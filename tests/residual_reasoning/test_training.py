import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.residual_reasoning import train
from experiments.residual_reasoning.common import read
from experiments.residual_reasoning.data import Dataset, prepare, unique_sudoku
from experiments.residual_reasoning.evaluate import evaluate_arm
from experiments.residual_reasoning.model import Architecture, METHODS, Reasoner
from experiments.residual_reasoning.objective import Objective, Training, paired_edit
from experiments.residual_reasoning.report import paired_rows, summarize
from experiments.residual_reasoning.study import jobs

CFG = read(Path(__file__).resolve().parents[2] / "config/residual/smoke.json")
SOL = (
    "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
)
PUZZLE = (
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
)


@pytest.fixture(scope="module", autouse=True)
def threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.mark.parametrize("method", METHODS)
def test_unrolled_backward_and_finite_inference(method):
    torch.manual_seed(4)
    model = Reasoner(Architecture(**CFG["architecture"]), 5, 5, method)
    x = torch.randint(5, (4, 5))
    y = x.cumsum(1) % 5
    loss, _ = Objective(model, Training(loops=2), False)(x, y, x >= 0)
    loss.backward()
    assert loss.isfinite()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )
    if method == "recurrent":
        assert (
            model.factor.weight.grad.norm() > 0 and model.memory.weight.grad.norm() > 0
        )
    with torch.no_grad():
        xh = model.encode(x)
        q, w = model.start(xh)
    q, w = model.step(xh, q, w, train=False)
    assert not q.requires_grad and torch.isfinite(q).all()


def test_zero_residual_and_workspace_independence():
    m = Reasoner(Architecture(**CFG["architecture"]), 5, 5, "recurrent")
    for head in (m.field.geometry, m.field.score):
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    xh = m.encode(torch.zeros(3, 5, dtype=torch.long))
    q, w = m.start(xh)
    q1, w1 = m.step(xh, q, w, train=True)
    q2, w2 = m.step(xh, q, w + 1, train=True)
    assert torch.equal(q1, q) and torch.equal(q2, q) and not torch.equal(w1, w2)
    assert torch.equal(m.decode(q1), m.decode(q2))


def test_signed_wrong_zero_and_local_decoder_margin():
    m = Reasoner(Architecture(**CFG["architecture"]), 5, 5, "direct")
    torch.nn.init.zeros_(m.field.score.weight)
    torch.nn.init.zeros_(m.field.score.bias)
    y = torch.arange(5).repeat(3, 1)
    anchor = m.codes[y]
    delta = torch.nn.functional.normalize(torch.randn_like(anchor), dim=-1) * 0.6
    assert torch.equal(m.decode(anchor + delta).argmax(-1), y)
    score = m.field(m.encode(y), anchor)[:, -1] / 5**0.5
    (1 - score).relu().mean().backward()
    assert torch.allclose(m.field.score.bias.grad, torch.tensor([-1.0]))


def test_edits_match_locations_and_leave_givens():
    y = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]])
    free = torch.tensor([[True, False, True, True], [True, True, True, True]])
    results = []
    for mode in ("swap", "random"):
        torch.manual_seed(45)
        changed, valid = paired_edit(y, free, 4, mode)
        assert valid.tolist() == [True, False]
        assert torch.equal(changed[0, ~free[0]], y[0, ~free[0]])
        assert changed[0].ne(y[0]).sum() == 2
        if mode == "swap":
            assert torch.equal(changed[0].sort().values, y[0].sort().values)
        results.append(changed[0] != y[0])
    assert torch.equal(*results)


def test_preparation_uniqueness_splits_and_checksum(tmp_path):
    assert unique_sudoku(PUZZLE, SOL)
    assert not unique_sudoku("0" * 81, SOL)
    with pytest.raises(ValueError):
        unique_sudoku("1" + PUZZLE[1:], SOL)
    for name, masks in [("train", range(5)), ("test", range(3, 8))]:
        with (tmp_path / f"{name}.csv").open("w") as fp:
            writer = csv.writer(fp)
            writer.writerow(["source", "question", "answer", "rating"])
            for i in masks:
                writer.writerow(["fixture", SOL[:i] + "." + SOL[i + 1 :], SOL, 0])
    out = tmp_path / "data"
    prepare(
        out,
        "sudoku",
        [2, 2, 2],
        3,
        16,
        8,
        tmp_path / "train.csv",
        tmp_path / "test.csv",
    )
    ids = [key for split in ["train", "dev", "test"] for key in Dataset(out, split).ids]
    assert len(set(ids)) == 6
    with (out / "test.npz").open("ab") as fp:
        fp.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        Dataset(out, "test")


def test_restart_replays_weights_and_shared_field_is_frozen(tmp_path):
    dataset = tmp_path / "data"
    prepare(dataset, "scan", [24, 8, 8], 3, 5, 5)

    def fit(out, **kw):
        train.run(
            cfg=CFG,
            method="recurrent",
            seed=11,
            data_path=dataset,
            out=out,
            device="cpu",
            **kw,
        )

    fit(tmp_path / "whole")
    original = train.Objective.forward
    calls = 0

    def crash(self, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(self, *args)

    with (
        patch.object(train.Objective, "forward", crash),
        pytest.raises(RuntimeError, match="interruption"),
    ):
        fit(tmp_path / "restart")
    with (tmp_path / "restart" / "train.jsonl").open("a") as fp:
        fp.write('{"partial')
    fit(tmp_path / "restart", resume=True)
    whole = train.checkpoint(tmp_path / "whole" / "checkpoint.pt")
    again = train.checkpoint(tmp_path / "restart" / "checkpoint.pt")
    assert all(torch.equal(t, again["model"][k]) for k, t in whole["model"].items())
    assert [
        json.loads(r)["step"]
        for r in (tmp_path / "restart" / "train.jsonl").read_text().splitlines()
    ] == [1, 2]
    fit(tmp_path / "frozen", reference=tmp_path / "whole" / "checkpoint.pt")
    frozen = train.checkpoint(tmp_path / "frozen" / "checkpoint.pt")["model"]
    for k in frozen:
        if k.startswith(("field.", "initial.", "input.")) or k in {"codes", "position"}:
            assert torch.equal(frozen[k], whole["model"][k])


def test_plan_independence_and_null_denominators():
    plan = jobs(CFG)
    assert len(plan["reference"]) == 2
    ids = ["a", "b"]
    init = np.array([False, False])
    results = paired_rows(
        {"fork": 0},
        {
            "recurrent": (ids, init, np.array([True, False])),
            "direct": (ids, init, np.array([False, False])),
        },
    )
    assert results[1]["helped"] == 1 and results[1]["difference"] == 0.5
    assert results[2]["n"] == 0 and results[2]["difference"] is None


def test_sudoku_training_entrypoint(tmp_path):
    for name, mask in [("train", range(12)), ("test", range(12, 16))]:
        with (tmp_path / f"{name}.csv").open("w") as fp:
            writer = csv.writer(fp)
            writer.writerow(["source", "question", "answer", "rating"])
            for i in mask:
                writer.writerow(["fixture", SOL[:i] + "." + SOL[i + 1 :], SOL, 0])
    data = tmp_path / "sudoku"
    prepare(
        data,
        "sudoku",
        [8, 4, 4],
        2,
        81,
        9,
        tmp_path / "train.csv",
        tmp_path / "test.csv",
    )
    cfg = CFG | {"architecture": CFG["architecture"] | {"latent": 12, "max_length": 81}}
    train.run(
        cfg=cfg,
        method="recurrent",
        seed=1,
        data_path=data,
        out=tmp_path / "model",
        device="cpu",
    )
    assert read(tmp_path / "model" / "complete.json")["step"] == 2
    output = tmp_path / "evaluation"
    evaluate_arm(
        output,
        tmp_path / "model" / "checkpoint.pt",
        Dataset(data, "dev"),
        cfg["evaluation"],
        torch.device("cpu"),
        {"seconds": 1.0},
    )
    metrics, _ = summarize(output, {}, [0])
    for row in metrics:
        assert row["tokens_total"] == 4 * 81
        assert row["editable_tokens_total"] == 4
        assert row["editable_tokens_correct"] <= row["tokens_correct"]
        assert row["editable_token_accuracy"] == row["editable_tokens_correct"] / 4


def test_wall_budget_saves_resumable_final_state(tmp_path):
    data = tmp_path / "scan"
    prepare(data, "scan", [16, 4, 4], 3, 5, 5)
    train.run(
        cfg=CFG,
        method="recurrent",
        seed=1,
        data_path=data,
        out=tmp_path / "model",
        override={"max_seconds": 0.000001},
        device="cpu",
    )
    summary = read(tmp_path / "model" / "complete.json")
    assert summary["stop_reason"] == "time_budget" and summary["step"] == 1


def test_miniature_campaign(tmp_path):
    import os
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "experiments.residual_reasoning.smoke",
        "--root",
        str(tmp_path),
    ]
    # Exercise Gloo/DDP in network-enabled CI. The GH200 acceptance job exercises NCCL.
    if os.environ.get("GITHUB_ACTIONS") == "true":
        cmd += ["--gpus", "2"]
    subprocess.run(
        cmd, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True, timeout=180
    )
    root = tmp_path / "smoke" / "campaign"
    assert read(root / "reports" / "coverage.json")["missing"] == []
    frozen = root / "evaluations/scan/seed-11/shared/identity"

    def base(arm):
        return [
            (r["id"], r["fork"], r["initial"])
            for r in map(
                json.loads,
                (frozen / arm / "test/examples.jsonl").read_text().splitlines(),
            )
        ]

    assert base("search2") == base("recurrent")
    with (root / "reports" / "metrics.csv").open(newline="") as metrics:
        rows = list(csv.DictReader(metrics))
    assert rows
    for row in rows:
        length = 10 if row["split"] == "long" else 5
        assert int(row["tokens_total"]) == int(row["n"]) * length
        assert row["editable_tokens_correct"] == row["tokens_correct"]
        assert row["editable_tokens_total"] == row["tokens_total"]
        assert float(row["token_accuracy"]) == int(row["tokens_correct"]) / int(
            row["tokens_total"]
        )
    assert (root / "reports" / "paired.csv").stat().st_size > 100
