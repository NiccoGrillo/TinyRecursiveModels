import csv
import json

import numpy as np
import pytest

from experiments.decision_stability.data import load_sudoku, smoke_data
from experiments.decision_stability.prepare_sudoku import prepare
from experiments.decision_stability.report import paired_interval, write_report
from experiments.decision_stability.metrics import summarize_interventions, summarize_rollouts


def test_csv_preparation_preserves_tokens_groups_and_source(tmp_path):
    data = smoke_data()
    source = tmp_path / "test.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "question", "answer", "rating"])
        for inp, lab in zip(data.inputs, data.labels):
            writer.writerow(["fixture", "".join(str(x - 1) if x > 1 else "." for x in inp),
                             "".join(str(x - 1) for x in lab), 1])
    out = tmp_path / "prepared"
    provenance = prepare(source, out, "https://example.org/pinned/test.csv")
    assert np.array_equal(np.load(out / "test/all__inputs.npy"), data.inputs)
    assert np.array_equal(np.load(out / "test/all__labels.npy"), data.labels)
    assert np.array_equal(np.load(out / "test/all__group_indices.npy"), [0, 1, 2])
    selected = load_sudoku(out, limit=1)
    assert selected.provenance["preparation"]["record"] == provenance
    with pytest.raises(FileExistsError):
        prepare(source, out, "https://example.org/pinned/test.csv")


def test_csv_preparation_rejects_malformed_tokens_before_writing(tmp_path):
    source = tmp_path / "test.csv"
    source.write_text("source,question,answer,rating\nx,broken,broken,1\n")
    with pytest.raises(ValueError, match="tokens"):
        prepare(source, tmp_path / "out", "fixture")
    assert not (tmp_path / "out").exists()


def test_paired_interval_resamples_puzzle_differences():
    assert paired_interval([]) == dict(examples=0, mean=None, lower=None, upper=None)
    assert paired_interval([1] * 8) == dict(examples=8, mean=1, lower=1, upper=1)
    assert paired_interval([-1] * 8)["upper"] == -1
    result = paired_interval([-1, 0, 1, 1], seed=5)
    assert result == paired_interval([-1, 0, 1, 1], seed=5)
    assert result["mean"] == 0.25
    assert result["lower"] < result["mean"] < result["upper"]


def test_report_rejects_incomplete_run(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(dict(status="running")))
    with pytest.raises(ValueError, match="incomplete"):
        write_report(tmp_path)


def test_report_normal_horizon_pairs_losses_and_recoveries(tmp_path):
    rows = [dict(sample_id=key, step=step, correct=value, rule_valid=value)
            for key, timeline in {"a": [True, False], "b": [False, True], "c": [True, True]}.items()
            for step, value in enumerate(timeline, 1)]
    manifest = dict(status="complete", run_kind="checkpoint_diagnostic",
                    arguments=dict(smoke=False, split_seed=0, seed=0, device="cpu", steps=2),
                    model_config=dict(halt_max_steps=1, forward_dtype="float32"),
                    dataset=dict(partition="dev"), wall_seconds=1, checkpoint_sha256="fixture", compute={})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "rollouts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    paired = [dict(sample_id=key, fork_step=1, offset=1, arm=arm, correct=value, prediction=[int(value)],
                   correct_at_fork=key != "b", changed_from_fork=key != "c",
                   matched_budget_ok=key != "a", budget_ok=key != "a" or arm == "clean",
                   zero_delta=arm == "clean", actual_relative_rms=0 if arm == "clean" else 0.01)
              for arm, outcomes in {"clean": dict(a=False, b=True, c=True),
                                    "random_plus": dict(a=True, b=False, c=True)}.items()
              for key, value in outcomes.items()]
    (tmp_path / "interventions.jsonl").write_text("\n".join(json.dumps(r) for r in paired))
    summary = dict(run_kind="checkpoint_diagnostic", natural=summarize_rollouts(rows),
                   interventions=summarize_interventions(paired))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    result = write_report(tmp_path, repetitions=100)
    normal = result["normal_horizon"]
    assert normal["accuracy"]["numerator"] == 2
    assert normal["harmed"]["numerator"] == normal["helped"]["numerator"] == 1
    assert normal["final_minus_normal"]["examples"] == 3
    assert normal["final_minus_normal"]["mean"] == 0
    matched = [p for p in result["paired"] if p["stratum"] == "all_arms_within_norm_tolerance"]
    assert all(p["examples"] == 2 and p["excluded"] == 1 for p in matched)
    attack = next(p for p in matched if p["arm"] == "random_plus")
    assert attack["final_minus_clean"]["mean"] == -0.5
    assert attack["final_accuracy"]["rate"] == 0.5
    assert attack["any_prediction_difference_vs_clean"]["rate"] == 0.5
    assert attack["ever_harmed_vs_clean"]["rate"] == 0.5
    assert attack["ever_helped_vs_clean"]["rate"] == 0
    # Rejoin the clean answer at offset 2: final accuracy must not hide a transient effect.
    paired += [dict(r, offset=2, correct=r["sample_id"] != "a",
                    prediction=[int(r["sample_id"] != "a")]) for r in list(paired)]
    (tmp_path / "interventions.jsonl").write_text("\n".join(json.dumps(r) for r in paired))
    summary["interventions"] = summarize_interventions(paired)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    result = write_report(tmp_path, repetitions=100)
    attack = next(p for p in result["paired"] if p["arm"] == "random_plus"
                  and p["stratum"] == "all_arms_within_norm_tolerance")
    assert attack["any_prediction_difference_vs_clean"]["rate"] == 0.5
    assert attack["final_prediction_difference_vs_clean"]["rate"] == 0
    assert attack["final_minus_clean"]["mean"] == 0
    summary["natural"]["final_accuracy"]["numerator"] = 99
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="disagrees"):
        write_report(tmp_path)
