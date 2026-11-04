import json

import numpy as np
import pytest
import torch

from experiments.decision_stability.data import load_sudoku, smoke_data, sudoku_valid
from experiments.decision_stability.metrics import exact_correct, summarize_interventions, summarize_rollouts
from experiments.decision_stability.toy import trajectory


def test_hazards_and_denominators():
    rows = [dict(sample_id=key, step=t + 1, correct=correct, rule_valid=correct)
            for key, seq in {"a": [True, False, True], "b": [False, False, True]}.items()
            for t, correct in enumerate(seq)]
    metrics = summarize_rollouts(rows)
    assert metrics["pooled_loss_hazard"] == dict(numerator=1, denominator=1, rate=1)
    assert metrics["pooled_recovery"] == dict(numerator=2, denominator=3, rate=2 / 3)
    assert metrics["ever_lost_after_first_correct"]["rate"] == 0.5
    assert metrics["per_step"][2]["loss_hazard"]["rate"] is None
    assert metrics["final_accuracy"]["rate"] == 1
    assert metrics["first_correct_steps"] == {"a": 1, "b": 3}


def test_paired_harm_help_and_first_loss_survival():
    rows = []
    cases = {"a": (True, {"clean": [True, True], "random_plus": [False, False]}),
             "b": (False, {"clean": [False, False], "random_plus": [False, True]})}
    for key, (initial, arms) in cases.items():
        for arm, timeline in arms.items():
            for j, value in enumerate(timeline):
                rows.append(dict(sample_id=key, fork_step=4, offset=j + 1, arm=arm, correct=value,
                                 correct_at_fork=initial, changed_from_fork=value != initial,
                                 budget_ok=True, matched_budget_ok=True, zero_delta=arm == "clean"))
    out = next(r for r in summarize_interventions(rows) if r["arm"] == "random_plus" and r["stratum"] == "all")
    assert out["final_harmed_vs_clean"]["rate"] == 0.5
    assert out["final_helped_vs_clean"]["rate"] == 0.5
    assert out["final_loss_given_correct_at_fork"]["rate"] == 1
    assert out["final_recovery_given_wrong_at_fork"]["rate"] == 1
    assert [x["survival"]["rate"] for x in out["first_loss_survival"]] == [0, 0]


def test_all_padding_is_never_correct():
    prediction = torch.tensor([[2, 3], [2, 3]])
    labels = torch.tensor([[-100, -100], [2, -100]])
    assert exact_correct(prediction, labels).tolist() == [False, True]


def test_verifier_checks_givens_and_boxes():
    data = smoke_data()
    labels, inputs = torch.tensor(data.labels), torch.tensor(data.inputs)
    assert sudoku_valid(labels, inputs).all()
    other_solution = (labels - 2 + 1) % 9 + 2
    assert not sudoku_valid(other_solution, inputs).any()
    invalid = labels.clone()
    invalid[:, 0] = invalid[:, 1]
    assert not sudoku_valid(invalid, inputs).any()


def prepare_fixture(root):
    path = root / "test"
    path.mkdir()
    data = smoke_data()
    labels = np.stack([(data.labels[0] - 2 + i) % 9 + 2 for i in range(8)])
    inputs = labels.copy()
    inputs[:, ::2] = 1
    metadata = dict(seq_len=81, vocab_size=11, num_puzzle_identifiers=1, pad_id=0)
    (path / "dataset.json").write_text(json.dumps(metadata))
    for key, arr in dict(inputs=inputs, labels=labels, puzzle_identifiers=np.zeros(8, dtype=int),
                         puzzle_indices=np.arange(9), group_indices=np.arange(9)).items():
        np.save(path / f"all__{key}.npy", arr)
    return path


def test_data_partitions_are_disjoint_repeatable_and_recorded(tmp_path):
    prepare_fixture(tmp_path)
    dev = load_sudoku(tmp_path, "dev", 7, 4)
    test = load_sudoku(tmp_path, "test", 7, 4)
    assert set(dev.sample_ids).isdisjoint(test.sample_ids)
    assert dev.sample_ids == load_sudoku(tmp_path, "dev", 7, 4).sample_ids
    assert len(dev.provenance["source_sha256"]) == 6
    with pytest.raises(ValueError, match="available"):
        load_sudoku(tmp_path, "dev", 7, 5)


def test_reject_augmented_data(tmp_path):
    path = prepare_fixture(tmp_path)
    np.save(path / "all__group_indices.npy", np.array([0, 4, 8]))
    with pytest.raises(ValueError, match="unaugmented"):
        load_sudoku(tmp_path, "dev", 0, 1)


def test_stable_linear_toy_loses_and_recovers_answer():
    rows = trajectory()
    assert rows[0]["correct"]
    assert not rows[3]["correct"]
    assert rows[13]["correct"]
    assert rows[6]["error"] == pytest.approx(-5 * (0.9**6 - 0.8**6))
