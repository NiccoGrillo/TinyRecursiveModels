import json
import subprocess
import sys

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.decision_stability.clock import (budget_plan, observation_audit, paired_binary_interval,
                                                  summarize, zero_event_upper)
from experiments.decision_stability.clock_report import verify
from experiments.decision_stability.model import load_model, smoke_config


@pytest.mark.parametrize("mlp", [True, False])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_dense_native_execution_matches_original_macro(mlp, dtype):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    c = smoke_config(mlp)
    c.update(seq_len=9, H_cycles=3, L_cycles=6, L_layers=2, forward_dtype=dtype)
    adapter = load_model(c)
    batch = dict(inputs=torch.randint(1, 11, (2, 9)), puzzle_identifiers=torch.zeros(2, dtype=torch.long))
    native = adapter.initial(batch)
    dense = adapter.initial(batch)
    with sdpa_kernel(SDPBackend.MATH):
        context = adapter.outer_context(batch)
        for _ in range(3):
            native, native_logits = adapter.step(native, batch)
            for _ in range(c["H_cycles"]):dense, dense_logits = adapter.outer_step(dense, batch, context=context)
            assert torch.equal(native.z_H, dense.z_H)
            assert torch.equal(native.z_L, dense.z_L)
            assert torch.equal(native_logits, dense_logits)


def test_exact_common_clock_budget():
    costs, common = budget_plan([6, 12], 2, 2184)
    assert costs == {6: 14, 12: 26}
    assert common == 182 and 2184 // costs[6] == 156 and 2184 // costs[12] == 84
    with pytest.raises(ValueError, match="multiple"):
        budget_plan([6, 12], 2, 936)  # 36 changed-schedule updates is not equal-cost to native


def test_native_output_semantics_are_not_conflated_with_dense_first_hit():
    # Correct at native endpoint 3; transient error at 4; correct again at 6/9.
    values = [False, True, True, False, True, True, True, True, True]
    rows = [dict(sample_id="a", outer_update=i + 1, correct=v, low_cycles=6) for i, v in enumerate(values)]
    audit = observation_audit([], rows, 6, 3, 6)
    assert audit["native_macro_loss_after_official_anchor"]["numerator"] == 0
    assert audit["dense_loss_after_official_anchor"]["numerator"] == 1
    assert audit["dense_excursion_missed_by_macro"]["numerator"] == 1
    assert audit["per_puzzle"][0]["first_native_macro_solve"] == 3
    assert audit["per_puzzle"][0]["first_dense_solve"] == 2
    # A later-solving puzzle lacks follow-up and must not dilute the risk set.
    late = [dict(sample_id="b", outer_update=i + 1, correct=i >= 7, low_cycles=6) for i in range(9)]
    audit = observation_audit([], rows + late, 6, 3, 6)
    assert audit["official_anchor_cohort"] == 1
    assert audit["not_eligible_official"] == 1


def test_crossover_risk_sets_and_block_time():
    # Toy clocks: L=1 costs 2 blocks, L=2 costs 3 blocks. Equal budget = 6.
    prefix = [dict(sample_id=i, outer_update=1, correct=v) for i, v in (("a", True), ("b", False))]
    anchors = {r["sample_id"]: r for r in prefix}
    sequences = {("a", 1): [True, True, True], ("a", 2): [False, True],
                 ("b", 1): [False, False, True], ("b", 2): [True, True]}
    branches = [dict(sample_id=i, low_cycles=l, branch_outer=k, outer_update=k + 1,
                     branch_block_evals=k * (l + 1), correct=c)
                for (i, l), seq in sequences.items() for k, c in enumerate(seq, 1)]
    result = summarize(prefix, branches, anchors, 1, 1, 1, 6, 1, bootstrap=50)
    correct = next(r for r in result["comparisons"] if r["low_cycles"] == 2 and r["cohort"] == "correct_at_fork")
    wrong = next(r for r in result["comparisons"] if r["low_cycles"] == 2 and r["cohort"] == "wrong_at_fork")
    assert correct["ever_lost"]["rate"] == 1 and correct["final_correct"]["rate"] == 1
    assert correct["mean_incorrect_block_fraction"] == 0.5
    assert wrong["ever_recovered"]["rate"] == 1
    assert wrong["mean_incorrect_block_fraction"] == 0.5
    assert correct["common_budget_grid"][0]["branch_block_evals"] == 6
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(prefix, branches[:-1], anchors, 1, 1, 1, 6, 1)


def test_zero_events_do_not_imply_certainty():
    assert zero_event_upper(300) == pytest.approx(0.00993608194)
    assert zero_event_upper(0) is None
    bound = paired_binary_interval([False] * 300, [False] * 300)
    assert bound["mean"] == 0 and bound["lower"] < 0 < bound["upper"]
    assert bound["upper"] == pytest.approx(1 - 0.0125**(1 / 300))
    exact = paired_binary_interval([True], [False])
    assert exact["mean"] == 1 and exact["lower"] == pytest.approx(2 * 0.0125 - 1)


def test_crossover_separates_native_macro_loss_and_intermediate_phases():
    prefix = [dict(sample_id="a", outer_update=k, correct=True) for k in range(1, 4)]
    anchors = {"a": prefix[-1]}
    branches = [dict(sample_id="a", low_cycles=l, branch_outer=k, outer_update=3 + k,
                     branch_block_evals=k * (l + 1) * 2, correct=not (l == 6 and k == 1))
                for l in (6, 12) for k in range(1, 182 // ((l + 1) * 2) + 1)]
    result = summarize(prefix, branches, anchors, 6, 3, 2, 182, 3, bootstrap=50)
    native = next(r for r in result["comparisons"] if r["low_cycles"] == 6 and r["cohort"] == "correct_at_fork")
    assert native["ever_lost"]["numerator"] == 1
    assert result["native_macro_ever_loss_from_fork"]["numerator"] == 0
    assert native["final_phase_mod_h"] == 1
    assert native["common_budget_grid"][0]["phase_mod_h"] == 1
    assert set(native["paired_difference_vs_native"]) == {"incorrect_block_fraction"}


def test_full_clock_command_accounting_and_independent_reanalysis(tmp_path):
    out = tmp_path / "clock"
    subprocess.run([sys.executable, "-m", "experiments.decision_stability.clock", "--smoke", "--smoke-attention",
                    "--out", str(out), "--bootstrap", "50"], check=True, capture_output=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["compute"] == dict(prefix_sample_block_evals=24, branch_sample_block_evals=120,
                                       readout_sample_evals=42, input_embedding_sample_evals=2, backward_calls=0)
    result = verify(out)
    assert result["common_block_interval"] == 15
    assert (out / "report.md").exists()
    with (out / "branches.jsonl").open("a") as handle:handle.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(out)
