# Working on TRM Decision Stability

Read `README.md`, `docs/PROJECT.md`, `docs/EXPERIMENTS.md`, and `docs/STATUS.md` before changing experiments. Use `docs/ROADMAP.md` for the next bounded task and `docs/RESOURCES.md` before making novelty claims. Record findings and executed checks in `docs/STATUS.md`.

The broad objective is to understand decision stability during continued computation. The pilot found recovery effects without observed retention loss. The immediate task is the prespecified same-state continuation-schedule study in `docs/NEXT_EXPERIMENT.md`, not a larger perturbation sweep. No improvement, learned manifold, or A* publication claim is established.

## Boundaries and commands

- Official base: `SamsungSAILMontreal/TinyRecursiveModels`, commit `c01103738605ba39d1430519b1ee0c62f4c707f8`.
- Prefer additive adapters; document and test any necessary upstream edit. Preserve MIT attribution.
- Supported backend: original `trm.py`, MLP and attention variants. Other TRM variants, FPRM, and InfiLoop require explicit adapters.
- Keep data, weights, outputs, credentials, and full paper downloads out of Git.
- Use the agreed compute budget. Long or paid training is not needed to validate a routine edit.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.decision_stability.run --smoke \
  --out runs/smoke-new --steps 8 --fork-steps 2 4 --horizon 4 --targeted
.venv/bin/python -m experiments.decision_stability.analyze runs/smoke-new
.venv/bin/python -m experiments.decision_stability.clock --smoke --smoke-attention \
  --out runs/clock-smoke-new
.venv/bin/python -m experiments.decision_stability.clock_report runs/clock-smoke-new
```

Install CPU torch separately, then `requirements-experiments.txt`. Full upstream training has additional dependencies in `README_UPSTREAM.md`.

## Scientific invariants

1. Preserve the full carry `(z_H,z_L)`, fixed input, and macro phase. Open-loop diagnostics bypass ACT resets; otherwise long runs silently become repeated short solves.
2. `z_L` interventions must preserve `z_H` and immediate logits bitwise. This is an exact readout fiber, not proof of a dynamically invariant manifold.
3. Native inner forward cuts early gradients and detaches carries. Future derivatives must use the full-gradient adapter, with forward parity at H=3/L=6 and multi-macro finite-difference tests.
4. Record realized post-cast perturbation norms, zero deltas, and budget failures. Compare common sample cohorts across arms; nominal BF16 budgets need not match.
5. Preset fork times and freeze hyperparameters before held-out evaluation. Labels are offline evaluation data; no oracle stopping or deployment-time correctness gating.
6. Report loss, recovery, final solve rate, paired harm/help, and denominators. Undefined rates are null. All-padding examples are not correct.
7. Gradient attacks anchor current predictions, not true labels. Count lookahead forwards/backwards and distinguish attack effects from prediction of natural errors.
8. Split by original puzzle, independently of noise seed. Never split augmented rows randomly. Exact deduplication does not establish absence of symmetry duplicates or training contamination.
9. Log checkpoint/data/source hashes, config, precision, device, seeds, IDs, phase, and compute. Missing checkpoints must fail, never silently use random weights.
10. Distinguish verified implementation, analytic illustration, trained-model evidence, and conjecture in reports.

11. In the clock study, dense readouts are observational; changing L is a dynamics intervention. Compare identical native-reachable states and equal block budgets. Dense errors do not imply failures at native macro outputs. Preserve the separate observation policies, pre-fork cohorts, and endpoint phases.
12. Zero-event bootstrap intervals are not equivalence evidence. Use the implemented exact binary bounds and the fixed effect-size gates. Do not expand the cohort based on observed results or treat correlated steps as independent puzzles.

For recurrence, intervention, data, or metric edits, run relevant scientific tests and CPU integration. Documentation edits need command/link consistency checks. The next useful result is the fixed clock study; its full run is not an implementation test. Seek counterexamples and simple baselines; do not pressure a critic to approve an idea or portray agent review as external peer review.
