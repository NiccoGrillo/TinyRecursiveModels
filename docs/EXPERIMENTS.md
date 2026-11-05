# First experiments and runbook

## Environment and inputs

For the first downloaded checkpoint, fixed CPU pilot settings, and exact commands, see [the public-checkpoint pilot](PILOT.md).

Install the diagnostic environment as in the README. Verified: Python 3.12.14, torch 2.7.0+cpu, and `requirements-experiments.txt`. GPU runs need a compatible CUDA torch build. The pinned upstream uses PyTorch SDPA; these diagnostics do not need FlashAttention. Full upstream training has additional dependencies and recipes in [README_UPSTREAM.md](../README_UPSTREAM.md).

Use a trusted checkpoint of the **original TRM architecture** plus the resolved `all_config.yaml` that produced it. A flat complete model JSON/YAML also works. `config/arch/trm.yaml` has unresolved Hydra interpolations and is not a checkpoint config. Accepted weights: native state dict, `model.` wrapper, or `_orig_mod.model.` wrapper. Loading uses `weights_only=True` and strict keys/shapes. Missing weights, incompatible architectures, and optimizer bundles fail; there is no random fallback.

After installing upstream data dependencies, its Sudoku preparation command is:

```bash
python -m dataset.build_sudoku_dataset \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 --num-aug 1000
```

This downloads data and can create a large augmented training set; it is unnecessary for smoke checks. The diagnostic reader uses only `test/dataset.json` and the five `test/all__*.npy` arrays for inputs, labels, identifiers, puzzle indices, and group indices. It rejects augmented groups. PAD=0, blank=1, digits=2..10. Labels must be complete valid Sudoku solutions respecting givens.

The prepared test set is divided into **project dev and held-out test halves** by stable seeded hash ordering of deduplicated input puzzles. Keep `--split-seed 0` fixed; `--seed` changes intervention noise and smoke initialization. Exact input duplicates are removed. Symmetry-equivalent duplicates and training contamination are not automatically audited. This split cannot undo upstream checkpoint selection on the same data: audit provenance or generate an independent final test set before publication.

## E0: verify the instrument

```bash
python -m pytest -q
python -m experiments.decision_stability.run --smoke \
  --out runs/smoke --steps 8 --fork-steps 2 4 --horizon 4 \
  --targeted --probe-horizon 2
```

Random tiny weights and synthetic puzzles check execution, not reasoning quality. Choose a new output directory for every run; overwrites are rejected.

## E1: natural continued reasoning

Set real paths; these placeholder files are not bundled:

```bash
TRM_CHECKPOINT=checkpoints/my-run/step_100000
TRM_CONFIG=checkpoints/my-run/all_config.yaml
TRM_DATA=data/sudoku-extreme-1k-aug-1000

python -m experiments.decision_stability.run \
  --config "$TRM_CONFIG" --checkpoint "$TRM_CHECKPOINT" --data "$TRM_DATA" \
  --device cuda --batch-size 8 --limit 128 --partition dev --split-seed 0 \
  --experiment rollout --steps 64 --out runs/dev-natural-native
```

Omit `--dtype` to use the config's precision. Reproduce normal-horizon accuracy first, then inspect 64 macro-steps; increase to 256 only after timing a pilot. We never reset carry during continuation. Q output is logged but does not halt a trajectory. Labels are never used for inference decisions. Observations start after macro-step 1.

Inspect final accuracy, ever-correct fraction, correct→wrong hazards, wrong→correct recovery, margins, and latent changes. Ever-correct is an offline oracle upper bound. Pooled transitions are correlated; do not treat them as independent samples in confidence intervals. The ever-lost-after-first-correct fraction has unequal follow-up; fixed-fork survival gives matched follow-up.

The rule verifier checks all rows, columns, boxes, legal digits, and givens. It reports coverage and reference accuracy at first success. A different valid solution can fail reference exact match. Verifier stopping is a meaningful Sudoku baseline, but is not available for every task.

## E2: paired output-preserving perturbations

Use FP32 initially to interpret small budgets, after comparing its natural performance with native precision. Changing dtype changes the numerical dynamics.

```bash
python -m experiments.decision_stability.run \
  --config "$TRM_CONFIG" --checkpoint "$TRM_CHECKPOINT" --data "$TRM_DATA" \
  --device cuda --dtype float32 --batch-size 4 --limit 128 \
  --partition dev --split-seed 0 --seed 0 \
  --steps 64 --fork-steps 4 8 16 --horizon 16 --epsilon 0.01 \
  --out runs/dev-paired-eps001-seed0
```

Fork each **unperturbed** trajectory at predetermined times. Arms share input and full state: clean, Gaussian direction, and its negative. Only `z_L` changes; immediate logits and `z_H` must remain bitwise equal. `--experiment intervene` also runs the natural path needed to obtain forks; `both` is the default.

Nominal per-example budget: `||delta z_L||_2 / ||z_L||_2 = epsilon`. Record realized post-cast relative RMS, actual L2, direction norm, zero deltas, and budget error. `all_arms_within_norm_tolerance` selects a **common cohort** for every arm: all arms must meet the tolerance (default 5%). This is a tolerance criterion, not exact equality of realized norms. Report full-cohort results and excluded counts too. BF16 can erase small deltas; change epsilon on dev only and document it.

Suggested dev sweep: epsilon `{0, 0.001, 0.01, 0.03}`, noise seeds `{0,1,2}`, forks `{4,8,16}`. Pilot timing first. Zero epsilon must reproduce entire clean continuations. Seeded noise for a sample is independent of batching, but GPU kernel numerics can still vary with batch size.

## E3: offline future-margin sensitivity

Add `--targeted --probe-horizon 2` to E2 with a new output directory. This adds `gradient_plus` (negative gradient of a soft-min future margin) and `gradient_minus` (opposite direction), at the same nominal budget. Anchor tokens are the current predictions. There is no ground-truth-guided attack and no iterative PGD.

The derivative traverses every H/L cycle and macro-step using the forward-equivalent adapter, without upstream gradient cuts. Keep evaluation horizon 16 longer than probe horizon 2. Probe horizons `{2,4,8}` can be explored on dev. Extra activation memory can be substantial; reduce batch size before silently changing the scientific horizon.

Attack success is not evidence that the clean model naturally makes the same error. Compare paired harmed/helped outcomes, and examine changes at and beyond the optimized horizon. Stage and margins may explain apparent effects; control them before claiming a mechanism.

## Output and compute

| File | Content |
| --- | --- |
| `manifest.json` | Status, code/checkpoint/data hashes, selected rows, config, precision, seeds, device, timing, computation |
| `rollouts.jsonl` | Per-example per-step predictions, correctness, rule validity, margins, Q output, residuals |
| `interventions.jsonl` | Same plus fork/arm/offset, anchor correctness, actual budget, common-cohort eligibility |
| `summary.json` | Natural metrics, paired harm/help, conditional final loss/recovery, first-loss survival |
| `report.md` | Readable accuracy, paired effects, budgets, and compute (generated by report command) |
| `report.json` | Per-fork puzzle-paired bootstrap intervals and final solve rates |
| `accuracy_by_step.csv` | Natural accuracy, loss, and recovery counts at every step |

Recompute with `python -m experiments.decision_stability.analyze runs/dev-natural-native`. Incomplete runs are rejected. The gradient arms' `direction_rms` records gradient magnitude for a later risk analysis; that prospective analysis is **not implemented or validated**.

Generate readable artifacts with `python -m experiments.decision_stability.report runs/dev-natural-native`. It checks the saved summary against raw records. Pointwise percentile intervals resample original puzzles separately for each fork/arm/cohort comparison. They do not pool repeated forks or seeds, correct for multiple comparisons, or estimate checkpoint variance. Small or all-zero paired samples can give degenerate intervals; do not interpret them as certainty.

One macro = `H_cycles*(L_cycles+1)` module calls, each with `L_layers` blocks. Manifest counts are **sample-macro evaluations** (batch size times macro calls), separated into natural, branch, and probe forwards. Backwards are separate batched-call and sample-objective counts, not FLOPs. Repeated clean branches are counted. Per-row `cumulative_block_calls` describes trajectory depth, not total experiment cost. Wall time includes logging/analysis; it is not a controlled speed benchmark.

## Before a training intervention

Define thresholds/strata on dev, then freeze the held-out protocol. For pooled analyses, bootstrap paired differences by original puzzle, clustering repeated forks and noise seeds; report checkpoint variance separately. Evaluate a label-free score at t with probe horizon H against natural flips **after t+H**, compared with current margin, trend, Q, and prediction stability. Charge extra probe compute against equal-cost ordinary reasoning. Pooled uncertainty, prospective prediction, and replicated trained-model studies remain future work.
