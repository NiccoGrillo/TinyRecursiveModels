# Next experiment: change the computation clock from the same state

## Current architecture direction — 12 November 2026, second research session

The leading candidate is now **solution-preserving residual mixing**: `q_next = q - alpha * J_c(q)^T W(q,w,x)c(q;x)`, with bounded positive-definite, low-rank residual-space mixing. It can move at some wrong stationary points that the previous state-side preconditioner and diagonal constraint weights cannot leave. Every common residual zero remains fixed. This is a distinction between update classes, not a claim of a new general optimizer: Sketched Newton–Raphson is a direct predecessor, and random residual mixing is a mandatory control.

The first full architecture is a grounded Sudoku factor-graph reasoner: Boolean, cell, row, column, box and clue residuals define valid solutions; a bounded per-factor workspace predicts their low-rank interactions. The controller is learned, the initial task residuals are specified. Generic learned latent residuals remain a later branch. No complete architecture has been trained or benchmarked. A tested PyTorch operator prototype is in `experiments/residual_reasoning/operators.py`.

Start with Section 1 of [notebook version 2.1](notebook/project_notebook.pdf), or its [LaTeX source](notebook/project_notebook.tex), for four adversarial critique rounds, the supplied QRR review assessment, exact proofs, limitations and the decisive comparison. The prior energy-preconditioner is retained as a baseline. E4/E5 remain optional historical diagnostics and do not gate architectural work.

## Historical specification (preserved)

Protocol frozen before running the new cohort, 2026-11-05. Implementation checks are complete; the trained-checkpoint study has **not** been run. This is a development study with explicit decision gates, not a registered report or a publication claim.

## Why this experiment

The pilot's interesting effect was delayed or accelerated recovery of a wrong answer. Correct answers survived both natural continuation and the tested perturbations. More tiny perturbation runs would not establish a retention problem.

A newly identified close paper, [When Recursive Models Finish Computing](https://arxiv.org/abs/2609.26487v1), substantially overlaps our original diagnostic direction; see [RESOURCES.md](RESOURCES.md). Our next question is more specific: **does the allocation of computation between reasoning-state and answer-state updates change retention or recovery, even when both continuations start from exactly the same state and receive the same block budget?**

This is motivated by the mismatch between our pilot and the InfiLoop retention diagnostic, not an exact reproduction of that figure. The latter's TRM checkpoint and complete figure protocol have not been identified. Matching its 26 blocks per outer update by changing L to 12 changes the dynamics of our L=6 checkpoint; it is not a harmless relabeling of time.

The follow-up review on 2026-11-06 also identified FPRM Appendix F as direct compute-allocation prior. The current protocol and decision thresholds remain unchanged, but an accuracy difference alone cannot be presented as a new allocation principle. [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md) develops the narrower coupled-feedback hypothesis and its limits.

## Dynamical interpretation and estimand

Let one low-state update be A and one high-state update be B, acting on the full state `(z_H,z_L)` with fixed input. One outer update is `F_L = B ∘ A^L`; the native macro is `F_6^3`. Reading the current answer between native outer updates does not change this map. Replacing `F_6` with `F_12` does.

The treatment is the **continuation schedule**, including its different number of high-state updates under equal compute. It is not a clean intervention on just elapsed time or just a single Jacobian. The estimand is conditional on states reached by the native schedule at a fixed time. It does not decompose the effects of a schedule used from initialization into reachability versus stabilization. A source-schedule × continuation-schedule factorial would be a later experiment only if warranted.

## Fixed design

| Item | Prespecified setting |
| --- | --- |
| Population | Same 1,024 project-dev puzzles, split seed 0, offset 32; excludes the old exploratory prefix |
| Checkpoints | Pinned gaoxin492 MLP step_16275 and attention step_21700, original TRM code |
| Numerical regimes | BF16 and FP32 separately, math SDPA, TF32 disabled, batch size 8 |
| Native prefix | 16 native macros = 48 outer updates = 672 block evaluations per puzzle |
| Fork | Copy both latent tensors at that native macro boundary; no reset, noise, gradients, or label-based branching |
| Continuations | L=6 and L=12, frozen identical weights and input |
| Budget per continuation | 2,184 block evaluations: 156 outer updates at L=6; 84 at L=12 |
| Common comparison grid | Every 182 blocks: 13 L=6 updates or 7 L=12 updates |
| Native observation audit | 36 further native outer updates after first solve, only when complete follow-up exists |
| Primary comparison | Attention checkpoint, BF16: L=12 minus L=6 dense ever-loss among puzzles correct before the fork |
| Practical effect threshold | 2 percentage points of paired excess dense ever-loss |
| Secondary comparisons | Other three checkpoint × dtype conditions; final correctness, recovery, incorrect block-time, native output retention |

One block means one `L_level` layer evaluation. Costs per outer update are `(L+1)*L_layers`, hence 14 versus 26. Equal numbers of outer updates would not match compute. Within one checkpoint the shared block shapes are fixed; this is not a FLOP match between MLP and attention architectures. Readout and embedding counts are reported separately. The four-condition study uses 20,643,840 sample-block evaluations, excluding readout/embedding/analysis costs.

The 182-block grid observes dense high-state outputs; some grid points lie between normal native macro outputs. At the default final budget, both schedules finish after a multiple of H=3 outer updates. Reports record endpoint phase explicitly. The native audit's 36 updates cost 504 blocks; this is **not** the 36×26-block window of the InfiLoop diagnostic.

Each precision has its own native prefix and resulting states. Branches within a precision share an exact starting state. Cross-precision differences concern the entire numerical regime, not an intervention on precision at one identical latent. One checkpoint per architecture cannot identify an architecture effect independently of training/checkpoint differences.

## Three distinct retention questions

1. **Native output retention:** after a correct native macro output, does a later native macro output become wrong? Report this both from the fixed fork over the whole continuation and from first solving over the fixed audit window.
2. **Intermediate excursions:** on that same native trajectory and the same official first-solve cohort, does a dense high-state readout become wrong between macro endpoints? A model that is right at macros 16 and 17 but wrong between them has an intermediate excursion, not a demonstrated failure of its normal output policy.
3. **Dense first-solve retention:** after the first correct dense readout, is a later dense readout wrong? This is descriptive and can include transient intermediate solutions that the original model would never emit.

`dense_loss_after_official_anchor` includes both off-macro and macro failures. `dense_excursion_missed_by_macro` counts puzzles with dense loss but **no** macro loss anywhere in that window; it is not a count of every off-macro error.

The crossover uses the same pre-treatment correct-at-fork cohort in both arms. Recovery uses the separate wrong-at-fork cohort. Never select puzzles according to whether an arm solves later. First-solve audits are descriptive, require the same complete follow-up window, and report excluded late/never-solving cases. Labels are used only for offline measurement, never for a schedule decision.

Besides any-time loss, report final correctness and the fraction of the block budget spent with an incorrect current high-state readout. That fraction treats the last high state as constant during the next outer computation, and the terminal update as having zero duration. It is a computational-state statistic, not a wall-clock output-policy measurement. Full per-update predictions remain available for reanalysis.

## Uncertainty and decision gates

The independent unit is an original puzzle, not an update, fork, or cell. The primary binary comparison uses conservative paired risk-difference bounds built from the two discordant-event counts, with simultaneous Clopper–Pearson bounds. Continuous block-time differences use a 2,000-resample paired puzzle bootstrap. Intervals are pointwise; secondary comparisons are exploratory rather than family-wise significance claims. No checkpoint-seed uncertainty is estimated.

The cohort size is fixed at 1,024 before outcomes are inspected. At least 300 correct-at-fork puzzles is an adequacy flag: zero losses in 300 independent puzzles gives a one-sided 95% upper bound of about 0.994%. It is not a power calculation for every alternative or a reason to keep sampling until a favorable result appears. Report the flag and bounds even if it is not reached. Exact deduplication does not remove all symmetry-related dependence or upstream test-selection bias.

| Result | Decision |
| --- | --- |
| Primary lower bound > +2 pp | Support a practically meaningful adverse continuation-schedule effect on native-reachable states; replicate on an independent cohort/checkpoint before a mechanism claim |
| Primary upper bound < +2 pp | Rule out the prespecified harmful effect size under this protocol; stop that explanation here |
| Primary interval overlaps +2 pp | Inconclusive about that effect size; do not call a null or a meaningful positive result from the point estimate alone |
| Dense native losses, stable native macro outputs | Observation-frequency result; no native output-retention repair is justified by it |
| L=12 loses answers, native macro outputs remain stable | Schedule-induced damage; do not present it as an existing native TRM defect |
| Both schedules stable with informative bounds | Stop this retention-repair direction for these conditions; do not increase attack strength to create a problem |
| Useful final accuracy/recovery improvement at equal cost | A possible computation-allocation direction; require a label-free policy and comparison with fixed schedules, verifier stopping, and TAPS |

A significant effect above zero does not establish an effect above 2 pp. A large point estimate with a wide interval is not enough. Numerical-regime-specific effects must be reported as such. None of these gates alone certifies conference novelty.

If there is a real useful schedule effect, the next method question is whether label-free trajectory information predicts **which allocation improves future correctness**, beyond depth, margins, Q, residuals and answer stability. [TAPS](https://arxiv.org/abs/2609.36653), fixed recurrence schedules, extra ordinary continuation, and Sudoku verifier stopping are mandatory comparators. A new controller or training loss is not implemented before that evidence exists.

## Checkpoints and launch

Reuse the MLP weights and prepared full evaluation dataset from [PILOT.md](PILOT.md). The new attention input is [gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att](https://huggingface.co/gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att/tree/33991b882368f425916d625a9ee614a7a85efe76), revision `33991b882368f425916d625a9ee614a7a85efe76`. This is a third-party reproduction, not a verified InfiLoop baseline. Published source matches the pinned upstream. Downloaded Python is never executed.

| Attention asset | SHA256 |
| --- | --- |
| `step_21700` | `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3` |
| `all_config.yaml` | `9e595c896d6484e9dbb3c3963d11bac224dd91d70eb040f11de04ac8303be6b3` |
| Published `trm.py` | `3454818b64a1abb45c062782d380aa4bb560e2cb35db9cc309cf367ca3ac262c` |

The fetch helper downloads only those pinned files, verifies hashes, and records provenance. Existing mismatched files fail. Training and EMA selection remain unverified. The launcher verifies both frozen checkpoint/config hashes, strict loading and the GPU before launching; path overrides relocate the same assets. It does not download benchmark data or start training.

```bash
.venv/bin/python scripts/fetch_attention_checkpoint.py
bash scripts/run_clock_study.sh
```

Default paths match the pilot. GPU execution needs an appropriate CUDA PyTorch installation; the CPU quickstart installs a CPU-only build. Customize only operational paths/device before the run:

```bash
TRM_DEVICE=cuda TRM_BATCH_SIZE=8 TRM_OUT=runs/clock-study-v1 \
  bash scripts/run_clock_study.sh
```

`TRM_PYTHON`, `TRM_DATA`, `TRM_MLP_CONFIG`, `TRM_MLP_CHECKPOINT`, `TRM_ATT_CONFIG`, `TRM_ATT_CHECKPOINT`, and `TRM_THREADS` are also supported. Explicit `TRM_DEVICE=cpu` is supported but this is a substantial study, not another laptop micro-pilot. Output root must be new; there is no automatic resume or outcome-driven expansion. The launcher runs attention BF16 first, then the three secondary conditions, with identical predetermined IDs and budgets. It verifies each report before proceeding and checks matching IDs across all four completed runs.

A single condition can be run directly:

```bash
.venv/bin/python -m experiments.decision_stability.clock \
  --config checkpoints/gaoxin492-sudoku-att-33991b8/all_config.yaml \
  --checkpoint checkpoints/gaoxin492-sudoku-att-33991b8/step_21700 \
  --data data/sudoku-extreme-public/prepared \
  --device cuda --dtype bfloat16 --attention-backend math \
  --limit 1024 --offset 32 --partition dev --split-seed 0 --seed 0 \
  --batch-size 8 --low-cycles 6 12 --fork-outer 48 --branch-blocks 2184 \
  --followup-outer 36 --out runs/clock-att-bf16
.venv/bin/python -m experiments.decision_stability.clock_report runs/clock-att-bf16
```

Each run writes `manifest.json`, `prefix.jsonl`, `branches.jsonl`, `anchors.json`, `summary.json`, and `report.md`. Raw outputs are hashed. The independent report command checks completeness, hashes, and recomputes summaries without loading a model. Keep raw files for audit; do not commit weights or bulk outputs. Commit concise result tables and their artifact hashes to `STATUS.md` after the study.
