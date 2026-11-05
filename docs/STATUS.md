# Status

## Initial implementation — 2026-11-04

Official TRM history and MIT licence are preserved at base `c01103738605ba39d1430519b1ee0c62f4c707f8`. Upstream models, layers, training, and dataset builders are unchanged.

Implemented: fixed-phase rollouts; full-gradient adapter; strict checkpoint loading; unaugmented Sudoku reader and deterministic dev/test split; antithetic random and optional future-gradient `z_L` interventions; immediate-logit invariance assertions; post-cast budgets; paired harm/help, retention/recovery and survival metrics; JSONL and hashed manifests; independent summary recomputation; analytic transient-growth toy.

Verified with Python 3.12.14 and torch 2.7.0+cpu:

- `python -m pytest -q`: **20 passed**.
- Exact native/full-gradient forward agreement for MLP and attention at H=3, L=6, two layers, across multiple macro-steps.
- Multi-macro finite-difference agreement, immediate-readout invariance, zero-epsilon trajectory identity, and no reset after normal halt depth.
- Strict checkpoint wrappers and missing-key rejection; handcrafted metric denominators; masking; verifier checks; deterministic disjoint data selection; BF16 zero-delta detection.
- End-to-end fixture checkpoint/data CLI, cost counts, summary recomputation, and overwrite rejection.
- Standalone documented smoke command completed: 8 macro-steps, forks at 2/4, four continuation steps, all five arms, and two-step gradient probes.

The fixture checkpoint has random weights. These are implementation checks, not benchmark evidence.

## Not established

At the initial implementation, no trained checkpoint evaluation had been executed. The first CPU pilot is recorded below. No GPU benchmark or full training run has been executed. No general empirical gain, new theorem, solution manifold, or publication novelty is claimed. Prospective natural-error prediction, training interventions, pooled uncertainty across repeated forks/seeds, and additional backends remain future work.

## Next step

The first bounded CPU pilot is complete. Prioritize matching the InfiLoop Figure 1 retention diagnostic before expanding perturbations: obtain its TRM-attention baseline checkpoint/protocol if available, observe every outer update, align follow-up to first solving, and compare native/FP32 dynamics. Keep the existing MLP pilot as a separate condition. No selective-stabilization training is justified yet, but our MLP null result is not a replication or refutation of the published attention-model failure.

## First public-checkpoint pilot — 2026-11-04

Protocol and commands: [PILOT.md](PILOT.md). Local readable index: `runs/README.md`. Weights, data, environment, and generated outputs remain Git-ignored.

Setup and verified implementation:

- Created `.venv` with Python 3.12.3, torch 2.7.0+cpu and pinned experiment requirements. No GPU is visible; no paid compute or training was launched. Exact installed packages are saved in `runs/environment.txt`.
- Downloaded the third-party `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-mlp` checkpoint at revision `256f32fcbe7123e8bf8c449410773a5ad311dbc5`. Published model source is byte-identical to the pinned upstream `trm.py`; weights load strictly. Source/config agreement does not establish complete training provenance or independently confirm EMA selection.
- Checkpoint SHA256: `20e9dc7ebf83b9b41a8b3f58f5fd94ee3a7eb0b0d245bdeeb14e2f1488d1daaf`; config SHA256: `811e64a1fa2e923e601791339b736bb3bab8bd350778ba116908d7965c021777`.
- Prepared all 422,786 rows of the revision-pinned public evaluation CSV, with unchanged token values/order and no augmentation. Exact deduplication found 422,786 unique inputs; project dev contains 211,393. CSV SHA256: `a2fd52aea23d331d5b4ee723c856236e838a9fb9a70e66f4e0e0cf26c338c6a8`. Only selected dev labels are checked/evaluated. No symmetry or training-contamination audit was performed.
- Added evaluation-only CSV preparation, manifest embedding of downloaded-input provenance, flushed batch progress, and a report command producing Markdown, JSON uncertainty, and per-depth CSV. Reports reject incomplete runs and disagreement with recomputed raw summaries. Upstream model/training files remain unchanged.
- `.venv/bin/python -m pytest -q`: **25 passed**. Includes original scientific tests, CPU checkpoint integration, CSV token/group/provenance checks, paired-bootstrap and common-cohort checks, report mismatch rejection, and overwrite rejection.
- Documented smoke command completed at `runs/smoke-first-pilot`, independently reanalyzed and reported. The analytic toy was saved to `runs/analytic-toy.json`. These are implementation/analytic checks, not trained-model evidence.
- Documentation local-link and new CLI help checks passed; `git diff --check` passed. Git ignore checks cover weights, data, outputs and `.venv`.

Two-puzzle timing checks completed at depth 16: both native BF16 and FP32 solved 2/2, taking 21.4 s and 7.1 s respectively. These overlapping, tiny timing checks are not controlled performance measurements or accuracy reproduction.

The 32-puzzle baseline runs completed:

| Run | Step 16 exact accuracy | Final exact accuracy | Correct → wrong transitions | Wall time |
| --- | --- | --- | --- | --- |
| `pilot-native-32` (BF16, 16 steps) | 25/32 (78.125%) | 25/32 (78.125%) | 0/334 at-risk transitions | 461.1 s |
| `pilot-fp32-32-depth64` (FP32, 64 steps) | 25/32 (78.125%) | 27/32 (84.375%) | 0/1,608 at-risk transitions | 542.2 s |

Same dev IDs and batch size in both runs. One puzzle is correct only in native precision at step 16, and a different puzzle only in FP32: equal aggregate accuracy does not mean equal trajectories. The publisher's reported 84.80% uses a different evaluation population; 32 examples do not reproduce or refute it.

FP32 never lost correctness after first becoming correct (0/27 ever-correct puzzles). Relative to step 16, step 64 helps 2/32 and harms 0/32; paired mean change +6.25 percentage points, pointwise 95% puzzle-bootstrap interval [0, 15.625] pp. The two late recoveries are 2/7 of the puzzles still wrong at step 16. The pooled recovery count is 22/408 wrong-state transitions; these transitions are correlated, not independent trials. First-valid rule-verifier coverage is 27/32 with reference accuracy 27/27. Native BF16 continuation beyond step 16 has not been measured.

This sample shows late recovery without the natural retention failures needed to motivate the proposed repair. It does not establish that such failures are absent in the population. The two baseline runs overlapped, so wall times are not controlled speed measurements.

The eight-puzzle zero-epsilon run completed in 372.8 s. All 1,152 branch records exactly match the corresponding natural trajectory in predictions, correctness, verifier results, margins, Q logits, and logged state norms/residuals; every realized delta is zero. Natural solve rate at step 32 is 6/8. This validates the control, not a perturbation benefit.

### Final intervention results and audit — 2026-11-05

`pilot-targeted-8` completed in 520.6 s: eight dev puzzles, FP32, epsilon 0.01, forks 4/8/16, continuation horizon 16, probe horizon 2, clean/random ±/gradient ±. Every comparison has eight puzzles, with five correct and three wrong at each fork.

| Fork | Arm(s) | Final solve | Final harm / help vs clean | Loss given correct at fork | Recovery given wrong at fork | Any prediction difference vs clean |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | Each of random ± and gradient ± | 6/8 | 0/8 / 0/8 | 0/5 | 1/3 | 3/8 |
| 8 | Gradient plus | 5/8 | 1/8 / 0/8 | 0/5 | 0/3 | 3/8 |
| 8 | Each of random ± and gradient minus | 6/8 | 0/8 / 0/8 | 0/5 | 1/3 | 3/8 |
| 16 | Each of random ± and gradient ± | 6/8 | 0/8 / 0/8 | 0/5 | 1/3 | 2/8 |

Rows grouping arms report each arm's count, not pooled independent samples. The step-8 gradient-plus final solve difference is −12.5 percentage points, pointwise 95% puzzle-bootstrap interval [−37.5, 0] pp. Other final solve differences are zero in this sample; degenerate bootstrap intervals do not establish population equivalence. No correct-at-fork puzzle changed its predicted answer relative to clean at any observed branch offset.

Post-hoc case inspection: puzzle `e524867ce8ff8ad0f9697194a08d65e3d935a8a3ff432cba98eac9c83585364b` is wrong at all three forks and naturally first solves at step 17. Gradient plus at fork 4 solves at step 7; at fork 8 it remains wrong through the branch endpoint at step 24; at fork 16 it solves at step 17. Random directions and gradient minus sometimes accelerate this same recovery as well. This is an observed effect on **recovery within a finite window**, not destruction of an already correct answer, proof of permanent failure, a gradient-specific mechanism, or a replicated stage effect. The probes use current predictions, never reference labels.

All 96 nonclean sample/fork/arm perturbations have nonzero realized deltas and pass the norm tolerance. Actual relative norms range from `0.009999998845160007` to `0.010000001639127731`. No examples are excluded, so full and common norm-qualified cohorts coincide. All immediate logits are preserved exactly. The targeted run charges 256 natural, 1,920 continuation, and 48 probe forward sample-macros, plus 6 backward batch calls / 24 sample objectives.

Executed final checks:

- All four primary manifests report `complete`. Each summary was independently recomputed with `analyze`; reports also check equality against raw-record recomputation.
- `PYTHONPATH=. .venv/bin/python runs/audit_pilot.py` passed: common/nested puzzle IDs, identical checkpoint, exact zero-control observables, clean branches matching natural paths, and natural prefixes matching across FP32 runs. Evidence and artifact hashes are in `runs/pilot-audit.json`.
- BF16 and FP32 step-16 decoded predictions match on 24/32 puzzles. Correctness differs on two; wrong answers differ on additional puzzles. Of the 25 correct at FP32 step 16, 0/25 lose correctness through step 64; 2/7 remaining wrong puzzles recover.
- The final scientific/report suite passed **25 tests**, including a transient-effect example whose final predictions rejoin clean. All output directories have Markdown reports, machine-readable JSON, and accuracy CSVs. Sum of the four run wall times is 1,896.8 s, with overlapping runs; this is not total exclusive CPU time or a speed benchmark.

Decision: the narrow observation that current-output-preserving changes can affect later answers is supported by this small checkpoint diagnostic. A stage pattern appears in one post-hoc recovery case and needs replication and margin controls. The retention-repair motivation remains unsupported by this sample: natural continuation recovered answers and never lost a correct one. Increase diagnostic coverage before designing or training a selective stabilizer; do not increase attack strength merely to manufacture correct-answer failures.

### Comparison with InfiLoop Figure 1 — 2026-11-05

Checked [InfiLoop v2, Section 2 and Figure 1](https://arxiv.org/html/2610.11570v2#S2), including the PDF page-3 figure/caption. It reports 19.7% solved-puzzle loss after 36 further outer updates for its self-attention TRM. The aligned cohort contains puzzles both models solve within 36 updates; trajectories span 72 outer updates, with 26 layer evaluations per update. The separately reported 20.9% concerns initially correct **cells**, not solved puzzles.

Our pilot instead uses third-party MLP weights, L=6, and logs after each H=3 macro-step (42 block-layer evaluations), with FP32 continuation only. The 64 logged steps execute 2,688 block-layer evaluations. Thus insufficient total layer count is not an established explanation; architecture, checkpoint, precision, recurrence schedule, observation frequency, and cohort differ. Intermediate losses followed by recovery between our observation points could be missed. Of our 27 ever-correct puzzles, 26 first solve by macro-step 28 and have at least 36 further logged steps; this does not equate their cohort or update definition to the paper's.

The [checked code release](https://github.com/pixeli99/InfiLoop/tree/63716a9c46723d3592c8ad4fc5819855f22aaa47) documents an InfiLoop checkpoint, but its inspected tree/READMEs did not identify the Figure-1 TRM baseline checkpoint or figure-generation script. Exact Figure-1 precision and implementation details remain unverified. No reproduction or new model run was performed in this comparison.

Revised interpretation: the pilot's absence of observed losses is specific to its MLP configuration and observation grid. It should not weaken the published motivation without a matched diagnostic. Next compare attention weights, preserve full state, log every outer update, and measure both pointwise correctness and first-loss survival over equal post-solving windows. Reproduce the paper's joint-model cohort only with both trajectories available; otherwise label a TRM-only cohort explicitly. Keep original and altered recurrence schedules distinct and record both. Do not explain the discrepancy solely by sample size or solve it by strengthening attacks.

Checks: inspected paper Section 2/Figure 1/compute appendix and pinned repository tree/READMEs; verified local manifest configuration, executed-layer count, and first-correct follow-up counts from saved results. Documentation consistency checks passed; no experiment/model code changed.
