# Status

## 25 November 2026 · Matched-update prefix-sum curriculum

Isambard job `8857953` completed in 8m 33s at revision `74a08dc8e1bdf9dd4c10531944c7e8537c11fa38`. On one GH200, the ordinary-descent reference trained for 3,072 updates either entirely at length 16 or in three equal phases at lengths 4, 8 and 16. Both arms used seed 11, batch 64, four supervised reasoning loops, the pilot objective and the same 512 length-16 development examples. The curriculum retained its weights and AdamW state between phases. The match is in updates. Sequence length changes the number of training tokens per update.

At four reasoning steps, length-16 dev token accuracy was 27.40% for direct training and 35.35% for the curriculum. Position four improved from 103/512 to 410/512 correct. Both arms had zero exact length-16 solves at all recorded depths. At depth 16, token accuracy fell to 23.86% and 24.60% respectively. Training took 223.39 s and 220.66 s, with peak PyTorch allocation of 0.165 and 0.167 GiB. This is one-seed development evidence of improved partial predictions with the chosen schedule. Extra inference steps degraded token accuracy after the four training loops.

The [diagnostic runner](../scripts/run_prefix_curriculum.py) used SHA-256 `b2d42efd87ba37620a915196d9015d3c8594e48a5c9c7e11614cb9b47b203d81`. Its direct arm reproduced the pinned trainer's weights and optimizer state bitwise in a three-update CPU check. Final checkpoint SHA-256 values were `f037a8d6a51ed071ccf5ddadf4f162c2e3563ce9e999a61a73c8d34e3f7e14bc` for direct training and `3788263e160d38e603e8728e30b802cf3f9ae7fda852c6b16bf8232e90f9b446` for the curriculum. The next bounded comparison should test additional length-16 training and vary the supervised unroll depth separately. The multi-seed controller comparison is still pending.

## 25 November 2026 · Development pilot

Isambard run `pilot-74a08dc-ZCDQLY` used revision `74a08dc8e1bdf9dd4c10531944c7e8537c11fa38`, seed 11, and the acceptance image. Final report job `8853665` finished 6m 56s after the first reference jobs started. The sealed run produced 16 model records and 20 development evaluations with complete coverage. The recurrent joint checkpoints were `82528e1e6c90ba32e837b2311c3ff44e1ef4f388ff3bb43eccf092e222b33642` (Sudoku) and `0f9753483e8f038a9909909ff5083bfb73cf94afafdea95e6168e01287ea924f` (prefix sums). All checkpoint hashes are in the archived `reports/training.csv`.

Every arm had zero exact solves on its 512-example dev cohort, both under the common wall budget and at every recorded depth through 16. The 24,576 input-by-arm-and-fork records contained zero exact hits, nonfinite trajectories, or wall-cap hits. The wall-budget recurrent mixer averaged about two steps at fork 4, compared with four for ordinary descent and about eight for direct correction. The correct-start cohorts were empty. Paired comparisons and memory interventions also had zero exact repair events, so this run cannot distinguish their effectiveness. Joint recurrent training used 32.0 s and 0.979 GiB peak PyTorch allocation for Sudoku and 26.1 s and 0.195 GiB for prefix sums, each across 250 updates.

Replaying the sealed checkpoints locally on the original dev inputs at depth four showed 13.71% accuracy on blank Sudoku cells for the jointly trained recurrent model, compared with 11.09% for predicting the most common training blank digit while copying clues. For prefix sums the corresponding accuracies were 18.59% and 18.13% for copying the first token and guessing zero elsewhere. Token metrics are now recorded at every evaluation policy endpoint, with a separate editable-token denominator. Rebuilding the archived report preserved all 384 original metric rows and the paired results.

A local CPU follow-up trained only the ordinary-descent reference from scratch on modular prefix sums at lengths 4, 8 and 16. Each run used the pilot architecture and objective, 2,048 training examples, 512 dev examples, data seed 2809, training seed 11, and 1,024 updates at batch 64. After four reasoning steps the exact counts were 175/512, 0/512 and 0/512, respectively. Token accuracies were 80.91%, 38.65% and 21.41%. The length-8 run was added after the length-4 result. These are separate fixed-length runs and provide no length-transfer estimate. The next bounded test is to compare a short-to-long curriculum against direct length-16 training at a matched update budget and measure per-position accuracy and exact solves before expanding the controller comparison.

## 25 November 2026 · Single-GH200 acceptance

The Isambard-AI acceptance passed at `74a08dc8e1bdf9dd4c10531944c7e8537c11fa38`. Image preparation job `8851688` completed in 7m 05s. Acceptance job `8851773` completed in 48s with exit code `0:0`. Its reports cover 18 model records and 66 split-level evaluations with zero missing records.

The synthetic FP32 recurrent update used a batch of 64, 81 Sudoku positions and four supervised loops on an NVIDIA GH200 120GB. It took 1.006 s and reached 1,048,961,024 bytes (0.977 GiB) of peak PyTorch-allocated GPU memory. The image SHA-256 was `b5524840a75f6d6dba8110f8e2b48389bb9baf841a0bd3147cb7904d12832b4a`.

The acceptance used synthetic inputs and establishes hardware and workflow readiness. Multi-GPU operation remains unchecked. The pinned Sudoku subset and modular prefix-sum data are prepared and staged on Isambard. Checks passed for split sizes, exact input-ID disjointness and transferred file hashes. The one-seed, dev-only development pilot plan sealed locally against these data. Its measured training and evaluation costs will inform the multi-seed study budget.

## 24 November 2026 · Training branch implementation

Implemented notebook Sections 2, 3 and 5 in `experiments/residual_reasoning/`. CPU verification covers the full higher-order training path, frozen-field transfer, shared-state repair, all solver families, Sudoku fixtures, exact interrupted-run replay and reports over a miniature campaign. Codex setup and single-node Slurm/Apptainer launchers are in the [runbook](RESIDUAL_TRAINING.md). The repository suite passed 72 checks locally, including 18 model records and 66 split-level evaluations in the complete fixture campaign. The Slurm submission dependencies were exercised with a local scheduler substitute. GH200 execution, NCCL, production data preparation, pilot timing and scientific outcomes remain pending. The November 20 notebook records the earlier design snapshot.

## 20 November 2026 · CRAM checkpoint audit and notebook v3.1

The experiment table and decision rules in Section 5 now include the score-only search comparison directly. They specify how to choose the repair cohort, measure retention, and interpret a tie at equal cost. This is an editorial clarification of the proposed study.

We checked the three published CRAM Sudoku checkpoints at upstream revision 62bbe830. For every full board with clues clamped, exact energy is at most zero for valid completions and at least 0.2671, 0.2963 and 0.2251 for invalid boards respectively (lower bounds rounded down). An integer-arithmetic audit and an independent exact-rational calculation agreed. The authors’ fitted thresholds lie in this gap. This establishes recognition under exact discrete scoring. The search can still exhaust its budget. The Maze-Unique result gives a complementary case where initialization matters despite perfect test solving.

Three design critiques and a fourth proof check changed the proposed study. Use untouched original-input rollout errors for paired repair, plus ordinary-start solve rate and correct-state retention. Include learned-score search with one- and two-token moves and strong cheap neural correctors. Compare balanced corruptions with equal-distance training controls. Test workspace recomputation only if persistent memory earns its extra cost. No v3 learner or benchmark was run. [Notebook §5](notebook/project_notebook.pdf) · [Full review, proof and reproduction](CRAM_REVIEW.md)

## Current direction — 16 November 2026, notebook v3.0

The main proposal is **learned latent residual coordination**, without symbolic Sudoku/SAT inference rules. The residual is split into geometric channels `g(q,x)` and a learned signed invalidity score `s(q,x)`. A bounded memory-conditioned SPD matrix mixes them before the update `q_next = q - alpha * J_c^T W c`, with `c=[g;s]`.

The concrete learning candidate combines local calibration near fixed generic answer embeddings, a signed invalidity margin on known incorrect candidates, and task loss through the actual recurrence. The signed margin replaces squared-energy ranking because it has nonzero output gradient at a wrong residual zero. It does not guarantee correct zeros, useful derivatives, or successful training. Use `s=sqrt(S)*s_bar` and documented per-position loss normalization for variable output length; this is a scale convention, not a transfer guarantee.

The decisive rivals include direct `q-g` correction, an invalidity-gated learned proposal `q-sH`, and a direct learned residual map `q-Gc`, alongside random/stateless mixing and established solvers. First compare controllers on a shared frozen learned field, then jointly trained systems whose competitors can adapt their own fields. Hardcoded residuals are optional controls. The attracting semantic-manifold branch remains separate from this anchor–workspace construction.

Read [the cleaned notebook](notebook/project_notebook.pdf), especially Sections 2–3 (architecture and learning) and Section 5 (study). [Version 2.1](notebook/archive/project_notebook_v2_1.pdf) preserves the full previous record. Only the mixing/VJP operator exists; the new residual heads, losses, controller and full benchmark are **unimplemented and untrained**. No new empirical gain or publication endorsement is claimed.

## Documentation and critique update — 16 November 2026

Completed one further adversarial design cycle and rewrote the guide around the current architecture, learning objective, mathematical limits and decisive study. Archived the prior 60-page PDF/source byte-for-byte. Retained the attracting-answer-set explanation in the new appendix and the stable E0–E5 ledger. This was a specification/documentation update, not an experiment or new training implementation.

The cycle rejected global denoising regression as a default (an ideal direct corrector can solve it immediately), and replaced squared-energy contrastive negatives with a signed learned invalidity margin. A wrong candidate's zero score receives nonzero output-level gradient, without teaching the globally correct direction. The critic required cheaper verifier-gated/direct correction baselines, explicit length scaling, and a distinction between full-column-rank anchor geometry and full-row-rank semantic manifolds. Practical usefulness and novelty remain open.

Validation: source/PDF rebuilt with XeLaTeX; references, local documentation links, archived-file identity, whitespace and rendered layout checked. No training or new benchmark run was launched; the previously reported eight operator tests remain implementation evidence only.

## Historical architecture decision — 12 November 2026 (superseded by v3.0)

The leading candidate is now **solution-preserving residual mixing**: `q_next = q - alpha * J_c(q)^T W(q,w,x)c(q;x)`, with bounded positive-definite, low-rank residual-space mixing. It can move at some wrong stationary points that the previous state-side preconditioner and diagonal constraint weights cannot leave. Every common residual zero remains fixed. This is a distinction between update classes, not a claim of a new general optimizer: Sketched Newton–Raphson is a direct predecessor, and random residual mixing is a mandatory control.

The first full architecture is a grounded Sudoku factor-graph reasoner: Boolean, cell, row, column, box and clue residuals define valid solutions; a bounded per-factor workspace predicts their low-rank interactions. The controller is learned, the initial task residuals are specified. Generic learned latent residuals remain a later branch. No complete architecture has been trained or benchmarked. A tested PyTorch operator prototype is in `experiments/residual_reasoning/operators.py`.

Start with Section 1 of [notebook version 2.1](notebook/project_notebook.pdf), or its [LaTeX source](notebook/project_notebook.tex), for four adversarial critique rounds, the supplied QRR review assessment, exact proofs, limitations and the decisive comparison. The prior energy-preconditioner is retained as a baseline. E4/E5 remain optional historical diagnostics and do not gate architectural work.

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

## Historical next step — superseded 12 November 2026

The first bounded CPU pilot is complete. Run the fixed [same-state clock study](NEXT_EXPERIMENT.md), implemented after the evidence and close-prior review below. Preserve dense-versus-native output semantics and distinguish schedule intervention from exact InfiLoop reproduction. Do not expand perturbation sweeps or train a selective stabilizer before this gate.

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

## Next-study implementation and critic review — 2026-11-05

Reviewed the pushed result summaries at commit `98689fb58da8bcb2791820f5674bd13d77ffc61b`. The previous raw runs and MLP/data assets were not present in this workspace; their reported outcomes were not independently recomputed in this turn. The evidence still favors recovery sensitivity over a demonstrated retention defect.

The literature review identified [When Recursive Models Finish Computing](https://arxiv.org/abs/2609.26487v1), a direct overlap that includes the pilot's MLP checkpoint, and [TAPS](https://arxiv.org/abs/2609.36653), relevant to any future inference controller. Added them prominently to [RESOURCES.md](RESOURCES.md). The original broad diagnostic cannot carry the novelty claim by itself.

Implemented the [fixed next study](NEXT_EXPERIMENT.md): dense native observation plus a causal continuation-schedule fork from the identical full state, equal block budgets, separate pre-fork retention/recovery cohorts, explicit native macro retention, exact paired binary bounds, and independent raw-record report verification. The primary condition is attention BF16; three secondary checkpoint/precision conditions share the same 1,024 new dev IDs. The 2 pp decision threshold and informative-null stopping rules are fixed in the runbook. No new training loss or larger perturbation sweep was added.

Added a hash-pinned attention downloader and a four-condition launcher with frozen-asset, architecture, strict-loading, device and output checks. Downloaded attention revision `33991b882368f425916d625a9ee614a7a85efe76` locally; its published model source is byte-identical to pinned upstream. Both BF16 and FP32 load strictly, and one native macro on two synthetic inputs agrees bitwise with three dense outer updates in both state tensors and logits. These checks use the real weights but are **compatibility checks, not benchmark measurements**. Asset hashes are in the next-study runbook; training/EMA provenance is not independently verified.

Executed checks:

- `.venv/bin/python -m pytest -q`: **35 passed** on Python 3.12.14 / torch 2.7.0+cpu. Covers original tests plus native/dense parity for MLP/attention and both precisions, exact equal-cost arithmetic, pre-treatment cohorts, fixed first-solve follow-up, dense-versus-macro error cases, intermediate endpoint phases, nondegenerate zero-event uncertainty, complete CLI accounting, report recomputation and corruption rejection.
- Pinned attention fetch helper verified all three assets; downloaded Python was not executed.
- Launcher shell syntax, new CLI help, documentation local links and `git diff --check` passed.
- Upstream model, training and dataset-builder source remains unchanged. Full raw datasets, weights and generated outputs remain ignored.

The critic reviewed the proposal, then the implementation and final protocol. It found no remaining scientific blocker after corrections to labeling, phase reporting, uncertainty and provenance checks; [CRITIC_REVIEW.md](CRITIC_REVIEW.md) records the objections. Its approval is for a consequential falsifiable experiment, not a novelty or publication guarantee.

**Not run:** the four-condition trained-checkpoint study. This host has no CUDA device; only implementation and compatibility checks were executed. The next action is the documented full run on suitable compute, followed by the prespecified decision gate. No scientific outcome for the new study is claimed.

## Further research and analytic deductions — 2026-11-06

The reviewed clock-study implementation was published as `2ccf2814a70a929cd3f91b6073513fda3b4c7842`; its [GitHub checks passed](https://github.com/NiccoGrillo/TinyRecursiveModels/actions/runs/42145692270). The user requested continued work with the critic on new insights. [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md) records the resulting narrow hypothesis, proofs, limitations and conditional experiment.

Derived the fixed-set inclusion `Fix(B A^L) ⊆ Fix(B A^(mL))` for TRM's alternating low/high update structure. An exact common equilibrium cannot be escaped by scheduling unchanged updates alone. Constructed an analytic example respecting shared additive input injection where the same equilibrium is attracting at L6, unstable at L12, and stabilized by a predicted high-update damping range. At equal low/high call counts, alternating 3/9 reduces the spectral radius of the linearized period map relative to 6/6; reversed order has the same eigenvalues but different transient gain.

These are elementary mathematical deductions and an **abstract scalar example**, not a literal Transformer weight construction, trained-model evidence, a new numerical-analysis theorem or an established reasoning improvement. The critic caught the additive-injection constraint in an earlier candidate and required the corrected construction and the phase-shift caveat. Generic allocation, adaptive scaling, invariant-measure readouts and feedback steering overlap existing work; the resource map now includes FPRM Appendix F, ImR, RoFB, ICS and classical coupled-solver analysis. ICS's primary abstract was retrieved, but its detailed claims still need full review.

Added executable `coupled_toy.py` and independent finite-difference checks through its nonlinear recurrence. `.venv/bin/python -m pytest -q`: **39 passed**. The toy command reproduced the documented spectra, equal-count comparison and damping threshold. Local documentation links and whitespace checks pass. This is analytic/implementation verification, not another small benchmark run. The original clock-study settings and decision gates are unchanged; no speculative schedule or damping method has been applied to the trained checkpoints.

## Living LaTeX project guide — 2026-11-06

Added [the project guide PDF](notebook/project_notebook.pdf) and its [self-contained editable LaTeX source](notebook/project_notebook.tex), with [build/update instructions](notebook/README.md). It is now the README's first reading recommendation. `AGENTS.md` instructs future work to update the guide and rebuild the PDF whenever results, theory or research decisions change.

The 30-page guide explains the actual two-state TRM computation, reconciles macro/outer/block units, traces completed pilot runs and their limitations, distinguishes recovery from retention with the recorded puzzle case, and introduces the coupled-feedback hypothesis before its derivations. The original transient example and conditional margin-bound idea remain documented. The frozen unrun clock study, conditional equal-count follow-up, prior-work boundaries, exact provenance, commands, glossary and update template are included. Mathematical proofs and implementation checks are explicitly separated from trained-model evidence. The empirical tables reproduce the committed reports; no fresh raw-data reanalysis or model experiment was performed in preparing this document.

Validation: XeLaTeX build completed with all references/citations resolved and no overflow warnings; all 30 rendered pages were visually inspected. Repository links in the LaTeX source resolve, shell syntax and `git diff --check` pass. Experiment code and the frozen study are unchanged; the prior 39-test result is reported as prior validation, not a new test run.

## Notebook v2 and architectural pivot — 2026-11-12

Completely reorganized the living LaTeX document around the persistent-workspace energy architecture. Added definitions, bounded metric construction, stationary-fiber and common-descent proofs, PL versus state convergence, decoder margins, concrete training and cost accounting, critic corrections to the QRR suggestions, and an expanded attracting-answer-set appendix. Preserved all detailed historical pilot results, TRM feedback calculations, E4/E5 protocols, uncertainty and provenance.

The critic audited the mathematics and novelty boundaries. The clock study is now optional, not a prerequisite; this is an explicit user-directed research pivot, not an empirical conclusion. New architecture implementation/training and the larger clock benchmark remain unexecuted. Validation for this documentation change: three-pass XeLaTeX compilation, reference/overflow checks, PDF page rendering and visual inspection, and repository diff/link consistency checks. No model tests or benchmarks were claimed or needed for this documentation-only update.

## Residual-mixing research session — 2026-11-12

Completed four adversarial architecture-review rounds and read the user-supplied QRR reviews. Replaced the sole-leading state-preconditioner proposal with a residual-space cross-factor controller hypothesis. Added exact class-separation example, common-root-family quantifiers, conditional local point/manifold attraction, Gaussian random-sketch control, direct SNR overlap and fixed-basis absorption objection to notebook v2.1. The old architecture and all historical data remain available.

Specified a first grounded Sudoku architecture: explicit Boolean/cell/unit/clue residuals and a learned bounded factor controller. Common residual zeros correspond exactly to valid completions and have full-column-rank Jacobians. Optional 3SAT transfer is witness finding. Full controller and task residual modules are not implemented; no training, GPU run or new accuracy result occurred.

Implemented isolated `experiments/residual_reasoning/operators.py` and focused tests in `tests/residual_reasoning/test_operators.py`. These check the bounded low-rank operation, exact wrong-minimum counterexample, zero preservation, J=0 limitation, and outer automatic derivatives through both residual and mixing parameters. Focused validation and document build results are recorded after final checks below. No existing TRM code or historical experiment protocol changed.

Final focused validation: **8 operator tests passed** on CPU, including rejection of disconnected residual graphs and zero outer gradients for graph-connected constant residuals. These are implementation checks only. Notebook v2.1 was compiled with three XeLaTeX passes, checked for references/overflows and rendered for visual review; repository diff/local-link checks completed.
