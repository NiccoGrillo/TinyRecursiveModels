# Research roadmap

## Current architecture direction — 12 November 2026, second research session

The leading candidate is now **solution-preserving residual mixing**: `q_next = q - alpha * J_c(q)^T W(q,w,x)c(q;x)`, with bounded positive-definite, low-rank residual-space mixing. It can move at some wrong stationary points that the previous state-side preconditioner and diagonal constraint weights cannot leave. Every common residual zero remains fixed. This is a distinction between update classes, not a claim of a new general optimizer: Sketched Newton–Raphson is a direct predecessor, and random residual mixing is a mandatory control.

The first full architecture is a grounded Sudoku factor-graph reasoner: Boolean, cell, row, column, box and clue residuals define valid solutions; a bounded per-factor workspace predicts their low-rank interactions. The controller is learned, the initial task residuals are specified. Generic learned latent residuals remain a later branch. No complete architecture has been trained or benchmarked. A tested PyTorch operator prototype is in `experiments/residual_reasoning/operators.py`.

Start with Section 1 of [notebook version 2.1](notebook/project_notebook.pdf), or its [LaTeX source](notebook/project_notebook.tex), for four adversarial critique rounds, the supplied QRR review assessment, exact proofs, limitations and the decisive comparison. The prior energy-preconditioner is retained as a baseline. E4/E5 remain optional historical diagnostics and do not gate architectural work.

## Next substantial task

Build the grounded Sudoku residual model and shared factor-graph controller. Compare state-side preconditioning, diagonal adaptive weights, fixed full residual mixing, stateless learned mixing, recurrent learned mixing, fresh random mixing, Sketched Newton–Raphson and a strong unrestricted factor-aware learned solver. Match information, initialization, training resources and total inference cost; separately allow strong controls to learn their own residual basis. Keep optional satisfiable-3SAT transfer distinct from UNSAT decision.

First compare fixed-compute solving with no adaptive stopping. Then cross architectures with a shared deployable readout rule and compare their native rules separately. Calibrate on development data; include failures and cap hits. Task sizes, seeds, data and accelerator budget must be frozen before training. This is one substantial architecture study, not a new sequence of small pilot gates.

Stop the memory-coordination claim if recurrent mixing fails to beat random/stateless mixing at comparable cost or if a learned static basis removes its advantage. The critic accepted the mathematics and a bounded architecture comparison, not A* novelty or demonstrated gains. Preserve the attracting-answer-set branch; regular semantic solution manifolds remain a separate question from discrete valid assignments times workspace.

## Historical roadmap — superseded as current priority

# Roadmap and decision gates

Updated after the completed pilot and close-prior review, 2026-11-05. Earlier plans to scale small perturbations or immediately build a risk predictor are superseded by this order.

## 1. Run the fixed clock study

Follow [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md): 1,024 new dev puzzles, two pinned checkpoints, two precisions, native-state forks, and equal recurrent-block budgets. This is one coherent study with a primary comparison and three prespecified secondary conditions. Code, launchers and independent reporting are implemented; the trained study is not yet run.

The study separates native output failures, intermediate excursions, and continuation-schedule effects. It is not an exact InfiLoop Figure-1 reproduction. Preserve its fixed sample, pre-fork cohorts, endpoint phases, source hashes and stopping rules. Do not run additional epsilon/horizon sweeps in parallel merely to find a striking case.

**Gate:** interpret the primary interval against the fixed +2 pp threshold. Separately require evidence of native macro-output losses before motivating a native retention repair. A failure created only by L=12 does not satisfy that requirement. Informative nulls close this direction for the tested conditions.

## 2. Follow the supported mechanism, if any

[RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md) records the critic-reviewed candidate: coupled feedback and timing, with an additive-core counterexample, exact fixed-state limitations and a conditional comparison at equal low/high update counts. This is a hypothesis and analytic illustration, not a demonstrated trained-model mechanism. FPRM Appendix F and TAPS rule out generic allocation/adaptive-scheduling novelty.

If a useful schedule effect survives the gate, first replicate it on an independent cohort and another checkpoint. To distinguish reachability from continuation stability, consider a source-schedule × continuation-schedule crossover only at this stage. Do not infer architecture causality from one differently trained checkpoint per architecture.

If a label-free schedule policy is justified, compare it against fixed schedules, equal-cost ordinary continuation, margin/residual/Q/answer-stability rules, TAPS, and Sudoku verifier stopping. Charge every probe, branch and readout. Tune only on dev; the public benchmark split cannot undo upstream test-selection bias.

**Gate:** prospective information improves final correctness or compute at matched costs, without oracle gating and without sacrificing recovery. A larger late-solving fraction alone is already covered by close work.

## 3. Train only after a demonstrated need

Selective stabilization remains conditional on a native retention problem and predictive/intervention value beyond simple baselines. Compare any proposed loss with unmodified TRM, extra ordinary supervision, random depth, isotropic/adversarial robustness, damping, and relevant Jacobian/normality regularization. A repair must improve solve/retain/compute tradeoffs across training seeds, not merely freeze wrong answers.

If native outputs remain stable, stop the retention-repair objective for this setup. A separate recovery/allocation project needs a new explicit hypothesis and a fresh novelty review; do not silently relabel a null as support for the old idea.

## 4. Publication-scale validation

Require multiple training seeds/checkpoints, another task or model family, complete cost accounting, independent evaluation data where feasible, and comparisons with the closest current work in [RESOURCES.md](RESOURCES.md). FPRM/InfiLoop need explicit full-state adapters and their own solver/accumulator/schedule accounting. A diagnostic discrepancy is not by itself a new method or an A* contribution.

## Immediate Codex task after the run

Audit all four manifests and raw summaries, verify common puzzle IDs, and update `STATUS.md` with primary and secondary effects, exact uncertainty, native macro retention, recovery, and artifact hashes. Apply the prespecified gate before proposing another experiment. If the interval is inconclusive, explain what effect sizes remain possible; do not pretend more of the same is automatically worthwhile.
