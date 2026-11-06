# Roadmap and decision gates

Updated after the completed pilot and close-prior review, 2026-11-05. Earlier plans to scale small perturbations or immediately build a risk predictor are superseded by this order.

## 1. Run the fixed clock study

Follow [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md): 1,024 new dev puzzles, two pinned checkpoints, two precisions, native-state forks, and equal recurrent-block budgets. This is one coherent study with a primary comparison and three prespecified secondary conditions. Code, launchers and independent reporting are implemented; the trained study is not yet run.

The study separates native output failures, intermediate excursions, and continuation-schedule effects. It is not an exact InfiLoop Figure-1 reproduction. Preserve its fixed sample, pre-fork cohorts, endpoint phases, source hashes and stopping rules. Do not run additional epsilon/horizon sweeps in parallel merely to find a striking case.

**Gate:** interpret the primary interval against the fixed +2 pp threshold. Separately require evidence of native macro-output losses before motivating a native retention repair. A failure created only by L=12 does not satisfy that requirement. Informative nulls close this direction for the tested conditions.

## 2. Follow the supported mechanism, if any

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
