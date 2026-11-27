# Research roadmap

The [frozen-refinement study](TRM_REFINEMENT.md) is closed. Extending training to 4,096 updates reduced breaks but gave neither refiner a net gain over the native fork. Native TRM lost no exact answers through macro depth 16, and all 94 final misses were never solved. At equal native work, an early Gaussian split produced one solution outside the native trajectories and lost 21 serial solves. A depth-12 split added zero serial misses. A mixing experiment should keep the native update available and measure the incremental contribution of mixing. See [the duration and late-branch record](STATUS.md#27-november-2026--training-duration-and-late-branch-results). The 896 held-out inputs remain sealed.

The [learned hypothesis pilot](STATUS.md#27-november-2026--learned-native-hypotheses-pilot) traded one serial solve for one new solve. Both proposed digits were wrong on the new solve, and both branches reached the same exact answer. The next comparison trains native TRM, scalar gating and directional mixing end to end from identical shared random weights. Compare curves at equal exposure and equal training time, with matched inference latency. See the [locked comparison](STATUS.md#27-november-2026--end-to-end-native-mixing-comparison).

The 20 November [CRAM assessment](CRAM_REVIEW.md) revises the substantial study in notebook v3.1. Include score-only single- and two-token search on the same field, pair natural error recovery across controllers, and assess ordinary-start solve rate and retention. Workspace recomputation is conditional on a persistent-controller advantage. The underlying rule-free model below is still the candidate.

## Current direction — 16 November 2026, notebook v3.0

The main proposal is **learned latent residual coordination**, without symbolic Sudoku/SAT inference rules. The residual is split into geometric channels `g(q,x)` and a learned signed invalidity score `s(q,x)`. A bounded memory-conditioned SPD matrix mixes them before the update `q_next = q - alpha * J_c^T W c`, with `c=[g;s]`.

The concrete learning candidate combines local calibration near fixed generic answer embeddings, a signed invalidity margin on known incorrect candidates, and task loss through the actual recurrence. The signed margin replaces squared-energy ranking because it has nonzero output gradient at a wrong residual zero. It does not guarantee correct zeros, useful derivatives, or successful training. Use `s=sqrt(S)*s_bar` and documented per-position loss normalization for variable output length; this is a scale convention, not a transfer guarantee.

The decisive rivals include direct `q-g` correction, an invalidity-gated learned proposal `q-sH`, and a direct learned residual map `q-Gc`, alongside random/stateless mixing and established solvers. First compare controllers on a shared frozen learned field, then jointly trained systems whose competitors can adapt their own fields. Hardcoded residuals are optional controls. The attracting semantic-manifold branch remains separate from this anchor–workspace construction.

Read [the cleaned notebook](notebook/project_notebook.pdf), especially Sections 2–3 (architecture and learning) and Section 5 (study). [Version 2.1](notebook/archive/project_notebook_v2_1.pdf) preserves the full previous record. The [training runbook](RESIDUAL_TRAINING.md) tracks the implemented architecture and its pending GPU measurements. No new empirical gain or publication endorsement is claimed.

## Implemented on the training branch

Commission the branch on one GH200 and then check the intended distributed topology. The development pilot covers verified-unique Sudoku and modular prefix sums. Read its per-update cost, memory, training curves, wrong-start counts and correct-start retention before choosing a training budget for the full study. The full profile includes two frozen fields, independently trained alternatives, extra-loop and length tests, and the Section 5 mechanism probes. E4/E5 remain optional historical work.

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
