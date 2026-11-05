# Roadmap and decision gates

## 1. Establish a trained baseline

Obtain a trusted original-TRM Sudoku checkpoint and its resolved config, or reproduce the upstream recipe within an agreed GPU budget. Record training seed, data, EMA use, steps, and source commit. Reproduce normal-horizon accuracy before extending the trajectory. Compare native precision and FP32. Pilot 128 dev puzzles, then 512 if runtime permits.

**Gate:** enough correct states and natural later errors to study retention, without unexplained reproduction discrepancies. If natural retention failures are negligible, report that result; do not manufacture relevance using extreme attacks.

## 2. Characterize sensitivity

Run zero epsilon, several small amplitudes, random antithetic directions, and multiple noise seeds. Add gradient arms with evaluation horizons longer than probe horizons. Inspect realized budgets, precision, stage, margins, and paired harm/help. Bootstrap by original puzzle; report checkpoint-seed variance separately. Match stage and margin before introducing more elaborate covariance-aware perturbations.

**Gate:** reproducible effects beyond budget/precision artifacts. Off-trajectory attacks alone do not explain the cause of natural failures.

## 3. Test prospective value

Build a label-free score at t using fixed probe horizon H; predict spontaneous changes or correctness loss strictly after t+H. Fit thresholds/models only on dev. Compare margin, margin trend, Q score, prediction stability, and residuals. Charge probe computation against equal-cost continued reasoning and verifier checks.

**Gate:** held-out value survives simple controls. Otherwise narrow to a diagnostic result or stop the proposed repair path.

## 4. Test selective stabilization only after evidence

Candidate: future-margin robustness on suitable training states, combined with the original task loss and opportunities to correct wrong answers. Ground-truth gating can be used in training, not assumed at deployment. The exact loss and state-sampling policy are not implemented or fixed yet.

Required comparisons at matched training budget:

- Unmodified TRM and extra ordinary supervision at the same depths.
- Random training depth/position where task structure permits.
- Isotropic noise and generic adversarial training, including full-state perturbations.
- Simple damping or smoothing, evaluated for both solving and retention.
- Spectral/Jacobian penalties and normality/commutativity regularization.
- Q-based, prediction-stability, and rule-verifier stopping where available.

**Gate:** improve the final solve/retain/compute tradeoff across training seeds. Reject an approach that merely freezes wrong answers or damages recovery.

## 5. Generalize and position

Add Maze or a controlled task with intermediate targets, then another model only if warranted. FPRM and InfiLoop require full solver/accumulator state and schedule accounting. Compare single trajectories with single trajectories; charge multi-schedule search fully.

Revisit QRR, retention/reuse, and neural-simulator stabilization as closest competing explanations. An A* conference is an ambition, not something an agent review can certify.

## Immediate Codex tasks

1. Audit the first bounded CPU checkpoint pilot in `STATUS.md` and expand to 128 dev puzzles only with a suitable runtime budget. Compare normal-horizon precision effects before interpreting continuation.
2. The JSONL report and per-fork puzzle-paired intervals are implemented. Add plots and clustered uncertainty for pooled forks/noise seeds when expanding the study.
3. Implement prospective-risk evaluation with a strict time gap beyond the probe and held-out baseline models.
4. Generalize the backend only after the Sudoku pilot establishes a reason.

No learned manifold estimator, novel training loss, FPRM/InfiLoop adapter, or validated stopping policy is included in the first implementation.
