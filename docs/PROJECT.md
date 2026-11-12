# Project: decision stability during continued reasoning

## Current architecture direction — 12 November 2026, second research session

The leading candidate is now **solution-preserving residual mixing**: `q_next = q - alpha * J_c(q)^T W(q,w,x)c(q;x)`, with bounded positive-definite, low-rank residual-space mixing. It can move at some wrong stationary points that the previous state-side preconditioner and diagonal constraint weights cannot leave. Every common residual zero remains fixed. This is a distinction between update classes, not a claim of a new general optimizer: Sketched Newton–Raphson is a direct predecessor, and random residual mixing is a mandatory control.

The first full architecture is a grounded Sudoku factor-graph reasoner: Boolean, cell, row, column, box and clue residuals define valid solutions; a bounded per-factor workspace predicts their low-rank interactions. The controller is learned, the initial task residuals are specified. Generic learned latent residuals remain a later branch. No complete architecture has been trained or benchmarked. A tested PyTorch operator prototype is in `experiments/residual_reasoning/operators.py`.

Start with Section 1 of [notebook version 2.1](notebook/project_notebook.pdf), or its [LaTeX source](notebook/project_notebook.tex), for four adversarial critique rounds, the supplied QRR review assessment, exact proofs, limitations and the decisive comparison. The prior energy-preconditioner is retained as a baseline. E4/E5 remain optional historical diagnostics and do not gate architectural work.

## Historical specification (preserved)

## Question and hypothesis

Can a recurrent reasoner keep a correct decoded answer stable while its internal state continues useful computation? Would controlling directions that threaten future decisions preserve more useful computation than contracting all latent directions?

The first testable hypothesis is narrower: **directions that preserve the current output can have different effects on later decisions at different stages of reasoning.** A useful result would connect these effects to natural retention failures beyond simple margin baselines, then motivate a repair that preserves correction of wrong answers. The first exploratory checkpoint diagnostics are recorded in [STATUS.md](STATUS.md); no replicated stage-dependent mechanism or selective-stabilization gain is established.

The 2026-11-05 evidence and literature review change the immediate priority. The pilot supports finite-window recovery sensitivity but not a native retention defect. Close completion/stability work also overlaps the generic diagnostic. [The next experiment](NEXT_EXPERIMENT.md) therefore tests a specific computation-allocation effect using identical starting states and equal block budgets. The original selective-stability hypothesis remains conditional; no new manifold or loss is proposed merely to preserve that framing.

## Necessary distinctions

A fixed-point equation need not have one globally unique solution; it can have multiple equilibria or a continuum. Uniqueness requires additional assumptions. TRM itself does not enforce global fixed-point convergence.

Many latent states can decode to one answer. That is different from multiple valid solutions. Sudoku answers are discrete, so a set of solutions is not automatically a smooth manifold. We begin with output fibers and finite-time decision stability, making no smooth-manifold claim.

Quotient reasoning, attractor networks, partial/output contraction, stochastic stopping, and recurrent stabilization already cover substantial parts of the broad intuition. See the [resource map](RESOURCES.md).

## TRM formulation

At a fixed phase,

$$s_t=(z_{H,t},z_{L,t}),\quad s_{t+1}=F_\theta(s_t;x),\quad \ell_t=R_\theta(z_{H,t}).$$

One $F_\theta$ application completes all H/L cycles. The input and puzzle identifier remain fixed. The ACT wrapper's reset is bypassed.

An intervention $\delta=(0,\delta_L)$ preserves **all current logits exactly**. Its future effect can nevertheless be

$$\Delta\ell_{t+k}\approx D(R_\theta\circ F_\theta^k)(s_t)(0,\delta_L).$$

The current readout fiber need not be forward invariant. An intersection of derivative kernels through a finite horizon is a local linear approximation, not a discovered manifold or a longer-horizon guarantee.

For cell $j$ and anchor token $a_j$, use the logit margin

$$m_{k,j}(s)=\ell_{k,j,a_j}(s)-\max_{b\ne a_j}\ell_{k,j,b}(s).$$

The implemented attack differentiates a soft minimum over cells at one future horizon. Anchors are current predicted tokens; this avoids true-label-guided attacks but does not imply correctness. Competitor maxima are piecewise differentiable; formal smooth bounds require handling competitor ties or considering each fixed competitor separately.

## Dynamical-systems connection

Partial/output contraction concerns selected quantities or subspaces. Transverse contraction permits motion along an orbit while attracting transverse perturbations. These motivate selective stability, but their assumptions and chosen directions must be justified in a learned model.

Non-normal transient amplification is particularly relevant: stable eigenvalues alone do not control finite-time gain, and changing Jacobian products can amplify perturbations before eventual decay. This is already studied in neural physical simulators. Transferring it to discrete decision retention needs evidence and comparisons.

The analytic toy uses

$$J=\begin{pmatrix}0.8&5\\0&0.9\end{pmatrix},\quad s_0=(0,-0.1),\quad m_t=1+(s_t)_1.$$

Both eigenvalues are stable, yet the margin becomes negative and later positive. Run `python -m experiments.decision_stability.toy`. This demonstrates a possibility, not a measurement of TRM.

## A limited theoretical target

For a smooth fixed-competitor future margin $m_k$, perturbation subspace $S$, and Hessian norm bounded by $L_k$ along relevant segments, Taylor's theorem gives

$$m_k(s+\delta)\ge m_k(s)-\varepsilon\|P_S\nabla m_k(s)\|-\tfrac12L_k\varepsilon^2,\quad \delta\in S,\ \|\delta\|\le\varepsilon.$$

Finite-horizon answer preservation requires positive bounds for every cell, competitor, and time. Without verified curvature bounds this is a heuristic, not a certificate. It preserves the selected answer, not its correctness. Assuming positive future baseline margins conditions on future survival; it cannot be presented as prospective natural-error prediction. This standard bound is a framing tool, not a claimed theorem contribution.

## Contribution requirements

1. A trained-model mechanism separating spontaneous late failures from artificial perturbation effects.
2. Incremental predictive or intervention value beyond margin, margin trend, Q, answer stability, and depth.
3. An improved solve/retain/compute tradeoff against extra supervision, random depth, generic robustness, damping, Jacobian control, and practical stopping.
4. Replication across seeds and another task or model family, with failures and complete cost accounting.

The first code supplies instruments for these questions; it does not establish those outcomes. Publication potential remains conditional on the evidence.
