# Research direction after the pilot: coupled feedback, not just more loops

## Current architecture direction — 12 November 2026

The main research direction is now a **persistent-workspace energy reasoner**. Its semantic update is `q_next = q - alpha * M(q,w,x) * grad_q E(q;x)`: the workspace changes a bounded positive-definite preconditioner, while energy and readout depend on semantic state and input only. Workspace therefore affects search away from stationarity and cannot move an exact stationary semantic state. This is a proposed architecture, not an implemented or trained result. Wrong stationary points are protected too.

The [living LaTeX guide](notebook/project_notebook.tex) and [PDF](notebook/project_notebook.pdf), version 2.0, are the authoritative integrated specification. They include the mathematical primer, exact assumptions/proofs, training design, cost-aware comparison with energy reasoners and stateful optimizers, QRR critique, and an expanded attracting-answer-set alternative. The older E4/E5 diagnostic gates below are historical and optional; they do not block architectural work. No new trained result was produced by this update.

## Historical specification (preserved)

Research and critic round, 2026-11-06. This note adds mathematical deductions and a conditional project direction. It does not change the frozen [next experiment](NEXT_EXPERIMENT.md), report new trained-model results, or establish publication novelty.

## The idea in plain language

TRM alternates between updating its reasoning state while the answer state is held still, and updating the answer state using the reasoning state. More reasoning updates therefore mean more computation against an answer state that has not yet been refreshed. Improving one of these two computations in isolation need not improve their interaction.

The strongest surviving hypothesis is: **some limits of continued reasoning arise from the timing and strength of feedback between the two states, even when the available fixed points do not disappear.** This shifts the question from choosing a point versus a manifold to understanding which coupled update rule reaches and stays near useful states.

The critic rejected generic versions of this proposal. FPRM already studies compute allocation; TAPS already adapts update scale. A paper would need a mechanism that predicts an effect and a useful intervention before observing its outcomes, beyond an off-training-schedule failure. The results below are reasons to investigate that mechanism, not proof it operates in trained TRM.

## 1. Changing a schedule can preserve a root and change its stability

Let A update only the low state, B update only the high state, and hold the input fixed. Define one outer update as

$$F_L=B\circ A^L.$$

**Elementary deduction:** for every positive integer m,

$$\operatorname{Fix}(F_L)\subseteq\operatorname{Fix}(F_{mL}).$$

Proof: if `F_L(h,l)=(h,l)`, the low state after `A^L` must already equal l, because B does not change it. A also leaves h unchanged. Thus `A^L(h,l)=(h,l)`, and B fixes that same state. Repeating `A^L` m times followed by B still returns the state.

In fact, `Fix(F_L)=Fix(A^L)∩Fix(B)`. If the fixed-h low dynamics have no nontrivial periodic points, these fixed sets are the same for every positive L. Contraction would be one sufficient condition for that extra premise; it is not assumed for trained TRM.

Consequently L=6→12 cannot delete an exact outer fixed point. This says nothing about its attraction, its basin, additional fixed points, or correctness. A fixed point of the native macro `F_6^3` need not be a fixed point of `F_6`: a three-cycle is possible. Finite-window answer stability is weaker still.

This gives a useful limit on control. An exact wrong `F_6` fixed state cannot be moved by switching to L=12 alone. More generally, if both `A(s)=s` and `B(s)=s`, every deterministic composition of these unchanged maps, including residual damping, leaves s fixed. Adaptive timing cannot escape such a state without a different map or an actual displacement. The statement also holds for exact fixed states of deterministic finite-precision implementations with identical operators; roundoff is not a magic escape mechanism.

This does not make scheduling useless: finite-budget wrong states can be unfinished computation, and near-equilibrium states can move. It prevents calling every wrong plateau a basin escape or promising that timing alone repairs every incorrect attractor.

## 2. A shared-core example where more inner convergence destabilizes the whole system

The actual TRM module adds its state and injection before its shared block stack. We therefore use a single scalar map with **additive injection**, rather than two independently parameterized update rules:

$$G(v)=v(v-1)(0.7v+0.2),\qquad x=1,$$

$$l\leftarrow G(l+h+x)\quad\text{L times},\qquad h\leftarrow G(h+l).$$

The origin is a common equilibrium because `G(0)=G(1)=0`. The conditional derivatives are `G'(1)=0.9` for the low update and `G'(0)=-0.2` for the high update. Each subsystem is locally contractive when the other state is held fixed.

Put `q=0.9^L` and `u=9(1-q)`. The exact derivative of the coupled outer map at the origin, in `(h,l)` order, is

$$J_L=\begin{pmatrix}-0.2(1+u)&-0.2q\\u&q\end{pmatrix}.$$

Its trace is `-2+2.8q` and determinant `-0.2q`. The three second-order discrete stability conditions reduce to

$$q>5/13.$$

Thus L=1 through 9 are locally stable and L≥10 are unstable. In particular:

| Update | Spectral radius at the same equilibrium | Interpretation |
| --- | --- | --- |
| L=6 | 0.670489 | Locally attracting |
| L=12 | 1.254233 | One unstable direction |
| L=12, high update damped by 0.8 | 0.694608 | Locally attracting again |

The stability inequalities have strict margins, so the L6/L12 distinction survives sufficiently small parameter changes. It is not a single exact cancellation. Nevertheless, an exactly initialized equilibrium remains there even when unstable; attraction describes nearby states.

For the damped high update `h_new=h+η(G(h+l)-h)`, with `0<η≤1`, the L12 stability threshold is

$$\eta<\frac{2(1+q)}{3-0.6q}\approx0.906137.$$

This supplies a predicted rescue, not just a way to break a model. Damping is a standard comparator, not our proposed novelty. The example matches TRM's shared additive update wiring but is **not** a construction of actual post-normalized Transformer weights, a theorem about trained checkpoints, or proof of ODE stiffness. Latent instability also need not change a discrete readout.

Reproduce the calculation with:

```bash
.venv/bin/python -m experiments.decision_stability.coupled_toy
```

Tests compare the derived derivative with finite differences through the actual nonlinear recurrence, rather than only checking the closed-form matrix against itself.

## 3. The same amount of each computation can be used more effectively

The example also has a positive prediction at exactly matched operation counts. Consider two successive high-state updates and allocate twelve low updates between them:

| Low-update counts before successive high updates | Low / high calls | TRM block-equivalent count | Spectral radius of the period map | Largest singular value of the period map |
| --- | --- | --- | --- | --- |
| 6, 6 | 12 / 2 | 28 | 0.449555 | 2.258716 |
| 3, 9 | 12 / 2 | 28 | 0.106288 | 2.982494 |
| 9, 3 | 12 / 2 | 28 | 0.106288 | 0.852055 |

Unequal allocation improves the worst-case local linearized convergence rate per equal-cost period in this example. This is not monotonic contraction or an improvement for every initial error. The 3,9 order permits larger finite-time amplification; faster eventual convergence need not improve a short-budget answer.

The block-equivalent count prices each scalar core call as two TRM layers to align the schedules; it is not a measured FLOP count for the scalar toy.

The reversed schedules are phase shifts of the same periodic policy. Their linearized period matrices are `J_9 J_3` and `J_3 J_9`, which have the same eigenvalues. Different outcomes from one shared initial state can reflect phase, transient amplification or nonlinear basin access; they cannot establish different local asymptotic stability merely from that reversal. The critic specifically required this distinction.

### Conditional follow-up on real models

Only after the current study gives a reason, compare repeated `(6,6)`, `(3,9)` and `(9,3)` from identical native states. Seventy-eight pairs per arm give 936 low updates, 156 high updates and 2,184 block evaluations each. This controls the number of answer updates, which differs between the current constant-L6 and constant-L12 arms. The total study cost grows with the added arm; it is not free computation.

Use final answers at matching phase and cost as the main practical outcome, alongside recovery, loss and dense trajectories. Reversal is a phase control. Prespecify the contrast, cohort and effect-size threshold before launching; this is a proposed follow-up, not an extra arm silently added to the frozen study. Do not select only visually striking late-recovery cases.

Even a difference here is not enough: all altered schedules differ from training. The stronger requirement is a state-dependent prediction of beneficial timing, or of a component-specific damping rescue, that generalizes beyond the states used to devise it. If only native scheduling works best, there is no demonstrated new inference method.

## What would connect the toy to a trained model?

At a common equilibrium of the individual additive updates, let P be the derivative of the shared module at the low-state evaluation and R its derivative at the high-state evaluation. They need not be equal: the module is shared, but evaluated at different injected states. Set

$$U_L=\sum_{j=1}^{L}P^j.$$

Then the full coupled derivative is

$$J_L=\begin{pmatrix}R(I+U_L)&RP^L\\U_L&P^L\end{pmatrix}.$$

If the inner iteration converges locally, `P^L→0`, the effective high-state feedback approaches `R(I-P)^{-1}`. Stability of P and R separately does not ensure stability of that coupled feedback. This is a standard numerical-analysis mechanism transferred to a precise architectural question, not a new general theorem.

Away from a common equilibrium, the derivatives vary along the inner trajectory and require ordered products. A single Jacobian eigenvalue is not a valid explanation of a finite rollout. Any practical account must predict finite-horizon effects on actual answers beyond residual, margin, Q, depth and numerical precision. It must also explain improvement or a prespecified rescue, not just harm from an unfamiliar schedule.

A separate finite-state bound clarifies what small residuals do and do not imply. With h held fixed, write `l6=A_h^6(l)` and `l12=A_h^6(l6)`. If `A_h^6` is K-Lipschitz on the relevant segment, then

$$\|l12-l6\|\leq K\|l6-l\|.$$

If the high update has low-input Lipschitz constant M on the corresponding segment, its L12-versus-L6 change is at most `MK||l6-l||`. This is a one-step statement with potentially large, unmeasured constants. It neither certifies long-horizon retention nor proves small residual means correct computation.

## Prior work and ideas we should not repackage

| Direction | Decision after checking close work |
| --- | --- |
| Stable answers on moving or non-point latent sets | QRR and [ImR](https://arxiv.org/abs/2610.10996) already address versions of this. Reject as a standalone novelty claim. |
| More outer updates versus more inner work | [FPRM Appendix F](https://arxiv.org/html/2606.18206v1) already compares allocation at an approximately fixed budget. Generic allocation is occupied. |
| Adapt step sizes from observed dynamics | [TAPS](https://arxiv.org/html/2609.36653v1) covers this, including inner/outer control and refresh frequency while retaining nested recurrence. Required baseline. |
| Steer incorrect states using readouts or changed input conditions | [RoFB](https://arxiv.org/html/2608.24136v2) and [ICS](https://arxiv.org/abs/2610.18547v2) are close prior work. ICS's primary abstract was checked; its detailed method/theory still needs full review before a competing method claim. |
| Inner solves and stability of coupled systems | [Spenke et al.](https://arxiv.org/abs/2303.08513) and [Sheehan et al.](https://doi.org/10.1155/2015/787198) already study related numerical issues. The math is grounding, not proof of ML novelty. |

The research opportunity, if real, is a **predictive causal account of coupled feedback in trained recurrent reasoners, with a useful equal-cost intervention**. The fixed-set distinction and shared-core counterexample sharpen that question. A toy plus a schedule sweep would not meet the publication ambition.

## Keep / stop criteria

Keep this line only if the prescribed clock study and a justified follow-up support a reproducible, useful effect that survives numerical/phase controls and simple scheduling/damping baselines. A component-specific prediction made before seeing the continuation is stronger than explaining a curve afterward. Validate any selected policy on new puzzles and checkpoints, and count all diagnostic computation.

Stop the method claim if the effects are only harmful distribution shift, isolated to one numerical regime without a corresponding explanation, or matched by native continuation and existing controllers. If wrong states are exact common equilibria, a pure-scheduling rescue claim is mathematically unavailable; a map-changing approach would need comparison with the existing alternatives. If no useful effect survives, retain these deductions as an explanatory note and redirect the project.
