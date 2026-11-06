# Research resources

Primary-source map updated 2026-11-06. Recheck versions before citing results. No full papers are vendored. Read the close completion, allocation and scheduling work before designing further diagnostics or claiming novelty.

## Closest work

| Resource | Relevance and caveat |
| --- | --- |
| [When Recursive Models Finish Computing](https://arxiv.org/abs/2609.26487v1) | Directly overlapping: extended TRM Sudoku recurrence, directional stability, and perturbations, including our MLP checkpoint. Its 1,000-puzzle study reports strong retention after solving. Generic late recovery, anisotropic stability, or another small z_L perturbation result is insufficient novelty. Full text reviewed; this substantially changes our priority. |
| [Scheduling Recursive Reasoning in Looped Transformers — TAPS](https://arxiv.org/abs/2609.36653) | Adaptive update scaling from trajectory progress/fluctuation, without retraining. Required controller baseline. Full-text update definition and Section 5.3 / Appendix C.4 checked: inner/outer control and controller refresh differ, while nested recurrence is retained. Code still needs inspection before implementing a comparator. |
| [TRM: Less is More](https://arxiv.org/abs/2510.04871) · [official code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | Starting model and training recipe. TRM updates answer and reasoning states; it does not enforce global fixed-point convergence. |
| [Fixed-Point Reasoners](https://arxiv.org/abs/2606.18206) · [code](https://github.com/nilskiKonjIzDunava/fprm) | Residual-scaled architecture and fixed-point halting. Distinguish sufficient theoretical contraction assumptions from enforced constraints. A future adapter must include solver/controller state and distinguish raw proposal residual from damped accepted movement. |
| [InfiLoop](https://arxiv.org/abs/2610.11570) · [code](https://github.com/pixeli99/InfiLoop) | Long-depth recurrence with accumulated memory. Preserve numerator and denominator, not just their ratio. Separate single-schedule performance from verifier-assisted schedule search and charge all attempts. |
| [Looped Reasoning Finishes Earlier Than You Think — QRR](https://arxiv.org/abs/2610.18174) · [code](https://github.com/camail-official/Quotient_Recurrent_Reasoner_QRR) | Exact phase symmetry supports latent motion while quotient dynamics converge. A generic “moving latent, stable answer” idea is already covered. Our proposed diagnostic does not assume dynamic decoupling of present-readout-preserving directions. |
| [Learning to Compute Is Not Learning to Retain or Reuse](https://arxiv.org/abs/2610.19768) | Separates computation, retention, and reuse using hidden-state interventions. Random depth/position supervision is a critical baseline. Retention failure and causal state experiments are already known. |
| [Controlling Transient Amplification Improves Long-horizon Rollouts](https://arxiv.org/abs/2605.08856) | Non-normal, non-commuting trajectory Jacobians in neural simulators. Its normality/commutativity regularization is a close comparator for any transient-amplification account. |
| [Invariant-Measure Reasoners — ImR](https://arxiv.org/abs/2610.10996) | Predicts using an expectation under an invariant measure. Generic stable readouts of non-fixed-point latent dynamics are already covered. Primary abstract checked. |
| [Readout Feedback — RoFB](https://arxiv.org/html/2608.24136v2) | Uses intermediate readout probabilities to add coupling to recurrent latent dynamics. Generic closed-loop steering is already covered. Primary introduction and method overview checked; detailed comparator implementation remains to audit. |
| [Input-Conditioned Search — ICS](https://arxiv.org/abs/2610.18547v2) | Searches input conditions to change the update map; a close comparator for escaping unsuccessful regimes. Primary abstract retrieved directly from arXiv; detailed method and fixed-point claims still need full review. |

**FPRM Appendix F matters separately from its main architecture:** it compares outer/inner compute allocation at an approximately fixed budget. A generic claim that spending more compute internally can be worse is therefore not new. The narrower conditional mechanism and explicit additive-core example are in [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md).

InfiLoop implementation inspected during planning: [`63716a9c46723d3592c8ad4fc5819855f22aaa47`](https://github.com/pixeli99/InfiLoop/tree/63716a9c46723d3592c8ad4fc5819855f22aaa47), particularly Sudoku model, verifier, and schedule files. No FPRM or InfiLoop backend is implemented here yet.

## Further related work

- [STARS / recurrent stability](https://arxiv.org/abs/2605.26733): Jacobian stabilization; distinguish empirical regularization from certified guarantees.
- [LayerNorm and gain control in looped transformers](https://arxiv.org/abs/2607.10681): normalization and local dynamics can explain apparent stability effects.
- [Learned stochastic stopping](https://arxiv.org/abs/2606.29983): a competing practical way to control extrapolation errors.
- [Solve the Loop / attractor models](https://arxiv.org/abs/2605.12466): distinguish this attractor approach from FPRM.
- [Equilibrium Reasoners](https://arxiv.org/abs/2605.21488): correctness versus reachability and initialization.
- [Probabilistic TRM](https://arxiv.org/abs/2605.19943): stochastic/parallel trajectories; count every attempt.
- [Ananke / contractive torus networks](https://arxiv.org/abs/2609.24737): attracting sets with tangent motion already exist as an architectural idea.
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377): equilibrium computation and implicit gradients.
- [Path Independence in Iterative Algorithmic Reasoning](https://arxiv.org/abs/2211.09961): convergence and history dependence are different properties.

## Dynamical-systems foundations

- [Spenke et al.: subproblem iterations in partitioned fluid–structure interaction](https://arxiv.org/abs/2303.08513): computation against unconverged coupling data and the cost of inner solves. Numerical-analysis precedent, not direct evidence about learned reasoning.
- [Sheehan et al.: iteration error and stability of coupled PDEs](https://doi.org/10.1155/2015/787198): component stability, incomplete iteration, and multirate coupling. Do not claim our small coupled-map example as a new general numerical theorem.
- [Wang and Slotine: partial contraction](https://people.csail.mit.edu/paulina/jclub/partial-contraction.pdf): selected stable quantities/subspaces and synchronization.
- [Manchester and Slotine: transverse contraction](https://arxiv.org/abs/1209.4433): orbital stability while phase continues; its metric assumptions need proof in a new model.
- [Output contraction for discrete-time nonlinear systems and RNNs](https://pubmed.ncbi.nlm.nih.gov/40669402/): output stability can be weaker than full-state contraction; check exact assumptions before transferring results.
- [Trefethen et al.: Hydrodynamic Stability Without Eigenvalues](https://people.maths.ox.ac.uk/trefethen/ttrd.pdf): why stable eigenvalues can coexist with finite-time gain.

## Code landmarks

| Path | What to inspect |
| --- | --- |
| `models/recursive_reasoning/trm.py` | Full carry, recurrence, gradient cuts, readout, ACT reset |
| `models/layers.py` | MLP/attention, normalization, casting, SDPA |
| `models/sparse_embedding.py` | Puzzle lookup and nonpersistent training buffers |
| `pretrain.py` | Wrappers, checkpoint state dict, resolved config, EMA |
| `dataset/build_sudoku_dataset.py` | Tokens, augmentation, original-puzzle groups |
| `experiments/decision_stability/model.py` | Fixed-phase adapter and full-gradient path |
| `experiments/decision_stability/clock.py` | Dense observation, same-state schedule fork, equal-budget outcomes and exact paired bounds |

Before comparing headline scores, record exact evaluation version, schedule search, verifier use, precision, and training budget. Results with different procedures are not directly interchangeable.
