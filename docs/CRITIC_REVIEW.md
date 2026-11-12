# Critical review

An independent critic agent reviewed the research framing and first-experiment design. This is internal adversarial review, not external peer review or a publication endorsement.

| Objection | Response |
| --- | --- |
| Broad manifold proposal overlaps existing work | Narrow to current-output-preserving directions and future consequences; cite QRR and other close work |
| Eigenvalues miss transient gain | Use finite-horizon margins and paired effects; label the toy as illustration |
| TRM cuts gradients | Full differentiable recurrence with forward and derivative tests |
| ACT resets invalidate extended rollouts | Continue the inner carry at a fixed phase |
| BF16 breaks equal nominal budgets | Measure actual norms and zero deltas; report full and common tolerance-qualified cohorts |
| Attack success is not native failure causation | Keep offline attack and prospective prediction separate; compare paired clean continuations |
| Freezing can harm solving | Report recovery, helped examples, and final solve rate |
| Oracle selection inflates outcomes | Preset times, offline labels, and separate rule verification |

At the time of this initial review there was no trained checkpoint evidence; subsequent exploratory runs are recorded in [STATUS.md](STATUS.md). Remaining risks: isotropic hidden perturbations may be off-distribution; the readout-preserving subspace is architecturally known, not learned. Margin, stage, precision, or generic recurrent sensitivity may explain observed effects. Simple supervision or stabilization may match any eventual repair.

The critic judged the diagnostic design worth executing with these constraints. It did not certify a new method or an A* publication outcome.

## Review after the checkpoint pilot — 2026-11-05

The critic was reactivated after reading the updated results and identifying close completion/stability prior work. Initial objections narrowed a broad protocol comparison into the [same-state clock study](NEXT_EXPERIMENT.md). It then reviewed the implementation, metrics and uncertainty independently of the implementation author.

| Objection | Incorporated resolution |
| --- | --- |
| More MLP retention/Jacobian/perturbation runs risk duplicating close work | One fixed causal continuation-schedule comparison; no new sweep or stabilizer |
| Logging more often and changing L are different operations | Dense native observation is separated from the L=12 intervention |
| Comparing 36 updates per arm confounds cost | 2,184 blocks per arm; 156 versus 84 outer updates; shared 182-block grid |
| Whole trajectories can differ before a schedule comparison | Both arms clone the same full native state at a fixed macro boundary |
| An arm-dependent solved cohort creates selection bias | Correct/wrong risk sets are fixed before branching |
| Transient readout errors may not affect normal outputs | Separate dense loss, native macro loss, and first-dense-solve audits |
| Some common-budget grid points are intermediate phases | Endpoint phases are recorded; default final endpoints are native phase zero |
| Zero-event bootstrap intervals create false certainty | Exact paired binary bounds; bootstrap retained only for continuous block-time differences |
| Precision can alter the starting state and solved cohort | Each numerical regime is separate; no causal paired-precision claim |
| A 2 pp point estimate does not establish a 2 pp effect | Interval-based support, exclusion and inconclusive rules |

Code review also caught an undefined variable in summarization, fixed before publication, and prompted FP32 diagnostic subtraction of stored latent tensors. Native recurrence parity and end-to-end report verification are tested.

Final internal verdict: **ready for a consequential, falsifiable protocol experiment; publication value remains conditional on the results.** The reviewer found no remaining blocker in corrected pairing, budgets or risk cohorts. It explicitly rejected interpreting L=12-induced failures as native L=6 instability. This is internal agent review, not external peer review or a guarantee of acceptance.

## Continued research review — 2026-11-06

At the user's request, continued beyond implementation. The strongest conditional direction was coupled feedback and timing, documented in [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md). The critic verified the fixed-set inclusion, identified a stable-inner/unstable-coupled example, and then caught that a generic two-input shared core did not respect TRM's actual addition before the block stack. The final example uses one scalar additive core and satisfies that wiring constraint; it still does not construct actual post-normalized Transformer weights.

The reviewer also identified direct allocation prior in FPRM Appendix F; required acknowledgement of classical coupled-solver analysis; distinguished a reversed periodic schedule's transient effects from its unchanged eigenvalues; and demanded a positive equal-cost prediction or predicted rescue beyond merely damaging an off-training schedule. The note includes those corrections and mathematical predictions. Common fixed states also expose a limit on schedule-only rescue.

The surviving proposal is a mechanism hypothesis requiring trained-model evidence. If changes only cause damage, or native/TAPS/simple damping matches any benefit, the result remains an explanatory note rather than a new reasoning method. Critic agreement does not establish an A* contribution.

## Architecture/document review — 2026-11-12

The persistent-workspace proposal is coherent and worth implementing directly, but its novelty is not a new principle of optimization. Strong stateful optimizer and learned-preconditioner controls are essential. Exact protection applies to every stationary semantic state, including wrong roots. Descent needs smoothness along accepted segments and bounded positive-definite metrics; PL gives energy-gap convergence, not automatically convergence to a chosen point or a correct argmax. Near-root workspace sensitivity is a one-step bound. A bounded workspace requires a bounded full update, not an unbounded residual addition.

For attracting answer sets, energy descent alone does not align the readout. Neighborhood-wide orthogonality and nonnegative potential do imply zero-set invariance for a well-posed continuous flow; orthogonality checked only at critical points is vacuous. LaSalle identifies the largest invariant zero-dissipation subset under its regional assumptions. Residual projection needs rank/conditioning analysis, and tangent Euler steps do not preserve curved manifolds exactly. GNN energy-plus-tangent precedents narrow the novelty claim but do not invalidate a distinct reasoning architecture. These qualifications are incorporated in the notebook. Internal agent review is not external peer review or publication acceptance.

## Four-round architecture session and supplied QRR reviews — 2026-11-12

1. **Reject weak novelty framing:** state-side SPD memory is learned optimization and freezes every aggregate-gradient zero, including wrong minima.
2. **Move control before aggregation:** residual-side weights can change cancellation; diagonal weighting is covered by longstanding constraint-penalty methods and can itself be blind.
3. **Specify off-diagonal mixing:** the exact residual pair c(q)=(q²−1, 2q(q−1)) has a strict wrong unweighted local minimum at zero. All diagonal weights freeze it; a bounded off-diagonal SPD residual metric can move it. This is availability of movement, not guaranteed basin escape. Common local attraction near regular residual zeros survives; global descent of the original energy is deliberately not claimed.
4. **Challenge the proposed mechanism:** fresh random mixing already moves almost surely under the stated nondegeneracy/distribution assumptions. Fixed full mixing can be absorbed into a residual basis. Sketched Newton–Raphson Section 3 already uses a residual metric frozen for its inner derivative and zero-residual interpolation. Thus new optimization-principle claims are rejected. Learned transferable factor coordination remains an empirical hypothesis.

The supplied review text (reviewer IDs 6QHq, 9Sdw, tDyW) rightly emphasizes architecture/stopping-rule confounding, checkpoint settings and coverage. Some allegations are unsupported: unique discrete answers do not imply isolated latent equilibria, multiple discrete answers do not imply a continuous manifold, repeated answer supervision does not require labeled trajectories, and gauge symmetry does not itself forbid outcome RL. No independent OpenReview-thread verification was claimed.

Final grounding revision: use explicit Sudoku Boolean and sum/clue residuals. Their common zeros are exactly valid one-hot completions, and the Boolean Jacobian provides full column rank at each zero. This closes the learned-residual-collapse gap for the first architecture, while narrowing scope to a grounded neuro-symbolic solver. It does not establish global convergence, a learned semantic manifold, or language-model transfer. Optional 3SAT is satisfiable-instance witness finding, not UNSAT certification.

**Critic verdict:** approve a bounded substantial architecture comparison; do not approve publication-ready novelty. Reject memory-coordination claims if random/stateless mixing or a learned static residual basis removes the advantage. Do not continue agent cycles merely to obtain approval. The notebook records the derivations and conditions in full.
