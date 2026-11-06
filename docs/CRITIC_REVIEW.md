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
