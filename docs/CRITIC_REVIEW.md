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
