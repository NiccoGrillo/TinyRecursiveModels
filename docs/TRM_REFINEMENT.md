# Frozen TRM Sudoku refinement

The standalone residual and direct-update models rarely produced a nearly solved board. This study asks whether a learned continuation can improve a competent native TRM candidate. The reference is the matched-training native TRM checkpoint that solved 78 of 128 historical held-out boards. Its SHA-256 is `533999676a1d44bdb84205461b55ecf1719f238cad2c2af699de78128e3dbf7f`.

## Development gates

1. `scripts/calibrate_trm_refinement.py` uses the first 256 prepared development boards. It tests existing macro depths 2, 4, 6, 8 and 12. At each fork the shared budget is the median measured time for four further native macro steps, including their readouts at batch size one. Eligible forks have at least 32 starts in each correctness cohort, at least eight native repairs and a positive net gain over stopping. Choose the eligible fork with most repairs, breaking ties by earlier depth. If no fork passes, stop before fitting refiners and examine the development traces. Save the full selected carry and 11 logits at every cell, and verify the reconstructed exact outcomes against the timing pass.
2. `scripts/run_trm_refinement_dev.py` builds a training bank from the 1,000 original training puzzles at that fork. The model sees genuine native errors and correct answers, and the teacher checkpoint remains frozen. Train direct, stateless and recurrent refiners with identical supervised cross-entropy across four updates, 512 optimizer steps, the same minibatches and the same initial values for shared parameters. Use the existing 12-dimensional fixed codebook for corrections. This first comparison keeps field-specific auxiliary losses out of all three arms, so improvement can be attributed to the tested update recipes without changing the supervision. Report parameter counts and measured training times.
3. On development, compare all trained arms with stopping and native continuation. Advance to the held-out comparison if a learned arm exceeds both baselines by at least two percentage points in exact solves and its retention is within one percentage point of native retention. The development gate is a screening rule for this recipe. A failed gate leaves the held-out inputs unopened and calls for inspection of training and inference cost.

## Common start and timing

Native continuation uses the full saved $(z_H,z_L)$ carry and the native macro phase. Both learned refiners see the same puzzle, per-cell $z_H$ and $z_L$, and full native logits, including the two reserved token classes. If $\ell_0$ is the native prediction vector and $q_t$ is the learned latent correction, the candidate readout is $\ell_0 + 2q_t C^\top/\tau$, where $C$ is the fixed codebook and $q_0=0$. A learned scalar gain starts at zero, so the initial learned update preserves the fork answer as well. No branch uses labels or Sudoku rules at inference.

Remeasure native continuation in the same allocation and rotate arm order across boards. Begin timing after the common TRM prefix. Charge refiner feature projection, every update and every readout. Use the last completed readout within the budget, or the fork prediction when no continuation fits. Keep the 16-macro reference cap and report learned continuation cap hits, numerical failures and setup time separately. Batch-one synchronized timing is the primary policy.

## Held-out decision

Use the remaining 896 prepared test inputs after excluding the 128 already examined. Verify their IDs against the cohort manifest and training/dev disjointness. Lock the fork, budget, checkpoints and decision rule before opening them. At the primary budget report exact solves, repairs out of wrong starts, losses out of correct starts, and example-paired helped and harmed counts against native continuation, stopping and the direct refiner. Report paired uncertainty over original input IDs. Any secondary latency budgets are descriptive.

- A gain is actionable if it is at least two percentage points in exact solves, the lower end of its paired 95% interval exceeds zero, and the lower end of the paired retention interval is at least minus one percentage point.
- If a direct refiner clears the native and stop thresholds and mixing's upper bound over direct is below two points, pursue direct learned refinement. If the interval still permits a meaningful mixer advantage, report that comparison as inconclusive.
- If neither learned recipe offers a meaningful gain over native continuation and the intervals exclude it, deprioritize this repair recipe. Wide intervals call for an uncertainty statement.
- An improvement from recurrent mixing over native continuation and direct refinement supports the update recipe on this fork. A memory claim also requires a gain over stateless mixing with comparable retention.
