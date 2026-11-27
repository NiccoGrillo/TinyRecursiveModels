# Fixed-bank development bridge

This deterministic comparator tests eight rigid grid symmetries with an inferred global color table. It uses Python's standard library. The 400 original ARC-AGI-1 public training tasks are the units of analysis. The task policy and denominator were recorded in PROTOCOL.md before scoring. We charge fitting every candidate and leave-one-demonstration-out refit. No cross-task training takes place.

From this directory, reproduce the scores with

```sh
git clone --filter=blob:none --sparse --no-checkout https://github.com/fchollet/ARC-AGI.git .cache/ARC-AGI
git -C .cache/ARC-AGI sparse-checkout set data/training
git -C .cache/ARC-AGI checkout 399030444e0ab0cc8b4e199870fb20b863846f34
python3 bridge.py .cache/ARC-AGI --out .cache/reproduction
```

From the TRM repository root, run `python3 -m unittest experiments.parallel_hypotheses_arc.test_bridge`. The scorer checks the data revision, a clean training directory and the task count. Compare the split digest with results/summary.json. results/manifest.json preserves a SHA-256 for each original JSON file. results/tasks.json records every selection, abstention, deterministic output digest and exact outcome. Only wall time and environment metadata should vary across identical data and scorer revisions.

## Development result

Twelve tasks admitted a demonstration-consistent program. Eleven solved every query on the first attempt (2.75% of 400, descriptive 95% Wilson interval 1.54–4.86%). Both ranking policies chose the same geometry on every task. Seven correct selected programs used geometry alone and four used color remapping alone. None of the correct selected programs used both. Across the 400 tasks, the mean within-task proportion of demonstrations correctly predicted after withholding each in turn was 2.5625%.

On aabf363d the only fitting geometry was identity. In one demonstration the 12 cells of color 2 became color 4 and a singleton 4 disappeared. In the other, 12 cells of color 3 became color 6 and the singleton 6 disappeared. In the query the 15 cells of color 8 became color 2 and the singleton 2 disappeared. A role-dependent recoloring explains this failure. The fixed table preserved the unseen 8 and mapped 2 to 4, missing 16 of 49 cells. This is a post hoc instance diagnosis. Broad ARC capability and a need for learned communication remain open.

A cellwise color substitution commutes with every coordinate permutation. High held-out combination accuracy on generated tasks from these two factors can therefore arise from a static factor bank. This hand-coded comparator is a shortcut diagnostic. Transfer of a learned model requires its own first-attempt outputs on a separately locked cohort and comparison to normal released solver policies.
