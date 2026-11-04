# Status

## Initial implementation — 2026-11-04

Official TRM history and MIT licence are preserved at base `c01103738605ba39d1430519b1ee0c62f4c707f8`. Upstream models, layers, training, and dataset builders are unchanged.

Implemented: fixed-phase rollouts; full-gradient adapter; strict checkpoint loading; unaugmented Sudoku reader and deterministic dev/test split; antithetic random and optional future-gradient `z_L` interventions; immediate-logit invariance assertions; post-cast budgets; paired harm/help, retention/recovery and survival metrics; JSONL and hashed manifests; independent summary recomputation; analytic transient-growth toy.

Verified with Python 3.12.14 and torch 2.7.0+cpu:

- `python -m pytest -q`: **20 passed**.
- Exact native/full-gradient forward agreement for MLP and attention at H=3, L=6, two layers, across multiple macro-steps.
- Multi-macro finite-difference agreement, immediate-readout invariance, zero-epsilon trajectory identity, and no reset after normal halt depth.
- Strict checkpoint wrappers and missing-key rejection; handcrafted metric denominators; masking; verifier checks; deterministic disjoint data selection; BF16 zero-delta detection.
- End-to-end fixture checkpoint/data CLI, cost counts, summary recomputation, and overwrite rejection.
- Standalone documented smoke command completed: 8 macro-steps, forks at 2/4, four continuation steps, all five arms, and two-step gradient probes.

The fixture checkpoint has random weights. These are implementation checks, not benchmark evidence.

## Not established

No trained checkpoint evaluation, GPU benchmark, or full training run has been executed. No empirical gain, new theorem, solution manifold, or publication novelty is claimed. Confidence intervals, prospective natural-error prediction, training interventions, and additional backends remain future work.

## Next step

Obtain a trusted original-TRM Sudoku checkpoint and resolved `all_config.yaml`, audit its data/selection provenance, then run the 128-puzzle dev pilot in `EXPERIMENTS.md`. Establish normal-horizon reproduction first.
