# First public-checkpoint CPU pilot — 2026-11-04

## Protocol and provenance

This is an exploratory dev pilot, not a held-out evaluation or training run. The laptop CPU budget reduces the roadmap's 128-example GPU pilot to 32 baseline puzzles and 8 intervention puzzles. No paid compute is used. Settings were fixed after two-puzzle timing checks and before inspecting the 32-puzzle baseline outcomes.

- Checkpoint: [gaoxin492/TinyRecursiveModels-Sudoku-Extreme-mlp](https://huggingface.co/gaoxin492/TinyRecursiveModels-Sudoku-Extreme-mlp/tree/256f32fcbe7123e8bf8c449410773a5ad311dbc5), revision `256f32fcbe7123e8bf8c449410773a5ad311dbc5`, `step_16275` and `all_config.yaml`.
- This is a third-party reproduction, not an official author checkpoint. The publisher reports 84.80% test accuracy. Config: seed 0, EMA enabled, global batch 4608, BF16, H=3/L=6, two MLP layers, 16-step normal horizon. Whether this weight file is EMA and the precise training/source/selection provenance are not independently verified.
- Published `trm.py` is byte-identical to this fork's pinned original `models/recursive_reasoning/trm.py`. Weight loading is strict with `weights_only=True`; downloaded Python is not executed. Compatibility does not audit unpublished training dependencies or contamination.
- Data: [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme/tree/58942f96baeb572ca3127e2a9e9c70f330783d6b), revision `58942f96baeb572ca3127e2a9e9c70f330783d6b`, full `test.csv` only (422,786 rows). No training download or augmentation.
- Prepare all evaluation rows before deduplication and split. Use project dev, split seed 0, noise seed 0, deterministic first 32 or first 8 IDs. Smaller cohorts are nested. No project held-out outcomes are inspected.
- The public test set was already used for reproduction evaluation. The project split does not make it a pristine external test. No symmetry-equivalence or training-overlap audit is claimed.

| Run | N | Precision | Natural steps | Forks | Branch horizon | Epsilon | Arms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pilot-native-32` | 32 | BF16 | 16 | — | — | — | Natural only |
| `pilot-fp32-32-depth64` | 32 | FP32 | 64 | — | — | — | Natural only |
| `pilot-zero-8` | 8 | FP32 | 32 | 4, 8, 16 | 16 | 0 | Clean, random ± |
| `pilot-targeted-8` | 8 | FP32 | 32 | 4, 8, 16 | 16 | 0.01 | Clean, random ±, gradient ± |

Use batch size 4, two CPU threads, probe horizon 2 and norm tolerance 5%. The branch horizon exceeds the gradient probe. The zero run must reproduce clean trajectories. Immediate readout equality is asserted for every intervention. Count all natural/branch/probe forwards and backwards. No hyperparameter sweep or learned repair is justified by this small pilot.

## Reproduce

Download `step_16275`, `all_config.yaml`, and `trm.py` from the pinned model revision into `checkpoints/gaoxin492-sudoku-256f32f/` (the local source copy is named `published_trm.py`). Download the pinned dataset's `test.csv` into `data/sudoku-extreme-public/`. These directories and all outputs are Git-ignored. Local `provenance.json` records source links, caveats, and file hashes; production manifests embed it and the CSV preparation provenance.

```bash
.venv/bin/python -m experiments.decision_stability.prepare_sudoku \
  data/sudoku-extreme-public/test.csv --out data/sudoku-extreme-public/prepared \
  --source-url https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/58942f96baeb572ca3127e2a9e9c70f330783d6b/test.csv

TRM_CHECKPOINT=checkpoints/gaoxin492-sudoku-256f32f/step_16275
TRM_CONFIG=checkpoints/gaoxin492-sudoku-256f32f/all_config.yaml
TRM_DATA=data/sudoku-extreme-public/prepared

.venv/bin/python -m experiments.decision_stability.run \
  --checkpoint "$TRM_CHECKPOINT" --config "$TRM_CONFIG" --data "$TRM_DATA" \
  --experiment rollout --limit 32 --steps 16 --batch-size 4 --threads 2 \
  --out runs/pilot-native-32
.venv/bin/python -m experiments.decision_stability.run \
  --checkpoint "$TRM_CHECKPOINT" --config "$TRM_CONFIG" --data "$TRM_DATA" \
  --dtype float32 --experiment rollout --limit 32 --steps 64 --batch-size 4 --threads 2 \
  --out runs/pilot-fp32-32-depth64
.venv/bin/python -m experiments.decision_stability.run \
  --checkpoint "$TRM_CHECKPOINT" --config "$TRM_CONFIG" --data "$TRM_DATA" \
  --dtype float32 --limit 8 --steps 32 --fork-steps 4 8 16 --horizon 16 \
  --epsilon 0 --batch-size 4 --threads 2 --out runs/pilot-zero-8
.venv/bin/python -m experiments.decision_stability.run \
  --checkpoint "$TRM_CHECKPOINT" --config "$TRM_CONFIG" --data "$TRM_DATA" \
  --dtype float32 --limit 8 --steps 32 --fork-steps 4 8 16 --horizon 16 \
  --epsilon 0.01 --targeted --probe-horizon 2 --batch-size 4 --threads 2 \
  --out runs/pilot-targeted-8

.venv/bin/python -m experiments.decision_stability.analyze runs/pilot-targeted-8
.venv/bin/python -m experiments.decision_stability.report runs/pilot-targeted-8
```

Output/preparation directories must be new. Reuse existing inputs but choose fresh run names for repetitions. Run the report command for each completed output directory. It recomputes the summary and rejects disagreement or incomplete runs.

## Reading results

Start with each run's `report.md`; `report.json` contains paired differences with 2,000-resample, seed-0 puzzle bootstrap intervals; `accuracy_by_step.csv` contains all natural depths. Reports distinguish any-time prediction divergence/harm/help from final outcomes, always against matching clean offsets. `summary.json` also includes first-loss survival at every branch offset. Raw JSONL and manifests retain predictions, budgets, hashes, IDs, and compute. The report JSON hashes its analysis source separately from the original run manifest.

Intervals are pointwise within a fork/arm/cohort, not simultaneous across stages. All arms use a common eligible cohort for the norm-matched table. No pooling across forks/seeds is performed, and no checkpoint variance is estimated. A degenerate interval on eight puzzles is not evidence of no population effect. Compare native and FP32 normal-horizon outcomes before interpreting FP32 continuation. This pilot does not measure native BF16 continuation beyond step 16.

Executed checks and measured outcomes are recorded in [STATUS.md](STATUS.md).
