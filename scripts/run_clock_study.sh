#!/usr/bin/env bash
# Fixed, four-condition development study. See docs/NEXT_EXPERIMENT.md.
set -euo pipefail
cd "$(dirname "$0")/.."
export TRM_PYTHON="${TRM_PYTHON:-.venv/bin/python}"
export TRM_DEVICE="${TRM_DEVICE:-cuda}"
export TRM_BATCH_SIZE="${TRM_BATCH_SIZE:-8}"
export TRM_THREADS="${TRM_THREADS:-2}"
export TRM_OUT="${TRM_OUT:-runs/clock-study-v1}"
export TRM_DATA="${TRM_DATA:-data/sudoku-extreme-public/prepared}"
export TRM_MLP_CONFIG="${TRM_MLP_CONFIG:-checkpoints/gaoxin492-sudoku-256f32f/all_config.yaml}"
export TRM_MLP_CHECKPOINT="${TRM_MLP_CHECKPOINT:-checkpoints/gaoxin492-sudoku-256f32f/step_16275}"
export TRM_ATT_CONFIG="${TRM_ATT_CONFIG:-checkpoints/gaoxin492-sudoku-att-33991b8/all_config.yaml}"
export TRM_ATT_CHECKPOINT="${TRM_ATT_CHECKPOINT:-checkpoints/gaoxin492-sudoku-att-33991b8/step_21700}"

"$TRM_PYTHON" - <<'PY'
import json, os
from pathlib import Path
import torch
from experiments.decision_stability.model import load_model, read_config, sha256

out = Path(os.environ['TRM_OUT'])
if out.exists():
    raise FileExistsError(f'Choose a new TRM_OUT: {out}')
device = os.environ['TRM_DEVICE']
if device.startswith('cuda'):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install CUDA PyTorch/use a GPU host, or explicitly set TRM_DEVICE=cpu.')
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('This GPU does not support the prespecified BF16 condition.')
metadata = json.loads((Path(os.environ['TRM_DATA']) / 'test/dataset.json').read_text())
expected = {
    'MLP': ('811e64a1fa2e923e601791339b736bb3bab8bd350778ba116908d7965c021777',
            '20e9dc7ebf83b9b41a8b3f58f5fd94ee3a7eb0b0d245bdeeb14e2f1488d1daaf'),
    'ATT': ('9e595c896d6484e9dbb3c3963d11bac224dd91d70eb040f11de04ac8303be6b3',
            '932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3'),
}
for kind in ('MLP', 'ATT'):
    for field, checksum in zip(('CONFIG', 'CHECKPOINT'), expected[kind]):
        if sha256(os.environ[f'TRM_{kind}_{field}']) != checksum:
            raise ValueError(f'{kind} {field}: hash differs from the frozen protocol; overrides only relocate the same assets')
    config = read_config(os.environ[f'TRM_{kind}_CONFIG'], metadata, int(os.environ['TRM_BATCH_SIZE']))
    if (config['H_cycles'], config['L_cycles'], config['L_layers'], config['halt_max_steps']) != (3, 6, 2, 16):
        raise ValueError(f'{kind}: configuration does not match this fixed protocol')
    if config['mlp_t'] != (kind == 'MLP'):
        raise ValueError(f'{kind}: wrong architecture')
    load_model(config, os.environ[f'TRM_{kind}_CHECKPOINT'], 'cpu')
print('Preflight passed. Fixed cohort: 1024; four conditions; 20,643,840 sample-block evaluations.', flush=True)
PY

for kind in ATT MLP; do
  config_variable="TRM_${kind}_CONFIG"
  checkpoint_variable="TRM_${kind}_CHECKPOINT"
  for dtype in bfloat16 float32; do
    run_out="$TRM_OUT/${kind,,}-$dtype"
    "$TRM_PYTHON" -m experiments.decision_stability.clock \
      --config "${!config_variable}" --checkpoint "${!checkpoint_variable}" --data "$TRM_DATA" \
      --device "$TRM_DEVICE" --dtype "$dtype" --attention-backend math \
      --limit 1024 --offset 32 --partition dev --split-seed 0 --seed 0 \
      --batch-size "$TRM_BATCH_SIZE" --threads "$TRM_THREADS" \
      --low-cycles 6 12 --fork-outer 48 --branch-blocks 2184 --followup-outer 36 \
      --out "$run_out"
    "$TRM_PYTHON" -m experiments.decision_stability.clock_report "$run_out"
  done
done

"$TRM_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['TRM_OUT'])
runs = [json.loads((root / f'{k}-{p}' / 'manifest.json').read_text())
        for k in ('att', 'mlp') for p in ('bfloat16', 'float32')]
assert all(r['status'] == 'complete' for r in runs)
ids = runs[0]['dataset']['selected_ids']
assert len(ids) == 1024 and all(r['dataset']['selected_ids'] == ids for r in runs)
print(f'All four runs complete with identical puzzle IDs. Reports: {root}', flush=True)
PY
