#!/usr/bin/env bash
set -euo pipefail
stage=${1:?Stage required}
runner="$RR_REPO/scripts/isambard/container.sh"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export RR_USE_GPU=1
if [[ $stage == report ]]; then
    export RR_USE_GPU=0
    exec "$runner" python -m experiments.residual_reasoning.report --root "$RR_ROOT" --require-complete
fi
if [[ $stage == smoke ]]; then
    srun --nodes=1 --ntasks=1 "$runner" python -m experiments.residual_reasoning.smoke \
        --root "$RR_ROOT" --device cuda --gpus "$RR_GPUS"
    exit
fi
cmd=(python)
if [[ $stage == reference || $stage == shared || $stage == joint ]]; then
    if (( RR_GPUS > 1 )); then
        cmd=(torchrun --standalone --nnodes=1 "--nproc-per-node=$RR_GPUS")
    fi
fi
srun --nodes=1 --ntasks=1 "$runner" "${cmd[@]}" -m experiments.residual_reasoning.study run \
    --config "$RR_CONFIG" --root "$RR_ROOT" --data "$RR_DATA" --stage "$stage" \
    --index "$SLURM_ARRAY_TASK_ID" --device cuda
