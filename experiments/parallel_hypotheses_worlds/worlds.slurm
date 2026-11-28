#!/usr/bin/env bash
#SBATCH --job-name=ph-two-worlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=02:00:00
set -euo pipefail
: "${PH_REPO:?}" "${PH_ARCHIVE:?}" "${PH_OUT:?}" "${PH_DATA:?}" "${PH_SIF:?}"
run="$PH_OUT/$SLURM_JOB_ID"
mkdir "$run"
scontrol show job "$SLURM_JOB_ID" > "$run/slurm.txt"
d=experiments/parallel_hypotheses_worlds
export APPTAINERENV_PYTHONPATH="$PH_REPO:$PH_REPO/scripts:$PH_REPO/$d"
unset DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR
options=(--nv --bind "$PH_REPO:$PH_REPO:ro" --bind "$PH_ARCHIVE:$PH_ARCHIVE:ro" --bind "$PH_OUT:$PH_OUT" --bind "$PH_DATA:$PH_DATA:ro" --pwd "$PH_REPO")
args=(--weights "$PH_ARCHIVE/checkpoint/checkpoint.pt" --config "$PH_ARCHIVE/checkpoint/all_config.yaml" --data "$PH_DATA/worlds-private.json" --manifest "$d/manifest.json")
if [[ ${1:-native} == native ]]; then
    srun apptainer exec "${options[@]}" "$PH_SIF" python "$d/native.py" "${args[@]}" --out "$run/result.json"
else
    srun apptainer exec "${options[@]}" "$PH_SIF" python "$d/train_route.py" "${args[@]}" --training "$PH_DATA/training-worlds-private.json" --seed "$1" --run "$run"
fi
