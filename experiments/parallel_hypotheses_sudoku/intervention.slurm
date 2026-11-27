#!/usr/bin/env bash
#SBATCH --job-name=ph-peer-write
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=00:15:00
set -euo pipefail
: "${PH_REPO:?}" "${PH_OUT:?}" "${PH_ARCHIVE:?}" "${PH_DATA:?}" "${PH_COHORT:?}" "${PH_SIF:?}"
export APPTAINERENV_PYTHONPATH="$PH_REPO/scripts:$PH_REPO"
unset DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR
run="$PH_OUT/$SLURM_JOB_ID"
mkdir "$run"
scontrol show job "$SLURM_JOB_ID" > "$run/slurm.txt"
srun apptainer exec --nv --bind "$PH_REPO:$PH_REPO:ro" --bind "$PH_ARCHIVE:$PH_ARCHIVE:ro" \
    --bind "$PH_DATA:$PH_DATA:ro" --bind "$PH_COHORT:$PH_COHORT:ro" --bind "$PH_OUT:$PH_OUT" \
    --pwd "$PH_REPO" "$PH_SIF" python experiments/parallel_hypotheses_sudoku/intervention.py \
    --device cuda --data "$PH_DATA" --cohort "$PH_COHORT" \
    --weights "$PH_ARCHIVE/checkpoint/checkpoint.pt" --config "$PH_ARCHIVE/checkpoint/all_config.yaml" \
    --calibration "$PH_ARCHIVE/dev-calibration/calibration.json" --edit "$PH_ARCHIVE/native-hypotheses/workspace-edit.pt" \
    --reference experiments/parallel_hypotheses_sudoku/gh200-results.json --out "$run/results.json"
