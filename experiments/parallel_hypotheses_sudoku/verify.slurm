#!/usr/bin/env bash
#SBATCH --job-name=ph-sudoku-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=00:15:00
set -euo pipefail
: "${PH_ARCHIVE:?}" "${PH_REPO:?}" "${PH_OUT:?}" "${PH_SIF:?}"
export APPTAINERENV_PYTHONPATH="$PH_REPO/scripts:$PH_REPO"
unset DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR
run="$PH_OUT/$SLURM_JOB_ID"
mkdir "$run"
scontrol show job "$SLURM_JOB_ID" > "$run/slurm.txt"
srun apptainer exec --nv --bind "$PH_REPO:$PH_REPO:ro" --bind "$PH_ARCHIVE:$PH_ARCHIVE:ro" \
    --bind "$PH_OUT:$PH_OUT" --pwd "$PH_REPO" "$PH_SIF" \
    python experiments/parallel_hypotheses_sudoku/replay.py --device cuda \
    --report "$PH_ARCHIVE/native-hypotheses/result.json" \
    --calibration "$PH_ARCHIVE/dev-calibration/calibration.json" \
    --bank "$PH_ARCHIVE/dev-calibration/fork_states.pt" \
    --weights "$PH_ARCHIVE/checkpoint/checkpoint.pt" --config "$PH_ARCHIVE/checkpoint/all_config.yaml" \
    --edit "$PH_ARCHIVE/native-hypotheses/workspace-edit.pt" \
    --out "$run/results.json" --predictions "$run/private-predictions.json"
