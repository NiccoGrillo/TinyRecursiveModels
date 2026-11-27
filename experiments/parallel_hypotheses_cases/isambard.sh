#!/usr/bin/env bash
#SBATCH --job-name=ph2-parity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
set -euo pipefail
: "${SLURM_JOB_ID:?Execute as a Slurm job}"
: "${RR_IMAGE:?Set to the existing Arm64 PyTorch SIF}"
: "${RR_ROOT:?Set to the project results directory}"
repo=$(git -C "${SLURM_SUBMIT_DIR:?}" rev-parse --show-toplevel)
out="$RR_ROOT/ph2-$SLURM_JOB_ID"
mkdir -p "$RR_ROOT"
mkdir "$out"
git -C "$repo" rev-parse HEAD > "$out/revision.txt"
git -C "$repo" diff --exit-code HEAD -- experiments/parallel_hypotheses_cases
sha256sum "$RR_IMAGE" > "$out/image.sha256"
scontrol show job "$SLURM_JOB_ID" > "$out/job.txt"
srun apptainer exec --nv --bind "$repo:$repo" --bind "$RR_ROOT:$RR_ROOT" --pwd "$repo" "$RR_IMAGE" \
    python -m experiments.parallel_hypotheses_cases.study --device cuda --out "$out/study"
