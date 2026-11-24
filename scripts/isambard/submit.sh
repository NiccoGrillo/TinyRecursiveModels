#!/usr/bin/env bash
# plan, smoke, or submit. Campaign submission is explicit.
set -euo pipefail
action=${1:-plan}
case "$action" in plan|smoke|submit) ;; *) echo "Usage: $0 [plan|smoke|submit]" >&2; exit 2;; esac
export RR_REPO
RR_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$RR_REPO"
export RR_CONFIG=${RR_CONFIG:-"$RR_REPO/config/residual/pilot.json"}
export RR_GPUS=${RR_GPUS:-1} RR_CONCURRENCY=${RR_CONCURRENCY:-2}
: "${RR_ROOT:?Set the absolute campaign output directory}"
: "${RR_DATA:?Set the absolute prepared-data directory}"
: "${RR_IMAGE:?Set the absolute SIF filename}"
for path in "$RR_CONFIG" "$RR_IMAGE" "$RR_ROOT" "$RR_DATA"; do
    [[ $path == /* ]] || { echo "Paths must be absolute: $path" >&2; exit 2; }
done
[[ $RR_GPUS =~ ^[1-4]$ && $RR_CONCURRENCY =~ ^[1-9][0-9]*$ ]] || exit 2
export RR_ROOT RR_DATA RR_IMAGE
runner="$RR_REPO/scripts/isambard/container.sh"
study=(python -m experiments.residual_reasoning.study)
if [[ $action == plan ]]; then
    exec "$runner" "${study[@]}" plan --config "$RR_CONFIG"
fi
RR_IMAGE_SHA256=$(sha256sum "$RR_IMAGE" | cut -d' ' -f1)
export RR_IMAGE_SHA256
mkdir -p "$RR_ROOT/slurm"
base=(--parsable --nodes=1 --ntasks=1 --cpus-per-task="${RR_CPUS:-8}" --export=ALL
      --chdir="$RR_REPO" --output="$RR_ROOT/slurm/%x-%A_%a.out")
if [[ -n ${RR_ACCOUNT:-} ]]; then base+=(--account="$RR_ACCOUNT"); fi
if [[ -n ${RR_PARTITION:-} ]]; then base+=(--partition="$RR_PARTITION"); fi
worker="$RR_REPO/scripts/isambard/job.sh"
if [[ $action == smoke ]]; then
    exec sbatch "${base[@]}" --job-name=rr-smoke --time=00:20:00 --gpus="$RR_GPUS" "$worker" smoke
fi
"$runner" "${study[@]}" seal --config "$RR_CONFIG" --data "$RR_DATA" --root "$RR_ROOT"
submit_stage() {
    local stage=$1 gpu_count=$2 dependency=${3:-} n identifier
    n=$("$runner" "${study[@]}" count --config "$RR_CONFIG" --stage "$stage")
    local opts=("${base[@]}" --job-name="rr-$stage" --array="0-$((n-1))%$RR_CONCURRENCY"
                --time="${RR_WALL:-04:00:00}" --gpus="$gpu_count")
    if [[ -n $dependency ]]; then opts+=(--dependency="afterok:$dependency"); fi
    identifier=$(sbatch "${opts[@]}" "$worker" "$stage")
    identifier=${identifier%%;*}
    echo "$stage $identifier" | tee -a "$RR_ROOT/slurm/submissions.txt" >&2
    echo "$identifier"
}
ref=$(submit_stage reference "$RR_GPUS")
joint=$(submit_stage joint "$RR_GPUS")
shared=$(submit_stage shared "$RR_GPUS" "$ref")
bank=$(submit_stage bank 1 "$ref")
ev=$(submit_stage eval 1 "$shared:$joint:$bank")
# Reuse the GPU queue for a brief CPU analysis job where the site requires a GPU request.
sbatch "${base[@]}" --job-name=rr-report --time=00:30:00 --gpus=1 \
    --dependency="afterok:$ev" "$worker" report | tee -a "$RR_ROOT/slurm/submissions.txt"
