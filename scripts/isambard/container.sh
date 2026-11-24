#!/usr/bin/env bash
# Run from the checkout or invoke by absolute path. RR_ROOT and RR_DATA are on shared storage.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${RR_IMAGE:?Set RR_IMAGE to the absolute path of the Arm64 PyTorch SIF}"
: "${RR_ROOT:?Set RR_ROOT to the output directory for this campaign}"
: "${RR_DATA:?Set RR_DATA to the parent directory of prepared tasks}"
mkdir -p "$RR_ROOT" "$RR_DATA"
args=(exec --bind "$repo:$repo" --bind "$RR_DATA:$RR_DATA" --bind "$RR_ROOT:$RR_ROOT" --pwd "$repo")
if [[ ${RR_USE_GPU:-0} == 1 ]]; then args+=(--nv); fi
exec apptainer "${args[@]}" "$RR_IMAGE" "$@"
