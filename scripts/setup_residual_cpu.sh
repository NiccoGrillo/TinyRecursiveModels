#!/usr/bin/env bash
# Local Codex worktree setup. The GH200 jobs use the NGC environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then
    py=${PYTHON_BIN:-python3}
    if [[ -z ${PYTHON_BIN:-} ]] && command -v python3.12 >/dev/null; then py=python3.12; fi
    "$py" -m venv .venv
fi
py=.venv/bin/python
"$py" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13), "Use Python 3.11 to 3.13; CI uses 3.12"'
args=()
if [[ $(uname -s) == Linux ]]; then args+=(--index-url https://download.pytorch.org/whl/cpu); fi
"$py" -m pip install torch==2.7.0 "${args[@]}"
"$py" -m pip install -r requirements-experiments.txt
