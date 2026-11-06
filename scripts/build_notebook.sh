#!/usr/bin/env bash
# Build the self-contained living guide with a normal XeLaTeX installation.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/runs/notebook-build"
source_file="$repo_root/docs/notebook/project_notebook.tex"
command -v xelatex >/dev/null || {
  echo "XeLaTeX is required. Install TeX Live with the source's packages, or use Overleaf (XeLaTeX)." >&2
  exit 1
}
mkdir -p "$build_dir"
for pass in 1 2 3; do
  xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory="$build_dir" "$source_file"
done
cp "$build_dir/project_notebook.pdf" "$repo_root/docs/notebook/project_notebook.pdf"
echo "Built docs/notebook/project_notebook.pdf"
