# Living project guide

Start with [the PDF](project_notebook.pdf). Its main sections explain the project history, completed tests, theory and the next experiment. Proofs, provenance, commands and a glossary are in the appendices.

[project_notebook.tex](project_notebook.tex) is the editable, self-contained source. Upload this one file to Overleaf and select **XeLaTeX**, or run from the repository root:

```bash
bash scripts/build_notebook.sh
```

A normal TeX Live installation needs the packages in the preamble (including fontspec, unicode-math, TikZ, fvextra and xurl) and Latin Modern fonts. Compilation intermediates go in the ignored `runs/notebook-build/` directory. Source and PDF are both versioned; no separate bibliography or image assets are required.

When results or the research direction change:

1. Update the dated status and stable experiment ledger; distinguish measured results, deductions, implementation checks and proposals.
2. Preserve the original hypotheses and frozen protocols. Explain what evidence changed a decision.
3. Update relevant tables, source revision, limitations and the notebook change log alongside `docs/STATUS.md`.
4. Rebuild, resolve references/overflow warnings and visually inspect the PDF. Commit source and PDF together.

The notebook supplements the detailed runbooks; it does not silently change their protocols. Version 1.0 explains the research snapshot at `8942518`; the completed pilot results are sourced from the committed reports, not a new raw-data reanalysis.
