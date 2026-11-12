# Living project guide

Start with [the PDF](project_notebook.pdf), **Section 1**. Version 2.1 adds four adversarial critique rounds, an assessment of the supplied QRR reviews, and the leading residual-mixing architecture. It includes the strict-local-minimum example, precise preservation/attraction claims, random-mixing and Sketched Newton–Raphson objections, and a concrete grounded Sudoku factor-controller design. The old state-preconditioner remains a baseline; the expanded attracting-answer-set appendix and all historical results are preserved.

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

The notebook supplements the detailed runbooks; it does not silently change their protocols. Version 2.1 (12 November 2026) continues the architectural pivot and supersedes version 2.0 as the current research decision. E4/E5 remain optional historical protocols. The completed pilot results are sourced from committed reports, not a new raw-data reanalysis. The operator prototype is tested; complete architectures and their gains remain untested. No new trained result is claimed.
