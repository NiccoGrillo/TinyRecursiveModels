# Living research guide

Read [project_notebook.pdf](project_notebook.pdf). **Version 3.1** records the 20 November 2026 design, single-GH200 hardware acceptance and development evidence through 26 November. Section 5 presents the CRAM checkpoint certificate, a score-only search comparison, and a natural-error repair endpoint. The [full research note](../CRAM_REVIEW.md) includes the proof and reproduction script.

- Sections 1–2: objective, notation, architecture and one update.
- Section 3: local calibration, signed learned invalidity, and task-through-unroll training. This is a proposed recipe, not a solved learning guarantee.
- Sections 4–5: mathematical limits and the decisive comparison against simpler learned correctors.
- Sections 6–7: actual evidence, critic verdict and relevant prior work.
- Appendix A: proofs and counterexamples.
- Appendix B: attracting answer sets, with intuition, equations, readout alignment and discrete-time limitations.
- Appendix C: compact stable E0–E5 ledger and links to the full historical record.

The [previous 60-page notebook](archive/project_notebook_v2_1.pdf) and its [source](archive/project_notebook_v2_1.tex) are archived byte-for-byte. Its grounded-Sudoku recommendation is superseded; its experiments and protocols are preserved rather than silently deleted.

The current [LaTeX source](project_notebook.tex) is self-contained. Select XeLaTeX in Overleaf, or run from the repository root:

```bash
bash scripts/build_notebook.sh
```

A normal TeX Live installation needs the packages in the preamble and Latin Modern fonts. Intermediates go to ignored `runs/notebook-build/`. Keep source and PDF together in commits.

For future updates, revise the core explanation rather than prepend a long research-session transcript. Keep observations, proofs, implementation checks and proposals distinct. Update `docs/STATUS.md`, retain experiment IDs and frozen protocols, resolve references/overflow warnings, and visually inspect the rebuilt PDF.
