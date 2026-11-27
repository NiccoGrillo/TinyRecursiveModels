# Parallel hypotheses · 27 November 2026

Start a resumed session with notebook.pdf, this file, the registered protocols and the live user queue. The base is 2de6e27b20e3decb6f264a58966a4c8aab4d3a8a. Use research/parallel-hypotheses. The previous project's branch and sealed 896-board Sudoku split are separate.

| Date | Decision and evidence |
| --- | --- |
| 27 Nov | Commissioned two independent designs and a literature review. Deferred a proof-by-cases fusion primitive because constructive disjunction and a mean/disagreement recurrent rival already perform the operation. |
| 27 Nov | Registered PH1 and the public ARC bank at 71a689d. Registered the additional serial reader at e8edd81. PH1 missed its continuation gate. A whole-message transplant cannot distinguish final-answer transmission from a useful intermediate. |
| 27 Nov | Registered PH2 at 95d7e7b, after PH1 and before PH2 training. The explicit two-world task separated conditional reads from the final lookup. The selector learned two constant addresses; its candidate set contained every answer. Closed the new-answer advantage on this task. |
| 27 Nov | Reviewed all three seeds, per-query estimates, non-overlap of split groups, and independent post-evidence critiques. Re-ran PH1, PH2 and the ARC scorer; predictions/task records reproduced byte-for-byte. Previous 78 CPU tests and 9 additional focused tests passed. |

PH1 trains both recurrent arms and the older results live in experiments/parallel_hypotheses/results. The serial control is reproduced with
`python -m experiments.parallel_hypotheses_serial.study --out runs/ph1-serial`. The predeclared serial search reached 251, 277 and 368/368 in seed order; report all three. Follow the other per-experiment READMEs for data, reproduction and run policy. Full reconstruction of the toy datasets uses only the source and the saved group memberships. The PH1, PH2 and ARC per-example outputs are under the respective results directories. No third-party checkpoints or datasets are redistributed.

## Resources and continuation

The phase-2 user queue was empty at inspection. No Slurm job was submitted. The historical job 8916461 was complete. One-GPU jobs and the existing Arm NGC container can run via the shared filesystem. This project has no explicit discretionary GPU envelope. Spent 0, committed 0, currently authorized to spend 0 GPU-hours. A 15-minute one-GPU hardware-only replay is prepared in experiments/parallel_hypotheses_cases/isambard.sh and shell-checked. Existing results provide no reason to spend it on accuracy. Both learned experiments used the local Apple ARM CPU with one numeric thread. A second full PH2 run peaked at 309,215,232 bytes RSS and reproduced the complete prediction ledger. PH1 peaked at 44,990,464 bytes. Peak RSS includes the interpreter, runtime, optimizer and corpus.

The distinguishing evidence a new project would need is a natural full-input case where two independently specified, unsuccessful proposals each carry a different needed fact, and patching that fact changes the next operation and yields an answer outside the old candidate set. Include an equally informed serial parent, an online selection/resampling policy, and role-matched donor controls. Audit the supply of operation boundaries. This is a criterion for reopening, not an experiment in progress. The PH1/PH2 reserve groups, Sudoku holdout and ARC evaluation split stay unopened. No active jobs or monitoring.

## Build the notebook

From docs/parallel_hypotheses run `tectonic notebook.tex` (XeLaTeX plus BibTeX also works with the listed packages and Latin Modern fonts). The source includes five companion TeX files and references.bib. The full source audit and links to releases are in prior-art.md.
