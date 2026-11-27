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

From docs/parallel_hypotheses run `tectonic notebook.tex` (XeLaTeX plus BibTeX also works with the listed packages and Latin Modern fonts). The source includes companion TeX files and references.bib. The full source audit and links to releases are in prior-art.md.

## Sudoku complementarity audit · 27 November 2026

Of the 76 archived cases, 50 still had two different, wrong endpoints and a serial miss on CPU. The selected blank and proposed pair of digits matched the archived values on 1/76 boards. On 14/50 replay-eligible boards, each branch was correct on at least one cell missed by the other. A chooser with access to the true digits could assemble 0/50 exact boards. The median number of cells wrong in both endpoints was 30. Averaging the endpoint logits solved 0/50 boards. The conditional subset is development evidence only. The CPU replay used no GPU-hours. The registered replay source is e5da9d1. The next proposed experiment is a frozen, site-aligned peer correction against shuffled corrections and untouched continuation. No new architecture training or Slurm job is pending.

## GH200 verification · 27 November 2026

Job 8930277 ran verification revision 44b60c4 on one GH200. Slurm reported COMPLETED, exit 0:0, elapsed 32 s, 0.00889 allocated GPU-hours against the one-job 0.25-hour cap. No additional GPU job is committed. The unused portion of the cap is not transferable to a different experiment. All 76 archived fingerprints passed, 17 boards have reciprocal correct digits, 0 have a fully correct cellwise union, 0 were solved by averaged logits, and the median shared error count is 29. See experiments/parallel_hypotheses_sudoku/gh200-results.json. Private per-board digit grids are under /projects/u6aag/trm-residual/niccogrillo.u6aag/parallel-hypotheses-verify-44b60c4eb21fb6ffee959d051db91365254b1aa8/8930277/private-predictions.json, SHA-256 36a014ae33a6b845b5ffb4dfbf26c0ec3604e393248bba3b2f2824b9b4dab521. The public metrics file matched the remote SHA-256 107d6172f7734488f430ed7765be609f787b3e4f704ae63d6e4411c8956dcb32. The fresh-puzzle intervention is specified separately; its GH200 execution requires a new allocation decision.
