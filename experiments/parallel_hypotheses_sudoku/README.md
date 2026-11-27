# Development-only complementarity audit

Before inspecting per-cell outcomes, select all 76 archived boards where both digit-conditioned endpoints were wrong and disagreed and the matched serial trajectory never reached an exact answer. The original full cohort is 256 development boards. Replay from the saved depth-two fork, reconstruct depth 12 then run the two learned hypotheses for two more macros and the ordinary serial trajectory for four. Report whether one path alone is correct at some cells and the other alone at others, the number of wrong cells shared by both, and how many complete boards a per-cell label oracle can assemble when both endpoints are wrong. Compare simple label-free average endpoint logits, plus the best whole candidate in the four-step serial run. Record concordance with archived selected cell, digit hypotheses and endpoint exact flags. CPU BF16 replay can diverge from GH200; report as new CPU evidence. These are selected discovery cases. Any gate or training policy will require fresh original boards. If exact oracle fusion gives little headroom, drop copying branch outputs as the next architecture; incomplete consequences could still matter for later learned deductions.

## Interpretation and next decision

A shared wrong cell rules out recovering that board solely by copying one of the two final top-one digits at each position. It leaves open information in logits and the possibility of a new deduction. Reciprocal exclusive correct cells can arise from chance variations. A replay must report the subset still wrong, distinct and missed by serial on CPU separately from all 76 archived cases. The summaries and labels are for diagnosis only.

The proposed next causal test uses fresh, original development puzzles, locked by ID and augmentation family. At a fixed depth generate two candidate states with the frozen executor. Use a label-free policy to select eligible disagreement cells and expose the same peer corrections to two otherwise identical continuations. One writes the correction at its corresponding cell; the other permutes whole correction vectors among the same cells within the same board, preserving the multiset and work. Fix permutation seed and dose before evaluation, verify post-cast correction norms and readout parity before continuation, and use a predeclared no-op for fewer than two sites. Include untouched continuations, simple logit pooling, matched serial work and a whole-trajectory selection policy. Paired exact solutions, gains and losses on all boards are primary. Determine whether the next update changes in a way predicted by the aligned content. A reproducible advantage over both misalignment and untouched continuation is the reason to consider training a message writer. A result explained only by damage in the shuffled arm is insufficient.

## Reproduction

From the checkout root use the fork's CPU PyTorch environment with:

    PYTHONPATH=.:scripts python experiments/parallel_hypotheses_sudoku/replay.py \
        --report /path/to/native-hypotheses/result.json \
        --calibration /path/to/dev-calibration/calibration.json \
        --bank /path/to/dev-calibration/fork_states.pt \
        --weights /path/to/checkpoint/checkpoint.pt \
        --config /path/to/checkpoint/all_config.yaml \
        --edit /path/to/native-hypotheses/workspace-edit.pt \
        --out /path/to/replay.json
    python experiments/parallel_hypotheses_sudoku/summarize.py /path/to/replay.json

These paths are relative to the archived project run trm-refinement-K8SH3r on shared project storage; the staged native checkpoint and config must match the recorded digests. No query grids, target grids, checkpoint files or fork tensors are added to the research branch.

The recorded CPU results are from revision e5da9d1. For a same-runtime read-only replay, use the recovery command above with --device cuda in the original GH200 container and an output directory on shared storage. This path now rejects a mismatch in device name, PyTorch version, proposed cell or digits, or archived stepwise exact flags. None of these fingerprints replaces saving full branch predictions in future campaigns. A single GPU with a 15-minute Slurm wall limit is a proposed ceiling of 0.25 requested GPU-hours, conditional on the user's authorization. No GPU validation or submission has taken place.
