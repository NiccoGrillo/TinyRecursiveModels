# TRM residual reasoning — Codex project handoff

25 November 2026 · Prepared for Niccolò Grillo's local Codex project on macOS.

## Pick up here

The implementation is on `NiccoGrillo/TinyRecursiveModels`, branch `research/residual-reasoning-training`. Its verified tip at handoff is `74a08dc8e1bdf9dd4c10531944c7e8537c11fa38`. Open a local checkout of this branch in Codex. If the branch has advanced, inspect the later commits before assuming this snapshot describes them. The next milestone is a **single-GH200 acceptance run on Isambard-AI**. No cluster job was submitted in the originating Work chat.

Read `AGENTS.md`, `docs/RESIDUAL_TRAINING.md`, and the newest entry in `docs/STATUS.md` first. For the scientific reasoning, read sections 2, 3 and 5 of `docs/notebook/project_notebook.tex` or the accompanying PDF. The notebook is the November 20 design snapshot. The November 24 runbook describes the implemented system. The CRAM audit and the earlier trained-checkpoint experiments are in `docs/CRAM_REVIEW.md` and `docs/PILOT.md`. [Implementation branch](https://github.com/NiccoGrillo/TinyRecursiveModels/tree/research/residual-reasoning-training) · [Training runbook at the tested revision](https://github.com/NiccoGrillo/TinyRecursiveModels/blob/74a08dc8e1bdf9dd4c10531944c7e8537c11fa38/docs/RESIDUAL_TRAINING.md)

## The scientific question

Can a reasoner use additional loops to repair an incorrect candidate and keep a correct answer stable while working memory continues to evolve? Niccolò wants the model to learn how to make corrections from examples. Hand-coded Sudoku rules in the inference update would weaken the intended generalization to other structured tasks and eventually text.

An earlier study of a third-party TRM checkpoint found 25/32 dev puzzles correct at macro-step 16 and 27/32 at step 64 in FP32. The 25 initially correct puzzles stayed correct at the logged macro-steps. This small historical result motivates measuring **repair and retention separately**. The CRAM audit supplied a second reason to do so. A score can recognize valid candidates while a search procedure still struggles to find them within budget. The strongest test gives simpler search and correction methods the same score and starting errors.

## What the branch implements

- The latent candidate `q` decodes to the answer. The workspace `w` stores history. A shared neural field reads input and candidate and returns geometric channels `g` and a signed invalidity score `s`. Training encourages both to vanish at embeddings of correct answers.
- For `c = [g; s]`, the main update is `q_next = q - alpha * J_c(q)^T W(q,w,c,x)c`. The low-rank bounded positive-definite matrix mixes channels before the vector–Jacobian product. This can produce a correction when ordinary energy gradients cancel. At an exact residual zero, changing memory cannot move the candidate. A wrong zero would also be protected.
- Preserve the inner and outer derivatives. Supply `Wc` as the cotangent to the VJP. During training, gradients through the completed update must reach the field and controller. The losses supervise decoded answers across recurrent steps, calibrate small perturbations near correct codewords, and teach a signed margin on known incorrect examples.
- The frozen-field comparison holds the encoder, initializer, field, and output interface fixed. Controllers face the same saved reference states. The joint comparison lets each system train its own field and initializer. Cheaper rivals include ordinary descent, direct geometric correction, invalidity-gated proposals, stateless mixing and score-only one/two-token search. The full matrix also includes fresh-random mixing, direct residual maps and established optimizers. Claims about memory depend on the stateless and memory-intervention controls.
- The two implemented tasks are verified-unique Sudoku and generated modular prefix sums. Sudoku uniqueness is checked during data preparation. The inference interface uses learned fields. The doubled-length sequence test measures length generalization. ARC and language-model transfer remain broader research aims.

## What has actually run

The recorded CPU suite passed **72 tests**. The miniature campaign produced **18 model records and 66 split-level evaluations** with complete coverage. Simulated interrupted training reproduced the uninterrupted CPU weights. The Slurm array/dependency wiring was checked against a scheduler substitute. CUDA memory and throughput, NCCL, production-scale data preparation, and scientific performance on the cluster still need measurement. Treat smoke scores as implementation checks.

## First cluster session

Keep Codex on the Mac and the user's ordinary Clifton-authenticated SSH session in Terminal. The desktop SSH-project feature starts a Codex server on the remote machine. BriCS directs users to run agents locally, review cluster commands and run them themselves. It disallows agents on login nodes and advises against persistent agent connections and unreviewed execution. Codex can independently inspect and fix the local checkout, prepare a complete launch block, and analyze retrieved outputs. Slurm handles the unattended jobs after the user's submission. [BriCS AI-agent guidance](https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/) · [Clifton login](https://docs.isambard.ac.uk/user-documentation/guides/login/) · [Codex SSH host behavior](https://learn.chatgpt.com/docs/remote-connections)

1. **Hardware acceptance.** Use the runbook's shared checkout and Arm64 PyTorch SIF setup. Use absolute `RR_IMAGE`, `RR_DATA`, and `RR_ROOT` paths, with a dedicated acceptance result root. No production dataset is needed. Prepare the user-reviewed submission `RR_GPUS=1 bash scripts/isambard/submit.sh smoke`. It requests one GPU for up to 20 minutes. Capture the job ID.
2. **Check acceptance.** Inspect the Slurm exit state and log, `$RR_ROOT/smoke/production-shape.json` and `$RR_ROOT/smoke/campaign/reports/coverage.json`. Record elapsed time and peak allocated bytes for the production-shaped synthetic update, and check for 18 model records and zero missing evaluations. A separate two- or four-GPU smoke run checks NCCL if that topology will be used.
3. **Development pilot.** Prepare the two datasets using the pinned inputs in the runbook. Choose a fresh pilot result root and `config/residual/pilot.json`. Preview the plan. One reviewed `bash scripts/isambard/submit.sh submit` issues the reference and joint training arrays, dependent shared-controller and bank arrays, evaluation, and final report. The pilot uses seed 11, 250 updates per learned model and dev-only evaluation. Concurrency limits apply per array, so several stages can use GPUs simultaneously. Use the pilot measurements to estimate the full study's cost against the allocation.
4. **Full study.** The separate `config/residual/study.json` uses three seeds, two reference fields, 5,000 updates per learned model, and ablations. Keep dev tuning and held-out reporting distinct. To recover an interrupted campaign, inspect the failed job and the sealed configuration before retrying. Completed matching jobs and checkpoints are reusable. Avoid duplicating an active campaign.

## Interpreting returned results

Start with `study.json`, `reports/coverage.json`, and `reports/README.md`. Compare methods at the same task, seed, split, reference, fork and policy. The primary shared-field comparison is the fraction of genuinely wrong starts repaired within the common measured time budget. Report its denominator alongside correct-start retention, ordinary-start solve rate, and cap hits. `paired.csv` exposes helped/harmed counts. `training.csv` exposes realized costs. Inspect `interventions.csv` before attributing a gain to memory, and trace anomalous outcomes by original input ID. Equal optimizer-step counts can entail different training costs.

Niccolò prefers clear explanations of why an experiment answers a question and what result would change the research decision. Keep observed results, tested implementation properties and hypotheses distinct. If a cheaper method matches the mixer at equal cost and retention, say so.
