# TRM Decision Stability

**Can a recurrent reasoner preserve a correct answer while continuing to compute?**

This project starts from [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), the official implementation of [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871). It studies whether changes to a reasoning state that leave the current output unchanged can affect future answer retention.

The working hypothesis is that useful internal computation and harmful future decision changes should be distinguished before imposing contraction on the whole latent state. This is a hypothesis to test, not a demonstrated improvement or a claim of publication novelty.

**Current decision:** the first checkpoint pilot found late recovery without observed correct-answer loss. The next study is a [same-state, equal-budget continuation-schedule comparison](docs/NEXT_EXPERIMENT.md), with dense observation and separate native-output retention. Do this before expanding perturbations or training a stabilizer. Close prior work has also narrowed the original novelty claim.

## Start here

| File | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Context, commands, and scientific constraints for Codex |
| [Project](docs/PROJECT.md) | Research question, mathematics, and novelty boundaries |
| [Experiments](docs/EXPERIMENTS.md) | Runnable commands, metrics, and interpretation |
| [First checkpoint pilot](docs/PILOT.md) | Public weights, provenance, fixed CPU protocol, and reports |
| [Next consequential experiment](docs/NEXT_EXPERIMENT.md) | Frozen protocol, launch commands, interpretation and stopping criteria |
| [Resources](docs/RESOURCES.md) | Papers, code, and relevance |
| [Roadmap](docs/ROADMAP.md) | Next tasks, baselines, and decision gates |
| [Status](docs/STATUS.md) | Implemented, verified, and still untested |
| [Critic review](docs/CRITIC_REVIEW.md) | Objections incorporated and remaining risks |
| [Upstream README](README_UPSTREAM.md) | Original training recipes and attribution |

## CPU quickstart

Run from the repository root. Python 3.12 and PyTorch 2.7.0 CPU were used for verification.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-experiments.txt
python -m pytest -q
python -m experiments.decision_stability.run \
  --smoke --out runs/smoke \
  --steps 8 --fork-steps 2 4 --horizon 4 --targeted --probe-horizon 2
python -m experiments.decision_stability.analyze runs/smoke
python -m experiments.decision_stability.report runs/smoke
```

The smoke test uses **random weights and two synthetic puzzles**. Its accuracy and perturbation effects are not research results. Output directories must be new. For trained weights, use the [checkpoint/data runbook](docs/EXPERIMENTS.md). No checkpoint or benchmark data is bundled, and no full training run has been performed for this project.

## Implemented experiments

1. Extend a frozen TRM at a fixed recursion phase, logging retention, recovery, rule validity, margins, Q output, and latent changes.
2. At predetermined times, fork the same state and perturb only `z_L`. Assert identical immediate logits; compare clean, random ±, and optional future-margin gradient ± continuations.
3. Save JSONL records, reproducibility manifests, and independently recomputable summaries; generate readable Markdown/CSV reports with per-fork puzzle-paired uncertainty.
4. Fork the identical native full state into L=6/L=12 schedules with equal block budgets; distinguish intermediate excursions from native macro-output failures, and report paired exact binary-effect bounds.

Gradient arms are **offline sensitivity attacks** with extra compute. They are not a stopping policy, a validated predictor of spontaneous errors, or evidence of the cause of native failures.

For the new study, reuse pilot inputs, run `.venv/bin/python scripts/fetch_attention_checkpoint.py`, then `bash scripts/run_clock_study.sh` on a GPU host with CUDA PyTorch. The [runbook](docs/NEXT_EXPERIMENT.md#checkpoints-and-launch) documents paths and overrides. The full 1,024-puzzle study has not been executed in this workspace; CPU-only implementation checks and strict attention-checkpoint compatibility have passed.

The additions are in `experiments/decision_stability/`, with tests in `tests/decision_stability/`. The upstream `models/`, `pretrain.py`, `dataset/`, and `config/` are unchanged. One macro-step uses `H_cycles*(L_cycles+1)*L_layers` block evaluations: **42** at the original Sudoku settings. Probe and branch costs are logged separately.

## Upstream and licence

Base: [`c01103738605ba39d1430519b1ee0c62f4c707f8`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/commit/c01103738605ba39d1430519b1ee0c62f4c707f8). Original Git history, [MIT licence](LICENSE), and [README](README_UPSTREAM.md) are preserved. Cite TRM and relevant prior work. These additions do not imply endorsement by upstream authors.
