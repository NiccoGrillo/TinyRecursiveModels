# Training the learned residual reasoner

The first cluster run answers a practical question. Can the field, controller and cheap correctors train, checkpoint and evaluate on exactly the same inputs at an affordable cost per update? The development pilot measures this on one seed. The full study asks whether the recurrent mixer repairs more genuine errors at equal inference time, preserves already correct answers and improves ordinary-start solving across seeds and tasks.

## Choose a run

| Configuration | Work performed | Data read for evaluation |
|---|---|---|
| [Smoke](../config/residual/smoke.json) | Two optimizer steps per learned model, tiny sequence dataset, reference banks, all reporting stages. A multi-GPU request also runs a distributed training check. | Synthetic dev, test and longer sequences |
| [Pilot](../config/residual/pilot.json) | Default cluster profile. Two tasks, seed 11, one reference, 250 updates for each learned model, five shared-field arms, two jointly trained rivals and two discrete searches. | Dev |
| [Study](../config/residual/study.json) | Both tasks, seeds 11/23/37, two reference fields, 5,000 updates per learned model, full controller matrix and four learning ablations. | Dev, test and a doubled-length transduction split |

These are initial experimental choices. The long study uses 64 examples per global update and four supervised loops. At the same number of updates, the VJP and direct methods have different training costs. Use the pilot's recorded durations and peak allocation to size the campaign. To run a separate equal-training-time comparison, set `training.max_seconds` to the same positive number in a copy of the config, use the same GPU count, and set `training.steps` high enough for each method to reach the time limit. Each run records which bound fired. The checked-in profiles do not consume an allocation until submitted.

## Local development and Codex

From a checkout of the implementation branch, on Linux or macOS with Python 3.12:

```bash
bash scripts/setup_residual_cpu.sh
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.residual_reasoning.smoke --root runs/local-acceptance
.venv/bin/python -m experiments.residual_reasoning.study plan --config config/residual/pilot.json
```

The Codex worktree setup and toolbar actions are in [.codex/environments/environment.toml](../.codex/environments/environment.toml). [AGENTS.md](../AGENTS.md) gives repository instructions and points to the result tables. The residual training path requires Python, NumPy and PyTorch. The test environment also installs the fork's diagnostic dependencies. It runs offline once dependencies and the two Sudoku CSVs are present.

## Isambard-AI commissioning

The launcher supports one node with one to four visible GPUs. One Slurm task owns the allocation. For distributed training it starts one PyTorch process per GPU using NCCL. Reference-state generation and timed evaluation each use one GPU. The same global batch applies at every GPU count. The precision baseline is FP32 with TF32 disabled. The attention implementation supports differentiation through a VJP. CUDA memory, throughput and multi-GPU operation need measurement on the allocated hardware. [BriCS ML environment](https://docs.isambard.ac.uk/user-documentation/applications/ML-packages/) · [Slurm](https://docs.isambard.ac.uk/user-documentation/guides/slurm/) · [Storage](https://docs.isambard.ac.uk/user-documentation/information/system-storage/)

**1. Prepare the checkout and image.** Use a shared checkout, a scratch location for inputs and the container, and project storage for results you want to retain. Run large transfers and preparation in the site's transfer or compute allocation. The image tag below is the Arm-compatible PyTorch example published by BriCS.

```bash
git clone --branch research/residual-reasoning-training https://github.com/NiccoGrillo/TinyRecursiveModels.git
cd TinyRecursiveModels
export RR_DATA="$SCRATCHDIR/trm-residual/data"
export RR_IMAGE="$SCRATCHDIR/trm-residual/pytorch_25.05-py3.sif"
export RR_ROOT="$PROJECTDIR/trm-residual/$USER/acceptance"
mkdir -p "$(dirname "$RR_IMAGE")" "$RR_DATA" "$RR_ROOT"
apptainer pull "$RR_IMAGE" docker://nvcr.io/nvidia/pytorch:25.05-py3
```

Reuse the SIF on subsequent submissions. Each submission records its SHA-256 in the training provenance. Set `RR_ACCOUNT` or `RR_PARTITION` if your allocation requires them. Configs should live in the checkout or under the mounted data/result directories. Use absolute paths.

**2. Run hardware acceptance.** This creates its own small dataset. On an initial single-GPU allocation:

```bash
export RR_GPUS=1
bash scripts/isambard/submit.sh smoke
squeue --me
```

Read `$RR_ROOT/slurm/` and `$RR_ROOT/smoke/campaign/reports/coverage.json`. A complete run has 18 model records and zero missing entries. To test the intended distributed topology, use a fresh acceptance root and set `RR_GPUS=2` or `4`. That job first exercises a real two-step, multi-process backward pass, then runs the small science workflow on one device. GPU acceptance also writes `smoke/production-shape.json` with peak memory and elapsed time for one synthetic update using the full Sudoku shape and study width, depth and per-rank batch. Smoke accuracy has no research interpretation.

**3. Stage the two tasks.** The pinned Sudoku source is the original dataset used by the fork. Fetch the CSVs once. The following commands store sources inside the mounted data directory:

```bash
mkdir -p "$RR_DATA/source"
revision=58942f96baeb572ca3127e2a9e9c70f330783d6b
for split in train test; do
    curl --fail --location --retry 3 \
      "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/$revision/$split.csv" \
      --output "$RR_DATA/source/$split.csv.part" && mv "$RR_DATA/source/$split.csv.part" "$RR_DATA/source/$split.csv"
done
scripts/isambard/container.sh python -m experiments.residual_reasoning.data sudoku \
    --train-csv "$RR_DATA/source/train.csv" --test-csv "$RR_DATA/source/test.csv" --out "$RR_DATA/sudoku"
scripts/isambard/container.sh python -m experiments.residual_reasoning.data scan --out "$RR_DATA/scan"
```

Both tasks use 8,192 training inputs, 512 dev inputs and 1,024 test inputs by default. The Sudoku loader streams unaugmented rows, selects distinct inputs by seeded hash, splits the training source into train/dev, and excludes their inputs when sampling the official test source. It checks every selected solution against the givens and searches for a second completion during preparation. This makes a different decoded board a legitimate negative. The solver never enters training dynamics or inference. Exact-input deduplication does not identify symmetry-equivalent boards. The new subset and training budget should be quoted with any result.

The non-puzzle task maps a sequence to its running sums modulo eight. It uses the same model and losses. Training/dev/test sequences have length 16. A separate 1,024-example split has length 32. All inputs are distinct across splits. This tests length generalization for an algorithmic sequence transformation. Prepared data includes fixed-length arrays per split, original-input hashes, editable-position metadata for training corruptions, source hashes and split sizes. The current implementation expects a single token label per position and unambiguous outputs.

**4. Preview and submit the development pilot.**

```bash
export RR_ROOT="$PROJECTDIR/trm-residual/$USER/pilot-v1"
export RR_CONFIG="$PWD/config/residual/pilot.json"
export RR_GPUS=1 RR_CONCURRENCY=2
bash scripts/isambard/submit.sh plan
bash scripts/isambard/submit.sh submit
```

The submission seals the config, code identity, datasets and complete indexed job plan in `study.json`. Reference pretraining and joint training can start independently. Shared controllers and state banks wait for references, evaluation waits for all model and bank arrays, and the report waits for evaluation. Array concurrency is per stage. `RR_WALL` controls each array element's wall limit and defaults to four hours. `RR_CPUS` defaults to eight. The launcher records job IDs under `slurm/submissions.txt`. Every dependency uses `afterok`.

**5. Run the multi-seed study** by choosing `$PWD/config/residual/study.json` and a fresh `RR_ROOT`, then using the same plan and submit commands. The full config is larger than the pilot. Use measured pilot resource costs to decide how many workers and concurrent elements to request. Freeze hyperparameters on dev before opening held-out reports. To run a deliberate subset, copy the JSON, edit tasks, methods, seeds or splits, and inspect its expanded plan.

## What trains

| Component | Implementation and comparison |
|---|---|
| Answer interface | One distinct frozen orthonormal code per output token. The fixed distance decoder reads only the continuous candidate. Each token has 12 latent coordinates in the main profile. |
| Initializer and field | Learned input embedding and sinusoidal positions feed the initializer. A separate attention trunk reads input plus candidate and emits geometric channels of size `S*d` and one pooled signed score `sqrt(S)*s_bar`. |
| Recurrent mixing | The controller reads input, candidate, both residual heads and the previous workspace. It predicts rank-two factors spanning every residual channel in one example. The existing bounded operator supplies the cotangent to the field VJP. The workspace updates from the same pre-step state through tanh. |
| Task and field losses | Average the fixed-codebook target KL at depths 1 through 4. Calibrate at anchors and within one quarter of the half-codeword distance. Score known-wrong candidates with a normalized signed margin. Reached candidates are detached only as auxiliary training examples. The task loss retains the full recurrence graph. |
| Negative examples | Average three sources with equal source weights: naturally wrong reached candidates, known-wrong answer codes from another example, and a swap of two distinct symbols on eligible sites. The random-edit ablation changes both symbols at the same eligible site pair. Empty source cohorts contribute zero with their eligibility fraction logged. |
| Frozen comparison | Train one identity-energy reference and one direct-correction reference independently. For each reference, copy and freeze the encoder, initializer, field and fixed interface. Only controller parameters are optimized. Parameter-free arms write an audited checkpoint with zero optimizer steps. |
| Joint comparison | Each arm trains its own model from the common interface seed. The free-energy arm learns all residual channels from the unrolled task loss, leaving its basis unconstrained. The unrestricted recurrent controls include a fixed decoder with target KL and a trainable readout with one-hot cross-entropy. |

The name `scan` in paths denotes the generated modular prefix-sum task. The data and model do not use the SCAN instruction-following benchmark. The `unrestricted` controls are matched building blocks developed on this branch. The fork's native TRM remains available through its existing training code. This suite does not reproduce published TRM, EBR or CRAM numbers.

### Controller and solver names

| Arm | One step |
|---|---|
| `recurrent`, `stateless` | Bounded low-rank residual metric before the VJP, with or without persistent workspace. |
| `fixed`, `fixed_learned`, `random` | One seeded frozen metric, one input-independent trainable factor, or a fresh random factor at each step. |
| `identity`, `diagonal`, `state_side` | Ordinary squared-residual gradient, its bounded diagonal residual weighting, or a bounded state-space metric after the ordinary VJP. |
| `direct`, `gated`, `residual_map` | Subtract the full geometric residual, learn a proposal gated by raw invalidity, or use a learned diagonal-plus-low-rank map from residual to correction. The learned rivals have the same residual and memory access. |
| `adam_energy`, `snr` | Apply bias-corrected Adam moments to the ordinary energy gradient (0.9/0.999 with smooth epsilon), or use a fresh Gaussian residual sketch and solve a damped rank-sized least-squares system. SNR charges one VJP per sketch column. |
| `search1`, `search2` | Propose one or two distinct token changes from the same decoded starting answer. Score fixed codewords with the reference's learned score. Search returns the best score seen, charges the initial score and every proposal/restart, and uses all output positions. Uphill probability and restart period are config values. |

The profile uses a fixed step of 0.5 outside the full-step direct arm. The mixer uses `m=1` and `b=2`. Low-rank maps and fixed-width rivals have different parameter counts, reported for every model. The sequence interface and hyperparameters are research choices for this implementation.

## Reading a result

Start with **`reports/README.md`** and **`reports/coverage.json`**. The overview groups seeds by task, split, condition and reference. On a shared frozen field, the primary endpoint is repair among examples wrong at the same reference fork, under the common time budget. Check retention among initially correct examples and ordinary-start solve rate alongside it. An empty denominator is blank in CSV and a dash in Markdown.

| File under the campaign root | Contents |
|---|---|
| `study.json` | Effective config, source identity, dataset fingerprints and stage/index mapping |
| `models/<task>/seed-<n>/<condition>/…/` | Model/optimizer checkpoint, manifest, training-loss JSONL and completion record with checksum, elapsed time and examples processed |
| `banks/`, `prices/` | Shared continuous candidates keyed by example ID and fork, and development-only reference timings for each length |
| `evaluations/…/<split>/examples.jsonl` | One record per original input and fork. Correctness trace, exact and token counts at each policy endpoint, elapsed batch time, operation counts, cap hits and numerical failures |
| `evaluations/…/<split>/diagnostics.jsonl` | Prespecified sample of residual and derivative probes, anchor calibration, input/candidate swaps and matched memory interventions |
| `reports/metrics.csv` | Per-seed solve/repair/retention, accuracy over all tokens and separately over editable tokens, raw counts, timing, cap and operation counts at every requested horizon and stopping policy. Editable tokens are the blank Sudoku cells or all prefix-sum positions. Older evaluations leave the token columns blank. |
| `reports/paired.csv` | Joined-by-input wall-budget differences, recurrent mixer minus rival. Helped/harmed counts and a pointwise paired bootstrap interval within each seed and fork |
| `reports/training.csv`, `reports/diagnostics.csv`, `reports/interventions.csv` | Parameter counts and training costs, diagnostic means, and matched continuation results including the dev-defined high-residual/low-gradient cohort |

**Budget semantics.** The fixed-depth curve includes depth zero and continues to 16 steps in the study. The shared-field arms start at identical saved candidates and cold-start their own memory at forks 0, 2 and 4. A joint system starts from its own initializer. Time budgets use the median of five warm development batch timings for four reference steps at the same length. For length 32, dev inputs are repeated and truncated for a label-free timing workload. Arm-specific synchronized timings include step and readout. Encoding/initialization or bank restoration and the initial readout are timed separately. The wall endpoint uses the last completed prefix inside the budget. Raw records expose the first over-budget time and the evaluation cap. These timings describe batched inference under the measured runtime. Patience is replayed independently on the same traces and falls back to the horizon if the decoded output never stabilizes. Ground truth only defines offline endpoints.

**Mechanism checks.** The first 64 examples have probes two steps after each fork. Record the ordinary gradient, separate head norms, one geometric Rademacher VJP probe, and responses to swapping input/answer partners. Two-step continuations from the exact same candidate and memory compare native mixing with resets, fresh random factors and identity mixing. The high-residual/low-gradient subset uses the upper residual quartile and lower ordinary-gradient quartile among dev wrong probes in the same arm/fork. Treat tiny and empty cohorts as such. Optional `evaluation.recompute_memory=true` adds the more expensive memory-recomputation intervention. Probe work runs outside inference timings. Per-example traces preserve first-correct times and transient losses even when endpoint correctness is equal.

**Inspect an incomplete campaign or resume it.**

```bash
scripts/isambard/container.sh python -m experiments.residual_reasoning.report --root "$RR_ROOT"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

The report lists missing evaluations explicitly. The final batch job also passes `--require-complete`. A training retry restores optimizer state and resumes from the last atomic checkpoint, reusing the same step-indexed batches and random seeds at the original world size. A torn progress log is reconciled to the checkpoint. Concurrent writers are excluded. Jobs with matching sealed, completed outputs return successfully. After an interrupted campaign has stopped, resubmitting the same config, data, image, GPU topology and root reuses its completed work. If the scheduler wall limit was too short, increase `RR_WALL` before retrying. A changed branch revision or study spec needs its own run root. Portable first-pass review needs only `study.json` and `reports/`. Keep the archived examples and checkpoints for paired reanalysis and exact reproduction.
