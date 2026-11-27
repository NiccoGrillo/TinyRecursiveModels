# Status

## 27 November 2026 · End-to-end native mixing comparison

This supersedes the warm-start comparison as the architectural test. Native TRM, the scalar gate and directional mixing start from identical random tensors for all shared parameters, with zero initial gates. They train all shared weights, including the native workspace and embeddings, from their first step. The archived checkpoint supplies the native architecture recipe and the earlier traces supply the frozen timing allowance, with no pretrained weights loaded into these arms.

[The runner](../scripts/run_trm_answer_from_scratch.py) uses the original TRM batch size 768, BF16 forward, native stablemax and halt loss, AdamATan2, sparse embedding SignSGD, the archived 2,000-step warmup and training EMA. The seed is 11 throughout. Each optimizer step offers the same deterministic augmented rows to all arms. The native arm records its ACT resets, and the gate arms replay those masks while learning their own halt targets. This keeps the actually supervised examples and order identical at common checkpoints, including repeated presentations during ACT. It is a comparison under a fixed supervision schedule. Separately optimized adaptive halting is outside this pilot.

Each arm has two 512-update rate trials with the same per-trial time cap, starting fresh in each case. Native tries its original dense rate and twice that rate. Both mixer arms use the selected native backbone rate and each tries that rate and four times that rate for its own head. Choose on depth-16 development exact solves, then blank accuracy, with a prespecified base-rate tie and unequal-step fallback. Reinitialize and train the main arm with the selected rate. Trial cost is reported separately. No held-out data enters tuning.

Stage one targets 4,096 updates and checkpoints every 512. Both exact solves and blank accuracy are evaluated on the 256 development boards at each checkpoint. The curve includes depth 1 through 16 exact counts, supervised examples, cumulative GPU training wall time, training token and halt loss, and post-cast mixer activity. Shared-time comparisons use the latest saved checkpoint at or before each common time; the equal-exposure comparison uses identical supervised-row digests. The decisive endpoint is the last synchronized readout within the frozen batch-one allowance, with native, scalar and directional checkpoints from the same latest equal-exposure step. Report pairwise gains/losses with exact conservative intervals.

If all arms reach 4,096 updates and any gains at least six depth-16 development solves from update 3,072 to 4,096, extend all three trajectories to 8,192. Resume model, EMA, optimizers, carry and RNG. Cap active training at 48 minutes in stage one and 96 minutes cumulatively, plus the equal tuning allowance of up to four minutes per trial. If the cap limits any arm, report the matched observed range and avoid a claim about unattained performance. The 896 held-out boards remain sealed. A final comparison requires a locked protocol and replication across seeds.

The warm-start chain was cancelled before any allocation ran. The replacement source is `011c6de`. Isambard submitted `8915525` (preflight), `8915526`–`8915528` (stage-one native, scalar, directional), `8915529` (curves and extension decision), and conditional extension jobs `8915530`–`8915533`. The CPU suite passed 78 tests. A BF16 CPU check verified identical shared random tensors and zero-gate carries/logits with the actual native model recipe. An eight-step CPU fixture verified both phases of optimizer and carry restore and equal effective supervised stream digests across all three arms. The GPU preflight uses the production optimizer.

## 27 November 2026 · Native answer-update mixing protocol

The standalone residual solver and frozen-refiner recipe are closed. This pilot fine-tunes the same competent native TRM checkpoint, SHA-256 `533999676a1d44bdb84205461b55ecf1719f238cad2c2af699de78128e3dbf7f`, as unmodified native TRM, workspace-conditioned scalar TRM, and workspace-conditioned directional TRM. At each high-state update the scalar branch adds $gd$ and the directional branch adds $gu(v^\top d)$ to the native proposal, where $d$ is the proposal minus the current high state. The small head uses the current answer and workspace, shares weights across tokens and cycles, bounds the gate by one half, and starts with zero gate and nonzero directions. Native input, loss, low-state updates, early-cycle gradient truncation and detached carry remain in place.

The [runner](../scripts/run_trm_answer_mixing.py) locks one seed, 1,024 optimizer updates and batch size 128. It shuffles the same 33,000 augmented rows. The native control records its actual ACT halt/reset masks and both variants replay them during training, ensuring that the same rows are supervised at every macro and update. The runner verifies this with a digest of effective supervised rows, including repeats. All native weights and the two added heads are trainable. All arms start fresh optimizer state, use native AdamATan2, embedding SignSGD, stablemax and halt supervision, a common 64-step warmup, then the checkpoint recipe's learned-weight and embedding learning rates, and native EMA. The fixed budget is matched by optimizer steps and supervision, with wall time and peak memory reported per arm.

The primary endpoint is the last synchronized single-board readout within one shared time allowance, fixed before training as the archived starting TRM's median full-puzzle time to macro depth 12. Include carry setup, all native steps, head computation, and readout in timed measurements. Rotate arm order by board, report exact results, paired gains/losses and conservative 95% intervals, plus achieved depth and cap counts. The depth-12 and depth-16 exact outcomes are secondary. The development gate for directional mixing is a gain of at least two percentage points and a positive paired lower bound against each control. The 896 held-out inputs remain sealed. If the gate is met, lock a follow-up across seeds before any held-out evaluation.

Isambard staged source revision `2ec2cd9`. Slurm jobs `8914896` (preflight), `8914897` (native), `8914898` (scalar), `8914899` (directional), and `8914900` (paired evaluation) were submitted with success dependencies. The complete CPU suite passed 78 tests, including zero-gate carry and readout parity, the directional formula and fixed-state invariant, nonzero gate gradients under native ACT loss, and full-gradient forward parity after updating the gate.

## 27 November 2026 · Learned native hypotheses pilot

[The pilot](../scripts/run_trm_hypothesis_pilot.py) uses the frozen native TRM at depth 12. Pick the blank with the smallest gap between its two leading digit logits. For each explicit digit hypothesis, a shared learned map reads the fork state at that cell and writes a bounded edit to the corresponding working-state vector. The input and answer state are identical across branches, so immediate logits are invariant. The native macro update is retained. Training differentiates through one full native macro for 256 updates on the same 1,000 original training boards. Correct hypotheses supervise the full board, and both correct and incorrect proposed digits supervise adherence at the chosen cell. The proposal uses only native logits at inference.

On all 256 development boards, compare four serial macros, two clean plus two under the second-ranked hypothesis, and two macros under each of the two explicit hypotheses. Each spends 168 native blocks after the shared prefix. Assess exact candidate coverage, per-board gains and losses, and the existing confidence selector with labels used only after selecting. Report branch disagreement and adherence at the proposed digit, so a lack of candidate improvement can be interpreted alongside whether the network actually used the hypotheses. This is a test of a candidate generator, before introducing a mixer.

Isambard job `8904575` completed in 2m 36s with exit `0:0`. All three policies had exact candidates on 162/256 boards and selected 162 exact boards without labels. Each learned split uniquely solved the same board that native TRM missed through depth 16, and missed the same board that serial solved. Median continuation times were 67.87 ms serial, 68.59 ms clean plus alternate, and 68.98 ms two hypotheses. The serial exact flags matched the archived late probe at every depth. The 896 held-out boards remain sealed.

The two conditioned edits differed after casting on 256/256 boards and final branch predictions differed on 77/256. Of the 98 wrong starts, 69 contained the correct digit among the two proposals at the selected blank. The leading and second-ranked branches returned their proposed digit after two macros on 211/256 and 37/256 boards, respectively. Both proposed digits were wrong on the unique learned solve, and both branches instead returned the same correct board. Median realized edit size was about 0.914 times the original working vector's RMS at the edited cell for each hypothesis. The result establishes a new reachable candidate and weak adherence to the alternative digit. It supports a matched digit-blind edit control before interpreting any benefit as hypothesis-specific or adding a mixer.

Report `native-hypotheses/result.json` SHA-256 `78bc28a84289430d26debceeac489a46fc7627abacb73c9a44c68a5c282184e7`. Edit checkpoint SHA-256 `7e2205b9147ffc0dcdf8b49691580937a6e6460f1861a98b7ed7cdb85e0246da`. The CPU suite passed 75 tests. A tiny CPU fit produced distinct edits from different digits while preserving immediate logits.

## 27 November 2026 · Training-duration and late-branch results

GH200 job `8899223` completed on 26 November in 6m 24s with exit `0:0`. The original archives contained weights only. The [duration runner](../scripts/run_trm_refinement_duration.py) replayed both arms from the original initialization and recovered every archived step-512 weight bitwise. Both arms then continued their live AdamW trajectories to 4,096 total updates. They used the same step-indexed minibatches, 1,000 training fork states, original optimizer and cell loss, 256 development states and four-step inference horizon. Each arm received 65,536 sample presentations. All ten model and optimizer checkpoints were saved and checksum-verified. The separate replay's step-512 metrics and per-board IDs matched exactly at inference steps 0 through 4.

The entries below show mean all-cell cross-entropy and repair/break counts at inference step four. Stopping at the native fork solves 291/1,000 training boards and 60/256 development boards.

| Optimizer updates | Direct train | Direct dev | Recurrent train | Recurrent dev |
|---:|---:|---:|---:|---:|
| 128 | 3.236, 0/0 | 4.154, 0/0 | 1.878, 0/7 | 2.402, 0/3 |
| 512 | 1.142, 0/34 | 1.479, 0/8 | 0.612, 0/37 | 0.830, 0/7 |
| 1,024 | 0.604, 0/41 | 0.881, 0/11 | 0.502, 1/9 | 0.705, 1/3 |
| 2,048 | 0.450, 2/13 | 0.714, 0/6 | 0.453, 1/8 | 0.678, 1/4 |
| 4,096 | 0.398, 1/9 | 0.725, 0/5 | 0.418, 0/3 | 0.707, 1/2 |

Extra optimization reduced the late checkpoints' damage and lowered training cell loss. At 4,096 updates, the direct and recurrent endpoints reached 283 and 288 exact on training and 55 and 59 on development. No inspected four-step checkpoint improved on stopping at the fork on either split. The recurrent development repair at 1,024, 2,048 and 4,096 updates was the same board. The duration test leaves the frozen recipe closed and motivates evaluating mixing only in a controlled comparison that preserves the competent native update. Its bounded training curve does not settle the architecture's ceiling. The cluster report is `refiner-duration/duration.json`, SHA-256 `f2cc4adbca48f4bfe0c589c2fc038e6400590121944b2421364e3be6270be494`; its manifest includes all ten checkpoint hashes.

GH200 job `8899228` completed in 1m 36s with exit `0:0`. The [native probe](../scripts/probe_native_trm_branches.py) reconstructed the shared depth-12 state from the saved depth-two fork, matching all intervening archived exact flags. All 256 development boards ran both policies. The fork solved 158. The shared clean steps through depth 14 covered 161. With 168 native blocks per policy, serial reached 162 and the Gaussian split covered 161. Serial alone solved one extra board; the branch alone solved zero. The per-board policy oracle stayed at 162. Label-free confidence selected 162 and 161 exact boards, respectively. Median continuation times, including selection and perturbation, were 73.67 and 74.71 ms; the prefix was shared and uncharged. The cluster report `native-branches-late.json` has SHA-256 `fbcda57879cb13262cb9ace7d93525813f771d3914648f9b075e28444290a100`.

Together, the early and late probes offer little room for a trigger choosing between these Gaussian split and serial policies. The development target is candidate generation under distinct learned hypotheses, compared with spending the same inference work on native TRM. A state-mixing comparison should isolate the contribution of mixing while keeping native TRM's update available. The 896 held-out boards remain sealed. The CPU suite passed 75 tests.

## 26 November 2026 · Native TRM retention and candidate search

Audited the original 256-board GH200 development calibration, job `8893794`, using [the native branch probe](../scripts/probe_native_trm_branches.py). The archived calibration SHA-256 was `34eee53a18fec002ffe33c2c21083b443f6a615c6bcd2f45dd4485720da7cc09`. At macro depths 1 through 16, the exact counts were 14, 60, 96, 120, 132, 143, 148, 150, 153, 154, 157, 158, 160, 161, 162, and 162. Zero boards transitioned from correct to wrong at adjacent logged macro readouts. At depth 16 all 94 wrong answers were never solved. The 101.21 ms native endpoint agreed with the archived exact flags at each board's observed depth. Of its 109 misses, zero were solved earlier, 15 were first solved later through depth 16, and 94 were never solved. An oracle that retained previously visited exact candidates adds zero at either endpoint. This ceiling applies to logged native macro outputs for this checkpoint and cohort.

The three frozen refiners and all of their trained-depth outputs also have a combined oracle of 60/256, exactly the fork total. This recipe is closed and the 896 held-out boards remain sealed. Candidate generation is the next target on this checkpoint.

The first two-path probe starts from the sealed depth-two development fork. Serial runs four native macros. The two-path policy runs two native macros on the clean state and two on one copy whose working state receives an ID-seeded Gaussian offset with relative RMS 0.10. The immediate logits must remain bitwise equal, the measured post-cast perturbation must be within 5% of its target, and the clean branch must reproduce the serial flags. Both policies generate four candidates plus the shared fork, with equal native block counts. We compare the exact candidate union at equal work, record any solutions outside the entire 16-depth native trace, and score a label-free selector that maximizes mean top-token log-probability on blanks. Timings include perturbation and selection overhead with batch size one and alternating arm order. Oracle coverage is an offline diagnostic.

GH200 job `8899159` completed in 43 seconds with exit `0:0`. At 168 native block evaluations per board, the serial and two-path candidate oracles reached 143/256 and 123/256. The split found one solution absent from the serial candidates and the full 16-depth trace, while serial found 21 absent from the split candidates. The confidence selector scored 143/256 serial and 122/256 split. Median batch-one continuation times were 69.93 and 70.83 ms in this run. All clean trajectories reproduced their archived exact flags. This fixed early split shows an alternative trajectory can reach a new solution, at a net loss of 20 candidate solves for this allocation. Any follow-up should charge a branch against more ordinary continuation and test a label-free stall trigger before allocating work away from the serial path. No training or held-out evaluation was run. The CPU suite passed 75 tests.

Audit SHA-256 `71031b625bd9d62b6183d31a61a3ddaf6e19c8cd6b53b0490a9be5365b445f59`. The cluster report `trm-refinement-K8SH3r/native-branches.json` has SHA-256 `c90999fa2192b93dab4f23f33148b7249f070c74fb871ca576260f6c2873ae01`. The executed script SHA-256 is `2d09d47bccf6760195424a2a6ec89864c700d9d25e77cf7788a6eeaf97744e24`.

## 26 November 2026 · Frozen refiner checkpoint replay

Isambard GH200 job `8898982` completed in 1m 44s with exit `0:0`. The [replay script](../scripts/replay_trm_refinement.py) verified the frozen checkpoints and development bank against the original manifests. It regenerated the native depth-two states for the same 1,000 original training inputs, in the original batch grouping. It used the 256 saved development states and ran no optimizer. The audit report is `trm-refinement-K8SH3r/refiner-replay/replay.json`, SHA-256 `c132b872c085ea9718a175eac2bab4f3a829ecf6a45323f572c1bb26a82a2ec4`. The 896 held-out boards remain unopened.

With the gain set to zero, all three refiners returned the same 60 correct development board IDs as stopping at the fork in the original 101.21 ms production evaluator. Every board received at least one update. No prediction was nonfinite. The direct control reached the 64-step cap on 212 boards, with its answer unchanged. Neither residual control reached the cap.

With learned gains restored, the native fork started at 291/1,000 exact on training and 60/256 on development. Each entry below gives all-cell mean cross-entropy and the counts of repairs/breaks relative to that split's fork. Baseline cross-entropy was 3.727 on training and 4.757 on development.

| Split | Extra steps | Direct (loss, repairs/breaks) | Stateless (loss, repairs/breaks) | Recurrent (loss, repairs/breaks) |
|---|---:|---:|---:|---:|
| Train | 1 | 2.627, 2/0 | 0.771, 0/41 | 0.651, 0/32 |
| Train | 2 | 1.732, 1/0 | 0.692, 0/40 | 0.620, 0/29 |
| Train | 3 | 1.094, 1/9 | 0.668, 0/39 | 0.610, 0/28 |
| Train | 4 | 1.142, 0/34 | 0.670, 0/42 | 0.612, 0/37 |
| Dev | 1 | 3.442, 0/0 | 1.048, 0/11 | 0.896, 0/7 |
| Dev | 2 | 2.354, 0/1 | 0.951, 0/8 | 0.853, 0/6 |
| Dev | 3 | 1.549, 0/4 | 0.913, 0/8 | 0.835, 0/6 |
| Dev | 4 | 1.479, 0/8 | 0.907, 0/10 | 0.830, 0/7 |

Blank-cell cross-entropy also fell for all arms on both splits. The four-step endpoint lost exact solutions even on the examples used to fit the refiners. The direct refiner's two first-step training repairs were transient. The original cell-level objective can improve while exact repair and retention deteriorate. This replay leaves open whether different supervision or optimization would help. The recipe stays paused and the development gate remains closed. The CPU suite passed 75 tests, and a separate CPU replay matched the GH200 development repair/break counts and the zero-gain solved IDs.

## 26 November 2026 · Native TRM refinement development result

Isambard calibration `8893794` and development run `8893949` completed with exit code `0:0` in 2m 08s and 3m 23s. Run `trm-refinement-K8SH3r` used the native TRM checkpoint `533999676a1d44bdb84205461b55ecf1719f238cad2c2af699de78128e3dbf7f` and the refiner code at `43ee675`. Calibration selected depth two with 60 correct and 196 wrong starts on 256 dev boards. The median time for four more native steps was 101.21 ms. Calibration measured 75 repairs and no losses within that budget. The later interleaved timing run remeasured native continuation and observed 87 repairs and no losses.

| Policy | Repairs | Losses | Final exact | Median continuation steps |
|---|---:|---:|---:|---:|
| Stop at depth two | 0 | 0 | 60/256 | 0 |
| Native TRM continuation | 87 | 0 | 147/256 | 5 |
| Direct refiner | 0 | 60 | 0/256 | 57 |
| Stateless refiner | 0 | 60 | 0/256 | 19 |
| Recurrent refiner | 0 | 60 | 0/256 | 19 |

No evaluation endpoint hit its step cap or returned nonfinite logits. All three learned arms failed the promotion gate. The 896 held-out boards remain unopened. A CPU replay from the checksum-verified development bank found 52, 50 and 53 exact solves for direct, stateless and recurrent refiners after four additional steps, respectively, versus 60 at the fork. Extra steps amplify the degradation, and the learned arms are already losing accuracy within their supervised horizon. This is a development result for the frozen-feature, task-loss-only refiner recipe.

Refiner checkpoint SHA-256 values were `3125c0b2a6f79a676a9131490709bd2a36b0b47710e4f26f373fdb88629ac990` (direct), `33c53d5b96f33636ad7037b934f216ccad7b389ccfd374d1e45c0906bdada649` (stateless), and `b6a1fa4a2d21f9a6f1b5a3fc532e3a06edcaf11eb078591905feafce6451ef54` (recurrent).

## 26 November 2026 · Native TRM refinement calibration

The next bounded comparison starts from genuine Sudoku candidates of the trained native TRM. The [protocol](TRM_REFINEMENT.md) fixes how to choose an informative fork from development data, compare same-budget continuation policies and reserve the 896 untouched test boards. Isambard calibration job `8893794` uses only 256 development puzzles and records full selected carry and logits if the gate passes. Development job `8893949` depends on calibration and runs only if that report selects a fork. The direct, stateless and recurrent refiner fixtures preserve the native fork logits and receive finite training gradients. The CPU repository suite passed 75 tests.

## 26 November 2026 · Sample-matched direct-update Sudoku result

Isambard job `8889450` finished successfully at 16:37 UK time. The direct-update and recurrent arms each completed 65,556 updates and 4,195,584 example presentations using the same step-seeded minibatch stream. The input arrays, dataset identity, seed, architecture configuration and held-out IDs matched. The direct-update arm took 1,630.00 training seconds compared with 6,600.02 seconds for recurrent mixing, a 4.05-fold speedup at equal training exposure. The final direct checkpoint SHA-256 was `34ff7f755ad577a2c679350fee8ebe012382b69e2587277cbe7eb6ee7efea7a8`.

| Arm | Training blank accuracy at depth four | Held-out blank accuracy at depth four | Held-out median wrong blanks | Held-out exact at depth four | Median depth-four latency |
|---|---:|---:|---:|---:|---:|
| Direct learned update | 49.35% | 46.90% | 30 | 0/128 | 7.82 ms |
| Recurrent residual update | 53.25% | 49.66% | 29 | 1/128 | 22.52 ms |

The CPU audit reproduced the direct arm's GPU exact traces. Neither arm solved any held-out board at the 100 ms or 750 ms endpoint. The direct arm corrected fewer blank cells at equal exposure and cost less to train and run. Both arms were far from solving full boards, including on the training sample. This one-seed comparison contrasts the current recipes, which also differ in field-specific supervision and parameter count.

## 26 November 2026 · Direct-update Sudoku comparison launched

Run `sudoku-direct-z4CC5v` compares the existing `unrestricted` arm to the matched-budget joint residual checkpoint. Both arms use the same byte-identical 33,000 augmented training rows, seed 11, batch size 64, four supervised steps, GH200 model and ordered 128-board held-out cohort. The step-seeded sampler presents the same minibatches in the same order. The direct arm must finish exactly 65,556 updates, matching all 4,195,584 sample presentations of the recurrent arm. The 6,600-second safety cap still applies. If it stops short, the job does not report an exposure-matched evaluation. The direct arm keeps the small model configuration and fixed codebook decoder. It predicts candidate corrections directly and has no learned residual field or field-specific auxiliary losses. The comparison includes those differences and the difference in parameter counts.

Preflight `8889408` checks training, checkpointing and scoring on development boards. Training job `8889450` replaced the time-only job `8889409` before it started. It writes `full/comparison.json` with the direct, recurrent, TRM and FPRM outcomes at the same inference budgets. The reporter verifies dataset hashes, input ordering and GPU model before making paired comparisons. Training time is measured as an outcome of the exposure-matched comparison.

## 26 November 2026 · Residual Sudoku checkpoint audit

The saved matched-budget residual checkpoint (`b1dbdbb7183e303ba2f17e81ac03eeb5cd827d8560bbecf890f808721f968181`) scored all 128 known held-out solution codewords correctly through the matched evaluator's shared encoding and scoring functions. Decoded digits matched the revision-pinned original Sudoku CSV. Native target IDs round-tripped to the same residual labels. A local CPU replay reproduced every held-out GPU exact-correctness trace by input ID and reasoning depth. No additional training was used.

| Cohort | Readout | Clues correct | Blanks correct | Median wrong blanks | Exact boards |
|---|---|---:|---:|---:|---:|
| 128 original training boards | Initialization | 100% | 27.54% | 40 | 0 |
| Training | Depth 1 | 100% | 47.58% | 29 | 0 |
| Training | Depth 4 | 100% | 53.25% | 27 | 3 |
| 128 held-out boards | Initialization | 100% | 26.46% | 41 | 0 |
| Held-out | Depth 1 | 100% | 45.49% | 30 | 0 |
| Held-out | Depth 4 | 100% | 49.66% | 29 | 1 |
| Held-out | At 100 ms | 92.05% | 36.97% | 36 | 0 |
| Held-out | At 750 ms | 92.05% | 37.07% | 36 | 0 |

The timed rows use the last completed depth from the original GPU timing trace for each board. At depth four, the held-out interquartile range was 25–32 wrong blanks and only the solved board had at most one wrong blank. Exactly one board became correct anywhere in the logged rollout. Early corrections improve token accuracy, but the training and held-out error distributions show that the model rarely builds a nearly solved board. The next comparison should use a direct learned proposal with the same small backbone and training data to see how much of the gap persists without the residual-mediated update. The full system result also includes a 3.18-fold difference in realized sample presentations under the common time budget.

## 26 November 2026 · Matched-budget Sudoku result

Native TRM (`8882873`) and the jointly trained residual reasoner (`8882874`) completed run `sudoku-matched-7Lr4Oe` with exit code `0:0`. The trainer records were both about 6,600 seconds. TRM took 17,351 optimizer steps and saw 13,325,568 sample presentations (40.01 GiB peak allocated). The residual model took 65,556 steps and saw 4,195,584 presentations (0.98 GiB peak allocated). The comparison holds the available training time and available training boards fixed. Model size, precision, batch size, objectives and realized exposure differ.

| System | Exact at 30 ms | Exact at 100 ms | Exact at 750 ms | Normal readout |
|---|---:|---:|---:|---:|
| Native TRM | 6/128 | 48/128 | 78/128 | 78/128 at its native stop |
| Joint residual reasoner | 1/128 | 0/128 | 0/128 | 1/128 at depth four |
| Released FPRM | 14/128 | 67/128 | 96/128 | 79/128 reached native stop within the timing cap, all correct |

All three sets of predictions were verified against the same ordered held-out IDs. Timings are synchronized eager batch-one measurements. The budget curves carry forward the last prediction when a run has already ended. TRM's median time to its native stop was 379.7 ms. The residual model's median time to its depth-four readout was 22.5 ms. The released FPRM curve has its own training provenance. At the tested training budget, the residual reasoner remains near the exact-solve floor while native TRM learns to solve many boards. Investigate candidate initialization and the learned field before expanding the controller matrix. The system comparison leaves their individual contributions unresolved.

Final checkpoint SHA-256 values were `533999676a1d44bdb84205461b55ecf1719f238cad2c2af699de78128e3dbf7f` for TRM and `b1dbdbb7183e303ba2f17e81ac03eeb5cd827d8560bbecf890f808721f968181` for the residual model.

## 26 November 2026 · Held-out controller comparison and Sudoku benchmark

Isambard run `length4-mixer-74a08dc-mmORam` (job `8882596`) froze the ordinary-descent curriculum continuation and compared controllers from the same depth-four states. The runtime was `74a08dc`. The parent checkpoint SHA-256 was `c54e771fe8c8d36c0583f1c1750bc18368f0a6ff728c2f1312ffe0bf012a0f6d`. The recurrent and stateless checkpoints were `23cc64e5c5970640204a5fee0f1e60c9fe51226ef2bcf2ad397d48543fd45674` and `967d744e1f5bedc5d6a442bc38cba857440786929036193a1b178675610678f4`. The budget was 14.62 ms per batch of 16, calibrated as four warmed reference steps on development inputs. The 512 held-out sequences had 430 correct and 82 wrong at the fork.

| Arm | Repaired | Broke | Final exact |
|---|---:|---:|---:|
| Stop at four | 0 | 0 | 430/512 |
| Ordinary descent | 10 | 19 | 421/512 |
| Direct correction | 5 | 285 | 150/512 |
| Gated correction | 20 | 93 | 357/512 |
| Stateless mixing | 12 | 12 | 430/512 |
| Recurrent mixing | 12 | 10 | 432/512 |
| One-token score search | 5 | 323 | 112/512 |
| Two-token score search | 0 | 178 | 252/512 |

The recurrent mixer helped 11 and harmed zero relative to ordinary descent. It repaired the same 12 inputs as stateless mixing and retained two more correct starts. Relative to stopping, the net gain was 2/512 (+0.39 percentage points, pointwise paired bootstrap interval −1.37 to +2.15 points). The held-out result supports a comparison with ordinary descent on this field. The gain over stopping and the incremental value of memory are uncertain. Further work shifts to whole-system Sudoku capability.

Isambard job `8882647` exercised the released TRM MLP and FPRM checkpoints on the same 32 Sudoku dev boards using synchronized eager batch-one inference. At 100 ms, TRM solved 19/32 and FPRM 20/32. At the 750 ms cap, both solved 25/32. TRM reached its native stop on all 32. FPRM reached its native stop on 20, so the 750 ms table reports its last affordable prediction for the remaining boards. This small checkpoint and timing check uses public weights with separate training provenance. The TRM weight SHA-256 was `20e9dc7ebf83b9b41a8b3f58f5fd94ee3a7eb0b0d245bdeeb14e2f1488d1daaf`. The FPRM weight SHA-256 was `6daec5f499d115beb14e23f3a9cf56d1166b99c1ccd36b185a19ea5dfec9a137` with upstream source `b291252387b2656d7a7bc2719d9317e6d812b283`.

The matched training comparison uses 1,000 common original training puzzles and the same 32 seeded upstream Sudoku augmentations per puzzle, giving 33,000 training rows. Both input encodings round-trip to the same puzzle and label arrays. It uses 512 dev and 1,024 separate held-out boards, with the first 128 held-out boards fixed for the short run's timing curves. The approved maximum is two GH200-hours per model. The plan allocates 110 minutes of training and a 115-minute Slurm cap to each arm, plus a shared preflight capped at ten minutes. TRM retains its native objective, EMA and BF16 path. The residual reasoner uses joint recurrent-mixer training in FP32. This is a comparison of the two systems under an equal time budget.

The matched run root is `sudoku-matched-7Lr4Oe` with cohort manifest SHA-256 `2ac636f1f6834cc7eedc5ef7ee2f296add953ee188fee085fd154ac1c96874ac`. Initial preflight `8882801` stopped after 55 seconds because the optimizer's CUDA build inherited an unwritable temporary path. Retry `8882839` compiled the same upstream optimizer in the container's `/tmp` and exposed a missing dataset-loader dependency. Preflight `8882872` added that dependency and passed training, checkpoint and evaluation checks for both arms. Native TRM `8882873` and recurrent residual `8882874` started at 09:49 UK time. External FPRM job `8882879` evaluated the identical 128 held-out input IDs. It solved 67/128 by 100 ms and 96/128 by 750 ms. It reached native termination on 79 boards within the cap.

## 26 November 2026 · Prefix-sum continuation and reasoning depth

All three tasks in Isambard array `8870010` completed in 5m 19s from first start to final exit. At runtime revision `74a08dc8e1bdf9dd4c10531944c7e8537c11fa38`, each branch restored the weights, AdamW state and step counter at update 3,072, then took 1,024 more updates on the same length-16 training set. The two curriculum branches began at the same checkpoint and supervised either four or eight reasoning steps. The direct branch continued with four. Dataset fingerprints and dev input IDs matched the prior study. The continuation runner reproduced uninterrupted training bitwise in a short CPU check, and 72 repository tests passed.

At inference depth four, length-16 token accuracy was 29.99% for direct training with four supervised steps, 39.77% for the curriculum with four, and 37.71% for the curriculum with eight. At inference depth eight the two curriculum branches reached 39.71% and 37.78%, respectively. All branches still solved 0/512 length-16 dev sequences at every recorded depth. The curriculum continuations took 75.36 s and 114.66 s in the training loop, respectively. The eight-step branch consumed about 52% more training time at the same update count.

The four-step curriculum continuation created a usable short-task reference. On the identical 512 length-4 dev sequences, 451 were correct at depth four. Continuing to depth 16 retained 390/451 and repaired 11/61, for 401/512 exact at depth 16. At length eight, only 12/512 were initially correct at depth four. The next bounded comparison should freeze the length-4 reference at a depth-four fork and give the mixer and cheaper controls identical starts and a common measured continuation budget. Use the development observations to design the protocol, and preserve a separately evaluated held-out cohort for confirmation.

Run `prefix-depth-74a08dc-2CB2yr` ended at checkpoint SHA-256 values `4f39b47e2ecf7623235ed42ada9ba78296efe291d54355266bee3ccc185bc33d` (direct, four), `c54e771fe8c8d36c0583f1c1750bc18368f0a6ff728c2f1312ffe0bf012a0f6d` (curriculum, four), and `df4ca798b49090a8898d863c28e9628c582816331d8899285d192ae3b297e7b0` (curriculum, eight).

## 25 November 2026 · Matched-update prefix-sum curriculum

Isambard job `8857953` completed in 8m 33s at revision `74a08dc8e1bdf9dd4c10531944c7e8537c11fa38`. On one GH200, the ordinary-descent reference trained for 3,072 updates either entirely at length 16 or in three equal phases at lengths 4, 8 and 16. Both arms used seed 11, batch 64, four supervised reasoning loops, the pilot objective and the same 512 length-16 development examples. The curriculum retained its weights and AdamW state between phases. The match is in updates. Sequence length changes the number of training tokens per update.

At four reasoning steps, length-16 dev token accuracy was 27.40% for direct training and 35.35% for the curriculum. Position four improved from 103/512 to 410/512 correct. Both arms had zero exact length-16 solves at all recorded depths. At depth 16, token accuracy fell to 23.86% and 24.60% respectively. Training took 223.39 s and 220.66 s, with peak PyTorch allocation of 0.165 and 0.167 GiB. This is one-seed development evidence of improved partial predictions with the chosen schedule. Extra inference steps degraded token accuracy after the four training loops.

The [diagnostic runner](../scripts/run_prefix_curriculum.py) used SHA-256 `b2d42efd87ba37620a915196d9015d3c8594e48a5c9c7e11614cb9b47b203d81`. Its direct arm reproduced the pinned trainer's weights and optimizer state bitwise in a three-update CPU check. Final checkpoint SHA-256 values were `f037a8d6a51ed071ccf5ddadf4f162c2e3563ce9e999a61a73c8d34e3f7e14bc` for direct training and `3788263e160d38e603e8728e30b802cf3f9ae7fda852c6b16bf8232e90f9b446` for the curriculum. The next bounded comparison should test additional length-16 training and vary the supervised unroll depth separately. The multi-seed controller comparison is still pending.

## 25 November 2026 · Development pilot

Isambard run `pilot-74a08dc-ZCDQLY` used revision `74a08dc8e1bdf9dd4c10531944c7e8537c11fa38`, seed 11, and the acceptance image. Final report job `8853665` finished 6m 56s after the first reference jobs started. The sealed run produced 16 model records and 20 development evaluations with complete coverage. The recurrent joint checkpoints were `82528e1e6c90ba32e837b2311c3ff44e1ef4f388ff3bb43eccf092e222b33642` (Sudoku) and `0f9753483e8f038a9909909ff5083bfb73cf94afafdea95e6168e01287ea924f` (prefix sums). All checkpoint hashes are in the archived `reports/training.csv`.

Every arm had zero exact solves on its 512-example dev cohort, both under the common wall budget and at every recorded depth through 16. The 24,576 input-by-arm-and-fork records contained zero exact hits, nonfinite trajectories, or wall-cap hits. The wall-budget recurrent mixer averaged about two steps at fork 4, compared with four for ordinary descent and about eight for direct correction. The correct-start cohorts were empty. Paired comparisons and memory interventions also had zero exact repair events, so this run cannot distinguish their effectiveness. Joint recurrent training used 32.0 s and 0.979 GiB peak PyTorch allocation for Sudoku and 26.1 s and 0.195 GiB for prefix sums, each across 250 updates.

Replaying the sealed checkpoints locally on the original dev inputs at depth four showed 13.71% accuracy on blank Sudoku cells for the jointly trained recurrent model, compared with 11.09% for predicting the most common training blank digit while copying clues. For prefix sums the corresponding accuracies were 18.59% and 18.13% for copying the first token and guessing zero elsewhere. Token metrics are now recorded at every evaluation policy endpoint, with a separate editable-token denominator. Rebuilding the archived report preserved all 384 original metric rows and the paired results.

A local CPU follow-up trained only the ordinary-descent reference from scratch on modular prefix sums at lengths 4, 8 and 16. Each run used the pilot architecture and objective, 2,048 training examples, 512 dev examples, data seed 2809, training seed 11, and 1,024 updates at batch 64. After four reasoning steps the exact counts were 175/512, 0/512 and 0/512, respectively. Token accuracies were 80.91%, 38.65% and 21.41%. The length-8 run was added after the length-4 result. These are separate fixed-length runs and provide no length-transfer estimate. The next bounded test is to compare a short-to-long curriculum against direct length-16 training at a matched update budget and measure per-position accuracy and exact solves before expanding the controller comparison.

## 25 November 2026 · Single-GH200 acceptance

The Isambard-AI acceptance passed at `74a08dc8e1bdf9dd4c10531944c7e8537c11fa38`. Image preparation job `8851688` completed in 7m 05s. Acceptance job `8851773` completed in 48s with exit code `0:0`. Its reports cover 18 model records and 66 split-level evaluations with zero missing records.

The synthetic FP32 recurrent update used a batch of 64, 81 Sudoku positions and four supervised loops on an NVIDIA GH200 120GB. It took 1.006 s and reached 1,048,961,024 bytes (0.977 GiB) of peak PyTorch-allocated GPU memory. The image SHA-256 was `b5524840a75f6d6dba8110f8e2b48389bb9baf841a0bd3147cb7904d12832b4a`.

The acceptance used synthetic inputs and establishes hardware and workflow readiness. Multi-GPU operation remains unchecked. The pinned Sudoku subset and modular prefix-sum data are prepared and staged on Isambard. Checks passed for split sizes, exact input-ID disjointness and transferred file hashes. The one-seed, dev-only development pilot plan sealed locally against these data. Its measured training and evaluation costs will inform the multi-seed study budget.

## 24 November 2026 · Training branch implementation

Implemented notebook Sections 2, 3 and 5 in `experiments/residual_reasoning/`. CPU verification covers the full higher-order training path, frozen-field transfer, shared-state repair, all solver families, Sudoku fixtures, exact interrupted-run replay and reports over a miniature campaign. Codex setup and single-node Slurm/Apptainer launchers are in the [runbook](RESIDUAL_TRAINING.md). The repository suite passed 72 checks locally, including 18 model records and 66 split-level evaluations in the complete fixture campaign. The Slurm submission dependencies were exercised with a local scheduler substitute. GH200 execution, NCCL, production data preparation, pilot timing and scientific outcomes remain pending. The November 20 notebook records the earlier design snapshot.

## 20 November 2026 · CRAM checkpoint audit and notebook v3.1

The experiment table and decision rules in Section 5 now include the score-only search comparison directly. They specify how to choose the repair cohort, measure retention, and interpret a tie at equal cost. This is an editorial clarification of the proposed study.

We checked the three published CRAM Sudoku checkpoints at upstream revision 62bbe830. For every full board with clues clamped, exact energy is at most zero for valid completions and at least 0.2671, 0.2963 and 0.2251 for invalid boards respectively (lower bounds rounded down). An integer-arithmetic audit and an independent exact-rational calculation agreed. The authors’ fitted thresholds lie in this gap. This establishes recognition under exact discrete scoring. The search can still exhaust its budget. The Maze-Unique result gives a complementary case where initialization matters despite perfect test solving.

Three design critiques and a fourth proof check changed the proposed study. Use untouched original-input rollout errors for paired repair, plus ordinary-start solve rate and correct-state retention. Include learned-score search with one- and two-token moves and strong cheap neural correctors. Compare balanced corruptions with equal-distance training controls. Test workspace recomputation only if persistent memory earns its extra cost. No v3 learner or benchmark was run. [Notebook §5](notebook/project_notebook.pdf) · [Full review, proof and reproduction](CRAM_REVIEW.md)

## Current direction — 16 November 2026, notebook v3.0

The main proposal is **learned latent residual coordination**, without symbolic Sudoku/SAT inference rules. The residual is split into geometric channels `g(q,x)` and a learned signed invalidity score `s(q,x)`. A bounded memory-conditioned SPD matrix mixes them before the update `q_next = q - alpha * J_c^T W c`, with `c=[g;s]`.

The concrete learning candidate combines local calibration near fixed generic answer embeddings, a signed invalidity margin on known incorrect candidates, and task loss through the actual recurrence. The signed margin replaces squared-energy ranking because it has nonzero output gradient at a wrong residual zero. It does not guarantee correct zeros, useful derivatives, or successful training. Use `s=sqrt(S)*s_bar` and documented per-position loss normalization for variable output length; this is a scale convention, not a transfer guarantee.

The decisive rivals include direct `q-g` correction, an invalidity-gated learned proposal `q-sH`, and a direct learned residual map `q-Gc`, alongside random/stateless mixing and established solvers. First compare controllers on a shared frozen learned field, then jointly trained systems whose competitors can adapt their own fields. Hardcoded residuals are optional controls. The attracting semantic-manifold branch remains separate from this anchor–workspace construction.

Read [the cleaned notebook](notebook/project_notebook.pdf), especially Sections 2–3 (architecture and learning) and Section 5 (study). [Version 2.1](notebook/archive/project_notebook_v2_1.pdf) preserves the full previous record. Only the mixing/VJP operator exists; the new residual heads, losses, controller and full benchmark are **unimplemented and untrained**. No new empirical gain or publication endorsement is claimed.

## Documentation and critique update — 16 November 2026

Completed one further adversarial design cycle and rewrote the guide around the current architecture, learning objective, mathematical limits and decisive study. Archived the prior 60-page PDF/source byte-for-byte. Retained the attracting-answer-set explanation in the new appendix and the stable E0–E5 ledger. This was a specification/documentation update, not an experiment or new training implementation.

The cycle rejected global denoising regression as a default (an ideal direct corrector can solve it immediately), and replaced squared-energy contrastive negatives with a signed learned invalidity margin. A wrong candidate's zero score receives nonzero output-level gradient, without teaching the globally correct direction. The critic required cheaper verifier-gated/direct correction baselines, explicit length scaling, and a distinction between full-column-rank anchor geometry and full-row-rank semantic manifolds. Practical usefulness and novelty remain open.

Validation: source/PDF rebuilt with XeLaTeX; references, local documentation links, archived-file identity, whitespace and rendered layout checked. No training or new benchmark run was launched; the previously reported eight operator tests remain implementation evidence only.

## Historical architecture decision — 12 November 2026 (superseded by v3.0)

The leading candidate is now **solution-preserving residual mixing**: `q_next = q - alpha * J_c(q)^T W(q,w,x)c(q;x)`, with bounded positive-definite, low-rank residual-space mixing. It can move at some wrong stationary points that the previous state-side preconditioner and diagonal constraint weights cannot leave. Every common residual zero remains fixed. This is a distinction between update classes, not a claim of a new general optimizer: Sketched Newton–Raphson is a direct predecessor, and random residual mixing is a mandatory control.

The first full architecture is a grounded Sudoku factor-graph reasoner: Boolean, cell, row, column, box and clue residuals define valid solutions; a bounded per-factor workspace predicts their low-rank interactions. The controller is learned, the initial task residuals are specified. Generic learned latent residuals remain a later branch. No complete architecture has been trained or benchmarked. A tested PyTorch operator prototype is in `experiments/residual_reasoning/operators.py`.

Start with Section 1 of [notebook version 2.1](notebook/project_notebook.pdf), or its [LaTeX source](notebook/project_notebook.tex), for four adversarial critique rounds, the supplied QRR review assessment, exact proofs, limitations and the decisive comparison. The prior energy-preconditioner is retained as a baseline. E4/E5 remain optional historical diagnostics and do not gate architectural work.

## Initial implementation — 2026-11-04

Official TRM history and MIT licence are preserved at base `c01103738605ba39d1430519b1ee0c62f4c707f8`. Upstream models, layers, training, and dataset builders are unchanged.

Implemented: fixed-phase rollouts; full-gradient adapter; strict checkpoint loading; unaugmented Sudoku reader and deterministic dev/test split; antithetic random and optional future-gradient `z_L` interventions; immediate-logit invariance assertions; post-cast budgets; paired harm/help, retention/recovery and survival metrics; JSONL and hashed manifests; independent summary recomputation; analytic transient-growth toy.

Verified with Python 3.12.14 and torch 2.7.0+cpu:

- `python -m pytest -q`: **20 passed**.
- Exact native/full-gradient forward agreement for MLP and attention at H=3, L=6, two layers, across multiple macro-steps.
- Multi-macro finite-difference agreement, immediate-readout invariance, zero-epsilon trajectory identity, and no reset after normal halt depth.
- Strict checkpoint wrappers and missing-key rejection; handcrafted metric denominators; masking; verifier checks; deterministic disjoint data selection; BF16 zero-delta detection.
- End-to-end fixture checkpoint/data CLI, cost counts, summary recomputation, and overwrite rejection.
- Standalone documented smoke command completed: 8 macro-steps, forks at 2/4, four continuation steps, all five arms, and two-step gradient probes.

The fixture checkpoint has random weights. These are implementation checks, not benchmark evidence.

## Not established

At the initial implementation, no trained checkpoint evaluation had been executed. The first CPU pilot is recorded below. No GPU benchmark or full training run has been executed. No general empirical gain, new theorem, solution manifold, or publication novelty is claimed. Prospective natural-error prediction, training interventions, pooled uncertainty across repeated forks/seeds, and additional backends remain future work.

## Historical next step — superseded 12 November 2026

The first bounded CPU pilot is complete. Run the fixed [same-state clock study](NEXT_EXPERIMENT.md), implemented after the evidence and close-prior review below. Preserve dense-versus-native output semantics and distinguish schedule intervention from exact InfiLoop reproduction. Do not expand perturbation sweeps or train a selective stabilizer before this gate.

## First public-checkpoint pilot — 2026-11-04

Protocol and commands: [PILOT.md](PILOT.md). Local readable index: `runs/README.md`. Weights, data, environment, and generated outputs remain Git-ignored.

Setup and verified implementation:

- Created `.venv` with Python 3.12.3, torch 2.7.0+cpu and pinned experiment requirements. No GPU is visible; no paid compute or training was launched. Exact installed packages are saved in `runs/environment.txt`.
- Downloaded the third-party `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-mlp` checkpoint at revision `256f32fcbe7123e8bf8c449410773a5ad311dbc5`. Published model source is byte-identical to the pinned upstream `trm.py`; weights load strictly. Source/config agreement does not establish complete training provenance or independently confirm EMA selection.
- Checkpoint SHA256: `20e9dc7ebf83b9b41a8b3f58f5fd94ee3a7eb0b0d245bdeeb14e2f1488d1daaf`; config SHA256: `811e64a1fa2e923e601791339b736bb3bab8bd350778ba116908d7965c021777`.
- Prepared all 422,786 rows of the revision-pinned public evaluation CSV, with unchanged token values/order and no augmentation. Exact deduplication found 422,786 unique inputs; project dev contains 211,393. CSV SHA256: `a2fd52aea23d331d5b4ee723c856236e838a9fb9a70e66f4e0e0cf26c338c6a8`. Only selected dev labels are checked/evaluated. No symmetry or training-contamination audit was performed.
- Added evaluation-only CSV preparation, manifest embedding of downloaded-input provenance, flushed batch progress, and a report command producing Markdown, JSON uncertainty, and per-depth CSV. Reports reject incomplete runs and disagreement with recomputed raw summaries. Upstream model/training files remain unchanged.
- `.venv/bin/python -m pytest -q`: **25 passed**. Includes original scientific tests, CPU checkpoint integration, CSV token/group/provenance checks, paired-bootstrap and common-cohort checks, report mismatch rejection, and overwrite rejection.
- Documented smoke command completed at `runs/smoke-first-pilot`, independently reanalyzed and reported. The analytic toy was saved to `runs/analytic-toy.json`. These are implementation/analytic checks, not trained-model evidence.
- Documentation local-link and new CLI help checks passed; `git diff --check` passed. Git ignore checks cover weights, data, outputs and `.venv`.

Two-puzzle timing checks completed at depth 16: both native BF16 and FP32 solved 2/2, taking 21.4 s and 7.1 s respectively. These overlapping, tiny timing checks are not controlled performance measurements or accuracy reproduction.

The 32-puzzle baseline runs completed:

| Run | Step 16 exact accuracy | Final exact accuracy | Correct → wrong transitions | Wall time |
| --- | --- | --- | --- | --- |
| `pilot-native-32` (BF16, 16 steps) | 25/32 (78.125%) | 25/32 (78.125%) | 0/334 at-risk transitions | 461.1 s |
| `pilot-fp32-32-depth64` (FP32, 64 steps) | 25/32 (78.125%) | 27/32 (84.375%) | 0/1,608 at-risk transitions | 542.2 s |

Same dev IDs and batch size in both runs. One puzzle is correct only in native precision at step 16, and a different puzzle only in FP32: equal aggregate accuracy does not mean equal trajectories. The publisher's reported 84.80% uses a different evaluation population; 32 examples do not reproduce or refute it.

FP32 never lost correctness after first becoming correct (0/27 ever-correct puzzles). Relative to step 16, step 64 helps 2/32 and harms 0/32; paired mean change +6.25 percentage points, pointwise 95% puzzle-bootstrap interval [0, 15.625] pp. The two late recoveries are 2/7 of the puzzles still wrong at step 16. The pooled recovery count is 22/408 wrong-state transitions; these transitions are correlated, not independent trials. First-valid rule-verifier coverage is 27/32 with reference accuracy 27/27. Native BF16 continuation beyond step 16 has not been measured.

This sample shows late recovery without the natural retention failures needed to motivate the proposed repair. It does not establish that such failures are absent in the population. The two baseline runs overlapped, so wall times are not controlled speed measurements.

The eight-puzzle zero-epsilon run completed in 372.8 s. All 1,152 branch records exactly match the corresponding natural trajectory in predictions, correctness, verifier results, margins, Q logits, and logged state norms/residuals; every realized delta is zero. Natural solve rate at step 32 is 6/8. This validates the control, not a perturbation benefit.

### Final intervention results and audit — 2026-11-05

`pilot-targeted-8` completed in 520.6 s: eight dev puzzles, FP32, epsilon 0.01, forks 4/8/16, continuation horizon 16, probe horizon 2, clean/random ±/gradient ±. Every comparison has eight puzzles, with five correct and three wrong at each fork.

| Fork | Arm(s) | Final solve | Final harm / help vs clean | Loss given correct at fork | Recovery given wrong at fork | Any prediction difference vs clean |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | Each of random ± and gradient ± | 6/8 | 0/8 / 0/8 | 0/5 | 1/3 | 3/8 |
| 8 | Gradient plus | 5/8 | 1/8 / 0/8 | 0/5 | 0/3 | 3/8 |
| 8 | Each of random ± and gradient minus | 6/8 | 0/8 / 0/8 | 0/5 | 1/3 | 3/8 |
| 16 | Each of random ± and gradient ± | 6/8 | 0/8 / 0/8 | 0/5 | 1/3 | 2/8 |

Rows grouping arms report each arm's count, not pooled independent samples. The step-8 gradient-plus final solve difference is −12.5 percentage points, pointwise 95% puzzle-bootstrap interval [−37.5, 0] pp. Other final solve differences are zero in this sample; degenerate bootstrap intervals do not establish population equivalence. No correct-at-fork puzzle changed its predicted answer relative to clean at any observed branch offset.

Post-hoc case inspection: puzzle `e524867ce8ff8ad0f9697194a08d65e3d935a8a3ff432cba98eac9c83585364b` is wrong at all three forks and naturally first solves at step 17. Gradient plus at fork 4 solves at step 7; at fork 8 it remains wrong through the branch endpoint at step 24; at fork 16 it solves at step 17. Random directions and gradient minus sometimes accelerate this same recovery as well. This is an observed effect on **recovery within a finite window**, not destruction of an already correct answer, proof of permanent failure, a gradient-specific mechanism, or a replicated stage effect. The probes use current predictions, never reference labels.

All 96 nonclean sample/fork/arm perturbations have nonzero realized deltas and pass the norm tolerance. Actual relative norms range from `0.009999998845160007` to `0.010000001639127731`. No examples are excluded, so full and common norm-qualified cohorts coincide. All immediate logits are preserved exactly. The targeted run charges 256 natural, 1,920 continuation, and 48 probe forward sample-macros, plus 6 backward batch calls / 24 sample objectives.

Executed final checks:

- All four primary manifests report `complete`. Each summary was independently recomputed with `analyze`; reports also check equality against raw-record recomputation.
- `PYTHONPATH=. .venv/bin/python runs/audit_pilot.py` passed: common/nested puzzle IDs, identical checkpoint, exact zero-control observables, clean branches matching natural paths, and natural prefixes matching across FP32 runs. Evidence and artifact hashes are in `runs/pilot-audit.json`.
- BF16 and FP32 step-16 decoded predictions match on 24/32 puzzles. Correctness differs on two; wrong answers differ on additional puzzles. Of the 25 correct at FP32 step 16, 0/25 lose correctness through step 64; 2/7 remaining wrong puzzles recover.
- The final scientific/report suite passed **25 tests**, including a transient-effect example whose final predictions rejoin clean. All output directories have Markdown reports, machine-readable JSON, and accuracy CSVs. Sum of the four run wall times is 1,896.8 s, with overlapping runs; this is not total exclusive CPU time or a speed benchmark.

Decision: the narrow observation that current-output-preserving changes can affect later answers is supported by this small checkpoint diagnostic. A stage pattern appears in one post-hoc recovery case and needs replication and margin controls. The retention-repair motivation remains unsupported by this sample: natural continuation recovered answers and never lost a correct one. Increase diagnostic coverage before designing or training a selective stabilizer; do not increase attack strength merely to manufacture correct-answer failures.

### Comparison with InfiLoop Figure 1 — 2026-11-05

Checked [InfiLoop v2, Section 2 and Figure 1](https://arxiv.org/html/2610.11570v2#S2), including the PDF page-3 figure/caption. It reports 19.7% solved-puzzle loss after 36 further outer updates for its self-attention TRM. The aligned cohort contains puzzles both models solve within 36 updates; trajectories span 72 outer updates, with 26 layer evaluations per update. The separately reported 20.9% concerns initially correct **cells**, not solved puzzles.

Our pilot instead uses third-party MLP weights, L=6, and logs after each H=3 macro-step (42 block-layer evaluations), with FP32 continuation only. The 64 logged steps execute 2,688 block-layer evaluations. Thus insufficient total layer count is not an established explanation; architecture, checkpoint, precision, recurrence schedule, observation frequency, and cohort differ. Intermediate losses followed by recovery between our observation points could be missed. Of our 27 ever-correct puzzles, 26 first solve by macro-step 28 and have at least 36 further logged steps; this does not equate their cohort or update definition to the paper's.

The [checked code release](https://github.com/pixeli99/InfiLoop/tree/63716a9c46723d3592c8ad4fc5819855f22aaa47) documents an InfiLoop checkpoint, but its inspected tree/READMEs did not identify the Figure-1 TRM baseline checkpoint or figure-generation script. Exact Figure-1 precision and implementation details remain unverified. No reproduction or new model run was performed in this comparison.

Revised interpretation: the pilot's absence of observed losses is specific to its MLP configuration and observation grid. It should not weaken the published motivation without a matched diagnostic. Next compare attention weights, preserve full state, log every outer update, and measure both pointwise correctness and first-loss survival over equal post-solving windows. Reproduce the paper's joint-model cohort only with both trajectories available; otherwise label a TRM-only cohort explicitly. Keep original and altered recurrence schedules distinct and record both. Do not explain the discrepancy solely by sample size or solve it by strengthening attacks.

Checks: inspected paper Section 2/Figure 1/compute appendix and pinned repository tree/READMEs; verified local manifest configuration, executed-layer count, and first-correct follow-up counts from saved results. Documentation consistency checks passed; no experiment/model code changed.

## Next-study implementation and critic review — 2026-11-05

Reviewed the pushed result summaries at commit `98689fb58da8bcb2791820f5674bd13d77ffc61b`. The previous raw runs and MLP/data assets were not present in this workspace; their reported outcomes were not independently recomputed in this turn. The evidence still favors recovery sensitivity over a demonstrated retention defect.

The literature review identified [When Recursive Models Finish Computing](https://arxiv.org/abs/2609.26487v1), a direct overlap that includes the pilot's MLP checkpoint, and [TAPS](https://arxiv.org/abs/2609.36653), relevant to any future inference controller. Added them prominently to [RESOURCES.md](RESOURCES.md). The original broad diagnostic cannot carry the novelty claim by itself.

Implemented the [fixed next study](NEXT_EXPERIMENT.md): dense native observation plus a causal continuation-schedule fork from the identical full state, equal block budgets, separate pre-fork retention/recovery cohorts, explicit native macro retention, exact paired binary bounds, and independent raw-record report verification. The primary condition is attention BF16; three secondary checkpoint/precision conditions share the same 1,024 new dev IDs. The 2 pp decision threshold and informative-null stopping rules are fixed in the runbook. No new training loss or larger perturbation sweep was added.

Added a hash-pinned attention downloader and a four-condition launcher with frozen-asset, architecture, strict-loading, device and output checks. Downloaded attention revision `33991b882368f425916d625a9ee614a7a85efe76` locally; its published model source is byte-identical to pinned upstream. Both BF16 and FP32 load strictly, and one native macro on two synthetic inputs agrees bitwise with three dense outer updates in both state tensors and logits. These checks use the real weights but are **compatibility checks, not benchmark measurements**. Asset hashes are in the next-study runbook; training/EMA provenance is not independently verified.

Executed checks:

- `.venv/bin/python -m pytest -q`: **35 passed** on Python 3.12.14 / torch 2.7.0+cpu. Covers original tests plus native/dense parity for MLP/attention and both precisions, exact equal-cost arithmetic, pre-treatment cohorts, fixed first-solve follow-up, dense-versus-macro error cases, intermediate endpoint phases, nondegenerate zero-event uncertainty, complete CLI accounting, report recomputation and corruption rejection.
- Pinned attention fetch helper verified all three assets; downloaded Python was not executed.
- Launcher shell syntax, new CLI help, documentation local links and `git diff --check` passed.
- Upstream model, training and dataset-builder source remains unchanged. Full raw datasets, weights and generated outputs remain ignored.

The critic reviewed the proposal, then the implementation and final protocol. It found no remaining scientific blocker after corrections to labeling, phase reporting, uncertainty and provenance checks; [CRITIC_REVIEW.md](CRITIC_REVIEW.md) records the objections. Its approval is for a consequential falsifiable experiment, not a novelty or publication guarantee.

**Not run:** the four-condition trained-checkpoint study. This host has no CUDA device; only implementation and compatibility checks were executed. The next action is the documented full run on suitable compute, followed by the prespecified decision gate. No scientific outcome for the new study is claimed.

## Further research and analytic deductions — 2026-11-06

The reviewed clock-study implementation was published as `2ccf2814a70a929cd3f91b6073513fda3b4c7842`; its [GitHub checks passed](https://github.com/NiccoGrillo/TinyRecursiveModels/actions/runs/42145692270). The user requested continued work with the critic on new insights. [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md) records the resulting narrow hypothesis, proofs, limitations and conditional experiment.

Derived the fixed-set inclusion `Fix(B A^L) ⊆ Fix(B A^(mL))` for TRM's alternating low/high update structure. An exact common equilibrium cannot be escaped by scheduling unchanged updates alone. Constructed an analytic example respecting shared additive input injection where the same equilibrium is attracting at L6, unstable at L12, and stabilized by a predicted high-update damping range. At equal low/high call counts, alternating 3/9 reduces the spectral radius of the linearized period map relative to 6/6; reversed order has the same eigenvalues but different transient gain.

These are elementary mathematical deductions and an **abstract scalar example**, not a literal Transformer weight construction, trained-model evidence, a new numerical-analysis theorem or an established reasoning improvement. The critic caught the additive-injection constraint in an earlier candidate and required the corrected construction and the phase-shift caveat. Generic allocation, adaptive scaling, invariant-measure readouts and feedback steering overlap existing work; the resource map now includes FPRM Appendix F, ImR, RoFB, ICS and classical coupled-solver analysis. ICS's primary abstract was retrieved, but its detailed claims still need full review.

Added executable `coupled_toy.py` and independent finite-difference checks through its nonlinear recurrence. `.venv/bin/python -m pytest -q`: **39 passed**. The toy command reproduced the documented spectra, equal-count comparison and damping threshold. Local documentation links and whitespace checks pass. This is analytic/implementation verification, not another small benchmark run. The original clock-study settings and decision gates are unchanged; no speculative schedule or damping method has been applied to the trained checkpoints.

## Living LaTeX project guide — 2026-11-06

Added [the project guide PDF](notebook/project_notebook.pdf) and its [self-contained editable LaTeX source](notebook/project_notebook.tex), with [build/update instructions](notebook/README.md). It is now the README's first reading recommendation. `AGENTS.md` instructs future work to update the guide and rebuild the PDF whenever results, theory or research decisions change.

The 30-page guide explains the actual two-state TRM computation, reconciles macro/outer/block units, traces completed pilot runs and their limitations, distinguishes recovery from retention with the recorded puzzle case, and introduces the coupled-feedback hypothesis before its derivations. The original transient example and conditional margin-bound idea remain documented. The frozen unrun clock study, conditional equal-count follow-up, prior-work boundaries, exact provenance, commands, glossary and update template are included. Mathematical proofs and implementation checks are explicitly separated from trained-model evidence. The empirical tables reproduce the committed reports; no fresh raw-data reanalysis or model experiment was performed in preparing this document.

Validation: XeLaTeX build completed with all references/citations resolved and no overflow warnings; all 30 rendered pages were visually inspected. Repository links in the LaTeX source resolve, shell syntax and `git diff --check` pass. Experiment code and the frozen study are unchanged; the prior 39-test result is reported as prior validation, not a new test run.

## Notebook v2 and architectural pivot — 2026-11-12

Completely reorganized the living LaTeX document around the persistent-workspace energy architecture. Added definitions, bounded metric construction, stationary-fiber and common-descent proofs, PL versus state convergence, decoder margins, concrete training and cost accounting, critic corrections to the QRR suggestions, and an expanded attracting-answer-set appendix. Preserved all detailed historical pilot results, TRM feedback calculations, E4/E5 protocols, uncertainty and provenance.

The critic audited the mathematics and novelty boundaries. The clock study is now optional, not a prerequisite; this is an explicit user-directed research pivot, not an empirical conclusion. New architecture implementation/training and the larger clock benchmark remain unexecuted. Validation for this documentation change: three-pass XeLaTeX compilation, reference/overflow checks, PDF page rendering and visual inspection, and repository diff/link consistency checks. No model tests or benchmarks were claimed or needed for this documentation-only update.

## Residual-mixing research session — 2026-11-12

Completed four adversarial architecture-review rounds and read the user-supplied QRR reviews. Replaced the sole-leading state-preconditioner proposal with a residual-space cross-factor controller hypothesis. Added exact class-separation example, common-root-family quantifiers, conditional local point/manifold attraction, Gaussian random-sketch control, direct SNR overlap and fixed-basis absorption objection to notebook v2.1. The old architecture and all historical data remain available.

Specified a first grounded Sudoku architecture: explicit Boolean/cell/unit/clue residuals and a learned bounded factor controller. Common residual zeros correspond exactly to valid completions and have full-column-rank Jacobians. Optional 3SAT transfer is witness finding. Full controller and task residual modules are not implemented; no training, GPU run or new accuracy result occurred.

Implemented isolated `experiments/residual_reasoning/operators.py` and focused tests in `tests/residual_reasoning/test_operators.py`. These check the bounded low-rank operation, exact wrong-minimum counterexample, zero preservation, J=0 limitation, and outer automatic derivatives through both residual and mixing parameters. Focused validation and document build results are recorded after final checks below. No existing TRM code or historical experiment protocol changed.

Final focused validation: **8 operator tests passed** on CPU, including rejection of disconnected residual graphs and zero outer gradients for graph-connected constant residuals. These are implementation checks only. Notebook v2.1 was compiled with three XeLaTeX passes, checked for references/overflows and rendered for visual review; repository diff/local-link checks completed.
