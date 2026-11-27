# PH1 reproduction

Run from the repository root with Python 3.12 and NumPy 2.3.5. The archived training used float32 CPU computation on an Apple M6. The process peak RSS was 44,990,464 bytes across all six models in one process. Platform, source digests and data digests are in results/summary.json and results/split.json. The reported latencies include Python and array execution for the full learned model. They exclude data creation, which is identical between arms.

```sh
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python experiments/parallel_hypotheses/study.py --out runs/ph1/results --weights runs/ph1/checkpoints
python experiments/parallel_hypotheses/transplant.py --out runs/ph1/results --weights runs/ph1/checkpoints
```

The source reconstructs the 384 training pairs and 96 development pairs in stable row order. The 96 reserve pairs have split membership only. Its labels are never constructed. Each seed/arm received 96,000 supervised query presentations. There is no additional tuning, ACT policy, verifier, learned evidence acquisition, or shared encoder pretraining. Both roles read all legal input features on every example via their own initial projection. Two recurrent updates plus one affine readout are charged identically. Checkpoints are reproducible from the training source and the six archived digests; a file's SHA256 can depend on NumPy's ZIP writer. The separately trained static model uses initial peer states. Its column labelled stale_prediction duplicates that model's prediction.

## Interpretation

The realized development advantage of fresh messages was −1.27 percentage points. The preregistered group-resampling interval covers zero and its upper bound lies below the required five-point effect. This interval describes the observed model seeds and operation vocabulary; repeated seeds expose substantial optimization variation. The six models all fit training perfectly. Formal claims about completely unseen operations, a natural benchmark, or what further compute could yield remain open. The explicit two-lookup executor solves all examples using known table semantics.

Two instances caution against attributing differences in separately trained readouts to test-time communication. For seed 31, ordered permutation IDs F=3,G=1,q=3 yield target 1; the dynamic and clamped dynamic models both predict 1, static predicts 0. At F=14,G=4,q=2, target 0, the dynamic model predicts 1; both clamping and separately trained static recover 0. A transplant at F=13,G=1,q=0 changes the predicted 3 to 0 for active donor F=3 (whose target is 0), while placebo F=12 preserves 3. Messages could convey a competing final answer, since each sender can compute the entire answer. Whole-message transplants cannot distinguish this from carrying an intermediate argument. The matched active and placebo donors also differ in other entries of F. Behavioral dependence on learned communication coexists with no attributable improvement over training with static peer inputs.
