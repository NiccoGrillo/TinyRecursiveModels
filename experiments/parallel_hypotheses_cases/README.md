# Reproduce the two-world diagnostic

From the checkout root run the following with Python 3.12, NumPy 2.2.6 and PyTorch 2.7.0. Pass a fresh output directory.

    python -m experiments.parallel_hypotheses_cases.study --out runs/ph2-cpu
    python -m unittest experiments.parallel_hypotheses_cases.test_study

The results directory contains per-row predictions, fitted parameters for each seed/arm, timings, training curves, code and split digests, paired intervals, donor counts and membership of all 300 unordered pairs. The program instantiates labels only for training and development. The recovered parameters use the names and shapes in Cases.named_parameters(). Use float32 for prediction parity.

The joint model uses 337 parameters. Its affine computations use 336 multiplications per query (two 4-by-8 symbol embeddings, a 16-by-16 join, and a 16-by-1 gate). Two four-row table reads add 32 multiplies, the two four-dimensional output terms add 8, and the interaction contributes 8. Count softmax, tanh, sigmoid, memory movement and framework overhead separately. Selection adds 16 multiplications for two proposals plus two mixture multiplies, two sigmoids, bias and addition. Both schedules execute two world lookups and one join. Vectorizing reads reduces dispatch in single-thread CPU measurements. No GPU timing was measured.

For an expressly authorized hardware calibration, run from the research checkout on shared storage. Inspect the current queue first, export RR_IMAGE pointing to the verified scratch SIF and RR_ROOT to the new project result directory, and use sbatch experiments/parallel_hypotheses_cases/isambard.sh. This requests one GPU for at most 15 minutes (0.25 requested GPU-hours). The script records the revision, image checksum and Slurm manifest and refuses an existing run directory. Site accounting may differ from requested GPU wall time. This job has not been submitted or GPU-tested. The CPU result gives no scientific reason to rerun these accuracy comparisons on a GPU.
