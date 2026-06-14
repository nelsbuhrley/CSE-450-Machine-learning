## Goal
Train a CNN classifier on the GTSRB German traffic-sign dataset (43 classes,
~39k 100x100 JPGs in `training/`) on the BYU cluster, iterating fast with minimal
GPU time. See README.md for the workflow.

## Environment
- Cluster: BYU ORC, SLURM. Account `rls62`. QOS: `test` (1h, fast/cheap, for
  iteration), `gpu` (3-day, full runs). Many GPU partitions (P100..B200) — a
  single modest GPU is plenty for this task.
- Conda env `gtsrb` (miniforge3) with PyTorch cu124. Build: `bash setup_env.sh`.
  Activate: `source activate.sh`.
- Login node has internet + a GPU view via nvidia-smi but DO NOT train on it;
  only run `--smoke --device cpu` checks there.

## Iterate
1. `python code/train.py --smoke --device cpu`  (validate on login, no GPU)
2. `bash dev_gpu.sh` then train interactively  (live tuning)
3. `sbatch submit.slurm <train.py args>`        (full runs)

## Conventions
- `code/train.py` is fully argparse-driven; keep defaults sane for GTSRB.
- `training/`, `models/`, `logs/` are gitignored (data is regenerable, models large).
- Each run writes best.pt/last.pt/metrics.csv/summary.json under `--out`.
- GTSRB has 30 near-identical frames per physical sign (`TRACK_FRAME.jpg`). Default
  `--group-split` keeps a sign's frames on one side of train/val (honest accuracy
  ~97-99%); `--random-split` is the old leaky per-image split (inflates to ~99.9%).
- `code/ensemble.py` soft-votes checkpoints on the identical held-out split.
  `experiment_grouped.slurm` runs train-both-then-ensemble end to end into new dirs.
