# GTSRB Traffic-Sign Classifier

Train a CNN on the **German Traffic Sign Recognition Benchmark** (43 classes,
~39k images) on the BYU cluster, set up for fast iteration with minimal GPU time.

## Layout
```
training/        43 class folders (00000..00042) of 100x100 JPGs   [gitignored]
code/train.py    configurable training script (smallcnn / resnet18)
code/ensemble.py score checkpoints individually + as a soft-vote ensemble
models/          checkpoints + metrics per run                     [gitignored]
setup_env.sh     one-time conda env build (PyTorch cu124)
activate.sh      source to activate the env (jobs + interactive)
dev_gpu.sh       grab an interactive GPU shell (test QOS)
submit.slurm     batch training job
experiment_grouped.slurm  train both models + ensemble on the honest split
```

## One-time setup (login node)
```bash
bash setup_env.sh          # builds conda env 'gtsrb' (~5 min, needs internet)
```

## The iteration loop (cheap → expensive)

**1. Validate the pipeline on the login node — zero GPU.**
```bash
source activate.sh
python code/train.py --smoke --device cpu      # ~10s, catches code bugs
```

**2. Live-tune on an interactive GPU (fast to schedule, 1h cap).**
```bash
bash dev_gpu.sh                                 # lands you on a GPU node
source activate.sh
python code/train.py --epochs 3 --out models/dev
```

**3. Full runs as batch jobs (queued, checkpointed).**
```bash
sbatch --qos=test --time=00:20:00 submit.slurm --epochs 10 --out models/run1
sbatch --qos=gpu  --time=01:00:00 submit.slurm --model resnet18 --epochs 60 --out models/run2
squeue --me                                     # watch it
tail -f logs/gtsrb-<jobid>.out
```

## Why this is GPU-cheap
- `--cache` (on by default) decodes every image into RAM **once**, so epochs run
  augmentation-only instead of re-reading 39k JPEGs each time — this keeps the GPU
  fed (an uncached run leaves the GPU ~0% utilized, starved by disk/decode).
- `--smoke` finds 90% of bugs on CPU before you ever touch a GPU.
- `test` QOS schedules in seconds and self-limits to 1h, so debug runs barely
  dent your allocation.
- Mixed precision (`--amp`, on by default) ~halves GPU time on real runs.
- Early stopping (`--patience`) ends runs once val accuracy plateaus.
- A SmallCNN reaches ~98% val accuracy in a few minutes on one GPU — no need for
  the big A100/H200 partitions.

## Honest evaluation — no track leakage
GTSRB photographs each physical sign as **30 consecutive frames** (`TRACK_FRAME.jpg`).
A naive per-image split scatters a sign's near-identical frames across train *and*
val, so the model "recognizes" val signs it effectively memorized — inflating
accuracy to ~99.9%. `train.py` defaults to **`--group-split`**, which keeps all
frames of a sign on one side (val = entirely *unseen* signs). Use `--random-split`
to reproduce the old, leaky behavior.

Ensembling two trained models (soft voting = average softmax) on the honest split:
```bash
python code/ensemble.py \
    --models models/grouped_smallcnn/best.pt models/grouped_resnet18/best.pt \
    --out models/grouped_ensemble.json
# or run the whole thing (train both + ensemble) as one job:
sbatch experiment_grouped.slurm
```
Both models are scored on the *same* held-out split, so the comparison and the
ensemble are aligned image-for-image.

## Each run writes to `--out`
- `best.pt` / `last.pt` — checkpoints (model + class names + args)
- `metrics.csv` — per-epoch loss/acc/time
- `summary.json` — best val accuracy and config

## Useful knobs
`--model {smallcnn,resnet18}` · `--img-size` · `--epochs` · `--batch-size`
`--lr` · `--val-split` · `--patience` · `--no-amp` · `--no-cache` · `--random-split` · `--device {auto,cuda,cpu}`
Run `python code/train.py --help` for all of them.
