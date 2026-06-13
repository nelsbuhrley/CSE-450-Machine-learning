# GTSRB Traffic-Sign Classifier — Project Notes & Handoff

This repo trains a CNN to classify German Traffic Sign Recognition Benchmark
(GTSRB) images. It is meant to be cloned onto BYU's "Mary Lou" supercomputer
and trained on a GPU via SLURM.

> **For Claude Code running on the cluster:** read the "Status / what's left"
> section first. The data pipeline is verified; the PyTorch model code has NOT
> yet been run end-to-end (no GPU/torch was available where it was written).

---

## The data

- It is **GTSRB**: 43 sign classes, **39,209 images**, all resized to 100×100
  RGB JPG.
- The images are split across **two sibling folders**, `training1/` and
  `training2/`. This is **NOT** a train/test split — it's just an arbitrary
  split by class id: `training1/` holds classes `00000`–`00020`, `training2/`
  holds `00021`–`00042`. The training script scans **both** and builds one
  43-class dataset. Class label = integer of the parent folder name
  (`00007/` → label 7).
- **Tracks (important):** filenames are `XXXXX_YYYYY.jpg` where `XXXXX` is a
  *track* = ~30 frames of the **same physical sign**. Track ids repeat across
  classes, so the grouping key is `(class, track_id)`. There are **1,307**
  such groups. A naive per-image train/val split leaks near-duplicate frames
  into val and inflates accuracy, so we split by track instead.
- **Class imbalance:** 210 to 2,250 images per class. Handled with
  class-weighted `CrossEntropyLoss`.
- There is **no held-out test set** in the data — we carve a validation set
  out of training via the track-aware split.

The `training1/`, `training2/` folders and the `*.zip` files live OUTSIDE this
repo (in the parent Downloads folder) and are `.gitignore`d. **Do not commit
the data.** Upload it to the cluster separately (see below).

## Files

| file | purpose |
|------|---------|
| `train_gtsrb.py` | The trainer. Scans data, track-aware split, CNN, train loop, checkpointing, per-class accuracy. |
| `submit.slurm`   | SLURM batch script for Mary Lou (GPU). |
| `environment.yml`| conda env (pytorch + torchvision + cuda). |
| `requirements.txt`| pip/venv alternative. |
| `.gitignore`     | keeps data, checkpoints, `runs/` out of git. |

## Design decisions (already implemented in train_gtsrb.py)

- **Architecture:** compact 3-block CNN (`SignCNN`) at 48×48 input — a known
  sweet spot for GTSRB (~98–99% val acc). `--arch resnet18` is also wired up.
- **Augmentation:** rotation ±15°, small affine translate/scale, color jitter.
  **No horizontal flip** — traffic signs are not mirror-symmetric.
- **Loss:** class-weighted CE with `label_smoothing=0.05`.
- **Optim:** AdamW + cosine LR schedule. Mixed precision (AMP) auto-enabled on
  CUDA.
- **Checkpointing:** writes `last.pt` every epoch and `best.pt` on val
  improvement; `--resume` continues from `last.pt` (so `--qos=standby`
  preemptable jobs are safe).
- **Outputs:** `runs/<name>/log.csv` (per-epoch metrics) and
  `per_class_acc.json` (best-epoch per-class accuracy).

## Status / what's left

VERIFIED (unit-tested against all 39,209 real images):
- `scan_dataset` finds 39,209 imgs / 43 classes.
- `track_aware_split` → 32,609 train / 6,600 val (16.8%), **zero** track-group
  leakage between splits, all 43 classes present on both sides, deterministic.
- (Fixed a bug here: an earlier version indexed `track_id[1]` (a single char)
  instead of the whole track id, which dumped every class into val. Fixed.)

NOT yet verified (do this on the cluster):
- A real forward/backward pass of `SignCNN` / `resnet18` (torch couldn't be
  installed in the authoring sandbox). Run the smoke test below first.
- That `environment.yml` resolves against the cluster's CUDA version — adjust
  `pytorch-cuda=12.1` to match `module avail cuda` on Mary Lou.

## Cluster workflow (BYU Mary Lou — rc.byu.edu)

1. **Clone** this repo on a login node.
2. **Upload the data** (NOT via git). From your laptop:
   `rsync -av training1 training2 you@ssh.rc.byu.edu:~/gtsrb_data/`
   so the cluster has `~/gtsrb_data/training1` and `~/gtsrb_data/training2`.
   Keep data on compute/scratch storage, not in `archive`.
3. **Build the environment** (one time):
   `module load miniconda3` (or `mamba`), then
   `conda env create -f environment.yml` → creates env `gtsrb`.
4. **Smoke test** on the short `test` QOS before a real run:
   ```
   salloc --time 00:20:00 --gpus 1 --ntasks 4 --mem 8G --qos test
   module load cuda && conda activate gtsrb
   python train_gtsrb.py --data-root ~/gtsrb_data --limit 2000 --epochs 1 --num-workers 4
   nvidia-smi    # confirm the GPU is actually being used
   ```
5. **Full run:** edit `DATA_ROOT` / env-activation lines in `submit.slurm`,
   then `sbatch submit.slurm`. Monitor: `squeue --me`,
   `tail -f slurm-<jobid>.out`. For live GPU use: ssh to the node and
   `module load nvtop && nvtop`.

### Verified cluster facts (from rc.byu.edu docs, June 2026)
- GPUs available: H200, L40S, H100, A100, V100, P100. Request with
  `--gpus=1` or pin a type, e.g. `--gpus=l40s:1`.
- `module load cuda` for the CUDA runtime.
- `--mem` and `--time` are **required** sbatch flags.
- `test` QOS: ≤1 h walltime, short queue — use for dev/smoke tests.
- `--qos=standby --requeue` = preemptable (more/faster GPUs, may be killed →
  our checkpointing + `--resume` handle this).

## Tuning ideas (later)
- Try `--arch resnet18`, or larger `--img-size 64`.
- Swap class-weighted loss for a `WeightedRandomSampler` if minority recall is
  weak; check `per_class_acc.json`.
- Add a proper held-out test set (the official GTSRB test set) for a final,
  unbiased accuracy number.
