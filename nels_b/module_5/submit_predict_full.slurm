#!/bin/bash
# Generate full-holdout prediction CSVs for run2 and the 3-model ensemble.
# Writes:
#   NorthWind_smallcnn100-random-Nels-module5-holdout-predictions.csv
#   NorthWind_ensemble3model-Nels-module5-holdout-predictions.csv
#
#SBATCH --job-name=predict_full
#SBATCH --account=rls62
#SBATCH --qos=gpu
#SBATCH --partition=m9g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source activate.sh

nvidia-smi --query-gpu=name --format=csv,noheader

srun python3 - <<'PYEOF'
import csv, sys, torch, glob
from pathlib import Path
from PIL import Image

sys.path.insert(0, "code")
from train import build_model, CLAHE
from predict import transform_for

device = torch.device("cuda")
images_dir = Path("holdout")
files = sorted(p.name for p in images_dir.glob("*.jpg"))
print(f"[predict] {len(files)} images | device={device}")


def run_model(ckpt_path, files, images_dir, device, batch=512):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    name, img_size, classes = a["model"], a["img_size"], ckpt["classes"]
    stn = a.get("stn", False)
    clahe = CLAHE(a.get("clahe_clip", 2.0), a.get("clahe_tiles", 4)) if a.get("clahe", False) else None
    tf = transform_for(img_size, clahe)
    tag = f"{name}@{img_size}" + ("+stn" if stn else "") + ("+clahe" if clahe else "")
    print(f"[predict] loading {tag} from {ckpt_path}")

    model = build_model(name, len(classes), stn=stn).to(device).eval()
    model.load_state_dict(ckpt["model"])

    probs = []
    with torch.no_grad():
        for i in range(0, len(files), batch):
            chunk = files[i:i+batch]
            x = torch.stack([tf(Image.open(images_dir / f).convert("RGB")) for f in chunk])
            out = model(x.to(device))
            probs.append(torch.softmax(out, dim=1).cpu())
            if (i // batch) % 5 == 0:
                print(f"  {i+len(chunk)}/{len(files)}")
    return torch.cat(probs).numpy(), tag


def write_csv(path, files, preds):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Filename", "ClassId"])
        for fn, p in zip(files, preds):
            w.writerow([fn, int(p)])
    print(f"[predict] wrote {path}")


import numpy as np

# --- run2 alone ---
p2, tag2 = run_model("models/run2/best.pt", files, images_dir, device)
write_csv("NorthWind_smallcnn100-random-Nels-module5-holdout-predictions.csv",
          files, p2.argmax(1))

# --- 3-model ensemble ---
ckpts = [
    "models/run2/best.pt",
    "models/grouped_smallcnn/best.pt",
    "models/stn_clahe/smallcnn_48/best.pt",
]
all_p = []
for ck in ckpts:
    p, tag = run_model(ck, files, images_dir, device)
    all_p.append(p)

ens = np.mean(all_p, axis=0)
write_csv("NorthWind_ensemble3model-Nels-module5-holdout-predictions.csv",
          files, ens.argmax(1))

print("[predict] done")
PYEOF

echo "=== DONE | $(date) ==="
