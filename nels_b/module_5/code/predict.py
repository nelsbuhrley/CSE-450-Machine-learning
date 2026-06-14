#!/usr/bin/env python
"""Score checkpoints + their soft-vote ensemble on a labeled holdout folder.

Reads a flat folder of images plus an answer CSV (Filename,ClassId), runs each
model and the ensemble, and reports accuracy, macro/weighted precision-recall-F1,
and top-k accuracy. Each model uses its own --img-size (read from its checkpoint).
Small enough to run on a login-node CPU (--device cpu).

  python code/predict.py \
      --models models/grouped_smallcnn/best.pt models/grouped_resnet18/best.pt \
      --images holdout/mini_holdout --answers holdout/mini_holdout_answers.csv \
      --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import build_model


def load_answers(path):
    rows = list(csv.DictReader(open(path)))
    files = [r["Filename"] for r in rows]
    y = np.array([int(r["ClassId"]) for r in rows], dtype=np.int64)
    return files, y


def transform_for(img_size):
    return v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5] * 3, [0.5] * 3),
    ])


@torch.no_grad()
def model_probs(ckpt_path, files, images_dir, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    name, img_size, classes = a["model"], a["img_size"], ckpt["classes"]
    stn = a.get("stn", False)
    tf = transform_for(img_size)
    x = torch.stack([tf(Image.open(Path(images_dir) / f).convert("RGB")) for f in files])

    model = build_model(name, len(classes), stn=stn).to(device).eval()
    model.load_state_dict(ckpt["model"])
    probs = []
    for i in range(0, len(x), 256):
        out = model(x[i:i + 256].to(device))
        probs.append(torch.softmax(out, dim=1).cpu())
    tag = f"{name}@{img_size}" + ("+stn" if stn else "")
    return torch.cat(probs).numpy(), tag, img_size


def topk_acc(probs, y, k):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def metrics(probs, y):
    pred = probs.argmax(1)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
        "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(fw),
        "top2_acc": topk_acc(probs, y, 2), "top5_acc": topk_acc(probs, y, 5),
        "errors": int((pred != y).sum()), "n": int(len(y)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parent.parent
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--images", default=str(repo / "holdout" / "mini_holdout"))
    p.add_argument("--answers", default=str(repo / "holdout" / "mini_holdout_answers.csv"))
    p.add_argument("--out", default=str(repo / "models" / "holdout_metrics.json"))
    p.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(dev)

    files, y = load_answers(args.answers)
    print(f"[predict] {len(files)} holdout images | device={device}\n")

    per_model_probs, report = {}, {}
    for ck in args.models:
        probs, tag, img_size = model_probs(ck, files, args.images, device)
        per_model_probs[tag] = probs
        report[tag] = metrics(probs, y)

    ens = np.mean(list(per_model_probs.values()), axis=0)
    report["ensemble"] = metrics(ens, y)

    # ---- table ----
    cols = ["accuracy", "f1_macro", "precision_macro", "recall_macro",
            "f1_weighted", "top2_acc", "top5_acc", "errors"]
    head = f"{'model':<22}" + "".join(f"{c:>12}" for c in cols)
    print(head); print("-" * len(head))
    for k, m in report.items():
        row = f"{k:<22}" + "".join(
            (f"{m[c]:>12d}" if c == "errors" else f"{m[c]:>12.4f}") for c in cols)
        print(row)

    # ---- ensemble error breakdown ----
    epred = ens.argmax(1)
    wrong = np.where(epred != y)[0]
    print(f"\n[predict] ensemble misclassified {len(wrong)}/{len(y)}:")
    for i in wrong:
        print(f"   {files[i]}: true={y[i]:>2d}  pred={epred[i]:>2d}  conf={ens[i, epred[i]]:.3f}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[predict] wrote {args.out}")


if __name__ == "__main__":
    main()
