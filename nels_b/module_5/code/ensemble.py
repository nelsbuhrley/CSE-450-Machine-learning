#!/usr/bin/env python
"""Evaluate trained checkpoints individually and as a soft-voting ensemble.

Each model is scored on the SAME held-out validation split used during training
(same seed / val-split / group-split), so the numbers are directly comparable and
the ensemble is aligned image-for-image. Each model uses its own --img-size.

  python code/ensemble.py \
      --models models/grouped_smallcnn/best.pt models/grouped_resnet18/best.pt \
      --out models/grouped_ensemble.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CachedTensorDataset, _preload, build_model, eval_transform, make_split


@torch.no_grad()
def model_probs(ckpt_path, data_dir, val_split, seed, group_split, workers, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    name, img_size, classes = a["model"], a["img_size"], ckpt["classes"]
    stn = a.get("stn", False)

    images, labels, _, groups = _preload(data_dir, img_size, workers)
    _, val_idx = make_split(labels.numpy(), groups, val_split, seed, group_split)
    ds = CachedTensorDataset(images, labels, val_idx, eval_transform())
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=workers)

    model = build_model(name, len(classes), stn=stn).to(device).eval()
    model.load_state_dict(ckpt["model"])

    probs, ys = [], []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(out, dim=1).cpu())
        ys.append(y)
    return torch.cat(probs), torch.cat(ys), name, img_size


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parent.parent
    p.add_argument("--models", nargs="+", required=True, help="paths to best.pt checkpoints")
    p.add_argument("--data-dir", default=str(repo_root / "training"))
    p.add_argument("--out", default=str(repo_root / "models" / "ensemble.json"))
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-split", action="store_true", default=True)
    p.add_argument("--random-split", dest="group_split", action="store_false")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    print(f"[ensemble] device={device} | split={'group' if args.group_split else 'per-image'}")

    all_probs, ref_y, results = [], None, []
    for ck in args.models:
        probs, ys, name, img_size = model_probs(
            ck, args.data_dir, args.val_split, args.seed, args.group_split, args.workers, device)
        if ref_y is None:
            ref_y = ys
        elif not torch.equal(ref_y, ys):
            raise RuntimeError("val splits differ across checkpoints — same seed/val-split/group-split required")
        acc = (probs.argmax(1) == ys).float().mean().item()
        results.append({"ckpt": ck, "model": name, "img_size": img_size, "val_acc": acc})
        all_probs.append(probs)
        print(f"[ensemble] {name:9s} @ {img_size}px : val_acc = {acc:.4f}  ({ck})")

    # Soft voting: average class probabilities, then argmax.
    ens = torch.stack(all_probs).mean(0)
    ens_acc = (ens.argmax(1) == ref_y).float().mean().item()
    best_single = max(r["val_acc"] for r in results)
    delta = ens_acc - best_single
    print(f"[ensemble] ENSEMBLE (mean softmax) : val_acc = {ens_acc:.4f}  "
          f"({delta:+.4f} vs best single)  on {len(ref_y)} val images")

    report = {"n_val": int(len(ref_y)), "group_split": args.group_split,
              "models": results, "ensemble_val_acc": ens_acc,
              "best_single_val_acc": best_single, "ensemble_gain": delta}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[ensemble] wrote {args.out}")


if __name__ == "__main__":
    main()
