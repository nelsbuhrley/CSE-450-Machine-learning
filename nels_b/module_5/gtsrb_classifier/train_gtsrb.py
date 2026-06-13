#!/usr/bin/env python3
"""
GTSRB traffic-sign CNN trainer (PyTorch).

Designed to run as a SLURM batch job on a GPU node (e.g. BYU "Mary Lou"),
but also runs on CPU for quick local smoke tests.

Key design choices (why this is set up the way it is):
  * The data lives in two folders (training1, training2) that together hold all
    43 GTSRB classes (00000..00042). This script scans BOTH and builds one
    43-class dataset. The split into two folders is NOT a train/test split.
  * GTSRB images come in "tracks": ~30 frames of the SAME physical sign.
    A random per-image train/val split leaks near-duplicate frames into the
    validation set and badly inflates accuracy. We split by (class, track)
    so all frames of a given sign stay on one side of the split.
  * Classes are imbalanced (210..2250 images). We use class-weighted loss.
  * Signs are NOT left-right symmetric, so we do NOT horizontal-flip augment.
  * Checkpointing + --resume so the job survives preemption (--qos=standby).

Usage (see README.md for the full cluster workflow):
    python train_gtsrb.py --data-root /path/to/folder_with_training1_and_training2
"""

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

try:
    from torchvision import models as tvmodels
except Exception:  # pragma: no cover
    tvmodels = None


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
NUM_CLASSES = 43
FNAME_RE = re.compile(r"(\d+)_(\d+)\.(?:jpg|jpeg|png|ppm)$", re.IGNORECASE)


def scan_dataset(data_root, subdirs):
    """Return (paths, labels, tracks).

    A 'sample' is one image. label = integer of the parent class folder
    (e.g. '00007' -> 7). track = (class_label, track_id_from_filename), used
    only for grouping so frames of one physical sign don't span the split.
    """
    paths, labels, tracks = [], [], []
    for sub in subdirs:
        base = os.path.join(data_root, sub)
        if not os.path.isdir(base):
            continue
        for cls_name in sorted(os.listdir(base)):
            cls_dir = os.path.join(base, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            try:
                label = int(cls_name)
            except ValueError:
                continue
            for fn in sorted(os.listdir(cls_dir)):
                m = FNAME_RE.search(fn)
                if not m:
                    if fn.lower().endswith((".jpg", ".jpeg", ".png", ".ppm")):
                        # unknown name format: give it its own track
                        track_id = fn
                    else:
                        continue
                else:
                    track_id = m.group(1)
                paths.append(os.path.join(cls_dir, fn))
                labels.append(label)
                tracks.append((label, track_id))
    return paths, labels, tracks


def track_aware_split(paths, labels, tracks, val_frac, seed):
    """Stratified-by-class, grouped-by-track train/val split.

    For each class we shuffle that class's tracks and move whole tracks into
    val until ~val_frac of the class's images are held out. This keeps the
    per-class balance of the split while preventing track leakage.
    """
    import random

    rng = random.Random(seed)

    # tracks per class -> list of image indices
    cls_track_idxs = defaultdict(lambda: defaultdict(list))
    for i, (lab, track_id) in enumerate(tracks):
        cls_track_idxs[lab][track_id].append(i)

    train_idx, val_idx = [], []
    for lab, track_map in cls_track_idxs.items():
        track_ids = list(track_map.keys())
        rng.shuffle(track_ids)
        n_total = sum(len(track_map[t]) for t in track_ids)
        n_val_target = int(round(val_frac * n_total))
        n_val = 0
        val_tracks = set()
        for t in track_ids:
            if n_val < n_val_target:
                val_tracks.add(t)
                n_val += len(track_map[t])
        for t in track_ids:
            (val_idx if t in val_tracks else train_idx).extend(track_map[t])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class GTSRBDataset(Dataset):
    def __init__(self, paths, labels, indices, transform):
        self.paths = paths
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        img = Image.open(self.paths[j]).convert("RGB")
        return self.transform(img), self.labels[j]


def build_transforms(img_size):
    # GTSRB-ish normalization; ImageNet stats are fine too.
    mean = [0.3403, 0.3121, 0.3214]
    std = [0.2724, 0.2608, 0.2669]
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # NOTE: deliberately NO RandomHorizontalFlip — signs aren't mirror-symmetric.
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SignCNN(nn.Module):
    """Compact 3-block CNN. Reaches ~98-99% val acc on GTSRB at 48px."""

    def __init__(self, num_classes=NUM_CLASSES, img_size=48):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # /2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # /4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # /8
        )
        feat = (img_size // 8)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((feat, feat)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * feat * feat, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(arch, img_size, device):
    if arch == "cnn":
        model = SignCNN(NUM_CLASSES, img_size)
    elif arch == "resnet18":
        if tvmodels is None:
            raise RuntimeError("torchvision.models unavailable")
        model = tvmodels.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"unknown arch {arch}")
    return model.to(device)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    per_cls_correct = defaultdict(int)
    per_cls_total = defaultdict(int)
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type,
                                enabled=(scaler is not None)):
                out = model(imgs)
                loss = criterion(out, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        pred = out.argmax(1)
        correct += (pred == targets).sum().item()
        total += imgs.size(0)
        if not train:
            for t, p in zip(targets.cpu().tolist(), pred.cpu().tolist()):
                per_cls_total[t] += 1
                per_cls_correct[t] += int(t == p)
    acc = correct / max(total, 1)
    per_cls_acc = {c: per_cls_correct[c] / per_cls_total[c]
                   for c in sorted(per_cls_total)} if not train else None
    return loss_sum / max(total, 1), acc, per_cls_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Folder that CONTAINS training1/ and training2/")
    ap.add_argument("--subdirs", nargs="+", default=["training1", "training2"])
    ap.add_argument("--out-dir", default="runs/gtsrb")
    ap.add_argument("--arch", choices=["cnn", "resnet18"], default="cnn")
    ap.add_argument("--img-size", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from <out-dir>/last.pt if present")
    ap.add_argument("--limit", type=int, default=0,
                    help="Use only N images total (debug/smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[info] device={device}  amp={use_amp}  arch={args.arch}  "
          f"img={args.img_size}  bs={args.batch_size}")

    # ---- data ----
    paths, labels, tracks = scan_dataset(args.data_root, args.subdirs)
    if not paths:
        raise SystemExit(f"No images found under {args.data_root} "
                         f"(subdirs={args.subdirs}). Check the path.")
    if args.limit:
        paths, labels, tracks = paths[:args.limit], labels[:args.limit], tracks[:args.limit]
    print(f"[info] found {len(paths)} images, {len(set(labels))} classes")

    train_idx, val_idx = track_aware_split(paths, labels, tracks,
                                           args.val_frac, args.seed)
    print(f"[info] train={len(train_idx)}  val={len(val_idx)}")

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = GTSRBDataset(paths, labels, train_idx, train_tf)
    val_ds = GTSRBDataset(paths, labels, val_idx, val_tf)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              drop_last=False, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin,
                            persistent_workers=args.num_workers > 0)

    # ---- class-weighted loss for imbalance ----
    counts = torch.zeros(NUM_CLASSES)
    for i in train_idx:
        counts[labels[i]] += 1
    counts = counts.clamp(min=1)
    class_weights = (counts.sum() / (NUM_CLASSES * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # ---- model / optim ----
    model = build_model(args.arch, args.img_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch, best_acc = 0, 0.0
    ckpt_last = os.path.join(args.out_dir, "last.pt")
    ckpt_best = os.path.join(args.out_dir, "best.pt")
    log_csv = os.path.join(args.out_dir, "log.csv")

    if args.resume and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["sched"])
        if scaler and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_acc = ck.get("best_acc", 0.0)
        print(f"[info] resumed from epoch {start_epoch} (best_acc={best_acc:.4f})")
    else:
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                    "val_loss", "val_acc", "lr", "sec"])

    # ---- loop ----
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, device, criterion,
                                       optimizer, scaler)
        va_loss, va_acc, per_cls = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:3d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | lr={lr_now:.2e} | {dt:.0f}s",
              flush=True)
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.5f}", f"{tr_acc:.5f}",
                                    f"{va_loss:.5f}", f"{va_acc:.5f}",
                                    f"{lr_now:.6e}", f"{dt:.1f}"])

        ck = {"model": model.state_dict(), "optim": optimizer.state_dict(),
              "sched": scheduler.state_dict(),
              "scaler": scaler.state_dict() if scaler else None,
              "epoch": epoch, "best_acc": best_acc, "args": vars(args)}
        torch.save(ck, ckpt_last)
        if va_acc > best_acc:
            best_acc = va_acc
            ck["best_acc"] = best_acc
            torch.save(ck, ckpt_best)
            with open(os.path.join(args.out_dir, "per_class_acc.json"), "w") as f:
                json.dump({str(k): v for k, v in (per_cls or {}).items()}, f, indent=2)

    print(f"[done] best val acc = {best_acc:.4f}  ->  {ckpt_best}")


if __name__ == "__main__":
    main()
