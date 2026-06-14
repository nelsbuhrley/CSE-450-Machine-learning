#!/usr/bin/env python
"""Train a CNN classifier on the GTSRB German traffic-sign dataset.

Layout expected (torchvision ImageFolder): <data-dir>/<class>/<image>.jpg
with 43 class folders (00000..00042).

Designed for fast iteration on a shared cluster:
  * --smoke      tiny subset + 1 epoch to validate the pipeline (use on login CPU)
  * --amp        mixed precision (default on) for ~2x throughput on modern GPUs
  * checkpoints  best model + a resumable last.pt, plus a metrics.csv per run

Examples
--------
  # Validate the whole pipeline in ~10s, no GPU needed:
  python code/train.py --smoke --device cpu

  # Real run (inside a GPU job / interactive session):
  python code/train.py --epochs 30 --model smallcnn --out models/run1
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    """Compact 3-block CNN. Fast and reaches ~98% val accuracy on GTSRB."""

    def __init__(self, num_classes: int = 43):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class STN(nn.Module):
    """Spatial Transformer front-end (Jaderberg et al. 2015).

    A small localization net predicts a 2x3 affine matrix; the input is then
    resampled through that warp (F.affine_grid + F.grid_sample) before the
    backbone sees it. Purely *geometric* — it can crop/zoom/rotate/shift toward
    the sign but does NOT change brightness/contrast. The final layer is
    identity-initialized, so training starts from a no-op warp and the net learns
    where to look using only the classification loss (no zoom/crop labels).
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        # AdaptiveAvgPool makes the localizer agnostic to --img-size.
        self.loc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 7, padding=3), nn.MaxPool2d(2), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, padding=2), nn.MaxPool2d(2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )
        # Identity affine [[1,0,0],[0,1,0]] so the STN is a no-op at init.
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.fc(self.loc(x)).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class STNWrapper(nn.Module):
    """Prepend an STN to any classification backbone."""

    def __init__(self, backbone: nn.Module, in_ch: int = 3):
        super().__init__()
        self.stn = STN(in_ch)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(self.stn(x))


MODELS = ["smallcnn", "resnet18", "resnet34", "mobilenet_v3_small", "efficientnet_b0"]


def build_model(name: str, num_classes: int, stn: bool = False) -> nn.Module:
    name = name.lower()
    if name == "smallcnn":
        m = SmallCNN(num_classes)
    else:
        from torchvision import models
        if name in ("resnet18", "resnet34"):
            m = getattr(models, name)(weights=None)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        elif name in ("mobilenet_v3_small", "efficientnet_b0"):
            m = getattr(models, name)(weights=None)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        else:
            raise ValueError(f"unknown model '{name}' (choose: {', '.join(MODELS)})")
    return STNWrapper(m) if stn else m


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def _clahe_gray(y, clip_limit, n_tiles):
    """Contrast-Limited Adaptive Histogram Equalization on one uint8 channel.

    Tiles the image n_tiles x n_tiles, builds a clipped-CDF mapping per tile, then
    bilinearly interpolates the four surrounding tile mappings at each pixel (the
    "interpolation" step that removes the blocky look of plain adaptive HE).
    """
    H, W = y.shape
    ty = tx = n_tiles
    ph, pw = (-H) % ty, (-W) % tx
    if ph or pw:
        y = np.pad(y, ((0, ph), (0, pw)), mode="reflect")
    Hp, Wp = y.shape
    th, tw = Hp // ty, Wp // tx
    clip = max(1.0, clip_limit * (th * tw) / 256.0)

    lut = np.empty((ty, tx, 256), dtype=np.float32)
    for i in range(ty):
        for j in range(tx):
            tile = y[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float32)
            excess = np.maximum(hist - clip, 0.0).sum()
            hist = np.minimum(hist, clip) + excess / 256.0   # clip + redistribute
            cdf = np.cumsum(hist)
            lut[i, j] = cdf / cdf[-1] * 255.0

    yc = np.clip(np.arange(Hp) / th - 0.5, 0, ty - 1)
    xc = np.clip(np.arange(Wp) / tw - 0.5, 0, tx - 1)
    i0 = np.floor(yc).astype(int); i1 = np.minimum(i0 + 1, ty - 1); fy = (yc - i0).astype(np.float32)
    j0 = np.floor(xc).astype(int); j1 = np.minimum(j0 + 1, tx - 1); fx = (xc - j0).astype(np.float32)
    v = y.astype(int)
    A = lut[i0[:, None], j0[None, :], v]; B = lut[i0[:, None], j1[None, :], v]
    C = lut[i1[:, None], j0[None, :], v]; D = lut[i1[:, None], j1[None, :], v]
    top = A * (1 - fx)[None, :] + B * fx[None, :]
    bot = C * (1 - fx)[None, :] + D * fx[None, :]
    out = top * (1 - fy)[:, None] + bot * fy[:, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)[:H, :W]


class CLAHE:
    """PIL RGB -> PIL RGB. Equalizes only the luminance (Y) channel so colors are
    preserved; brightens dark/backlit signs and normalizes local contrast. Inserted
    right after Resize in every pipeline (train cache, streaming, and inference)."""

    def __init__(self, clip_limit=2.0, n_tiles=4):
        self.clip_limit = float(clip_limit)
        self.n_tiles = int(n_tiles)

    def __call__(self, img):
        ycbcr = np.asarray(img.convert("YCbCr")).copy()
        ycbcr[..., 0] = _clahe_gray(ycbcr[..., 0], self.clip_limit, self.n_tiles)
        return Image.fromarray(ycbcr, "YCbCr").convert("RGB")


def make_clahe(args):
    return CLAHE(args.clahe_clip, args.clahe_tiles) if args.clahe else None


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class CachedTensorDataset(Dataset):
    """Decoded+resized uint8 CHW images held in RAM; transform applied per sample.

    The expensive JPEG decode + resize happens once in _preload(); each epoch then
    only runs cheap augmentation on the in-memory tensor, so the GPU stops starving.
    """

    def __init__(self, images, labels, indices, transform):
        self.images = images        # uint8 [N,3,H,W], shared by both splits (COW across workers)
        self.labels = labels        # long  [N]
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.transform(self.images[idx]), int(self.labels[idx])


def _track_id(path, cls):
    """Group key for a frame: GTSRB names files TRACK_FRAME.ext (30 frames per sign).
    Combine with class so tracks are globally unique across class folders."""
    stem = Path(path).stem                       # e.g. 00007_00021
    track = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return f"{cls}:{track}"


def _preload(data_dir, img_size, workers, clahe=None):
    """Decode + resize (+ optional CLAHE) every image once into one uint8 tensor.

    CLAHE is baked into the cache here (a deterministic preprocess, not augmentation),
    so it's computed once rather than every epoch."""
    steps = [v2.Resize((img_size, img_size))]
    if clahe is not None:
        steps.append(clahe)
    steps.append(v2.PILToTensor())
    pre = v2.Compose(steps)
    ds = datasets.ImageFolder(data_dir, transform=pre)
    groups = np.array([_track_id(p, c) for p, c in ds.samples])  # aligned with load order
    loader = DataLoader(ds, batch_size=512, num_workers=workers, shuffle=False)
    chunks, labels = [], []
    print(f"[data] caching {len(ds)} images @ {img_size}px into RAM (one-time)...")
    for imgs, lbls in tqdm(loader, unit="batch"):
        chunks.append(imgs)
        labels.append(lbls)
    images = torch.cat(chunks)
    labels = torch.cat(labels)
    mb = images.element_size() * images.nelement() / 1e6
    print(f"[data] cached {tuple(images.shape)} uint8 = {mb:.0f} MB in RAM")
    return images, labels, ds.classes, groups


def make_split(labels, groups, val_split, seed, group_split):
    """Train/val indices. group_split keeps all frames of one physical sign on the
    same side (honest accuracy); otherwise a per-image stratified split (leaks
    near-duplicate frames into val and inflates accuracy)."""
    labels = np.asarray(labels)
    if group_split:
        from sklearn.model_selection import GroupShuffleSplit
        sp = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        return next(sp.split(np.zeros(len(labels)), labels, groups))
    from sklearn.model_selection import StratifiedShuffleSplit
    sp = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    return next(sp.split(np.zeros(len(labels)), labels))


def eval_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5] * 3, [0.5] * 3),
    ])


def build_loaders(args):
    if not args.cache:
        return _build_loaders_streaming(args)

    images, labels, classes, groups = _preload(args.data_dir, args.img_size, args.workers, make_clahe(args))
    if args.clahe:
        print(f"[data] CLAHE baked into cache (clip={args.clahe_clip}, tiles={args.clahe_tiles})")

    # Augmentation runs on cached uint8 tensors — no per-epoch JPEG decode/resize.
    train_tf = v2.Compose([
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),   # uint8 [0,255] -> float [0,1]
        v2.Normalize([0.5] * 3, [0.5] * 3),
    ])
    eval_tf = eval_transform()

    train_idx, val_idx = make_split(labels.numpy(), groups, args.val_split, args.seed, args.group_split)
    kind = "group/track" if args.group_split else "per-image"
    print(f"[data] {kind} split: {len(train_idx)} train / {len(val_idx)} val "
          f"({len(set(groups[val_idx]))} val tracks held out)")

    train_ds = CachedTensorDataset(images, labels, train_idx, train_tf)
    val_ds = CachedTensorDataset(images, labels, val_idx, eval_tf)

    pin = args.device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin,
                            persistent_workers=args.workers > 0)
    return train_loader, val_loader, classes


def _build_loaders_streaming(args):
    """Original path: decode + augment from disk every epoch (used by --no-cache / --smoke)."""
    clahe = make_clahe(args)
    clahe_step = [clahe] if clahe is not None else []
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        *clahe_step,
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        *clahe_step,
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # Two views of the same folder so train/val get different transforms.
    base = datasets.ImageFolder(args.data_dir)
    targets = np.array(base.targets)

    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))

    if args.smoke:  # shrink to validate the pipeline quickly
        train_idx = train_idx[: args.smoke_n]
        val_idx = val_idx[: args.smoke_n // 2]

    train_ds = Subset(datasets.ImageFolder(args.data_dir, transform=train_tf), train_idx)
    val_ds = Subset(datasets.ImageFolder(args.data_dir, transform=eval_tf), val_idx)

    pin = args.device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=False,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin,
                            persistent_workers=args.workers > 0)
    return train_loader, val_loader, base.classes


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, amp=False):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device} amp={amp} out={out}")

    train_loader, val_loader, classes = build_loaders(args)
    print(f"[train] {len(classes)} classes | "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images")

    model = build_model(args.model, len(classes), stn=args.stn).to(device)
    if args.stn:
        print("[train] Spatial Transformer front-end enabled (identity-initialized)")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type) if amp else None

    metrics_path = out / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "sec"])

    best_acc, bad_epochs = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer, scaler, amp)
        va_loss, va_acc = run_epoch(model, val_loader, device, criterion, amp=amp)
        scheduler.step()
        dt = time.time() - t0
        print(f"[{epoch:3d}/{args.epochs}] "
              f"train {tr_loss:.3f}/{tr_acc:.3f} | val {va_loss:.3f}/{va_acc:.3f} | {dt:.1f}s")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                                    f"{va_loss:.4f}", f"{va_acc:.4f}", f"{dt:.1f}"])

        ckpt = {"model": model.state_dict(), "classes": classes, "args": vars(args),
                "epoch": epoch, "val_acc": va_acc}
        torch.save(ckpt, out / "last.pt")
        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            torch.save(ckpt, out / "best.pt")
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience:
                print(f"[train] early stop: no val improvement for {args.patience} epochs")
                break

    summary = {"best_val_acc": best_acc, "epochs_run": epoch, "model": args.model,
               "img_size": args.img_size, "classes": len(classes),
               "group_split": args.group_split, "stn": args.stn, "clahe": args.clahe}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done. best val acc = {best_acc:.4f} | artifacts in {out}/")


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(repo_root / "training"))
    p.add_argument("--out", default=str(repo_root / "models" / "run"))
    p.add_argument("--model", default="smallcnn", choices=MODELS)
    p.add_argument("--stn", action="store_true",
                   help="prepend an identity-initialized Spatial Transformer (learned crop/zoom/rotate)")
    p.add_argument("--clahe", action="store_true",
                   help="CLAHE contrast normalization on luminance (brightens dark/backlit signs)")
    p.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit")
    p.add_argument("--clahe-tiles", type=int, default=4, help="CLAHE tile grid (n x n)")
    p.add_argument("--img-size", type=int, default=48)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=5, help="early-stop patience (0 disables)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--cache", action="store_true", default=True,
                   help="preload all images into RAM once (default; keeps the GPU fed)")
    p.add_argument("--no-cache", dest="cache", action="store_false",
                   help="decode + augment from disk every epoch")
    p.add_argument("--group-split", action="store_true", default=True,
                   help="split by sign track so frames of one sign don't leak into val (default; honest)")
    p.add_argument("--random-split", dest="group_split", action="store_false",
                   help="legacy per-image split (leaks near-duplicate frames into val)")
    p.add_argument("--smoke", action="store_true", help="tiny subset + few steps to test the pipeline")
    p.add_argument("--smoke-n", type=int, default=512)
    args = p.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.workers = min(args.workers, 2)
        args.cache = False  # streaming path already subsets to a tiny set for a fast check
    return args


if __name__ == "__main__":
    main()
