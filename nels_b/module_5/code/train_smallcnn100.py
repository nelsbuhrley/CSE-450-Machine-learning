#!/usr/bin/env python
"""Train SmallCNN on GTSRB @ 100px with the honest group/track split.

Self-contained, no command-line flags: every setting is a constant in the
CONFIG block below. This reproduces the `models/run2_grouped` run exactly and is
meant to be read top-to-bottom and explained.

Run it from the module_5 directory (after `source activate.sh`):

    python code/train_smallcnn100.py
"""
from pathlib import Path
import csv, json, random, time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

# --------------------------------------------------------------------------- #
# CONFIG  — exactly how run2_grouped was trained
# --------------------------------------------------------------------------- #
DATA_DIR     = Path(__file__).resolve().parent.parent / "training"   # 43 class folders of JPGs
OUT_DIR      = Path(__file__).resolve().parent.parent / "models" / "run2_grouped_simple"

IMG_SIZE     = 100        # resize every image to 100x100
EPOCHS       = 50         # max epochs (early stopping usually ends sooner)
BATCH_SIZE   = 512
LR           = 3e-3       # AdamW learning rate
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15       # 15% of sign tracks held out for validation
PATIENCE     = 10         # stop if val accuracy doesn't improve for this many epochs
SEED         = 42
WORKERS      = 4          # dataloader processes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"   # mixed precision: faster on GPU, no-op on CPU


# --------------------------------------------------------------------------- #
# Model — compact 3-block CNN (~0.3M params), reaches ~99% val on GTSRB
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    def __init__(self, num_classes=43):
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


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class CachedDataset(Dataset):
    """Holds decoded+resized uint8 images in RAM; applies the transform per sample.

    The expensive JPEG decode + resize happens once; each epoch only runs cheap
    augmentation on the in-memory tensors, so the GPU never starves on disk I/O.
    """

    def __init__(self, images, labels, indices, transform):
        self.images, self.labels, self.indices, self.transform = images, labels, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.transform(self.images[idx]), int(self.labels[idx])


def track_id(path, cls):
    """GTSRB names files TRACK_FRAME.jpg — 30 near-identical frames per physical sign.
    The track id groups those frames so they never split across train and val."""
    stem = Path(path).stem                       # e.g. 00007_00021
    track = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return f"{cls}:{track}"


def load_data():
    # 1) Decode + resize every image once into one big uint8 tensor in RAM.
    pre = v2.Compose([v2.Resize((IMG_SIZE, IMG_SIZE)), v2.PILToTensor()])
    ds = datasets.ImageFolder(DATA_DIR, transform=pre)
    groups = np.array([track_id(p, c) for p, c in ds.samples])

    loader = DataLoader(ds, batch_size=512, num_workers=WORKERS, shuffle=False)
    chunks, labels = [], []
    print(f"[data] caching {len(ds)} images @ {IMG_SIZE}px into RAM (one-time)...")
    for imgs, lbls in loader:
        chunks.append(imgs); labels.append(lbls)
    images, labels = torch.cat(chunks), torch.cat(labels)
    print(f"[data] cached {tuple(images.shape)} uint8 "
          f"= {images.element_size() * images.nelement() / 1e6:.0f} MB")

    # 2) Honest split: keep all frames of a sign on one side (val = unseen signs).
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels.numpy(), groups))
    print(f"[data] group split: {len(train_idx)} train / {len(val_idx)} val "
          f"({len(set(groups[val_idx]))} val tracks held out)")

    # 3) Train gets augmentation; val gets only normalization.
    train_tf = v2.Compose([
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToDtype(torch.float32, scale=True),       # uint8 [0,255] -> float [0,1]
        v2.Normalize([0.5] * 3, [0.5] * 3),
    ])
    eval_tf = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5] * 3, [0.5] * 3),
    ])

    pin = DEVICE == "cuda"
    train_loader = DataLoader(CachedDataset(images, labels, train_idx, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS,
                              pin_memory=pin, persistent_workers=WORKERS > 0)
    val_loader = DataLoader(CachedDataset(images, labels, val_idx, eval_tf),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS,
                            pin_memory=pin, persistent_workers=WORKERS > 0)
    return train_loader, val_loader, ds.classes


# --------------------------------------------------------------------------- #
# One pass over a loader (train if an optimizer is given, else evaluate)
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.enable_grad() if train else torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device(DEVICE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device} amp={USE_AMP} out={OUT_DIR}")

    train_loader, val_loader, classes = load_data()

    model = SmallCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(device.type) if USE_AMP else None

    metrics_path = OUT_DIR / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "sec"])

    best_acc, bad_epochs, epoch = 0.0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        dt = time.time() - t0
        print(f"[{epoch:3d}/{EPOCHS}] train {tr_loss:.3f}/{tr_acc:.3f} | "
              f"val {va_loss:.3f}/{va_acc:.3f} | {dt:.1f}s")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                                    f"{va_loss:.4f}", f"{va_acc:.4f}", f"{dt:.1f}"])

        ckpt = {"model": model.state_dict(), "classes": classes, "epoch": epoch, "val_acc": va_acc}
        torch.save(ckpt, OUT_DIR / "last.pt")
        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            torch.save(ckpt, OUT_DIR / "best.pt")
        else:
            bad_epochs += 1
            if PATIENCE and bad_epochs >= PATIENCE:
                print(f"[train] early stop: no val improvement for {PATIENCE} epochs")
                break

    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"best_val_acc": best_acc, "epochs_run": epoch, "model": "smallcnn",
         "img_size": IMG_SIZE, "classes": len(classes), "group_split": True}, indent=2))
    print(f"[train] done. best val acc = {best_acc:.4f} | artifacts in {OUT_DIR}/")


if __name__ == "__main__":
    main()
