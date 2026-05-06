"""Step 10 — Train 3D malignancy classifier from extracted nodule patches.

Input:  work/malignancy_patches.npz   (from 09_extract_patches.py)
Splits: work/splits.json (same as segmentation, patient-level)

Architecture: MONAI DenseNet121 (3D), classes=5 (malignancy 1..5).
Loss:         CrossEntropy with class weights for imbalance.
Augment:      random flips, intensity jitter, gaussian noise.
Output:       work/runs/malignancy.pt  (best balanced acc on val)

Run:  python 10_train_malignancy.py
      python 10_train_malignancy.py --epochs 60 --bs 64
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from monai.networks.nets import DenseNet121

from configs import WORK, RUNS_DIR, SPLITS_JSON


HU_LO = -1024.0
HU_HI = 600.0


def normalize_patch(p):
    p = np.clip(p.astype(np.float32), HU_LO, HU_HI)
    return (p - HU_LO) / (HU_HI - HU_LO)


class PatchDataset(Dataset):
    def __init__(self, patches, labels, augment=False):
        self.patches = patches
        self.labels = labels.astype(np.int64) - 1  # 1..5 -> 0..4
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = normalize_patch(self.patches[idx])
        if self.augment:
            if np.random.rand() < 0.5: x = x[::-1].copy()
            if np.random.rand() < 0.5: x = x[:, ::-1].copy()
            if np.random.rand() < 0.5: x = x[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x + np.random.randn(*x.shape).astype(np.float32) * 0.02
            if np.random.rand() < 0.3:
                x = x * (1 + (np.random.rand() - 0.5) * 0.2)
            x = np.clip(x, 0, 1)
        return torch.from_numpy(x).unsqueeze(0).float(), int(self.labels[idx])


def warmup_cosine(opt, n_warm, n_total):
    def lr_lambda(step):
        if step < n_warm:
            return step / max(1, n_warm)
        progress = (step - n_warm) / max(1, n_total - n_warm)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(opt, lr_lambda)


def evaluate(model, dl, device, n_classes=5):
    model.train(False)
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
            pred = logits.argmax(1).cpu().numpy()
            all_pred.append(pred); all_true.append(y.numpy())
    all_pred = np.concatenate(all_pred); all_true = np.concatenate(all_true)
    acc = float((all_pred == all_true).mean())
    per_class = []
    for c in range(n_classes):
        m = all_true == c
        if m.sum() == 0: continue
        per_class.append(float((all_pred[m] == c).mean()))
    bal_acc = float(np.mean(per_class)) if per_class else 0.0
    is_susp_pred = (all_pred >= 3).astype(int)
    is_susp_true = (all_true >= 3).astype(int)
    if is_susp_true.sum() > 0 and (1 - is_susp_true).sum() > 0:
        tp = int(((is_susp_pred == 1) & (is_susp_true == 1)).sum())
        fp = int(((is_susp_pred == 1) & (is_susp_true == 0)).sum())
        fn = int(((is_susp_pred == 0) & (is_susp_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        bin_f1 = 2 * prec * rec / max(prec + rec, 1e-7)
    else:
        bin_f1 = 0.0
    return acc, bal_acc, bin_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default=str(WORK / "malignancy_patches.npz"))
    ap.add_argument("--splits", default=str(SPLITS_JSON))
    ap.add_argument("--out", default=str(RUNS_DIR / "malignancy.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print(f"Loading patches from {args.patches} ...")
    data = np.load(args.patches)
    patches = data["patches"]
    labels = data["labels"]
    pids = data["pids"]
    # Drop invalid (label==0 should not exist in 1..5 scale)
    valid = (labels >= 1) & (labels <= 5)
    if (~valid).any():
        print(f"  dropping {int((~valid).sum())} patches with invalid labels")
        patches = patches[valid]; labels = labels[valid]; pids = pids[valid]

    splits = json.load(open(args.splits))
    train_pids = set(splits["train"])
    val_pids = set(splits["val"])
    test_pids = set(splits["test"])
    train_mask = np.array([p in train_pids for p in pids])
    val_mask = np.array([p in val_pids for p in pids])
    test_mask = np.array([p in test_pids for p in pids])
    print(f"Total {len(patches)} patches | train {train_mask.sum()} | val {val_mask.sum()} | test {test_mask.sum()}")

    train_ds = PatchDataset(patches[train_mask], labels[train_mask], augment=True)
    val_ds = PatchDataset(patches[val_mask], labels[val_mask], augment=False)
    test_ds = PatchDataset(patches[test_mask], labels[test_mask], augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                         pin_memory=True, persistent_workers=args.workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, GPU: "
          f"{torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=5).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: DenseNet121-3D ({n_params:.1f}M params)")

    train_labels = labels[train_mask].astype(int) - 1
    class_counts = np.bincount(train_labels, minlength=5).astype(np.float32)
    class_weights = (class_counts.sum() / (5 * np.maximum(class_counts, 1))).astype(np.float32)
    print(f"Class counts (train): {class_counts.astype(int).tolist()}")
    print(f"Class weights        : {[round(float(w), 3) for w in class_weights]}")
    cw = torch.from_numpy(class_weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_steps = args.epochs * len(train_dl)
    sched = warmup_cosine(opt, n_warm=int(0.05 * n_steps), n_total=n_steps)
    scaler = torch.amp.GradScaler("cuda")

    log_path = Path(args.out).parent / "malignancy_log.txt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_balacc = 0.0
    for epoch in range(args.epochs):
        model.train(True)
        t0 = time.time(); tot = 0.0; n = 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        train_loss = tot / max(n, 1)

        val_acc, val_bal, val_f1 = evaluate(model, val_dl, device)
        dt = time.time() - t0
        msg = (f"epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  "
               f"val_acc={val_acc:.4f}  val_bal_acc={val_bal:.4f}  "
               f"val_susp_f1={val_f1:.4f}  ({dt:.0f}s)")
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(msg + "\n")

        if val_bal > best_balacc:
            best_balacc = val_bal
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_bal_acc": val_bal, "val_acc": val_acc,
                        "n_classes": 5, "patch_size": 32,
                        "hu_lo": HU_LO, "hu_hi": HU_HI}, args.out)
            print(f"  -> saved {Path(args.out).name} (bal_acc={best_balacc:.4f})")

    print("\nLoading best ckpt for final test eval ...")
    ck = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    test_acc, test_bal, test_f1 = evaluate(model, test_dl, device)
    print(f"TEST  acc={test_acc:.4f}  bal_acc={test_bal:.4f}  susp_f1={test_f1:.4f}")
    final = {"test_acc": test_acc, "test_bal_acc": test_bal, "test_susp_f1": test_f1,
             "best_val_bal_acc": best_balacc}
    (Path(args.out).parent / "malignancy_metrics.json").write_text(json.dumps(final, indent=2))
    print(f"Done. Saved {args.out} + malignancy_metrics.json")


if __name__ == "__main__":
    main()
