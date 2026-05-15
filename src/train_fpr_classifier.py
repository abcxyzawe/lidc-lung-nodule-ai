"""Train binary 3D classifier for false-positive reduction.

Input: fpr_train.npz with fields (patches [N,P,P,P] int16 HU, labels [N], confs [N])
Output: work/runs/fpr.pt - DenseNet121-3D binary classifier checkpoint

Run:
  python train_fpr_classifier.py --train work/academic/fpr_val.npz \
      --val work/academic/fpr_test.npz --epochs 30 --bs 32
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from monai.networks.nets import DenseNet121

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import RUNS_DIR  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HU_LO, HU_HI = -1024.0, 600.0


def normalize_hu(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), HU_LO, HU_HI)
    return (x - HU_LO) / (HU_HI - HU_LO)


class FPRDataset(Dataset):
    def __init__(self, npz_path, augment=False):
        d = np.load(npz_path)
        self.patches = d["patches"]
        self.labels = d["labels"].astype(np.int64)
        self.confs = d["confs"].astype(np.float32)
        self.augment = augment
        print(f"  {npz_path}: {len(self.patches)} samples "
              f"({(self.labels==1).sum()} pos, {(self.labels==0).sum()} neg)")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        x = normalize_hu(self.patches[i])
        if self.augment:
            for ax in range(3):
                if np.random.rand() < 0.5:
                    x = np.flip(x, axis=ax).copy()
            k = np.random.randint(0, 4)
            if k:
                x = np.rot90(x, k, axes=(1, 2)).copy()
        x = torch.from_numpy(x).unsqueeze(0)
        return x, int(self.labels[i])


def run_validation(model, loader):
    model.train(False)
    all_p, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                p = torch.softmax(model(x), dim=1)[:, 1]
            all_p.append(p.cpu().numpy())
            all_y.append(y.numpy())
    p = np.concatenate(all_p); y = np.concatenate(all_y)
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float(y.mean())
    best_f1, best_thr = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 17):
        pred = (p > t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-7)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return {"auc": float(auc), "ap": float(ap),
            "best_f1": float(best_f1), "best_thr": best_thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(RUNS_DIR / "fpr.pt"))
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    print("Datasets:")
    train_ds = FPRDataset(args.train, augment=True)
    val_ds = FPRDataset(args.val, augment=False) if args.val else None

    n_pos = (train_ds.labels == 1).sum()
    n_neg = (train_ds.labels == 0).sum()
    weights = np.where(train_ds.labels == 1, 1.0 / n_pos, 1.0 / n_neg)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.bs, sampler=sampler,
                          num_workers=2, pin_memory=True)
    val_dl = (DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                          num_workers=2, pin_memory=True) if val_ds else None)

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    pos_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {pos_weight}")

    best_auc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n_seen = 0.0, 0
        for x, y in train_dl:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * x.size(0); n_seen += x.size(0)
        sched.step()
        train_loss = loss_sum / max(n_seen, 1)
        msg = f"  ep{ep:>3}  loss={train_loss:.4f}  lr={opt.param_groups[0]['lr']:.5f}  ({time.time()-t0:.0f}s)"
        if val_dl is not None and ep % 3 == 0:
            metrics = run_validation(model, val_dl)
            msg += (f"  | val AUC={metrics['auc']:.3f} AP={metrics['ap']:.3f} "
                    f"F1={metrics['best_f1']:.3f}@{metrics['best_thr']:.2f}")
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                torch.save({"model": model.state_dict(),
                            "epoch": ep, "auc": metrics["auc"],
                            "best_thr": metrics["best_thr"]}, args.out)
                msg += "  [BEST -> saved]"
        print(msg, flush=True)

    print(f"\nDone. Best val AUC = {best_auc:.4f}. Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
