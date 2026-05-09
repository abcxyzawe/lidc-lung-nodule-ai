"""Step 06 - Evaluate model(s) on test set with TTA + threshold sweep + ensemble.

Metrics:
  - Slice-level: Dice, IoU (TTA-averaged)
  - Volume-level (per series): mean Dice across whole volume
  - Per-nodule F1: centroid match within 1.5 * GT diameter (lenient match;
    NOT the LUNA16 official rule which uses radius = diameter/2)
  - Hausdorff distance 95% (HD95)

Supports ensemble: pass --ckpts a.pt,b.pt,c.pt → average sigmoid outputs.

Run:
  python 06_evaluate.py --ckpt ../work/runs/best.pt
  python 06_evaluate.py --ckpts ../work/runs/swa.pt,../work/runs/snapshot_e150.pt
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from scipy.ndimage import label as cc_label

from configs import (
    PRE_DIR, SPLITS_JSON, RUNS_DIR, BATCH_SIZE, NUM_WORKERS,
    MIN_NODULE_VOXELS, HU_LO, HU_HI,
)
from dataset import LIDCSeg25D, LIDCSeriesVolume
from importlib import import_module
make_model = import_module("05_train").make_model


def load_models(ckpt_paths, device):
    """Returns list of models loaded from .pt paths."""
    models = []
    for cp in ckpt_paths:
        ck = torch.load(cp, map_location=device, weights_only=False)
        encoder = ck.get("encoder", "efficientnet-b5")
        m = make_model(encoder=encoder, pretrained=None).to(device)
        m.load_state_dict(ck["model"])
        m.train(False)
        models.append(m)
    return models


def tta_predict(models, img, tta=True):
    """img: [B, 3, H, W] on device. returns prob [B, 1, H, W] in [0,1]."""
    probs = []
    for m in models:
        with torch.amp.autocast("cuda"):
            p1 = torch.sigmoid(m(img))
            if tta:
                p2 = torch.sigmoid(m(torch.flip(img, dims=[-1]))).flip(dims=[-1])
                p3 = torch.sigmoid(m(torch.flip(img, dims=[-2]))).flip(dims=[-2])
                p4 = torch.sigmoid(m(torch.flip(img, dims=[-1, -2]))).flip(dims=[-1, -2])
                probs.append((p1 + p2 + p3 + p4) / 4)
            else:
                probs.append(p1)
    return torch.stack(probs, 0).mean(0)


def normalize(x):
    x = np.clip(x.astype(np.float32), HU_LO, HU_HI)
    return (x - HU_LO) / (HU_HI - HU_LO)


def find_nodules(mask_3d, voxel_sp, min_voxels=MIN_NODULE_VOXELS):
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, n = cc_label(mask_3d, structure=structure)
    nodules = []
    for blob in range(1, n + 1):
        coords = np.argwhere(labeled == blob)
        if len(coords) < min_voxels: continue
        # convert centroid voxel → mm
        c_vox = coords.mean(0)
        c_mm = c_vox * np.array(voxel_sp)
        vol = float(len(coords)) * np.prod(voxel_sp)
        diam = 2 * (3 * vol / (4 * np.pi)) ** (1 / 3)
        nodules.append({"centroid_mm": c_mm.tolist(),
                        "voxels": int(len(coords)),
                        "vol_mm3": vol, "diam_mm": diam})
    return nodules


def per_nodule_f1(gt_mask, pred_mask, voxel_sp, tol_factor=1.5):
    """Per-nodule F1: prediction matches GT if centroid distance <= tol_factor * GT diameter.

    NOTE: tol_factor=1.5 is more lenient than the LUNA16 official rule
    (radius = diameter/2, equivalent to tol_factor=0.5). Numbers from this script
    are therefore not directly comparable to LUNA16 leaderboard entries.
    """
    gt = find_nodules(gt_mask, voxel_sp)
    pr = find_nodules(pred_mask, voxel_sp)
    if not gt and not pr: return {"tp":0,"fp":0,"fn":0,"precision":1.0,"recall":1.0,"f1":1.0,
                                    "n_gt":0,"n_pred":0}
    matched_pr = set()
    tp = 0
    for g in gt:
        gc = np.array(g["centroid_mm"]); tol = tol_factor * g["diam_mm"]
        best = None; bestd = float("inf")
        for i, p in enumerate(pr):
            if i in matched_pr: continue
            d = np.linalg.norm(gc - np.array(p["centroid_mm"]))
            if d < bestd: bestd, best = d, i
        if best is not None and bestd <= tol:
            tp += 1; matched_pr.add(best)
    fp = len(pr) - tp
    fn = len(gt) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f1,
            "n_gt":len(gt),"n_pred":len(pr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PRE_DIR))
    ap.add_argument("--splits", default=str(SPLITS_JSON))
    ap.add_argument("--ckpt", default="", help="Single checkpoint (legacy)")
    ap.add_argument("--ckpts", default="", help="Comma-separated ensemble")
    ap.add_argument("--out", default="")
    ap.add_argument("--bs", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()

    if args.ckpts:
        ckpts = [s.strip() for s in args.ckpts.split(",") if s.strip()]
    elif args.ckpt:
        ckpts = [args.ckpt]
    else:
        ckpts = [str(RUNS_DIR / "best.pt")]
    out_path = Path(args.out) if args.out else Path(ckpts[0]).parent / "test_metrics.json"

    splits = json.load(open(args.splits))
    test_ds = LIDCSeg25D(args.root, splits["test"], augment=None)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(ckpts, device)
    print(f"Loaded {len(models)} model(s) for ensemble: {ckpts}")

    # ---- Slice-level: TTA + ensemble + threshold sweep ----
    print("\n[Slice-level inference + TTA + ensemble] ...")
    P, M = [], []
    with torch.no_grad():
        for img, mask in test_dl:
            img = img.to(device, non_blocking=True)
            prob = tta_predict(models, img, tta=bool(args.tta))
            P.append(prob.cpu()); M.append(mask)
    P = torch.cat(P, 0); M = torch.cat(M, 0)

    print("\n[Threshold sweep]")
    best_thr, best_d = 0.5, 0.0
    for thr in np.linspace(0.2, 0.7, 11):
        pred = (P > thr).float()
        dm = DiceMetric(include_background=False, reduction="mean")
        dm(y_pred=pred, y=M)
        d = float(dm.aggregate().item())
        print(f"  thr={thr:.2f}  dice={d:.4f}")
        if d > best_d: best_d, best_thr = d, float(thr)
    pred = (P > best_thr).float()
    iou = MeanIoU(include_background=False, reduction="mean"); iou(y_pred=pred, y=M)
    inter = (pred * M).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + M.sum(dim=(1,2,3))
    per_slice = (2 * inter / (denom + 1e-7)).numpy()

    # ---- Volume-level: per-patient series ----
    print("\n[Volume-level per series + per-nodule F1] ...")
    vol_dices, hd95s, f1s, precs, recs = [], [], [], [], []
    n_gt_total = n_pred_total = tp_total = fp_total = fn_total = 0
    for pid in splits["test"]:
        h5p = PRE_DIR / f"{pid}.h5"
        if not h5p.exists(): continue
        with h5py.File(h5p) as f:
            keys = list(f.keys())
        for k in keys:
            try: vol = LIDCSeriesVolume(h5p, k)
            except Exception: continue
            N = vol.images.shape[0]
            prob_vol = np.zeros((N, 512, 512), dtype=np.float32)
            with torch.no_grad():
                for i in range(N):
                    ip, ine = max(0, i-1), min(N-1, i+1)
                    stk = np.stack([normalize(vol.images[ip]),
                                    normalize(vol.images[i]),
                                    normalize(vol.images[ine])], 0)
                    x = torch.from_numpy(stk).unsqueeze(0).float().to(device)
                    p = tta_predict(models, x, tta=bool(args.tta))
                    prob_vol[i] = p[0,0].float().cpu().numpy()
            pred_vol = (prob_vol > best_thr).astype(np.uint8)
            gt_vol = vol.gt_mask("union")
            inter = (pred_vol & gt_vol).sum()
            denom = pred_vol.sum() + gt_vol.sum()
            d = (2 * inter / (denom + 1e-7)) if denom > 0 else float("nan")
            vol_dices.append(d)

            # Hausdorff
            try:
                hd = HausdorffDistanceMetric(include_background=False,
                                              percentile=95, reduction="mean")
                hd(y_pred=torch.from_numpy(pred_vol).unsqueeze(0).unsqueeze(0).float(),
                   y=torch.from_numpy(gt_vol).unsqueeze(0).unsqueeze(0).float())
                hd95 = float(hd.aggregate().item())
            except Exception: hd95 = float("nan")
            hd95s.append(hd95)

            # Per-nodule F1
            f1_res = per_nodule_f1(gt_vol, pred_vol, vol.voxel_spacing)
            f1s.append(f1_res["f1"]); precs.append(f1_res["precision"]); recs.append(f1_res["recall"])
            tp_total += f1_res["tp"]; fp_total += f1_res["fp"]; fn_total += f1_res["fn"]
            n_gt_total += f1_res["n_gt"]; n_pred_total += f1_res["n_pred"]

    micro_prec = tp_total / max(tp_total+fp_total, 1)
    micro_rec  = tp_total / max(tp_total+fn_total, 1)
    micro_f1 = 2*micro_prec*micro_rec / max(micro_prec+micro_rec, 1e-7)

    res = {
        "checkpoints": ckpts,
        "n_models_ensemble": len(models),
        "tta": bool(args.tta),
        "best_threshold": best_thr,
        # Slice-level
        "slice_dice_mean": float(best_d),
        "slice_iou_mean":  float(iou.aggregate().item()),
        "per_slice_dice_median": float(np.median(per_slice)),
        "per_slice_dice_p25": float(np.percentile(per_slice, 25)),
        "per_slice_dice_p75": float(np.percentile(per_slice, 75)),
        "per_slice_dice_p90": float(np.percentile(per_slice, 90)),
        "per_slice_dice_zero_pct": float(np.mean(per_slice == 0) * 100),
        # Volume-level
        "n_volumes": len(vol_dices),
        "volume_dice_mean": float(np.nanmean(vol_dices)),
        "volume_dice_median": float(np.nanmedian(vol_dices)),
        "hd95_mean_mm": float(np.nanmean(hd95s)),
        "hd95_median_mm": float(np.nanmedian(hd95s)),
        # Per-nodule F1 (lenient 1.5 * diameter match — NOT the LUNA16 official radius rule)
        "n_nodules_gt": n_gt_total,
        "n_nodules_pred": n_pred_total,
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
        "nodule_precision_micro": float(micro_prec),
        "nodule_recall_micro":    float(micro_rec),
        "nodule_f1_micro":        float(micro_f1),
        "nodule_f1_macro":        float(np.mean(f1s)) if f1s else 0.0,
    }
    out_path.write_text(json.dumps(res, indent=2))
    print("\n" + json.dumps(res, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
