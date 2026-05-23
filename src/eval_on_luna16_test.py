"""Score a trained model on LUNA16 subset 9 (held-out test fold).

Supports --model-type:
  finetune -- UNet++ EfficientNet-B5 SCSE 2.5D (run003/run005 ckpts)
  lidc     -- LIDC winner UNet++ 3-ch (stage2_full best.pt, baseline)
  monai    -- MONAI RetinaNet 3D TorchScript bundle

Matching rule: LUNA16 official pred_center within gt_diameter/2 of GT center.
(NOT the 15mm fixed rule used in LIDC test_panel -- see commit db48359)

Usage (V100):
  # Smoke 5 scans
  python src/eval_on_luna16_test.py \
      --ckpt /workspace/work/runs_luna16/run003_scratch_full/best.pt \
      --luna16-dir /workspace/datasetLuna16 --subset 9 --n-scans 5 \
      --output-json /workspace/work/runs_luna16/eval_run003_smoke.json

  # Full run005
  python src/eval_on_luna16_test.py \
      --ckpt /workspace/work/runs_luna16/run005_mine_luna16/best.pt \
      --luna16-dir /workspace/datasetLuna16 --subset 9 \
      --output-json /workspace/work/runs_luna16/eval_run005_subset9.json

  # MONAI baseline
  python src/eval_on_luna16_test.py --model-type monai \
      --luna16-dir /workspace/datasetLuna16 --subset 9 \
      --output-json /workspace/work/runs_luna16/eval_monai_subset9.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

# ---------------------------------------------------------------------------
# Constants -- mirror finetune_luna16.py; no magic numbers inline
# ---------------------------------------------------------------------------
HU_LO = -1000.0
HU_HI = 200.0
PATCH_HW = 128
PATCH_SLICES = 5
PROB_THRESHOLD = 0.5
LUNA16_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
MIN_BLOB_VOXELS = 2
STRIDE_Z = 1
STRIDE_HW = 64


# ---------------------------------------------------------------------------
# I/O helpers (same logic as finetune_luna16.py)
# ---------------------------------------------------------------------------

def parse_annotations(csv_path: Path) -> Dict[str, List[dict]]:
    """Return {seriesuid: [{coord_world_xyz, diameter_mm}, ...]}."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    out: Dict[str, List[dict]] = {}
    for _, row in df.iterrows():
        uid = str(row["seriesuid"])
        entry = {
            "coord_world_xyz": [float(row["coordX"]),
                                 float(row["coordY"]),
                                 float(row["coordZ"])],
            "diameter_mm": float(row["diameter_mm"]),
        }
        out.setdefault(uid, []).append(entry)
    return out


def load_scan(mhd_path: Path):
    """Load .mhd. Returns (vol_zyx int16, spacing_zyx, origin_zyx) all ZYX order."""
    img = sitk.ReadImage(str(mhd_path))
    vol_zyx = sitk.GetArrayFromImage(img).astype(np.int16)
    spacing_zyx = tuple(reversed(img.GetSpacing()))
    origin_zyx = tuple(reversed(img.GetOrigin()))
    return vol_zyx, spacing_zyx, origin_zyx


def world_to_voxel_zyx(coord_xyz: list, origin_zyx: tuple,
                        spacing_zyx: tuple) -> np.ndarray:
    """LUNA16 world XYZ mm -> voxel ZYX index."""
    x, y, z = coord_xyz
    oz, oy, ox = origin_zyx
    sz, sy, sx = spacing_zyx
    return np.array([(z - oz) / sz, (y - oy) / sy, (x - ox) / sx], dtype=np.float32)


def normalize_hu(vol: np.ndarray, lo: float = HU_LO, hi: float = HU_HI) -> np.ndarray:
    """Clip HU to [lo, hi] and scale to [0, 1] float32."""
    return (np.clip(vol.astype(np.float32), lo, hi) - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# 2.5D sliding-window inference -- exact match to training pipeline
# ---------------------------------------------------------------------------

def run_inference_25d(
    model: torch.nn.Module,
    vol_norm: np.ndarray,
    device: torch.device,
    patch_hw: int = PATCH_HW,
    n_slices: int = PATCH_SLICES,
    stride_hw: int = STRIDE_HW,
    stride_z: int = STRIDE_Z,
) -> np.ndarray:
    """Slide 2.5D patch across volume; return prob_vol [Z,Y,X] float32.

    Each voxel receives the MAX probability over all overlapping patches.
    Max-fusion preserves nodule peaks; averaging would dilute them.
    """
    model.train(False)
    nZ, nY, nX = vol_norm.shape
    half_sl = n_slices // 2
    half_hw = patch_hw // 2
    prob_vol = np.zeros((nZ, nY, nX), dtype=np.float32)

    def _centers(n, half, stride):
        cs = list(range(half, n - half, stride))
        if not cs or cs[-1] < n - half - 1:
            cs.append(n - half - 1)
        return cs

    z_centers = _centers(nZ, half_sl, stride_z)
    y_centers = _centers(nY, half_hw, stride_hw)
    x_centers = _centers(nX, half_hw, stride_hw)

    batch_inputs: list = []
    batch_positions: list = []
    BATCH = 16

    def _flush(b_in, b_pos):
        x_t = torch.from_numpy(np.stack(b_in)).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        for (cz, cy, cx), pp in zip(b_pos, probs):
            p2d = pp[0]
            y0, y1 = cy - half_hw, cy + half_hw
            x0, x1 = cx - half_hw, cx + half_hw
            prob_vol[cz, y0:y1, x0:x1] = np.maximum(prob_vol[cz, y0:y1, x0:x1], p2d)

    for cz in z_centers:
        for cy in y_centers:
            for cx in x_centers:
                z_idxs = [max(0, min(nZ - 1, cz + off))
                           for off in range(-half_sl, half_sl + 1)]
                patch = vol_norm[z_idxs,
                                 cy - half_hw: cy + half_hw,
                                 cx - half_hw: cx + half_hw]
                if patch.shape[1] < patch_hw or patch.shape[2] < patch_hw:
                    continue
                batch_inputs.append(patch.astype(np.float32))
                batch_positions.append((cz, cy, cx))
                if len(batch_inputs) >= BATCH:
                    _flush(batch_inputs, batch_positions)
                    batch_inputs, batch_positions = [], []
    if batch_inputs:
        _flush(batch_inputs, batch_positions)
    return prob_vol


# ---------------------------------------------------------------------------
# Blob detection: prob_vol -> candidate centroids
# ---------------------------------------------------------------------------

def detect_blobs(
    prob_vol: np.ndarray,
    threshold: float = PROB_THRESHOLD,
    min_voxels: int = MIN_BLOB_VOXELS,
) -> List[dict]:
    """CC-label thresholded prob map. Returns [{centroid_zyx, max_prob, n_voxels}]."""
    labeled, n_cc = ndimage.label(prob_vol > threshold)
    if n_cc == 0:
        return []
    out = []
    for lbl in range(1, n_cc + 1):
        mask = labeled == lbl
        n_vox = int(mask.sum())
        if n_vox < min_voxels:
            continue
        zz, yy, xx = np.where(mask)
        weights = prob_vol[zz, yy, xx]
        w_sum = float(weights.sum())
        if w_sum < 1e-9:
            cz, cy, cx = float(zz.mean()), float(yy.mean()), float(xx.mean())
        else:
            cz = float((zz * weights).sum() / w_sum)
            cy = float((yy * weights).sum() / w_sum)
            cx = float((xx * weights).sum() / w_sum)
        out.append({
            "centroid_zyx": np.array([cz, cy, cx]),
            "max_prob": float(prob_vol[zz, yy, xx].max()),
            "n_voxels": n_vox,
        })
    return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_finetune_model(
    ckpt_path: Path, device: torch.device, in_channels: int = PATCH_SLICES
) -> torch.nn.Module:
    """Load UNet++ EfficientNet-B5 SCSE from finetune_luna16.py checkpoint."""
    import segmentation_models_pytorch as smp
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_in_ch = ck.get("in_channels", PATCH_SLICES)
    mdl = smp.UnetPlusPlus(
        encoder_name="efficientnet-b5", encoder_weights=None,
        in_channels=in_channels, classes=1, decoder_attention_type="scse",
    )
    if ckpt_in_ch != in_channels:
        sd = ck["model"]
        for key in [k for k in sd if "conv_stem.weight" in k or "_conv_stem.weight" in k]:
            w = sd[key]
            w_avg = w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            sd[key] = w_avg * (ckpt_in_ch / in_channels)
        mdl.load_state_dict(sd, strict=False)
    else:
        mdl.load_state_dict(ck["model"], strict=True)
    mdl = mdl.to(device)
    mdl.train(False)
    print(f"[model] ep={ck.get('epoch','?')}  "
          f"val_dice={ck.get('val_dice','N/A')}  "
          f"cpm={ck.get('val_froc_cpm','N/A')}")
    return mdl


def load_lidc_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """3-ch LIDC winner UNet++ for LUNA16 baseline comparison."""
    return load_finetune_model(ckpt_path, device, in_channels=3)


def load_monai_predict_fn():
    """Return predict_nodules_monai from src/webapp/predict_monai.py."""
    webapp_dir = Path(__file__).resolve().parent / "webapp"
    sys.path.insert(0, str(webapp_dir))
    from predict_monai import predict_nodules_monai
    return predict_nodules_monai


# ---------------------------------------------------------------------------
# LUNA16 matching rule
# ---------------------------------------------------------------------------

def dist_mm_zyx(vox_a: np.ndarray, vox_b: np.ndarray, spacing_zyx: tuple) -> float:
    """Euclidean distance in mm between ZYX voxel coords."""
    sz, sy, sx = spacing_zyx
    d = vox_a - vox_b
    return float(np.sqrt((d[0] * sz) ** 2 + (d[1] * sy) ** 2 + (d[2] * sx) ** 2))


def match_preds_to_gt(
    preds: List[dict],
    gt_list: List[dict],
    spacing_zyx: tuple,
    origin_zyx: tuple,
) -> Tuple[List[dict], List[bool]]:
    """LUNA16 official matching: pred within gt_diameter/2 of GT centre.

    Greedy (highest-confidence pred claims GT first).
    Returns (matched_preds_list, gt_matched_bool_list).
    """
    gt_vox = [world_to_voxel_zyx(g["coord_world_xyz"], origin_zyx, spacing_zyx)
              for g in gt_list]
    gt_matched = [False] * len(gt_list)
    matched = []
    for pred in sorted(preds, key=lambda p: -p["max_prob"]):
        pc = pred["centroid_zyx"]
        best_gi, best_dist = -1, float("inf")
        for gi, (gv, g) in enumerate(zip(gt_vox, gt_list)):
            if gt_matched[gi]:
                continue
            d = dist_mm_zyx(pc, gv, spacing_zyx)
            if d <= g["diameter_mm"] / 2.0 and d < best_dist:
                best_dist, best_gi = d, gi
        is_tp = best_gi >= 0
        if is_tp:
            gt_matched[best_gi] = True
        matched.append({
            "centroid_zyx": pc.tolist(),
            "max_prob": pred["max_prob"],
            "n_voxels": pred.get("n_voxels", 0),
            "matched_gt_idx": best_gi,
            "is_tp": is_tp,
            "distance_mm": best_dist if is_tp else -1.0,
        })
    return matched, gt_matched


# ---------------------------------------------------------------------------
# FROC computation
# ---------------------------------------------------------------------------

def compute_froc_metrics(
    all_tp_scores: List[float],
    all_fp_scores: List[float],
    n_gt_total: int,
    n_scans: int,
    fp_rates: List[float] = LUNA16_FP_RATES,
) -> dict:
    """Build FROC curve, interpolate sensitivity at LUNA16 FP rates, compute CPM."""
    combined = [(s, True) for s in all_tp_scores] + [(s, False) for s in all_fp_scores]
    combined.sort(key=lambda t: -t[0])
    froc_pts = [{"threshold": 1.0, "sensitivity": 0.0, "fp_per_scan": 0.0,
                 "tp_cum": 0, "fp_cum": 0}]
    tp_cum = fp_cum = 0
    seen: set = set()
    for score, is_tp in combined:
        tp_cum += int(is_tp)
        fp_cum += int(not is_tp)
        if score not in seen:
            froc_pts.append({
                "threshold": float(score),
                "sensitivity": tp_cum / max(n_gt_total, 1),
                "fp_per_scan": fp_cum / max(n_scans, 1),
                "tp_cum": tp_cum, "fp_cum": fp_cum,
            })
            seen.add(score)
    froc_pts.sort(key=lambda p: p["fp_per_scan"])
    fp_arr = np.array([p["fp_per_scan"] for p in froc_pts])
    sens_arr = np.array([p["sensitivity"] for p in froc_pts])
    luna_sens = {}
    for fp_t in fp_rates:
        if len(fp_arr) == 0 or fp_t <= fp_arr.min():
            s = float(sens_arr[0]) if len(sens_arr) else 0.0
        elif fp_t >= fp_arr.max():
            s = float(sens_arr[-1])
        else:
            s = float(np.interp(fp_t, fp_arr, sens_arr))
        luna_sens[fp_t] = round(s, 4)
    cpm = float(np.mean(list(luna_sens.values())))
    best_f1 = best_thr = best_sens = best_prec = 0.0
    for pt in froc_pts:
        tp_, fp_ = pt["tp_cum"], pt["fp_cum"]
        prec = tp_ / max(tp_ + fp_, 1)
        rec = tp_ / max(n_gt_total, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-7)
        if f1 > best_f1:
            best_f1, best_thr, best_sens, best_prec = f1, pt["threshold"], rec, prec
    return {
        "froc_points": froc_pts, "luna_sens": luna_sens,
        "cpm": round(cpm, 4), "best_f1": round(best_f1, 4),
        "best_thr": round(best_thr, 4), "best_sensitivity": round(best_sens, 4),
        "best_precision": round(best_prec, 4),
        "n_gt_total": n_gt_total, "n_scans": n_scans,
        "n_tp": len(all_tp_scores), "n_fp": len(all_fp_scores),
    }


def bootstrap_cpm_ci(
    all_tp_scores: List[float],
    all_fp_scores: List[float],
    n_gt_total: int,
    n_scans: int,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired bootstrap 95% CI on CPM (1000 resamples)."""
    rng = np.random.default_rng(seed)
    items = [(s, True) for s in all_tp_scores] + [(s, False) for s in all_fp_scores]
    n = len(items)
    if n == 0:
        return {"cpm_ci_lo": 0.0, "cpm_ci_hi": 0.0}
    boots = []
    for _ in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        b_tp = [items[i][0] for i in idxs if items[i][1]]
        b_fp = [items[i][0] for i in idxs if not items[i][1]]
        boots.append(compute_froc_metrics(b_tp, b_fp, n_gt_total, n_scans)["cpm"])
    return {
        "cpm_ci_lo": round(float(np.percentile(boots, 2.5)), 4),
        "cpm_ci_hi": round(float(np.percentile(boots, 97.5)), 4),
    }


def stratified_sensitivity(per_scan_results: List[dict], best_thr: float) -> dict:
    """Sensitivity per diameter stratum at best_thr: 4-6 mm, 6-15 mm, >15 mm."""
    strata = {
        "4_6mm":  {"tp": 0, "fn": 0},
        "6_15mm": {"tp": 0, "fn": 0},
        "gt15mm": {"tp": 0, "fn": 0},
    }
    for scan in per_scan_results:
        gt_tp_set = {
            mp["matched_gt_idx"]
            for mp in scan["matched_preds"]
            if mp["is_tp"] and mp["max_prob"] >= best_thr
        }
        for gi, g in enumerate(scan["gt_list"]):
            d = g["diameter_mm"]
            if d < 4.0:
                continue
            key = "4_6mm" if d < 6.0 else ("gt15mm" if d > 15.0 else "6_15mm")
            strata[key]["tp" if gi in gt_tp_set else "fn"] += 1
    return {
        k: {
            "sensitivity": round(v["tp"] / max(v["tp"] + v["fn"], 1), 4),
            "tp": v["tp"], "fn": v["fn"], "total_gt": v["tp"] + v["fn"],
        }
        for k, v in strata.items()
    }


# ---------------------------------------------------------------------------
# MONAI inference adapter (bbox -> centroid format)
# ---------------------------------------------------------------------------

def run_monai_inference(predict_fn, vol_zyx: np.ndarray, spacing_zyx: tuple) -> List[dict]:
    """Wrap predict_nodules_monai; convert MONAI bbox output to blob-dict format.

    MONAI RetinaNet returns 3D bounding boxes.
    centroid_zyx_voxel already computed in predict_monai.py from bbox centre.
    confidence used as max_prob for FROC ranking.
    """
    nodules = predict_fn(vol_zyx, spacing_zyx)
    return [
        {
            "centroid_zyx": np.array(n["centroid_zyx_voxel"]),
            "max_prob": float(n["confidence"]),
            "n_voxels": int(n.get("voxels", 1)),
        }
        for n in nodules
    ]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Score a model on LUNA16 held-out subset."
    )
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Checkpoint (.pt). Required for finetune/lidc.")
    ap.add_argument("--model-type", choices=["finetune", "monai", "lidc"],
                    default="finetune")
    ap.add_argument("--luna16-dir", type=Path,
                    default=Path("/workspace/datasetLuna16"))
    ap.add_argument("--annotations", type=Path, default=None,
                    help="annotations.csv path (default: <luna16-dir>/annotations.csv)")
    ap.add_argument("--subset", type=int, default=9)
    ap.add_argument("--n-scans", type=int, default=0,
                    help="Limit to first N scans (0=all, 5 for smoke test)")
    ap.add_argument("--prob-thresh", type=float, default=PROB_THRESHOLD)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--patch-hw", type=int, default=PATCH_HW)
    ap.add_argument("--n-slices", type=int, default=PATCH_SLICES)
    ap.add_argument("--stride-hw", type=int, default=STRIDE_HW)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="Skip bootstrap CI (faster for smoke test)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[scorer] device={device}  model_type={args.model_type}")

    luna16_dir = args.luna16_dir
    subset_dir = luna16_dir / f"subset{args.subset}"
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset dir not found: {subset_dir}")
    annotations_csv = args.annotations or (luna16_dir / "annotations.csv")
    if not annotations_csv.exists():
        raise FileNotFoundError(f"annotations.csv not found: {annotations_csv}")

    all_annotations = parse_annotations(annotations_csv)
    mhd_files = sorted(subset_dir.glob("*.mhd"))
    if args.n_scans > 0:
        mhd_files = mhd_files[:args.n_scans]
    print(f"[scorer] subset{args.subset}: {len(mhd_files)} scans")
    scope_uids = {f.stem for f in mhd_files}
    n_gt_scope = sum(len(v) for uid, v in all_annotations.items() if uid in scope_uids)
    print(f"[scorer] GT nodules in scope: {n_gt_scope}")

    monai_fn = None
    if args.model_type == "finetune":
        if args.ckpt is None:
            raise ValueError("--ckpt required for --model-type finetune")
        model = load_finetune_model(args.ckpt, device, in_channels=args.n_slices)
    elif args.model_type == "lidc":
        if args.ckpt is None:
            raise ValueError("--ckpt required for --model-type lidc")
        model = load_lidc_model(args.ckpt, device)
    else:
        model = None
        monai_fn = load_monai_predict_fn()
        print("[scorer] MONAI bundle loaded.")

    all_tp_scores: List[float] = []
    all_fp_scores: List[float] = []
    per_scan_results: List[dict] = []
    n_gt_total = 0

    t0 = time.time()
    for scan_idx, mhd_path in enumerate(mhd_files):
        uid = mhd_path.stem
        gt_list = all_annotations.get(uid, [])
        n_gt_total += len(gt_list)
        print(f"  [{scan_idx+1}/{len(mhd_files)}] {uid[:55]}  GT={len(gt_list)}",
              flush=True)

        vol_zyx, spacing_zyx, origin_zyx = load_scan(mhd_path)

        if args.model_type in ("finetune", "lidc"):
            vol_norm = normalize_hu(vol_zyx, lo=HU_LO, hi=HU_HI)
            prob_vol = run_inference_25d(
                model, vol_norm, device,
                patch_hw=args.patch_hw, n_slices=args.n_slices,
                stride_hw=args.stride_hw,
            )
            candidates = detect_blobs(prob_vol, threshold=args.prob_thresh)
        else:
            candidates = run_monai_inference(monai_fn,
                                             vol_zyx.astype(np.float32),
                                             spacing_zyx)

        print(f"       candidates={len(candidates)}", flush=True)

        if len(candidates) == 0:
            matched_preds, gt_matched = [], [False] * len(gt_list)
        else:
            matched_preds, gt_matched = match_preds_to_gt(
                candidates, gt_list, spacing_zyx, origin_zyx
            )

        for mp in matched_preds:
            (all_tp_scores if mp["is_tp"] else all_fp_scores).append(mp["max_prob"])

        per_scan_results.append({
            "uid": uid, "gt_list": gt_list,
            "matched_preds": matched_preds, "gt_matched": gt_matched,
            "spacing_zyx": spacing_zyx, "n_candidates": len(candidates),
        })

    elapsed = time.time() - t0
    print(f"\n[scorer] Done: {elapsed:.0f}s ({elapsed/max(len(mhd_files),1):.1f}s/scan)")
    print(f"[scorer] GT={n_gt_total}  TP={len(all_tp_scores)}  FP={len(all_fp_scores)}")

    froc = compute_froc_metrics(all_tp_scores, all_fp_scores, n_gt_total, len(mhd_files))
    print(f"\n=== FROC Results (LUNA16 subset {args.subset}, {len(mhd_files)} scans) ===")
    print(f"  CPM        : {froc['cpm']:.4f}")
    print(f"  Best F1    : {froc['best_f1']:.4f}  (thr={froc['best_thr']:.3f})")
    print(f"  Sensitivity: {froc['best_sensitivity']:.4f}")
    print(f"  Precision  : {froc['best_precision']:.4f}")
    print(f"  FP/scan    : {froc['n_fp']/max(len(mhd_files),1):.2f}")
    print("\n  Sensitivity @ LUNA16 FP rates:")
    for fp_r in LUNA16_FP_RATES:
        print(f"    {fp_r:5.3f} FP/scan -> sens={froc['luna_sens'][fp_r]:.4f}")

    ci: dict = {}
    if not args.no_bootstrap and (len(all_tp_scores) + len(all_fp_scores)) > 0:
        print(f"\n[scorer] Bootstrap CI ({BOOTSTRAP_N} resamples)...", flush=True)
        ci = bootstrap_cpm_ci(all_tp_scores, all_fp_scores, n_gt_total, len(mhd_files))
        print(f"  CPM 95% CI: [{ci['cpm_ci_lo']:.4f}, {ci['cpm_ci_hi']:.4f}]")

    strat = stratified_sensitivity(per_scan_results, froc["best_thr"])
    print(f"\n  Stratified sensitivity @ thr={froc['best_thr']:.3f}:")
    for k, v in strat.items():
        print(f"    {k:8s}: sens={v['sensitivity']:.4f}  ({v['tp']}/{v['total_gt']})")

    if args.n_scans > 0:
        print(f"\n[SMOKE] n_scans={args.n_scans}")
        total_cands = sum(r["n_candidates"] for r in per_scan_results)
        print(f"[SMOKE] Total candidates: {total_cands}")
        if per_scan_results and per_scan_results[0]["matched_preds"]:
            mp0 = per_scan_results[0]["matched_preds"][0]
            print(f"[SMOKE] First pred ZYX={mp0['centroid_zyx']}  prob={mp0['max_prob']:.4f}")
        print(f"[SMOKE] TP={len(all_tp_scores)}  FP={len(all_fp_scores)}")

    output = {
        "meta": {
            "model_type": args.model_type,
            "ckpt": str(args.ckpt) if args.ckpt else None,
            "luna16_dir": str(luna16_dir),
            "subset": args.subset,
            "n_scans": len(mhd_files),
            "n_gt_total": n_gt_total,
            "prob_thresh": args.prob_thresh,
            "patch_hw": args.patch_hw,
            "n_slices": args.n_slices,
            "hu_lo": HU_LO, "hu_hi": HU_HI,
            "matching_rule": "LUNA16_official_pred_within_gt_diameter_half",
            "elapsed_s": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "froc": froc, "bootstrap_ci": ci, "stratified": strat,
        "per_scan": [
            {
                "uid": r["uid"],
                "n_gt": len(r["gt_list"]),
                "n_candidates": r["n_candidates"],
                "n_tp": sum(1 for mp in r["matched_preds"] if mp["is_tp"]),
                "n_fp": sum(1 for mp in r["matched_preds"] if not mp["is_tp"]),
                "gt_matched": r["gt_matched"],
                "predictions": [
                    {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in mp.items()}
                    for mp in r["matched_preds"]
                ],
            }
            for r in per_scan_results
        ],
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2))
        print(f"\n[scorer] Results saved to {args.output_json}")
    else:
        print("\n[scorer] --output-json not set; results not persisted.")

    return output


if __name__ == "__main__":
    main()
