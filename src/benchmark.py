"""Benchmark AI pipeline against LIDC ground truth on a panel of patients.

For each patient in the panel:
  1. Load HU volume + GT mask from work/preprocessed/<pid>.h5
  2. Run lungmask + UNet++ + post-processing with given parameters
  3. Match predicted nodules to GT nodules by 3D centroid distance
  4. Compute precision, recall, F1, size error

Output: JSON metrics aggregated across panel.

Use directly:
  python benchmark.py --threshold 0.65 --min-voxels 200 --max-elong 4 \
    --merge-dist 10 --subpleural-dist 2 --ensemble 1

Or via tuner.py for grid search.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

# Reuse webapp predict logic
sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from predict import (  # noqa: E402
    clean_mask,
    filter_subpleural,
    find_nodules,
    get_malignancy_model,
    get_model,
    get_swa_model,
    merge_nearby_nodules,
    normalize,
    predict_nodules,
    segment_lung,
)
from configs import PRE_DIR  # noqa: E402


# Diverse panel covering small, large, multi-nodule, screening cases
DEFAULT_PANEL = [
    "LIDC-IDRI-0001",  # small
    "LIDC-IDRI-0002",  # 1 large 17.5mm mal=4.5
    "LIDC-IDRI-0003",
    "LIDC-IDRI-0094",
    "LIDC-IDRI-0220",  # multiple
    "LIDC-IDRI-0303",  # 28mm
    "LIDC-IDRI-0447",
    "LIDC-IDRI-0595",
    "LIDC-IDRI-0651",
    "LIDC-IDRI-0751",  # 27mm + 25mm
    "LIDC-IDRI-0936",  # 17mm
    "LIDC-IDRI-0940",
]


def gt_nodules_for(h5_path: Path, voxel_sp: tuple, merge_dist_mm: float = 12.0) -> list:
    """Extract clinically-relevant GT nodules (polygon ≥ 3mm) from preprocessed h5.

    Merges across radiologists by physical centroid distance.
    Returns list of dict with: diam_mm, voxels, vol_mm3, centroid_zyx_voxel, n_radiologists,
    malignancy_mean (or None).
    """
    out = []
    with h5py.File(h5_path, "r") as f:
        for series_key in f.keys():
            g = f[series_key]
            masks = g["mask_per_rad"][:]
            nodule_meta = json.loads(g.attrs["nodule_meta"])
            voxel_vol = float(np.prod(voxel_sp))

            by = defaultdict(list)
            for e in nodule_meta:
                by[(e["rad_channel"], e["nodule_id"])].append(e)

            per_rad = []
            for (rad_ch, nod_id), entries in by.items():
                slice_idxs = [int(e["slice_idx"]) for e in entries
                              if 0 <= int(e["slice_idx"]) < masks.shape[0]]
                if not slice_idxs: continue
                sub = masks[slice_idxs, rad_ch]
                if sub.sum() < 5: continue
                coords = np.argwhere(sub > 0)
                if len(coords) == 0: continue
                cz_local = int(coords[:, 0].mean())
                cz = slice_idxs[min(cz_local, len(slice_idxs) - 1)]
                cy = int(coords[:, 1].mean())
                cx = int(coords[:, 2].mean())
                voxels = int(sub.sum())
                vol_mm3 = voxels * voxel_vol
                diam = 2 * (3 * vol_mm3 / (4 * np.pi)) ** (1 / 3)
                mal = entries[0].get("malignancy", "")
                per_rad.append({
                    "rad": rad_ch, "nid": nod_id,
                    "centroid_zyx_voxel": [cz, cy, cx],
                    "voxels": voxels, "vol_mm3": float(vol_mm3),
                    "diam_mm": float(diam),
                    "malignancy": int(mal) if str(mal).isdigit() else None,
                })

            # Merge across radiologists by physical centroid distance
            per_rad.sort(key=lambda n: -n["vol_mm3"])
            used = [False] * len(per_rad)
            for i, n in enumerate(per_rad):
                if used[i]: continue
                cluster = [n]; used[i] = True
                ca = np.array(n["centroid_zyx_voxel"]) * np.array(voxel_sp)
                for j in range(i + 1, len(per_rad)):
                    if used[j]: continue
                    m = per_rad[j]
                    cb = np.array(m["centroid_zyx_voxel"]) * np.array(voxel_sp)
                    if np.linalg.norm(ca - cb) <= merge_dist_mm:
                        cluster.append(m); used[j] = True
                rads = sorted(set(c["rad"] for c in cluster))
                mals = [c["malignancy"] for c in cluster if c["malignancy"] is not None]
                diams = [c["diam_mm"] for c in cluster]
                out.append({
                    "diam_mm": float(np.mean(diams)),
                    "diam_mm_max": float(np.max(diams)),
                    "centroid_zyx_voxel": n["centroid_zyx_voxel"],
                    "n_radiologists": len(rads),
                    "malignancy_mean": float(np.mean(mals)) if mals else None,
                })
    return out


def load_patient(pid: str) -> dict:
    """Load HU volume, voxel spacing, GT nodules from preprocessed h5."""
    h5p = PRE_DIR / f"{pid}.h5"
    if not h5p.exists():
        return None
    with h5py.File(h5p, "r") as f:
        # Take first series (usually only 1 per patient in LIDC)
        series_key = next(iter(f.keys()))
        g = f[series_key]
        images = g["images"][:].astype(np.int16)
        pixel_sp = json.loads(g.attrs["pixel_spacing"])
        slice_thk = float(g.attrs["slice_thickness"])
        voxel_sp = (slice_thk, pixel_sp[0], pixel_sp[1])
    gt = gt_nodules_for(h5p, voxel_sp)
    return {"pid": pid, "vol": images, "voxel_sp": voxel_sp, "gt_nodules": gt}


def run_pipeline(vol: np.ndarray, voxel_sp: tuple, params: dict) -> list:
    """Run AI pipeline with given params. Returns list of predicted nodules."""
    # Lung segmentation
    lung = segment_lung(vol)

    # Predict
    _prob, pred = predict_nodules(
        vol,
        threshold=params["threshold"],
        tta=params.get("tta", True),
        ensemble=params.get("ensemble", True),
    )

    # Filters
    pred = (pred & lung).astype(np.uint8)
    if params.get("clean_mask", True):
        pred = clean_mask(pred, voxel_sp)

    nodules, labeled = find_nodules(
        pred, voxel_sp,
        min_voxels=params["min_voxels"],
        max_elongation=params["max_elong"],
    )
    if params["merge_dist"] > 0:
        nodules = merge_nearby_nodules(nodules, voxel_sp, max_dist_mm=params["merge_dist"])
    if params["subpleural_dist"] > 0:
        nodules = filter_subpleural(nodules, lung, voxel_sp, min_dist_mm=params["subpleural_dist"])
    return nodules


def match_nodules(pred: list, gt: list, voxel_sp: tuple, max_dist_mm: float = 15.0) -> dict:
    """Match each GT nodule to nearest predicted nodule.

    Returns: tp, fp, fn, precision, recall, f1, size_errors.
    """
    if not gt and not pred:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "size_errors": [], "matches": []}
    matched_pred = set()
    tp = 0
    matches = []
    size_errors = []
    for g in gt:
        gc = np.array(g["centroid_zyx_voxel"]) * np.array(voxel_sp)
        best_idx, best_dist = None, float("inf")
        for i, p in enumerate(pred):
            if i in matched_pred: continue
            pc = np.array(p["centroid_zyx_voxel"]) * np.array(voxel_sp)
            d = np.linalg.norm(gc - pc)
            if d < best_dist:
                best_dist, best_idx = d, i
        if best_idx is not None and best_dist <= max_dist_mm:
            tp += 1
            matched_pred.add(best_idx)
            p = pred[best_idx]
            matches.append({
                "gt_diam": g["diam_mm"], "pred_diam": p["diameter_mm"],
                "dist_mm": float(best_dist), "gt_mal": g.get("malignancy_mean"),
            })
            if g["diam_mm"] > 0:
                size_errors.append(abs(p["diameter_mm"] - g["diam_mm"]) / g["diam_mm"])
    fp = len(pred) - tp
    fn = len(gt) - tp
    prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 1.0
    rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 1.0
    f1 = 2 * prec * rec / max(prec + rec, 1e-7) if (prec + rec) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "size_errors": size_errors, "matches": matches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--min-voxels", type=int, default=200)
    ap.add_argument("--max-elong", type=float, default=4.0)
    ap.add_argument("--merge-dist", type=float, default=10.0)
    ap.add_argument("--subpleural-dist", type=float, default=2.0)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--clean-mask", type=int, default=1)
    ap.add_argument("--match-dist", type=float, default=15.0,
                    help="Max centroid distance (mm) for AI↔GT matching")
    ap.add_argument("--patients", default=",".join(DEFAULT_PANEL))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    params = {
        "threshold": args.threshold, "min_voxels": args.min_voxels,
        "max_elong": args.max_elong, "merge_dist": args.merge_dist,
        "subpleural_dist": args.subpleural_dist,
        "ensemble": bool(args.ensemble), "clean_mask": bool(args.clean_mask),
        "tta": True,
    }

    # Warm up models
    print(f"Params: {params}", flush=True)
    get_model(); get_swa_model(); get_malignancy_model()
    pids = [p.strip() for p in args.patients.split(",") if p.strip()]
    per_patient = []

    t_total = time.time()
    for pid in pids:
        t0 = time.time()
        data = load_patient(pid)
        if data is None:
            print(f"  [SKIP] {pid}: no h5", flush=True)
            continue
        try:
            pred_nodules = run_pipeline(data["vol"], data["voxel_sp"], params)
        except Exception as e:
            print(f"  [ERR] {pid}: {type(e).__name__}: {e}", flush=True)
            continue
        match = match_nodules(pred_nodules, data["gt_nodules"], data["voxel_sp"],
                              max_dist_mm=args.match_dist)
        dt = time.time() - t0
        per_patient.append({
            "pid": pid, "n_gt": len(data["gt_nodules"]), "n_pred": len(pred_nodules),
            **{k: match[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
            "mean_size_err": float(np.mean(match["size_errors"])) if match["size_errors"] else None,
            "time_s": float(dt),
        })
        print(f"  [{pid}] gt={len(data['gt_nodules']):2} pred={len(pred_nodules):2} "
              f"tp={match['tp']} fp={match['fp']} fn={match['fn']} "
              f"P={match['precision']:.2f} R={match['recall']:.2f} F1={match['f1']:.2f} "
              f"({dt:.1f}s)", flush=True)

    # Aggregate
    tp_t = sum(p["tp"] for p in per_patient)
    fp_t = sum(p["fp"] for p in per_patient)
    fn_t = sum(p["fn"] for p in per_patient)
    micro_p = tp_t / max(tp_t + fp_t, 1)
    micro_r = tp_t / max(tp_t + fn_t, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-7)
    macro_f1 = float(np.mean([p["f1"] for p in per_patient])) if per_patient else 0.0
    size_errs = [e for p in per_patient if p["mean_size_err"] is not None for e in [p["mean_size_err"]]]
    mean_size_err = float(np.mean(size_errs)) if size_errs else None
    total_time = time.time() - t_total

    result = {
        "params": params,
        "n_patients": len(per_patient),
        "tp_total": tp_t, "fp_total": fp_t, "fn_total": fn_t,
        "precision_micro": float(micro_p),
        "recall_micro": float(micro_r),
        "f1_micro": float(micro_f1),
        "f1_macro": macro_f1,
        "mean_size_err": mean_size_err,
        "total_time_s": float(total_time),
        "per_patient": per_patient,
    }

    print()
    print(f"=== AGGREGATE ===")
    print(f"  Patients:         {len(per_patient)}/{len(pids)}")
    print(f"  Precision (micro):{micro_p:.3f}")
    print(f"  Recall (micro):   {micro_r:.3f}")
    print(f"  F1 (micro):       {micro_f1:.3f}")
    print(f"  F1 (macro):       {macro_f1:.3f}")
    if mean_size_err is not None:
        print(f"  Mean size err:    {mean_size_err*100:.1f}% (lower better)")
    print(f"  Total time:       {total_time:.0f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
