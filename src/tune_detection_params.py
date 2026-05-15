"""Sweep post-processing params on cached probability volumes.

Usage:
  # 1) Cache prob volumes for the panel (slow, GPU)
  python tune_detection_params.py --panel val --cache-only

  # 2) Sweep grid (fast, CPU-only on cached probs)
  python tune_detection_params.py --panel val --quick     # small grid for smoke test
  python tune_detection_params.py --panel val             # full grid

Output:
  work/academic/tuning_results.csv     — one row per (post-proc config × FROC point)
  work/academic/tuning_summary.csv     — one row per post-proc config (CPM + FP/scan)
  work/academic/best_config.json       — selected config

Selection rule (verbatim from user plan):
  primary  : max CPM
  tie-break: max sensitivity @ 1 FP/scan
  tie-break: min FP/scan at default threshold
  tie-break: better sens for nodules >= 6mm
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import (
    binary_closing, binary_dilation, binary_erosion,
    distance_transform_edt, label as cc_label,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from configs import WORK, PRE_DIR  # noqa: E402
from panels import load_panel  # noqa: E402
from benchmark import gt_nodules_for, load_patient  # noqa: E402

ACADEMIC_DIR = WORK / "academic"
PROBS_DIR = ACADEMIC_DIR / "probs"
LUNG_DIR = ACADEMIC_DIR / "lung_masks"

LUNA16_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


# ------------------------ Cache helpers ------------------------

def cache_prob_volume(pid: str, tta: bool = True) -> Path:
    """Cache TTA-averaged probability volume for one patient. Returns cache path."""
    cache_path = PROBS_DIR / f"{pid}_tta{int(tta)}.npy"
    if cache_path.exists():
        return cache_path
    # Lazy import (heavy GPU bring-up)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
    from predict import predict_nodules, get_model, get_swa_model
    get_model(); get_swa_model()
    data = load_patient(pid)
    if data is None:
        return None
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    prob, _ = predict_nodules(data["vol"], threshold=0.5, tta=tta, ensemble=True)
    np.save(cache_path, prob.astype(np.float16))
    return cache_path


def cache_lung_mask(pid: str, vol: np.ndarray) -> np.ndarray:
    """Cache lung mask. Returns uint8 [N,H,W]."""
    cache_path = LUNG_DIR / f"{pid}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
    from predict import segment_lung
    LUNG_DIR.mkdir(parents=True, exist_ok=True)
    lung = segment_lung(vol).astype(np.uint8)
    np.save(cache_path, lung)
    return lung


# ------------------------ Patient bundle (precomputed per threshold) ------------------------

def precompute_threshold_blobs(prob_vol: np.ndarray, lung_mask: np.ndarray,
                                voxel_sp: tuple, threshold: float,
                                clean: bool = True,
                                core_threshold: float = 0.85) -> list:
    """For one (patient, threshold), CC-label and return blob list.

    Each blob dict: id, voxels, coords (Nx3 int), centroid_mm (3,), elong, bbox.
    Used by the sweep loop to filter without re-labeling.

    Diameter is reported using the core-threshold (prob > core_threshold) high-
    confidence interior — closer to clinical measurement than full-mask sphere.
    """
    pred = (prob_vol > threshold).astype(np.uint8)
    pred = pred & lung_mask
    if clean:
        st = np.ones((1, 3, 3), dtype=np.uint8)
        b = binary_closing(pred.astype(bool), structure=st, iterations=1)
        b = binary_erosion(b, structure=st, iterations=1)
        b = binary_dilation(b, structure=st, iterations=1)
        pred = b.astype(np.uint8)
    if pred.sum() == 0:
        return []
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    lab, n = cc_label(pred, structure=structure)
    if n == 0:
        return []
    sp = np.array(voxel_sp, dtype=np.float32)
    blobs = []
    flat = lab.ravel()
    # Compute coords once via argwhere per blob (faster than np.where on 3D for many blobs)
    for bi in range(1, n + 1):
        coords = np.argwhere(lab == bi)
        if len(coords) < 8:
            continue  # tiny artefact; even smallest min_voxels (20) will reject
        physical = coords.astype(np.float32) * sp
        centered = physical - physical.mean(axis=0)
        # Elongation via PCA
        try:
            cov = np.cov(centered.T)
            eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
            if eig[2] < 1e-6:
                elong = 99.0
            else:
                elong = float(np.sqrt(max(eig[0], 0) / max(eig[2], 1e-6)))
        except np.linalg.LinAlgError:
            elong = 99.0
        centroid_mm = (coords.mean(axis=0) * sp).astype(np.float32)
        bmin = coords.min(axis=0)
        bmax = coords.max(axis=0) + 1
        probs_inside = prob_vol[coords[:, 0], coords[:, 1], coords[:, 2]]
        conf = float(probs_inside.mean())
        # Tight diameter from CORE (prob > core_threshold), like predict.find_nodules
        core_voxels = int((probs_inside > core_threshold).sum())
        vol_full = float(len(coords)) * float(np.prod(voxel_sp))
        diam_full = 2 * (3 * vol_full / (4 * np.pi)) ** (1 / 3)
        if core_voxels >= 8:
            vol_core = float(core_voxels) * float(np.prod(voxel_sp))
            diam = 2 * (3 * vol_core / (4 * np.pi)) ** (1 / 3)
        else:
            diam = diam_full * 0.7
        blobs.append({
            "id": bi,
            "voxels": int(len(coords)),
            "centroid_mm": centroid_mm,            # (3,) physical mm
            "centroid_voxel": coords.mean(axis=0), # (3,) voxel
            "elongation": elong,
            "bbox": list(bmin) + list(bmax),
            "confidence": conf,
            "diameter_mm": float(diam),
            "diameter_full_mm": float(diam_full),
            "core_voxels": core_voxels,
        })
    return blobs


def filter_and_match(blobs: list, lung_dist_mm: np.ndarray, voxel_sp: tuple,
                     gt_nodules: list, *,
                     min_voxels: int, max_elong: float, merge_dist_mm: float,
                     subpleural_min_mm: float,
                     match_min_radius_mm: float = 5.0):
    """Apply post-proc filters to precomputed blobs, then match to GT.

    lung_dist_mm: precomputed distance transform of lung (mm), 3D array.
    """
    sp = np.array(voxel_sp, dtype=np.float32)
    # Filter by voxels + elongation
    cands = [b for b in blobs if b["voxels"] >= min_voxels and b["elongation"] <= max_elong]
    # Merge nearby (sort by size desc)
    cands = sorted(cands, key=lambda b: -b["voxels"])
    used = [False] * len(cands)
    merged = []
    for i, a in enumerate(cands):
        if used[i]: continue
        used[i] = True
        cluster = [a]
        ca = a["centroid_mm"]
        for j in range(i + 1, len(cands)):
            if used[j]: continue
            cb = cands[j]["centroid_mm"]
            if np.linalg.norm(ca - cb) <= merge_dist_mm:
                cluster.append(cands[j]); used[j] = True
        if len(cluster) == 1:
            merged.append(a)
        else:
            total_vox = sum(c["voxels"] for c in cluster)
            total_core = sum(c.get("core_voxels", 0) for c in cluster)
            new_centroid = a["centroid_mm"]
            new_centroid_voxel = a["centroid_voxel"]
            vol_full = float(total_vox) * float(np.prod(voxel_sp))
            diam_full = 2 * (3 * vol_full / (4 * np.pi)) ** (1 / 3)
            if total_core >= 8:
                vol_core = float(total_core) * float(np.prod(voxel_sp))
                diam = 2 * (3 * vol_core / (4 * np.pi)) ** (1 / 3)
            else:
                diam = diam_full * 0.7
            merged.append({
                **a, "voxels": total_vox, "core_voxels": total_core,
                "diameter_mm": float(diam),
                "diameter_full_mm": float(diam_full),
                "centroid_mm": new_centroid, "centroid_voxel": new_centroid_voxel,
            })
    # Subpleural filter
    if subpleural_min_mm > 0:
        kept = []
        for b in merged:
            cz, cy, cx = [int(round(v)) for v in b["centroid_voxel"]]
            cz = max(0, min(lung_dist_mm.shape[0] - 1, cz))
            cy = max(0, min(lung_dist_mm.shape[1] - 1, cy))
            cx = max(0, min(lung_dist_mm.shape[2] - 1, cx))
            if lung_dist_mm[cz, cy, cx] >= subpleural_min_mm:
                kept.append(b)
        preds = kept
    else:
        preds = merged
    # Match to GT (LUNA16 radius rule)
    matched = []
    used_p = set()
    for gi, g in enumerate(gt_nodules):
        gc = np.array(g["centroid_zyx_voxel"], dtype=np.float32) * sp
        tol = max(g["diam_mm"] / 2.0, match_min_radius_mm)
        best, bestd = None, float("inf")
        for pi, p in enumerate(preds):
            if pi in used_p: continue
            d = np.linalg.norm(gc - p["centroid_mm"])
            if d < bestd:
                bestd, best = d, pi
        if best is not None and bestd <= tol:
            matched.append((gi, best))
            used_p.add(best)
    n_pred = len(preds)
    tp = len(matched)
    fp = n_pred - tp
    fn = len(gt_nodules) - tp
    # Stratified by GT diameter
    matched_gi = {m[0] for m in matched}
    strat = {"sm_4_6": [0, 0], "md_6_15": [0, 0], "lg_15p": [0, 0]}  # [tp, n_gt]
    for gi, g in enumerate(gt_nodules):
        d = g["diam_mm"]
        if 4 <= d < 6: bucket = "sm_4_6"
        elif 6 <= d < 15: bucket = "md_6_15"
        elif d >= 15: bucket = "lg_15p"
        else: continue
        strat[bucket][1] += 1
        if gi in matched_gi: strat[bucket][0] += 1
    return {
        "tp": tp, "fp": fp, "fn": fn, "n_pred": n_pred,
        "n_gt": len(gt_nodules), "preds": preds, "matched": matched,
        "strat": strat,
    }


# ------------------------ Main sweep ------------------------

def load_patient_bundle(pid: str, force_cache_now: bool = False) -> dict:
    """Load prob_vol + lung_mask + lung_dist + GT for one patient, using caches."""
    cache_path = PROBS_DIR / f"{pid}_tta1.npy"
    if not cache_path.exists():
        if not force_cache_now:
            return None  # caller must run --cache-only first
        cache_prob_volume(pid, tta=True)
    prob = np.load(cache_path).astype(np.float32)
    data = load_patient(pid)
    if data is None:
        return None
    lung = cache_lung_mask(pid, data["vol"])
    # Precompute distance transform once per patient (slowest single op besides CC)
    sp = data["voxel_sp"]
    lung_dist = distance_transform_edt(lung, sampling=list(sp)).astype(np.float32)
    return {
        "pid": pid, "prob": prob, "lung": lung, "lung_dist": lung_dist,
        "voxel_sp": sp, "gt": data["gt_nodules"], "vol_shape": data["vol"].shape,
    }


def compute_cpm_at_config(patients: list, config: dict, thresholds: list,
                           blob_cache: dict) -> dict:
    """For one post-proc config, sweep thresholds and compute FROC + CPM.

    blob_cache[(pid, thr)] -> precomputed blob list (so we don't re-CC label).
    """
    points = []
    strat_at_default = {"sm_4_6": [0, 0], "md_6_15": [0, 0], "lg_15p": [0, 0]}
    for thr in thresholds:
        tp = fp = fn = n_gt = 0
        for pat in patients:
            blobs = blob_cache[(pat["pid"], thr)]
            res = filter_and_match(
                blobs, pat["lung_dist"], pat["voxel_sp"], pat["gt"],
                min_voxels=config["min_voxels"],
                max_elong=config["max_elong"],
                merge_dist_mm=config["merge_dist_mm"],
                subpleural_min_mm=config["subpleural_min_mm"],
            )
            tp += res["tp"]; fp += res["fp"]; fn += res["fn"]; n_gt += res["n_gt"]
            # accumulate stratified at default (closest to 0.40)
            if abs(thr - config.get("default_thr", 0.40)) < 1e-6:
                for k in strat_at_default:
                    strat_at_default[k][0] += res["strat"][k][0]
                    strat_at_default[k][1] += res["strat"][k][1]
        sens = tp / max(n_gt, 1)
        fpps = fp / max(len(patients), 1)
        points.append({"threshold": thr, "sensitivity": sens, "fp_per_scan": fpps,
                        "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt})
    pts_sorted = sorted(points, key=lambda p: p["fp_per_scan"])
    fp_arr = np.array([p["fp_per_scan"] for p in pts_sorted])
    sens_arr = np.array([p["sensitivity"] for p in pts_sorted])
    luna_sens = []
    for fp_target in LUNA16_FP_RATES:
        if fp_target <= fp_arr.min():
            s = float(sens_arr[0])
        elif fp_target >= fp_arr.max():
            s = float(sens_arr[-1])
        else:
            s = float(np.interp(fp_target, fp_arr, sens_arr))
        luna_sens.append({"fp_per_scan": fp_target, "sensitivity": s})
    cpm = float(np.mean([s["sensitivity"] for s in luna_sens]))
    sens_at_1 = next(s["sensitivity"] for s in luna_sens if s["fp_per_scan"] == 1.0)
    sens_at_2 = next(s["sensitivity"] for s in luna_sens if s["fp_per_scan"] == 2.0)
    sens_at_4 = next(s["sensitivity"] for s in luna_sens if s["fp_per_scan"] == 4.0)
    return {
        "config": config, "points": points, "luna_sens": luna_sens,
        "cpm": cpm, "sens_at_1fp": sens_at_1, "sens_at_2fp": sens_at_2,
        "sens_at_4fp": sens_at_4, "stratified": strat_at_default,
    }


def best_f1_point(points: list) -> dict:
    """Return the point with best F1 = 2*P*R/(P+R), where P = TP/(TP+FP)."""
    best, best_f1 = None, -1.0
    for p in points:
        tp = p["tp"]; fp = p["fp"]; fn = p["fn"]
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = p["sensitivity"]
        f1 = 2 * prec * rec / max(prec + rec, 1e-7) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best = {**p, "precision": prec, "f1": f1}
    return best


def select_best(results: list) -> dict:
    """Pick the config whose best-F1 operating point is highest.

    Rationale: user wants AI results that match GT most closely, which is exactly
    F1 (jointly maximises precision AND recall). CPM-only selection picks
    permissive configs that never hit clinically usable FP/scan rates.

    Tie-breakers: max CPM, max sens@1FP if reachable, then better mid+large
    nodule sensitivity.
    """
    def key(r):
        bp = best_f1_point(r["points"])
        sm = r["stratified"]["md_6_15"][0] / max(r["stratified"]["md_6_15"][1], 1) \
             + r["stratified"]["lg_15p"][0] / max(r["stratified"]["lg_15p"][1], 1)
        return (
            -bp["f1"],       # max best-F1 across thresholds
            -r["cpm"],       # tie-break: max CPM
            -bp["sensitivity"],
            -sm,
        )
    results_sorted = sorted(results, key=key)
    chosen = results_sorted[0]
    chosen["best_f1_point"] = best_f1_point(chosen["points"])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["debug", "val", "test"], default="debug")
    ap.add_argument("--cache-only", action="store_true",
                    help="Just cache prob volumes, don't sweep")
    ap.add_argument("--quick", action="store_true",
                    help="Smaller grid for fast smoke test")
    ap.add_argument("--max-patients", type=int, default=0,
                    help="Cap patients (0 = all in panel)")
    ap.add_argument("--out-prefix", default="tuning",
                    help="Output filename prefix in work/academic/")
    args = ap.parse_args()

    pids = load_panel(args.panel)
    if args.max_patients > 0:
        pids = pids[: args.max_patients]
    print(f"Panel '{args.panel}' — {len(pids)} patients")
    ACADEMIC_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Cache phase (only if missing) ----
    missing = [p for p in pids if not (PROBS_DIR / f"{p}_tta1.npy").exists()]
    if missing:
        print(f"\n[Cache] {len(missing)} prob volumes missing -- running TTA inference (slow)")
        for i, pid in enumerate(missing):
            t0 = time.time()
            try:
                r = cache_prob_volume(pid, tta=True)
            except Exception as e:
                print(f"  [{i+1}/{len(missing)}] {pid}: FAILED ({type(e).__name__}: {e})",
                      flush=True)
                continue
            if r is None:
                print(f"  [{i+1}/{len(missing)}] {pid}: SKIP (no preprocessed h5)")
            else:
                print(f"  [{i+1}/{len(missing)}] {pid}: cached in {time.time()-t0:.1f}s",
                      flush=True)
    if args.cache_only:
        print("\n--cache-only set; done.")
        return

    # ---- Load all patient bundles ----
    print(f"\n[Load] {len(pids)} patient bundles")
    patients = []
    for pid in pids:
        t0 = time.time()
        bundle = load_patient_bundle(pid, force_cache_now=False)
        if bundle is None:
            print(f"  {pid}: SKIP")
            continue
        patients.append(bundle)
        print(f"  {pid}: prob {bundle['prob'].shape}, gt={len(bundle['gt'])}, "
              f"lung_dist {bundle['lung_dist'].shape}  ({time.time()-t0:.1f}s)",
              flush=True)
    print(f"  Loaded {len(patients)} patients")

    # ---- Define grid ----
    if args.quick:
        # Wider threshold range to actually hit FP/scan <= 4 for realistic operating points
        thresholds = [0.20, 0.40, 0.60, 0.75, 0.85, 0.92, 0.97]
        grid = {
            "min_voxels":         [20, 50, 120],
            "max_elong":          [3.0, 4.0, 99.0],
            "merge_dist_mm":      [6, 10],
            "subpleural_min_mm":  [0, 1.5, 2.0],
        }
    else:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
        grid = {
            "min_voxels":         [20, 30, 50, 80, 120, 200],
            "max_elong":          [3.0, 4.0, 5.0, 99.0],
            "merge_dist_mm":      [6, 8, 10, 12],
            "subpleural_min_mm":  [0, 1.0, 1.5, 2.0],
        }
    n_configs = (len(grid["min_voxels"]) * len(grid["max_elong"])
                  * len(grid["merge_dist_mm"]) * len(grid["subpleural_min_mm"]))
    print(f"\n[Grid] {n_configs} configs × {len(thresholds)} thresholds = "
          f"{n_configs * len(thresholds)} ops/patient × {len(patients)} patients")

    # ---- Precompute blobs per (patient, threshold) ----
    print(f"\n[Precompute] CC labels for {len(patients)} patients × {len(thresholds)} thresholds")
    blob_cache = {}
    t_pre = time.time()
    for pi, pat in enumerate(patients):
        for thr in thresholds:
            blob_cache[(pat["pid"], thr)] = precompute_threshold_blobs(
                pat["prob"], pat["lung"], pat["voxel_sp"], thr
            )
        if (pi + 1) % max(1, len(patients) // 10) == 0:
            print(f"  {pi+1}/{len(patients)} done ({time.time()-t_pre:.0f}s)", flush=True)
    print(f"  Precompute done in {time.time()-t_pre:.0f}s")

    # ---- Sweep ----
    print(f"\n[Sweep] running {n_configs} configs")
    all_results = []
    csv_rows = []
    summary_rows = []
    t_sw = time.time()
    cfg_idx = 0
    for mv in grid["min_voxels"]:
        for me in grid["max_elong"]:
            for md in grid["merge_dist_mm"]:
                for sp_d in grid["subpleural_min_mm"]:
                    cfg = {"min_voxels": mv, "max_elong": me,
                           "merge_dist_mm": md, "subpleural_min_mm": sp_d,
                           "default_thr": 0.40}
                    res = compute_cpm_at_config(patients, cfg, thresholds, blob_cache)
                    all_results.append(res)
                    cfg_idx += 1
                    cfg_tag = f"mv{mv}_me{me}_md{md}_sp{sp_d}"
                    summary_rows.append({
                        "config": cfg_tag, **cfg, "cpm": res["cpm"],
                        "sens_at_1fp": res["sens_at_1fp"],
                        "sens_at_2fp": res["sens_at_2fp"],
                        "sens_at_4fp": res["sens_at_4fp"],
                        "sens_small_4_6mm": res["stratified"]["sm_4_6"][0]
                            / max(res["stratified"]["sm_4_6"][1], 1),
                        "sens_med_6_15mm": res["stratified"]["md_6_15"][0]
                            / max(res["stratified"]["md_6_15"][1], 1),
                        "sens_large_15p": res["stratified"]["lg_15p"][0]
                            / max(res["stratified"]["lg_15p"][1], 1),
                    })
                    for p in res["points"]:
                        csv_rows.append({"config": cfg_tag, **cfg, **p})
                    if cfg_idx % max(1, n_configs // 10) == 0:
                        print(f"  {cfg_idx}/{n_configs} configs ({time.time()-t_sw:.0f}s) "
                              f"latest CPM={res['cpm']:.3f}", flush=True)
    print(f"\nSweep done in {time.time()-t_sw:.0f}s")

    # ---- Write CSVs ----
    csv_path = ACADEMIC_DIR / f"{args.out_prefix}_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"  Wrote {csv_path} ({len(csv_rows)} rows)")
    summary_csv = ACADEMIC_DIR / f"{args.out_prefix}_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"  Wrote {summary_csv} ({len(summary_rows)} rows)")

    # ---- Select best ----
    best = select_best(all_results)
    bp = best["best_f1_point"]
    print(f"\n=== Best config (best-F1={bp['f1']:.4f} at thr={bp['threshold']:.2f}, "
          f"CPM={best['cpm']:.4f}) ===")
    print(f"  min_voxels = {best['config']['min_voxels']}")
    print(f"  max_elong  = {best['config']['max_elong']}")
    print(f"  merge_dist = {best['config']['merge_dist_mm']} mm")
    print(f"  subpleural = {best['config']['subpleural_min_mm']} mm")
    print(f"  At thr={bp['threshold']:.2f}: sens={bp['sensitivity']:.3f}  prec={bp['precision']:.3f}  "
          f"FP/scan={bp['fp_per_scan']:.2f}  TP={bp['tp']} FP={bp['fp']} FN={bp['fn']}")
    print(f"  Sens @ 1 FP/scan = {best['sens_at_1fp']:.3f}")
    print(f"  Sens @ 2 FP/scan = {best['sens_at_2fp']:.3f}")
    print(f"  Sens @ 4 FP/scan = {best['sens_at_4fp']:.3f}")
    print(f"  Stratified: small {best['stratified']['sm_4_6']}  "
          f"medium {best['stratified']['md_6_15']}  large {best['stratified']['lg_15p']}")
    # Pick three operating points: Sensitive / Balanced / Strict
    # Sensitive : highest sens with FP/scan <= 4 (or just lowest threshold)
    # Balanced  : best F1
    # Strict    : highest precision with sens >= 80% of best
    def op_at(target_fp):
        pts = sorted(best["points"], key=lambda p: abs(p["fp_per_scan"] - target_fp))
        return pts[0]
    op_balanced = bp  # best F1 across thresholds
    # Sensitive: lowest threshold (highest sens, accept higher FP)
    op_sensitive = sorted(best["points"], key=lambda p: p["threshold"])[0]
    # Strict: highest threshold (lowest FP)
    op_strict    = sorted(best["points"], key=lambda p: -p["threshold"])[0]
    best_thr = op_balanced["threshold"]
    print(f"\n  Operating points:")
    print(f"    SENSITIVE (~4 FP/scan) thr={op_sensitive['threshold']:.2f} "
          f"sens={op_sensitive['sensitivity']:.3f} FP/scan={op_sensitive['fp_per_scan']:.2f}")
    print(f"    BALANCED  (~1 FP/scan) thr={op_balanced['threshold']:.2f} "
          f"sens={op_balanced['sensitivity']:.3f} FP/scan={op_balanced['fp_per_scan']:.2f}")
    print(f"    STRICT    (~0.25 FP)   thr={op_strict['threshold']:.2f} "
          f"sens={op_strict['sensitivity']:.3f} FP/scan={op_strict['fp_per_scan']:.2f}")

    out_json = ACADEMIC_DIR / "best_config.json"
    out_json.write_text(json.dumps({
        "panel": args.panel,
        "n_patients": len(patients),
        "thresholds_swept": thresholds,
        "grid": grid,
        "best_config": best["config"],
        "best_threshold_op_at_1fp": best_thr,
        "operating_points": {
            "sensitive": op_sensitive,
            "balanced":  op_balanced,
            "strict":    op_strict,
        },
        "cpm": best["cpm"],
        "sens_at_1fp": best["sens_at_1fp"],
        "sens_at_2fp": best["sens_at_2fp"],
        "sens_at_4fp": best["sens_at_4fp"],
        "stratified": {k: {"tp": v[0], "n_gt": v[1],
                            "sens": v[0] / max(v[1], 1)}
                        for k, v in best["stratified"].items()},
        "froc_points": best["points"],
        "luna_sens": best["luna_sens"],
    }, indent=2, default=str))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
