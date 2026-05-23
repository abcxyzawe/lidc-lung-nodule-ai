"""Temporary re-eval script: compare luna16_radius vs fixed_15mm on test panel.

Uses cached prob volumes from work/academic/probs/ (no GPU needed).
Output: work/academic/MINE_LUNA16_RULE.json

Run: python src/reeval_luna16_rule.py
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))

from configs import WORK, PRE_DIR
from panels import load_panel
from benchmark import load_patient
from tune_detection_params import (
    precompute_threshold_blobs, filter_and_match, LUNA16_FP_RATES
)

PROBS_DIR = WORK / "academic" / "probs"
LUNG_DIR = WORK / "academic" / "lung_masks"
THRESHOLD_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                   0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
POST_CFG = {"min_voxels": 200, "max_elong": 4.0,
            "merge_dist_mm": 10.0, "subpleural_min_mm": 2.0}


def load_cached_patient(pid):
    prob_path = PROBS_DIR / f"{pid}_tta1.npy"
    lung_path = LUNG_DIR / f"{pid}.npy"
    if not prob_path.exists() or not lung_path.exists():
        return None
    data = load_patient(pid)
    if data is None:
        return None
    prob = np.load(prob_path).astype(np.float32)
    lung = np.load(lung_path)
    sp = data["voxel_sp"]
    ldist = distance_transform_edt(lung, sampling=list(sp)).astype(np.float32)
    return {"pid": pid, "prob": prob, "lung": lung, "ldist": ldist,
            "voxel_sp": sp, "gt": data["gt_nodules"]}


def eval_threshold(patient_data, thr, match_rule, post):
    tp = fp = fn = n_gt = 0
    for pd in patient_data:
        blobs = precompute_threshold_blobs(
            pd["prob"], pd["lung"], pd["voxel_sp"], thr)
        if match_rule == "luna16_radius":
            res = filter_and_match(
                blobs, pd["ldist"], pd["voxel_sp"], pd["gt"],
                min_voxels=post["min_voxels"], max_elong=post["max_elong"],
                merge_dist_mm=post["merge_dist_mm"],
                subpleural_min_mm=post["subpleural_min_mm"],
                match_min_radius_mm=3.0,
            )
            tp += res["tp"]; fp += res["fp"]; fn += res["fn"]
        else:
            # fixed_15mm: reuse the filter stage but override matching
            res = filter_and_match(
                blobs, pd["ldist"], pd["voxel_sp"], pd["gt"],
                min_voxels=post["min_voxels"], max_elong=post["max_elong"],
                merge_dist_mm=post["merge_dist_mm"],
                subpleural_min_mm=post["subpleural_min_mm"],
                match_min_radius_mm=3.0,  # use luna16 for filter stage
            )
            # Re-match with fixed 15mm
            preds = res["preds"]
            sp = np.array(pd["voxel_sp"])
            matched = []
            used_p = set()
            for gi, g in enumerate(pd["gt"]):
                gc = np.array(g["centroid_zyx_voxel"], dtype=np.float32) * sp
                best, bestd = None, float("inf")
                for pi, p in enumerate(preds):
                    if pi in used_p:
                        continue
                    d = float(np.linalg.norm(gc - p["centroid_mm"]))
                    if d < bestd:
                        bestd, best = d, pi
                if best is not None and bestd <= 15.0:
                    matched.append((gi, best))
                    used_p.add(best)
            tp += len(matched)
            fp += len(preds) - len(matched)
            fn += len(pd["gt"]) - len(matched)
        n_gt += len(pd["gt"])
    sens = tp / max(n_gt, 1)
    fp_per_scan = fp / max(len(patient_data), 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-7)
    return {"thr": thr, "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt,
            "sens": sens, "fp_per_scan": fp_per_scan, "prec": prec, "f1": f1}


def compute_froc(patient_data, match_rule):
    points = []
    print(f"  Sweeping {len(THRESHOLD_SWEEP)} thresholds ...", flush=True)
    for thr in THRESHOLD_SWEEP:
        r = eval_threshold(patient_data, thr, match_rule, POST_CFG)
        print(f"    thr={thr:.2f}  sens={r['sens']:.3f}  FP/scan={r['fp_per_scan']:.2f}"
              f"  F1={r['f1']:.3f}  TP={r['tp']} FP={r['fp']} FN={r['fn']}", flush=True)
        points.append(r)
    pts_sorted = sorted(points, key=lambda p: p["fp_per_scan"])
    fp_arr = np.array([p["fp_per_scan"] for p in pts_sorted])
    sens_arr = np.array([p["sens"] for p in pts_sorted])
    luna_sens = []
    for fp_target in LUNA16_FP_RATES:
        if fp_target <= fp_arr.min():
            s = float(sens_arr[0])
        elif fp_target >= fp_arr.max():
            s = float(sens_arr[-1])
        else:
            s = float(np.interp(fp_target, fp_arr, sens_arr))
        luna_sens.append({"fp_per_scan": fp_target, "sensitivity": s})
    cpm = float(np.mean([x["sensitivity"] for x in luna_sens]))
    best = max(points, key=lambda x: x["f1"])
    return {"cpm": cpm, "luna16_points": luna_sens, "best": best, "all_points": points}


def stratified(patient_data, thr, match_rule):
    buckets = {"small_4_6mm": (4, 6), "medium_6_15mm": (6, 15),
               "large_15mm_plus": (15, 999)}
    bs = {k: {"tp": 0, "fn": 0, "n_gt": 0} for k in buckets}
    for pd in patient_data:
        blobs = precompute_threshold_blobs(
            pd["prob"], pd["lung"], pd["voxel_sp"], thr)
        if match_rule == "luna16_radius":
            res = filter_and_match(
                blobs, pd["ldist"], pd["voxel_sp"], pd["gt"],
                min_voxels=POST_CFG["min_voxels"], max_elong=POST_CFG["max_elong"],
                merge_dist_mm=POST_CFG["merge_dist_mm"],
                subpleural_min_mm=POST_CFG["subpleural_min_mm"],
                match_min_radius_mm=3.0)
            matched_gi = {m[0] for m in res["matched"]}
        else:
            res = filter_and_match(
                blobs, pd["ldist"], pd["voxel_sp"], pd["gt"],
                min_voxels=POST_CFG["min_voxels"], max_elong=POST_CFG["max_elong"],
                merge_dist_mm=POST_CFG["merge_dist_mm"],
                subpleural_min_mm=POST_CFG["subpleural_min_mm"],
                match_min_radius_mm=3.0)
            preds = res["preds"]
            sp = np.array(pd["voxel_sp"])
            matched = []
            used_p = set()
            for gi, g in enumerate(pd["gt"]):
                gc = np.array(g["centroid_zyx_voxel"], dtype=np.float32) * sp
                best_p, bestd = None, float("inf")
                for pi, p in enumerate(preds):
                    if pi in used_p:
                        continue
                    d = float(np.linalg.norm(gc - p["centroid_mm"]))
                    if d < bestd:
                        bestd, best_p = d, pi
                if best_p is not None and bestd <= 15.0:
                    matched.append((gi, best_p))
                    used_p.add(best_p)
            matched_gi = {m[0] for m in matched}

        for gi, g in enumerate(pd["gt"]):
            d = g["diam_mm"]
            for name, (lo, hi) in buckets.items():
                if lo <= d < hi:
                    bs[name]["n_gt"] += 1
                    if gi in matched_gi:
                        bs[name]["tp"] += 1
                    else:
                        bs[name]["fn"] += 1
                    break
    return {k: {"n_gt": v["n_gt"], "tp": v["tp"], "fn": v["fn"],
                "sensitivity": v["tp"] / v["n_gt"] if v["n_gt"] > 0 else None}
            for k, v in bs.items()}


def main():
    print("Loading test panel ...", flush=True)
    pids = load_panel("test")
    patient_data = []
    t0 = time.time()
    for pid in pids:
        pd = load_cached_patient(pid)
        if pd:
            patient_data.append(pd)
    n_loaded = len(patient_data)
    print(f"Loaded {n_loaded}/{len(pids)} patients ({time.time()-t0:.1f}s)", flush=True)
    n_gt_total = sum(len(pd["gt"]) for pd in patient_data)
    print(f"Total GT nodules: {n_gt_total}", flush=True)

    print("\n=== LUNA16 radius rule (max(diam/2, 3mm)) ===", flush=True)
    r_luna = compute_froc(patient_data, "luna16_radius")
    print(f"\nCPM = {r_luna['cpm']:.4f}")
    for p in r_luna["luna16_points"]:
        print(f"  FP/scan={p['fp_per_scan']:>6.3f}  sens={p['sensitivity']:.3f}")
    b = r_luna["best"]
    print(f"Best thr={b['thr']}  F1={b['f1']:.4f}  sens={b['sens']:.3f}  "
          f"prec={b['prec']:.3f}  FP/scan={b['fp_per_scan']:.2f}")

    print("\n=== fixed_15mm rule ===", flush=True)
    r_fixed = compute_froc(patient_data, "fixed_15mm")
    print(f"\nCPM = {r_fixed['cpm']:.4f}")
    for p in r_fixed["luna16_points"]:
        print(f"  FP/scan={p['fp_per_scan']:>6.3f}  sens={p['sensitivity']:.3f}")
    b2 = r_fixed["best"]
    print(f"Best thr={b2['thr']}  F1={b2['f1']:.4f}  sens={b2['sens']:.3f}  "
          f"prec={b2['prec']:.3f}  FP/scan={b2['fp_per_scan']:.2f}")

    best_thr_luna = r_luna["best"]["thr"]
    print(f"\n=== Stratified (luna16 rule, thr={best_thr_luna}) ===", flush=True)
    strat = stratified(patient_data, best_thr_luna, "luna16_radius")
    for name, s in strat.items():
        sens_str = f"{s['sensitivity']:.3f}" if s["sensitivity"] is not None else "N/A"
        print(f"  {name:22} n_gt={s['n_gt']:3}  tp={s['tp']:3}  fn={s['fn']:3}  "
              f"sens={sens_str}")

    out = {
        "n_patients": n_loaded,
        "n_gt_total": n_gt_total,
        "panel": "test",
        "post_proc": POST_CFG,
        "match_rules_compared": {
            "luna16_radius": {
                "rule_description": "centroid <= max(GT_diam/2, 3mm) — LUNA16 official",
                "cpm": r_luna["cpm"],
                "best_f1": r_luna["best"]["f1"],
                "best_threshold": r_luna["best"]["thr"],
                "sensitivity_at_best": r_luna["best"]["sens"],
                "precision_at_best": r_luna["best"]["prec"],
                "fp_per_scan_at_best": r_luna["best"]["fp_per_scan"],
                "luna16_points": r_luna["luna16_points"],
            },
            "fixed_15mm": {
                "rule_description": "centroid <= 15mm fixed — LIDC project baseline",
                "cpm": r_fixed["cpm"],
                "best_f1": r_fixed["best"]["f1"],
                "best_threshold": r_fixed["best"]["thr"],
                "sensitivity_at_best": r_fixed["best"]["sens"],
                "precision_at_best": r_fixed["best"]["prec"],
                "fp_per_scan_at_best": r_fixed["best"]["fp_per_scan"],
                "luna16_points": r_fixed["luna16_points"],
            },
        },
        "stratified_luna16_at_best_thr": strat,
        "note": (
            "Baseline F1=0.618 from commit cd48359 was measured on test_panel (99 patients) "
            "using tune_detection_params + filter_and_match pipeline (not academic_bench). "
            "This re-eval uses same post-proc params but compares matching rules."
        ),
    }
    out_path = WORK / "academic" / "MINE_LUNA16_RULE.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

    # Summary table
    print("\n=== COMPARISON TABLE ===")
    print(f"{'Rule':<25} {'CPM':>8} {'Best F1':>8} {'Sens':>7} {'Prec':>7} {'FP/scan':>8}")
    print("-" * 67)
    for rule, r in [("luna16_radius", r_luna), ("fixed_15mm", r_fixed)]:
        b = r["best"]
        print(f"{rule:<25} {r['cpm']:>8.4f} {b['f1']:>8.4f} {b['sens']:>7.3f} "
              f"{b['prec']:>7.3f} {b['fp_per_scan']:>8.2f}")


if __name__ == "__main__":
    main()
