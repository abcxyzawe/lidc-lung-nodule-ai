"""Academic-grade benchmark for lung nodule detection pipeline.

Produces FROC + CPM + ablation + failure analysis:

  1. FROC curve and CPM (Competition Performance Metric)
     - Sensitivity at FP/scan = [0.125, 0.25, 0.5, 1, 2, 4, 8]
       (the 7 operating points are the LUNA16-standard set)
     - CPM = mean of those 7 sensitivities
  2. Size-stratified analysis (small 4-6mm, medium 6-15mm, large >15mm)
  3. Ablation: full vs -TTA vs -lung-mask vs -post-process
  4. Failure analysis: top 3 FN + top 3 FP with PNG export

NOTE — matching rule:
  We match a predicted nodule to a GT nodule using a *fixed* centroid distance
  threshold (MATCH_TOL_MM = 15 mm). This is inspired by common nodule-detection
  evaluation practice but is NOT the LUNA16 official matching rule, which uses
  a per-nodule radius criterion (a candidate matches GT when centroid distance
  <= GT_diameter / 2). Our fixed 15 mm is more lenient than LUNA16 for small
  nodules. CPM numbers from this script are therefore not directly comparable
  to the LUNA16 leaderboard.

All outputs to work/academic/. Run: python academic_bench.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from predict import (  # noqa: E402
    clean_mask, filter_subpleural, find_nodules,
    get_malignancy_model, get_model, get_swa_model,
    merge_nearby_nodules, predict_nodules, segment_lung,
)
from benchmark import load_patient  # noqa: E402
from panels import load_panel, DEBUG_PANEL  # noqa: E402
from configs import WORK  # noqa: E402

ACADEMIC_DIR = WORK / "academic"
PROBS_DIR = ACADEMIC_DIR / "probs"
FAILURE_DIR = ACADEMIC_DIR / "failure_cases"

LUNA16_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # LUNA16-standard set
THRESHOLD_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
MATCH_TOL_MM = 15.0  # fallback for fixed-distance matching
MIN_RADIUS_MM = 5.0  # minimum effective radius for LUNA16-style matching (very small nodules need some tolerance)


def infer_and_cache(pid, vol, voxel_sp, tta):
    cache_path = PROBS_DIR / f"{pid}_tta{int(tta)}.npy"
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    prob, _ = predict_nodules(vol, threshold=0.5, tta=tta, ensemble=True)
    np.save(cache_path, prob.astype(np.float16))
    return prob


def detect_at_threshold(prob_vol, lung_mask, voxel_sp, threshold,
                        use_lung_mask, use_postproc, min_voxels=200):
    pred = (prob_vol > threshold).astype(np.uint8)
    if use_lung_mask:
        pred = (pred & lung_mask).astype(np.uint8)
    if use_postproc:
        pred = clean_mask(pred, voxel_sp)
    nodules, labeled = find_nodules(
        pred, voxel_sp, min_voxels=min_voxels, max_elongation=4.0,
        prob_volume=prob_vol, core_threshold=0.85,
    )
    if use_postproc:
        nodules = merge_nearby_nodules(nodules, voxel_sp, max_dist_mm=10.0)
        if use_lung_mask:
            nodules = filter_subpleural(nodules, lung_mask, voxel_sp, min_dist_mm=2.0)
    for n in nodules:
        zmin, ymin, xmin, zmax, ymax, xmax = n["bbox_zyx_voxel"]
        blob = (labeled[zmin:zmax, ymin:ymax, xmin:xmax] == n["id"])
        if blob.any():
            n["confidence"] = float(prob_vol[zmin:zmax, ymin:ymax, xmin:xmax][blob].mean())
        else:
            n["confidence"] = 0.0
    return nodules


def match_predictions_to_gt(pred_nodules, gt_nodules, voxel_sp,
                            rule: str = "luna16_radius",
                            tol_mm: float = MATCH_TOL_MM,
                            min_radius_mm: float = MIN_RADIUS_MM):
    """Match predicted nodules to GT nodules.

    rule:
      - "luna16_radius": A prediction matches a GT if centroid distance
        <= max(GT_diameter/2, min_radius_mm). This is the official LUNA16
        rule. min_radius_mm provides tolerance for very small annotations.
      - "fixed_15mm": Fixed centroid distance <= tol_mm regardless of size.
    """
    if not gt_nodules or not pred_nodules:
        return [], list(range(len(gt_nodules))), list(range(len(pred_nodules)))
    used_pred = set()
    matched = []
    for gi, g in enumerate(gt_nodules):
        gc = np.array(g["centroid_zyx_voxel"]) * np.array(voxel_sp)
        if rule == "luna16_radius":
            tol = max(g["diam_mm"] / 2, min_radius_mm)
        else:
            tol = tol_mm
        best_pi, best_d = None, float("inf")
        for pi, p in enumerate(pred_nodules):
            if pi in used_pred: continue
            pc = np.array(p["centroid_zyx_voxel"]) * np.array(voxel_sp)
            d = np.linalg.norm(gc - pc)
            if d < best_d: best_d, best_pi = d, pi
        if best_pi is not None and best_d <= tol:
            matched.append((gi, best_pi, best_d))
            used_pred.add(best_pi)
    unmatched_gt = [i for i in range(len(gt_nodules)) if i not in {m[0] for m in matched}]
    unmatched_pred = [i for i in range(len(pred_nodules)) if i not in used_pred]
    return matched, unmatched_gt, unmatched_pred


def assess_at_threshold(patient_data, threshold, use_lung_mask=True, use_postproc=True,
                        match_rule: str = "luna16_radius",
                        min_voxels: int = 200, max_elongation: float = 4.0,
                        merge_dist_mm: float = 10.0, subpleural_dist_mm: float = 2.0):
    tp = fp = fn = n_gt = n_pred = 0
    for pd in patient_data:
        nodules = detect_at_threshold(pd["prob"], pd["lung"], pd["voxel_sp"],
                                      threshold, use_lung_mask, use_postproc,
                                      min_voxels=min_voxels)
        # Apply elongation/merge/subpleural overrides if defaults differ
        # (currently detect_at_threshold uses fixed 4.0 / 10.0 / 2.0; for sweep
        # we pass these via separate code path in tuner — see tune_detection_params.py)
        matched, u_gt, u_pred = match_predictions_to_gt(
            nodules, pd["gt"], pd["voxel_sp"], rule=match_rule
        )
        tp += len(matched); fp += len(u_pred); fn += len(u_gt)
        n_gt += len(pd["gt"]); n_pred += len(nodules)
    sens = tp / max(n_gt, 1)
    fp_per_scan = fp / max(len(patient_data), 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-7)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "n_gt": n_gt, "n_pred": n_pred,
            "sensitivity": float(sens), "fp_per_scan": float(fp_per_scan),
            "precision": float(prec), "f1": float(f1)}


def compute_froc(patient_data, thresholds=THRESHOLD_SWEEP, match_rule="luna16_radius"):
    points = []
    print(f"  Sweeping {len(thresholds)} thresholds for FROC ...", flush=True)
    for thr in thresholds:
        res = assess_at_threshold(patient_data, thr, match_rule=match_rule)
        points.append({"threshold": thr, "fp_per_scan": res["fp_per_scan"],
                       "sensitivity": res["sensitivity"],
                       "tp": res["tp"], "fp": res["fp"], "fn": res["fn"]})
        print(f"    thr={thr:.2f}  sens={res['sensitivity']:.3f}  FP/scan={res['fp_per_scan']:.2f}  TP={res['tp']} FP={res['fp']} FN={res['fn']}", flush=True)
    points_sorted = sorted(points, key=lambda p: p["fp_per_scan"])
    fp_arr = np.array([p["fp_per_scan"] for p in points_sorted])
    sens_arr = np.array([p["sensitivity"] for p in points_sorted])
    luna_sens = []
    for fp_target in LUNA16_FP_RATES:
        if fp_target <= fp_arr.min():
            sens = float(sens_arr[0])
        elif fp_target >= fp_arr.max():
            sens = float(sens_arr[-1])
        else:
            sens = float(np.interp(fp_target, fp_arr, sens_arr))
        luna_sens.append({"fp_per_scan": fp_target, "sensitivity": sens})
    cpm = float(np.mean([s["sensitivity"] for s in luna_sens]))
    return {"all_points": points, "luna16_points": luna_sens, "cpm": cpm}


def plot_froc(froc, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    pts = sorted(froc["all_points"], key=lambda p: p["fp_per_scan"])
    ax.plot([p["fp_per_scan"] for p in pts], [p["sensitivity"] for p in pts],
            "b-o", linewidth=2, markersize=4, label="Threshold sweep")
    lp = froc["luna16_points"]
    ax.scatter([p["fp_per_scan"] for p in lp], [p["sensitivity"] for p in lp],
               c="red", s=80, marker="*", zorder=5,
               label=f"LUNA16 FP rates (CPM={froc['cpm']:.3f})")
    for p in lp:
        ax.annotate(f"{p['sensitivity']:.2f}", (p["fp_per_scan"], p["sensitivity"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("False Positives per scan (log scale)")
    ax.set_ylabel("Sensitivity")
    n_total = pts[0]["tp"] + pts[0]["fn"]
    ax.set_title(f"FROC Curve - UNet++ B5 ensemble + TTA - LIDC panel ({n_total} GT nodules)")
    ax.set_xlim(0.05, 30); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def stratified(patient_data, threshold=0.65, match_rule="luna16_radius"):
    buckets = {"small_4_6mm": (4, 6), "medium_6_15mm": (6, 15), "large_15mm_plus": (15, 999)}
    bs = {k: {"tp": 0, "fn": 0, "n_gt": 0} for k in buckets}
    for pd in patient_data:
        nodules = detect_at_threshold(pd["prob"], pd["lung"], pd["voxel_sp"],
                                      threshold, True, True)
        matched, u_gt, _ = match_predictions_to_gt(nodules, pd["gt"], pd["voxel_sp"], rule=match_rule)
        matched_idx = {m[0] for m in matched}
        for gi, g in enumerate(pd["gt"]):
            d = g["diam_mm"]
            for name, (lo, hi) in buckets.items():
                if lo <= d < hi:
                    bs[name]["n_gt"] += 1
                    if gi in matched_idx: bs[name]["tp"] += 1
                    else: bs[name]["fn"] += 1
                    break
    out = {}
    for name, s in bs.items():
        n = s["n_gt"]
        out[name] = {"n_gt": n, "tp": s["tp"], "fn": s["fn"],
                     "sensitivity": s["tp"] / n if n > 0 else None}
    return out


def ablation(patient_data, threshold=0.65):
    print("  1/4: Full pipeline", flush=True)
    full = assess_at_threshold(patient_data, threshold, True, True)
    print("  2/4: -Post-process", flush=True)
    no_pp = assess_at_threshold(patient_data, threshold, True, False)
    print("  3/4: -Lung mask", flush=True)
    no_lung = assess_at_threshold(patient_data, threshold, False, True)
    print("  4/4: -TTA", flush=True)
    pat_no_tta = []
    for pd in patient_data:
        prob_no = infer_and_cache(pd["pid"], pd["vol"], pd["voxel_sp"], tta=False)
        pat_no_tta.append({**pd, "prob": prob_no})
    no_tta = assess_at_threshold(pat_no_tta, threshold, True, True)
    keys = ("sensitivity", "precision", "f1", "fp_per_scan", "tp", "fp", "fn")
    return {
        "full":         {k: full[k] for k in keys},
        "no_postproc":  {k: no_pp[k] for k in keys},
        "no_lung_mask": {k: no_lung[k] for k in keys},
        "no_tta":       {k: no_tta[k] for k in keys},
    }


def export_failure_png(vol, voxel_sp, gt, pred, title, out_path):
    if pred is not None:
        cz = int(round(pred["centroid_zyx_voxel"][0]))
    elif gt is not None:
        cz = int(round(gt["centroid_zyx_voxel"][0]))
    else:
        return
    cz = max(0, min(vol.shape[0] - 1, cz))
    sl = vol[cz].astype(np.float32)
    img = np.clip(sl, -1000, 200)
    img = ((img + 1000) / 1200 * 255).astype(np.uint8)
    pim = Image.fromarray(img, mode="L").convert("RGB")
    draw = ImageDraw.Draw(pim)
    if gt is not None:
        gz, gy, gx = gt["centroid_zyx_voxel"]
        r = max(8, (gt["diam_mm"] / 2) / voxel_sp[1])
        draw.rectangle([gx - r, gy - r, gx + r, gy + r], outline=(80, 230, 80), width=3)
        draw.text((max(0, gx - r), max(0, gy - r - 14)),
                  f"GT {gt['diam_mm']:.1f}mm (rad {gt['n_radiologists']}/4)",
                  fill=(80, 230, 80))
    if pred is not None:
        zmin, ymin, xmin, zmax, ymax, xmax = pred["bbox_zyx_voxel"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 80, 80), width=3)
        draw.text((xmin, min(vol.shape[1]-12, ymax + 2)),
                  f"Pred {pred['diameter_mm']:.1f}mm conf={pred.get('confidence', 0)*100:.0f}%",
                  fill=(255, 80, 80))
    draw.text((6, 6), title, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pim.save(out_path)


def failure(patient_data, threshold=0.65, match_rule="luna16_radius"):
    all_fn, all_fp = [], []
    for pd in patient_data:
        nodules = detect_at_threshold(pd["prob"], pd["lung"], pd["voxel_sp"],
                                      threshold, True, True)
        matched, u_gt, u_pred = match_predictions_to_gt(nodules, pd["gt"], pd["voxel_sp"], rule=match_rule)
        for gi in u_gt:
            all_fn.append({"pd": pd, "gt": pd["gt"][gi], "diam": pd["gt"][gi]["diam_mm"]})
        for pi in u_pred:
            n = nodules[pi]
            all_fp.append({"pd": pd, "pred": n, "conf": n.get("confidence", 0)})
    all_fn.sort(key=lambda x: -x["diam"])
    all_fp.sort(key=lambda x: -x["conf"])
    out = {"false_negatives": [], "false_positives": []}
    for i, fn in enumerate(all_fn[:3], 1):
        path = FAILURE_DIR / f"fn_{i}_{fn['pd']['pid']}.png"
        export_failure_png(fn["pd"]["vol"], fn["pd"]["voxel_sp"], fn["gt"], None,
                           f"FN #{i}: {fn['pd']['pid']} - missed {fn['diam']:.1f}mm GT", path)
        out["false_negatives"].append({
            "pid": fn["pd"]["pid"], "gt_diam_mm": fn["diam"],
            "gt_pos": fn["gt"]["centroid_zyx_voxel"],
            "n_radiologists": fn["gt"]["n_radiologists"],
            "malignancy_mean": fn["gt"].get("malignancy_mean"),
            "image": str(path.relative_to(WORK)),
        })
    for i, fp in enumerate(all_fp[:3], 1):
        path = FAILURE_DIR / f"fp_{i}_{fp['pd']['pid']}.png"
        export_failure_png(fp["pd"]["vol"], fp["pd"]["voxel_sp"], None, fp["pred"],
                           f"FP #{i}: {fp['pd']['pid']} - bogus {fp['pred']['diameter_mm']:.1f}mm", path)
        out["false_positives"].append({
            "pid": fp["pd"]["pid"], "diam_mm": fp["pred"]["diameter_mm"],
            "confidence": fp["conf"], "pos": fp["pred"]["centroid_zyx_voxel"],
            "elongation": fp["pred"].get("elongation"),
            "image": str(path.relative_to(WORK)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["debug", "val", "test"], default="debug",
                    help="Which evaluation panel to use")
    ap.add_argument("--patients", default="",
                    help="Comma-separated patient IDs (overrides --panel)")
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--match-rule", choices=["luna16_radius", "fixed_15mm"],
                    default="luna16_radius",
                    help="Matching rule for predictions vs GT")
    ap.add_argument("--out-name", default="results",
                    help="Output JSON name in work/academic/")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    ACADEMIC_DIR.mkdir(parents=True, exist_ok=True)
    if args.patients:
        pids = [p.strip() for p in args.patients.split(",") if p.strip()]
        panel_name = "custom"
    else:
        pids = load_panel(args.panel)
        panel_name = args.panel
    print(f"=== Academic benchmark on {panel_name} panel ({len(pids)} patients) ===", flush=True)
    print(f"=== Match rule: {args.match_rule} ===\n", flush=True)
    print("Loading models ...", flush=True)
    get_model(); get_swa_model(); get_malignancy_model()

    print(f"\n[Step 1] Load + cache prob volumes (TTA on)", flush=True)
    patient_data = []
    for i, pid in enumerate(pids):
        t0 = time.time()
        data = load_patient(pid)
        if data is None:
            print(f"  [{i+1}/{len(pids)}] {pid}: SKIP", flush=True); continue
        lung = segment_lung(data["vol"])
        prob = infer_and_cache(pid, data["vol"], data["voxel_sp"], tta=True)
        patient_data.append({"pid": pid, "vol": data["vol"], "voxel_sp": data["voxel_sp"],
                            "gt": data["gt_nodules"], "lung": lung, "prob": prob})
        print(f"  [{i+1}/{len(pids)}] {pid}: {len(data['gt_nodules'])} GT, {time.time()-t0:.1f}s", flush=True)

    print(f"\n[Step 2] FROC + CPM", flush=True)
    froc = compute_froc(patient_data, match_rule=args.match_rule)
    plot_froc(froc, ACADEMIC_DIR / f"froc_{panel_name}.png")
    print(f"\n  CPM = {froc['cpm']:.4f}")
    for p in froc["luna16_points"]:
        print(f"    FP/scan={p['fp_per_scan']:>6.3f}  sens={p['sensitivity']:.3f}")

    print(f"\n[Step 3] Stratified @ thr={args.threshold}", flush=True)
    strat = stratified(patient_data, args.threshold, match_rule=args.match_rule)
    for name, s in strat.items():
        sens = f"{s['sensitivity']:.3f}" if s['sensitivity'] is not None else "N/A"
        print(f"  {name:18} (n_gt={s['n_gt']:3}): sens={sens}  TP={s['tp']} FN={s['fn']}")

    abl = None
    if not args.skip_ablation:
        print(f"\n[Step 4] Ablation @ thr={args.threshold}", flush=True)
        abl = ablation(patient_data, args.threshold)
        print(f"\n  {'Variant':<20} {'Sens':>7} {'Prec':>7} {'F1':>7} {'FP/scan':>10}")
        for name, m in abl.items():
            print(f"  {name:<20} {m['sensitivity']:>7.3f} {m['precision']:>7.3f} {m['f1']:>7.3f} {m['fp_per_scan']:>10.2f}")

    print(f"\n[Step 5] Failure analysis @ thr={args.threshold}", flush=True)
    failures = failure(patient_data, args.threshold, match_rule=args.match_rule)
    for fn in failures["false_negatives"]:
        print(f"  FN: {fn['pid']} {fn['gt_diam_mm']:.1f}mm (mal={fn['malignancy_mean']})")
    for fp in failures["false_positives"]:
        print(f"  FP: {fp['pid']} {fp['diam_mm']:.1f}mm conf={fp['confidence']*100:.0f}%")

    match_rule_desc = (
        "LUNA16 official: centroid <= max(GT_diameter/2, 5mm)" if args.match_rule == "luna16_radius"
        else "fixed centroid distance <= 15 mm"
    )
    summary = {
        "panel": panel_name,
        "n_patients": len(patient_data),
        "patient_ids": [pd["pid"] for pd in patient_data],
        "default_threshold": args.threshold,
        "froc": froc, "stratified": strat,
        "ablation": abl, "failure_analysis": failures,
        "tools": {
            "seg_model": "UNet++ EfficientNet-B5 (best.pt + swa.pt ensemble)",
            "lung_mask": "Lungmask R231 (Hofmanninger 2020)",
            "filters": "min_voxels=200, max_elongation=4, merge<10mm, subpleural>2mm",
            "match_rule": match_rule_desc,
            "luna16_fp_rates_used": LUNA16_FP_RATES,
        },
    }
    out_json = ACADEMIC_DIR / f"{args.out_name}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n=== Done ===")
    print(f"FROC plot:   {ACADEMIC_DIR / 'froc.png'}")
    print(f"Failure PNG: {FAILURE_DIR}")
    print(f"Summary:     {out_json}")


if __name__ == "__main__":
    main()
