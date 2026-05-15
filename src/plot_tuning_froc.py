"""Generate FROC comparison plot: baseline vs tuned, with operating points marked.

Reads work/academic/best_config.json and best['points'] (FROC of tuned config),
overlays baseline FROC computed on-the-fly from cached probs.

Run: python plot_tuning_froc.py --panel val
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from configs import WORK  # noqa: E402
from panels import load_panel  # noqa: E402
from benchmark import load_patient  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from tune_detection_params import (  # noqa: E402
    PROBS_DIR, cache_lung_mask, precompute_threshold_blobs,
    filter_and_match, LUNA16_FP_RATES,
)
from compare_baseline_vs_tuned import BASELINE  # noqa: E402

ACADEMIC_DIR = WORK / "academic"


def froc_for_config(patients, cfg, thresholds):
    pts = []
    for thr in thresholds:
        tp = fp = fn = n_gt = 0
        for pat in patients:
            blobs = precompute_threshold_blobs(pat["prob"], pat["lung"],
                                                pat["voxel_sp"], thr)
            res = filter_and_match(
                blobs, pat["lung_dist"], pat["voxel_sp"], pat["gt"],
                min_voxels=cfg["min_voxels"], max_elong=cfg["max_elong"],
                merge_dist_mm=cfg["merge_dist_mm"],
                subpleural_min_mm=cfg["subpleural_min_mm"],
            )
            tp += res["tp"]; fp += res["fp"]; fn += res["fn"]; n_gt += res["n_gt"]
        sens = tp / max(n_gt, 1)
        pts.append({"threshold": thr, "sensitivity": sens,
                     "fp_per_scan": fp / max(len(patients), 1),
                     "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt})
    return pts


def cpm(pts):
    sorted_pts = sorted(pts, key=lambda p: p["fp_per_scan"])
    fp_arr = np.array([p["fp_per_scan"] for p in sorted_pts])
    sens_arr = np.array([p["sensitivity"] for p in sorted_pts])
    luna = []
    for ft in LUNA16_FP_RATES:
        if ft <= fp_arr.min(): s = float(sens_arr[0])
        elif ft >= fp_arr.max(): s = float(sens_arr[-1])
        else: s = float(np.interp(ft, fp_arr, sens_arr))
        luna.append(s)
    return float(np.mean(luna)), luna


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["debug", "val", "test"], default="val")
    ap.add_argument("--config", default=str(ACADEMIC_DIR / "best_config.json"))
    ap.add_argument("--out", default=str(ACADEMIC_DIR / "froc_baseline_vs_tuned.png"))
    args = ap.parse_args()

    pids = load_panel(args.panel)
    cfg_data = json.loads(Path(args.config).read_text())
    tuned_cfg = {k: v for k, v in cfg_data["best_config"].items() if k != "default_thr"}
    base_cfg = {k: v for k, v in BASELINE.items() if k != "threshold"}

    print(f"Loading {len(pids)} patient bundles...")
    patients = []
    for pid in pids:
        cp = PROBS_DIR / f"{pid}_tta1.npy"
        if not cp.exists(): continue
        d = load_patient(pid)
        if d is None: continue
        prob = np.load(cp).astype(np.float32)
        lung = cache_lung_mask(pid, d["vol"])
        sp = d["voxel_sp"]
        ldist = distance_transform_edt(lung, sampling=list(sp)).astype(np.float32)
        patients.append({"pid": pid, "prob": prob, "lung": lung,
                         "lung_dist": ldist, "voxel_sp": sp, "gt": d["gt_nodules"]})
    print(f"  {len(patients)} loaded")

    thresholds = cfg_data["thresholds_swept"]
    print(f"Computing FROC for tuned config...")
    tuned_pts = froc_for_config(patients, tuned_cfg, thresholds)
    print(f"Computing FROC for baseline...")
    base_pts = froc_for_config(patients, base_cfg, thresholds)

    cpm_t, _ = cpm(tuned_pts)
    cpm_b, _ = cpm(base_pts)
    print(f"  Tuned    CPM={cpm_t:.4f}")
    print(f"  Baseline CPM={cpm_b:.4f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for pts, label, color in [
        (sorted(base_pts, key=lambda p: p["fp_per_scan"]),
         f"Baseline (CPM={cpm_b:.3f})", "#ff7f0e"),
        (sorted(tuned_pts, key=lambda p: p["fp_per_scan"]),
         f"Tuned (CPM={cpm_t:.3f})", "#1f77b4"),
    ]:
        ax.plot([p["fp_per_scan"] for p in pts],
                [p["sensitivity"] for p in pts],
                marker="o", linewidth=2, markersize=5,
                label=label, color=color)
    # Mark LUNA16 reference FP rates
    for fr in LUNA16_FP_RATES:
        ax.axvline(fr, color="grey", alpha=0.15, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 30)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positives per scan (log scale)")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"FROC: baseline vs tuned ({args.panel} panel, "
                 f"{len(patients)} patients, "
                 f"{sum(p['gt'].__len__() for p in patients)} GT nodules)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
