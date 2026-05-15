"""Side-by-side: current webapp defaults vs sweep-best config.

Both configs are evaluated on the same panel (using cached prob volumes).
Reports per-patient TP/FP/FN deltas + aggregate sensitivity / FP-per-scan / CPM.

Run:
  python compare_baseline_vs_tuned.py --panel val
  python compare_baseline_vs_tuned.py --panel test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from configs import WORK  # noqa: E402
from panels import load_panel  # noqa: E402
from benchmark import load_patient  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from tune_detection_params import (  # noqa: E402
    PROBS_DIR, cache_lung_mask, precompute_threshold_blobs,
    filter_and_match, LUNA16_FP_RATES, compute_cpm_at_config,
)

ACADEMIC_DIR = WORK / "academic"

# Current webapp/configs.py defaults at time of writing
BASELINE = {
    "threshold": 0.65,
    "min_voxels": 200,
    "max_elong": 4.0,
    "merge_dist_mm": 10.0,
    "subpleural_min_mm": 2.0,
}


def evaluate(patients, cfg: dict, threshold: float):
    tp = fp = fn = n_gt = 0
    for pat in patients:
        blobs = precompute_threshold_blobs(pat["prob"], pat["lung"],
                                            pat["voxel_sp"], threshold)
        res = filter_and_match(
            blobs, pat["lung_dist"], pat["voxel_sp"], pat["gt"],
            min_voxels=cfg["min_voxels"], max_elong=cfg["max_elong"],
            merge_dist_mm=cfg["merge_dist_mm"],
            subpleural_min_mm=cfg["subpleural_min_mm"],
        )
        tp += res["tp"]; fp += res["fp"]; fn += res["fn"]; n_gt += res["n_gt"]
    sens = tp / max(n_gt, 1)
    prec = tp / max(tp + fp, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt,
            "sensitivity": sens, "precision": prec,
            "fp_per_scan": fp / max(len(patients), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["debug", "val", "test"], default="val")
    ap.add_argument("--config", default=str(ACADEMIC_DIR / "best_config.json"))
    ap.add_argument("--out", default=str(ACADEMIC_DIR / "baseline_vs_tuned.json"))
    args = ap.parse_args()

    pids = load_panel(args.panel)
    cfg_data = json.loads(Path(args.config).read_text())
    tuned = cfg_data["best_config"]
    tuned_thr = cfg_data["best_threshold_op_at_1fp"]

    print(f"Panel '{args.panel}' ({len(pids)} patients)")
    print(f"BASELINE: thr={BASELINE['threshold']:.2f}  min_vox={BASELINE['min_voxels']}  "
          f"elong={BASELINE['max_elong']}  merge={BASELINE['merge_dist_mm']}  "
          f"subp={BASELINE['subpleural_min_mm']}")
    print(f"TUNED   : thr={tuned_thr:.2f}  min_vox={tuned['min_voxels']}  "
          f"elong={tuned['max_elong']}  merge={tuned['merge_dist_mm']}  "
          f"subp={tuned['subpleural_min_mm']}")

    print(f"\n[Load] patient bundles")
    patients = []
    for pid in pids:
        cache_p = PROBS_DIR / f"{pid}_tta1.npy"
        if not cache_p.exists():
            print(f"  {pid}: no prob cache (skip)")
            continue
        data = load_patient(pid)
        if data is None: continue
        prob = np.load(cache_p).astype(np.float32)
        lung = cache_lung_mask(pid, data["vol"])
        sp = data["voxel_sp"]
        ldist = distance_transform_edt(lung, sampling=list(sp)).astype(np.float32)
        patients.append({"pid": pid, "prob": prob, "lung": lung, "lung_dist": ldist,
                          "voxel_sp": sp, "gt": data["gt_nodules"]})
    print(f"  Loaded {len(patients)} patients")

    print(f"\n[Eval] BASELINE")
    base_cfg = {k: v for k, v in BASELINE.items() if k != "threshold"}
    base = evaluate(patients, base_cfg, BASELINE["threshold"])
    print(f"  TP={base['tp']}  FP={base['fp']}  FN={base['fn']}")
    print(f"  Sens={base['sensitivity']:.3f}  Prec={base['precision']:.3f}  "
          f"FP/scan={base['fp_per_scan']:.2f}")

    print(f"\n[Eval] TUNED")
    tuned_cfg = {k: v for k, v in tuned.items() if k != "default_thr"}
    tuned_eval = evaluate(patients, tuned_cfg, tuned_thr)
    print(f"  TP={tuned_eval['tp']}  FP={tuned_eval['fp']}  FN={tuned_eval['fn']}")
    print(f"  Sens={tuned_eval['sensitivity']:.3f}  Prec={tuned_eval['precision']:.3f}  "
          f"FP/scan={tuned_eval['fp_per_scan']:.2f}")

    print(f"\n=== DELTA ===")
    print(f"  Sens:    {tuned_eval['sensitivity'] - base['sensitivity']:+.3f}")
    print(f"  Prec:    {tuned_eval['precision'] - base['precision']:+.3f}")
    print(f"  FP/scan: {tuned_eval['fp_per_scan'] - base['fp_per_scan']:+.2f}")
    print(f"  TP:      {tuned_eval['tp'] - base['tp']:+d}")
    print(f"  FP:      {tuned_eval['fp'] - base['fp']:+d}")
    print(f"  FN:      {tuned_eval['fn'] - base['fn']:+d}")

    Path(args.out).write_text(json.dumps({
        "panel": args.panel, "n_patients": len(patients),
        "baseline_config": BASELINE, "tuned_config": {**tuned, "threshold": tuned_thr},
        "baseline_metrics": base, "tuned_metrics": tuned_eval,
    }, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
