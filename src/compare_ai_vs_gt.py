"""Per-patient AI vs GT comparison table.

For each patient on the chosen panel, runs the configured AI pipeline
(uses cached prob volumes), matches to GT (LUNA16 radius rule), prints:

  - GT nodules (size, malignancy, position)
  - AI predictions (size, confidence, position)
  - Match table: which GT was found, by which prediction, residual error

Usage:
  python compare_ai_vs_gt.py --panel debug
  python compare_ai_vs_gt.py --panel val --config work/academic/best_config.json
  python compare_ai_vs_gt.py --panel test                  # uses best_config.json by default
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
from tune_detection_params import (  # noqa: E402
    PROBS_DIR, LUNG_DIR, cache_lung_mask,
    precompute_threshold_blobs, filter_and_match,
)
from scipy.ndimage import distance_transform_edt  # noqa: E402

ACADEMIC_DIR = WORK / "academic"


def run_one(pid: str, cfg: dict, threshold: float):
    cache_path = PROBS_DIR / f"{pid}_tta1.npy"
    if not cache_path.exists():
        return None
    data = load_patient(pid)
    if data is None: return None
    prob = np.load(cache_path).astype(np.float32)
    lung = cache_lung_mask(pid, data["vol"])
    sp = data["voxel_sp"]
    lung_dist = distance_transform_edt(lung, sampling=list(sp)).astype(np.float32)
    blobs = precompute_threshold_blobs(prob, lung, sp, threshold)
    res = filter_and_match(
        blobs, lung_dist, sp, data["gt_nodules"],
        min_voxels=cfg["min_voxels"], max_elong=cfg["max_elong"],
        merge_dist_mm=cfg["merge_dist_mm"],
        subpleural_min_mm=cfg["subpleural_min_mm"],
    )
    return {"pid": pid, "gt": data["gt_nodules"], "voxel_sp": sp, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["debug", "val", "test"], default="debug")
    ap.add_argument("--config", default=str(ACADEMIC_DIR / "best_config.json"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg_data = json.loads(Path(args.config).read_text())
    cfg = cfg_data["best_config"]
    thr = args.threshold if args.threshold is not None else cfg_data["best_threshold_op_at_1fp"]
    print(f"Using config: thr={thr:.2f}  {cfg}\n")

    pids = load_panel(args.panel)
    rows = []
    grand_tp = grand_fp = grand_fn = grand_n_gt = 0
    for pid in pids:
        r = run_one(pid, cfg, thr)
        if r is None:
            print(f"[SKIP] {pid} (no cache or h5)")
            continue
        rows.append(r)
        grand_tp += r["tp"]; grand_fp += r["fp"]; grand_fn += r["fn"]; grand_n_gt += r["n_gt"]
        print(f"\n=== {pid} ===  GT={r['n_gt']}  Pred={r['n_pred']}  "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']}")
        print(f"  GT nodules:")
        matched_gi = {m[0] for m in r["matched"]}
        for gi, g in enumerate(r["gt"]):
            tag = "MATCHED" if gi in matched_gi else "MISSED"
            mal = g.get("malignancy_mean", "?")
            mal_s = f"{mal:.1f}" if isinstance(mal, (int, float)) else str(mal)
            print(f"    GT#{gi}: {g['diam_mm']:5.1f}mm  rads={g['n_radiologists']}/4  "
                  f"mal={mal_s}  pos={g['centroid_zyx_voxel']}  [{tag}]")
        print(f"  AI predictions ({r['n_pred']}):")
        matched_pi = {m[1] for m in r["matched"]}
        for pi, p in enumerate(r["preds"]):
            tag = "TP" if pi in matched_pi else "FP"
            cv = p["centroid_voxel"]
            print(f"    AI#{pi}: {p['diameter_mm']:5.1f}mm  conf={p['confidence']:.2f}  "
                  f"elong={p['elongation']:.2f}  vox={p['voxels']}  "
                  f"pos=[{cv[0]:.0f},{cv[1]:.0f},{cv[2]:.0f}]  [{tag}]")
        if r["matched"]:
            print(f"  Match details:")
            for gi, pi in r["matched"]:
                g = r["gt"][gi]; p = r["preds"][pi]
                gc = np.array(g["centroid_zyx_voxel"]) * np.array(r["voxel_sp"])
                d = np.linalg.norm(gc - p["centroid_mm"])
                size_err = abs(p["diameter_mm"] - g["diam_mm"])
                print(f"    GT#{gi} ({g['diam_mm']:.1f}mm) <- AI#{pi} ({p['diameter_mm']:.1f}mm)  "
                      f"  centroid err={d:.1f}mm  size err={size_err:.1f}mm")

    n_pat = len(rows)
    sens = grand_tp / max(grand_n_gt, 1)
    fpps = grand_fp / max(n_pat, 1)
    prec = grand_tp / max(grand_tp + grand_fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-7)
    print(f"\n{'=' * 60}")
    print(f"=== AGGREGATE ({n_pat} patients) ===")
    print(f"  TP/FP/FN:     {grand_tp} / {grand_fp} / {grand_fn}")
    print(f"  Sensitivity:  {sens:.3f}  ({grand_tp}/{grand_n_gt})")
    print(f"  Precision:    {prec:.3f}")
    print(f"  F1:           {f1:.3f}")
    print(f"  FP/scan:      {fpps:.2f}")

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps({
            "panel": args.panel, "n_patients": n_pat,
            "config": cfg, "threshold": thr,
            "tp": grand_tp, "fp": grand_fp, "fn": grand_n_gt - grand_tp,
            "sensitivity": sens, "precision": prec, "f1": f1,
            "fp_per_scan": fpps,
            "per_patient": [{
                "pid": r["pid"], "n_gt": r["n_gt"], "n_pred": r["n_pred"],
                "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            } for r in rows],
        }, indent=2, default=str))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
