"""For each GT nodule on test_panel, probe the model's prob volume at the GT centroid.

Outputs distribution of:
  - peak prob in 5-voxel sphere around GT centroid
  - whether nodule is "invisible" (peak < 0.3), "weak" (0.3-0.7), or "visible" (>=0.7)
  - stratified by GT size, n_radiologists, malignancy

Use to understand the model's blind spots — which nodule types need retraining attention.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from configs import WORK  # noqa: E402
from panels import load_panel  # noqa: E402
from benchmark import load_patient  # noqa: E402

PROBS_DIR = WORK / "academic" / "probs"


def probe_centroid(prob_vol, centroid, r=8):
    cz, cy, cx = [int(round(c)) for c in centroid]
    z0, z1 = max(0, cz-3), min(prob_vol.shape[0], cz+4)
    y0, y1 = max(0, cy-r), min(prob_vol.shape[1], cy+r+1)
    x0, x1 = max(0, cx-r), min(prob_vol.shape[2], cx+r+1)
    roi = prob_vol[z0:z1, y0:y1, x0:x1]
    return float(roi.max()) if roi.size > 0 else 0.0


def main():
    pids = load_panel("test")
    rows = []
    for pid in pids:
        cp = PROBS_DIR / f"{pid}_tta1.npy"
        if not cp.exists(): continue
        d = load_patient(pid)
        if d is None: continue
        prob = np.load(cp).astype(np.float32)
        for g in d["gt_nodules"]:
            peak = probe_centroid(prob, g["centroid_zyx_voxel"])
            rows.append({
                "pid": pid,
                "diam_mm": g["diam_mm"],
                "n_rads": g["n_radiologists"],
                "malignancy": g.get("malignancy_mean"),
                "peak_prob": peak,
                "visibility": "invisible" if peak < 0.3 else ("weak" if peak < 0.7 else "visible"),
            })
    print(f"Total GT nodules probed: {len(rows)}")
    n_inv = sum(1 for r in rows if r["visibility"] == "invisible")
    n_weak = sum(1 for r in rows if r["visibility"] == "weak")
    n_vis = sum(1 for r in rows if r["visibility"] == "visible")
    print(f"  invisible (peak<0.3): {n_inv} ({100*n_inv/len(rows):.1f}%)")
    print(f"  weak (0.3-0.7):       {n_weak} ({100*n_weak/len(rows):.1f}%)")
    print(f"  visible (>=0.7):      {n_vis} ({100*n_vis/len(rows):.1f}%)")
    print()
    # Stratified by size
    print("=== By size ===")
    for lo, hi, name in [(3, 6, "small 3-6mm"), (6, 10, "medium 6-10mm"),
                         (10, 20, "med-large 10-20mm"), (20, 999, "large >=20mm")]:
        bucket = [r for r in rows if lo <= r["diam_mm"] < hi]
        if not bucket: continue
        n_i = sum(1 for r in bucket if r["visibility"] == "invisible")
        n_v = sum(1 for r in bucket if r["visibility"] == "visible")
        print(f"  {name:<22} n={len(bucket):>3}  visible={n_v:>3} ({100*n_v/len(bucket):>5.1f}%)  "
              f"invisible={n_i:>3} ({100*n_i/len(bucket):>5.1f}%)")
    print()
    # Stratified by n_radiologists (consensus)
    print("=== By radiologist consensus ===")
    for nr in [1, 2, 3, 4]:
        bucket = [r for r in rows if r["n_rads"] == nr]
        if not bucket: continue
        n_i = sum(1 for r in bucket if r["visibility"] == "invisible")
        n_v = sum(1 for r in bucket if r["visibility"] == "visible")
        print(f"  {nr}/4 rads             n={len(bucket):>3}  visible={n_v:>3} "
              f"({100*n_v/len(bucket):>5.1f}%)  invisible={n_i:>3} ({100*n_i/len(bucket):>5.1f}%)")
    print()
    # Stratified by malignancy
    print("=== By GT malignancy ===")
    for lo, hi, name in [(0, 2.5, "low (1-2)"), (2.5, 3.5, "indeterminate (2.5-3.5)"),
                         (3.5, 5.1, "high (3.5-5)")]:
        bucket = [r for r in rows if r["malignancy"] is not None and lo <= r["malignancy"] < hi]
        if not bucket: continue
        n_i = sum(1 for r in bucket if r["visibility"] == "invisible")
        n_v = sum(1 for r in bucket if r["visibility"] == "visible")
        print(f"  {name:<25} n={len(bucket):>3}  visible={n_v:>3} ({100*n_v/len(bucket):>5.1f}%)  "
              f"invisible={n_i:>3} ({100*n_i/len(bucket):>5.1f}%)")

    # Save
    out = WORK / "academic" / "missed_nodules_analysis.json"
    out.write_text(json.dumps({
        "n_total": len(rows), "summary": {"invisible": n_inv, "weak": n_weak, "visible": n_vis},
        "rows": rows,
    }, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
