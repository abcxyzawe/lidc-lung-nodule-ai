"""Verify: does running AI on FULL CT (all DICOM slices) give different results
than running on preprocessed h5 (only nodule±3 slices)?

For each test patient:
  1. Load FULL CT from DICOM (all slices, no filtering)
  2. Load preprocessed h5 (kept slices only)
  3. Run AI on both with SAME model + post-proc config
  4. Compare:
     - Total slices                               (full vs kept count)
     - Nodule predictions                          (count, location, size)
     - Slices with FP outside known-nodule regions (potential drift cause)

Run: python verify_full_vs_preprocessed.py --patients LIDC-IDRI-0001,LIDC-IDRI-0002
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from configs import DICOM_ROOT, PRE_DIR
from benchmark import load_patient
from predict import (
    read_dicom_series, predict_nodules, segment_lung,
    clean_mask, find_nodules, merge_nearby_nodules, filter_subpleural,
    get_model, get_swa_model,
)


def find_dicoms(pid: str) -> list:
    """Find all .dcm files for a patient (one series)."""
    p_dir = DICOM_ROOT / pid
    if not p_dir.exists(): return []
    # Group by SeriesInstanceUID, return the one with most slices
    from collections import defaultdict
    by_series = defaultdict(list)
    for dp in p_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dp), stop_before_pixels=True, force=True,
                                  specific_tags=["SeriesInstanceUID"])
            by_series[str(ds.SeriesInstanceUID)].append(str(dp))
        except Exception:
            continue
    if not by_series: return []
    # Pick the series with most slices (CT axial usually)
    best_uid = max(by_series, key=lambda k: len(by_series[k]))
    return by_series[best_uid]


def predict_with_postproc(vol, voxel_sp, threshold=0.97, min_voxels=120,
                          max_elong=4.0, merge_dist_mm=10.0, subpleural_min_mm=0.0):
    """Full pipeline: predict + lung mask + post-proc, return nodules + prob_vol."""
    lung = segment_lung(vol)
    prob, pred = predict_nodules(vol, threshold=threshold, tta=True, ensemble=True)
    pred = (pred & lung).astype(np.uint8)
    pred = clean_mask(pred, voxel_sp)
    nodules, labeled = find_nodules(pred, voxel_sp, min_voxels=min_voxels,
                                     max_elongation=max_elong,
                                     prob_volume=prob, core_threshold=0.85)
    nodules = merge_nearby_nodules(nodules, voxel_sp, max_dist_mm=merge_dist_mm)
    nodules = filter_subpleural(nodules, lung, voxel_sp, min_dist_mm=subpleural_min_mm)
    return nodules, prob, lung


def compare_one(pid: str):
    print(f"\n{'='*70}\n=== {pid} ===\n{'='*70}")

    # 1. Full CT from DICOM
    dcm_paths = find_dicoms(pid)
    if not dcm_paths:
        print(f"  No DICOM found")
        return None
    print(f"  Full DICOM: {len(dcm_paths)} slices")
    vol_full, voxel_sp_full = read_dicom_series(dcm_paths)
    print(f"  Full volume shape: {vol_full.shape}, voxel_sp={voxel_sp_full}")

    # 2. Preprocessed h5
    h5p = PRE_DIR / f"{pid}.h5"
    if not h5p.exists():
        print(f"  No preprocessed h5"); return None
    with h5py.File(h5p) as f:
        key = next(iter(f.keys()))
        vol_pre = f[key]["images"][:]
        pix_sp = json.loads(f[key].attrs["pixel_spacing"])
        slice_thk = float(f[key].attrs["slice_thickness"])
    voxel_sp_pre = (slice_thk, pix_sp[0], pix_sp[1])
    print(f"  Preprocessed h5: {vol_pre.shape[0]} slices ({100*vol_pre.shape[0]/len(dcm_paths):.0f}% of full)")

    # GT
    data = load_patient(pid)
    gts = data["gt_nodules"] if data else []
    print(f"  GT (kept-h5 view): {len(gts)} nodules")
    for g in gts:
        print(f"    GT  {g['diam_mm']:5.1f}mm @ {g['centroid_zyx_voxel']} (rads {g['n_radiologists']}/4)")

    # Warm models once
    get_model(); get_swa_model()

    # 3. Predict on full CT
    print(f"\n  [PRED FULL CT]")
    nods_full, prob_full, lung_full = predict_with_postproc(vol_full, voxel_sp_full)
    print(f"    {len(nods_full)} nodules predicted")
    for n in nods_full:
        print(f"    AI  {n['diameter_mm']:5.1f}mm vox={n['voxels']} elong={n['elongation']:.2f} "
              f"@ [{n['centroid_zyx_voxel'][0]:.0f},{n['centroid_zyx_voxel'][1]:.0f},"
              f"{n['centroid_zyx_voxel'][2]:.0f}]")

    # 4. Predict on preprocessed (kept) slices
    print(f"\n  [PRED PREPROCESSED (kept slices only)]")
    nods_pre, prob_pre, _ = predict_with_postproc(vol_pre, voxel_sp_pre)
    print(f"    {len(nods_pre)} nodules predicted")
    for n in nods_pre:
        print(f"    AI  {n['diameter_mm']:5.1f}mm vox={n['voxels']} elong={n['elongation']:.2f} "
              f"@ [{n['centroid_zyx_voxel'][0]:.0f},{n['centroid_zyx_voxel'][1]:.0f},"
              f"{n['centroid_zyx_voxel'][2]:.0f}]")

    # 5. Aggregate prob distribution per slice
    print(f"\n  [PROB STATS]")
    print(f"    Full CT: mean={prob_full.mean():.4f}  max={prob_full.max():.4f}")
    print(f"             slices with any prob>0.5: {(prob_full.max(axis=(1,2)) > 0.5).sum()}/{prob_full.shape[0]}")
    print(f"             slices with any prob>0.97: {(prob_full.max(axis=(1,2)) > 0.97).sum()}/{prob_full.shape[0]}")
    print(f"    Pre h5:  mean={prob_pre.mean():.4f}  max={prob_pre.max():.4f}")
    print(f"             slices with any prob>0.5: {(prob_pre.max(axis=(1,2)) > 0.5).sum()}/{prob_pre.shape[0]}")
    print(f"             slices with any prob>0.97: {(prob_pre.max(axis=(1,2)) > 0.97).sum()}/{prob_pre.shape[0]}")

    return {
        "pid": pid,
        "n_slices_full": int(len(dcm_paths)),
        "n_slices_pre": int(vol_pre.shape[0]),
        "n_pred_full": len(nods_full),
        "n_pred_pre": len(nods_pre),
        "n_gt": len(gts),
        "voxel_sp_full": voxel_sp_full,
        "voxel_sp_pre": voxel_sp_pre,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients",
                    default="LIDC-IDRI-0001,LIDC-IDRI-0002,LIDC-IDRI-0003,LIDC-IDRI-0094")
    args = ap.parse_args()

    pids = [p.strip() for p in args.patients.split(",") if p.strip()]
    rows = []
    for pid in pids:
        try:
            r = compare_one(pid)
            if r: rows.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED on {pid}: {e}")

    print(f"\n{'='*70}\n=== SUMMARY ===\n{'='*70}")
    print(f"{'PID':<18} {'slices_full':>12} {'slices_pre':>11} {'pred_full':>10} {'pred_pre':>9} {'n_gt':>5}")
    for r in rows:
        print(f"  {r['pid']:<18} {r['n_slices_full']:>12} {r['n_slices_pre']:>11} "
              f"{r['n_pred_full']:>10} {r['n_pred_pre']:>9} {r['n_gt']:>5}")


if __name__ == "__main__":
    main()
