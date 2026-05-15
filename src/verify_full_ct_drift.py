"""Phase B — Verify distribution shift on full CT (10-20 patients).

Extends verify_full_vs_preprocessed.py with:
  - 10-20 patients (not 3) — statistical signal
  - Full final-detection metrics on full CT (not just slice fire counts)
  - FP location classification: apex / base / heart / mid-lung / vessel-heavy
  - Decision report: should we add negative slice sampler?

Decision criteria (per user plan B):
  Trigger negative-sampler retrain IF either:
    (a) FP/scan_full > FP/scan_preprocessed + 0.5
    (b) >30% of FPs land in apex/base/heart/diaphragm zones (i.e., outside the
        plausible nodule region the model saw in training)

Run:
  python verify_full_ct_drift.py                          # default 15 test patients
  python verify_full_ct_drift.py --n-patients 20
  python verify_full_ct_drift.py --patients LIDC-IDRI-0001,LIDC-IDRI-0002,...
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import h5py
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from configs import DICOM_ROOT, PRE_DIR, WORK
from panels import load_panel
from benchmark import load_patient
from predict import (
    read_dicom_series, predict_nodules, segment_lung,
    clean_mask, find_nodules, merge_nearby_nodules, filter_subpleural,
    get_model, get_swa_model,
)


REPORT_PATH = WORK / "academic" / "FULL_CT_DRIFT_VERIFY.md"


def find_dicoms(pid: str) -> list:
    p_dir = DICOM_ROOT / pid
    if not p_dir.exists(): return []
    by_series = defaultdict(list)
    for dp in p_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dp), stop_before_pixels=True, force=True,
                                  specific_tags=["SeriesInstanceUID"])
            by_series[str(ds.SeriesInstanceUID)].append(str(dp))
        except Exception:
            continue
    if not by_series: return []
    best_uid = max(by_series, key=lambda k: len(by_series[k]))
    return by_series[best_uid]


def predict_with_postproc(vol, voxel_sp, threshold=0.97, min_voxels=120,
                           max_elong=4.0, merge_dist_mm=10.0, subpleural_min_mm=0.0):
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


def classify_fp_location(centroid_voxel, n_slices_total: int,
                          lung_mask: np.ndarray) -> str:
    """Classify a predicted nodule centroid into anatomical zones.

    Heuristic based on z-position fraction (apex=top of CT, base=bottom):
      apex     : top 12.5% of slices
      upper    : 12.5%-37.5%
      mid      : 37.5%-62.5%
      lower    : 62.5%-87.5%
      base     : bottom 12.5%
    Plus check lung_mask:
      heart    : centroid is inside the heart shadow (no lung at this z+y)
      vessel   : centroid is in lung but very near the mediastinum (centre)
    """
    cz, cy, cx = [int(round(c)) for c in centroid_voxel]
    z_frac = cz / max(n_slices_total - 1, 1)
    z_zone = ("apex" if z_frac < 0.125 else
              "upper" if z_frac < 0.375 else
              "mid" if z_frac < 0.625 else
              "lower" if z_frac < 0.875 else
              "base")
    # Heart check: at this z, is there lung at the centroid (cy,cx)?
    cz = max(0, min(lung_mask.shape[0]-1, cz))
    cy = max(0, min(lung_mask.shape[1]-1, cy))
    cx = max(0, min(lung_mask.shape[2]-1, cx))
    in_lung = bool(lung_mask[cz, cy, cx])
    H = lung_mask.shape[1]; W = lung_mask.shape[2]
    near_mediastinum = abs(cx - W//2) < W * 0.12  # within ±12% of x-centre
    if not in_lung:
        return f"{z_zone}/heart-or-outside"
    if near_mediastinum and z_zone in ("mid", "upper"):
        return f"{z_zone}/vessel-mediastinum"
    return z_zone


def match_to_gt(preds, gts, voxel_sp, min_radius_mm=5.0):
    """LUNA16-radius match. Returns list of (pred_idx, gt_idx | None) for each pred."""
    sp = np.array(voxel_sp, dtype=np.float32)
    if not gts:
        return [(i, None) for i in range(len(preds))]
    used_g = set()
    results = []
    for pi, p in enumerate(preds):
        pc = np.array(p["centroid_zyx_voxel"]) * sp
        best_gi, best_d = None, float("inf")
        for gi, g in enumerate(gts):
            if gi in used_g: continue
            gc = np.array(g["centroid_zyx_voxel"]) * sp
            tol = max(g["diam_mm"] / 2, min_radius_mm)
            d = np.linalg.norm(pc - gc)
            if d < tol and d < best_d:
                best_d, best_gi = d, gi
        if best_gi is not None:
            used_g.add(best_gi)
            results.append((pi, best_gi))
        else:
            results.append((pi, None))
    return results


def analyze_one(pid: str):
    print(f"\n--- {pid} ---")
    dcm_paths = find_dicoms(pid)
    if not dcm_paths: return None

    # Full CT
    vol_full, voxel_sp = read_dicom_series(dcm_paths)
    N_full = vol_full.shape[0]

    # Preprocessed h5
    h5p = PRE_DIR / f"{pid}.h5"
    if not h5p.exists(): return None
    with h5py.File(h5p) as f:
        key = next(iter(f.keys()))
        vol_pre = f[key]["images"][:]
        pix_sp = json.loads(f[key].attrs["pixel_spacing"])
        slice_thk = float(f[key].attrs["slice_thickness"])
    voxel_sp_pre = (slice_thk, pix_sp[0], pix_sp[1])

    data = load_patient(pid)
    gts = data["gt_nodules"] if data else []

    get_model(); get_swa_model()

    # FULL CT prediction
    nods_full, prob_full, lung_full = predict_with_postproc(vol_full, voxel_sp)
    matches_full = match_to_gt(nods_full, gts, voxel_sp)
    n_tp_full = sum(1 for _, gi in matches_full if gi is not None)
    n_fp_full = len(nods_full) - n_tp_full
    fp_zones = []
    for pi, gi in matches_full:
        if gi is None:  # FP
            zone = classify_fp_location(nods_full[pi]["centroid_zyx_voxel"],
                                          N_full, lung_full)
            fp_zones.append(zone)

    # PREPROCESSED prediction
    nods_pre, _, lung_pre = predict_with_postproc(vol_pre, voxel_sp_pre)
    matches_pre = match_to_gt(nods_pre, gts, voxel_sp_pre)
    n_tp_pre = sum(1 for _, gi in matches_pre if gi is not None)
    n_fp_pre = len(nods_pre) - n_tp_pre

    print(f"  Full CT: {N_full} slices, {len(nods_full)} preds (TP={n_tp_full} FP={n_fp_full})")
    print(f"  Preproc: {vol_pre.shape[0]} slices, {len(nods_pre)} preds (TP={n_tp_pre} FP={n_fp_pre})")
    print(f"  GT: {len(gts)} nodules")
    if fp_zones:
        from collections import Counter
        zc = Counter(fp_zones)
        print(f"  FP zones (full CT): {dict(zc)}")

    return {
        "pid": pid,
        "n_slices_full": N_full,
        "n_slices_pre": vol_pre.shape[0],
        "n_gt": len(gts),
        "full": {"n_pred": len(nods_full), "tp": n_tp_full, "fp": n_fp_full,
                  "fp_zones": fp_zones},
        "pre": {"n_pred": len(nods_pre), "tp": n_tp_pre, "fp": n_fp_pre},
    }


def write_report(rows):
    n_pat = len(rows)
    if n_pat == 0:
        print("No data to report"); return

    total_n_full = sum(r["full"]["n_pred"] for r in rows)
    total_tp_full = sum(r["full"]["tp"] for r in rows)
    total_fp_full = sum(r["full"]["fp"] for r in rows)
    total_n_pre = sum(r["pre"]["n_pred"] for r in rows)
    total_tp_pre = sum(r["pre"]["tp"] for r in rows)
    total_fp_pre = sum(r["pre"]["fp"] for r in rows)
    total_gt = sum(r["n_gt"] for r in rows)

    fp_per_scan_full = total_fp_full / n_pat
    fp_per_scan_pre = total_fp_pre / n_pat
    sens_full = total_tp_full / max(total_gt, 1)
    sens_pre = total_tp_pre / max(total_gt, 1)
    prec_full = total_tp_full / max(total_n_full, 1)
    prec_pre = total_tp_pre / max(total_n_pre, 1)

    # FP zone breakdown
    from collections import Counter
    all_zones = []
    for r in rows: all_zones.extend(r["full"]["fp_zones"])
    zc = Counter(all_zones)
    n_zones = sum(zc.values())
    risky_zones = sum(v for z, v in zc.items()
                       if "apex" in z or "base" in z or "heart" in z or "mediastinum" in z)
    risky_pct = 100 * risky_zones / max(n_zones, 1)

    # Decision
    delta_fp = fp_per_scan_full - fp_per_scan_pre
    trigger_negsampler = (delta_fp > 0.5) or (risky_pct > 30)

    md = [
        "# Full-CT distribution shift verification",
        f"\nGenerated {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## Aggregate ({n_pat} test patients, {total_gt} GT nodules)\n",
        "| Metric | Full CT | Preprocessed |",
        "|---|---:|---:|",
        f"| Total predictions | {total_n_full} | {total_n_pre} |",
        f"| TP | {total_tp_full} | {total_tp_pre} |",
        f"| FP | {total_fp_full} | {total_fp_pre} |",
        f"| Sensitivity | {sens_full:.3f} | {sens_pre:.3f} |",
        f"| Precision | {prec_full:.3f} | {prec_pre:.3f} |",
        f"| FP/scan | {fp_per_scan_full:.2f} | {fp_per_scan_pre:.2f} |",
        f"\n**Δ FP/scan (full − pre): {delta_fp:+.2f}**",
        f"\n## FP zone breakdown (full CT only)\n",
        "| Zone | Count | % |",
        "|---|---:|---:|",
    ]
    for zone, count in sorted(zc.items(), key=lambda x: -x[1]):
        md.append(f"| {zone} | {count} | {100*count/n_zones:.1f}% |")
    md.append(f"\n**Risky-zone FP rate (apex/base/heart/mediastinum): {risky_pct:.1f}%**\n")

    md.append("## Decision\n")
    md.append("Triggers for retraining with negative slice sampler:")
    md.append("- (a) ΔFP/scan > 0.5 — " + ("✅ TRIGGERED" if delta_fp > 0.5 else "❌ not triggered"))
    md.append("- (b) Risky-zone FP > 30% — " +
              ("✅ TRIGGERED" if risky_pct > 30 else "❌ not triggered"))
    md.append("")
    if trigger_negsampler:
        md.append("**→ RUN re-preprocess + EXP01/EXP04/EXP05 (with negative sampler)**")
    else:
        md.append("**→ SKIP negative sampler. Distribution shift not significant.**")
        md.append("**→ Focus on EXP02/EXP03 (consensus2, Tversky) only.**")

    md.append("\n## Per-patient detail\n")
    md.append("| PID | full slices | pre slices | GT | full TP/FP | pre TP/FP |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['pid']} | {r['n_slices_full']} | {r['n_slices_pre']} | {r['n_gt']} "
            f"| {r['full']['tp']}/{r['full']['fp']} | {r['pre']['tp']}/{r['pre']['fp']} |"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(md))
    print(f"\nWrote {REPORT_PATH}")
    return {
        "delta_fp_per_scan": delta_fp,
        "risky_zone_fp_pct": risky_pct,
        "trigger_negsampler": trigger_negsampler,
        "n_patients": n_pat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=15,
                    help="How many test_panel patients to verify")
    ap.add_argument("--patients", default="",
                    help="Comma-separated PIDs (overrides --n-patients)")
    args = ap.parse_args()

    if args.patients:
        pids = [p.strip() for p in args.patients.split(",") if p.strip()]
    else:
        pids = load_panel("test")[:args.n_patients]

    print(f"Verifying {len(pids)} patients")
    rows = []
    t0 = time.time()
    for pid in pids:
        try:
            r = analyze_one(pid)
            if r: rows.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED {pid}: {e}")
    print(f"\nTotal {time.time()-t0:.0f}s for {len(rows)} patients\n")
    decision = write_report(rows)
    if decision:
        print(f"\n=== DECISION ===")
        print(f"  ΔFP/scan: {decision['delta_fp_per_scan']:+.2f}")
        print(f"  Risky-zone FP: {decision['risky_zone_fp_pct']:.1f}%")
        print(f"  → Negative sampler retraining: "
              f"{'YES' if decision['trigger_negsampler'] else 'NO'}")


if __name__ == "__main__":
    main()
