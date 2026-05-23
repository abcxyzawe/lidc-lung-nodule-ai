"""Quick rescue eval for LUNA16 fine-tuned model.

Loads 95 cached prob volumes from work/academic/probs_luna16_ft/.
Tests 4 thresholds only. Uses luna16_radius matching (per existing eval).
Writes work/runs_luna16/run001/QUICK_EVAL.json.
No FROC sweep, no bootstrap CI.

Run:
    python src/quick_eval_ft.py
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    binary_closing, binary_dilation, binary_erosion,
    distance_transform_edt, label as cc_label,
)

# Paths
ROOT = Path("E:/Phan Tich Ung Thu")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "webapp"))

from configs import WORK, PRE_DIR  # noqa: E402
from benchmark import load_patient  # noqa: E402
from tune_detection_params import (  # noqa: E402
    precompute_threshold_blobs,
    filter_and_match,
    cache_lung_mask,
)

PROBS_DIR = WORK / "academic" / "probs_luna16_ft"
OUT_DIR = WORK / "runs_luna16" / "run001"
OUT_JSON = OUT_DIR / "QUICK_EVAL.json"

# 4 thresholds only — fast eval
THRESHOLDS = [0.30, 0.50, 0.65, 0.85]

# Post-proc params — same as eval_finetune_on_lidc.py
POST_CFG = {
    "min_voxels": 200,
    "max_elong": 4.0,
    "merge_dist_mm": 10.0,
    "subpleural_min_mm": 2.0,
}

# Reference numbers for verdict
MINE_PRE_F1 = 0.5551  # pre-FT fixed_15mm best F1
MONAI_F1    = 0.5455  # MONAI fixed_15mm best F1


def _match_fixed_15mm(preds, gt_nodules, voxel_sp):
    """Match using fixed 15mm centroid distance (non-official, conservative)."""
    sp = np.array(voxel_sp, dtype=np.float32)
    used_p = set()
    tp = 0
    for g in gt_nodules:
        gc = np.array(g["centroid_zyx_voxel"], dtype=np.float32) * sp
        best_pi, best_d = None, float("inf")
        for pi, p in enumerate(preds):
            if pi in used_p:
                continue
            d = float(np.linalg.norm(gc - p["centroid_mm"]))
            if d < best_d:
                best_d, best_pi = d, pi
        if best_pi is not None and best_d <= 15.0:
            tp += 1
            used_p.add(best_pi)
    fp = len(preds) - tp
    fn = len(gt_nodules) - tp
    return tp, fp, fn


def score_one_threshold(patient_data, thr):
    """Compute aggregate sens/prec/F1/FP-per-scan at one threshold.

    Uses luna16_radius matching (primary) and fixed_15mm (secondary).
    Returns dict with both rule results.
    """
    tp_lr = fp_lr = fn_lr = 0
    tp_f15 = fp_f15 = fn_f15 = 0
    n_gt_total = 0

    for pd in patient_data:
        try:
            blobs = precompute_threshold_blobs(
                pd["prob"], pd["lung"], pd["voxel_sp"], thr
            )
            res = filter_and_match(
                blobs, pd["ldist"], pd["voxel_sp"], pd["gt"],
                min_voxels=POST_CFG["min_voxels"],
                max_elong=POST_CFG["max_elong"],
                merge_dist_mm=POST_CFG["merge_dist_mm"],
                subpleural_min_mm=POST_CFG["subpleural_min_mm"],
                match_min_radius_mm=3.0,
            )
            # luna16_radius result from filter_and_match
            tp_lr += res["tp"]
            fp_lr += res["fp"]
            fn_lr += res["fn"]

            # fixed_15mm re-match on same preds list
            ptp, pfp, pfn = _match_fixed_15mm(
                res["preds"], pd["gt"], pd["voxel_sp"]
            )
            tp_f15 += ptp
            fp_f15 += pfp
            fn_f15 += pfn

            n_gt_total += len(pd["gt"])
        except Exception as exc:
            print(f"  WARN: {pd['pid']} thr={thr} failed: {exc}", flush=True)

    n_scans = len(patient_data)

    def _metrics(tp, fp, fn, n_gt, n_s):
        sens = tp / max(n_gt, 1)
        prec = tp / max(tp + fp, 1)
        f1   = 2 * prec * sens / max(prec + sens, 1e-9)
        fps  = fp / max(n_s, 1)
        return {"sens": round(sens, 4), "prec": round(prec, 4),
                "f1": round(f1, 4), "fp_per_scan": round(fps, 3),
                "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt}

    lr  = _metrics(tp_lr,  fp_lr,  fn_lr,  n_gt_total, n_scans)
    f15 = _metrics(tp_f15, fp_f15, fn_f15, n_gt_total, n_scans)
    return {"thr": thr, "luna16_radius": lr, "fixed_15mm": f15}


def load_bundles():
    """Load all 95 cached patients.  Returns list of bundle dicts."""
    prob_files = sorted(PROBS_DIR.glob("LIDC-IDRI-*.npy"))
    print(f"Found {len(prob_files)} cached prob files in {PROBS_DIR}", flush=True)
    bundles = []
    failed = []
    for i, pf in enumerate(prob_files):
        pid = pf.stem  # e.g. LIDC-IDRI-0003
        try:
            data = load_patient(pid)
            if data is None:
                print(f"  [{i+1}/{len(prob_files)}] {pid}: no h5 — skip", flush=True)
                failed.append(pid)
                continue
            prob = np.load(pf).astype(np.float32)
            lung = cache_lung_mask(pid, data["vol"])
            sp   = data["voxel_sp"]
            ldist = distance_transform_edt(lung, sampling=list(sp)).astype(np.float32)
            bundles.append({
                "pid": pid, "prob": prob, "lung": lung,
                "ldist": ldist, "voxel_sp": sp, "gt": data["gt_nodules"],
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(prob_files):
                print(f"  Loaded {i+1}/{len(prob_files)} ...", flush=True)
        except Exception as exc:
            print(f"  [{i+1}/{len(prob_files)}] {pid}: ERROR {exc}", flush=True)
            failed.append(pid)
    n_gt = sum(len(b["gt"]) for b in bundles)
    print(f"\nLoaded {len(bundles)} patients  GT nodules: {n_gt}  "
          f"Failed/skipped: {len(failed)}", flush=True)
    return bundles, n_gt, failed


def verdict(best_f1):
    """Map best F1 to one of 4 verdict categories."""
    if best_f1 < 0.40:
        cat = "CATASTROPHIC_FORGETTING"
        rec = "ROLLBACK — use pre-FT winner ckpt cd48359 for thesis"
    elif best_f1 < 0.55:
        cat = "REGRESSION"
        rec = "No catastrophic forgetting but regression; run2 ablation needed"
    elif best_f1 < 0.60:
        cat = "NEUTRAL"
        rec = "Fine-tune neutral (no harm, no gain); can use either ckpt"
    else:
        cat = "IMPROVED"
        rec = "Fine-tune improved; declare success, publish post-FT results"
    return cat, rec


def main():
    result = {
        "eval_date": "2026-05-22",
        "probs_dir": str(PROBS_DIR),
        "thresholds_tested": THRESHOLDS,
        "post_proc": POST_CFG,
        "reference": {
            "mine_pre_fixed15mm_f1": MINE_PRE_F1,
            "monai_fixed15mm_f1": MONAI_F1,
        },
        "rows": [],
        "best_luna16_radius": None,
        "best_fixed_15mm": None,
        "verdict": None,
        "partial": False,
    }

    try:
        # Step 1: Load bundles
        print("=" * 60, flush=True)
        print("[Step 1] Loading patient bundles ...", flush=True)
        bundles, n_gt, failed = load_bundles()
        result["n_patients"] = len(bundles)
        result["n_gt"] = n_gt
        result["failed_pids"] = failed

        if len(bundles) == 0:
            print("ERROR: no bundles loaded — cannot eval", flush=True)
            result["error"] = "no bundles loaded"
            return result

        # Step 2: Score 4 thresholds
        print(f"\n[Step 2] Scoring {len(THRESHOLDS)} thresholds ...", flush=True)
        print(f"\n{'thr':>6} | {'rule':>14} | {'sens':>6} | {'prec':>6} | {'F1':>6} | {'FP/scan':>7}", flush=True)
        print("-" * 58, flush=True)

        rows = []
        for thr in THRESHOLDS:
            r = score_one_threshold(bundles, thr)
            rows.append(r)
            for rule_key in ("luna16_radius", "fixed_15mm"):
                m = r[rule_key]
                print(
                    f"{thr:>6.2f} | {rule_key:>14} | "
                    f"{m['sens']:>6.3f} | {m['prec']:>6.3f} | "
                    f"{m['f1']:>6.3f} | {m['fp_per_scan']:>7.2f}",
                    flush=True
                )
            print("", flush=True)

        result["rows"] = rows

        # Step 3: Find best per rule
        best_lr  = max(rows, key=lambda r: r["luna16_radius"]["f1"])
        best_f15 = max(rows, key=lambda r: r["fixed_15mm"]["f1"])
        result["best_luna16_radius"] = best_lr
        result["best_fixed_15mm"]    = best_f15

        bf1_f15 = best_f15["fixed_15mm"]["f1"]
        bf1_lr  = best_lr["luna16_radius"]["f1"]

        print("=" * 60, flush=True)
        print(f"[Results] Best F1 (fixed_15mm)    = {bf1_f15:.4f}  "
              f"@ thr={best_f15['thr']}  "
              f"(pre-FT ref={MINE_PRE_F1:.4f}  MONAI ref={MONAI_F1:.4f})", flush=True)
        print(f"[Results] Best F1 (luna16_radius) = {bf1_lr:.4f}  "
              f"@ thr={best_lr['thr']}", flush=True)

        # Step 4: Verdict — use fixed_15mm F1 as primary (consistent with pre-FT eval)
        cat, rec = verdict(bf1_f15)
        result["verdict"] = cat
        result["recommendation"] = rec

        print(f"\n[Verdict] Best post-FT F1 (fixed_15mm) = {bf1_f15:.4f}", flush=True)
        print(f"[Verdict] {cat}", flush=True)
        print(f"[Decision] {rec}", flush=True)

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\nCRITICAL ERROR: {exc}\n{tb}", flush=True)
        result["error"] = str(exc)
        result["traceback"] = tb
        result["partial"] = True

    finally:
        # Always write JSON — even on partial / crash
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved: {OUT_JSON}", flush=True)

    return result


if __name__ == "__main__":
    main()
