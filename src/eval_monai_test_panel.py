"""MONAI vs Mine head-to-head benchmark on the locked test_panel (99 patients).

Steps
------
1. Loop 99 test patients -> run predict_nodules_monai -> save predictions JSON.
   Also loop 99 val patients -> save for op-point sweep.
2. Sweep score_thresh on val_panel -> pick best F1 operating point for MONAI.
3. Compute on test_panel at that op-point: F1, recall, precision, FP/scan.
4. FROC + CPM (7 standard FP/scan rates, fixed-15mm matching -- same rule as mine).
5. Stratified by nodule size: 3-6 / 6-10 / 10-20 / >=20 mm.
6. Bootstrap 95% CI (1000 resamples, paired) + Wilcoxon on per-patient F1.
7. Write work/academic/MONAI_VS_MINE.md.

Matching rule: fixed 15 mm centroid distance (same as honest eval, commit db08c2e).
Mine results loaded from: work/academic/fpr_v3_stage2winner_eval_test.json (thr=0.85).

Run:
    python src/eval_monai_test_panel.py [--skip-val-inference] [--skip-test-inference]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "webapp"))

from configs import WORK, SPLITS_JSON  # noqa: E402
from panels import load_panel  # noqa: E402
from benchmark import load_patient  # noqa: E402
from academic_bench import match_predictions_to_gt  # noqa: E402
from webapp.predict_monai import predict_nodules_monai  # noqa: E402

ACADEMIC_DIR = WORK / "academic"
MONAI_PREDS_TEST = ACADEMIC_DIR / "monai_test_panel_preds.json"
MONAI_PREDS_VAL  = ACADEMIC_DIR / "monai_val_panel_preds.json"
REPORT_MD        = ACADEMIC_DIR / "MONAI_VS_MINE.md"

LUNA16_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SCORE_THRESH_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
MATCH_TOL_MM = 15.0  # fixed 15 mm -- same as honest eval (commit db08c2e)

SIZE_BUCKETS = {
    "3-6mm":  (3.0,  6.0),
    "6-10mm": (6.0, 10.0),
    "10-20mm":(10.0, 20.0),
    ">=20mm": (20.0, 9999.0),
}

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_monai_on_panel(pids: list, out_path: Path, desc: str) -> dict:
    """Predict all patients in panel, cache to JSON. Returns {pid: [nodules]}.

    Saves incrementally after every patient so a crash can be resumed.
    """
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        done_pids = set(existing.get("per_patient", {}).keys())
    else:
        existing = {"per_patient": {}, "timing": {}}
        done_pids = set()

    per_patient: dict = existing["per_patient"]
    timing: dict = existing["timing"]
    todo = [p for p in pids if p not in done_pids]

    if not todo:
        print(f"  [{desc}] All {len(pids)} patients already cached in {out_path.name}.")
        return per_patient

    print(f"  [{desc}] {len(done_pids)} cached, {len(todo)} to run ...", flush=True)
    ACADEMIC_DIR.mkdir(parents=True, exist_ok=True)

    for i, pid in enumerate(todo, 1):
        t0 = time.time()
        data = load_patient(pid)
        if data is None:
            print(f"  [{i}/{len(todo)}] {pid}: SKIP (no h5)", flush=True)
            per_patient[pid] = []
            timing[pid] = 0.0
            continue
        try:
            nodules = predict_nodules_monai(data["vol"], data["voxel_sp"])
        except MemoryError:
            print(f"  [{i}/{len(todo)}] {pid}: OOM on GPU -- falling back to CPU", flush=True)
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            import webapp.predict_monai as pm
            pm._detector = None
            nodules = predict_nodules_monai(data["vol"], data["voxel_sp"])
        except Exception as exc:
            print(f"  [{i}/{len(todo)}] {pid}: ERROR {type(exc).__name__}: {exc}", flush=True)
            per_patient[pid] = []
            timing[pid] = 0.0
            continue

        dt = time.time() - t0
        per_patient[pid] = nodules
        timing[pid] = float(dt)
        n_gt = len(data["gt_nodules"])
        print(
            f"  [{i}/{len(todo)}] {pid}: {len(nodules):3} preds  "
            f"gt={n_gt}  ({dt:.1f}s)",
            flush=True,
        )
        # Incremental save
        payload = {
            "per_patient": per_patient,
            "timing": timing,
            "total_patients": len(pids),
            "completed": len(per_patient),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total_s = sum(v for v in timing.values() if isinstance(v, (int, float)))
    print(f"  [{desc}] Done. Total inference: {total_s:.0f}s ({total_s/60:.1f}min)", flush=True)
    return per_patient


# ---------------------------------------------------------------------------
# GT loading helpers
# ---------------------------------------------------------------------------

def get_gt_vsp_for_panel(pids: list) -> tuple:
    """Return (gt_map, vsp_map) for all pids in one pass."""
    gt_map: dict = {}
    vsp_map: dict = {}
    for pid in pids:
        data = load_patient(pid)
        if data:
            gt_map[pid]  = data["gt_nodules"]
            vsp_map[pid] = data["voxel_sp"]
        else:
            gt_map[pid]  = []
            vsp_map[pid] = (1.0, 1.0, 1.0)
    return gt_map, vsp_map


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def pairwise_metrics(preds_map: dict, gt_map: dict, vsp_map: dict,
                     score_thresh: float, pids: list) -> dict:
    """Compute aggregate + per-patient metrics at a given score threshold."""
    tp = fp = fn = 0
    per_patient = []
    for pid in pids:
        raw_nods = preds_map.get(pid, [])
        gts      = gt_map.get(pid, [])
        vsp      = vsp_map.get(pid, (1.0, 1.0, 1.0))
        nods = [n for n in raw_nods if n.get("confidence", 0) >= score_thresh]
        matched, u_gt, u_pred = match_predictions_to_gt(
            nods, gts, vsp, rule="fixed_15mm", tol_mm=MATCH_TOL_MM
        )
        p_tp = len(matched); p_fp = len(u_pred); p_fn = len(u_gt)
        tp += p_tp; fp += p_fp; fn += p_fn
        prec = p_tp / max(p_tp + p_fp, 1)
        rec  = p_tp / max(p_tp + p_fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_patient.append({
            "pid": pid,
            "tp": p_tp, "fp": p_fp, "fn": p_fn,
            "n_gt": len(gts), "n_pred": len(nods),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
        })

    n_scans = len(pids)
    sens = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1   = 2 * prec * sens / max(prec + sens, 1e-9)
    fp_ps = fp / max(n_scans, 1)
    return {
        "score_thresh": score_thresh,
        "tp": tp, "fp": fp, "fn": fn,
        "sensitivity": float(sens),
        "precision": float(prec),
        "f1": float(f1),
        "fp_per_scan": float(fp_ps),
        "n_scans": n_scans,
        "per_patient": per_patient,
    }


def sweep_for_best_f1(preds_map, gt_map, vsp_map, pids) -> tuple:
    """Sweep SCORE_THRESH_SWEEP, return (best_thresh, best_result)."""
    best_thr, best_f1, best_result = None, -1.0, None
    print("  Sweeping score_thresh on val panel ...", flush=True)
    for thr in SCORE_THRESH_SWEEP:
        res = pairwise_metrics(preds_map, gt_map, vsp_map, thr, pids)
        print(
            f"    thr={thr:.2f}  sens={res['sensitivity']:.3f}  "
            f"prec={res['precision']:.3f}  f1={res['f1']:.3f}  "
            f"fp/scan={res['fp_per_scan']:.2f}",
            flush=True,
        )
        if res["f1"] > best_f1:
            best_f1, best_thr, best_result = res["f1"], thr, res
    print(f"  Best val F1={best_f1:.4f} @ thr={best_thr}", flush=True)
    return best_thr, best_result


def compute_froc(preds_map, gt_map, vsp_map, pids) -> dict:
    """Build FROC curve by sweeping SCORE_THRESH_SWEEP."""
    points = []
    for thr in SCORE_THRESH_SWEEP:
        res = pairwise_metrics(preds_map, gt_map, vsp_map, thr, pids)
        points.append({
            "threshold": thr,
            "fp_per_scan": res["fp_per_scan"],
            "sensitivity": res["sensitivity"],
            "tp": res["tp"], "fp": res["fp"], "fn": res["fn"],
        })
    pts_sorted = sorted(points, key=lambda p: p["fp_per_scan"])
    fp_arr   = np.array([p["fp_per_scan"] for p in pts_sorted])
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
    cpm = float(np.mean([p["sensitivity"] for p in luna_sens]))
    return {"all_points": pts_sorted, "luna16_points": luna_sens, "cpm": cpm}


def stratified_metrics(preds_map, gt_map, vsp_map, pids, score_thresh) -> dict:
    """Sensitivity stratified by nodule diameter."""
    buckets = {k: {"tp": 0, "fn": 0, "n_gt": 0} for k in SIZE_BUCKETS}
    for pid in pids:
        raw_nods = preds_map.get(pid, [])
        gts      = gt_map.get(pid, [])
        vsp      = vsp_map.get(pid, (1.0, 1.0, 1.0))
        nods = [n for n in raw_nods if n.get("confidence", 0) >= score_thresh]
        matched, u_gt, _ = match_predictions_to_gt(
            nods, gts, vsp, rule="fixed_15mm", tol_mm=MATCH_TOL_MM
        )
        matched_gi = {m[0] for m in matched}
        for gi, g in enumerate(gts):
            d = g["diam_mm"]
            for bname, (lo, hi) in SIZE_BUCKETS.items():
                if lo <= d < hi:
                    buckets[bname]["n_gt"] += 1
                    if gi in matched_gi:
                        buckets[bname]["tp"] += 1
                    else:
                        buckets[bname]["fn"] += 1
                    break
    out = {}
    for bname, s in buckets.items():
        n = s["n_gt"]
        out[bname] = {
            "n_gt": n, "tp": s["tp"], "fn": s["fn"],
            "sensitivity": s["tp"] / n if n > 0 else None,
        }
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI + Wilcoxon
# ---------------------------------------------------------------------------

def bootstrap_ci(values_a: np.ndarray, values_b: np.ndarray,
                 stat_fn, n_boot: int = 1000, alpha: float = 0.05,
                 rng_seed: int = 42) -> tuple:
    """Paired bootstrap CI for stat_fn(a) - stat_fn(b).

    Returns (delta_observed, ci_lo, ci_hi).
    Uses only standard numpy -- no pickle, no external serialisation.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(values_a)
    obs_delta = stat_fn(values_a) - stat_fn(values_b)
    boot_deltas = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_deltas[k] = stat_fn(values_a[idx]) - stat_fn(values_b[idx])
    ci_lo = float(np.percentile(boot_deltas, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
    return float(obs_delta), ci_lo, ci_hi


def wilcoxon_p(a: list, b: list) -> float:
    """Paired Wilcoxon signed-rank p-value. Returns nan if scipy unavailable."""
    try:
        from scipy.stats import wilcoxon
        diffs = np.array(a) - np.array(b)
        if np.all(diffs == 0):
            return 1.0
        _, p = wilcoxon(a, b, alternative="two-sided")
        return float(p)
    except ImportError:
        return float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Load "mine" results from existing eval artifacts
# ---------------------------------------------------------------------------

def load_mine_results(test_pids: list) -> dict:
    """Load mine aggregate from fpr_v3_stage2winner_eval_test.json.

    Per-patient breakdown is reconstructed from the .npz FPR candidate file
    (sources array = patient IDs, confs = FPR confidence, labels = TP/FP flag).
    """
    eval_path = ACADEMIC_DIR / "fpr_v3_stage2winner_eval_test.json"
    if not eval_path.exists():
        print("  [mine] WARNING: eval_test.json not found.")
        return {"available": False}

    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    best = eval_data.get("with_fpr_best_f1", {})
    fpr_thr = float(eval_data.get("ckpt_thr", 0.85))

    npz_path = ACADEMIC_DIR / "fpr_v3_stage2winner_test.npz"
    if not npz_path.exists():
        print("  [mine] npz not found -- per-patient F1 unavailable.")
        return {"available": True, "aggregate": best, "per_patient_f1": None, "fpr_thr": fpr_thr}

    # allow_pickle=False is safe: only plain numeric / string arrays
    npz = np.load(str(npz_path), allow_pickle=False)
    # Expected arrays: sources (patient IDs), labels (0/1), confs (FPR prob)
    if "sources" not in npz or "labels" not in npz or "confs" not in npz:
        print(f"  [mine] npz keys unexpected: {list(npz.keys())} -- per-patient F1 unavailable.")
        return {"available": True, "aggregate": best, "per_patient_f1": None, "fpr_thr": fpr_thr}

    arr_sources = npz["sources"]   # shape (N,) dtype str
    arr_labels  = npz["labels"]    # shape (N,) dtype uint8
    arr_confs   = npz["confs"]     # shape (N,) dtype float32

    per_patient_f1: dict = {}
    for pid in test_pids:
        mask = arr_sources == pid
        if mask.sum() == 0:
            per_patient_f1[pid] = None
            continue
        p_confs  = arr_confs[mask]
        p_labels = arr_labels[mask]
        kept = p_confs >= fpr_thr
        tp = int((kept & (p_labels == 1)).sum())
        fp = int((kept & (p_labels == 0)).sum())
        fn = int(((~kept) & (p_labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_patient_f1[pid] = float(f1)

    return {
        "available": True,
        "aggregate": best,
        "per_patient_f1": per_patient_f1,
        "fpr_thr": fpr_thr,
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    mine_agg: dict,
    monai_test_agg: dict,
    monai_froc: dict,
    monai_strat: dict,
    mine_strat: Optional[dict],
    delta_f1: float,
    ci_lo: float,
    ci_hi: float,
    wilcoxon_pval: float,
    monai_best_thr: float,
    val_best_f1: float,
) -> str:
    mine_f1  = mine_agg.get("f1", float("nan"))
    monai_f1 = monai_test_agg["f1"]
    winner_label = "MINE" if mine_f1 >= monai_f1 else "MONAI"

    p_str = f"{wilcoxon_pval:.4f}" if wilcoxon_pval == wilcoxon_pval else "n/a"
    if ci_lo == ci_lo and ci_hi == ci_hi:
        ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
    else:
        ci_str = "[n/a, n/a]"
    d_str = f"{delta_f1:+.4f}" if delta_f1 == delta_f1 else "n/a"

    lines = [
        "# MONAI vs Mine -- Head-to-Head on test_panel (99 patients, locked)",
        "",
        "> Matching rule: fixed 15 mm centroid distance (NOT official LUNA16 radius rule -- see commit db08c2e).",
        f"> MONAI operating point chosen on val_panel: score_thresh={monai_best_thr:.2f} (val F1={val_best_f1:.4f})",
        "> Mine operating point: FPR threshold=0.85 (from fpr_v3_stage2winner_eval_test.json)",
        "",
        "## One-liner verdict",
        "",
        f"**mine F1={mine_f1:.4f} vs MONAI F1={monai_f1:.4f}, winner: {winner_label}, "
        f"delta(mine-MONAI)={d_str} 95%CI {ci_str}, Wilcoxon p={p_str}**",
        "",
        "## Aggregate metrics -- test_panel",
        "",
        "| Metric | mine (Stage2+FPRv3) | MONAI (RetinaNet3D) | Delta |",
        "|--------|--------------------|--------------------|-------|",
    ]

    def row(label, mine_v, monai_v, fmt=".4f"):
        if mine_v is None or monai_v is None or (mine_v != mine_v) or (monai_v != monai_v):
            return f"| {label} | N/A | N/A | -- |"
        delta = mine_v - monai_v
        sign = "+" if delta >= 0 else ""
        return f"| {label} | {mine_v:{fmt}} | {monai_v:{fmt}} | {sign}{delta:{fmt}} |"

    lines += [
        row("F1 (micro)", mine_agg.get("f1"), monai_f1),
        row("Recall / Sensitivity", mine_agg.get("sens"), monai_test_agg["sensitivity"]),
        row("Precision", mine_agg.get("prec"), monai_test_agg["precision"]),
        row("FP / scan", mine_agg.get("fp_per_scan"), monai_test_agg["fp_per_scan"], ".3f"),
    ]
    if isinstance(mine_agg.get("tp"), int) and isinstance(monai_test_agg.get("tp"), int):
        lines += [
            row("TP", mine_agg.get("tp"), monai_test_agg["tp"], "d"),
            row("FP", mine_agg.get("fp"), monai_test_agg["fp"], "d"),
            row("FN", mine_agg.get("fn"), monai_test_agg["fn"], "d"),
        ]
    lines += [
        "",
        f"Bootstrap 95% CI on delta(F1): {ci_str}  (1000 paired resamples)",
        f"Wilcoxon signed-rank (per-patient F1): p={p_str}",
        "",
    ]

    # FROC
    lines += [
        "## FROC -- MONAI on test_panel",
        "",
        "| FP/scan | MONAI sens |",
        "|---------|-----------|",
    ]
    for pt in monai_froc["luna16_points"]:
        lines.append(f"| {pt['fp_per_scan']:.3f} | {pt['sensitivity']:.3f} |")
    lines += [
        "",
        f"**MONAI CPM = {monai_froc['cpm']:.4f}**",
        "",
    ]

    # Stratified
    lines += [
        "## Stratified sensitivity (fixed 15mm matching)",
        "",
        "| Size bucket | n_GT | mine sens | MONAI sens | delta |",
        "|-------------|------|-----------|-----------|-------|",
    ]
    for bname in SIZE_BUCKETS:
        m_s = mine_strat.get(bname) if mine_strat else None
        o_s = monai_strat.get(bname, {})
        m_v = m_s["sensitivity"] if m_s and m_s.get("sensitivity") is not None else None
        o_v = o_s.get("sensitivity")
        n_gt = o_s.get("n_gt", "?")
        m_str = f"{m_v:.3f}" if m_v is not None else "N/A"
        o_str = f"{o_v:.3f}" if o_v is not None else "N/A"
        d_delta = f"{m_v - o_v:+.3f}" if (m_v is not None and o_v is not None) else "--"
        lines.append(f"| {bname} | {n_gt} | {m_str} | {o_str} | {d_delta} |")
    lines += [""]

    # Decision
    if mine_f1 == mine_f1 and monai_f1 == monai_f1:  # not nan
        if mine_f1 > monai_f1 + 0.02:
            retrain_str = "**NO** -- mine outperforms MONAI by >0.02 F1; no retrain justified."
            verdict = (
                f"Mine is clearly better (delta={mine_f1-monai_f1:+.4f}). "
                "MONAI baseline does NOT justify retraining from scratch."
            )
            thesis_bullets = [
                "Frame thesis as: 'Custom 2.5D UNet++ with hard-negative FPR achieves superior performance "
                "vs pretrained 3D RetinaNet on LIDC-IDRI test_panel.'",
                "Competitive advantage: domain-specific FPR stage, lower model complexity.",
            ]
        elif abs(mine_f1 - monai_f1) <= 0.02:
            retrain_str = "**BORDERLINE** -- delta < 0.02 F1; difference may not be significant."
            verdict = (
                f"Results within 0.02 F1 (delta={mine_f1-monai_f1:+.4f}, CI {ci_str}). "
                "Consider MONAI as competitive baseline; CI determines significance."
            )
            thesis_bullets = [
                "Frame thesis as: 'Custom pipeline achieves comparable performance to pretrained MONAI bundle.'",
                "Optionally: fine-tune MONAI on LIDC train split and compare again.",
            ]
        else:
            retrain_str = "**YES** -- MONAI significantly outperforms; retrain or fine-tune recommended."
            verdict = (
                f"MONAI beats mine (delta={mine_f1-monai_f1:+.4f}). "
                "Strong evidence to retrain or fine-tune with MONAI architecture."
            )
            thesis_bullets = [
                "Frame thesis as: 'Pretrained 3D RetinaNet outperforms custom 2.5D pipeline; "
                "fine-tuning on LIDC further improves.'",
                "Next step: fine-tune MONAI bundle on LIDC train_panel, re-evaluate on test_panel.",
            ]
    else:
        retrain_str = "**UNKNOWN** -- one or both F1 values unavailable."
        verdict = "Could not determine winner due to missing data."
        thesis_bullets = []

    lines += [
        "## Decision: Retrain?",
        "",
        f"**Retrain recommendation: {retrain_str}**",
        "",
        verdict,
        "",
        "### Thesis framing",
        "",
    ]
    for b in thesis_bullets:
        lines.append(f"- {b}")
    lines += [
        "",
        "---",
        "",
        f"*Generated by eval_monai_test_panel.py -- {__import__('datetime').date.today()}*",
    ]
    return "\n".join(l for l in lines if l is not None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MONAI vs Mine benchmark on test_panel.")
    ap.add_argument("--skip-val-inference",  action="store_true",
                    help="Skip inference on val panel (use cached JSON if present)")
    ap.add_argument("--skip-test-inference", action="store_true",
                    help="Skip inference on test panel (use cached JSON if present)")
    ap.add_argument("--force-cpu",           action="store_true",
                    help="Force MONAI inference on CPU (overrides CUDA detection)")
    args = ap.parse_args()

    if args.force_cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[INFO] CPU inference forced via --force-cpu")

    t_wall = time.time()

    test_pids = load_panel("test")
    val_pids  = load_panel("val")
    print(f"Panels: test={len(test_pids)}, val={len(val_pids)}", flush=True)

    # ------------------------------------------------------------------
    # Step 1 -- MONAI inference on val + test
    # ------------------------------------------------------------------
    print("\n[Step 1a] MONAI inference on val_panel ...", flush=True)
    if args.skip_val_inference and MONAI_PREDS_VAL.exists():
        val_preds = json.loads(MONAI_PREDS_VAL.read_text(encoding="utf-8"))["per_patient"]
        print(f"  Loaded {len(val_preds)} val patients from cache.", flush=True)
    else:
        val_preds = run_monai_on_panel(val_pids, MONAI_PREDS_VAL, "val")

    print("\n[Step 1b] MONAI inference on test_panel ...", flush=True)
    if args.skip_test_inference and MONAI_PREDS_TEST.exists():
        test_preds = json.loads(MONAI_PREDS_TEST.read_text(encoding="utf-8"))["per_patient"]
        print(f"  Loaded {len(test_preds)} test patients from cache.", flush=True)
    else:
        test_preds = run_monai_on_panel(test_pids, MONAI_PREDS_TEST, "test")

    # ------------------------------------------------------------------
    # Step 2 -- Load GT + voxel spacings
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading GT for both panels (single pass each) ...", flush=True)
    t0 = time.time()
    val_gt,  val_vsp  = get_gt_vsp_for_panel(val_pids)
    test_gt, test_vsp = get_gt_vsp_for_panel(test_pids)
    print(f"  GT loaded in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Step 3 -- Sweep val for best MONAI op-point
    # ------------------------------------------------------------------
    print("\n[Step 3] Sweep score_thresh on val_panel ...", flush=True)
    best_thr, val_best_result = sweep_for_best_f1(val_preds, val_gt, val_vsp, val_pids)
    val_best_f1 = val_best_result["f1"]

    # ------------------------------------------------------------------
    # Step 4 -- MONAI test metrics at best_thr
    # ------------------------------------------------------------------
    print(f"\n[Step 4] MONAI test metrics at score_thresh={best_thr} ...", flush=True)
    monai_test = pairwise_metrics(test_preds, test_gt, test_vsp, best_thr, test_pids)
    print(
        f"  MONAI test  F1={monai_test['f1']:.4f}  "
        f"recall={monai_test['sensitivity']:.4f}  prec={monai_test['precision']:.4f}  "
        f"fp/scan={monai_test['fp_per_scan']:.3f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 5 -- FROC on test
    # ------------------------------------------------------------------
    print("\n[Step 5] FROC on test_panel ...", flush=True)
    monai_froc = compute_froc(test_preds, test_gt, test_vsp, test_pids)
    print(f"  MONAI CPM = {monai_froc['cpm']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Step 6 -- Stratified
    # ------------------------------------------------------------------
    print(f"\n[Step 6] Stratified @ score_thresh={best_thr} ...", flush=True)
    monai_strat = stratified_metrics(test_preds, test_gt, test_vsp, test_pids, best_thr)
    for bname, s in monai_strat.items():
        sens_str = f"{s['sensitivity']:.3f}" if s["sensitivity"] is not None else "N/A"
        print(f"  {bname:10s}  n_gt={s['n_gt']:3}  tp={s['tp']:3}  fn={s['fn']:3}  sens={sens_str}")

    # ------------------------------------------------------------------
    # Step 7 -- Load mine results
    # ------------------------------------------------------------------
    print("\n[Step 7] Loading mine results ...", flush=True)
    mine_data = load_mine_results(test_pids)
    mine_agg_raw = mine_data.get("aggregate", {})
    mine_agg = {
        "f1":         mine_agg_raw.get("f1", float("nan")),
        "sens":       mine_agg_raw.get("sens", float("nan")),
        "prec":       mine_agg_raw.get("prec", float("nan")),
        "fp_per_scan":mine_agg_raw.get("fp_per_scan", float("nan")),
        "tp":         mine_agg_raw.get("tp"),
        "fp":         mine_agg_raw.get("fp"),
        "fn":         mine_agg_raw.get("fn"),
    }
    print(
        f"  mine  F1={mine_agg['f1']:.4f}  "
        f"recall={mine_agg['sens']:.4f}  prec={mine_agg['prec']:.4f}  "
        f"fp/scan={mine_agg['fp_per_scan']:.3f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 8 -- Bootstrap CI + Wilcoxon
    # ------------------------------------------------------------------
    print("\n[Step 8] Bootstrap CI + Wilcoxon ...", flush=True)
    mine_pp_f1 = mine_data.get("per_patient_f1")
    monai_pp   = {r["pid"]: r["f1"] for r in monai_test["per_patient"]}

    delta_f1 = ci_lo = ci_hi = w_pval = float("nan")
    if mine_pp_f1 is not None:
        common_pids = [p for p in test_pids if mine_pp_f1.get(p) is not None and p in monai_pp]
        if common_pids:
            mine_arr  = np.array([mine_pp_f1[p] for p in common_pids], dtype=float)
            monai_arr = np.array([monai_pp[p]    for p in common_pids], dtype=float)
            delta_f1, ci_lo, ci_hi = bootstrap_ci(mine_arr, monai_arr, np.mean, n_boot=1000)
            w_pval = wilcoxon_p(mine_arr.tolist(), monai_arr.tolist())
            print(f"  Paired patients: {len(common_pids)}")
            print(
                f"  delta(mine-MONAI) = {delta_f1:+.4f}  "
                f"95%CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  Wilcoxon p={w_pval:.4f}",
                flush=True,
            )
        else:
            print("  No valid paired patients found.", flush=True)
    else:
        delta_f1 = mine_agg["f1"] - monai_test["f1"]
        print(f"  Per-patient mine F1 unavailable -- aggregate delta: {delta_f1:+.4f}", flush=True)

    # ------------------------------------------------------------------
    # Step 9 -- Write report
    # ------------------------------------------------------------------
    print("\n[Step 9] Writing MONAI_VS_MINE.md ...", flush=True)
    report_text = write_report(
        mine_agg=mine_agg,
        monai_test_agg=monai_test,
        monai_froc=monai_froc,
        monai_strat=monai_strat,
        mine_strat=None,  # mine per-size breakdown not in existing eval JSON
        delta_f1=delta_f1,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        wilcoxon_pval=w_pval,
        monai_best_thr=best_thr,
        val_best_f1=val_best_f1,
    )
    REPORT_MD.write_text(report_text, encoding="utf-8")
    print(f"  Written: {REPORT_MD}", flush=True)

    # ------------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------------
    total_wall = time.time() - t_wall
    print("\n" + "=" * 64)
    print("FINAL SUMMARY")
    print("=" * 64)
    print(
        f"  mine   F1={mine_agg['f1']:.4f}  "
        f"recall={mine_agg['sens']:.4f}  prec={mine_agg['prec']:.4f}  "
        f"fp/scan={mine_agg['fp_per_scan']:.3f}"
    )
    print(
        f"  MONAI  F1={monai_test['f1']:.4f}  "
        f"recall={monai_test['sensitivity']:.4f}  prec={monai_test['precision']:.4f}  "
        f"fp/scan={monai_test['fp_per_scan']:.3f}"
    )
    if delta_f1 == delta_f1:
        print(f"  delta(mine-MONAI) F1 = {delta_f1:+.4f}")
    if ci_lo == ci_lo:
        print(f"  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  Wilcoxon p={w_pval:.4f}")
    print(f"  MONAI CPM = {monai_froc['cpm']:.4f}")
    print(f"  Wall time : {total_wall:.0f}s ({total_wall/60:.1f} min)")
    print(f"\n  Report  : {REPORT_MD}")
    print(f"  Preds   : {MONAI_PREDS_TEST}")
    winner_label = "MINE" if mine_agg.get("f1", 0) >= monai_test["f1"] else "MONAI"
    print(f"\n  ** WINNER: {winner_label} **")
    print("=" * 64)


if __name__ == "__main__":
    main()
