"""Re-eval MONAI cached predictions with luna16_radius matching rule.

Loads pre-cached MONAI val + test panel predictions from JSON.
Sweeps score_thresh on val_panel to find best-F1 op-point under rule=luna16_radius.
Applies op-point to test_panel, computes F1/CPM/FROC/stratified.
Also runs fixed_15mm for 4-cell comparison table.

Output: work/academic/MONAI_LUNA16_RULE.json

Run: python src/reeval_monai_luna16_rule.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))

from configs import WORK
from benchmark import load_patient
from panels import load_panel

ACADEMIC_DIR = WORK / "academic"
LUNA16_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SCORE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                    0.80, 0.85, 0.90, 0.95]


# ---------------------------------------------------------------------------
# GT loading (cached per pid)
# ---------------------------------------------------------------------------
_GT_CACHE: dict = {}


def gt_for(pid: str) -> tuple:
    """Return (gt_nodules, voxel_sp) for a patient, cached."""
    if pid not in _GT_CACHE:
        data = load_patient(pid)
        if data is None:
            _GT_CACHE[pid] = ([], (1.0, 1.0, 1.0))
        else:
            _GT_CACHE[pid] = (data["gt_nodules"], data["voxel_sp"])
    return _GT_CACHE[pid]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_preds(pred_nodules: list, gt_nodules: list, voxel_sp: tuple,
                rule: str) -> tuple:
    """
    Returns (tp, fp, fn, matched_gi_set).

    rule="luna16_radius": tol = max(gt_diam/2, 3.0mm)
    rule="fixed_15mm":    tol = 15.0mm
    """
    if not gt_nodules:
        return 0, len(pred_nodules), 0, set()
    if not pred_nodules:
        return 0, 0, len(gt_nodules), set()

    sp = np.array(voxel_sp)
    used_pred = set()
    matched_gi = set()

    for gi, g in enumerate(gt_nodules):
        gc = np.array(g["centroid_zyx_voxel"], dtype=np.float32) * sp
        tol = max(g["diam_mm"] / 2.0, 3.0) if rule == "luna16_radius" else 15.0
        best_pi, best_d = None, float("inf")
        for pi, p in enumerate(pred_nodules):
            if pi in used_pred:
                continue
            pc = np.array(p["centroid_zyx_voxel"], dtype=np.float32) * sp
            d = float(np.linalg.norm(gc - pc))
            if d < best_d:
                best_d, best_pi = d, pi
        if best_pi is not None and best_d <= tol:
            matched_gi.add(gi)
            used_pred.add(best_pi)

    tp = len(matched_gi)
    fp = len(pred_nodules) - len(used_pred)
    fn = len(gt_nodules) - tp
    return tp, fp, fn, matched_gi


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def eval_panel_at_thresh(pids: list, per_patient_preds: dict,
                         score_thr: float, rule: str) -> dict:
    """Evaluate all patients at a given score threshold."""
    tp = fp = fn = n_gt = n_patients = 0
    for pid in pids:
        all_preds = per_patient_preds.get(pid, [])
        gt, vsp = gt_for(pid)
        if not gt and not all_preds:
            continue
        n_patients += 1
        active = [p for p in all_preds if p["confidence"] >= score_thr]
        _tp, _fp, _fn, _ = match_preds(active, gt, vsp, rule)
        tp += _tp; fp += _fp; fn += _fn
        n_gt += len(gt)

    sens = tp / max(n_gt, 1)
    fp_per_scan = fp / max(n_patients, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2.0 * prec * sens / max(prec + sens, 1e-9)
    return {
        "score_thr": score_thr, "tp": tp, "fp": fp, "fn": fn,
        "n_gt": n_gt, "n_patients": n_patients,
        "sensitivity": float(sens), "fp_per_scan": float(fp_per_scan),
        "precision": float(prec), "f1": float(f1),
    }


def compute_froc_and_cpm(pids: list, per_patient_preds: dict, rule: str) -> dict:
    """Sweep thresholds, build FROC, compute CPM."""
    points = []
    print(f"  Sweeping {len(SCORE_THRESHOLDS)} thresholds (rule={rule})...", flush=True)
    for thr in SCORE_THRESHOLDS:
        r = eval_panel_at_thresh(pids, per_patient_preds, thr, rule)
        print(f"    thr={thr:.2f}  F1={r['f1']:.4f}  sens={r['sensitivity']:.3f}"
              f"  FP/scan={r['fp_per_scan']:.2f}  TP={r['tp']} FP={r['fp']} FN={r['fn']}",
              flush=True)
        points.append(r)

    pts_sorted = sorted(points, key=lambda p: p["fp_per_scan"])
    fp_arr = np.array([p["fp_per_scan"] for p in pts_sorted])
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

    cpm = float(np.mean([x["sensitivity"] for x in luna_sens]))
    best = max(points, key=lambda x: x["f1"])
    return {"cpm": cpm, "luna16_points": luna_sens, "best": best, "all_points": points}


def stratified_at_thresh(pids: list, per_patient_preds: dict,
                         score_thr: float, rule: str) -> dict:
    buckets = {"small_4_6mm": (4, 6), "medium_6_15mm": (6, 15), "large_15mm_plus": (15, 999)}
    bs = {k: {"tp": 0, "fn": 0, "n_gt": 0} for k in buckets}
    for pid in pids:
        all_preds = per_patient_preds.get(pid, [])
        gt, vsp = gt_for(pid)
        if not gt:
            continue
        active = [p for p in all_preds if p["confidence"] >= score_thr]
        _, _, _, matched_gi = match_preds(active, gt, vsp, rule)
        for gi, g in enumerate(gt):
            d = g["diam_mm"]
            for name, (lo, hi) in buckets.items():
                if lo <= d < hi:
                    bs[name]["n_gt"] += 1
                    if gi in matched_gi:
                        bs[name]["tp"] += 1
                    else:
                        bs[name]["fn"] += 1
                    break
    return {k: {**v, "sensitivity": v["tp"] / v["n_gt"] if v["n_gt"] > 0 else None}
            for k, v in bs.items()}


# ---------------------------------------------------------------------------
# Bootstrap CI + Wilcoxon
# ---------------------------------------------------------------------------

def per_patient_f1(pids: list, per_patient_preds: dict,
                   score_thr: float, rule: str) -> np.ndarray:
    """Return per-patient F1 array (0.0 if no GT and no pred)."""
    out = []
    for pid in pids:
        all_preds = per_patient_preds.get(pid, [])
        gt, vsp = gt_for(pid)
        if not gt and not all_preds:
            continue
        active = [p for p in all_preds if p["confidence"] >= score_thr]
        tp, fp, fn, _ = match_preds(active, gt, vsp, rule)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(len(gt), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out.append(f1)
    return np.array(out)


def bootstrap_delta_ci(a: np.ndarray, b: np.ndarray,
                        n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap CI for mean(a) - mean(b). Paired (same patients)."""
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(a[idx].mean() - b[idx].mean())
    d = np.array(deltas)
    return {
        "observed_delta": float(a.mean() - b.mean()),
        "ci_95_lo": float(np.percentile(d, 2.5)),
        "ci_95_hi": float(np.percentile(d, 97.5)),
        "n_samples": n,
        "n_boot": n_boot,
    }


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value. Returns nan if scipy not available."""
    try:
        from scipy.stats import wilcoxon
        n = min(len(a), len(b))
        diff = a[:n] - b[:n]
        if np.all(diff == 0):
            return 1.0
        _, p = wilcoxon(diff, zero_method="wilcox", correction=False)
        return float(p)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=== MONAI re-eval with luna16_radius + fixed_15mm ===", flush=True)

    # Load cached MONAI preds
    test_path = ACADEMIC_DIR / "monai_test_panel_preds.json"
    val_path = ACADEMIC_DIR / "monai_val_panel_preds.json"
    monai_test = json.loads(test_path.read_text())["per_patient"]
    monai_val = json.loads(val_path.read_text())["per_patient"]

    val_pids = load_panel("val")
    test_pids = load_panel("test")
    print(f"Val panel: {len(val_pids)} pids | Test panel: {len(test_pids)} pids", flush=True)

    # Pre-warm GT cache
    print("Pre-loading GT (val + test)...", flush=True)
    for pid in set(val_pids + test_pids):
        gt_for(pid)
    print(f"GT cache warmed in {time.time()-t0:.1f}s", flush=True)

    # ---- LUNA16_RADIUS: sweep on val, apply best op-point to test ----
    print("\n[1/4] Val sweep — luna16_radius", flush=True)
    val_froc_luna = compute_froc_and_cpm(val_pids, monai_val, "luna16_radius")
    best_thr_luna = val_froc_luna["best"]["score_thr"]
    print(f"  Best val op-point: thr={best_thr_luna}  F1={val_froc_luna['best']['f1']:.4f}", flush=True)

    print(f"\n[2/4] Test eval — luna16_radius @ thr={best_thr_luna}", flush=True)
    test_luna = eval_panel_at_thresh(test_pids, monai_test, best_thr_luna, "luna16_radius")
    test_froc_luna = compute_froc_and_cpm(test_pids, monai_test, "luna16_radius")
    strat_luna = stratified_at_thresh(test_pids, monai_test, best_thr_luna, "luna16_radius")

    print(f"  Test luna16_radius: F1={test_luna['f1']:.4f}  CPM={test_froc_luna['cpm']:.4f}"
          f"  sens={test_luna['sensitivity']:.3f}  prec={test_luna['precision']:.3f}"
          f"  FP/scan={test_luna['fp_per_scan']:.2f}", flush=True)

    # ---- FIXED_15MM: sweep on val, apply best op-point to test ----
    print("\n[3/4] Val sweep — fixed_15mm", flush=True)
    val_froc_fixed = compute_froc_and_cpm(val_pids, monai_val, "fixed_15mm")
    best_thr_fixed = val_froc_fixed["best"]["score_thr"]
    print(f"  Best val op-point: thr={best_thr_fixed}  F1={val_froc_fixed['best']['f1']:.4f}", flush=True)

    print(f"\n[4/4] Test eval — fixed_15mm @ thr={best_thr_fixed}", flush=True)
    test_fixed = eval_panel_at_thresh(test_pids, monai_test, best_thr_fixed, "fixed_15mm")
    test_froc_fixed = compute_froc_and_cpm(test_pids, monai_test, "fixed_15mm")
    strat_fixed = stratified_at_thresh(test_pids, monai_test, best_thr_fixed, "fixed_15mm")

    print(f"  Test fixed_15mm:   F1={test_fixed['f1']:.4f}  CPM={test_froc_fixed['cpm']:.4f}"
          f"  sens={test_fixed['sensitivity']:.3f}  prec={test_fixed['precision']:.3f}"
          f"  FP/scan={test_fixed['fp_per_scan']:.2f}", flush=True)

    # ---- Bootstrap CI + Wilcoxon (luna16_radius: mine vs MONAI) ----
    print("\n[Bootstrap] Computing per-patient F1 arrays...", flush=True)
    # Mine luna16_radius reference numbers from MINE_LUNA16_RULE.json
    mine_ref = json.loads((ACADEMIC_DIR / "MINE_LUNA16_RULE.json").read_text())
    mine_thr_luna = mine_ref["match_rules_compared"]["luna16_radius"]["best_threshold"]
    mine_f1_luna = mine_ref["match_rules_compared"]["luna16_radius"]["best_f1"]
    mine_cpm_luna = mine_ref["match_rules_compared"]["luna16_radius"]["cpm"]
    mine_thr_fixed = mine_ref["match_rules_compared"]["fixed_15mm"]["best_threshold"]
    mine_f1_fixed = mine_ref["match_rules_compared"]["fixed_15mm"]["best_f1"]
    mine_cpm_fixed = mine_ref["match_rules_compared"]["fixed_15mm"]["cpm"]

    # Need mine test preds to compute per-patient delta — use MONAI preds against itself
    # for bootstrap; for mine we use aggregate stats since probe volumes are not re-run here.
    # Bootstrap is computed on MONAI per-patient F1 only (self-consistency check).
    monai_pp_luna = per_patient_f1(test_pids, monai_test, best_thr_luna, "luna16_radius")
    monai_pp_fixed = per_patient_f1(test_pids, monai_test, best_thr_fixed, "fixed_15mm")

    print(f"  MONAI per-patient F1 (luna): mean={monai_pp_luna.mean():.4f}  n={len(monai_pp_luna)}")
    print(f"  MONAI per-patient F1 (fixed): mean={monai_pp_fixed.mean():.4f}  n={len(monai_pp_fixed)}")

    # Use mine's MINE_LUNA16_RULE probs for per-patient delta if available
    # Since mine has no per-patient array here, we compare aggregate numbers and
    # note the limitation. Bootstrap only on MONAI internal for variance estimate.
    mono_boot_luna = bootstrap_delta_ci(monai_pp_luna, monai_pp_luna * 0 + mine_f1_luna)
    wilcox_p = wilcoxon_p(monai_pp_luna, monai_pp_luna * 0 + mine_f1_luna)

    delta_f1_luna = mine_f1_luna - test_luna["f1"]
    delta_cpm_luna = mine_cpm_luna - test_froc_luna["cpm"]
    delta_f1_fixed = mine_f1_fixed - test_fixed["f1"]
    delta_cpm_fixed = mine_cpm_fixed - test_froc_fixed["cpm"]

    # ---- Summary ----
    print("\n=== 4-CELL COMPARISON TABLE ===")
    print(f"{'':20} {'fixed_15mm':>22} {'luna16_radius':>22}")
    print("-" * 66)
    print(f"{'mine':20} F1={mine_f1_fixed:.4f} CPM={mine_cpm_fixed:.4f}"
          f"  F1={mine_f1_luna:.4f} CPM={mine_cpm_luna:.4f}")
    print(f"{'MONAI':20} F1={test_fixed['f1']:.4f} CPM={test_froc_fixed['cpm']:.4f}"
          f"  F1={test_luna['f1']:.4f} CPM={test_froc_luna['cpm']:.4f}")
    print(f"{'delta (mine-MONAI)':20} dF1={delta_f1_fixed:+.4f} dCPM={delta_cpm_fixed:+.4f}"
          f"  dF1={delta_f1_luna:+.4f} dCPM={delta_cpm_luna:+.4f}")

    decision_luna_f1 = "mine" if delta_f1_luna > 0 else "MONAI"
    decision_luna_cpm = "mine" if delta_cpm_luna > 0 else "MONAI"
    decision_fixed_f1 = "mine" if delta_f1_fixed > 0 else "MONAI"
    decision_fixed_cpm = "mine" if delta_cpm_fixed > 0 else "MONAI"
    both_rules_mine_wins = (delta_f1_luna > 0 and delta_f1_fixed > 0 and
                             delta_cpm_luna > 0 and delta_cpm_fixed > 0)
    print(f"\nDecision F1   luna16_radius: {decision_luna_f1} wins  (+{abs(delta_f1_luna):.4f})")
    print(f"Decision CPM  luna16_radius: {decision_luna_cpm} wins  (+{abs(delta_cpm_luna):.4f})")
    print(f"Mine wins both rules: {both_rules_mine_wins}")

    # ---- Output JSON ----
    out = {
        "description": (
            "MONAI re-eval on cached predictions with luna16_radius + fixed_15mm rules. "
            "Op-point tuned on val_panel, applied to test_panel."
        ),
        "monai_test": {
            "luna16_radius": {
                "val_best_thr": best_thr_luna,
                "val_best_f1": val_froc_luna["best"]["f1"],
                "test_f1": test_luna["f1"],
                "test_sensitivity": test_luna["sensitivity"],
                "test_precision": test_luna["precision"],
                "test_fp_per_scan": test_luna["fp_per_scan"],
                "test_tp": test_luna["tp"],
                "test_fp": test_luna["fp"],
                "test_fn": test_luna["fn"],
                "test_n_gt": test_luna["n_gt"],
                "test_cpm": test_froc_luna["cpm"],
                "test_luna16_points": test_froc_luna["luna16_points"],
                "test_stratified": strat_luna,
            },
            "fixed_15mm": {
                "val_best_thr": best_thr_fixed,
                "val_best_f1": val_froc_fixed["best"]["f1"],
                "test_f1": test_fixed["f1"],
                "test_sensitivity": test_fixed["sensitivity"],
                "test_precision": test_fixed["precision"],
                "test_fp_per_scan": test_fixed["fp_per_scan"],
                "test_tp": test_fixed["tp"],
                "test_fp": test_fixed["fp"],
                "test_fn": test_fixed["fn"],
                "test_n_gt": test_fixed["n_gt"],
                "test_cpm": test_froc_fixed["cpm"],
                "test_luna16_points": test_froc_fixed["luna16_points"],
                "test_stratified": strat_fixed,
            },
        },
        "mine_reference": {
            "luna16_radius": {
                "best_thr": mine_thr_luna, "f1": mine_f1_luna, "cpm": mine_cpm_luna,
            },
            "fixed_15mm": {
                "best_thr": mine_thr_fixed, "f1": mine_f1_fixed, "cpm": mine_cpm_fixed,
            },
        },
        "deltas_mine_minus_monai": {
            "luna16_radius": {
                "delta_f1": delta_f1_luna,
                "delta_cpm": delta_cpm_luna,
                "winner_f1": decision_luna_f1,
                "winner_cpm": decision_luna_cpm,
            },
            "fixed_15mm": {
                "delta_f1": delta_f1_fixed,
                "delta_cpm": delta_cpm_fixed,
                "winner_f1": decision_fixed_f1,
                "winner_cpm": decision_fixed_cpm,
            },
        },
        "bootstrap_luna16_radius": {
            **mono_boot_luna,
            "note": (
                "Bootstrap paired against per-patient MONAI F1 vs. scalar mine mean. "
                "Mine per-patient array not available (prob volumes needed). "
                "Treat CI as MONAI variance estimate only."
            ),
            "wilcoxon_p_approx": wilcox_p,
        },
        "verdict": {
            "mine_wins_both_rules_f1_cpm": both_rules_mine_wins,
            "fine_tune_luna16_recommended": not both_rules_mine_wins,
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out_path = ACADEMIC_DIR / "MONAI_LUNA16_RULE.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}", flush=True)
    print(f"Wall clock: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
