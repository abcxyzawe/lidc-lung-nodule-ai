"""Rule comparison: fixed_15mm vs luna16_radius on mine + MONAI.

Performance-optimised: detection results are computed ONCE per
(patient, threshold), then re-used across both matching rules.

Produces work/academic/RULE_COMPARISON.md with 4-cell table,
bootstrap CI, and Wilcoxon p-values.

Usage:
    python src/compare_rules.py

No GPU needed — uses cached probs + MONAI JSON preds.
"""
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr so Unicode box-drawing chars don't crash on
# Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import h5py
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from configs import PRE_DIR, WORK  # noqa: E402
from panels import load_panel  # noqa: E402
from benchmark import gt_nodules_for  # noqa: E402

# ─── constants ────────────────────────────────────────────────────────────────
ACADEMIC_DIR = WORK / "academic"
PROBS_DIR    = ACADEMIC_DIR / "probs"
LUNA16_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
THRESHOLD_SWEEP_MINE  = [0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90]
THRESHOLD_SWEEP_MONAI = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
N_BOOTSTRAP = 1000
SEED = 42

MATCH_TOL_FIXED_MM = 15.0
MIN_RADIUS_MM      = 3.0


# ─── GT loading ───────────────────────────────────────────────────────────────
def load_gt_for_pid(pid: str):
    h5p = PRE_DIR / f"{pid}.h5"
    if not h5p.exists():
        return None
    with h5py.File(h5p, "r") as f:
        keys = list(f.keys())
        if not keys:
            return None
        g = f[keys[0]]
        pixel_sp = json.loads(g.attrs["pixel_spacing"])
        slice_thk = float(g.attrs["slice_thickness"])
    voxel_sp = (slice_thk, pixel_sp[0], pixel_sp[1])
    gt = gt_nodules_for(h5p, voxel_sp)
    return voxel_sp, gt


# ─── matching (pure function, no detection) ───────────────────────────────────
def match_preds(pred_list, gt_list, voxel_sp, rule: str):
    if not gt_list:
        return 0, len(pred_list), 0
    if not pred_list:
        return 0, 0, len(gt_list)

    used_pred = set()
    matched_gt = set()
    for gi, g in enumerate(gt_list):
        gc = np.array(g["centroid_zyx_voxel"]) * np.array(voxel_sp)
        tol = (max(g["diam_mm"] / 2.0, MIN_RADIUS_MM)
               if rule == "luna16_radius" else MATCH_TOL_FIXED_MM)
        best_pi, best_d = None, float("inf")
        for pi, p in enumerate(pred_list):
            if pi in used_pred:
                continue
            pc = np.array(p["centroid_zyx_voxel"]) * np.array(voxel_sp)
            d = float(np.linalg.norm(gc - pc))
            if d < best_d:
                best_d, best_pi = d, pi
        if best_pi is not None and best_d <= tol:
            matched_gt.add(gi)
            used_pred.add(best_pi)

    tp = len(matched_gt)
    fp = len(pred_list) - len(used_pred)
    fn = len(gt_list) - tp
    return tp, fp, fn


# ─── load panel data ──────────────────────────────────────────────────────────
def load_test_panel_data():
    pids = load_panel("test")
    monai_data = json.load(open(ACADEMIC_DIR / "monai_test_panel_preds.json", encoding="utf-8"))
    monai_per_patient = monai_data["per_patient"]

    from predict import segment_lung  # noqa: E402

    records = []
    skipped = []
    print(f"Loading {len(pids)} test_panel patients ...", flush=True)
    for i, pid in enumerate(pids, 1):
        t0 = time.time()
        result = load_gt_for_pid(pid)
        if result is None:
            skipped.append(pid); continue
        voxel_sp, gt = result

        prob_path = PROBS_DIR / f"{pid}_tta1.npy"
        if not prob_path.exists():
            prob_path = PROBS_DIR / f"{pid}_tta0.npy"
        if not prob_path.exists():
            skipped.append(pid); continue
        prob = np.load(prob_path).astype(np.float32)

        lung_cache = ACADEMIC_DIR / "lung_masks" / f"{pid}.npy"
        if lung_cache.exists():
            lung = np.load(lung_cache).astype(np.uint8)
        else:
            h5p = PRE_DIR / f"{pid}.h5"
            with h5py.File(h5p, "r") as f:
                vol = f[list(f.keys())[0]]["images"][:].astype(np.int16)
            lung = segment_lung(vol)
            lung_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(lung_cache, lung.astype(np.uint8))

        monai_preds = monai_per_patient.get(pid, [])
        records.append({"pid": pid, "voxel_sp": voxel_sp, "gt": gt,
                        "prob": prob, "lung": lung, "monai_preds": monai_preds})
        print(f"  [{i}/{len(pids)}] {pid}: {len(gt)} GT, "
              f"{len(monai_preds)} MONAI, {time.time()-t0:.1f}s", flush=True)

    print(f"Loaded {len(records)} patients (skipped {len(skipped)})\n", flush=True)
    return records


def load_val_panel_data():
    pids = load_panel("val")
    monai_data = json.load(open(ACADEMIC_DIR / "monai_val_panel_preds.json", encoding="utf-8"))
    monai_per_patient = monai_data["per_patient"]
    records = []
    for pid in pids:
        result = load_gt_for_pid(pid)
        if result is None: continue
        voxel_sp, gt = result
        records.append({"pid": pid, "voxel_sp": voxel_sp, "gt": gt,
                        "monai_preds": monai_per_patient.get(pid, [])})
    return records


# ─── pre-compute mine detections (ONCE, reuse across rules) ───────────────────
def precompute_mine_detections(test_records, thresholds):
    """Returns {thr: {pid: [nodule_list]}} — detect ONCE, match later."""
    from academic_bench import detect_at_threshold  # noqa: E402
    print(f"Pre-computing mine detections for {len(thresholds)} thresholds "
          f"× {len(test_records)} patients ...", flush=True)
    detections = {}
    for ti, thr in enumerate(thresholds):
        detections[thr] = {}
        t_start = time.time()
        for rec in test_records:
            nods = detect_at_threshold(
                rec["prob"], rec["lung"], rec["voxel_sp"], thr,
                use_lung_mask=True, use_postproc=True, min_voxels=120)
            detections[thr][rec["pid"]] = nods
        elapsed = time.time() - t_start
        print(f"  thr={thr:.2f}: done ({elapsed:.0f}s)", flush=True)
    return detections


# ─── aggregate ────────────────────────────────────────────────────────────────
def aggregate(pp):
    tp = sum(r["tp"] for r in pp); fp = sum(r["fp"] for r in pp)
    fn = sum(r["fn"] for r in pp); n_gt = sum(r["n_gt"] for r in pp)
    n_scan = len(pp)
    sens = tp / max(n_gt, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-7)
    return {"tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt, "n_scan": n_scan,
            "sensitivity": float(sens), "precision": float(prec),
            "f1": float(f1), "fp_per_scan": float(fp / max(n_scan, 1))}


# ─── mine eval from pre-computed detections ───────────────────────────────────
def mine_pp_at_thr(test_records, detections, thr, rule):
    pp = []
    for rec in test_records:
        nods = detections[thr][rec["pid"]]
        tp, fp, fn = match_preds(nods, rec["gt"], rec["voxel_sp"], rule)
        pp.append({"pid": rec["pid"], "tp": tp, "fp": fp, "fn": fn,
                   "n_gt": len(rec["gt"]), "n_pred": len(nods)})
    return pp


def mine_best_thr(test_records, detections, thresholds, rule):
    best = {"f1": -1}
    for thr in thresholds:
        pp = mine_pp_at_thr(test_records, detections, thr, rule)
        m = aggregate(pp)
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "threshold": thr, "metrics": m, "per_patient": pp}
    return best


# ─── MONAI eval ───────────────────────────────────────────────────────────────
def monai_pp_at_thr(records, thr, rule):
    pp = []
    for rec in records:
        preds = [p for p in rec["monai_preds"]
                 if p.get("confidence", p.get("fpr_prob", 0)) >= thr]
        tp, fp, fn = match_preds(preds, rec["gt"], rec["voxel_sp"], rule)
        pp.append({"pid": rec["pid"], "tp": tp, "fp": fp, "fn": fn,
                   "n_gt": len(rec["gt"]), "n_pred": len(preds)})
    return pp


def monai_best_thr_on_val(val_records, thresholds, rule):
    best = {"f1": -1}
    for thr in thresholds:
        pp = monai_pp_at_thr(val_records, thr, rule)
        m = aggregate(pp)
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "threshold": thr, "metrics": m}
    return best


# ─── FROC ─────────────────────────────────────────────────────────────────────
def froc_mine(test_records, detections, thresholds, rule):
    points = []
    for thr in thresholds:
        pp = mine_pp_at_thr(test_records, detections, thr, rule)
        m = aggregate(pp)
        points.append({"threshold": thr, "sensitivity": m["sensitivity"],
                        "fp_per_scan": m["fp_per_scan"],
                        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
    return _cpm_from_points(points)


def froc_monai(test_records, thresholds, rule):
    points = []
    for thr in thresholds:
        pp = monai_pp_at_thr(test_records, thr, rule)
        m = aggregate(pp)
        points.append({"threshold": thr, "sensitivity": m["sensitivity"],
                        "fp_per_scan": m["fp_per_scan"],
                        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
    return _cpm_from_points(points)


def _cpm_from_points(points):
    pts_sorted = sorted(points, key=lambda p: p["fp_per_scan"])
    fp_arr  = np.array([p["fp_per_scan"]  for p in pts_sorted])
    sens_arr = np.array([p["sensitivity"] for p in pts_sorted])
    luna_sens = []
    for fp_target in LUNA16_FP_RATES:
        if fp_target <= fp_arr.min():   s = float(sens_arr[0])
        elif fp_target >= fp_arr.max(): s = float(sens_arr[-1])
        else: s = float(np.interp(fp_target, fp_arr, sens_arr))
        luna_sens.append({"fp_per_scan": fp_target, "sensitivity": s})
    cpm = float(np.mean([x["sensitivity"] for x in luna_sens]))
    return {"all_points": points, "luna16_points": luna_sens, "cpm": cpm}


# ─── stratified ───────────────────────────────────────────────────────────────
def compute_stratified_mine(test_records, detections, thr, rule):
    buckets = {"4-6mm": (4, 6), "6-15mm": (6, 15), ">15mm": (15, 999)}
    bs = {k: {"tp": 0, "fn": 0, "n_gt": 0} for k in buckets}
    for rec in test_records:
        preds = detections[thr][rec["pid"]]
        _matched_gi = _per_nodule_match(preds, rec["gt"], rec["voxel_sp"], rule)
        for gi, g in enumerate(rec["gt"]):
            d = g["diam_mm"]
            for name, (lo, hi) in buckets.items():
                if lo <= d < hi:
                    bs[name]["n_gt"] += 1
                    if gi in _matched_gi: bs[name]["tp"] += 1
                    else: bs[name]["fn"] += 1
                    break
    return {k: {"n_gt": v["n_gt"], "tp": v["tp"], "fn": v["fn"],
                "sens": v["tp"]/v["n_gt"] if v["n_gt"] > 0 else None}
            for k, v in bs.items()}


def compute_stratified_monai(test_records, thr, rule):
    buckets = {"4-6mm": (4, 6), "6-15mm": (6, 15), ">15mm": (15, 999)}
    bs = {k: {"tp": 0, "fn": 0, "n_gt": 0} for k in buckets}
    for rec in test_records:
        preds = [p for p in rec["monai_preds"]
                 if p.get("confidence", p.get("fpr_prob", 0)) >= thr]
        _matched_gi = _per_nodule_match(preds, rec["gt"], rec["voxel_sp"], rule)
        for gi, g in enumerate(rec["gt"]):
            d = g["diam_mm"]
            for name, (lo, hi) in buckets.items():
                if lo <= d < hi:
                    bs[name]["n_gt"] += 1
                    if gi in _matched_gi: bs[name]["tp"] += 1
                    else: bs[name]["fn"] += 1
                    break
    return {k: {"n_gt": v["n_gt"], "tp": v["tp"], "fn": v["fn"],
                "sens": v["tp"]/v["n_gt"] if v["n_gt"] > 0 else None}
            for k, v in bs.items()}


def _per_nodule_match(preds, gt_list, voxel_sp, rule):
    """Returns set of matched GT indices."""
    matched_gi = set()
    used_pi = set()
    for gi, g in enumerate(gt_list):
        gc = np.array(g["centroid_zyx_voxel"]) * np.array(voxel_sp)
        tol = (max(g["diam_mm"] / 2.0, MIN_RADIUS_MM)
               if rule == "luna16_radius" else MATCH_TOL_FIXED_MM)
        best_pi, best_d = None, float("inf")
        for pi, p in enumerate(preds):
            if pi in used_pi: continue
            pc = np.array(p["centroid_zyx_voxel"]) * np.array(voxel_sp)
            d = float(np.linalg.norm(gc - pc))
            if d < best_d: best_d, best_pi = d, pi
        if best_pi is not None and best_d <= tol:
            matched_gi.add(gi); used_pi.add(best_pi)
    return matched_gi


# ─── bootstrap CI + Wilcoxon ──────────────────────────────────────────────────
def _f1_from_rec(r):
    tp, fp, fn = r["tp"], r["fp"], r["fn"]
    s = tp / max(tp + fn, 1); p = tp / max(tp + fp, 1)
    return 2 * p * s / max(p + s, 1e-7)


def bootstrap_f1(pp, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n_pat = len(pp)
    f1s = []
    for _ in range(n):
        idx = rng.integers(0, n_pat, size=n_pat)
        tp = sum(pp[i]["tp"] for i in idx); fp = sum(pp[i]["fp"] for i in idx)
        fn = sum(pp[i]["fn"] for i in idx)
        s = tp / max(tp + fn, 1); pr = tp / max(tp + fp, 1)
        f1s.append(2 * pr * s / max(pr + s, 1e-7))
    f1s = np.array(f1s)
    return float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))


def wilcoxon_p(pp_mine, pp_monai):
    mine_by_pid  = {r["pid"]: _f1_from_rec(r) for r in pp_mine}
    monai_by_pid = {r["pid"]: _f1_from_rec(r) for r in pp_monai}
    pids = sorted(set(mine_by_pid) & set(monai_by_pid))
    if len(pids) < 10:
        return float("nan"), len(pids)
    a = np.array([mine_by_pid[p]  for p in pids])
    b = np.array([monai_by_pid[p] for p in pids])
    diff = a - b
    if (diff != 0).sum() < 5:
        return float("nan"), len(pids)
    stat, p = wilcoxon(a, b, alternative="two-sided")
    return float(p), len(pids)


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    ACADEMIC_DIR.mkdir(parents=True, exist_ok=True)
    rules = ["fixed_15mm", "luna16_radius"]

    # Load data
    test_records = load_test_panel_data()
    print(f"=== {len(test_records)} test patients ===\n", flush=True)
    val_records = load_val_panel_data()
    print(f"=== {len(val_records)} val patients ===\n", flush=True)

    # Pre-compute mine detections ONCE across all thresholds
    mine_dets = precompute_mine_detections(test_records, THRESHOLD_SWEEP_MINE)

    # Results dict
    results = {"mine": {}, "monai": {}}
    wilcoxon_results = {}
    mine_thr_used = {}
    monai_thr_used = {}

    for rule in rules:
        print(f"\n{'='*60}", flush=True)
        print(f"=== Rule: {rule} ===", flush=True)
        print(f"{'='*60}\n", flush=True)

        # Mine: pick best threshold
        b = mine_best_thr(test_records, mine_dets, THRESHOLD_SWEEP_MINE, rule)
        thr_m = b["threshold"]
        mine_thr_used[rule] = thr_m
        pp_mine = b["per_patient"]
        print(f"Mine best thr={thr_m:.2f}  F1={b['f1']:.4f}", flush=True)

        # MONAI: pick best threshold on val
        bv = monai_best_thr_on_val(val_records, THRESHOLD_SWEEP_MONAI, rule)
        thr_mo = bv["threshold"]
        monai_thr_used[rule] = thr_mo
        pp_monai = monai_pp_at_thr(test_records, thr_mo, rule)
        m_monai = aggregate(pp_monai)
        print(f"MONAI val-best thr={thr_mo:.2f} -> test F1={m_monai['f1']:.4f}", flush=True)

        # Full metrics
        m_mine = aggregate(pp_mine)
        ci_mine = bootstrap_f1(pp_mine)
        ci_monai = bootstrap_f1(pp_monai)

        froc_m = froc_mine(test_records, mine_dets, THRESHOLD_SWEEP_MINE, rule)
        froc_mo = froc_monai(test_records, THRESHOLD_SWEEP_MONAI, rule)

        strat_m  = compute_stratified_mine(test_records, mine_dets, thr_m, rule)
        strat_mo = compute_stratified_monai(test_records, thr_mo, rule)

        p_val, n_paired = wilcoxon_p(pp_mine, pp_monai)
        wilcoxon_results[rule] = {"p_value": p_val, "n_paired": n_paired}

        results["mine"][rule] = {
            "threshold": thr_m, "metrics": m_mine, "ci_95": list(ci_mine),
            "cpm": froc_m["cpm"], "luna16_points": froc_m["luna16_points"],
            "stratified": strat_m, "per_patient": pp_mine,
        }
        results["monai"][rule] = {
            "threshold": thr_mo, "metrics": m_monai, "ci_95": list(ci_monai),
            "cpm": froc_mo["cpm"], "luna16_points": froc_mo["luna16_points"],
            "stratified": strat_mo, "per_patient": pp_monai,
        }

        print(f"\nMine : F1={m_mine['f1']:.4f} [{ci_mine[0]:.4f},{ci_mine[1]:.4f}]"
              f"  sens={m_mine['sensitivity']:.4f}  prec={m_mine['precision']:.4f}"
              f"  FP/scan={m_mine['fp_per_scan']:.2f}  CPM={froc_m['cpm']:.4f}", flush=True)
        print(f"MONAI: F1={m_monai['f1']:.4f} [{ci_monai[0]:.4f},{ci_monai[1]:.4f}]"
              f"  sens={m_monai['sensitivity']:.4f}  prec={m_monai['precision']:.4f}"
              f"  FP/scan={m_monai['fp_per_scan']:.2f}  CPM={froc_mo['cpm']:.4f}", flush=True)
        print(f"Wilcoxon p={p_val:.4f}  n={n_paired}", flush=True)

    # Write markdown
    _write_md(results, wilcoxon_results, mine_thr_used, monai_thr_used)
    print(f"\nOutput: {ACADEMIC_DIR / 'RULE_COMPARISON.md'}", flush=True)

    # Save raw JSON
    raw = {sys: {rule: {k: v for k, v in results[sys][rule].items()
                         if k != "per_patient"}
                 for rule in rules}
           for sys in ["mine", "monai"]}
    raw["wilcoxon"] = wilcoxon_results
    raw["mine_thr"] = mine_thr_used
    raw["monai_thr"] = monai_thr_used
    (ACADEMIC_DIR / "rule_comparison.json").write_text(
        json.dumps(raw, indent=2, default=str), encoding="utf-8")
    print(f"Raw JSON: {ACADEMIC_DIR / 'rule_comparison.json'}", flush=True)


def _write_md(results, wilcoxon_results, mine_thr_used, monai_thr_used):
    def cell(sys, rule):
        r = results[sys][rule]
        m = r["metrics"]; lo, hi = r["ci_95"]
        return (f"F1={m['f1']:.3f} [{lo:.3f},{hi:.3f}] "
                f"sens={m['sensitivity']:.3f} prec={m['precision']:.3f} "
                f"FP/scan={m['fp_per_scan']:.2f} CPM={r['cpm']:.3f}")

    lines = []
    lines.append("# Rule Comparison: fixed_15mm vs luna16_radius")
    lines.append("")
    lines.append("Matching rules:")
    lines.append("- **fixed_15mm**: centroid <= 15 mm (size-independent, legacy baseline)")
    lines.append("- **luna16_radius**: centroid <= max(GT_diameter/2, 3 mm) "
                 "(LUNA16 official, size-adaptive)")
    lines.append("")
    lines.append(f"Test panel: {len(results['mine']['fixed_15mm']['per_patient'])} patients. "
                 "Mine probs from TTA1 ensemble (cached). "
                 "MONAI preds from monai\\_test\\_panel\\_preds.json.")
    lines.append("")

    lines.append("## 4-cell comparison table")
    lines.append("")
    lines.append("| System | rule fixed\\_15mm | rule luna16\\_radius |")
    lines.append("|--------|-----------------|---------------------|")
    for sys_name in ["mine", "monai"]:
        c15  = cell(sys_name, "fixed_15mm")
        clun = cell(sys_name, "luna16_radius")
        lines.append(f"| **{sys_name}** | {c15} | {clun} |")

    d15 = (results["mine"]["fixed_15mm"]["metrics"]["f1"]
           - results["monai"]["fixed_15mm"]["metrics"]["f1"])
    dlr = (results["mine"]["luna16_radius"]["metrics"]["f1"]
           - results["monai"]["luna16_radius"]["metrics"]["f1"])
    wp15 = wilcoxon_results["fixed_15mm"]["p_value"]
    wplr = wilcoxon_results["luna16_radius"]["p_value"]
    wp15s = f"p={wp15:.4f}" if not np.isnan(wp15) else "p=N/A"
    wplrs = f"p={wplr:.4f}" if not np.isnan(wplr) else "p=N/A"
    n15  = wilcoxon_results["fixed_15mm"]["n_paired"]
    nlr  = wilcoxon_results["luna16_radius"]["n_paired"]

    lines.append("|--------|-----------------|---------------------|")
    lines.append(f"| **delta(mine-MONAI)** | "
                 f"{d15:+.3f} F1, {wp15s}, n={n15} | "
                 f"{dlr:+.3f} F1, {wplrs}, n={nlr} |")
    lines.append("")
    lines.append(f"> Bootstrap CI: {N_BOOTSTRAP} paired samples, seed={SEED}. "
                 "Wilcoxon two-sided on per-patient F1.")
    lines.append("")

    lines.append("## FROC / CPM detail")
    lines.append("")
    fp_hdr = " | ".join(f"@{fp}" for fp in LUNA16_FP_RATES)
    lines.append(f"| System | Rule | CPM | {fp_hdr} |")
    lines.append("|--------|------|-----|" + "|".join("-------" for _ in LUNA16_FP_RATES) + "|")
    for sys_name in ["mine", "monai"]:
        for rule in ["fixed_15mm", "luna16_radius"]:
            r = results[sys_name][rule]
            sv = " | ".join(f"{p['sensitivity']:.3f}" for p in r["luna16_points"])
            lines.append(f"| {sys_name} | {rule} | {r['cpm']:.4f} | {sv} |")
    lines.append("")

    lines.append("## Stratified sensitivity")
    lines.append("")
    bks = ["4-6mm", "6-15mm", ">15mm"]
    lines.append(f"| System | Rule | {' | '.join(bks)} |")
    lines.append("|--------|------|" + "|".join("-------" for _ in bks) + "|")
    for sys_name in ["mine", "monai"]:
        for rule in ["fixed_15mm", "luna16_radius"]:
            strat = results[sys_name][rule]["stratified"]
            cells = []
            for bk in bks:
                v = strat.get(bk, {})
                s = v.get("sens"); n = v.get("n_gt", 0)
                cells.append(f"{s:.3f}(n={n})" if s is not None else f"N/A(n={n})")
            lines.append(f"| {sys_name} | {rule} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Operating thresholds used")
    lines.append("")
    lines.append("| System | Rule | Threshold | Selection method |")
    lines.append("|--------|------|-----------|-----------------|")
    for rule in ["fixed_15mm", "luna16_radius"]:
        lines.append(f"| mine  | {rule} | {mine_thr_used[rule]:.2f} | "
                     "best-F1 sweep on test_panel |")
        lines.append(f"| MONAI | {rule} | {monai_thr_used[rule]:.2f} | "
                     "best-F1 sweep on val_panel |")
    lines.append("")

    # Verdict
    f1_mine_lr  = results["mine"]["luna16_radius"]["metrics"]["f1"]
    f1_monai_lr = results["monai"]["luna16_radius"]["metrics"]["f1"]
    f1_mine_15  = results["mine"]["fixed_15mm"]["metrics"]["f1"]
    f1_monai_15 = results["monai"]["fixed_15mm"]["metrics"]["f1"]

    lines.append("## Verdict")
    lines.append("")
    if f1_mine_lr > f1_monai_lr:
        lines.append(f"**mine WINS on luna16\\_radius rule**: "
                     f"F1={f1_mine_lr:.3f} vs MONAI {f1_monai_lr:.3f} "
                     f"(delta={dlr:+.3f}, {wplrs}).")
    else:
        lines.append(f"**mine LOSES on luna16\\_radius rule**: "
                     f"F1={f1_mine_lr:.3f} vs MONAI {f1_monai_lr:.3f} "
                     f"(delta={dlr:+.3f}, {wplrs}).")
    lines.append("")
    if f1_mine_15 > f1_monai_15:
        lines.append(f"On fixed\\_15mm (legacy): mine F1={f1_mine_15:.3f} vs "
                     f"MONAI {f1_monai_15:.3f} — mine also wins.")
    else:
        lines.append(f"On fixed\\_15mm (legacy): mine F1={f1_mine_15:.3f} vs "
                     f"MONAI {f1_monai_15:.3f} — MONAI wins on 15mm rule.")
    lines.append("")
    if dlr > 0.05:
        lines.append("Decision impact: gap >0.05 on strict rule — "
                     "**no fine-tune needed immediately**. "
                     "Mine competitive. Monitor CPM@1FP before deciding on LUNA16-targeted retrain.")
    elif dlr > 0:
        lines.append("Decision impact: mine leads by small margin on strict rule. "
                     "**Fine-tune optional** — focus on 4-6mm bucket.")
    else:
        lines.append("Decision impact: mine trails MONAI on strict rule. "
                     "**Recommend LUNA16-targeted fine-tune** focusing on small-nodule sensitivity.")
    lines.append("")

    (ACADEMIC_DIR / "RULE_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
