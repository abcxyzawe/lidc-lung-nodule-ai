"""Run LIDC test_panel scoring for LUNA16 fine-tuned best.pt.

Runs inference on local 4090 (no V100 needed).
Caches prob volumes under work/academic/probs_luna16_ft/
Produces:
  work/runs_luna16/run001/MINE_POST_FT_TEST_PANEL.json
  work/academic/THESIS_VERDICT_FINAL.md

Matching rules: fixed_15mm + luna16_radius
Bootstrap CI 1000 paired delta mine_post vs MONAI
Wilcoxon p-value (patient-level F1)

Run:
  python src/eval_finetune_on_lidc.py [--force-recache]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))

from configs import WORK, PRE_DIR
from panels import load_panel
from benchmark import load_patient
from tune_detection_params import (
    precompute_threshold_blobs,
    filter_and_match,
    LUNA16_FP_RATES,
    cache_lung_mask,
)
from scipy.ndimage import distance_transform_edt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN001_DIR = WORK / "runs_luna16" / "run001"
BEST_PT = RUN001_DIR / "best.pt"
PROBS_DIR = WORK / "academic" / "probs_luna16_ft"

# Threshold sweep (same range as reeval_luna16_rule.py)
THRESHOLD_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Post-proc (same as reeval_luna16_rule.py)
POST_CFG = {"min_voxels": 200, "max_elong": 4.0,
            "merge_dist_mm": 10.0, "subpleural_min_mm": 2.0}

# Reference numbers (from MINE_LUNA16_RULE.json + MONAI_LUNA16_RULE.json)
MINE_PRE = {
    "fixed_15mm":    {"f1": 0.5551, "cpm": 0.4727},
    "luna16_radius": {"f1": 0.5209, "cpm": 0.4394},
}
MONAI_REF = {
    "fixed_15mm":    {"f1": 0.5455, "cpm": 0.5609},
    "luna16_radius": {"f1": 0.5178, "cpm": 0.5297},
}

# LUNA16 fine-tune HU window (wider than LIDC HU_LO=-1350)
HU_LO_FT = -1000.0
HU_HI_FT = 200.0
PATCH_SLICES = 5


def normalize_hu_ft(x):
    x = np.clip(x.astype(np.float32), HU_LO_FT, HU_HI_FT)
    return (x - HU_LO_FT) / (HU_HI_FT - HU_LO_FT)


def load_ft_model():
    """Load LUNA16 fine-tuned model (5-channel UNet++ EfficientNet-B5)."""
    import segmentation_models_pytorch as smp
    if not BEST_PT.exists():
        raise FileNotFoundError(
            f"best.pt not found at {BEST_PT}. "
            "Run src/pull_run001.py first to sync from V100."
        )
    ck = torch.load(str(BEST_PT), map_location=DEVICE, weights_only=False)
    in_ch = ck.get("in_channels", PATCH_SLICES)
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b5", encoder_weights=None,
        in_channels=in_ch, classes=1, decoder_attention_type="scse",
    ).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.train(False)
    print(f"Loaded ft model: epoch={ck.get('epoch')}  "
          f"cpm={ck.get('val_froc_cpm','N/A')}  in_ch={in_ch}")
    return model, in_ch


def infer_volume(model, vol, n_channels, tta=True):
    """2.5D inference with LUNA16 HU normalization. Returns prob float32 [Z,H,W]."""
    N = vol.shape[0]
    prob = np.zeros_like(vol, dtype=np.float32)
    half = n_channels // 2
    with torch.no_grad():
        for i in range(N):
            idxs = [max(0, min(N - 1, i + off)) for off in range(-half, half + 1)]
            stk = np.stack([normalize_hu_ft(vol[j]) for j in idxs], axis=0)
            x = torch.from_numpy(stk).unsqueeze(0).float().to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p1 = torch.sigmoid(model(x))
                if tta:
                    p2 = torch.sigmoid(model(torch.flip(x, dims=[-1]))).flip(dims=[-1])
                    p = (p1 + p2) / 2
                else:
                    p = p1
            prob[i] = p[0, 0].float().cpu().numpy()
    return prob


def cache_ft_probs(model, in_ch, pids, force=False):
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    for i, pid in enumerate(pids):
        cp = PROBS_DIR / f"{pid}.npy"
        if cp.exists() and not force:
            continue
        data = load_patient(pid)
        if data is None:
            print(f"  [{i+1}/{len(pids)}] {pid}: MISS (no h5)")
            continue
        t0 = time.time()
        prob = infer_volume(model, data["vol"], in_ch, tta=True)
        np.save(cp, prob.astype(np.float16))
        print(f"  [{i+1}/{len(pids)}] {pid}: cached ({time.time()-t0:.1f}s)", flush=True)


def load_patient_bundle(pid):
    prob_path = PROBS_DIR / f"{pid}.npy"
    if not prob_path.exists():
        return None
    data = load_patient(pid)
    if data is None:
        return None
    prob = np.load(prob_path).astype(np.float32)
    lung = cache_lung_mask(pid, data["vol"])
    sp = data["voxel_sp"]
    ldist = distance_transform_edt(lung, sampling=list(sp)).astype(np.float32)
    return {"pid": pid, "prob": prob, "lung": lung, "ldist": ldist,
            "voxel_sp": sp, "gt": data["gt_nodules"]}


def _match_fixed_15mm(res, patient_dict):
    """Re-match filter_and_match results using fixed 15mm centroid rule."""
    preds = res["preds"]
    sp = np.array(patient_dict["voxel_sp"])
    matched = []
    used_p = set()
    for gi, g in enumerate(patient_dict["gt"]):
        gc = np.array(g["centroid_zyx_voxel"], dtype=np.float32) * sp
        best_pi, bestd = None, float("inf")
        for pi, p in enumerate(preds):
            if pi in used_p:
                continue
            d = float(np.linalg.norm(gc - p["centroid_mm"]))
            if d < bestd:
                bestd, best_pi = d, pi
        if best_pi is not None and bestd <= 15.0:
            matched.append((gi, best_pi))
            used_p.add(best_pi)
    ptp = len(matched)
    pfp = len(preds) - ptp
    pfn = len(patient_dict["gt"]) - ptp
    return ptp, pfp, pfn, set(m[0] for m in matched)


def score_threshold(patient_data, thr, match_rule):
    """Score one threshold over all patients. Returns aggregate + per-patient F1."""
    tp = fp = fn = n_gt = 0
    per_pat = []
    for pd in patient_data:
        blobs = precompute_threshold_blobs(pd["prob"], pd["lung"], pd["voxel_sp"], thr)
        res = filter_and_match(
            blobs, pd["ldist"], pd["voxel_sp"], pd["gt"],
            min_voxels=POST_CFG["min_voxels"], max_elong=POST_CFG["max_elong"],
            merge_dist_mm=POST_CFG["merge_dist_mm"],
            subpleural_min_mm=POST_CFG["subpleural_min_mm"],
            match_min_radius_mm=3.0,
        )
        if match_rule == "luna16_radius":
            ptp, pfp, pfn = res["tp"], res["fp"], res["fn"]
        else:
            ptp, pfp, pfn, _ = _match_fixed_15mm(res, pd)

        tp += ptp; fp += pfp; fn += pfn
        n_gt += len(pd["gt"])
        pg_prec = ptp / max(ptp + pfp, 1)
        pg_rec = ptp / max(len(pd["gt"]), 1)
        pg_f1 = 2 * pg_prec * pg_rec / max(pg_prec + pg_rec, 1e-7)
        per_pat.append({"pid": pd["pid"], "tp": ptp, "fp": pfp, "fn": pfn,
                        "n_gt": len(pd["gt"]), "f1": pg_f1})

    sens = tp / max(n_gt, 1)
    fp_per_scan = fp / max(len(patient_data), 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-7)
    return {"thr": thr, "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt,
            "sens": sens, "fp_per_scan": fp_per_scan, "prec": prec, "f1": f1,
            "per_patient": per_pat}


def compute_froc(patient_data, match_rule):
    print(f"  Sweeping {len(THRESHOLD_SWEEP)} thresholds for rule={match_rule} ...", flush=True)
    all_pts = []
    for thr in THRESHOLD_SWEEP:
        r = score_threshold(patient_data, thr, match_rule)
        print(f"    thr={thr:.2f}  sens={r['sens']:.3f}  FP/scan={r['fp_per_scan']:.2f}"
              f"  F1={r['f1']:.3f}  TP={r['tp']} FP={r['fp']} FN={r['fn']}", flush=True)
        all_pts.append(r)

    pts_sorted = sorted(all_pts, key=lambda p: p["fp_per_scan"])
    fp_arr = np.array([p["fp_per_scan"] for p in pts_sorted])
    sens_arr = np.array([p["sens"] for p in pts_sorted])
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
    best = max(all_pts, key=lambda x: x["f1"])
    return {
        "cpm": cpm,
        "luna16_points": luna_sens,
        "best": {k: v for k, v in best.items() if k != "per_patient"},
        "all_points": [{k: v for k, v in p.items() if k != "per_patient"} for p in all_pts],
        "per_patient_at_best_f1": best["per_patient"],
    }


def stratified_at_best(patient_data, best_thr, match_rule):
    buckets = {"small_4_6mm": (4, 6), "medium_6_15mm": (6, 15), "large_15mm_plus": (15, 999)}
    bs = {k: {"tp": 0, "fn": 0, "n_gt": 0} for k in buckets}
    for pd in patient_data:
        blobs = precompute_threshold_blobs(pd["prob"], pd["lung"], pd["voxel_sp"], best_thr)
        res = filter_and_match(
            blobs, pd["ldist"], pd["voxel_sp"], pd["gt"],
            min_voxels=POST_CFG["min_voxels"], max_elong=POST_CFG["max_elong"],
            merge_dist_mm=POST_CFG["merge_dist_mm"],
            subpleural_min_mm=POST_CFG["subpleural_min_mm"],
            match_min_radius_mm=3.0,
        )
        if match_rule == "luna16_radius":
            matched_gi = {m[0] for m in res["matched"]}
        else:
            _, _, _, matched_gi = _match_fixed_15mm(res, pd)

        for gi, g in enumerate(pd["gt"]):
            d = g["diam_mm"]
            for name, (lo, hi) in buckets.items():
                if lo <= d < hi:
                    bs[name]["n_gt"] += 1
                    if gi in matched_gi:
                        bs[name]["tp"] += 1
                    else:
                        bs[name]["fn"] += 1
                    break

    return {k: {"n_gt": v["n_gt"], "tp": v["tp"], "fn": v["fn"],
                "sensitivity": v["tp"] / v["n_gt"] if v["n_gt"] > 0 else None}
            for k, v in bs.items()}


def bootstrap_ci(per_patient_f1, n_boot=1000, seed=42):
    """1000-rep bootstrap paired delta mine_post - MONAI_aggregate."""
    rng = np.random.default_rng(seed)
    mine_arr = np.array([p["f1"] for p in per_patient_f1])
    n = len(mine_arr)
    monai_scalar = MONAI_REF["fixed_15mm"]["f1"]
    deltas = mine_arr - monai_scalar
    obs_delta = float(deltas.mean())
    boot_deltas = [float(deltas[rng.integers(0, n, size=n)].mean()) for _ in range(n_boot)]
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    return {"observed_delta": obs_delta, "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
            "n_patients": n, "n_boot": n_boot,
            "note": "Per-patient delta vs MONAI aggregate F1=0.5455 (MONAI per-patient N/A)"}


def wilcoxon_test(per_patient_f1):
    from scipy.stats import wilcoxon
    mine_arr = np.array([p["f1"] for p in per_patient_f1])
    monai_scalar = MONAI_REF["fixed_15mm"]["f1"]
    diffs = mine_arr - monai_scalar
    try:
        stat, pv = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        return {"statistic": float(stat), "p_value": float(pv),
                "significant": bool(pv < 0.05)}
    except Exception as exc:
        return {"error": str(exc)}


def write_thesis_verdict(result, out_path):
    f15 = result["fixed_15mm"]
    lr = result["luna16_radius"]
    mine_post_f15_f1 = f15["best"]["f1"]
    mine_post_f15_cpm = f15["cpm"]
    mine_post_lr_f1 = lr["best"]["f1"]
    mine_post_lr_cpm = lr["cpm"]
    boot = result.get("bootstrap_ci", {})
    wilc = result.get("wilcoxon", {})

    f1_beats_monai = mine_post_f15_f1 >= MONAI_REF["fixed_15mm"]["f1"]
    cpm_within_3pt = mine_post_f15_cpm >= (MONAI_REF["fixed_15mm"]["cpm"] - 0.03)
    f1_regression = mine_post_f15_f1 < MINE_PRE["fixed_15mm"]["f1"]

    if f1_regression:
        verdict = "ROLLBACK"
        narrative = "mine_post REGRESSION vs mine_pre -- do NOT claim improvement."
        action = "Rollback to winner ckpt cd48359. Do NOT publish fine-tuned weights."
        v100_rec = "SHUTDOWN V100 (no run2 needed -- regression confirms fine-tune hurt LIDC performance)."
    elif f1_beats_monai and cpm_within_3pt:
        verdict = "SUCCESS"
        narrative = "mine_post matches or beats MONAI on both F1 and CPM."
        action = (
            "Publish narrative: LIDC-trained model matches MONAI RetinaNet 3D "
            "on held-out LIDC test panel after LUNA16 fine-tuning."
        )
        v100_rec = "SHUTDOWN V100 (result achieved, no run2 needed)."
    elif f1_beats_monai and not cpm_within_3pt:
        verdict = "PARTIAL"
        narrative = "mine_post F1 >= MONAI but CPM substantially lower -- comparable F1, higher precision."
        action = (
            "Acceptable thesis claim with precision-recall tradeoff noted. "
            "Consider run2 ablation to close CPM gap."
        )
        v100_rec = "KEEP V100 ALIVE for run2 (CPM gap needs attention)."
    else:
        verdict = "BELOW_MONAI"
        narrative = "mine_post below MONAI on F1 -- fine-tuning did not close the gap."
        action = "Run run2 ablation (longer training, FPR re-tune, threshold re-sweep)."
        v100_rec = "KEEP V100 ALIVE for run2."

    delta_f1 = mine_post_f15_f1 - MONAI_REF["fixed_15mm"]["f1"]
    delta_cpm = mine_post_f15_cpm - MONAI_REF["fixed_15mm"]["cpm"]
    strat = result.get("strat_fixed", {})
    sm = strat.get("small_4_6mm", {}).get("sensitivity") or 0
    md = strat.get("medium_6_15mm", {}).get("sensitivity") or 0
    lg = strat.get("large_15mm_plus", {}).get("sensitivity") or 0
    ci_lo = boot.get("ci_95_lo", "N/A")
    ci_hi = boot.get("ci_95_hi", "N/A")
    p_val = wilc.get("p_value", "N/A")
    boot_obs = boot.get("observed_delta", "N/A")

    lines = [
        "# THESIS VERDICT FINAL",
        f"Generated: 2026-05-22",
        f"Panel: LIDC test_panel (99 patients, {result['n_patients']} loaded, {result['n_gt']} GT nodules)",
        "",
        "## 6-Cell Comparison Table",
        "",
        "|                    | fixed_15mm                        | luna16_radius                     |",
        "|--------------------|-----------------------------------|-----------------------------------|",
        f"| mine (pre-FT)      | F1={MINE_PRE['fixed_15mm']['f1']:.4f} CPM={MINE_PRE['fixed_15mm']['cpm']:.4f} | F1={MINE_PRE['luna16_radius']['f1']:.4f} CPM={MINE_PRE['luna16_radius']['cpm']:.4f} |",
        f"| mine (post-FT)     | F1={mine_post_f15_f1:.4f} CPM={mine_post_f15_cpm:.4f} | F1={mine_post_lr_f1:.4f} CPM={mine_post_lr_cpm:.4f} |",
        f"| MONAI RetinaNet 3D | F1={MONAI_REF['fixed_15mm']['f1']:.4f} CPM={MONAI_REF['fixed_15mm']['cpm']:.4f} | F1={MONAI_REF['luna16_radius']['f1']:.4f} CPM={MONAI_REF['luna16_radius']['cpm']:.4f} |",
        "",
        "## Deltas (mine_post vs MONAI, fixed_15mm)",
        f"- Delta F1:  {delta_f1:+.4f}  (positive = mine_post better)",
        f"- Delta CPM: {delta_cpm:+.4f}  (positive = mine_post better)",
        "",
        "## Statistical Tests (fixed_15mm, 1000-rep bootstrap)",
        f"- Bootstrap observed delta: {boot_obs:+.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]",
        f"- Wilcoxon signed-rank p-value: {p_val:.4f}",
        "",
        "## Stratified Sensitivity (fixed_15mm, best threshold)",
        "| Size bucket       | mine post-FT | MONAI   |",
        "|-------------------|--------------|---------|",
        f"| small 4-6mm       | {sm:.3f}        | 0.522   |",
        f"| medium 6-15mm     | {md:.3f}        | 0.600   |",
        f"| large 15mm+       | {lg:.3f}        | 0.541   |",
        "",
        f"## Verdict: {verdict}",
        narrative,
        "",
        "## Recommendation",
        action,
        "",
        "## V100 Decision",
        v100_rec,
        "",
        "---",
        "Notes:",
        "- LUNA16 val CPM=0.9708 is patch-level on subset 9 (88 scans) -- NOT comparable to slice-level LIDC CPM.",
        "- fixed_15mm: centroid distance <= 15mm (non-official, conservative matching rule).",
        "- luna16_radius: centroid <= max(GT_diam/2, 3mm) -- closer to LUNA16 official rule.",
        "- MONAI per-patient F1 not available; bootstrap uses MONAI aggregate F1 as reference.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-recache", action="store_true",
                    help="Re-run inference even if prob files exist")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    model, in_ch = load_ft_model()
    pids = load_panel("test")
    print(f"\nTest panel: {len(pids)} patients")

    print("\n[Step 1] Cache fine-tuned prob volumes ...")
    cache_ft_probs(model, in_ch, pids, force=args.force_recache)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[Step 2] Load patient bundles ...")
    patient_data = []
    for pid in pids:
        bundle = load_patient_bundle(pid)
        if bundle:
            patient_data.append(bundle)
    n_loaded = len(patient_data)
    n_gt_total = sum(len(pd["gt"]) for pd in patient_data)
    print(f"  Loaded {n_loaded}/{len(pids)} patients  GT nodules: {n_gt_total}")

    print("\n[Step 3] FROC sweep -- fixed_15mm ...")
    r_fixed = compute_froc(patient_data, "fixed_15mm")
    print(f"\n  CPM={r_fixed['cpm']:.4f}  Best F1={r_fixed['best']['f1']:.4f}  "
          f"thr={r_fixed['best']['thr']}  sens={r_fixed['best']['sens']:.3f}  "
          f"prec={r_fixed['best']['prec']:.3f}")

    print("\n[Step 4] FROC sweep -- luna16_radius ...")
    r_luna = compute_froc(patient_data, "luna16_radius")
    print(f"\n  CPM={r_luna['cpm']:.4f}  Best F1={r_luna['best']['f1']:.4f}  "
          f"thr={r_luna['best']['thr']}  sens={r_luna['best']['sens']:.3f}  "
          f"prec={r_luna['best']['prec']:.3f}")

    print("\n[Step 5] Stratified sensitivity ...")
    strat_fixed = stratified_at_best(patient_data, r_fixed["best"]["thr"], "fixed_15mm")
    strat_luna = stratified_at_best(patient_data, r_luna["best"]["thr"], "luna16_radius")
    for name, s in strat_fixed.items():
        sens_str = f"{s['sensitivity']:.3f}" if s["sensitivity"] is not None else "N/A"
        print(f"  fixed_15mm  {name:22} n_gt={s['n_gt']:3}  tp={s['tp']:3}  sens={sens_str}")

    print("\n[Step 6] Statistical tests ...")
    mine_pp = r_fixed["per_patient_at_best_f1"]
    boot = bootstrap_ci(mine_pp, n_boot=1000, seed=42)
    wilc = wilcoxon_test(mine_pp)
    pv = wilc.get("p_value", "ERR")
    print(f"  Bootstrap delta={boot['observed_delta']:+.4f}  "
          f"CI=[{boot['ci_95_lo']:.4f},{boot['ci_95_hi']:.4f}]")
    print(f"  Wilcoxon p={pv}")

    result = {
        "eval_date": "2026-05-22",
        "ckpt": str(BEST_PT),
        "ckpt_epoch": 1,
        "ckpt_luna16_val_cpm": 0.9708,
        "n_patients": n_loaded,
        "n_gt": n_gt_total,
        "post_proc": POST_CFG,
        "fixed_15mm": r_fixed,
        "luna16_radius": r_luna,
        "strat_fixed": strat_fixed,
        "strat_luna": strat_luna,
        "bootstrap_ci": boot,
        "wilcoxon": wilc,
        "reference": {"mine_pre": MINE_PRE, "monai": MONAI_REF},
    }

    out_json = RUN001_DIR / "MINE_POST_FT_TEST_PANEL.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_json}")

    out_md = WORK / "academic" / "THESIS_VERDICT_FINAL.md"
    verdict = write_thesis_verdict(result, out_md)

    f15 = result["fixed_15mm"]
    lr = result["luna16_radius"]
    print("\n=== 6-CELL FINAL TABLE ===")
    print(f"{'':20} {'fixed_15mm':35} {'luna16_radius'}")
    print(f"{'mine (pre)':20} F1={MINE_PRE['fixed_15mm']['f1']:.4f} "
          f"CPM={MINE_PRE['fixed_15mm']['cpm']:.4f}"
          f"     F1={MINE_PRE['luna16_radius']['f1']:.4f} CPM={MINE_PRE['luna16_radius']['cpm']:.4f}")
    print(f"{'mine (post-FT)':20} F1={f15['best']['f1']:.4f} "
          f"CPM={f15['cpm']:.4f}"
          f"     F1={lr['best']['f1']:.4f} CPM={lr['cpm']:.4f}")
    print(f"{'MONAI':20} F1={MONAI_REF['fixed_15mm']['f1']:.4f} "
          f"CPM={MONAI_REF['fixed_15mm']['cpm']:.4f}"
          f"     F1={MONAI_REF['luna16_radius']['f1']:.4f} CPM={MONAI_REF['luna16_radius']['cpm']:.4f}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
