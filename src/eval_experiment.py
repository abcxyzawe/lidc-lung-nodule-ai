"""Evaluate a trained experiment on val_panel + locked test_panel.

Pipeline:
  1. Cache prob volumes on val_panel + test_panel using exp's checkpoint
  2. Sweep post-processing on val_panel -> per-exp best_config
  3. Apply best_config on locked test_panel -> final metrics
  4. Per-bucket sensitivity + invisible/weak/visible analysis
  5. Save work/runs_exp/<name>/eval.json + append row to comparison.csv

Run:
  python eval_experiment.py --exp consensus2
  python eval_experiment.py --exp all       # eval every experiment with a ckpt
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from configs import WORK, PRE_DIR
from panels import load_panel
from benchmark import load_patient
from tune_detection_params import (
    PROBS_DIR, cache_lung_mask, precompute_threshold_blobs,
    filter_and_match, LUNA16_FP_RATES,
)
from experiments import EXPERIMENTS, get_experiment
import segmentation_models_pytorch as smp

EXP_RUNS = WORK / "runs_exp"
EXP_PROBS_ROOT = WORK / "academic" / "probs_exp"
COMPARISON_CSV = WORK / "academic" / "experiments_comparison.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_exp_model(exp_name):
    cfg = get_experiment(exp_name)
    if exp_name in ("baseline", "exp00_baseline"):
        # Use existing best.pt + swa.pt
        from importlib import import_module
        train_mod = import_module("05_train")
        model = train_mod.make_model(encoder=cfg.encoder, pretrained=None).to(DEVICE)
        ck_path = WORK / "runs" / "best.pt"
        ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
    else:
        ck_path = EXP_RUNS / exp_name / "best.pt"
        if not ck_path.exists():
            raise FileNotFoundError(f"No checkpoint at {ck_path}. Train first.")
        ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
        in_ch = ck.get("in_channels", cfg.in_channels)
        model = smp.UnetPlusPlus(
            encoder_name=cfg.encoder, encoder_weights=None,
            in_channels=in_ch, classes=1, decoder_attention_type="scse",
        ).to(DEVICE)
        model.load_state_dict(ck["model"])
    model.train(False)
    return model, cfg


def normalize(x):
    from configs import HU_LO, HU_HI
    x = np.clip(x.astype(np.float32), HU_LO, HU_HI)
    return (x - HU_LO) / (HU_HI - HU_LO)


def predict_volume(model, vol, n_channels, tta=True):
    N, H, W = vol.shape
    prob = np.zeros((N, H, W), dtype=np.float32)
    half = n_channels // 2
    with torch.no_grad():
        for i in range(N):
            idxs = [max(0, min(N-1, i + off)) for off in range(-half, half + 1)]
            stk = np.stack([normalize(vol[j]) for j in idxs], axis=0)
            x = torch.from_numpy(stk).unsqueeze(0).float().to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                p1 = torch.sigmoid(model(x))
                if tta:
                    p2 = torch.sigmoid(model(torch.flip(x, dims=[-1]))).flip(dims=[-1])
                    p = (p1 + p2) / 2
                else:
                    p = p1
            prob[i] = p[0, 0].float().cpu().numpy()
    return prob


def cache_exp_probs(exp_name, panel_pids, model, n_channels):
    """Cache prob volumes for this experiment under work/academic/probs_exp/<name>/."""
    d = EXP_PROBS_ROOT / exp_name
    d.mkdir(parents=True, exist_ok=True)
    for i, pid in enumerate(panel_pids):
        cp = d / f"{pid}.npy"
        if cp.exists(): continue
        data = load_patient(pid)
        if data is None: continue
        t0 = time.time()
        prob = predict_volume(model, data["vol"], n_channels)
        np.save(cp, prob.astype(np.float16))
        print(f"  [{i+1}/{len(panel_pids)}] {pid}: cached ({time.time()-t0:.1f}s)", flush=True)


def evaluate_panel(exp_name, panel_name, cfg_post, threshold):
    """Apply config post-processing, return aggregate metrics + stratified."""
    pids = load_panel(panel_name)
    probs_dir = EXP_PROBS_ROOT / exp_name
    tp = fp = fn = n_gt = 0
    strat = {"sm_4_6": [0, 0], "md_6_15": [0, 0], "lg_15p": [0, 0]}
    for pid in pids:
        cp = probs_dir / f"{pid}.npy"
        if not cp.exists(): continue
        data = load_patient(pid)
        if data is None: continue
        prob = np.load(cp).astype(np.float32)
        lung = cache_lung_mask(pid, data["vol"])
        sp = data["voxel_sp"]
        ldist = distance_transform_edt(lung, sampling=list(sp)).astype(np.float32)
        blobs = precompute_threshold_blobs(prob, lung, sp, threshold)
        res = filter_and_match(
            blobs, ldist, sp, data["gt_nodules"],
            min_voxels=cfg_post["min_voxels"], max_elong=cfg_post["max_elong"],
            merge_dist_mm=cfg_post["merge_dist_mm"],
            subpleural_min_mm=cfg_post["subpleural_min_mm"],
        )
        tp += res["tp"]; fp += res["fp"]; fn += res["fn"]; n_gt += res["n_gt"]
        matched_gi = {m[0] for m in res["matched"]}
        for gi, g in enumerate(data["gt_nodules"]):
            d = g["diam_mm"]
            if 4 <= d < 6: bk = "sm_4_6"
            elif 6 <= d < 15: bk = "md_6_15"
            elif d >= 15: bk = "lg_15p"
            else: continue
            strat[bk][1] += 1
            if gi in matched_gi: strat[bk][0] += 1
    sens = tp / max(n_gt, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-7)
    return {
        "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt,
        "sens": sens, "prec": prec, "f1": f1,
        "fp_per_scan": fp / max(len(pids), 1),
        "stratified": {k: {"tp": v[0], "n_gt": v[1],
                            "sens": v[0] / max(v[1], 1)} for k, v in strat.items()},
    }


def quick_threshold_sweep_on_val(exp_name, thresholds=(0.30, 0.50, 0.70, 0.85, 0.95)):
    """Sweep thresholds on val with default post-proc, pick best F1."""
    POST = {"min_voxels": 120, "max_elong": 4.0, "merge_dist_mm": 10, "subpleural_min_mm": 0}
    best = None
    for thr in thresholds:
        m = evaluate_panel(exp_name, "val", POST, thr)
        m["threshold"] = thr; m["post"] = POST
        print(f"  thr={thr:.2f}  sens={m['sens']:.3f} prec={m['prec']:.3f} "
              f"F1={m['f1']:.3f} FP/sc={m['fp_per_scan']:.2f}")
        if best is None or m["f1"] > best["f1"]: best = m
    return best


def visibility_analysis(exp_name):
    """For each test_panel GT centroid, peak prob in 5-vox cube. Bucket as Phase 3."""
    pids = load_panel("test")
    probs_dir = EXP_PROBS_ROOT / exp_name
    rows = []
    for pid in pids:
        cp = probs_dir / f"{pid}.npy"
        if not cp.exists(): continue
        data = load_patient(pid)
        if data is None: continue
        prob = np.load(cp).astype(np.float32)
        for g in data["gt_nodules"]:
            cz, cy, cx = [int(round(c)) for c in g["centroid_zyx_voxel"]]
            z0, z1 = max(0, cz-3), min(prob.shape[0], cz+4)
            y0, y1 = max(0, cy-8), min(prob.shape[1], cy+9)
            x0, x1 = max(0, cx-8), min(prob.shape[2], cx+9)
            roi = prob[z0:z1, y0:y1, x0:x1]
            peak = float(roi.max()) if roi.size else 0.0
            rows.append({"diam_mm": g["diam_mm"], "n_rads": g["n_radiologists"],
                          "peak_prob": peak,
                          "bucket": "invisible" if peak < 0.3 else
                                    "weak" if peak < 0.7 else "visible"})
    return rows


def eval_full_ct_fp_rate(exp_name, n_patients=10):
    """KEY metric for fix-distribution-shift evaluation.

    For N test patients: run inference on FULL DICOM (all slices, not preprocessed)
    and report how many predicted nodules per scan. Compare with the per-scan GT
    count. The "extra" predictions on full CT vs on kept-slice h5 measure the
    distribution shift FP injection.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
    import pydicom
    from collections import defaultdict
    from configs import DICOM_ROOT
    from predict import (
        read_dicom_series, segment_lung, clean_mask,
        find_nodules, merge_nearby_nodules, filter_subpleural,
    )
    cfg = get_experiment(exp_name)
    model, _ = load_exp_model(exp_name)
    n_ch = cfg.in_channels

    pids = load_panel("test")[:n_patients]
    rows = []
    for pid in pids:
        # find DICOM
        p_dir = DICOM_ROOT / pid
        if not p_dir.exists():
            continue
        by_series = defaultdict(list)
        for dp in p_dir.rglob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(dp), stop_before_pixels=True, force=True,
                                     specific_tags=["SeriesInstanceUID"])
                by_series[str(ds.SeriesInstanceUID)].append(str(dp))
            except Exception:
                continue
        if not by_series: continue
        best_uid = max(by_series, key=lambda k: len(by_series[k]))
        dcm_paths = by_series[best_uid]
        try:
            vol_full, voxel_sp = read_dicom_series(dcm_paths)
        except Exception as e:
            print(f"  {pid}: read fail {e}"); continue
        N = vol_full.shape[0]
        # Predict on full CT (single-pass without TTA for speed during eval)
        prob = np.zeros_like(vol_full, dtype=np.float32)
        from configs import HU_LO, HU_HI
        def normalize(x):
            x = np.clip(x.astype(np.float32), HU_LO, HU_HI)
            return (x - HU_LO) / (HU_HI - HU_LO)
        half = n_ch // 2
        with torch.no_grad():
            for i in range(N):
                idxs = [max(0, min(N-1, i + off)) for off in range(-half, half + 1)]
                stk = np.stack([normalize(vol_full[j]) for j in idxs], axis=0)
                x = torch.from_numpy(stk).unsqueeze(0).float().to(DEVICE)
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    p = torch.sigmoid(model(x))
                prob[i] = p[0, 0].float().cpu().numpy()
        # Post-proc
        lung = segment_lung(vol_full)
        thr = 0.97  # use Phase 1 strict threshold
        pred = (prob > thr).astype(np.uint8) & lung
        pred = clean_mask(pred, voxel_sp)
        nodules, _ = find_nodules(pred, voxel_sp, min_voxels=120, max_elongation=4.0,
                                   prob_volume=prob, core_threshold=0.85)
        nodules = merge_nearby_nodules(nodules, voxel_sp, max_dist_mm=10.0)
        nodules = filter_subpleural(nodules, lung, voxel_sp, min_dist_mm=0.0)
        # GT count (from preprocessed)
        data = load_patient(pid)
        n_gt = len(data["gt_nodules"]) if data else 0
        slices_with_signal = int((prob.max(axis=(1,2)) > 0.5).sum())
        rows.append({
            "pid": pid, "n_slices_full": N, "n_pred_full": len(nodules),
            "n_gt_pre": n_gt, "slices_signal_gt0_5": slices_with_signal,
            "extra_preds": max(0, len(nodules) - n_gt),
        })
        print(f"  {pid}: slices={N}, pred_full={len(nodules)}, gt={n_gt}, "
              f"signal_slices={slices_with_signal}, extra={max(0,len(nodules)-n_gt)}",
              flush=True)
    del model; torch.cuda.empty_cache()
    if not rows:
        return {"per_patient": [], "avg_extra_pred": None,
                "avg_signal_slice_pct": None}
    avg_extra = np.mean([r["extra_preds"] for r in rows])
    avg_pct = np.mean([100*r["slices_signal_gt0_5"]/max(r["n_slices_full"],1) for r in rows])
    print(f"  AVG extra preds (FP injection from full CT): {avg_extra:.2f}/scan")
    print(f"  AVG % slices firing signal>0.5: {avg_pct:.1f}%")
    return {"per_patient": rows, "avg_extra_pred": float(avg_extra),
            "avg_signal_slice_pct": float(avg_pct)}


def evaluate_one(exp_name):
    print(f"\n{'='*70}\n[{exp_name}] EVAL\n{'='*70}")
    out_dir = EXP_RUNS / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_exp_model(exp_name)
    n_ch = cfg.in_channels

    # 1. Cache probs val + test (skips existing)
    print(f"\n[Cache] val_panel ({len(load_panel('val'))} patients)")
    cache_exp_probs(exp_name, load_panel("val"), model, n_ch)
    print(f"\n[Cache] test_panel ({len(load_panel('test'))} patients)")
    cache_exp_probs(exp_name, load_panel("test"), model, n_ch)
    del model; torch.cuda.empty_cache()

    # 2. Pick threshold on val
    print(f"\n[Sweep] threshold on val_panel")
    best_val = quick_threshold_sweep_on_val(exp_name)
    print(f"\n  Best val: thr={best_val['threshold']} F1={best_val['f1']:.3f}")

    # 3. Apply on locked test
    print(f"\n[Test] applying best config on test_panel (LOCKED)")
    test_metrics = evaluate_panel(exp_name, "test", best_val["post"], best_val["threshold"])
    print(f"  TP={test_metrics['tp']} FP={test_metrics['fp']} FN={test_metrics['fn']}")
    print(f"  sens={test_metrics['sens']:.3f}  prec={test_metrics['prec']:.3f}  "
          f"F1={test_metrics['f1']:.3f}  FP/scan={test_metrics['fp_per_scan']:.2f}")
    print(f"  Strat: small {test_metrics['stratified']['sm_4_6']}  "
          f"med {test_metrics['stratified']['md_6_15']}  "
          f"large {test_metrics['stratified']['lg_15p']}")

    # 4. Visibility analysis
    print(f"\n[Visibility] test_panel GT centroids")
    vis = visibility_analysis(exp_name)
    n = len(vis); n_inv = sum(1 for r in vis if r["bucket"] == "invisible")
    n_w = sum(1 for r in vis if r["bucket"] == "weak")
    n_v = sum(1 for r in vis if r["bucket"] == "visible")
    print(f"  invisible {n_inv}/{n} ({100*n_inv/max(n,1):.1f}%)  "
          f"weak {n_w}/{n} ({100*n_w/max(n,1):.1f}%)  "
          f"visible {n_v}/{n} ({100*n_v/max(n,1):.1f}%)")

    # 5. Full-CT FP rate (KEY for distribution-shift verification)
    print(f"\n[FullCT] FP rate verification on 10 test patients (full DICOM, single-pass)")
    full_ct_eval = eval_full_ct_fp_rate(exp_name, n_patients=10)

    # 6. Save eval.json + append to comparison
    eval_out = {
        "experiment": exp_name, "config": cfg.__dict__,
        "best_val": best_val, "test_metrics": test_metrics,
        "visibility_test": {"invisible": n_inv, "weak": n_w, "visible": n_v, "total": n},
        "full_ct_fp": full_ct_eval,
    }
    (out_dir / "eval.json").write_text(json.dumps(eval_out, indent=2, default=str))

    row = {
        "experiment": exp_name,
        "test_sens": test_metrics["sens"], "test_prec": test_metrics["prec"],
        "test_f1": test_metrics["f1"], "test_fp_per_scan": test_metrics["fp_per_scan"],
        "test_tp": test_metrics["tp"], "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"],
        "sens_small": test_metrics["stratified"]["sm_4_6"]["sens"],
        "sens_medium": test_metrics["stratified"]["md_6_15"]["sens"],
        "sens_large": test_metrics["stratified"]["lg_15p"]["sens"],
        "invisible_pct": 100 * n_inv / max(n, 1),
        "best_threshold": best_val["threshold"],
        "fullct_avg_extra_pred": full_ct_eval["avg_extra_pred"],
        "fullct_avg_signal_slice_pct": full_ct_eval["avg_signal_slice_pct"],
    }
    new_csv = not COMPARISON_CSV.exists()
    with open(COMPARISON_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_csv: w.writeheader()
        w.writerow(row)
    print(f"\nWrote {out_dir / 'eval.json'} + appended to {COMPARISON_CSV}")
    return eval_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    help="Experiment name or 'all' to eval every trained one")
    args = ap.parse_args()
    if args.exp == "all":
        for name in EXPERIMENTS:
            try:
                evaluate_one(name)
            except Exception as e:
                print(f"[{name}] FAILED: {type(e).__name__}: {e}")
    else:
        evaluate_one(args.exp)


if __name__ == "__main__":
    main()
