"""Apply trained FPR classifier to candidate patches and recompute metrics.

Loads a previously extracted candidate dataset (fpr_test.npz), runs the trained
FPR classifier on each patch, and computes:
  - Operating-point FROC (re-rank by classifier prob, threshold sweep)
  - Compare to baseline (no FPR, just seg confidence)

Run:
  python apply_fpr_classifier.py --candidates work/academic/fpr_test.npz \
      --ckpt work/runs/fpr.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import DenseNet121

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import RUNS_DIR, WORK  # noqa: E402
from train_fpr_classifier import normalize_hu  # noqa: E402

ACADEMIC = WORK / "academic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ckpt", default=str(RUNS_DIR / "fpr.pt"))
    ap.add_argument("--out", default=str(ACADEMIC / "fpr_eval.json"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.train(False)
    fpr_thr = ck.get("best_thr", 0.5)
    print(f"FPR ckpt: epoch {ck.get('epoch')}  AUC {ck.get('auc'):.3f}  thr {fpr_thr:.2f}")

    d = np.load(args.candidates)
    patches = d["patches"]
    labels = d["labels"]
    confs = d["confs"]
    sources_arr = d["sources"]  # may be a numpy str array
    sources = [str(s) for s in sources_arr]
    print(f"Candidates: {len(patches)}  ({(labels==1).sum()} pos / {(labels==0).sum()} neg)")

    fpr_probs = np.zeros(len(patches), dtype=np.float32)
    bs = 64
    with torch.no_grad():
        for i in range(0, len(patches), bs):
            batch = np.stack(
                [normalize_hu(patches[j]) for j in range(i, min(i+bs, len(patches)))], 0
            )
            x = torch.from_numpy(batch).unsqueeze(1).float().to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                p = torch.softmax(model(x), dim=1)[:, 1]
            fpr_probs[i:i+x.size(0)] = p.cpu().numpy()
    print(f"FPR probs: mean={fpr_probs.mean():.3f}  pos_mean={fpr_probs[labels==1].mean():.3f}  "
          f"neg_mean={fpr_probs[labels==0].mean():.3f}")

    n_pat = len(set(sources))
    print(f"\n=== Eval at multiple FPR thresholds ===")
    print(f"  N patients: {n_pat}")

    results = []
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        keep = fpr_probs >= thr
        tp = int(((labels == 1) & keep).sum())
        fp = int(((labels == 0) & keep).sum())
        fn = int((labels == 1).sum() - tp)
        sens = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-7)
        fpps = fp / n_pat
        results.append({"thr": thr, "tp": tp, "fp": fp, "fn": fn,
                         "sens": sens, "prec": prec, "f1": f1, "fp_per_scan": fpps})
        print(f"  fpr_thr={thr:.2f}  TP={tp:>3} FP={fp:>4} FN={fn:>3}  "
              f"sens={sens:.3f} prec={prec:.3f} F1={f1:.3f} FP/scan={fpps:.2f}")

    keep = np.ones(len(patches), dtype=bool)
    base_tp = int(((labels == 1) & keep).sum())
    base_fp = int(((labels == 0) & keep).sum())
    base_fn = int((labels == 1).sum() - base_tp)
    base_sens = base_tp / max(base_tp + base_fn, 1)
    base_prec = base_tp / max(base_tp + base_fp, 1)
    base_f1 = 2 * base_prec * base_sens / max(base_prec + base_sens, 1e-7)
    print(f"\n=== Compare ===")
    print(f"  WITHOUT FPR: TP={base_tp} FP={base_fp} FN={base_fn} "
          f"sens={base_sens:.3f} prec={base_prec:.3f} F1={base_f1:.3f} "
          f"FP/scan={base_fp/n_pat:.2f}")
    best_f1_row = max(results, key=lambda r: r["f1"])
    print(f"  WITH FPR (best F1 thr={best_f1_row['thr']:.2f}): "
          f"TP={best_f1_row['tp']} FP={best_f1_row['fp']} FN={best_f1_row['fn']} "
          f"sens={best_f1_row['sens']:.3f} prec={best_f1_row['prec']:.3f} "
          f"F1={best_f1_row['f1']:.3f} FP/scan={best_f1_row['fp_per_scan']:.2f}")

    Path(args.out).write_text(json.dumps({
        "ckpt": args.ckpt, "candidates": args.candidates,
        "ckpt_thr": fpr_thr,
        "n_patients": n_pat, "n_candidates": int(len(patches)),
        "without_fpr": {"tp": base_tp, "fp": base_fp, "fn": base_fn,
                         "sens": base_sens, "prec": base_prec, "f1": base_f1,
                         "fp_per_scan": base_fp / n_pat},
        "with_fpr_thresholds": results,
        "with_fpr_best_f1": best_f1_row,
    }, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
