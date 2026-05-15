"""FPR retraining with proper academic protocol.

Old pipeline (Phase 2):
  - Extracted candidates from val_panel for FPR training (96 patients)
  - Tested on test_panel candidates
  - Issue: small training set (1283 cands), threshold chosen on test (mild leak in spirit)

New pipeline (this script):
  Stage 1: Extract candidates from train_panel  (812 patients ≈ 8x more data)
  Stage 2: Train FPR on train_panel candidates
  Stage 3: Pick FPR threshold on val_panel candidates  (held-out from training)
  Stage 4: Lock test_panel candidates, single eval, no further tuning

Run end-to-end (assumes new segmentation model exists at <ckpt_path>):
  python retrain_fpr_pipeline.py --seg-ckpt work/runs_exp/stage2_full/best.pt
  python retrain_fpr_pipeline.py --seg-ckpt work/runs/best.pt --tag baseline_fpr_v2

For each candidate uses extract_fpr_candidates.py + train_fpr_classifier.py
+ apply_fpr_classifier.py with the proper sequencing.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import WORK

ACADEMIC = WORK / "academic"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def run(label, cmd, allow_fail=False):
    print(f"\n{'='*70}\n[{label}] {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, env=ENV, cwd=str(Path(__file__).resolve().parent))
    dt = time.time() - t0
    print(f"[{label}] exit={p.returncode}  ({dt:.0f}s)")
    if p.returncode != 0 and not allow_fail:
        sys.exit(f"FAILED at: {label}")
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-ckpt", default="work/runs/best.pt",
                    help="Segmentation checkpoint to generate candidates with")
    ap.add_argument("--tag", default="fpr_v2",
                    help="Tag for output files (e.g., fpr_v2_stage2winner)")
    ap.add_argument("--threshold", type=float, default=0.40,
                    help="Seg threshold for candidate generation (permissive)")
    ap.add_argument("--epochs", type=int, default=40,
                    help="FPR classifier training epochs")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--skip-cache", action="store_true",
                    help="Assume probs already cached (skip the slow GPU step)")
    args = ap.parse_args()

    cache_train_npz = ACADEMIC / f"{args.tag}_train.npz"
    cache_val_npz = ACADEMIC / f"{args.tag}_val.npz"
    cache_test_npz = ACADEMIC / f"{args.tag}_test.npz"
    fpr_ckpt = WORK / "runs" / f"{args.tag}.pt"
    eval_val_json = ACADEMIC / f"{args.tag}_eval_val.json"
    eval_test_json = ACADEMIC / f"{args.tag}_eval_test.json"

    # Pre-step: cache prob volumes with the chosen segmentation ckpt for all 3 panels.
    # If args.seg_ckpt is the default (work/runs/best.pt), this is a no-op for val+test
    # but train_panel will need ~1-2h GPU. For new stage2 ckpt, all 3 panels need caching.
    probs_dir = ACADEMIC / "probs"  # default location used by extract_fpr_candidates
    if not args.skip_cache:
        for panel in ("train", "val", "test"):
            run(f"CACHE probs for {panel}_panel",
                [sys.executable, "cache_panel_probs.py",
                 "--panel", panel,
                 "--ckpt", args.seg_ckpt,
                 "--out-dir", str(probs_dir)])

    # Stage 1: extract train_panel candidates (THE NEW BIG DATASET)
    if not cache_train_npz.exists():
        run("EXTRACT train_panel candidates",
            [sys.executable, "extract_fpr_candidates.py",
             "--panel", "train", "--threshold", str(args.threshold),
             "--out", str(cache_train_npz)])
    else:
        print(f"[skip] {cache_train_npz} exists")

    # Extract val + test (smaller, used for threshold + locked eval)
    if not cache_val_npz.exists():
        run("EXTRACT val_panel candidates",
            [sys.executable, "extract_fpr_candidates.py",
             "--panel", "val", "--threshold", str(args.threshold),
             "--out", str(cache_val_npz)])
    if not cache_test_npz.exists():
        run("EXTRACT test_panel candidates (LOCKED)",
            [sys.executable, "extract_fpr_candidates.py",
             "--panel", "test", "--threshold", str(args.threshold),
             "--out", str(cache_test_npz)])

    # Stage 2: train FPR on train_panel, monitor on val_panel
    run(f"TRAIN FPR on train_panel ({args.epochs}ep)",
        [sys.executable, "train_fpr_classifier.py",
         "--train", str(cache_train_npz),
         "--val", str(cache_val_npz),
         "--epochs", str(args.epochs),
         "--bs", str(args.bs),
         "--out", str(fpr_ckpt)])

    # Stage 3: pick threshold on val_panel (already done inside train_fpr_classifier
    # which writes best_thr — but we re-evaluate to make it explicit)
    run("EVAL FPR on val_panel (threshold sweep)",
        [sys.executable, "apply_fpr_classifier.py",
         "--candidates", str(cache_val_npz),
         "--ckpt", str(fpr_ckpt),
         "--out", str(eval_val_json)])

    # Stage 4: locked test_panel single eval
    run("EVAL FPR on test_panel (LOCKED, single shot)",
        [sys.executable, "apply_fpr_classifier.py",
         "--candidates", str(cache_test_npz),
         "--ckpt", str(fpr_ckpt),
         "--out", str(eval_test_json)])

    # Summary
    print(f"\n=== FPR RETRAIN COMPLETE ===")
    print(f"  Checkpoint: {fpr_ckpt}")
    print(f"  Val eval:   {eval_val_json}")
    print(f"  Test eval:  {eval_test_json}")
    val = json.loads(eval_val_json.read_text())
    test = json.loads(eval_test_json.read_text())
    print(f"\n  VAL  best F1 row: {val['with_fpr_best_f1']}")
    print(f"  TEST best F1 row: {test['with_fpr_best_f1']}")


if __name__ == "__main__":
    main()
