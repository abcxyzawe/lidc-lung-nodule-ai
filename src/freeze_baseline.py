"""Phase A — Freeze current production baseline before any retraining.

Copies current checkpoints, configs, and result JSONs to work/baseline_frozen/<TIMESTAMP>/
so we can roll back / compare against a known-good demo state.

Run: python freeze_baseline.py
"""
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
SRC = ROOT / "src"
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
FROZEN_DIR = WORK / "baseline_frozen" / TIMESTAMP


def copy_safe(src, dst):
    if not src.exists():
        print(f"  [SKIP] {src} (not found)")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    print(f"  [OK]   {src.name} -> {dst.relative_to(WORK)}")
    return True


def main():
    print(f"=== Freezing baseline to {FROZEN_DIR} ===\n")
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoints
    print("[1/4] Checkpoints (work/runs/*.pt)")
    for ckpt in ["best.pt", "swa.pt", "fpr.pt", "malignancy.pt",
                  "snapshot_e030.pt", "snapshot_e058.pt", "snapshot_e085.pt",
                  "last.pt"]:
        copy_safe(WORK / "runs" / ckpt, FROZEN_DIR / "runs" / ckpt)

    # 2. Configs
    print("\n[2/4] Configs + webapp pipeline")
    for f in ["configs.py", "panels.py"]:
        copy_safe(SRC / f, FROZEN_DIR / "src" / f)
    copy_safe(SRC / "webapp" / "app.py", FROZEN_DIR / "src" / "webapp" / "app.py")
    copy_safe(SRC / "webapp" / "predict.py", FROZEN_DIR / "src" / "webapp" / "predict.py")

    # 3. Phase 1+2+3 results
    print("\n[3/4] Academic results (work/academic/)")
    academic = WORK / "academic"
    for f in ["TUNING_SUMMARY.md", "best_config.json",
              "tuning_results.csv", "tuning_summary.csv",
              "baseline_vs_tuned.json", "baseline_vs_tuned_TEST.json",
              "compare_ai_vs_gt_TEST.json", "fpr_eval.json", "fpr_eval_strict.json",
              "missed_nodules_analysis.json", "froc_baseline_vs_tuned.png"]:
        copy_safe(academic / f, FROZEN_DIR / "academic" / f)

    # 4. Top-level docs
    print("\n[4/4] Top-level docs")
    for f in ["PROJECT_REPORT_FULL.md", "RETRAIN_README.md",
              "ACADEMIC_EVALUATION.md", "DOCUMENTATION.md", "README.md"]:
        copy_safe(ROOT / f, FROZEN_DIR / f)

    # 5. Manifest
    manifest = {
        "timestamp": TIMESTAMP,
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Pre-retrain baseline freeze. Restore by copying back to original paths.",
        "current_metrics_test_panel_locked": {
            "phase1_strict": {
                "sens": 0.536, "prec": 0.670, "f1": 0.596, "fp_per_scan": 0.69,
                "config": "min_voxels=120, max_elong=4, merge=10mm, subp=0, threshold=0.97",
            },
            "phase2_sensitive": {
                "sens": 0.804, "prec": 0.301, "f1": 0.439, "fp_per_scan": 3.61,
                "config": "FPR @ 0.30, candidate threshold 0.40, min_voxels 20",
            },
            "phase2_balanced": {
                "sens": 0.696, "prec": 0.381, "f1": 0.492, "fp_per_scan": 2.19,
                "config": "FPR @ 0.70",
            },
            "baseline_old": {
                "sens": 0.424, "prec": 0.675, "f1": 0.522, "fp_per_scan": 0.54,
                "config": "min_voxels=200, threshold=0.65, subpleural=2.0",
            },
        },
        "fpr_classifier": {
            "test_auc": 0.830, "best_thr": 0.80, "best_f1": 0.521,
            "trained_on": "val_panel candidates (1283 samples)",
            "protocol_issue": "Trained on val, tested on test. Not leak but not clean.",
        },
        "restore_instructions": [
            "1. cp -r work/baseline_frozen/<TIMESTAMP>/runs/* work/runs/",
            "2. cp work/baseline_frozen/<TIMESTAMP>/src/configs.py src/",
            "3. cp work/baseline_frozen/<TIMESTAMP>/src/webapp/* src/webapp/",
            "4. Restart webapp: cd src/webapp && uvicorn app:app --reload --port 8000",
        ],
    }
    (FROZEN_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {FROZEN_DIR / 'MANIFEST.json'}")
    total_size = sum(f.stat().st_size for f in FROZEN_DIR.rglob("*") if f.is_file())
    print(f"\n=== Done. Frozen baseline: {FROZEN_DIR} ({total_size/1e6:.0f} MB) ===")


if __name__ == "__main__":
    main()
