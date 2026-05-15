"""End-to-end orchestration: cache val -> sweep -> apply -> cache test -> final eval.

Run autonomously:
  python run_full_tuning_pipeline.py
  python run_full_tuning_pipeline.py --quick      # smaller grid for speed
  python run_full_tuning_pipeline.py --skip-cache # if probs already cached
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run(label: str, cmd: list, allow_fail: bool = False) -> int:
    print(f"\n{'=' * 70}\n[{label}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, env=ENV, cwd=str(ROOT))
    dt = time.time() - t0
    print(f"\n[{label}] exit={p.returncode}  ({dt:.0f}s)", flush=True)
    if p.returncode != 0 and not allow_fail:
        sys.exit(f"FAILED at: {label}")
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Use smaller sweep grid")
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--skip-test", action="store_true",
                    help="Don't cache+evaluate test panel")
    args = ap.parse_args()

    quick = ["--quick"] if args.quick else []

    # 1. Cache val_panel
    if not args.skip_cache:
        run("CACHE val_panel",
            [sys.executable, "tune_detection_params.py", "--panel", "val", "--cache-only"])

    # 2. Sweep on val_panel
    run("SWEEP val_panel",
        [sys.executable, "tune_detection_params.py", "--panel", "val"] + quick)

    # 3. Compare baseline vs tuned on val
    run("COMPARE baseline vs tuned (val)",
        [sys.executable, "compare_baseline_vs_tuned.py", "--panel", "val"])

    # 4. FROC plot
    run("PLOT FROC (val)",
        [sys.executable, "plot_tuning_froc.py", "--panel", "val"])

    # 5. Apply best config to webapp/configs.py
    run("APPLY best config",
        [sys.executable, "apply_best_config.py"])

    if args.skip_test:
        print("\nSkipping test panel.")
        return

    # 6. Cache test_panel
    run("CACHE test_panel",
        [sys.executable, "tune_detection_params.py", "--panel", "test", "--cache-only"])

    # 7. Compare baseline vs tuned on test (locked)
    run("COMPARE baseline vs tuned (test)",
        [sys.executable, "compare_baseline_vs_tuned.py", "--panel", "test",
         "--out", str(ROOT.parent / "work/academic/baseline_vs_tuned_TEST.json")])

    # 8. FROC plot on test
    run("PLOT FROC (test)",
        [sys.executable, "plot_tuning_froc.py", "--panel", "test",
         "--out", str(ROOT.parent / "work/academic/froc_baseline_vs_tuned_TEST.png")])

    # 9. Per-patient compare on test
    run("COMPARE AI vs GT (test, per-patient)",
        [sys.executable, "compare_ai_vs_gt.py", "--panel", "test",
         "--out", str(ROOT.parent / "work/academic/compare_ai_vs_gt_TEST.json")])

    print(f"\n{'=' * 70}\nALL DONE.")
    print("Artifacts in work/academic/:")
    print("  best_config.json")
    print("  tuning_results.csv  tuning_summary.csv")
    print("  baseline_vs_tuned.json (val)")
    print("  baseline_vs_tuned_TEST.json (locked test)")
    print("  froc_baseline_vs_tuned.png  froc_baseline_vs_tuned_TEST.png")
    print("  compare_ai_vs_gt_TEST.json")


if __name__ == "__main__":
    main()
