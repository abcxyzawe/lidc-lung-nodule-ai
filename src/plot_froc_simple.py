"""Plot FROC from already-computed JSON data (no re-evaluation).

Uses best_config.json which already contains the tuned FROC points.
Adds the baseline operating point computed manually below as a single marker.

Run: python plot_froc_simple.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ACADEMIC = ROOT / "work" / "academic"


def main():
    cfg = json.loads((ACADEMIC / "best_config.json").read_text())
    val_cmp = json.loads((ACADEMIC / "baseline_vs_tuned.json").read_text())
    test_cmp = json.loads((ACADEMIC / "baseline_vs_tuned_TEST.json").read_text())

    # Tuned FROC from val_panel
    pts = sorted(cfg["froc_points"], key=lambda p: p["fp_per_scan"])
    fps = [p["fp_per_scan"] for p in pts]
    sens = [p["sensitivity"] for p in pts]

    # Baseline single point (val + test)
    base_val = val_cmp["baseline_metrics"]
    tuned_val = val_cmp["tuned_metrics"]
    base_test = test_cmp["baseline_metrics"]
    tuned_test = test_cmp["tuned_metrics"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(fps, sens, "b-o", linewidth=2, markersize=6,
            label="Tuned config (val_panel FROC sweep)")
    ax.scatter([base_val["fp_per_scan"]], [base_val["sensitivity"]],
               s=180, marker="*", c="orange", edgecolor="black", zorder=5,
               label=f"Baseline (val): sens={base_val['sensitivity']:.3f} "
                      f"FP/sc={base_val['fp_per_scan']:.2f}")
    ax.scatter([tuned_val["fp_per_scan"]], [tuned_val["sensitivity"]],
               s=180, marker="P", c="blue", edgecolor="black", zorder=5,
               label=f"Tuned (val):    sens={tuned_val['sensitivity']:.3f} "
                      f"FP/sc={tuned_val['fp_per_scan']:.2f}")
    ax.scatter([base_test["fp_per_scan"]], [base_test["sensitivity"]],
               s=180, marker="*", c="red", edgecolor="black", zorder=5,
               label=f"Baseline (TEST): sens={base_test['sensitivity']:.3f} "
                      f"FP/sc={base_test['fp_per_scan']:.2f}")
    ax.scatter([tuned_test["fp_per_scan"]], [tuned_test["sensitivity"]],
               s=180, marker="P", c="green", edgecolor="black", zorder=5,
               label=f"Tuned (TEST):    sens={tuned_test['sensitivity']:.3f} "
                      f"FP/sc={tuned_test['fp_per_scan']:.2f}")

    ax.set_xscale("log")
    ax.set_xlim(0.05, 30)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("False Positives per scan (log scale)")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Lung-nodule detection: baseline vs F1-tuned\n"
                  f"Tuned config: mv120 elong4 merge10 subp0  |  CPM (val) = {cfg['cpm']:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    out = ACADEMIC / "froc_baseline_vs_tuned.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
