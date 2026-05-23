"""
render_thesis_charts.py — Render matplotlib figures cho luận văn LIDC-IDRI.

Không dùng TensorBoard. Parse log files trực tiếp, render 4 figures (8 files).

Output: work/academic/figures/
  - stage2_loss_dice.png/.pdf         (Figure 1)
  - fpr_malignancy_curves.png/.pdf    (Figure 2)
  - stage2_overview_combined.png/.pdf (Figure 3)
  - malignancy_overview_combined.png/.pdf (Figure 4)

Reuse parse functions từ tb_convert.py (import trực tiếp).

Author: ml-engineer (autopilot)
Date:   2026-05-22
"""

import pathlib
import sys
import re

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO     = pathlib.Path(__file__).resolve().parent.parent   # E:\Phan Tich Ung Thu
WORK     = pathlib.Path(__file__).resolve().parent          # E:\Phan Tich Ung Thu\work

STAGE2_LOG = WORK / "runs_exp" / "stage2_full" / "log.txt"
MAL_LOG    = WORK / "runs" / "mal_train.log"

OUT_DIR    = WORK / "academic" / "figures"

# ---------------------------------------------------------------------------
# Color palette — professional navy/teal/orange (no neon)
# ---------------------------------------------------------------------------
C_NAVY    = "#1B3A6B"
C_TEAL    = "#1A7A8A"
C_ORANGE  = "#D4600A"
C_RUST    = "#A63D2F"
C_GREY    = "#6B7280"
C_LIGHT   = "#E8EFF7"
C_GRID    = "#D1D5DB"

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.labelweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "grid.color":         C_GRID,
    "grid.linewidth":     0.6,
    "grid.alpha":         0.8,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   C_GRID,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "figure.dpi":         200,
})

# ---------------------------------------------------------------------------
# Parse helpers (mirrored from tb_convert.py — no circular import needed)
# ---------------------------------------------------------------------------

def parse_stage2_log(path: pathlib.Path) -> list:
    """
    Parse Stage 2 milestone log.
    Returns list of dicts: {epoch, train_loss (may be None), val_dice}
    """
    records = []
    pat = re.compile(
        r"ep\s+(\d+)/\d+\s+"
        r"loss=([\d.?]+)\s+"
        r"val_dice=([\d.]+)"
    )
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                ep        = int(m.group(1))
                loss_raw  = m.group(2)
                vdice     = float(m.group(3))
                loss      = float(loss_raw) if re.fullmatch(r"[\d.]+", loss_raw) else None
                records.append({"epoch": ep, "train_loss": loss, "val_dice": vdice})
    return records


def parse_mal_log(path: pathlib.Path) -> list:
    """
    Parse FPR+Malignancy dense log (60 epochs).
    Returns list of dicts: {epoch, train_loss, val_acc, val_bal_acc, val_susp_f1}
    """
    records = []
    pat = re.compile(
        r"epoch\s+(\d+)/\d+\s+"
        r"loss=([\d.]+)\s+"
        r"val_acc=([\d.]+)\s+"
        r"val_bal_acc=([\d.]+)\s+"
        r"val_susp_f1=([\d.]+)"
    )
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                records.append({
                    "epoch":       int(m.group(1)),
                    "train_loss":  float(m.group(2)),
                    "val_acc":     float(m.group(3)),
                    "val_bal_acc": float(m.group(4)),
                    "val_susp_f1": float(m.group(5)),
                })
    return records

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_fig(fig, stem: str) -> tuple:
    """Save figure as both PNG and PDF. Returns (png_path, pdf_path)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    return png_path, pdf_path


def annotate_best(ax, x_best, y_best, label_text,
                  line_color=C_ORANGE, text_offset=(6, 0)):
    """Draw vertical dashed line + annotation box at best epoch."""
    ax.axvline(x=x_best, color=line_color, linewidth=1.4,
               linestyle="--", alpha=0.85, zorder=3)
    ax.annotate(
        label_text,
        xy=(x_best, y_best),
        xytext=(x_best + text_offset[0], y_best + text_offset[1]),
        fontsize=9,
        color=line_color,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=line_color,
                        lw=1.2, connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=line_color, alpha=0.9),
        zorder=5,
    )


def add_swa_band(ax, swa_start=136, x_max=180, alpha=0.07):
    """Shade SWA region."""
    ax.axvspan(swa_start, x_max, alpha=alpha, color=C_TEAL, zorder=0)
    ax.text(
        swa_start + 1, ax.get_ylim()[0],
        "SWA", fontsize=8, color=C_TEAL, alpha=0.7,
        va="bottom", ha="left",
    )

# ===========================================================================
# Figure 1: stage2_loss_dice.png — 2 subplots dọc
# ===========================================================================

def render_figure1(stage2: list) -> tuple:
    """
    Figure 1: Stage 2 train_loss (top) + val_dice (bottom), milestone sparse.
    """
    # Separate points with/without loss (ep 16 has loss=None)
    eps_loss  = [r["epoch"] for r in stage2 if r["train_loss"] is not None]
    vals_loss = [r["train_loss"] for r in stage2 if r["train_loss"] is not None]
    eps_dice  = [r["epoch"] for r in stage2]
    vals_dice = [r["val_dice"] for r in stage2]

    best_ep    = 16
    best_dice  = 0.8676

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={"hspace": 0.45}
    )

    fig.suptitle(
        "Stage 2 — Huấn luyện Phân đoạn Nốt\n"
        "(UNet++ EfficientNet-B5 SCSE, Tversky Loss, 180 Epochs)",
        fontsize=14, fontweight="bold", y=0.98
    )

    # --- Subplot 1: Train Loss ---
    ax1.plot(eps_loss, vals_loss,
             color=C_NAVY, linewidth=2.0, marker="o",
             markersize=6, markerfacecolor="white",
             markeredgecolor=C_NAVY, markeredgewidth=1.5,
             label="Train Loss (Tversky)", zorder=4)

    # Mark ep 16 as missing with different symbol
    ax1.axvline(x=best_ep, color=C_ORANGE, linewidth=1.2,
                linestyle="--", alpha=0.6, zorder=3)
    ax1.text(best_ep + 1, min(vals_loss) + 0.002,
             "ep 16\n(loss không\nghi lại)",
             fontsize=8, color=C_ORANGE, alpha=0.8)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss (Tversky)")
    ax1.set_title("(a) Train Loss theo Epoch")
    ax1.set_xlim(0, 185)
    ax1.set_ylim(min(vals_loss) - 0.008, max(vals_loss) + 0.012)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper right")

    # SWA band on loss plot
    ax1.axvspan(136, 180, alpha=0.07, color=C_TEAL, zorder=0)
    ax1.text(137, ax1.get_ylim()[0] + 0.001, "SWA",
             fontsize=8, color=C_TEAL, alpha=0.75)

    # Milestone note
    ax1.text(0.02, 0.97,
             f"Dữ liệu sparse: {len(eps_loss)} mốc / 180 epoch",
             transform=ax1.transAxes, fontsize=8.5, color=C_GREY,
             va="top", ha="left", style="italic")

    # --- Subplot 2: Val Dice ---
    ax2.plot(eps_dice, vals_dice,
             color=C_TEAL, linewidth=2.0, marker="s",
             markersize=6, markerfacecolor="white",
             markeredgecolor=C_TEAL, markeredgewidth=1.5,
             label="Val Dice", zorder=4)

    # Highlight best epoch 16
    annotate_best(
        ax2, x_best=best_ep, y_best=best_dice,
        label_text=f"Tốt nhất\nEpoch {best_ep}\nDice = {best_dice:.4f}",
        line_color=C_ORANGE,
        text_offset=(8, -0.015)
    )

    # SWA band
    ax2.axvspan(136, 180, alpha=0.07, color=C_TEAL, zorder=0)
    ax2.text(137, 0.705, "SWA", fontsize=8, color=C_TEAL, alpha=0.75)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Dice Score")
    ax2.set_title("(b) Val Dice Score theo Epoch")
    ax2.set_xlim(0, 185)
    ax2.set_ylim(0.70, 0.90)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax2.grid(True, axis="y")
    ax2.legend(loc="lower left")

    # Horizontal reference line at best dice
    ax2.axhline(y=best_dice, color=C_ORANGE, linewidth=0.8,
                linestyle=":", alpha=0.5, zorder=2)

    return save_fig(fig, "stage2_loss_dice")


# ===========================================================================
# Figure 2: fpr_malignancy_curves.png — 3 subplots dọc
# ===========================================================================

def render_figure2(mal: list) -> tuple:
    """
    Figure 2: FPR+Malignancy — train_loss, val_bal_acc, val_susp_f1.
    """
    eps       = [r["epoch"] for r in mal]
    losses    = [r["train_loss"] for r in mal]
    bal_accs  = [r["val_bal_acc"] for r in mal]
    susp_f1s  = [r["val_susp_f1"] for r in mal]

    best_idx     = int(np.argmax(bal_accs))
    best_ep      = eps[best_idx]         # 52
    best_bal_acc = bal_accs[best_idx]    # 0.4593
    best_f1_at   = susp_f1s[best_idx]   # 0.6542

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 12),
        gridspec_kw={"hspace": 0.50}
    )

    fig.suptitle(
        "FPR + Malignancy Classifier — Huấn luyện DenseNet121-3D\n"
        "(60 Epochs, Adam lr=1e-4, Weighted Cross-Entropy)",
        fontsize=14, fontweight="bold", y=0.98
    )

    # --- Subplot 1: Train Loss ---
    ax1.plot(eps, losses,
             color=C_NAVY, linewidth=2.0,
             label="Train Loss")

    ax1.fill_between(eps, losses, alpha=0.08, color=C_NAVY)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("(a) Train Loss theo Epoch")
    ax1.set_xlim(0, 62)
    ax1.set_ylim(min(losses) - 0.04, max(losses) + 0.04)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper right")

    # Annotate start/end values
    ax1.annotate(f"{losses[0]:.4f}",
                 xy=(eps[0], losses[0]),
                 xytext=(eps[0] + 2, losses[0] - 0.04),
                 fontsize=8.5, color=C_NAVY,
                 arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.8))
    ax1.annotate(f"{losses[-1]:.4f}",
                 xy=(eps[-1], losses[-1]),
                 xytext=(eps[-1] - 12, losses[-1] - 0.06),
                 fontsize=8.5, color=C_NAVY,
                 arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.8))

    reduction_pct = (1 - losses[-1] / losses[0]) * 100
    ax1.text(0.98, 0.95,
             f"Giam {reduction_pct:.0f}%\n({losses[0]:.3f} -> {losses[-1]:.4f})",
             transform=ax1.transAxes, fontsize=9,
             color=C_NAVY, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=C_LIGHT,
                       edgecolor=C_NAVY, alpha=0.9))

    # --- Subplot 2: Val Balanced Accuracy ---
    ax2.plot(eps, bal_accs,
             color=C_TEAL, linewidth=2.0, marker="o",
             markersize=3.5, markerfacecolor=C_TEAL,
             label="Val Balanced Accuracy", zorder=4)

    annotate_best(
        ax2, x_best=best_ep, y_best=best_bal_acc,
        label_text=f"Tot nhat\nEpoch {best_ep}\nBal Acc = {best_bal_acc:.4f}",
        line_color=C_ORANGE,
        text_offset=(3, -0.012)
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Balanced Accuracy")
    ax2.set_title("(b) Val Balanced Accuracy theo Epoch")
    ax2.set_xlim(0, 62)
    bal_min = min(bal_accs)
    bal_max = max(bal_accs)
    margin  = (bal_max - bal_min) * 0.15
    ax2.set_ylim(bal_min - margin, bal_max + margin)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax2.grid(True, axis="y")
    ax2.legend(loc="lower right")

    # Baseline random = 0.2 (5 classes)
    ax2.axhline(y=0.2, color=C_GREY, linewidth=0.9,
                linestyle=":", alpha=0.7, zorder=2)
    ax2.text(1, 0.2 + 0.002, "Random (5 classes)",
             fontsize=8, color=C_GREY, ha="left")

    # --- Subplot 3: Val Suspicious F1 ---
    ax3.plot(eps, susp_f1s,
             color=C_RUST, linewidth=2.0, marker="s",
             markersize=3.5, markerfacecolor=C_RUST,
             label="Val Suspicious F1", zorder=4)

    # Mark best epoch
    ax3.axvline(x=best_ep, color=C_ORANGE, linewidth=1.4,
                linestyle="--", alpha=0.85, zorder=3)
    ax3.annotate(
        f"Epoch {best_ep}\nSusp F1 = {best_f1_at:.4f}",
        xy=(best_ep, best_f1_at),
        xytext=(best_ep + 3, best_f1_at - 0.012),
        fontsize=9, color=C_ORANGE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_ORANGE,
                        lw=1.2, connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C_ORANGE, alpha=0.9),
        zorder=5,
    )

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Val Suspicious F1")
    ax3.set_title("(c) Val Suspicious F1 theo Epoch")
    ax3.set_xlim(0, 62)
    f1_min = min(susp_f1s)
    f1_max = max(susp_f1s)
    margin = (f1_max - f1_min) * 0.15
    ax3.set_ylim(f1_min - margin, f1_max + margin)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax3.grid(True, axis="y")
    ax3.legend(loc="lower right")

    return save_fig(fig, "fpr_malignancy_curves")


# ===========================================================================
# Figure 3: stage2_overview_combined.png — twin y-axis
# ===========================================================================

def render_figure3(stage2: list) -> tuple:
    """
    Figure 3: Stage 2 train_loss + val_dice on twin y-axis.
    """
    eps_loss  = [r["epoch"] for r in stage2 if r["train_loss"] is not None]
    vals_loss = [r["train_loss"] for r in stage2 if r["train_loss"] is not None]
    eps_dice  = [r["epoch"] for r in stage2]
    vals_dice = [r["val_dice"] for r in stage2]

    best_ep   = 16
    best_dice = 0.8676

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    fig.suptitle(
        "Stage 2 — Tong quan Qua trinh Huan luyen (Twin Y-Axis)\n"
        "UNet++ EfficientNet-B5 SCSE | 180 Epochs | V100 GPU",
        fontsize=13, fontweight="bold"
    )

    # Loss on left axis
    l1, = ax1.plot(eps_loss, vals_loss,
                   color=C_NAVY, linewidth=2.2, marker="o",
                   markersize=6, markerfacecolor="white",
                   markeredgecolor=C_NAVY, markeredgewidth=1.6,
                   label="Train Loss (Tversky)", zorder=4)

    # Dice on right axis
    l2, = ax2.plot(eps_dice, vals_dice,
                   color=C_TEAL, linewidth=2.2, marker="s",
                   markersize=6, markerfacecolor="white",
                   markeredgecolor=C_TEAL, markeredgewidth=1.6,
                   label="Val Dice Score", zorder=4)

    # Best epoch marker
    ax1.axvline(x=best_ep, color=C_ORANGE, linewidth=1.5,
                linestyle="--", alpha=0.85, zorder=3,
                label=f"Best epoch ({best_ep})")
    ax2.annotate(
        f"Best\nEp {best_ep}\nDice={best_dice:.4f}",
        xy=(best_ep, best_dice),
        xytext=(best_ep + 10, best_dice - 0.025),
        fontsize=9, color=C_ORANGE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_ORANGE,
                        lw=1.2, connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=C_ORANGE, alpha=0.95),
        zorder=6,
    )

    # SWA region
    ax1.axvspan(136, 180, alpha=0.07, color=C_TEAL, zorder=0)
    ax1.text(137, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 0.074,
             "SWA", fontsize=8, color=C_TEAL, alpha=0.75)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss (Tversky)", color=C_NAVY, fontweight="bold")
    ax2.set_ylabel("Val Dice Score", color=C_TEAL, fontweight="bold")
    ax1.tick_params(axis="y", colors=C_NAVY)
    ax2.tick_params(axis="y", colors=C_TEAL)
    ax1.set_xlim(0, 185)
    ax1.set_ylim(0.06, 0.20)
    ax2.set_ylim(0.70, 0.92)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax1.grid(True, axis="y", color=C_GRID, linewidth=0.6, alpha=0.7)

    # Combined legend
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return save_fig(fig, "stage2_overview_combined")


# ===========================================================================
# Figure 4: malignancy_overview_combined.png — multi-metric normalized
# ===========================================================================

def render_figure4(mal: list) -> tuple:
    """
    Figure 4: FPR+Malignancy — 3 metrics trên 1 plot.
    Dùng 2 y-axis: loss (left, absolute), metrics (right, absolute 0-1).
    Thêm annotation epoch 52 best.
    """
    eps      = [r["epoch"] for r in mal]
    losses   = [r["train_loss"] for r in mal]
    bal_accs = [r["val_bal_acc"] for r in mal]
    susp_f1s = [r["val_susp_f1"] for r in mal]

    best_idx     = int(np.argmax(bal_accs))
    best_ep      = eps[best_idx]
    best_bal_acc = bal_accs[best_idx]
    best_f1      = susp_f1s[best_idx]

    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax2 = ax1.twinx()

    fig.suptitle(
        "FPR + Malignancy Classifier — Tong quan Huan luyen (60 Epochs)\n"
        "DenseNet121-3D | Adam lr=1e-4 | Weighted Cross-Entropy",
        fontsize=13, fontweight="bold"
    )

    # Loss on left axis
    l1, = ax1.plot(eps, losses,
                   color=C_NAVY, linewidth=2.2, linestyle="-",
                   label="Train Loss", zorder=3)
    ax1.fill_between(eps, losses, alpha=0.07, color=C_NAVY)

    # Metrics on right axis
    l2, = ax2.plot(eps, bal_accs,
                   color=C_TEAL, linewidth=2.0, marker="o",
                   markersize=3, markerfacecolor=C_TEAL,
                   label="Val Balanced Accuracy", zorder=4)

    l3, = ax2.plot(eps, susp_f1s,
                   color=C_RUST, linewidth=2.0, marker="s",
                   markersize=3, markerfacecolor=C_RUST,
                   label="Val Suspicious F1", zorder=4)

    # Best epoch vertical line
    ax1.axvline(x=best_ep, color=C_ORANGE, linewidth=1.5,
                linestyle="--", alpha=0.85, zorder=5)

    # Annotation at best bal_acc
    ax2.annotate(
        f"Best Epoch {best_ep}\nBal Acc = {best_bal_acc:.4f}\nSusp F1 = {best_f1:.4f}",
        xy=(best_ep, best_bal_acc),
        xytext=(best_ep - 20, best_bal_acc + 0.025),
        fontsize=9, color=C_ORANGE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_ORANGE,
                        lw=1.2, connectionstyle="arc3,rad=-0.2"),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=C_ORANGE, alpha=0.95),
        zorder=6,
    )

    # Random baseline for bal_acc (5 classes = 0.2)
    ax2.axhline(y=0.2, color=C_GREY, linewidth=0.9,
                linestyle=":", alpha=0.6, zorder=2)
    ax2.text(1.5, 0.205, "Random (5 class)", fontsize=8, color=C_GREY)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=C_NAVY, fontweight="bold")
    ax2.set_ylabel("Validation Metric (0-1)", color=C_TEAL, fontweight="bold")
    ax1.tick_params(axis="y", colors=C_NAVY)
    ax2.tick_params(axis="y", colors=C_TEAL)
    ax1.set_xlim(0, 62)
    ax1.set_ylim(min(losses) - 0.05, max(losses) + 0.05)
    ax2.set_ylim(0.25, 0.85)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax1.grid(True, axis="y", color=C_GRID, linewidth=0.6, alpha=0.7)

    # Combined legend
    lines  = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.92)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return save_fig(fig, "malignancy_overview_combined")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("render_thesis_charts.py — Generating thesis figures")
    print(f"  Stage2 log : {STAGE2_LOG}")
    print(f"  Mal log    : {MAL_LOG}")
    print(f"  Output dir : {OUT_DIR}")
    print()

    # Verify input files exist
    for p in (STAGE2_LOG, MAL_LOG):
        if not p.exists():
            print(f"ERROR: Input file not found: {p}")
            sys.exit(1)

    # Parse
    print("[1/2] Parsing Stage 2 log...")
    stage2 = parse_stage2_log(STAGE2_LOG)
    print(f"      Parsed {len(stage2)} milestone records")

    print("[2/2] Parsing FPR+Malignancy log...")
    mal = parse_mal_log(MAL_LOG)
    print(f"      Parsed {len(mal)} epoch records")
    print()

    # Render
    print("=== Rendering Figure 1: stage2_loss_dice ===")
    png1, pdf1 = render_figure1(stage2)
    plt.close("all")
    print(f"  PNG: {png1}  ({png1.stat().st_size:,} bytes)")
    print(f"  PDF: {pdf1}  ({pdf1.stat().st_size:,} bytes)")

    print("=== Rendering Figure 2: fpr_malignancy_curves ===")
    png2, pdf2 = render_figure2(mal)
    plt.close("all")
    print(f"  PNG: {png2}  ({png2.stat().st_size:,} bytes)")
    print(f"  PDF: {pdf2}  ({pdf2.stat().st_size:,} bytes)")

    print("=== Rendering Figure 3: stage2_overview_combined ===")
    png3, pdf3 = render_figure3(stage2)
    plt.close("all")
    print(f"  PNG: {png3}  ({png3.stat().st_size:,} bytes)")
    print(f"  PDF: {pdf3}  ({pdf3.stat().st_size:,} bytes)")

    print("=== Rendering Figure 4: malignancy_overview_combined ===")
    png4, pdf4 = render_figure4(mal)
    plt.close("all")
    print(f"  PNG: {png4}  ({png4.stat().st_size:,} bytes)")
    print(f"  PDF: {pdf4}  ({pdf4.stat().st_size:,} bytes)")

    print()
    print("=== Summary ===")
    all_files = [png1, pdf1, png2, pdf2, png3, pdf3, png4, pdf4]
    total_bytes = sum(p.stat().st_size for p in all_files)
    print(f"  8 files generated in: {OUT_DIR}")
    print(f"  Total size: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")
    # PNG must be > 50KB; PDF vector files are typically 25-60KB (paths, not pixels)
    failed_png = [p for p in all_files if p.suffix == ".png" and p.stat().st_size < 50_000]
    failed_pdf = [p for p in all_files if p.suffix == ".pdf" and p.stat().st_size < 5_000]
    failed = failed_png + failed_pdf
    if failed:
        print(f"WARNING: {len(failed)} file(s) suspiciously small:")
        for p in failed:
            print(f"    {p}  ({p.stat().st_size:,} bytes)")
    else:
        print("  Sanity check PASSED:")
        print("    PNG files all > 50KB (raster OK)")
        print("    PDF files all > 5KB (vector OK — typical 30-60KB for matplotlib charts)")


if __name__ == "__main__":
    main()
