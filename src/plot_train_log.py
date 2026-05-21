"""Parse a stage2_full training log and plot loss + val_dice curves.

Run: python src/plot_train_log.py
Outputs:
  work/academic/stage2_train_curve.png
"""
import re
from pathlib import Path
import matplotlib.pyplot as plt

LOG = Path(__file__).resolve().parent.parent / "work/runs_exp/stage2_full/log.txt"
OUT = Path(__file__).resolve().parent.parent / "work/academic/stage2_train_curve.png"

EP_RE = re.compile(r"ep\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+val_dice=([\d.]+)")

epochs, losses, val_dices, best_eps = [], [], [], []
for line in LOG.read_text(encoding="utf-8").splitlines():
    m = EP_RE.search(line)
    if not m:
        continue
    ep = int(m.group(1))
    loss = float(m.group(3))
    val = float(m.group(4))
    epochs.append(ep)
    losses.append(loss)
    val_dices.append(val)
    if "saved best.pt" in line:
        best_eps.append(ep)

print(f"Parsed {len(epochs)} epoch samples")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(epochs, losses, "o-", color="#ff6b6b", linewidth=2, markersize=6, label="Train loss")
ax1.set_ylabel("Tversky loss", fontsize=11)
ax1.set_title("Stage2 winner — training curves (sampled checkpoints)", fontsize=12, fontweight="bold")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper right")
ax1.axvline(136, color="orange", linestyle="--", alpha=0.6, label="SWA start ep 136")
ax1.text(138, max(losses) * 0.95, "SWA start", color="orange", fontsize=9)

best_idx = val_dices.index(max(val_dices))
ax2.plot(epochs, val_dices, "o-", color="#4ecdc4", linewidth=2, markersize=6, label="Val Dice")
ax2.axhline(0.8676, color="green", linestyle=":", alpha=0.5)
ax2.text(180, 0.871, "Final best 0.8676", color="green", fontsize=9, ha="right")
ax2.scatter([epochs[best_idx]], [val_dices[best_idx]], s=200, marker="*",
            color="gold", edgecolor="black", linewidth=1.5, zorder=10,
            label=f"Best @ ep {epochs[best_idx]} = {val_dices[best_idx]:.4f}")
ax2.axvline(136, color="orange", linestyle="--", alpha=0.6)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Validation Dice score", fontsize=11)
ax2.grid(alpha=0.3)
ax2.legend(loc="lower right")
ax2.set_ylim(0.70, 0.90)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Wrote {OUT}")
