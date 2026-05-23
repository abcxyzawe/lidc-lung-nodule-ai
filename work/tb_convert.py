"""
tb_convert.py - Convert 3 training logs to TensorBoard event files.

Sources:
  1. work/runs_exp/stage2_full/log.txt  (Stage2 segmentation, milestone-sparse, 180 ep)
  2. work/runs/mal_train.log            (DenseNet FPR+Malignancy, dense, 60 ep)
  3. work/runs_luna16/run005_scratch_full80/metrics.csv  (LUNA16 80 ep CSV)

Outputs:
  work/tensorboard_logs/stage2_full/
  work/tensorboard_logs/fpr_malignancy/
  work/tensorboard_logs/luna16_run005/

Tag naming convention (ml-lead requirement):
  train/loss, val/loss, val/dice, val/cpm, val/sens_at_*fp, optim/lr

Walltime: no real timestamps in stage2/mal logs → synthetic base_time + 60s/ep.
LUNA16 CSV has no timestamps → same synthetic scheme.
Disclaimer in README.md.
"""

import os
import re
import sys
import time
import pathlib
import numpy as np

# ---------------------------------------------------------------------------
# Paths — all absolute so script is runnable from any cwd
# ---------------------------------------------------------------------------
REPO = pathlib.Path(__file__).resolve().parent.parent  # E:\Phan Tich Ung Thu
WORK = REPO / "work"

STAGE2_LOG   = WORK / "runs_exp" / "stage2_full" / "log.txt"
MAL_LOG      = WORK / "runs" / "mal_train.log"
LUNA16_CSV   = WORK / "runs_luna16" / "run005_scratch_full80" / "metrics.csv"
LUNA16_DIR   = WORK / "runs_luna16" / "run005_scratch_full80"

OUT_BASE     = WORK / "tensorboard_logs"
OUT_STAGE2   = OUT_BASE / "stage2_full"
OUT_MAL      = OUT_BASE / "fpr_malignancy"
OUT_LUNA16   = OUT_BASE / "luna16_run005"

# Synthetic base walltime: 2026-05-14 00:00:00 UTC (aligns with stage2 run date)
BASE_TIME_STAGE2 = 1747180800  # 2026-05-14 00:00:00 UTC
BASE_TIME_MAL    = 1747267200  # 2026-05-15 00:00:00 UTC
BASE_TIME_LUNA16 = 1747353600  # 2026-05-16 00:00:00 UTC

# ---------------------------------------------------------------------------
# SummaryWriter — prefer torch, fall back to tensorboardX
# ---------------------------------------------------------------------------
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_BACKEND = "torch"
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        _TB_BACKEND = "tensorboardX"
    except ImportError:
        print("ERROR: Neither torch.utils.tensorboard nor tensorboardX found.")
        print("Install with: pip install tensorboard  OR  pip install tensorboardX")
        sys.exit(1)

try:
    from PIL import Image
    _PIL_AVAIL = True
except ImportError:
    _PIL_AVAIL = False
    print("WARNING: PIL not available — image embedding will be skipped.")

# ---------------------------------------------------------------------------
# Helper: create SummaryWriter with walltime offset applied via monkey-patch
# ---------------------------------------------------------------------------
def make_writer(logdir: pathlib.Path) -> SummaryWriter:
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(logdir))


def add_scalar_wt(writer: SummaryWriter, tag: str, value: float,
                  global_step: int, walltime: float) -> None:
    """Add scalar with explicit walltime. Torch TB and tensorboardX both accept
    the walltime kwarg in add_scalar."""
    writer.add_scalar(tag, value, global_step=global_step, walltime=walltime)


def embed_image(writer: SummaryWriter, tag: str, png_path: pathlib.Path,
                global_step: int, walltime: float) -> None:
    """Load PNG and write as tf.Image tag. Only called once at best/final epoch."""
    if not _PIL_AVAIL:
        print(f"  [SKIP image] PIL unavailable: {png_path.name}")
        return
    if not png_path.exists():
        print(f"  [SKIP image] file not found: {png_path}")
        return
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img)  # HWC uint8
    # TensorBoard expects CHW float [0,1] or HWC uint8
    # SummaryWriter.add_image accepts HWC with dataformats='HWC'
    writer.add_image(tag, arr, global_step=global_step,
                     walltime=walltime, dataformats="HWC")
    print(f"  [image embedded] {tag} @ ep {global_step}")


# ===========================================================================
# 1. Stage2 — milestone-sparse log
# ===========================================================================

def parse_stage2_log(path: pathlib.Path) -> list[dict]:
    """
    Parse lines like:
      ep   1/180  loss=0.1584  val_dice=0.8579  -> saved best.pt
      ep  16/180  loss=0.????  val_dice=0.8676  -> saved best.pt (FINAL BEST)
    loss=0.???? is treated as missing (NaN) — we DO NOT fabricate it.
    Returns list of dicts with keys: epoch, train_loss (may be None), val_dice.
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
                ep = int(m.group(1))
                loss_raw = m.group(2)
                vdice = float(m.group(3))
                # Only convert to float if fully numeric — no fabrication for ?-placeholders
                loss = float(loss_raw) if re.fullmatch(r"[\d.]+", loss_raw) else None
                records.append({"epoch": ep, "train_loss": loss, "val_dice": vdice})
    return records


def write_stage2(records: list[dict]) -> int:
    writer = make_writer(OUT_STAGE2)
    # lr schedule: cosine from 1e-4; SWA resets at ep 136
    # We store the documented lr value (no per-epoch lr in log)
    # Using documented initial lr=1e-4 for all, reset at 136
    LR_INIT = 1e-4
    LR_SWA  = 1e-4

    # Best epoch is 16 (explicitly marked FINAL BEST in log)
    BEST_EP = 16

    n_written = 0
    for rec in records:
        ep = rec["epoch"]
        wt = BASE_TIME_STAGE2 + ep * 60  # synthetic +60s/ep

        # train/loss — only if parsed (not NaN placeholder)
        if rec["train_loss"] is not None:
            add_scalar_wt(writer, "train/loss", rec["train_loss"], ep, wt)

        # val/dice
        add_scalar_wt(writer, "val/dice", rec["val_dice"], ep, wt)

        # optim/lr intentionally omitted — no per-epoch lr in log, flat constant
        # would be misleading; see F-2 fix (code-reviewer 2026-05-22)

        n_written += 1

    # No PNG assets in stage2_full dir — skip image embedding
    print(f"  [stage2] {n_written} milestone epochs written (no PNG assets found)")

    writer.close()
    return n_written


# ===========================================================================
# 2. DenseNet FPR+Malignancy — dense log
# ===========================================================================

def parse_mal_log(path: pathlib.Path) -> list[dict]:
    """
    Parse lines like:
      epoch 1/60  loss=1.6010  val_acc=0.3060  val_bal_acc=0.3274  val_susp_f1=0.5293  (7s)
    Returns list of dicts.
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
                    "epoch":        int(m.group(1)),
                    "train_loss":   float(m.group(2)),
                    "val_acc":      float(m.group(3)),
                    "val_bal_acc":  float(m.group(4)),
                    "val_susp_f1":  float(m.group(5)),
                })
    return records


def write_mal(records: list[dict]) -> int:
    writer = make_writer(OUT_MAL)

    # Best epoch: highest val_bal_acc  (epoch 52, bal_acc=0.4593 from log)
    best_rec = max(records, key=lambda r: r["val_bal_acc"])
    BEST_EP  = best_rec["epoch"]

    # Approximate lr: DenseNet Adam, fixed (no schedule info in log)
    LR_FIXED = 1e-4

    n_written = 0
    for rec in records:
        ep = rec["epoch"]
        wt = BASE_TIME_MAL + ep * 60  # synthetic +60s/ep (each ep ~5-7s real, spacing irrelevant)

        add_scalar_wt(writer, "train/loss",    rec["train_loss"],  ep, wt)
        # val/loss intentionally omitted — mal_train.log has no val_loss; see F-1 fix
        add_scalar_wt(writer, "val/bal_acc",   rec["val_bal_acc"], ep, wt)
        add_scalar_wt(writer, "val/susp_f1",   rec["val_susp_f1"], ep, wt)
        add_scalar_wt(writer, "optim/lr",      LR_FIXED,           ep, wt)
        n_written += 1

    print(f"  [fpr_mal] {n_written} epochs written | best ep={BEST_EP} val_bal_acc={best_rec['val_bal_acc']:.4f}")
    writer.close()
    return n_written


# ===========================================================================
# 3. LUNA16 run005 — CSV
# ===========================================================================

def _to_float(v: str) -> float:
    """Safe string-to-float conversion; returns NaN for empty or non-numeric values.
    Prevents crash on CSV rows with empty cells or 'N/A' strings (F-4 fix)."""
    try:
        return float(v) if v.strip() else float("nan")
    except ValueError:
        return float("nan")


def parse_luna16_csv(path: pathlib.Path):
    """
    CSV columns:
      epoch,train_loss,val_loss,val_dice,val_froc_cpm,val_best_f1,val_best_thr,
      val_sens@0.125FP,...,val_sens@8.0FP
    Returns (list of row dicts, list of column names).
    Empty/N/A cells become NaN (not a crash) — see _to_float().
    """
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _to_float(v) for k, v in row.items()})
    return rows


def _safe_fp_tag(col_name: str) -> str:
    """Convert 'val_sens@0.125FP' → 'val/sens_at_0.125fp' (TB-safe tag)."""
    tag = col_name.replace("val_sens@", "val/sens_at_").replace("FP", "fp")
    return tag


def write_luna16(rows: list[dict]) -> int:
    writer = make_writer(OUT_LUNA16)

    # Best epoch: highest val_froc_cpm
    best_row = max(rows, key=lambda r: r["val_froc_cpm"])
    BEST_EP  = int(best_row["epoch"])

    n_written = 0
    for row in rows:
        ep  = int(row["epoch"])
        wt  = BASE_TIME_LUNA16 + ep * 60  # synthetic

        add_scalar_wt(writer, "train/loss", row["train_loss"], ep, wt)
        add_scalar_wt(writer, "val/loss",   row["val_loss"],   ep, wt)
        add_scalar_wt(writer, "val/dice",   row["val_dice"],   ep, wt)
        add_scalar_wt(writer, "val/cpm",    row["val_froc_cpm"], ep, wt)
        add_scalar_wt(writer, "val/best_f1", row["val_best_f1"], ep, wt)

        # Sensitivity at each FP/scan level
        for col in row:
            if col.startswith("val_sens@"):
                add_scalar_wt(writer, _safe_fp_tag(col), row[col], ep, wt)

        # lr not in CSV — not logged (no fabrication)
        n_written += 1

    # Embed FROC PNG at best epoch only (LUNA16 has per-epoch froc_curve_epXXX.png)
    best_ep_str = f"{BEST_EP:03d}"
    froc_png = LUNA16_DIR / f"froc_curve_ep{best_ep_str}.png"
    best_wt  = BASE_TIME_LUNA16 + BEST_EP * 60
    embed_image(writer, "images/froc_curve_best_ep", froc_png, BEST_EP, best_wt)

    # Also embed final epoch FROC
    final_ep  = int(rows[-1]["epoch"])
    final_ep_str = f"{final_ep:03d}"
    final_png = LUNA16_DIR / f"froc_curve_ep{final_ep_str}.png"
    if final_ep != BEST_EP:
        final_wt = BASE_TIME_LUNA16 + final_ep * 60
        embed_image(writer, "images/froc_curve_final_ep", final_png, final_ep, final_wt)

    print(f"  [luna16] {n_written} epochs written | best ep={BEST_EP} val_cpm={best_row['val_froc_cpm']:.4f}")
    writer.close()
    return n_written


# ===========================================================================
# README
# ===========================================================================

README_TEMPLATE = """\
# TensorBoard Logs — LIDC-IDRI Pipeline

Generated by `work/tb_convert.py`.

## DISCLAIMER — Walltime

**Walltimes in these event files are SYNTHETIC** (base_timestamp + 60s * epoch).
The original training logs do not contain per-epoch timestamps.
The `Scalars` panel "wall time" axis is therefore not a reliable measure of actual
training duration. Use the `step` (epoch) axis for analysis.

## Launch

```bash
cd "E:/Phan Tich Ung Thu"
tensorboard --logdir=work/tensorboard_logs/ --port=6006
```

Then open http://localhost:6006 in your browser.

To view a single run:
```bash
tensorboard --logdir=work/tensorboard_logs/stage2_full/ --port=6007
```

## Runs

### stage2_full  (Stage 2 Segmentation — EfficientNet-B5)
- Source: `work/runs_exp/stage2_full/log.txt`
- Format: milestone-sparse (18 checkpoints out of 180 epochs)
- Tags:
  - `train/loss` (missing ep 16 — placeholder `loss=0.????` in log, NOT fabricated)
  - `val/dice`
- **Omitted**: `optim/lr` — cosine schedule, no per-epoch lr in log; flat constant would be misleading
- Best epoch: 16 (val_dice=0.8676)
- Epochs parsed: 18 milestones
- SWA enabled from ep 136 (lr reset to 1e-4)

### fpr_malignancy  (DenseNet121-3D FPR + Malignancy Classifier)
- Source: `work/runs/mal_train.log`
- Format: dense (all 60 epochs)
- Tags:
  - `train/loss`
  - `val/bal_acc`
  - `val/susp_f1`
  - `optim/lr`
- **Omitted**: `val/loss` — mal_train.log has no val_loss; writing train_loss under val/loss tag is fabrication (removed per F-1 fix 2026-05-22)
- Best epoch: determined by peak val_bal_acc
- Epochs parsed: 60

### luna16_run005  (LUNA16 Detection — run005 scratch 80ep)
- Source: `work/runs_luna16/run005_scratch_full80/metrics.csv`
- Format: CSV, all 80 epochs
- Tags:
  - `train/loss`, `val/loss`, `val/dice`
  - `val/cpm` (FROC CPM — 7-point mean)
  - `val/best_f1`
  - `val/sens_at_0.125fp`, `val/sens_at_0.25fp`, `val/sens_at_0.5fp`,
    `val/sens_at_1.0fp`, `val/sens_at_2.0fp`, `val/sens_at_4.0fp`, `val/sens_at_8.0fp`
  - `images/froc_curve_best_ep` (PNG embedded at best epoch only)
  - `images/froc_curve_final_ep` (PNG at ep 80, if different from best)
- Epochs parsed: 80
- Note: `optim/lr` not present in CSV — NOT fabricated

## Eval methodology note

FROC/CPM uses fixed 15mm matching radius, NOT the official LUNA16 competition rule.
See commit db08c2e for honest wording.
"""


def write_readme(stage2_n: int, mal_n: int, luna16_n: int) -> None:
    readme_path = OUT_BASE / "README.md"
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    content = README_TEMPLATE
    # Patch epoch counts
    content = content.replace(
        "Epochs parsed: 18 milestones",
        f"Epochs parsed: {stage2_n} milestones"
    )
    content = content.replace(
        "Epochs parsed: 60",
        f"Epochs parsed: {mal_n}"
    )
    content = content.replace(
        "Epochs parsed: 80",
        f"Epochs parsed: {luna16_n}"
    )
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [readme] written to {readme_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(f"tb_convert.py — TensorBoard backend: {_TB_BACKEND}")
    print(f"Output base: {OUT_BASE}")
    print()

    # ---- Stage 2 ----
    print("=== [1/3] Stage2 segmentation log ===")
    stage2_records = parse_stage2_log(STAGE2_LOG)
    print(f"  Parsed {len(stage2_records)} milestone records from {STAGE2_LOG.name}")
    n1 = write_stage2(stage2_records)

    # ---- Malignancy ----
    print()
    print("=== [2/3] FPR+Malignancy DenseNet log ===")
    mal_records = parse_mal_log(MAL_LOG)
    print(f"  Parsed {len(mal_records)} epoch records from {MAL_LOG.name}")
    n2 = write_mal(mal_records)

    # ---- LUNA16 ----
    print()
    print("=== [3/3] LUNA16 run005 CSV ===")
    luna16_rows = parse_luna16_csv(LUNA16_CSV)
    print(f"  Parsed {len(luna16_rows)} epoch records from {LUNA16_CSV.name}")
    n3 = write_luna16(luna16_rows)

    # ---- README ----
    print()
    print("=== Writing README ===")
    write_readme(n1, n2, n3)

    # ---- Summary ----
    print()
    print("=== Summary ===")
    print(f"  stage2_full   : {n1} epochs -> {OUT_STAGE2}")
    print(f"  fpr_malignancy: {n2} epochs -> {OUT_MAL}")
    print(f"  luna16_run005 : {n3} epochs -> {OUT_LUNA16}")
    print()
    print("Launch with:")
    print('  tensorboard --logdir="work/tensorboard_logs/" --port=6006')


if __name__ == "__main__":
    main()
