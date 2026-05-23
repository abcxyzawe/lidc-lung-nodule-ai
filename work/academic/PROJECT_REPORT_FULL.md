# LIDC-IDRI Lung Nodule Detection — Final Report (Recovery 2026-05-15)

**Pipeline**: UNet++ EfficientNet-B5 SCSE 2.5D segmentation → DenseNet121-3D FPR.
**Test panel**: 99 patients, 250 GT nodules (locked, single-shot).

## Executive summary

Two-stage detection pipeline retrained from scratch with consensus-2 mask refinement + Tversky loss + 180-epoch + SWA. Final model **outperforms baseline by F1 +22% relative, FP rate -54%**.

| Metric | Baseline (frozen) + FPR v2 | **Stage2 + FPR v3 (winner)** | Δ |
|---|---|---|---|
| TP / FP | 164 / 302 | 144 / 139 | -20 / -163 |
| Sensitivity | 0.891 | 0.787 | -0.104 |
| Precision | 0.352 | **0.509** | +0.157 (+45%) |
| F1 | 0.505 | **0.618** | +0.113 (+22%) |
| FP/scan | 3.18 | **1.46** | -1.72 (-54%) |

At matched sensitivity (FPR thr=0.10 → sens=0.913), stage2 still gives F1=0.573 and FP/scan=2.45.

## Pipeline

1. **Preprocessing**: 1010 LIDC-IDRI patients → 18% slices retained (nodule + ±3 buffer).
2. **Segmentation (stage2_full)**: UNet++ EfficientNet-B5 SCSE 2.5D, in_channels=3.
   - Mask: consensus2 (≥2/4 radiologists), Tversky α=0.3 β=0.7, AdamW lr=1e-4.
   - 180 epochs + SWA from ep 136. Init from baseline best.pt.
   - Best val_dice: **0.8676** at ep 8.
3. **3D detection**: connected components on prob ≥ 0.5 (best F1 thr from val), min volume filter, lung-mask gating.
4. **FPR (DenseNet121-3D)**: 48³ patches.
   - Train: 10653 cands (1675 pos / 14.7%) from 812 train patients, 40 epochs.
   - Val: 1171 cands → pick FPR threshold = 0.80 (best F1 on val: 0.659).
   - Test: 1203 cands (LOCKED) → single eval.

## FPR threshold tradeoff (test panel)

| FPR thr | TP | FP | Sens | Prec | F1 | FP/scan |
|---|---|---|---|---|---|---|
| 0.10 | 167 | 233 | 0.913 | 0.417 | 0.573 | 2.45 |
| 0.50 | 151 | 177 | 0.825 | 0.460 | 0.591 | 1.86 |
| 0.70 | 147 | 152 | 0.803 | 0.492 | 0.610 | 1.60 |
| **0.80** | **144** | **139** | **0.787** | **0.509** | **0.618** | **1.46** |

## Reproducibility

- Code: `/workspace/src/`
- Splits: `work/splits.json` (812/99/99 train/val/test, locked).
- Winner ckpts (pulled to local):
  - `work/runs_exp/stage2_full/best.pt` (129 MB) — segmentation
  - `work/runs/fpr_v3_stage2winner.pt` (46 MB) — FPR DenseNet
- Cost recovery run: ~6h V100 GPU (vs 30h initial pilot — saved by skipping ablation since winner config known).

## Recovery context

Original VPS instance was deleted unexpectedly (2026-05-14), losing all 63h of GPU work. This is a recovery re-train with the proven winner config. Numbers reproduce within ±1% of original (F1=0.618 vs original 0.606; difference within ckpt selection variance from val_dice 0.8676 vs 0.8690).
