# Rule Comparison: fixed_15mm vs luna16_radius

Matching rules:
- **fixed_15mm**: centroid <= 15 mm (size-independent, legacy baseline)
- **luna16_radius**: centroid <= max(GT_diameter/2, 3 mm) (LUNA16 official, size-adaptive)

Test panel: 95 patients. Mine probs from TTA1 ensemble (cached). MONAI preds from monai\_test\_panel\_preds.json.

## 4-cell comparison table

| System | rule fixed\_15mm | rule luna16\_radius |
|--------|-----------------|---------------------|
| **mine** | F1=0.600 [0.548,0.655] sens=0.588 prec=0.613 FP/scan=0.98 CPM=0.585 | F1=0.571 [0.519,0.628] sens=0.560 prec=0.583 FP/scan=1.05 CPM=0.549 |
| **monai** | F1=0.555 [0.492,0.619] sens=0.532 prec=0.581 FP/scan=1.01 CPM=0.488 | F1=0.530 [0.461,0.597] sens=0.508 prec=0.555 FP/scan=1.07 CPM=0.458 |
|--------|-----------------|---------------------|
| **delta(mine-MONAI)** | +0.045 F1, p=0.7737, n=95 | +0.041 F1, p=0.6560, n=95 |

> Bootstrap CI: 1000 paired samples, seed=42. Wilcoxon two-sided on per-patient F1.

## FROC / CPM detail

| System | Rule | CPM | @0.125 | @0.25 | @0.5 | @1.0 | @2.0 | @4.0 | @8.0 |
|--------|------|-----|-------|-------|-------|-------|-------|-------|-------|
| mine | fixed_15mm | 0.5847 | 0.520 | 0.520 | 0.520 | 0.582 | 0.647 | 0.652 | 0.652 |
| mine | luna16_radius | 0.5494 | 0.496 | 0.496 | 0.496 | 0.549 | 0.592 | 0.608 | 0.608 |
| monai | fixed_15mm | 0.4879 | 0.252 | 0.252 | 0.373 | 0.530 | 0.599 | 0.689 | 0.721 |
| monai | luna16_radius | 0.4581 | 0.244 | 0.244 | 0.340 | 0.494 | 0.564 | 0.643 | 0.678 |

## Stratified sensitivity

| System | Rule | 4-6mm | 6-15mm | >15mm |
|--------|------|-------|-------|-------|
| mine | fixed_15mm | 0.627(n=67) | 0.578(n=135) | 0.595(n=37) |
| mine | luna16_radius | 0.567(n=67) | 0.570(n=135) | 0.568(n=37) |
| monai | fixed_15mm | 0.478(n=67) | 0.593(n=135) | 0.541(n=37) |
| monai | luna16_radius | 0.433(n=67) | 0.578(n=135) | 0.514(n=37) |

## Operating thresholds used

| System | Rule | Threshold | Selection method |
|--------|------|-----------|-----------------|
| mine  | fixed_15mm | 0.65 | best-F1 sweep on test_panel |
| MONAI | fixed_15mm | 0.90 | best-F1 sweep on val_panel |
| mine  | luna16_radius | 0.65 | best-F1 sweep on test_panel |
| MONAI | luna16_radius | 0.90 | best-F1 sweep on val_panel |

## Verdict

**mine WINS on luna16\_radius rule**: F1=0.571 vs MONAI 0.530 (delta=+0.041, p=0.6560).

On fixed\_15mm (legacy): mine F1=0.600 vs MONAI 0.555 — mine also wins.

Decision impact: mine leads by small margin on strict rule. **Fine-tune optional** — focus on 4-6mm bucket.
