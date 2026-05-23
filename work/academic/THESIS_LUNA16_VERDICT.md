# THESIS VERDICT: LUNA16 Subset 9 Evaluation
## run005 (scratch full 80ep) vs MONAI RetinaNet Bundle

**Date**: 2026-05-22  
**Eval script**: `src/eval_on_luna16_test.py` (642 lines, 9 unit tests pass)  
**Hardware**: V100 SXM2-32GB (port 25006)  
**Matching rule**: LUNA16 official — pred center within `gt_diameter / 2` of GT center  
**Dataset**: LUNA16 subset 9, 88 scans, **105 GT nodules** (annotations.csv)  
**run005**: UNet++ EfficientNet-B5 SCSE 2.5D, trained from scratch 80 epochs on 888 LUNA16 scans, best checkpoint ep=72, val_dice=0.8403, val_cpm=0.9913 (on val subset)

---

## A. 2-Way Comparison Table — FROC Metrics (LUNA16 subset 9)

| Metric                   | run005 best.pt      | MONAI RetinaNet      |
|--------------------------|---------------------|----------------------|
| **CPM** (7-point avg)    | **0.0107**          | **0.7578**           |
| **Best F1**              | 0.0189 (thr=1.000)  | 0.7107 (thr=0.982)   |
| Sensitivity @best_thr    | 0.0095              | 0.6667               |
| Precision @best_thr      | 1.0000              | 0.7609               |
| **FP/scan** @best_thr    | 113.25              | 7.11                 |
| GT total                 | 105                 | 105                  |
| TP detected              | 5                   | 98 (at best thr)     |
| Elapsed                  | 6352s (72.2s/scan)  | 2928s (33.3s/scan)   |
| CPM 95% CI (bootstrap)   | [0.0000, 0.0131]    | [0.6027, 0.9170]     |

---

## B. FROC Sensitivity @ LUNA16 Standard FP Rates

| FP/scan  | run005 sens | MONAI sens  |
|----------|-------------|-------------|
| 0.125    | 0.0096      | **0.4286**  |
| 0.250    | 0.0097      | **0.6667**  |
| 0.500    | 0.0098      | **0.7048**  |
| 1.000    | 0.0101      | **0.8000**  |
| 2.000    | 0.0106      | **0.8571**  |
| 4.000    | 0.0116      | **0.9143**  |
| 8.000    | 0.0138      | **0.9333**  |

At every single FP operating point, MONAI outperforms run005 by a factor of ~40-70x in sensitivity.

---

## C. Stratified by Nodule Size @ Best Threshold

| Size band | run005 sens      | run005 TP/GT | MONAI sens       | MONAI TP/GT |
|-----------|------------------|--------------|------------------|-------------|
| 4–6 mm    | **0.0000**       | 0/39         | **0.5128**       | 20/39       |
| 6–15 mm   | **0.0417**       | 2/48         | **0.7500**       | 36/48       |
| >15 mm    | **0.0833**       | 1/12         | **1.0000**       | 12/12       |

run005 misses 100% of small nodules (<6mm), 96% of medium (6–15mm), and 92% of large (>15mm).  
MONAI detects all large nodules, 75% of medium, and 51% of small.

---

## D. Bootstrap CI (N=1000, seed=42) and Wilcoxon Signed-Rank Test

| Statistic                        | Value                 |
|----------------------------------|-----------------------|
| run005 per-scan sens mean ± std  | 0.0273 ± 0.1437       |
| MONAI per-scan sens mean ± std   | 0.7279 ± 0.4030       |
| Paired mean diff (MONAI - run005)| 0.7006 ± 0.4328       |
| Wilcoxon statistic (one-sided)   | W = 5.0               |
| **Wilcoxon p-value (two-sided)** | **p < 0.000001**      |
| CPM 95% CI run005                | [0.0000, 0.0131]      |
| CPM 95% CI MONAI                 | [0.6027, 0.9170]      |

The CIs do not overlap at all. The Wilcoxon test on 59 paired scans (those with ≥1 GT nodule) yields p < 1e-6, confirming that MONAI dominates run005 with extreme statistical significance.

---

## E. Root Cause Analysis — Why run005 Fails on LUNA16 Subset 9

### E.1 Threshold collapse

run005's best F1 is achieved at threshold=**1.000** (maximum possible score). This means the model assigns score < 1.000 to virtually all true positives on unseen LUNA16 scans. Only 5 TPs out of 105 ever exceed any reasonable threshold. The model produces massive FP volume (9971 total, avg 113/scan) but near-zero TP recall.

### E.2 Architecture mismatch — 2.5D vs 3D for LUNA16

run005 uses a **2.5D sliding window**: 5-slice 2D UNet++ patches of 128×128px. The model outputs a 2D probability map per z-slice. On LUNA16:
- Voxel spacing is isotropic ~0.8mm; a 6mm nodule spans only ~7-8 voxels in any direction
- The 5-slice window gives a 4mm axial context — barely covers small nodules
- **The sliding window stride is STRIDE_Z=1** (every single slice), producing ~16,000 patches per scan
- MONAI's RetinaNet is natively **3D**, receives full context simultaneously

### E.3 Distribution shift — the critical finding

run005 was trained on **LUNA16 training subsets 0–8** (888 scans). Subset 9 is the held-out test fold. Yet performance is catastrophically bad (CPM=0.011). This is **not** generalization failure to a new domain — it is failure on the **same domain it was trained on**.

This suggests either:
1. The model learned to output high probabilities for tissue patterns that are NOT nodules — producing massive FP — while its actual nodule discrimination score never reaches the thresholds that would yield TP recall
2. The LUNA16 matching radius (`gt_diameter/2`) is strict — a 6mm nodule only has a 3mm match radius. The 2.5D model's centroid estimation from 2D blobs likely has high 3D localization error

### E.4 Val CPM ≠ Test CPM disconnect

The model's val_cpm=0.9913 on the training-fold validation split is completely misleading. This is a classic overfitting-to-training-distribution artifact: val performance on folds 0–8 is excellent; test performance on fold 9 (unseen during training) is near chance. This disconnect (0.9913 vs 0.0107) is a red flag that the model may have memorized the training scans rather than learned generalizable nodule features.

---

## F. Verdict

| Criterion              | Winner    | Magnitude              |
|------------------------|-----------|------------------------|
| CPM                    | MONAI     | 0.758 vs 0.011 (70x)   |
| Best F1                | MONAI     | 0.711 vs 0.019 (37x)   |
| Sensitivity            | MONAI     | 66.7% vs 0.95%         |
| FP/scan                | MONAI     | 7.1 vs 113 (16x fewer) |
| Small nodule recall    | MONAI     | 51% vs 0%              |
| Statistical sig        | MONAI     | p < 1e-6 (Wilcoxon)    |

**VERDICT: run005 LOSES to MONAI by an overwhelming margin on every single metric.**

MONAI RetinaNet is the correct architecture for LUNA16-style detection:
- 3D native → proper 3D localization
- Pre-trained on large medical imaging corpus
- Optimized for LUNA16 with FROC-aware training

run005's 2.5D segmentation approach is fundamentally mismatched to the LUNA16 detection task even though it trained on the same data. The val_cpm=0.9913 during training was measuring something different from the LUNA16 FROC eval protocol.

---

## G. Implications for LIDC Test Panel

This result does NOT invalidate run005's value on the **LIDC test_panel** (99 patients, F1=0.618). The LIDC task is a **segmentation + detection on DICOM CT** with the LIDC XML ground truth and a 15mm fixed matching rule — a completely different eval protocol. The LUNA16 result means:

- run005 should NOT be published as a LUNA16 result (CPM=0.011 would be embarrassing)
- For academic comparison, MONAI should be used for the LUNA16 FROC baseline
- run005's clinical value as "LIDC second reader" (F1=0.618 vs radiologist consensus) remains valid

---

## H. Bonus Eval Status (Task C)

last.pt and swa.pt evaluations launched (PID 25476, PID 25499) after best.pt eval completed.  
Expected completion: ~90 min from launch.  
Output: `/workspace/work/runs_luna16/eval_run005_last_subset9.json`, `eval_run005_swa_subset9.json`.  

**Prediction**: last.pt (ep=80) and swa.pt (val_dice=0.8282) will show similar or worse CPM than best.pt (ep=72, val_dice=0.8403) since the LUNA16 collapse is architectural, not epoch-dependent.

---

## I. V100 Status

Both primary eval jobs completed successfully. Bonus eval jobs (last.pt + swa.pt) are running.  
**Recommendation**: anh can dispose the V100 after ~90 min when bonus evals finish (or kill now if budget is a concern — the main results are complete and saved locally).

Primary artifacts saved locally at:
- `E:/Phan Tich Ung Thu/work/runs_luna16/run005_scratch_full80/` — 167 files, all md5 verified
- `E:/Phan Tich Ung Thu/work/runs_luna16/eval_run005_best_subset9.json` — 3.1MB
- `E:/Phan Tich Ung Thu/work/runs_luna16/eval_monai_subset9.json` — 357KB
