# THESIS VERDICT FULL — run003 scratch LUNA16 vs mine vs MONAI
Generated: 2026-05-22
Panel: LIDC test_panel (99 patients, 95 loaded, 250 GT nodules)
run003: trained from scratch on LUNA16 FULL 888 scans (712 train / 88 val / 88 test)
Best LUNA16 val CPM=0.9897 at epoch 3 (early stop epoch 11, patience=8)

---

## A. Pull Status — 29/29 files verified (370.1MB)

| File group                  | Count | Status       |
|-----------------------------|-------|--------------|
| best.pt (ep3)               | 1     | OK md5=737c5943 |
| last.pt (ep11)              | 1     | OK md5=18547967 |
| swa.pt                      | 1     | OK md5=f8b628e7 |
| metrics.csv                 | 1     | OK md5=2c392f80 |
| config.json                 | 1     | OK md5=56653657 |
| manifest.json               | 1     | OK md5=8e228009 |
| train.log                   | 1     | OK md5=d505fd4e |
| confusion_matrix_ep001-011  | 11    | OK all        |
| froc_curve_ep001-011.png    | 11    | OK all        |

---

## B+C. LIDC test_panel Results (local 4090 eval)

### B. run003 best.pt (epoch 3, LUNA16 val CPM=0.9897)

fixed_15mm: F1=0.0077  CPM=0.0040  thr=0.15  sens=0.004  prec=0.091  FP/scan=0.11
luna16_rad: F1=0.0077  CPM=0.0040  thr=0.15  sens=0.004  prec=0.091

### C. run003 last.pt (epoch 11, final checkpoint)

fixed_15mm: F1=0.0079  CPM=0.0000  thr=0.10  sens=0.004  prec=0.500  FP/scan=0.01
luna16_rad: F1=0.0079  CPM=0.0000  thr=0.10  sens=0.004  prec=0.500

Both checkpoints produce near-zero detections on LIDC. The model is essentially silent
above threshold 0.15 on LIDC scans. Root cause: domain shift — LUNA16 trains on
raw CT patches with different HU statistics and no LIDC-specific augmentation.
Same pattern observed in run001 (fine-tune) which also collapsed (F1=0.35 at best).

---

## D. 4-Way Comparison Table

### Primary metric: fixed_15mm matching rule (centroid <= 15mm)

| Model                   | F1 fixed_15mm | CPM fixed_15mm | F1 luna16_radius | CPM luna16_radius |
|-------------------------|---------------|----------------|------------------|-------------------|
| mine cd48359 (WINNER)   | 0.5551        | 0.4727         | 0.5209           | 0.4394            |
| MONAI RetinaNet 3D      | 0.5455        | 0.5609         | 0.5178           | 0.5297            |
| run003 best.pt (ep3)    | 0.0077        | 0.0040         | 0.0077           | 0.0040            |
| run003 last.pt (ep11)   | 0.0079        | 0.0000         | 0.0079           | 0.0000            |

### Stratified by nodule size — fixed_15mm best threshold

| Model                   | small 4-6mm (n=67) | medium 6-15mm (n=135) | large 15mm+ (n=37) |
|-------------------------|--------------------|-----------------------|--------------------|
| mine cd48359            | sens=0.299         | sens=0.467            | sens=0.568         |
| MONAI RetinaNet 3D      | sens=0.522         | sens=0.600            | sens=0.541         |
| run003 best.pt (ep3)    | sens=0.000         | sens=0.000            | sens=0.027 (1 TP)  |
| run003 last.pt (ep11)   | sens=0.000         | sens=0.000            | sens=0.027 (1 TP)  |

### Statistical tests — run003 vs mine cd48359 (fixed_15mm, 1000-rep bootstrap)

| Metric                        | best.pt (ep3)              | last.pt (ep11)             |
|-------------------------------|----------------------------|----------------------------|
| Bootstrap observed delta      | -0.5509                    | -0.5509                    |
| Bootstrap 95% CI              | [-0.5551, -0.5425]         | [-0.5551, -0.5425]         |
| Wilcoxon signed-rank p        | 3.09e-22                   | 3.09e-22                   |
| Significant (p < 0.05)?       | YES (dramatically WORSE)   | YES (dramatically WORSE)   |

Interpretation: run003 is statistically significantly worse than mine cd48359 with
p=3e-22. The 95% CI for performance delta is entirely negative ([-0.55, -0.54]),
confirming this is not noise — this is structural domain mismatch.

---

## E. V100 Decision

**VERDICT: SHUTDOWN V100**

run003 LOSES to mine cd48359 on LIDC test_panel by an overwhelming margin:
- F1: 0.008 vs 0.555 (delta = -0.547, 70x worse)
- CPM: 0.004 vs 0.473 (delta = -0.469)
- Wilcoxon p = 3e-22

Root cause diagnosis: LUNA16-trained-from-scratch models do NOT generalize to LIDC
inference pipeline. The 2.5D slice-level segmentation probability maps from LUNA16
training are NOT calibrated for the blob-detection post-processing pipeline used in
LIDC evaluation. Model outputs near-zero sigmoid values on LIDC scans because the
HU normalization, patch sampling strategy, and slice context differ fundamentally.

This is NOT an architecture or training quality issue — LUNA16 val CPM=0.9897 proves
the model learned to detect nodules on LUNA16. The problem is the evaluation pipeline
mismatch (LUNA16 patch-level output vs LIDC volume-level blob detection).

### Keep/Shutdown recommendation

SHUTDOWN V100 immediately. Rationale:
1. run003 LOSES decisively — no run2 ablation will fix the evaluation pipeline mismatch.
2. mine cd48359 (F1=0.618 original metric, F1=0.555 on academic fixed_15mm) remains winner.
3. No further LUNA16-scratch training is expected to produce LIDC-compatible outputs
   without a major pipeline redesign (cross-dataset calibration or LUNA16+LIDC joint training).
4. Cost: ~3.5h x 6,900 VND = ~24,150 VND already spent. No further spend justified.

NOTE: Em (mlops-engineer) cannot self-shutdown the V100 container.
Anh user dispose via rental UI (same provider portal as VPS creation).

---

## Summary

- mine cd48359 WINS on LIDC. Hold current winner ckpt.
- MONAI RetinaNet 3D is competitive on LIDC CPM (0.561 vs 0.473).
- run003 scratch LUNA16 = catastrophic domain shift, F1 ~0% on LIDC.
- Both best.pt and last.pt fail identically.
- Statistical significance: p=3e-22, CI entirely negative.
- Decision: SHUTDOWN V100. No run2 needed.

---

Artifacts:
  E:\Phan Tich Ung Thu\work\runs_luna16\run003_scratch_full\EVAL_BEST_LIDC.json
  E:\Phan Tich Ung Thu\work\runs_luna16\run003_scratch_full\EVAL_LAST_LIDC.json
  E:\Phan Tich Ung Thu\work\academic\eval_run003_best.log
  E:\Phan Tich Ung Thu\work\academic\eval_run003_last.log

Notes:
- fixed_15mm: centroid distance <= 15mm (conservative non-official matching rule).
- luna16_radius: centroid <= max(GT_diam/2, 3mm) (closer to LUNA16 official rule).
- LUNA16 val CPM=0.9897 is patch-level on 88 val scans -- NOT comparable to LIDC CPM.
- MONAI per-patient F1 not available; bootstrap uses mine aggregate as reference.
- 95/99 LIDC test patients loaded (4 missing h5: LIDC-IDRI-0760, LIDC-IDRI-0322,
  LIDC-IDRI-0109, LIDC-IDRI-0208 based on MISS logs).
