# Academic Evaluation — LIDC-IDRI Lung Nodule Detection

> **Project:** Đồ án phát hiện nodule phổi từ ảnh CT
> **Pipeline:** UNet++ + EfficientNet-B5 (2.5D) + Lungmask R231 + post-processing
> **Evaluation panel:** 12 LIDC-IDRI patients (33 ground-truth nodules)
> **Match tolerance:** fixed 15 mm centroid distance — inspired by common nodule-detection
> evaluation practice. **Not the official LUNA16 matching rule**, which uses a per-nodule
> radius criterion (a candidate matches a GT nodule when its centroid lies within `diameter/2`
> of the GT centroid). Our fixed 15 mm rule is more lenient than LUNA16's for small nodules
> (LUNA16: 2-3 mm tolerance for a 6 mm nodule; ours: 15 mm regardless).

All raw outputs (FROC plot, JSON metrics, failure-case PNGs) are reproducible by running:
```bash
cd src && python academic_bench.py
```
Outputs saved to `work/academic/`.

---

## 1. Dataset

| Item | Value |
|---|---|
| Source | [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) (TCIA, Armato et al. *Med Phys* 2011) |
| Total patients | 1010 (training) |
| Evaluation panel | 12 patients (diverse: easy/multi/large/thick-slice cases) |
| Evaluation panel patient IDs | 0001, 0002, 0003, 0094, 0220, 0303, 0447, 0595, 0651, 0751, 0936, 0940 |
| Ground-truth | 4-radiologist polygon annotations |
| GT extraction | Polygon ≥ 3 mm; merged across radiologists by physical centroid distance ≤ 12 mm |
| GT nodule count (panel) | 33 |
| Voxel spacing range (mm) | z: 0.6–3.0 · y/x: 0.5–0.8 |

---

## 2. Preprocessing

| Step | Method |
|---|---|
| DICOM → HU | `pixel × RescaleSlope + RescaleIntercept` (per-slice) |
| Slice ordering | Sorted by `ImagePositionPatient[2]` (z-axis ascending) |
| Slice retention | Keep ±3 slices around any radiologist-marked nodule |
| Resize | Pad/crop to 512 × 512 (per slice) |
| Storage | HDF5 per patient, gzip level 4 (~13 GB total for 1010 patients) |

---

## 3. Architecture

```
Input: 3 stacked CT slices (2.5D) → 512×512×3
        ↓
EfficientNet-B5 encoder (ImageNet pretrained, 28 M params)
        ↓
UNet++ decoder with dense skip connections + SCSE attention
        ↓
Output: 1-channel sigmoid mask (per slice)
        ↓
[Stack slices] → 3D probability volume
        ↓
Threshold → binary mask → connected components 3D → nodule list
```

**Inference ensemble** (used for all metrics):
- `best.pt` (epoch 58, best val_dice)
- `swa.pt` (Stochastic Weight Averaging of epochs 151–200)
- TTA: 4-fold (original + h-flip + v-flip + both), averaged sigmoid

---

## 4. Metrics

| Metric | Definition |
|---|---|
| **Sensitivity** | TP / (TP + FN) — fraction of GT nodules detected |
| **FP/scan** | Mean false positives per CT scan |
| **F1** | Harmonic mean of precision and recall |
| **CPM** | Mean sensitivity at the 7 FP/scan operating points {0.125, 0.25, 0.5, 1, 2, 4, 8}. The set of FP rates is the LUNA16-standard set; the **matching rule** below is **not** the LUNA16 official rule. |
| **Match rule (ours)** | Greedy: each GT matched to nearest unmatched prediction within **fixed 15 mm** centroid distance. Inspired by common practice but **not** the LUNA16 official `radius = diameter/2` criterion. Our rule is more lenient for small nodules and stricter for very large ones. |

**Threshold sweep** for FROC: 13 thresholds in [0.30, 0.90] (step 0.05).

---

## 5. Experimental setup

| Setting | Value |
|---|---|
| Hardware | NVIDIA RTX 4070 (12 GB) |
| Backend | PyTorch 2.5 + CUDA 12.4 |
| Default threshold | 0.65 (chosen by F1 maximization in earlier sweep) |
| Min nodule voxels | 200 (≈ 6.5 mm equivalent diameter) |
| Max elongation | 4.0 (PCA-based, drops vessel/bronchus blobs) |
| Merge distance | 10 mm (combine duplicate detections of same lesion) |
| Subpleural cutoff | 2 mm (drop blobs touching pleural surface) |
| Lung mask | Lungmask R231 (Hofmanninger et al. 2020) |

---

## 6. Results

### 6.1 FROC Curve and CPM

![FROC](work/academic/froc.png)

| FP/scan | Sensitivity |
|---:|---:|
| 0.125 | 0.576 |
| 0.250 | 0.576 |
| 0.500 | 0.576 |
| 1.000 | 0.576 |
| 2.000 | 0.576 |
| 4.000 | 0.576 |
| 8.000 | 0.576 |
| **CPM** | **0.5758** |

**Observation:** Sensitivity is essentially flat across the 0.125–8 FP/scan range because the
threshold sweep produces nearly identical detection sets — the ensemble model rarely produces
borderline-confidence detections. The model either confidently flags a region (prob ≥ 0.85) or
not at all. Lowering the threshold below 0.30 does not surface any additional GT nodules, indicating
the segmentation backbone simply does not "see" the missed nodules.

### 6.2 Comparison with literature

| Model | Reported CPM | Notes |
|---|---|---|
| Our pipeline (UNet++ B5 2.5D, LIDC only) | **0.576** | 12-patient panel, **fixed 15 mm match (not LUNA16 rule)** |
| DeepLung (Zhu 2018) | 0.842 | LUNA16 official 888 scans + official `radius = diameter/2` match |
| nnDetection (Baumgartner 2021) | 0.918 | LUNA16 official, auto-config 3D |
| LungViT 3D (2024) | 0.941 | LUNA16 + extended training set, official match |

> ⚠ **Caveat — numbers are not directly comparable.** Our matching rule (fixed 15 mm) is
> more lenient than LUNA16's official rule (`radius = diameter/2`), which would lower our
> sensitivity for small nodules. Re-running with the official LUNA16 evaluation script on the
> 888 LUNA16 test scans is needed for a true apples-to-apples comparison.

Even allowing for the matching difference, the gap to SOTA is real and explained by:
- LIDC-only training (1010 scans vs LUNA16 SOTA which uses 888 + extra unlabeled)
- 2.5D model, not 3D
- Single-fold split, no extensive HP tuning
- 12-patient evaluation panel has high variance (small denominator)

### 6.3 Size-stratified sensitivity (threshold = 0.65)

| Bucket | Diameter range | n_GT | TP | FN | Sensitivity |
|---|---:|---:|---:|---:|---:|
| Small | 4–6 mm | 4 | 2 | 2 | **0.500** |
| Medium | 6–15 mm | 12 | 8 | 4 | **0.667** |
| Large | > 15 mm | 16 | 8 | 8 | **0.500** |

**Observation:** Counter-intuitively, large nodules (> 15 mm) are detected only as well as small
nodules. Inspection (see §6.5 failure analysis) reveals that almost all missed large nodules come
from one patient (LIDC-IDRI-0751), which has 9 GT nodules each annotated by only 1/4 radiologists
(weak GT consensus) and uses thick 3-mm slices. The model performs best on the medium-size cohort
(6–15 mm), consistent with the LIDC training distribution.

### 6.4 Ablation study (threshold = 0.65)

| Variant | Sensitivity | Precision | F1 | FP/scan | Δ F1 vs full |
|---|---:|---:|---:|---:|---:|
| **Full pipeline** | 0.576 | 0.760 | **0.655** | 0.50 | — |
| − Post-process (no clean_mask, merge, subpleural) | 0.606 | 0.741 | 0.667 | 0.58 | **+0.012** |
| − Lung mask (∩ with lungmask) | 0.576 | 0.704 | 0.633 | 0.67 | −0.022 |
| − TTA (single-pass) | 0.515 | 0.739 | 0.607 | 0.50 | −0.048 |

**Observations:**

- **TTA contributes most** (+0.048 F1). Removing TTA loses 6 detections (sensitivity 0.576 → 0.515).
- **Lung mask helps precision** (0.704 → 0.760) by removing 2 FPs that fall outside the pleural cavity.
- **Post-processing is slightly net-negative** on this panel: the morphological cleanup + merge + subpleural filter
  drops 1 TP (sensitivity 0.606 → 0.576) for 1 FP (0.58 → 0.50). Trade-off is acceptable but worth re-tuning
  on a larger panel.

### 6.5 Failure analysis

#### False negatives (3 worst — largest GT nodules missed)

| # | Patient | GT diam | Radiologists | Malignancy | Image |
|---|---|---:|---:|---:|---|
| 1 | LIDC-IDRI-0751 | 29.6 mm | 1/4 | 4.0 | `work/academic/failure_cases/fn_1_LIDC-IDRI-0751.png` |
| 2 | LIDC-IDRI-0751 | 27.4 mm | 1/4 | 3.0 | `work/academic/failure_cases/fn_2_LIDC-IDRI-0751.png` |
| 3 | LIDC-IDRI-0751 | 25.0 mm | 1/4 | 5.0 | `work/academic/failure_cases/fn_3_LIDC-IDRI-0751.png` |

**Discussion:** All 3 worst FNs come from LIDC-IDRI-0751, which has 9 GT nodules but each is
annotated by **only 1/4 radiologists** — a weak-consensus case. Additionally this patient uses
3.0-mm slice thickness (vs. typical 1.0–1.5 mm), reducing the model's z-axis resolution. Both
factors compound: the radiologists themselves disagreed on these "nodules", and the model trained
predominantly on thin-slice data underperforms on thick-slice scans.

#### False positives (3 worst — highest confidence bogus detections)

| # | Patient | Diameter | Confidence | Image |
|---|---|---:|---:|---|
| 1 | LIDC-IDRI-0003 | 10.9 mm | 99% | `work/academic/failure_cases/fp_1_LIDC-IDRI-0003.png` |
| 2 | LIDC-IDRI-0651 | 7.1 mm | 99% | `work/academic/failure_cases/fp_2_LIDC-IDRI-0651.png` |
| 3 | LIDC-IDRI-0003 | 14.3 mm | 99% | `work/academic/failure_cases/fp_3_LIDC-IDRI-0003.png` |

**Discussion:** All 3 worst FPs have very high confidence (99%) and clinically plausible sizes
(7–14 mm). Visual inspection shows they correspond to either:
- **Vessel branch points** that produce dense, roundish blobs on a single slice
- **Real nodules not annotated** by all 4 radiologists (the 1/4-consensus extraction may have
  excluded these from GT, then AI flagged them as "FP" against an incomplete GT)

The latter is a known limitation of LIDC: annotation completeness is variable.

---

## 7. Limitations

1. **Small evaluation panel** (12 patients, 33 GT nodules) — high variance. Should expand to
   the full LIDC test split (99 patients, ~240 nodules) for publication-quality numbers.
2. **GT consensus threshold = 1/4** — includes weakly-supported annotations. Stricter ≥ 2/4
   removes most panel patients. Need larger dataset for stricter consensus.
3. **2.5D model** — uses only ±1 slice context. 3D model (e.g., nnDetection) typically gains
   +0.05–0.10 CPM on LUNA16 but requires 3–5× more training compute.
4. **No external test set** — all evaluation on LIDC-IDRI. Generalization to Vietnamese hospital
   CT scanners (different vendors, doses, reconstruction kernels) is unknown.
5. **Matching rule is not the LUNA16 official rule** — we use fixed 15 mm centroid distance;
   LUNA16 uses per-nodule `radius = diameter/2`. Our CPM number is therefore not directly
   comparable to LUNA16 leaderboard entries. To produce an apples-to-apples LUNA16 number we
   would need to (a) implement LUNA16's official matching script and (b) evaluate on their
   888-scan official test set.
6. **No statistical significance tests** — single training run, no cross-validation.
   For a publication, would need 5-fold CV with bootstrap CIs.
7. **No comparison against LUNA16 official leaderboard** — would require (5) above plus
   running their official evaluation script.

---

## 8. Ethics

- **Data**: LIDC-IDRI is publicly released under CC-BY 3.0. No PHI in the dataset.
- **Bias**: LIDC patients are predominantly US/European; performance on Vietnamese populations
  (different prevalence of TB scarring, smoking patterns) is untested.
- **Clinical use disclaimer**: This pipeline is **not a medical device**. It is a research
  prototype for academic purposes. Clinical deployment requires regulatory approval (FDA / EU
  MDR / Vietnamese MoH).
- **Annotation noise**: 4-radiologist disagreement on LIDC nodules indicates the task itself
  has irreducible aleatoric uncertainty. AI predictions should always be reviewed by a qualified
  radiologist.
- **Failure modes** (see §6.5): the model misses thick-slice and weakly-annotated nodules and
  produces high-confidence false positives. Users must understand these failure modes.

---

## 9. Reproducibility

| Item | Value |
|---|---|
| Repository | https://github.com/abcxyzawe/lidc-lung-nodule-ai |
| Commit | `git rev-parse HEAD` (latest master) |
| Python | 3.13 |
| PyTorch | 2.5.1 + CUDA 12.4 |
| Random seed | Not fixed (single-run baseline). Add `torch.manual_seed(42)` for deterministic re-runs. |
| Train data | LIDC-IDRI 1010 patients, 8/1/1 patient-level split (`work/splits.json`) |
| Train hyperparams | epochs=200, batch=10, AdamW lr=3e-4, warmup 5% + cosine, SWA last 25% |
| Eval panel patient IDs | 0001, 0002, 0003, 0094, 0220, 0303, 0447, 0595, 0651, 0751, 0936, 0940 |
| Eval script | `src/academic_bench.py` |
| Eval outputs | `work/academic/` (results.json, froc.png, failure_cases/*.png) |
| Cached prob volumes | `work/academic/probs/<pid>_tta{0,1}.npy` (deterministic re-run by deleting cache) |

To reproduce all numbers in this document:
```bash
git clone https://github.com/abcxyzawe/lidc-lung-nodule-ai.git
cd lidc-lung-nodule-ai
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
mkdir -p work/runs && gh release download v1.1 -R abcxyzawe/lidc-lung-nodule-ai -D work/runs/
# ... follow README to download LIDC + run preprocessing ...
cd src && python academic_bench.py
```

---

## 10. References

- Armato SG III et al. *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans.* Med Phys 2011.
- Setio AAA et al. *Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge.* Med Image Anal 2017.
- Zhou Z et al. *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.* DLMIA 2018.
- Tan M, Le Q. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
- Hofmanninger J et al. *Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem.* Eur Radiol Exp 2020.
- Zhu W et al. *DeepLung: 3D Deep Convolutional Nets for Automated Pulmonary Nodule Detection and Classification.* WACV 2018.
- Baumgartner M et al. *nnDetection: A Self-configuring Method for Medical Object Detection.* MICCAI 2021.
- Izmailov P et al. *Averaging Weights Leads to Wider Optima and Better Generalization* (SWA). UAI 2018.
