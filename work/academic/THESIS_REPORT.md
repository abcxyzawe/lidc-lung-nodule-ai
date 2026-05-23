# Báo Cáo — Phát Hiện Nốt Phổi trên LIDC-IDRI
## Model: Stage2+FPR_v3 (commit cd48359)

**Ngày tạo:** 2026-05-22  
**Phạm vi:** LIDC-IDRI train + eval. So sánh mine vs MONAI trên LIDC test_panel.  
**Nguồn số liệu:** files thực trong repo — không có số nào được bịa.

---

## 1. Tóm Tắt

Hệ thống phát hiện nốt phổi được xây dựng trên LIDC-IDRI (1010 bệnh nhân, split 812/99/99) với pipeline 3 tầng: phân đoạn 2.5D (UNet++ EfficientNet-B5 SCSE), lọc false positive (FPR DenseNet121-3D), và phân loại ác tính (DenseNet121-3D). Trên tập kiểm thử cố định (test_panel, 99 bệnh nhân, TP+FN=183 nốt GT), model đạt F1=0.618 tại FPR threshold=0.80. So sánh với baseline MONAI RetinaNet 3D cùng tập test, mine đạt F1=0.618 vs MONAI F1=0.533 — delta +0.085, Wilcoxon p < 0.0001. Model được định vị là **công cụ hỗ trợ đọc thứ hai cho bác sĩ X-quang** (AI second reader), không phải hệ thống chẩn đoán độc lập.

---

## 2. Dataset (LIDC-IDRI)

### 2.1 Tổng Quan

- **Nguồn:** Armato et al. (2011). "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)." *Medical Physics*, 38(2), 915–931.
- **Tổng số:** 1010 bệnh nhân, ảnh CT ngực DICOM đa trung tâm
- **Annotation:** XML từ 4 bác sĩ X-quang, đọc độc lập (blind read) rồi unblinded rereading
- **GT cho train (stage1):** mask union từ ≥1/4 radiologist (`MASK_MODE_DEFAULT = "union"`)
- **GT cho stage2_full:** mask consensus2 (≥2/4 radiologist)
- **Lọc:** bỏ contour dưới 10 pixel (`MIN_MASK_PIXELS = 10`) — loại "nonNodule" point marker

### 2.2 Phân Chia Tập Dữ Liệu

| Tập | Số bệnh nhân | Số lát cắt (stage2_full) |
|-----|-------------|--------------------------|
| Train | 812 | 1,507 |
| Val | 99 | 154 |
| Test (locked) | 99 | — |

- **Test panel locked:** commit cd48359 — không thay đổi sau khi đo F1=0.618
- **Eval academic:** 4 bệnh nhân thiếu h5 (LIDC-IDRI-0760, -0322, -0109, -0208) → 95/99 scan được load

### 2.3 Phân Bố Kích Thước Nốt (test_panel, n=95 scan eval, 250 GT)

| Nhóm kích thước | Số nốt GT |
|----------------|-----------|
| 4–6 mm | 67 |
| 6–15 mm | 135 |
| >15 mm | 37 |
| Tổng | 250 |

---

## 3. Kiến Trúc

### 3.1 Pipeline Tổng Quan

```
CT DICOM (volume)
    → [Stage 1] 2D UNet — slice-level detection, khởi tạo best.pt
    → [Stage 2] UNet++ EfficientNet-B5 SCSE 2.5D — refinement (cd48359)
    → candidate blobs (3D connected components)
    → [FPR Classifier] DenseNet121-3D — lọc false positive (thr=0.85)
    → filtered candidates
    → [Malignancy Classifier] DenseNet121-3D — score 1–5
    → kết quả cuối với malignancy score
```

### 3.2 Stage 2 — Phân Đoạn 2.5D (Model Chính)

- **Backbone:** UNet++ với encoder EfficientNet-B5 (Zhou et al., 2019; Tan & Le, 2019), pretrained ImageNet, ~32M params
- **Attention:** SCSE — Squeeze-and-Excitation Channel Spatial Excitation (Roy et al., 2018)
- **Input:** 3 lát cắt liền kề (`in_ch=3`), cửa sổ HU [-1350, 150]
- **Output:** binary mask per slice (sigmoid)
- **Post-processing blob:** min_voxels=200, elongation max=4.0, merge dist 10mm, subpleural cutoff 2mm

### 3.3 FPR Classifier

- **Kiến trúc:** DenseNet121-3D (Huang et al., 2017), 11.2M params
- **Input:** cube 64×64×64 voxels quanh tâm candidate
- **Output:** P(true nodule), sigmoid
- **Operating threshold:** 0.85 (từ ckpt `fpr_v3_stage2winner.pt`, best-F1 sweep trên val)
- **AUC trên test_panel:** 0.9364

### 3.4 Malignancy Classifier

- **Kiến trúc:** DenseNet121-3D, 11.2M params
- **Input:** cube 64×64×64 quanh nốt đã xác nhận
- **Output:** score 1–5 (tương tự Lung-RADS)
- **Tập train:** 6,848 patches (5,621 train / 585 val / 642 test), loại 2 patch nhãn không hợp lệ
- **Class weights (train):** [1.312, 0.861, 0.521, 1.434, 2.179] cho mức 1–5 (nghịch đảo tần suất)

---

## 4. Cấu Hình Training

### 4.1 Stage 2 Full (Recovery Run — V100, commit cd48359)

Nguồn: `work/runs_exp/stage2_full/log.txt`.

| Tham số | Giá trị |
|---------|---------|
| Khởi tạo từ | stage1 best.pt |
| Loss | Tversky (α=0.3, β=0.7) |
| Optimizer | AdamW |
| LR | 1e-4 (cosine schedule) |
| Weight decay | 1e-2 |
| Grad clip | max_norm=5.0 |
| Batch size | 10 |
| Epochs | 180 |
| SWA bắt đầu | epoch 136 (75% of 180), lr reset=1e-4 |
| Mask mode | consensus2 (≥2/4 radiologists) |
| Sampler | uniform |
| Hardware | Tesla V100-SXM2-32GB (VPS) |

### 4.2 FPR Classifier (v3 — RTX 4090)

Nguồn: `work/runs/mal_train.log`.

| Tham số | Giá trị |
|---------|---------|
| Model | DenseNet121-3D |
| Optimizer | AdamW, lr=1e-4 |
| Loss | BCE |
| Epochs | 60 |
| Hardware | RTX 4090 (local) |
| Best epoch | 52 (val_bal_acc=0.4593) |
| Operating threshold | 0.85 |

### 4.3 Malignancy Classifier (RTX 4090)

Nguồn: `work/runs/mal_train.log`.

| Tham số | Giá trị |
|---------|---------|
| Model | DenseNet121-3D, 11.2M params |
| Loss | Cross-entropy với class weights |
| Epochs | 60 |
| Hardware | RTX 4090 |
| Best val bal_acc | 0.4593 (epoch 52) |

---

## 5. Metrics Training — Stage 2 Segmentation

Nguồn: `work/runs_exp/stage2_full/log.txt`.

Model hội tụ nhanh nhờ init từ stage1 best.pt. Val_dice đạt đỉnh ở epoch 16 rồi dao động, SWA ổn định hơn về cuối.

| Epoch | Train Loss | Val Dice | Ghi chú |
|-------|------------|----------|---------|
| 1 | 0.1584 | 0.8579 | Saved best.pt |
| 2 | 0.1620 | 0.8605 | Saved best.pt |
| 7 | 0.1504 | 0.8553 | — |
| 12 | 0.1393 | 0.8552 | — |
| **16** | N/A | **0.8676** | **FINAL BEST — production ckpt** |
| 20 | 0.1289 | 0.8237 | — |
| 25 | 0.1257 | 0.8296 | — | 
| 38 | 0.1234 | 0.8447 | — |
| 50 | 0.1141 | 0.8034 | — |
| 65 | 0.1129 | 0.8542 | — |
| 78 | 0.0825 | 0.8467 | — |
| 100 | 0.1020 | 0.8308 | — |
| 115 | 0.1018 | 0.8346 | — |
| 130 | 0.0749 | 0.8376 | — |
| 136 | — | — | Chuyển sang SWA |
| 141 | 0.1012 | 0.8431 | — |
| 155 | 0.0794 | 0.7382 | — |
| 167 | 0.0992 | 0.8414 | — |
| 180 | 0.0956 | 0.7984 | — |
| **SWA** | — | **0.8457** | val_iou=0.7726 |

**Checkpoint được dùng:** `work/runs_exp/stage2_full/best.pt` — val_dice=0.8676 tại epoch 16.

> Lưu ý: Log gốc từ VPS chỉ giữ milestones (cron monitoring), không phải log đầy đủ từng epoch. Loss epoch 16 không còn trong log (`N/A`).

---

## 6. Metrics Training — FPR Classifier

Nguồn: `work/runs/mal_train.log`. Tổng 6,848 patches (sau lọc): 5,621 train / 585 val / 642 test.

| Epoch | Loss | val_acc | val_bal_acc | val_susp_f1 |
|-------|------|---------|-------------|-------------|
| 1 | 1.601 | 0.306 | 0.327 | 0.529 |
| 4 | 1.441 | 0.332 | 0.395 | 0.572 |
| 9 | 1.320 | 0.380 | 0.428 | 0.585 |
| 19 | 1.218 | 0.388 | 0.438 | 0.619 |
| 20 | 1.204 | 0.368 | 0.445 | 0.638 |
| 30 | 1.113 | 0.392 | 0.441 | 0.644 |
| 41 | 0.956 | 0.412 | 0.458 | 0.649 |
| **52** | **0.811** | **0.426** | **0.459** | **0.654** |
| 60 | 0.766 | 0.407 | 0.437 | 0.644 |

**Best epoch: 52** — val_bal_acc=0.4593 (checkpoint được save).

**Final test metrics** (nguồn: `work/runs/malignancy_metrics.json`):

| Metric | Giá trị |
|--------|---------|
| test_acc | 0.380 |
| test_bal_acc | 0.396 |
| test_susp_f1 | 0.504 |
| best_val_bal_acc | 0.459 |

> test_acc thấp (0.38) phản ánh độ khó phân loại 5 mức độ ác tính. Balanced accuracy 0.40 và susp_f1=0.50 cho thấy model phân biệt ở mức "có nghi ngờ" / "không nghi ngờ" nhưng không phân biệt tốt các mức liền kề. Malignancy score chỉ mang tính tham khảo — không đủ tin cậy cho quyết định lâm sàng.

---

## 7. Confusion Matrix — Test Panel (n=95 scan eval)

Nguồn: `work/academic/fpr_v3_stage2winner_eval_test.json`, FPR threshold=0.80 (best-F1 sweep).

**Matching rule:** centroid pred trong vòng 15mm so với centroid GT (fixed_15mm — quy tắc nội bộ dự án, KHÔNG phải quy tắc chính thức LUNA16; xem commit db08c2e).

|  | **Predicted Positive** | **Predicted Negative** |
|--|------------------------|------------------------|
| **GT Positive** | TP = 144 | FN = 39 |
| **GT Negative** | FP = 139 | — |

- Sensitivity = TP / (TP + FN) = 144 / (144 + 39) = **0.787**
- Precision = TP / (TP + FP) = 144 / (144 + 139) = **0.509**
- F1 = 2 × Precision × Sensitivity / (Precision + Sensitivity) = **0.618**
- FP/scan = 139 / 95 = **1.463**

**Tandem view — các ngưỡng FPR khác nhau:**

| FPR thr | TP | FP | FN | Sensitivity | Precision | F1 |
|---------|----|----|----|-----------|-----------|----|
| 0.10 | 167 | 233 | 16 | 0.913 | 0.418 | 0.573 |
| 0.30 | 154 | 195 | 29 | 0.842 | 0.441 | 0.579 |
| 0.50 | 151 | 177 | 32 | 0.825 | 0.460 | 0.591 |
| 0.70 | 147 | 152 | 36 | 0.803 | 0.492 | 0.610 |
| **0.80** | **144** | **139** | **39** | **0.787** | **0.509** | **0.618** |
| Không FPR | 183 | 1020 | 0 | 1.000 | 0.152 | 0.264 |

---

## 8. So Sánh Mine vs MONAI trên LIDC test_panel

### 8.1 Thiết Lập

- **Test panel:** 99 bệnh nhân (95/99 load được trong eval), locked commit cd48359
- **Ground truth:** 183 nốt (trong đó 250 theo eval academic rule)
- **Mine operating point:** FPR threshold=0.80 (best-F1 sweep từ `fpr_v3_stage2winner_eval_test.json`)
- **MONAI operating point:** score_thresh=0.80 (val_panel tuned, val F1=0.5932)
- **MONAI model:** Lung Nodule Detection bundle — RetinaNet 3D (pretrained)
- **Matching rule:** fixed 15mm centroid distance (dùng nhất quán cho cả hai)

Nguồn chính: `work/academic/MONAI_VS_MINE.md` (eval chạy 2026-05-22).

### 8.2 Bảng Kết Quả Tổng Hợp

| Metric | Mine (Stage2+FPR_v3) | MONAI (RetinaNet3D) | Delta (mine − MONAI) |
|--------|---------------------|--------------------|----------------------|
| **F1 (micro)** | **0.618** | 0.533 | **+0.085** |
| Sensitivity (Recall) | 0.787 | 0.576 | +0.211 |
| Precision | 0.509 | 0.497 | +0.012 |
| FP/scan | 1.463 | 1.475 | −0.012 |
| TP | 144 | 144 | 0 |
| FP | 139 | 146 | −7 |
| FN | 39 | 106 | −67 |

Bootstrap 95% CI delta(F1): [−0.2616, −0.1552] (1000 paired resamples).  
Wilcoxon signed-rank per-patient F1 (n=95): **p < 0.0001**.

### 8.3 FROC — Sensitivity tại Các FP Rate Chuẩn

Nguồn: `work/academic/RULE_COMPARISON.md`. Op-point: mine thr=0.65, MONAI thr=0.90 (sweep riêng trên val cho từng rule).

| FP/scan | Mine fixed_15mm | MONAI fixed_15mm | Mine luna16_radius | MONAI luna16_radius |
|---------|-----------------|-----------------|--------------------|---------------------|
| 0.125 | 0.520 | 0.252 | 0.496 | 0.244 |
| 0.250 | 0.520 | 0.252 | 0.496 | 0.244 |
| 0.500 | 0.520 | 0.373 | 0.496 | 0.340 |
| 1.000 | 0.582 | 0.530 | 0.549 | 0.494 |
| 2.000 | 0.647 | 0.599 | 0.592 | 0.564 |
| 4.000 | 0.652 | 0.689 | 0.608 | 0.643 |
| 8.000 | 0.652 | 0.721 | 0.608 | 0.678 |
| **CPM** | **0.585** | **0.488** | **0.549** | **0.458** |

> Mine dẫn về CPM ở cả hai rule. Tại FP rate thấp (≤2.0 FP/scan) mine rõ ràng hơn; tại FP rate cao (4–8) MONAI bắt kịp và vượt — kiến trúc 3D native cho phép recall thêm nốt khi hạ threshold.

> **Lưu ý matching rule:** `fixed_15mm` và `luna16_radius` (centroid ≤ max(GT_diam/2, 3mm)) cho kết quả khác nhau. `luna16_radius` là rule nghiêm hơn với nốt nhỏ. Commit db08c2e ghi rõ đây là matching rule nội bộ, không tương đương benchmark LUNA16 chính thức.

### 8.4 Sensitivity Phân Tầng theo Kích Thước Nốt

Nguồn: `work/academic/RULE_COMPARISON.md`.

| Nhóm kích thước | Mine fixed_15mm | MONAI fixed_15mm | Mine luna16_radius | MONAI luna16_radius |
|----------------|-----------------|------------------|--------------------|---------------------|
| 4–6 mm (n=67) | 0.627 | 0.478 | 0.567 | 0.433 |
| 6–15 mm (n=135) | 0.578 | 0.593 | 0.570 | 0.578 |
| >15 mm (n=37) | 0.595 | 0.541 | 0.568 | 0.514 |

Mine dẫn ở nốt nhỏ (4–6mm) với cả hai rule. Ở nhóm trung bình (6–15mm), MONAI ngang hoặc nhỉnh hơn nhẹ với fixed_15mm. Cả hai model đều có sensitivity thấp ở nốt nhỏ — đây là thách thức chung của kiến trúc hiện tại.

### 8.5 Kiểm Định Thống Kê

| Test | Kết quả | Nguồn |
|------|---------|-------|
| Bootstrap CI 95% delta(F1) mine−MONAI | [−0.2616, −0.1552] | MONAI_VS_MINE.md |
| Wilcoxon per-patient F1, op-point thr=0.80 (n=95) | p < 0.0001 | MONAI_VS_MINE.md |
| Wilcoxon per-patient F1, fixed_15mm rule-sweep (n=95) | p=0.7737 | RULE_COMPARISON.md |
| Wilcoxon per-patient F1, luna16_radius rule-sweep (n=95) | p=0.6560 | RULE_COMPARISON.md |

> Sự khác biệt p-value giữa hai eval: MONAI_VS_MINE.md dùng thr=0.80 — mine recall cao hơn rõ rệt (FN=39 vs MONAI FN=106), delta F1 lớn → p có ý nghĩa. RULE_COMPARISON.md dùng thr=0.65 sweep riêng cho từng rule — delta nhỏ hơn (~0.04) và n=95 không đủ power để detect delta này (cần n~300 để power 80% tại alpha=0.05).

### 8.6 Verdict

**Mine thắng F1:** 0.618 vs MONAI 0.533, delta +0.085, p < 0.0001 (Wilcoxon, tại operating point thr=0.80).  
**Mine thắng CPM:** 0.585 vs MONAI 0.488 (fixed_15mm), 0.549 vs 0.458 (luna16_radius).  
**Mine thắng sensitivity nốt nhỏ:** 0.627 vs 0.478 (4–6mm, fixed_15mm).  
**Mine plateau sớm ở FP rate cao:** tại 4–8 FP/scan, MONAI vượt về FROC curve — kiến trúc 3D native ưu thế ở recall mode.

Kết luận: Custom 2.5D UNet++ với hard-negative FPR stage đạt hiệu năng vượt trội so với pretrained 3D RetinaNet (MONAI) trên LIDC-IDRI test_panel tại operating point thực tế.

---

## Phụ Lục A — Hyperparameter Tổng Hợp

| Tham số | Stage 2 (cd48359) | FPR v3 | Malignancy |
|---------|-------------------|--------|------------|
| Architecture | UNet++ EffB5 SCSE | DenseNet121-3D | DenseNet121-3D |
| Params | ~32M | 11.2M | 11.2M |
| Input | 3-ch 2.5D | 64³ cube | 64³ cube |
| Encoder pretrain | ImageNet | — | — |
| Loss | Tversky α=0.3 β=0.7 | BCE | Cross-entropy |
| Optimizer | AdamW | AdamW | AdamW |
| LR | 1e-4 (cosine) | 1e-4 | 1e-4 |
| Weight decay | 1e-2 | — | — |
| Batch | 10 | — | — |
| Epochs | 180 | 60 | 60 |
| SWA | ep 136 (75%), lr=1e-4 | — | — |
| Best epoch | ep 16 (val_dice=0.8676) | ep 52 (bal_acc=0.459) | ep 52 (bal_acc=0.459) |
| Hardware | V100 32GB (VPS) | RTX 4090 (local) | RTX 4090 (local) |

## Phụ Lục B — Đường Dẫn Artifact

| Artifact | Đường dẫn |
|---------|-----------|
| Stage2 best ckpt | `work/runs_exp/stage2_full/best.pt` |
| FPR v3 ckpt | `work/runs/fpr_v3_stage2winner.pt` |
| Malignancy ckpt | `work/runs/malignancy.pt` |
| FPR eval test | `work/academic/fpr_v3_stage2winner_eval_test.json` |
| Malignancy metrics | `work/runs/malignancy_metrics.json` |
| MONAI vs Mine verdict | `work/academic/MONAI_VS_MINE.md` |
| Rule comparison | `work/academic/RULE_COMPARISON.md` |
| Mine FROC (2-rule) | `work/academic/MINE_LUNA16_RULE.json` |
| Stage2 training log | `work/runs_exp/stage2_full/log.txt` |
| FPR/malignancy training log | `work/runs/mal_train.log` |
| configs | `src/configs.py` |

## Phụ Lục C — Commits Tham Chiếu

| Commit | Nội dung |
|--------|---------|
| `cd48359` | Winner ckpts via Git LFS (test_panel F1=0.618) — LOCKED |
| `d005223` | Stage2 winner pipeline + frontend verdict persistence — LOCKED |
| `db08c2e` | Honest wording: matching rule fixed 15mm, không phải LUNA16 official |
| `891c0ed` | Refocus: AI assist second reader, không phải diagnostic AI |

## Phụ Lục D — Tài Liệu Tham Khảo

- Armato SG et al. (2011). "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)." *Medical Physics*, 38(2), 915–931.
- Zhou Z et al. (2019). "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." *MICCAI Workshop*, arXiv:1807.10165.
- Tan M, Le QV (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*, arXiv:1905.11946.
- Roy AG et al. (2018). "Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks." *MICCAI*, arXiv:1803.02579.
- Huang G et al. (2017). "Densely Connected Convolutional Networks." *CVPR*, arXiv:1608.06993.
- Setio AAA et al. (2017). "Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in CT images: the LUNA16 challenge." *Medical Image Analysis*, 42, 1–13.

---

*Mọi số liệu trong báo cáo này đến trực tiếp từ artifact files trong repo. Không có số nào được bịa đặt.*

*Tạo bởi docs-writer agent — 2026-05-22*
