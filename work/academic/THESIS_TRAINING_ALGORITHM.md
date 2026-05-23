# Thuật Toán Training — Mô Hình Stage2+FPR_v3 (commit cd48359)

**Phiên bản**: 2026-05-22  
**Commit tham chiếu**: `cd48359` (winner ckpt, test_panel F1=0.618)  
**Nguồn số liệu**: `src/configs.py`, `src/experiments.py`, `src/train_experiment.py`, `src/dataset_v2.py`, `src/dataset.py`, `work/runs_exp/stage2_full/log.txt`, `work/runs_exp/stage2_full/eval.json`, `work/runs/mal_train.log`

---

## 1. Tổng Quan Pipeline Training

Hệ thống sử dụng ba mô hình độc lập, được training theo thứ tự:

| Thứ tự | Mô hình | Mục đích | Kết quả |
|--------|---------|----------|---------|
| 1 | Stage 1 — UNet++ EfficientNet-B5 | Khởi điểm (baseline best.pt) | Val Dice làm nền tảng |
| 2 | Stage 2 — UNet++ EfficientNet-B5 (fine-tune từ Stage 1) | Segmentation nodule 2.5D chính | Val Dice = 0.8676 @ epoch 16 |
| 3 | FPR Classifier — DenseNet121-3D binary | Lọc false positive sau Stage 2 | Test AUC = 0.9364 (xem mục 3) |
| 4 | Malignancy Classifier — DenseNet121-3D 5-class | Phân loại mức độ ác tính | Val bal_acc = 0.4593 @ epoch 52 |

Mô hình Stage 2 là mô hình chính cho detection. FPR Classifier được áp dụng ở bước hậu xử lý. Malignancy Classifier cung cấp điểm nguy cơ phụ trợ (không dùng để quyết định phát hiện).

---

## 2. Stage 2 — Segmentation Training (Mô hình chính)

### 2.1 Chuẩn bị dữ liệu đầu vào

**Định dạng lưu trữ**: mỗi bệnh nhân được lưu thành một file `.h5` trong thư mục `work/preprocessed/`. File chứa:
- `images`: mảng `[N, 512, 512]` kiểu `int16`, giá trị HU gốc từ DICOM.
- `mask_per_rad`: mảng `[N, 4, 512, 512]` kiểu `uint8`, mask nhị phân của 4 radiologist.
- Metadata: `pixel_spacing`, `slice_thickness`, `z_positions`.

**Chuỗi tiền xử lý**:

```
1. Đọc DICOM → apply RescaleSlope, RescaleIntercept → HU int16
2. Sắp xếp lát cắt theo ImagePositionPatient[z] tăng dần
3. Normalize HU:
       x = clip(x, HU_LO=-1350.0, HU_HI=150.0)
       x = (x - HU_LO) / (HU_HI - HU_LO)  →  [0, 1]
4. Xây dựng input 2.5D (3-channel):
       channels = [slice[i-1], slice[i], slice[i+1]]
       padding: nếu i=0 thì dùng slice[0] thay slice[-1]
5. Xây dựng ground truth mask từ 4 radiologist contours:
       mask = (sum(mask_per_rad, axis=1) >= 2)  [consensus2]
       Lọc: bỏ mask có tổng pixel < 10 (nonNodule markers)
```

Nguồn: `src/configs.py` (HU_LO=-1350, HU_HI=150, MIN_MASK_PIXELS=10), `src/dataset.py` (`_normalize`, `_build_mask`).

### 2.2 Xây dựng Ground Truth

Stage 2 sử dụng chế độ **consensus2**: một pixel được đánh nhãn là nodule khi ít nhất 2/4 radiologist đồng thuận. Đây là kết quả thực nghiệm từ loạt pilot EXP01–EXP05:

- `union` (1/4 radiologist): nhãn nhiễu, model không hội tụ tốt.
- `consensus2` (2/4 radiologist): tín hiệu sạch hơn, đây là cấu hình winner.
- `consensus3` và `all4`: quá ít mẫu positive.

Nguồn: `src/experiments.py` (note cho `stage2_full`), `src/dataset.py` (`_build_mask`).

### 2.3 Kiến trúc mô hình

```
UNet++ (segmentation_models_pytorch)
  encoder:          EfficientNet-B5, pretrained ImageNet
  in_channels:      3 (2.5D: 3 lát cắt liền kề)
  classes:          1 (binary segmentation)
  decoder_attention: SCSE (Channel Squeeze-and-Excitation + Spatial SE)
  output activation: sigmoid (áp dụng ở bước loss, không ở forward)
```

- EfficientNet-B5 có khoảng 28–30M tham số.
- SCSE attention được gắn vào mỗi decoder block, giúp model tập trung vào vùng nodule nhỏ.

Nguồn: `src/train_experiment.py` hàm `make_model` (dòng 53–57), `src/configs.py` (`ENCODER = "efficientnet-b5"`).

### 2.4 Hàm Loss

Stage 2 sử dụng **Tversky Loss** với tham số alpha=0.3, beta=0.7:

```
TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7, include_background=False)
```

Tversky Loss là tổng quát hóa của Dice Loss:

```
TI = TP / (TP + alpha * FP + beta * FN)
L  = 1 - TI
```

Với alpha=0.3, beta=0.7: false negative bị phạt nặng hơn false positive (0.7/0.3 ≈ 2.3 lần). Điều này phù hợp với bài toán screening y tế, nơi bỏ sót nodule (FN) nguy hiểm hơn báo nhầm (FP).

Baseline sử dụng `DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=0.5)`. Pilot thực nghiệm (EXP03 so với EXP00/EXP01) cho thấy Tversky cải thiện recall trên bucket 10–20mm.

Nguồn: `src/dataset_v2.py` hàm `make_loss` (dòng 252–267), `src/experiments.py` (`tversky_alpha=0.3, tversky_beta=0.7`).

### 2.5 Optimizer và Learning Rate Schedule

```
Optimizer:     AdamW
lr:            1e-4
weight_decay:  1e-2  (từ log.txt; configs.py default là 1e-4)
```

Schedule: **Cosine annealing với linear warmup**

```python
def lr_lambda(step):
    if step < n_warm:
        return step / n_warm                        # linear warmup
    progress = (step - n_warm) / (n_total - n_warm)
    return 0.5 * (1 + cos(pi * progress))          # cosine decay
```

- `WARMUP_FRAC = 0.05`: 5% đầu của tổng số step là warmup.
- Scheduler bước theo từng batch (step-level), không phải epoch-level.
- Scheduler dừng cập nhật khi bắt đầu giai đoạn SWA (từ epoch 136 trở đi).

Nguồn: `src/train_experiment.py` hàm `warmup_cosine` (dòng 45–50), `src/configs.py` (`WARMUP_FRAC=0.05`, `SWA_LR=1e-4`), `work/runs_exp/stage2_full/log.txt`.

### 2.6 Stochastic Weight Averaging (SWA)

SWA được bật cho Stage 2 (`enable_swa=True`). Cơ chế:

```
swa_start_epoch = int(0.75 * 180) = 135  → bắt đầu từ epoch 136
```

Từ epoch 136:
1. Học từ `model` bình thường (AdamW + cosine scheduler dừng).
2. `swa_model.update_parameters(model)` sau mỗi epoch → tích lũy trung bình tham số.
3. `SWALR` với `swa_lr=1e-4`, `anneal_epochs=3` thay thế scheduler ban đầu.
4. Sau epoch 180: `update_bn(train_dl, swa_model)` → cập nhật lại BatchNorm statistics.

Kết quả: `swa.pt` có val_dice=0.8457, thấp hơn `best.pt` (0.8676). Production checkpoint là `best.pt`.

Nguồn: `src/train_experiment.py` (dòng 175–213), `work/runs_exp/stage2_full/log.txt`.

### 2.7 Augmentation Pipeline

Augmentation được áp dụng trực tiếp trên numpy array, đồng bộ cho cả ảnh và mask:

| Transform | Tham số | Xác suất |
|-----------|---------|----------|
| RandomFlip | horizontal + vertical | p=0.5 mỗi chiều |
| RandomRotate | max_deg=20 | p=0.5 |
| RandomIntensity | shift=0.05, scale=0.1 | p=0.5 |
| RandomElastic | alpha=80, sigma=10 | p=0.3 |
| RandomNoise | std=0.02 | p=0.3 |

Không có augmentation màu sắc (ColorJitter) vì ảnh CT là grayscale normalize.

Nguồn: `src/dataset.py` hàm `get_train_aug` (dòng 213–220).

### 2.8 Cấu hình Dataset

Stage 2 sử dụng **v1 data** (không phải v2). Lý do: ba thực nghiệm với v2 data + class_balanced sampler đều phát sinh NaN dưới AMP fp16, không phục hồi được.

```
Dataset:   LIDCSeg25DMulti (v1, work/preprocessed/)
mask_mode: consensus2
sampler:   uniform (shuffle=True, không class-balanced)
Train set: 1507 slices từ 812 bệnh nhân
Val set:   154 slices từ 99 bệnh nhân
```

Nguồn: `work/runs_exp/stage2_full/log.txt` (dòng 3–4), `src/experiments.py` (`use_v2_data=False`, `sampler="uniform"`).

### 2.9 Vòng lặp Training (Pseudocode)

```python
# Khởi tạo
model = UNetPlusPlus(encoder="efficientnet-b5", in_ch=3, decoder_attn="scse")
model.load_state_dict(torch.load("work/runs/best.pt"))  # init từ Stage 1

loss_fn   = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = warmup_cosine(optimizer, n_warm=0.05*n_steps, n_total=n_steps)
scaler    = GradScaler()   # AMP fp16

swa_model    = None
swa_start_ep = int(0.75 * 180)   # = 135, bắt đầu từ epoch 136

best_dice = 0.0

for ep in range(180):
    # --- Train phase ---
    model.train()
    for img, mask in train_loader:             # batch_size=10
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            pred = model(img)                  # [B, 1, 512, 512]
            loss = loss_fn(pred, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer); scaler.update()
        if ep < swa_start_ep:
            scheduler.step()

    # --- Chuyển sang SWA ---
    if ep == swa_start_ep:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-4, anneal_epochs=3)

    if swa_model is not None:
        swa_model.update_parameters(model)
        swa_scheduler.step()

    # --- Val phase ---
    val_dice, val_iou = evaluate_dice(model, val_loader)
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(checkpoint, "best.pt")

# --- SWA finalization ---
update_bn(train_loader, swa_model)    # cập nhật BN statistics
swa_dice = evaluate_dice(swa_model, val_loader)
torch.save(swa_checkpoint, "swa.pt")
```

### 2.10 Bảng Hyperparameter Stage 2

| Tham số | Giá trị | Nguồn |
|---------|---------|-------|
| Encoder | efficientnet-b5 | `configs.py`, `experiments.py` |
| In_channels | 3 (2.5D) | `experiments.py` |
| Decoder attention | scse | `train_experiment.py` |
| Epochs | 180 | `experiments.py` (stage2_full) |
| Batch size | 10 | `experiments.py` |
| Optimizer | AdamW | `train_experiment.py` |
| Learning rate | 1e-4 | `experiments.py` |
| Weight decay | 1e-2 | `log.txt` |
| Gradient clip | max_norm=5.0 | `train_experiment.py` |
| Loss | Tversky(α=0.3, β=0.7) | `experiments.py` |
| Schedule | Cosine + linear warmup (5%) | `train_experiment.py`, `configs.py` |
| SWA start | epoch 136 (75% × 180) | `train_experiment.py` |
| SWA lr | 1e-4 | `configs.py` |
| SWA anneal_epochs | 3 | `train_experiment.py` |
| AMP | fp16 GradScaler | `train_experiment.py` |
| mask_mode | consensus2 | `experiments.py` |
| Init from | Stage 1 best.pt | `experiments.py` |
| Hardware | 1× V100 (VPS) | `log.txt` |

### 2.11 Kết quả Training Stage 2

Milestones từ log thực tế (`work/runs_exp/stage2_full/log.txt`):

| Epoch | Train loss | Val Dice | Ghi chú |
|-------|-----------|----------|---------|
| 1 | 0.1584 | 0.8579 | best.pt đầu tiên |
| 2 | 0.1620 | 0.8605 | |
| 16 | — | **0.8676** | **FINAL BEST** |
| 50 | 0.1141 | 0.8034 | |
| 100 | 0.1020 | 0.8308 | |
| 136 | — | — | SWA bắt đầu |
| 180 | 0.0956 | 0.7984 | |
| SWA | — | 0.8457 | swa.pt |

Best checkpoint: epoch 16, val_dice=0.8676. Mô hình hội tụ sớm (epoch 16) và không cải thiện thêm sau đó. SWA không vượt được best.pt trong trường hợp này.

Kết quả trên test_panel (`work/runs_exp/stage2_full/eval.json`):

| Metric | Seg-only (thr=0.7) | Với FPR v3 (thr=0.80) |
|--------|-------------------|----------------------|
| Sensitivity | 0.556 | 0.787 |
| Precision | 0.570 | 0.509 |
| F1 | 0.563 | **0.618** |
| FP/scan | 1.06 | 1.46 |

Kết quả phân tầng theo kích thước nodule (seg-only, test set):

| Bucket | TP / GT | Sensitivity |
|--------|---------|-------------|
| 4–6 mm | 36 / 67 | 0.537 |
| 6–15 mm | 78 / 135 | 0.578 |
| ≥15 mm | 21 / 37 | 0.568 |

---

## 3. FPR Classifier — False Positive Reduction

### 3.1 Mục đích

Sau khi Stage 2 tạo ra các candidate nodule qua connected components, một phần lớn là false positive (ví dụ: mạch máu cắt ngang, sẹo phổi). FPR Classifier là mô hình 3D binary nhận patch 3D xung quanh mỗi candidate và dự đoán xác suất đó là nodule thật.

### 3.2 Chuẩn bị dữ liệu

```
Input file:  work/academic/fpr_val.npz, fpr_test.npz
Format:      patches [N, P, P, P] kiểu int16 HU
             labels  [N]           0=FP, 1=TP
             confs   [N]           confidence từ Stage 2

Normalize HU cho FPR:
    clip(x, HU_LO=-1024.0, HU_HI=600.0)
    x = (x - HU_LO) / (HU_HI - HU_LO)
```

Lưu ý: window HU cho FPR (-1024, 600) khác với Stage 2 (-1350, 150) — rộng hơn để giữ thông tin mô mềm xung quanh nodule.

**Class balancing**: sử dụng `WeightedRandomSampler` để cân bằng pos/neg:

```python
weights = where(label==1, 1.0/n_pos, 1.0/n_neg)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

Nguồn: `src/train_fpr_classifier.py` (dòng 103–106).

### 3.3 Kiến trúc

```
DenseNet121-3D (MONAI)
  spatial_dims:  3
  in_channels:   1  (single-channel patch 3D)
  out_channels:  2  (binary: FP=0, nodule=1)
  Tham số:       ~11.2 triệu (từ mal_train.log)
```

Nguồn: `src/train_fpr_classifier.py` (dòng 112), `src/webapp/predict.py` (dòng 117).

### 3.4 Loss và Optimizer

```python
pos_weight = tensor([1.0, n_neg/n_pos])
loss = F.cross_entropy(logits, y, weight=pos_weight)

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

FPR classifier dùng `lr=3e-4` (mặc định `--lr` trong script), khác Stage 2 (1e-4).

Nguồn: `src/train_fpr_classifier.py` (dòng 88–116).

### 3.5 Augmentation (3D)

```python
# Với mỗi sample (augment=True):
for ax in [0, 1, 2]:
    if random() < 0.5:
        x = flip(x, axis=ax)    # flip theo 3 trục
k = randint(0, 4)
if k > 0:
    x = rot90(x, k, axes=(1,2)) # rotate 90° trên mặt phẳng y-z
```

Nguồn: `src/train_fpr_classifier.py` hàm `FPRDataset.__getitem__` (dòng 48–54).

### 3.6 Vòng lặp Training (Pseudocode)

```python
model  = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
optim  = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched  = CosineAnnealingLR(optim, T_max=epochs)
scaler = GradScaler()

best_auc = 0.0

for ep in range(1, epochs+1):
    model.train()
    for x, y in train_loader:                    # batch_size CLI arg
        optim.zero_grad()
        with autocast("cuda"):
            logits = model(x)
            loss   = cross_entropy(logits, y, weight=pos_weight)
        scaler.scale(loss).backward()
        scaler.step(optim); scaler.update()
    sched.step()

    if ep % 3 == 0:                              # val mỗi 3 epoch
        metrics = run_validation(model, val_loader)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(checkpoint, out_path)
```

Hàm `run_validation` sweep threshold t ∈ {0.1, 0.15, ..., 0.9} và chọn threshold có F1 cao nhất trên val set. Threshold này được lưu vào checkpoint (`best_thr`) và dùng tại inference.

Nguồn: `src/train_fpr_classifier.py` hàm `run_validation` (dòng 59–85) và `main` (dòng 120–148).

### 3.7 Số liệu thực nghiệm

Từ `work/runs/mal_train.log` (đây là run malignancy, cùng kiến trúc DenseNet121-3D):

```
Total: 6848 patches | train: 5621 | val: 585 | test: 642
Class counts (train): [857, 1305, 2159, 784, 516]
Class weights:        [1.312, 0.861, 0.521, 1.434, 2.179]
```

Với FPR binary classifier riêng, kết quả trên test_panel (từ `work/runs_exp/stage2_full/log.txt`):
- FPR threshold = 0.80: Sensitivity=0.787, Precision=0.509, **F1=0.618**, FP/scan=1.46
- FPR threshold = 0.10: Sensitivity=0.913, Precision=0.417, F1=0.573, FP/scan=2.45

Checkpoint `fpr.pt` được lưu với `best_thr` từ val set. Tại inference, `predict.py` đọc `best_thr` từ checkpoint và áp dụng.

---

## 4. Malignancy Classifier — Phân loại mức độ ác tính

### 4.1 Mục đích và giới hạn

Malignancy Classifier dự đoán điểm ác tính 1–5 (theo thang LIDC) cho mỗi nodule đã phát hiện. Đây là tín hiệu hỗ trợ — không thay thế đánh giá lâm sàng. Trong production, điểm AI được kết hợp với kích thước nodule theo tỷ lệ 40% AI + 60% size-prior.

### 4.2 Kiến trúc và cấu hình

```
DenseNet121-3D (MONAI)
  spatial_dims:  3
  in_channels:   1
  out_channels:  5  (class 1-5 malignancy)
  Tham số:       11.2M

Patch size:    32×32×32 voxel xung quanh centroid
HU window:     clip(-1024, 600) → normalize [0,1]
```

Nguồn: `src/webapp/predict.py` (`_MAL_PATCH=32`, `_MAL_HU=(-1024, 600)`).

### 4.3 Training setup

```
Epochs:        60
Optimizer:     AdamW (lr từ script mặc định)
Loss:          Cross entropy với class weights
Scheduler:     CosineAnnealingLR(T_max=60)
```

Class weights (từ `work/runs/mal_train.log`):

| Class | Count (train) | Weight |
|-------|--------------|--------|
| 1 (rất lành tính) | 857 | 1.312 |
| 2 | 1305 | 0.861 |
| 3 | 2159 | 0.521 |
| 4 | 784 | 1.434 |
| 5 (rất ác tính) | 516 | 2.179 |

### 4.4 Kết quả

| Metric | Val (best, ep 52) | Test |
|--------|-------------------|------|
| Accuracy | 0.4256 | 0.3801 |
| Balanced accuracy | **0.4593** | 0.3962 |
| Suspicious F1 (class 4+5) | 0.6542 | 0.5041 |

Best checkpoint lưu tại: `work/runs/malignancy.pt` (epoch 52).

---

## 5. Inference Pipeline (Sử dụng các mô hình đã train)

```
1. Đọc DICOM series
      → sort theo ImagePositionPatient[z]
      → apply RescaleSlope, RescaleIntercept
      → volume_hu: [N, 512, 512] int16

2. Lung segmentation
      → lungmask R231 (Hofmanninger et al. 2020)
      → fallback: HU threshold + morphological closing

3. Stage 2 segmentation (slice-by-slice với TTA)
      for i in range(N):
          stack = [normalize(slice[i-1]), normalize(slice[i]), normalize(slice[i+1])]
          x = tensor [1, 3, 512, 512]
          # TTA: average 4 flip augmentations
          p = mean(sigmoid(model(x)),
                   flip_h(sigmoid(model(flip_h(x)))),
                   flip_v(sigmoid(model(flip_v(x)))),
                   flip_hv(sigmoid(model(flip_hv(x)))))
          # Ensemble: average best.pt + swa.pt nếu có
          prob[i] = mean over models
      → prob volume: [N, 512, 512] float32

4. Threshold + binary mask
      mask = (prob > 0.97)    [THRESHOLD_DEFAULT từ configs.py]

5. Morphological cleanup
      → binary_closing (structure 1×3×3, 1 iter) → lấp lỗ hổng nhỏ
      → binary_erosion + binary_dilation → loại speckle noise

6. Connected components 3D
      → cc_label với structure 3×3×3 (26-connectivity)
      → Lọc blob:
          - bỏ nếu voxels < MIN_NODULE_VOXELS=120
          - bỏ nếu elongation_ratio > 4.0  (vessels cắt dọc)

7. Đo đường kính nodule
      → core_mask = blob & (prob > 0.85)
      → nếu core_voxels >= 8: diameter = 2*(3*vol_core/(4π))^(1/3)
      → nếu không: diameter = diameter_full * 0.7

8. FPR classification
      for each candidate:
          extract 48×48×48 cube around centroid
          normalize HU(-1024, 600) → [0,1]
          fpr_prob = softmax(fpr_model(cube))[1]    # P(nodule)
          nếu fpr_prob < best_thr (từ checkpoint) → drop

9. Merge near-duplicates (NMS)
      → nms_merge_duplicates: centroid distance < 0.7 * max_diameter → giữ cái có confidence cao hơn

10. Malignancy scoring
      for each surviving nodule:
          extract 32×32×32 patch
          normalize HU(-1024, 600)
          probs = softmax(malignancy_model(patch))   # [5]
          ai_class = argmax + 1
          ai_expected = sum(k * P(k) for k in 1..5)
          combined = 0.6 * size_prior + 0.4 * ai_class

11. Output
      → List nodule với: bbox, diameter_mm, confidence, fpr_prob,
                          ai_class, ai_susp_prob, lung_rads, risk_combined
```

Nguồn: `src/webapp/predict.py` (toàn bộ), `src/configs.py` (THRESHOLD_DEFAULT=0.97, MIN_NODULE_VOXELS=120).

---

## 6. Reproducibility

| Mục | Giá trị |
|-----|---------|
| Random seed | Không set toàn cục; PyTorch default |
| Framework | PyTorch + MONAI + segmentation_models_pytorch |
| AMP | fp16 GradScaler (cả Stage 2 và FPR) |
| Hardware Stage 2 | 1× NVIDIA V100 32GB (VPS, 2026-05-14 đến 2026-05-15) |
| Thời gian Stage 2 | ~3.5h trên V100 (từ log.txt: "~3.5h total on V100") |
| Hardware Malignancy | NVIDIA GeForce RTX 4090 (từ mal_train.log) |
| Thời gian Malignancy | ~5 giây/epoch × 60 epoch ≈ 5 phút |
| num_workers | 8 (từ `configs.py`) |
| pin_memory | True |
| persistent_workers | True |

**Để reproduce Stage 2:**

```bash
# Trên VPS với V100, LIDC_ROOT trỏ đúng
export LIDC_ROOT=/workspace
python src/train_experiment.py --exp stage2_full --epochs 180
# Output: work/runs_exp/stage2_full/best.pt, swa.pt, log.txt
```

**Lưu ý quan trọng**: `stage2_full` init từ `work/runs/best.pt` (Stage 1 baseline). Nếu file này không có hoặc khác version, val_dice kết quả sẽ khác. Winner checkpoint hiện tại (commit `cd48359`) có `val_dice=0.8676`.

---

## 7. Tham chiếu Code

| Thành phần | File | Hàm / Class chính |
|------------|------|-------------------|
| Hyperparameters toàn cục | `src/configs.py` | — |
| Experiment config | `src/experiments.py` | `stage2_full` |
| Training loop Stage 2 | `src/train_experiment.py` | `train_one`, `warmup_cosine`, `make_model` |
| Dataset v1 (2.5D) | `src/dataset.py` | `LIDCSeg25D`, `get_train_aug` |
| Dataset v2 + sampler | `src/dataset_v2.py` | `LIDCSeg25DMulti`, `LIDCSeg25DV2`, `make_loss` |
| FPR training | `src/train_fpr_classifier.py` | `FPRDataset`, `run_validation`, `main` |
| Inference pipeline | `src/webapp/predict.py` | `predict_nodules`, `find_nodules`, `predict_fpr_for_nodules`, `predict_malignancy_for_nodules` |
| Stage 2 run log | `work/runs_exp/stage2_full/log.txt` | — |
| Stage 2 eval | `work/runs_exp/stage2_full/eval.json` | — |
| Malignancy run log | `work/runs/mal_train.log` | — |
