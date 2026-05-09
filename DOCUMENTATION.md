# Nodura — Tài liệu kỹ thuật chi tiết

Đồ án **AI phát hiện và đánh giá nguy cơ ung thư phổi** từ ảnh CT, dùng dataset LIDC-IDRI.

> Đây là tài liệu giải thích chi tiết toàn bộ thuật toán + cách thức thực hiện. Để cài đặt + chạy, xem [README.md](README.md).

---

## Mục lục

1. [Bài toán](#1-bài-toán)
2. [Dataset](#2-dataset--lidc-idri)
3. [Pipeline tổng thể](#3-pipeline-tổng-thể)
4. [Tiền xử lý DICOM](#4-tiền-xử-lý-dicom--02_preprocesspy)
5. [Segmentation Model — UNet++ B5](#5-segmentation-model--05_trainpy)
6. [Malignancy Classifier — DenseNet121-3D](#6-malignancy-classifier--10_train_malignancypy)
7. [Lung Segmentation — Lungmask R231](#7-lung-segmentation--lungmask-r231-pretrained)
8. [Hậu xử lý 3D](#8-hậu-xử-lý-3d--07_detect_3dpy)
9. [Render 3D — Marching Cubes + Plotly](#9-render-3d--08_render_3dpy)
10. [Clinical Decision Support](#10-clinical-decision-support--clinicalpy)
11. [Webapp Architecture](#11-webapp-architecture)
12. [Tech Stack](#12-tech-stack-summary)
13. [Kết quả](#13-kết-quả)
14. [Hạn chế & Hướng phát triển](#14-hạn-chế--hướng-phát-triển)

---

## 1. Bài toán

| Mục | Chi tiết |
|---|---|
| **Input** | Ảnh CT phổi (DICOM, 100-500 slices/bệnh nhân, 512×512 grayscale) |
| **Output** | Phát hiện + đo + đánh giá nguy cơ ác tính của các nodule phổi |
| **Mục tiêu** | Hỗ trợ bác sĩ chẩn đoán sớm ung thư phổi |

⚠️ **Quan trọng:** AI này **KHÔNG thay thế** chẩn đoán bác sĩ. Chỉ là decision-support — giúp ưu tiên case + đưa ra số liệu khách quan.

---

## 2. Dataset — LIDC-IDRI

| Thông số | Giá trị |
|---|---|
| Bệnh nhân | 1010 |
| DICOM slices | ~244,000 |
| Annotations | 4 radiologist độc lập vẽ polygon contour + đánh giá |
| Per-nodule labels | id, polygon từng slice, malignancy 1-5, calcification, sphericity, margin, lobulation, spiculation, texture, internal_structure |
| Source | [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/), public, ~125 GB |

**Tại sao 4 radiologist?** Phòng case bác sĩ nhìn nhầm → consensus 4 người tin cậy hơn. Dùng **union mask** (≥1/4 đánh dấu) cho training.

---

## 3. Pipeline tổng thể

```
┌──────────────┐
│ DICOM raw    │  125 GB, 1010 bệnh nhân, 244k slices
└──────┬───────┘
       ↓
[01] validate     DICOM ↔ XML matching, build SOP index
[02] preprocess   DICOM → HDF5 (1 file/bệnh nhân, gzip lossless)
[03] split        Train/val/test 80/10/10 stratified theo malignancy
[05] train        Segmentation: UNet++ + EfficientNet-B5 + SCSE
[06] evaluate     Metrics: Dice, IoU, F1 nodule, HD95
[07] detect_3d    Inference 3D từng bệnh nhân
[08] render_3d    Plotly Mesh3d HTML + STL từng nodule
[09] extract      Crop 32³ patch + label malignancy
[10] train_mal    Classifier: DenseNet121-3D
[webapp]          FastAPI backend + Next.js frontend
```

---

## 4. Tiền xử lý DICOM — `02_preprocess.py`

### 4.1. Đọc DICOM
- **pydicom** đọc 244k file `.dcm`
- Sort theo `ImagePositionPatient[2]` (toạ độ z) — đảm bảo thứ tự slice từ chân lên đầu
- Convert pixel raw → **Hounsfield Units (HU)**:
  ```
  HU = pixel × RescaleSlope + RescaleIntercept
  ```
  Bảng tham khảo HU:
  | Mô | HU |
  |---|---|
  | Khí phổi | -1000 |
  | Nhu mô phổi | -700 đến -900 |
  | Mỡ | -100 |
  | Nước | 0 |
  | Mô mềm | 0 đến 100 |
  | Xương | > 400 |

### 4.2. Build mask từ polygon XML
- Mỗi nodule có **4 channel** (1 cho mỗi radiologist)
- Polygon → bitmap mask 512×512 bằng `PIL.ImageDraw.polygon`
- Tensor cuối: `[N_slices, 4, H, W]` uint8

### 4.3. Lưu HDF5 (gzip level 4)
```
work/preprocessed/<PatientID>.h5
├── images          [N, 512, 512] int16    HU values
├── mask_per_rad    [N, 4, 512, 512] uint8 4 radiologist masks
├── sop_uids        [N] string             DICOM SOP UIDs
├── z_positions     [N] float32            mm
└── attrs:
    ├── pixel_spacing       (yx mm)
    ├── slice_thickness     (mm)
    ├── image_position_first
    ├── image_orientation
    └── nodule_meta         JSON list per-nodule full metadata
```

→ **125 GB DICOM xuống 13 GB HDF5** (chỉ giữ slice ±3 quanh nodule + gzip).

### 4.4. Multiprocessing
- `concurrent.futures.ProcessPoolExecutor` với (CPU - 1) workers
- Resume tự động (skip nếu file h5 đã tồn tại)
- ~2-4 giờ trên i7 8-core

---

## 5. Segmentation Model — `05_train.py`

### 5.1. Kiến trúc

```
Input: 3 stacked CT slices (2.5D approach)
       Slice [i-1, i, i+1] → 512×512×3 (ImageNet-compatible)
            ↓
   ╔═══════════════════════════╗
   ║ EfficientNet-B5           ║   Encoder, 28M params
   ║   - MBConv blocks          ║   ImageNet pretrained
   ║   - Squeeze-Excitation     ║   Compound scaling
   ╚═══════════════════════════╝
            ↓
   ╔═══════════════════════════╗
   ║ UNet++ Decoder            ║   Dense skip connections
   ║   - Nested decoders        ║   = 5 paths nối encoder ↔ decoder
   ║   - SCSE attention         ║   Spatial + Channel SE
   ╚═══════════════════════════╝
            ↓
        512×512×1   binary mask (sigmoid)
```

### 5.2. Tại sao chọn các thành phần?

| Component | Lý do |
|---|---|
| **2.5D (3 slice)** | Có context z-direction nhưng vẫn dùng được pretrained 2D backbone (ImageNet) |
| **UNet++** | Dense skip > UNet thường — giảm semantic gap, nodule nhỏ rõ hơn |
| **EfficientNet-B5** | Sweet spot accuracy/memory: fit V100 16GB, accuracy > B4 |
| **SCSE attention** | Spatial + Channel attention → model tập trung vùng có nodule, giảm FP |

### 5.3. Loss function — DiceFocal
```python
loss = 1.0 × DiceLoss + 0.5 × FocalLoss(γ=2.0)
```
| Loss | Vai trò |
|---|---|
| **Dice** | Tốt cho overlap, handle class imbalance (nodule chiếm <0.1% pixel) |
| **Focal** | Nhân `(1-p)^γ` → giảm trọng số example dễ, tập trung example khó |

### 5.4. Optimizer + LR schedule
```
AdamW (lr=3e-4, weight_decay=1e-4)
  ├─ Warmup 5% step (0 → 3e-4)
  ├─ Cosine decay (3e-4 → 0 over 75% epochs)
  └─ SWA (Stochastic Weight Averaging) last 25% epochs
       averaged_weights = mean(weights at epoch 151..200)
       → flatter loss landscape, better generalization
```

### 5.5. Augmentation (training only)
- Random horizontal flip
- Random rotation ±15°
- Random intensity jitter (HU shift ±10%)
- Random elastic deformation (sim biến dạng giải phẫu)
- Random gaussian noise

### 5.6. Tricks nâng cao

| Trick | Tác dụng |
|---|---|
| **Mixed precision (AMP fp16)** | 2× nhanh hơn, half VRAM |
| **Snapshot ensemble** | Lưu top-3 checkpoint theo val_dice (E58, E85, E30) |
| **TTA (Test-Time Augmentation)** | Predict 4× (original + hflip + vflip + both) → average |
| **`persistent_workers=True`** | DataLoader workers không reset mỗi epoch |
| **`pin_memory=True`** | DMA transfer nhanh hơn lên GPU |

### 5.7. Hyperparams chính
| Param | Value |
|---|---|
| Epochs | 200 |
| Batch size | 10 (V100 32GB) |
| Image size | 512×512 |
| Buffer slices | ±3 quanh slice có nodule |
| Mask mode | union (≥1/4 radiologist) |

### 5.8. Training time
- ~1 ngày trên V100 16GB thuê (~$0.50-1.00/h)

---

## 6. Malignancy Classifier — `10_train_malignancy.py`

### 6.1. Tại sao cần?
Segmentation chỉ nói "có nốt", **không** nói "nốt này có ác tính không". LIDC có sẵn label malignancy 1-5 từ 4 radiologist, dùng để train classifier riêng.

### 6.2. Kiến trúc — DenseNet121-3D

```
Input: 32×32×32 voxel patch quanh centroid nodule
       (HU normalized → [0, 1] với HU_LO=-1024, HU_HI=600)
            ↓
   DenseNet121 (3D version từ MONAI):
     - Dense blocks: layer i nối với mọi layer trước đó
     - Growth rate k=12-32 channels
     - Compactness (gradient flow tốt hơn ResNet)
     - 11.2M params (3D)
            ↓
   Global Average Pool 3D → Linear(5)
            ↓
   Softmax → P(class 1..5)
```

### 6.3. Patch extraction — `09_extract_patches.py`
```python
for entry in nodule_meta:
    if not entry["malignancy"]: continue
    # Lấy mask của 1 radiologist cho 1 nodule cụ thể
    mask = mask_per_rad[slice_idx, rad_channel]
    centroid = centroid_3d(mask_across_slices)
    patch = images[centroid_z ± 16, centroid_y ± 16, centroid_x ± 16]
    pad_with(-1024)  # air fill nếu gần biên
    save(patch, malignancy_int)
```

| Thông số | Giá trị |
|---|---|
| Tổng patches | **6850** |
| Distribution | 1: 15% · 2: 23% · 3: 38% · 4: 14% · 5: 10% |

→ Lệch về class 3 ("indeterminate"). Cần class weights khi train.

### 6.4. Loss + tricks
- **CrossEntropyLoss với class weights** (inverse frequency)
- **Label smoothing 0.05**
- **AdamW + warmup + cosine**, 60 epochs
- **AMP fp16**
- Augment: random flip 3 axes + intensity jitter + gaussian noise

### 6.5. Kết quả

| Metric | Value | Note |
|---|---|---|
| Best val balanced accuracy | **0.4593** | Random = 0.20 (5-class) |
| Val susp-F1 (class ≥ 4) | **0.6542** | Binary "suspicious" tốt |
| Test bal-acc | 0.3962 | Generalization gap |
| Test susp-F1 | 0.5041 | |
| Train time | **~5 phút** trên RTX 4090 |

### 6.6. Tại sao bal_acc thấp?
- 1200 nodule là dataset rất nhỏ cho 5-class
- Distribution lệch nặng về class 3
- Model học "binary suspicious" tốt hơn fine-grained 5-class

---

## 7. Lung Segmentation — Lungmask R231 (pretrained)

### Vấn đề
HU threshold tự viết (`HU < -320 inside body`) **rò rỉ** vào ruột/dạ dày trên CT toàn thân → nốt giả hiển thị "outside lung" trong 3D viewer.

### Giải pháp — pretrained UNet
- **Lungmask R231** ([Hofmanninger et al. 2020](https://github.com/JoHof/lungmask), Đại học Vienna)
- U-Net 2D train trên **231 CT phổi đa-bệnh viện**
- Public weights, ~120 MB, auto-download khi `pip install lungmask`
- Chính xác **99%+**, infer 2-3s/volume trên RTX 4070
- **Không cần train**

### Tích hợp
```python
# predict.py
inferer = LMInferer(modelname="R231", force_cpu=False)
lung_mask = (inferer.apply(volume_hu) > 0).astype(np.uint8)
pred_in_lung = pred_mask & lung_mask  # → loại 95% FP toàn thân
```

---

## 8. Hậu xử lý 3D — `07_detect_3d.py`

```python
# 1. Slice-by-slice inference (with TTA)
prob_volume = sigmoid(model(every_slice))  # [N, H, W] float32

# 2. Threshold + lung intersection
mask_3d = (prob_volume > 0.5)
mask_3d = mask_3d & lung_mask

# 3. Connected components 3D (26-connectivity)
labeled, n_blobs = scipy.ndimage.label(
    mask_3d, structure=ones(3,3,3)
)

# 4. Filter + measure each blob
nodules = []
for blob_id in range(1, n_blobs + 1):
    voxels = (labeled == blob_id).sum()
    if voxels < 60:           # < ~4.5mm = noise
        continue
    coords = np.argwhere(labeled == blob_id)
    centroid = coords.mean(axis=0)
    bbox = (coords.min(0), coords.max(0))

    # Compute physical measurements
    volume_mm3 = voxels * Δz * Δy * Δx       # voxel size từ DICOM
    diameter_mm = 2 * ((3 * volume) / (4π))^(1/3)  # equivalent sphere

    nodules.append({...})
```

---

## 9. Render 3D — `08_render_3d.py`

### 9.1. Marching Cubes Algorithm
Convert binary mask 3D → mesh surface (triangles):

```
Input: 3D binary array [N, H, W]
Algorithm:
  1. Slide cube 2×2×2 voxel through volume
  2. 256 cube configurations (precomputed table)
  3. Each config → triangle vertices via interpolation
Output: vertices[V, 3] + faces[F, 3]
```

→ Mesh có thể export **STL** (mở Blender/MeshLab) hoặc render Plotly Mesh3d.

### 9.2. Plotly Mesh3d (interactive 3D)
- HTML interactive (xoay/zoom/click legend)
- Backend Three.js (WebGL)
- Lung shell **trong suốt 10%** + nodule màu đặc 85%
- Top 30 nodule lớn nhất render mesh (tránh quá nặng)
- Mỗi nodule một màu khác (palette warm: red → orange → magenta)

---

## 10. Clinical Decision Support — `clinical.py`

### 10.1. Brock Model — McWilliams NEJM 2013

**Logistic regression** train trên 2961 bệnh nhân Pan-Canadian Early Detection Study. Coefficients **đã công bố**, mình chỉ plug-in (không cần train):

```python
log_odds = -6.7892
         + 0.0287 × (age - 62)
         + 0.6011 × (sex == female)
         + 0.2961 × family_hx
         + 0.2953 × emphysema
         - 5.3854 × ((size_mm/10)^-0.5 - 1.581)   # Box-Cox transform
         - 0.1466 × type_code  (solid=0, non-solid=1, part-solid=2)
         + 0.6581 × upper_lobe
         - 0.0824 × nodule_count
         + 0.7729 × spiculated
P(cancer in 4 years) = sigmoid(log_odds)
```

**Validated AUC 0.970** trong PanCan study. Source: [NEJM 2013;369:910-919](https://www.nejm.org/doi/10.1056/NEJMoa1214726).

### 10.2. USPSTF 2021 (rule-based)
Đủ điều kiện tầm soát LDCT (low-dose CT) hằng năm nếu **TẤT CẢ**:
- 50 ≤ age ≤ 80
- pack-years ≥ 20
- Đang hút HOẶC bỏ < 15 năm

### 10.3. Lung-RADS v2022 (ACR)
| Diameter solid | Category | Action |
|---|---|---|
| < 6mm | LR2 | Bỏ qua, không follow-up |
| 6-8mm | LR3 | Theo dõi 6 tháng |
| 8-15mm | LR4A | Suspicious — chụp lại 3 tháng |
| 15-30mm | LR4B | Very suspicious — PET-CT/biopsy |
| > 30mm | LR4X | Cực kỳ suspicious — biopsy |

### 10.4. Diagnosis engine
6 mức `action_band`: **urgent / soon / routine / none**

Logic:
```
if hemoptysis + actionable_nodule:
    → URGENT  "Khám chuyên khoa TRONG 2 TUẦN" (NICE NG12)
elif max_diameter ≥ 30mm OR Brock ≥ 30%:
    → URGENT  "PET-CT + sinh thiết"
elif max_diameter ≥ 15mm OR Brock ≥ 10%:
    → SOON    "PET-CT + chụp lại 3 tháng"
elif max_diameter ≥ 8mm OR Brock ≥ 5%:
    → SOON    "Chụp lại CT phổi sau 3 tháng"
elif max_diameter ≥ 6mm:
    → ROUTINE "Chụp lại CT sau 6-12 tháng" (Lung-RADS 3)
elif nodules exist:
    → NONE    "Không cần follow-up" (chỉ < 6mm)
else:
    → NONE    "Phổi sạch"
+ Append USPSTF eligibility nếu đủ điều kiện
```

### 10.5. Output diagnosis
```
{
  summary: "3 cần can thiệp, 2 theo dõi, 5 bỏ qua"
  action_title: "PET-CT + sinh thiết"
  action_band: "urgent"
  action: <full paragraph khuyến nghị>
  nodule_buckets: { actionable: 3, monitor: 2, incidental: 5 }
  risk_factors: [
    {factor: "Tuổi 67 (>65)", weight: "medium"},
    {factor: "40 gói-năm (>30)", weight: "high"},
    {factor: "COPD", weight: "medium"},
    ...
  ]
  index_nodule_id: 8
  index_reason: "đường kính 18.2mm · Brock 24.1% · AI nghi 87% · ở thuỳ trên"
}
```

---

## 11. Webapp Architecture

### 11.1. Backend — `src/webapp/`
```
FastAPI + Uvicorn (port 8081)
├── app.py           routes:
│                      GET    /api/health
│                      GET    /api/cases
│                      POST   /api/analyze       (multipart)
│                      GET    /api/events/{id}   (Server-Sent Events)
│                      GET    /api/case/{id}
│                      DELETE /api/case/{id}
│                      + CORSMiddleware (allow http://localhost:3000)
├── predict.py       3 model loading + inference pipeline
└── clinical.py      Brock + USPSTF + Lung-RADS + diagnose()
```

### 11.2. Frontend — `frontend/`
```
Next.js 16 + TypeScript + Tailwind + shadcn/ui (port 3000)
├── src/app/page.tsx              Home (upload + history)
├── src/app/case/[id]/page.tsx    Result view
├── src/lib/api.ts                fetch + XHR upload + EventSource
├── src/lib/types.ts              TS types khớp backend
└── src/components/
    ├── upload-form.tsx           3-step wizard (files / patient / symptoms)
    ├── diagnosis-card.tsx        Action title + buckets + risk factors
    ├── nodule-spotlight.tsx      Index nodule KPIs
    ├── findings-table.tsx        All nodules table
    ├── plot-3d.tsx               Iframe sandbox cho Plotly
    ├── topbar.tsx                Brand "Nodura" + sticky
    ├── patient-context-bar.tsx
    ├── verdict-banner.tsx        Fallback cho case cũ chưa có diagnosis
    ├── history-list.tsx
    └── ui/*                      shadcn primitives
```

### 11.3. Pipeline 1 case upload

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. User upload DICOM folder/zip (browser)                            │
│ 2. POST /api/analyze (multipart)                                     │
│    → server save files vào uploads/<case_id>/                        │
│    → create background task                                          │
│    → respond 200 {case_id, name, n_dicoms} ngay                      │
│ 3. Frontend mở EventSource /api/events/{case_id}                     │
│ 4. Background thread (asyncio.to_thread):                            │
│      a. Read DICOM (~2s)                                             │
│      b. Lungmask R231 inference (~3s, GPU)                           │
│      c. UNet++ B5 + TTA on every slice (~30-90s, GPU)                │
│      d. mask ∩ lung                                                  │
│      e. Connected components 3D, drop voxels < 60                    │
│      f. DenseNet121-3D classify each nodule (~0.1s/nodule)           │
│      g. Brock + USPSTF + symptom_concern + diagnose()                │
│      h. Plotly Mesh3d HTML render                                    │
│      i. Save results/<case_id>/{meta.json, 3d.html}                  │
│      → emit progress 0..100% qua PROGRESS dict                       │
│ 5. SSE stream về frontend, update progress bar + log                 │
│ 6. Khi pct=100 → frontend redirect /case/{id}                        │
│ 7. /case/{id} fetch /api/case/{id} → render full report              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 12. Tech Stack Summary

| Layer | Tools |
|---|---|
| **Data** | LIDC-IDRI · pydicom · h5py |
| **DL framework** | PyTorch 2.5 + CUDA 12.4 |
| **Models** | UNet++ (segmentation_models_pytorch) · DenseNet121 (MONAI) · lungmask R231 (3rd party) |
| **Loss** | DiceFocalLoss (MONAI) · CrossEntropy with class weights |
| **Train tricks** | AMP fp16 · warmup+cosine LR · SWA · top-K snapshot ensemble · TTA |
| **Eval metrics** | Dice (MONAI) · IoU · HD95 · per-nodule F1 (centroid match within 1.5 × GT diameter — lenient, not LUNA16 official `radius = diameter/2`) |
| **3D** | scipy.ndimage.label · scikit-image marching_cubes · trimesh · Plotly |
| **Clinical** | Brock NEJM 2013 · USPSTF 2021 · Lung-RADS ACR v2022 · NICE NG12 |
| **Backend** | FastAPI · Uvicorn · CORS middleware · Server-Sent Events |
| **Frontend** | Next.js 16 (App Router) · TypeScript · Tailwind 4 · shadcn/ui · lucide-react |
| **Infrastructure** | V100 thuê (training seg) · RTX 4090 thuê (training malignancy) · RTX 4070 local (webapp inference) |

---

## 13. Kết quả

### 13.1. Segmentation (UNet++ B5 + TTA)
| Metric | Value | Note |
|---|---|---|
| Test slice dice | **0.7502** | Average overlap |
| Per-slice dice median | 0.8559 | Nửa case rất tốt |
| Per-slice dice P75 | 0.9153 | |
| Per-slice dice P90 | 0.9459 | Top 10% xuất sắc |
| Test slice IoU | 0.6528 | Intersection over Union |
| Volume dice mean | 0.3529 | Thấp do FP nhiều |
| HD95 median | 132 mm | Boundary noise |
| **Recall nodule (lenient match, 1.5 × GT diameter — không phải LUNA16 chính thức)** | **97.5%** | Bắt được hầu hết nodule thật theo quy tắc này |
| Precision nodule | 7.6% | Nhiều FP nhỏ |

### 13.2. Malignancy (DenseNet121-3D)
| Metric | Value | Note |
|---|---|---|
| Best val balanced accuracy | **0.4593** | 5-class, random=0.20 |
| Val suspicious-F1 (binary 4-5) | **0.6542** | Tốt cho phân biệt suspicious |
| Test acc | 0.3801 | |
| Test bal-acc | 0.3962 | Generalization gap |
| Test susp-F1 | 0.5041 | |

### 13.3. Inference time (RTX 4070, 1 case ~80 slices)
| Step | Time |
|---|---|
| Read DICOM | 1-3s |
| Lungmask R231 | 3-5s |
| UNet++ B5 + TTA | 30-60s |
| Connected components | 1-2s |
| Malignancy classify | 0.1-0.5s |
| Brock + USPSTF + diagnose | < 0.1s |
| Plotly Mesh3d render | 5-10s |
| **Tổng** | **~40-90s** |

---

## 14. Hạn chế & Hướng phát triển

### Hạn chế hiện tại
- Dataset chỉ **1010 bệnh nhân** → generalize sang dataset khác cần fine-tune
- Malignancy classifier 5-class còn yếu (bal_acc 0.46) — cần thêm data
- **Không phân biệt được spiculation** tự động → Brock dùng default `spiculated=False`
- 3D rendering hơi nặng với CT toàn thân (>500 slices)
- Chưa multi-timepoint comparison (so sánh CT cũ vs mới)

### Có thể cải tiến
- **Train 3D UNet** thay 2.5D → context z-direction tốt hơn
- Thêm **classifier riêng cho spiculation, calcification, sphericity** → cải thiện Brock
- Dùng **pretrained foundation model** (TotalSegmentator, MedSAM, BiomedCLIP)
- **Anatomical localization** chi tiết: thuỳ trên/dưới/giữa, segment phế quản nào
- **Multi-timepoint comparison** — track growth giữa các lần chụp
- Export **DICOM SR (Structured Report)** để integrate PACS bệnh viện
- **Volume-level batched inference** (hiện đang per-slice → 60s; batched có thể xuống 15s)
- **Web Worker** cho frontend Plotly render (đỡ block UI)

### Để đem vào lâm sàng thật cần thêm
- Validation trên **multi-center dataset** (không chỉ LIDC)
- IRB/ethics approval
- DICOM compliance
- HIPAA/GDPR compliance cho data bệnh nhân
- Liability insurance
- FDA / EU MDR clearance (nếu deploy thật)

---

## 15. Tham khảo

- **LIDC-IDRI Dataset:** Armato SG III et al. *Med Phys* 2011
- **UNet++:** Zhou Z et al. *DLMIA 2018*
- **EfficientNet:** Tan M, Le Q. *ICML 2019*
- **DenseNet:** Huang G et al. *CVPR 2017*
- **Brock model:** McWilliams A et al. *NEJM 2013;369:910-919*
- **USPSTF Lung Cancer Screening:** *JAMA 2021;325(10):962-970*
- **Lung-RADS v2022:** American College of Radiology
- **NICE NG12:** Suspected cancer recognition and referral
- **Lungmask R231:** Hofmanninger J et al. *Eur Radiol Exp* 2020
- **MONAI:** Medical Open Network for AI

---

**Repo:** https://github.com/abcxyzawe/lidc-lung-nodule-ai
**License:** MIT (đồ án học tập)
