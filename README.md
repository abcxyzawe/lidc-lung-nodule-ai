# LIDC-IDRI Lung Nodule AI — End-to-End Pipeline + Webapp

Đồ án phát hiện nodule phổi từ ảnh CT (LIDC-IDRI dataset).
Bao gồm full pipeline (validate → preprocess → train → eval → 3D detect → render) **và webapp FastAPI** để demo trực tiếp với GPU local.

**Kết quả:** Test slice Dice **0.7502** (TTA) · Per-slice median 0.8559 · Recall nodule 97.5%.

---

## 1. Cài đặt

### Yêu cầu
- Python ≥ 3.10
- NVIDIA GPU với CUDA ≥ 12.4 (đã test trên RTX 4070 và V100). CPU cũng chạy được nhưng inference chậm hơn 30-50×.
- ~150 GB ổ cứng nếu chạy lại pipeline từ đầu (DICOM gốc), ~3 GB nếu chỉ chạy webapp + checkpoint.

### Bước cài
```bash
git clone <repo-url>
cd phan-tich-ung-thu

# 1. PyTorch — cài đúng version cho CUDA của bạn
# CUDA 12.4 (khuyến nghị):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 2. Còn lại
pip install -r requirements.txt
```

### Lấy dataset
- Tải LIDC-IDRI từ [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) (~125 GB DICOM).
- XML annotations (4 radiologist contours): tải kèm hoặc từ [tcia-lidc-xml](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).
- Đặt vào cấu trúc:
  ```
  E:/Phan Tich Ung Thu/
  ├── manifest-1600709154662/LIDC-IDRI/...   (DICOM)
  └── tcia-lidc-xml/...                       (XML)
  ```
- Hoặc chỉnh `LIDC_ROOT` trong env: `export LIDC_ROOT=/đường/dẫn/khác`.

### Lấy checkpoint đã train sẵn (cho webapp)
Trained checkpoints không nằm trong git (~1.2 GB). Hỏi người trong team file `work/runs/best.pt` (hoặc `swa.pt`).
Để vào `work/runs/best.pt` là chạy webapp được ngay.

---

## 2. Cấu trúc project

```
phan-tich-ung-thu/
├── README.md                  bạn đang đọc
├── requirements.txt
├── src/
│   ├── configs.py             paths + hyperparameters chung
│   ├── utils.py
│   ├── dataset.py             PyTorch Dataset (importable)
│   ├── 01_validate.py         verify DICOM ↔ XML mapping
│   ├── 02_preprocess.py       DICOM → HDF5 (multiprocess)
│   ├── 03_split.py            train/val/test split 8/1/1
│   ├── 05_train.py            UNet++ + EfficientNet-B5 + SCSE
│   ├── 06_evaluate.py         test metrics + TTA + threshold sweep
│   ├── 07_detect_3d.py        inference → 3D mask + nodule list
│   ├── 08_render_3d.py        Plotly HTML + STL + PNG
│   └── webapp/                FastAPI demo
│       ├── app.py             server + endpoints
│       ├── predict.py         lung seg + AI inference + 3D render
│       ├── templates/         index.html (upload), result.html (3D)
│       └── static/style.css
├── work/                      [gitignore] generated artifacts
│   ├── preprocessed/<pid>.h5
│   ├── splits.json
│   └── runs/                  checkpoints + logs
└── outputs/                   [gitignore] step 07-08 outputs
```

---

## 3. Chạy webapp (cách nhanh nhất để demo)

```bash
cd src/webapp
python app.py
# Mở browser: http://127.0.0.1:8081
```

**Cách dùng:**
1. Click **"📁 Chọn folder DICOM"** → chọn folder của 1 bệnh nhân (ví dụ `manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0001/.../...`).
2. Hoặc click **"🗜️ Chọn file .zip"** → upload file zip chứa folder DICOM.
3. Bấm **"Phân tích bằng AI"** → progress bar live (upload → đọc DICOM → lung seg → AI → 3D render).
4. Tự động chuyển sang trang kết quả: 3D Plotly có vỏ phổi trong suốt + nodule màu, bảng nodule chi tiết, timing từng bước.

**Hiệu năng (RTX 4070):**
- Đọc DICOM ~1-3s
- Lung segmentation ~3-5s (HU threshold + scipy)
- AI inference ~30-90s (~0.1-0.2s/slice với TTA 4×)
- Render 3D mesh ~5-10s
- Tổng: ~40-110s tuỳ kích thước CT

**Lưu ý:**
- Webapp chỉ hiện top-30 nodule lớn nhất ra 3D (tránh render quá nặng), bảng full vẫn liệt kê hết.
- Pred được lọc giao với vùng phổi → bỏ FP toàn thân.
- Threshold mặc định 0.5, MIN_NODULE_VOXELS=30 (~3.5mm). Đổi trong `src/configs.py` nếu cần.

---

## 4. Chạy full pipeline (training + eval)

### Local hoặc remote GPU

```bash
cd src
python 01_validate.py        # 15-17 phút — index toàn bộ DICOM
python 02_preprocess.py      # 2-4 giờ — convert DICOM → HDF5
python 03_split.py           # instant
python 05_train.py           # 5-12 giờ V100 (200 epochs, B5)
python 06_evaluate.py --ckpt ../work/runs/best.pt
python 07_detect_3d.py --top 12 --ckpt ../work/runs/best.pt
python 08_render_3d.py       # → outputs/index.html
```

### Override path khi chạy trên server khác
```bash
LIDC_ROOT=/workspace python 05_train.py --root /workspace/preprocessed --splits /workspace/splits.json
```

---

## 5. Hyperparameters chính (xem `src/configs.py`)

| Tham số | Default | Mô tả |
|---|---|---|
| Encoder | `efficientnet-b5` | ImageNet pretrained |
| Loss | DiceFocalLoss | dice_w=1, focal_w=0.5, gamma=2 |
| Batch size | 10 | V100 32GB OK |
| LR | 3e-4 | AdamW + warmup 5% + cosine decay |
| Epochs | 200 | + SWA last 25% |
| Mask mode | `union` | ≥1/4 radiologist đánh dấu |
| Min mask px | 10 | bỏ point markers <10 px |
| Buffer slices | ±3 | giữ slice quanh nodule |
| Threshold | 0.5 | sweep tự động trong evaluate |
| TTA | true | hflip + vflip + both |
| Min nodule voxels (3D) | 30 | drop blob nhỏ |

---

## 6. Output 3D (từ pipeline)

Mỗi `outputs/<patient>/<series>/`:
- `report.html` — tổng hợp series (mở trước)
- `3d_interactive.html` — Plotly Mesh3d
- `6views.png` — 6 góc 3D collage
- `nodule_NN.stl` — STL từng khối u (Blender/MeshLab)
- `nodule_NN_slices.png` — axial/coronal/sagittal qua tâm nodule

---

## 7. Lưu ý quan trọng

⚠️ **AI này KHÔNG chẩn đoán ung thư.** Chỉ làm segmentation (tìm + đo nodule). Để biết ác/lành tính cần thêm classification model + bệnh sử + theo dõi growth + sinh thiết.

⚠️ **Recall cao (97.5%) nhưng precision thấp** trên dataset này — model bắt nhiều FP nhỏ (mạch máu, phế quản, sẹo). Dùng đường kính ≥ 4mm theo Lung-RADS để lọc clinically relevant.

⚠️ Dataset chỉ có 1010 bệnh nhân, generalization sang dataset thật cần fine-tune lại.

---

## 8. License & Citation

Dataset: LIDC-IDRI © The Cancer Imaging Archive (TCIA), CC-BY 3.0.
Code: MIT (đồ án học tập).
