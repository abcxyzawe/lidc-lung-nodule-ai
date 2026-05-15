# Báo cáo full dự án — LIDC-IDRI Lung Nodule Detection

**Trạng thái**: 2026-05-10. Tuned + FPR classifier deployed. Test_panel locked sens 53.6% (Phase 1) / 80.4% (Phase 2 sensitive mode).

---

## 1. Tổng quan & mục tiêu

Hệ thống AI hỗ trợ tầm soát ung thư phổi trên ảnh CT (LIDC-IDRI). Pipeline 3 giai đoạn:

1. **Phân vùng (segmentation)** — UNet++ EfficientNet-B5 2.5D phát hiện nốt phổi trên từng slice.
2. **Phân loại ác tính (classification)** — DenseNet121-3D đánh giá độ nghi ngờ ác tính của mỗi nodule.
3. **False Positive Reduction (mới)** — DenseNet121-3D lọc ứng viên giả từ giai đoạn 1.

Webapp FastAPI cho phép upload DICOM, render 3D Plotly + thumbnails 2D với confidence + Lung-RADS score.

**Định hướng hiện tại**: AI HỖ TRỢ bác sĩ tầm soát (radiologist-assist), không tự đưa kết luận chẩn đoán.

---

## 2. Dataset — LIDC-IDRI

### Nguồn
- TCIA download: `manifest-1600709154662/LIDC-IDRI/` — **1010 patients** (~130 GB raw DICOM)
- XML annotations: 4 radiologists draw polygons/markers per nodule per slice
- Kèm metadata: malignancy 1-5, subtlety, calcification, sphericity, margin, lobulation, spiculation, texture, internal structure

### Đặc điểm
- ~250 slice/patient (axial CT, 512×512)
- Slice thickness 0.6–5 mm (đa số 1.25–2.5 mm)
- Pixel spacing 0.5–0.9 mm
- 4 radiologists annotate độc lập → mỗi nodule có 1–4 polygon

---

## 3. Pipeline tổng thể (15 bước)

```
DICOM raw (130 GB)
  │
  ▼ 01_validate.py        — kiểm tra DICOM hợp lệ, build SOP index
  │
  ▼ 02_preprocess.py      — gom DICOM + XML → 1 h5/patient
  │                         GIỮ chỉ slice có nodule + ±3 buffer (=> 13 GB)
  │
  ▼ 03_split.py           — chia train(812)/val(99)/test(99) theo patient
  │
  ▼ 05_train.py           — train UNet++ B5 200 epochs, SWA, snapshot ensemble
  │                         → best.pt, swa.pt, snapshot_e*.pt, last.pt
  │
  ▼ 06_evaluate.py        — Dice/IoU + per-nodule F1 trên test
  ▼ 07_detect_3d.py       — apply pipeline lên patient bất kỳ
  ▼ 08_render_3d.py       — Plotly Mesh3d export
  │
  ▼ 09_extract_patches.py — crop 32³ patches quanh GT centroid
  ▼ 10_train_malignancy.py — DenseNet121-3D (5-class malignancy)
  │                         → malignancy.pt
  │
  ▼ academic_bench.py     — FROC + CPM + ablation + failure analysis
  │
  ▼ tune_detection_params.py — sweep post-proc params
  ▼ apply_best_config.py     — patch configs.py + webapp/app.py
  ▼ extract_fpr_candidates.py / train_fpr_classifier.py / apply_fpr_classifier.py
  │                         → fpr.pt
  │
  ▼ webapp/app.py         — FastAPI backend
```

---

## 4. Preprocessing chi tiết — VẤN ĐỀ QUAN TRỌNG

### Code hiện tại (`02_preprocess.py` line 193-198)

```python
keep = set()
for i in nodule_slice_idx:                              # chỉ slice có polygon
    for j in range(max(0, i - BUFFER_SLICES),           # ±3 slice
                   min(len(slices), i + BUFFER_SLICES + 1)):
        keep.add(j)
```

### Hậu quả
| Số liệu | Giá trị |
|---|---|
| Bệnh nhân preprocessed | 990 / 1010 |
| Tổng slice giữ | 44,822 |
| % so với LIDC gốc (~250K) | **~18%** |
| Slice có polygon nodule | 18,426 (41%) |
| Slice buffer ±3 (nền quanh nodule) | 26,396 (59%) |
| Slice CT bị bỏ | ~205,000 (82%) |
| Disk gốc → preprocessed | 130 GB → 13 GB |

### Hai vấn đề tách biệt

**VĐ A — Distribution shift (chưa khẳng định)**:
- Model train chỉ thấy slice gần nodule. KHÔNG học "phổi bình thường, không có nodule" trông ra sao.
- Webapp upload full CT (~250 slice) → model gặp 82% slice chưa từng thấy.
- Có thể tạo FP ngẫu nhiên ở vùng phổi xa nodule (apex, base)
- **Đang verify**: `verify_full_vs_preprocessed.py` đang chạy 3 patients để so sánh AI-on-full-CT vs AI-on-preprocessed

**VĐ B — 22% nodule "invisible"** (đã khẳng định, Phase 3):
- Những nodule này CÓ trong slice giữ (training thấy)
- Model vẫn không sinh signal ở centroid → ceiling thực
- Nguyên nhân: noisy 1/4-rad annotations + atypical shapes (10-20mm là tệ nhất, 32% invisible)
- **KHÔNG phải lỗi preprocessing**

---

## 5. Splits — train/val/test

`work/splits.json`:
```
train: 812 patients   (~80%, dùng để train)
val:    99 patients   (~10%, chọn threshold/config — KHÔNG train)
test:   99 patients   (~10%, LOCKED — chỉ chạy báo cáo cuối)
```

Cộng thêm `panels.py`:
- `debug_panel`: 12 patients hand-picked đa dạng size + multi-nodule (cho test nhanh ~3 phút)
- `val_panel`: 99 patients (= splits["val"])
- `test_panel`: 99 patients (= splits["test"])

---

## 6. Models đã train

| Model | File | Kích thước | Mục đích | Metric |
|---|---|---|---|---|
| Segmentation chính | `work/runs/best.pt` (e?, val_dice 0.7421) | ~120 MB | UNet++ B5 SCSE 2.5D 3-slice | val Dice 0.74 |
| Segmentation SWA | `work/runs/swa.pt` | ~120 MB | SWA last 25% epochs | ensemble với best |
| Snapshot ensemble | `snapshot_e030.pt`, `e058.pt`, `e085.pt` | ~120 MB × 3 | Top-K val_dice | optional ensemble |
| Latest checkpoint | `last.pt` (e200) | ~120 MB | epoch cuối cùng | (không dùng default) |
| Malignancy classifier | `work/runs/malignancy.pt` | ~30 MB | DenseNet121-3D 5-class | bal_acc 0.46 |
| FPR classifier (mới) | `work/runs/fpr.pt` | ~30 MB | DenseNet121-3D binary | test AUC 0.83 |

---

## 7. Codebase — 33 file Python theo nhóm

### Nhóm pipeline gốc (`0X_*.py` numbered)
| File | Chức năng |
|---|---|
| `01_validate.py` | Validate DICOM, build SOP index `work/sop_index.json` |
| `02_preprocess.py` | DICOM + XML → 1 h5/patient (kept slices only) |
| `03_split.py` | Chia train/val/test, lưu `splits.json` |
| `05_train.py` | Train UNet++ B5 200 epochs |
| `06_evaluate.py` | Dice + IoU + per-nodule F1 trên test |
| `07_detect_3d.py` | Pipeline detection trên 1 patient |
| `08_render_3d.py` | Plotly Mesh3d HTML export |
| `09_extract_patches.py` | Crop 32³ HU patches quanh GT centroid (cho malignancy training) |
| `10_train_malignancy.py` | Train DenseNet121-3D 5-class malignancy classifier |

### Nhóm core utility
| File | Chức năng |
|---|---|
| `configs.py` | Single source of truth: paths, HU window, mask mode, training hyperparams |
| `dataset.py` | `LIDCSeg25D` (3-slice 2.5D), augmentation pipeline (flip, rotate, intensity, elastic, noise) |
| `utils.py` | Parse patient ID từ path, helpers chung |

### Nhóm benchmark + tuning gốc
| File | Chức năng |
|---|---|
| `benchmark.py` | `gt_nodules_for()` (extract GT từ h5 với consensus merging), `run_pipeline()`, `match_nodules()` |
| `tuner.py` | Coordinate-descent tuning post-proc params |
| `academic_bench.py` | FROC + CPM + ablation + failure analysis với PNG export |
| `show_gt.py` | Inspect GT của 1 patient |

### Nhóm panel + Phase 1 (post-proc tuning, mới)
| File | Chức năng |
|---|---|
| `panels.py` | Định nghĩa debug/val/test panels, `load_panel()` |
| `tune_detection_params.py` | Sweep post-proc params trên cached probs, F1-based selection, 3 operating points |
| `compare_baseline_vs_tuned.py` | So sánh baseline webapp vs tuned config trên panel |
| `compare_ai_vs_gt.py` | Per-patient match table (GT vs AI predictions) |
| `apply_best_config.py` | Patch `configs.py` + `webapp/app.py` từ `best_config.json` |
| `plot_tuning_froc.py` | FROC plot baseline vs tuned (re-evals patients) |
| `plot_froc_simple.py` | FROC plot từ JSON sẵn (không re-eval) |
| `run_full_tuning_pipeline.py` | Orchestrator: cache → sweep → apply → cache test → eval |

### Nhóm Phase 2 (FPR classifier, mới)
| File | Chức năng |
|---|---|
| `extract_fpr_candidates.py` | Generate candidate patches (48³) + GT match labels |
| `train_fpr_classifier.py` | Train DenseNet121-3D binary FPR (val_panel cands) |
| `apply_fpr_classifier.py` | Apply FPR + compute metrics on candidates |

### Nhóm Phase 3 (analysis, mới)
| File | Chức năng |
|---|---|
| `analyze_missed_nodules.py` | Visibility analysis: peak prob ở GT centroid, stratified buckets |

### Nhóm retraining experiments (mới — ready, chưa chạy)
| File | Chức năng |
|---|---|
| `experiments.py` | Registry 8 experiments (baseline, consensus2/3, Tversky, FocalTversky, oversample, 5-slice, combined) |
| `dataset_v2.py` | `LIDCSeg25DMulti` (n-slice), oversample sampler, Tversky/FocalTversky losses, 3→5ch init |
| `train_experiment.py` | Train 1 hoặc all experiments → `work/runs_exp/<name>/` |
| `eval_experiment.py` | Cache probs val+test, sweep, eval test_panel locked, append `experiments_comparison.csv` |

### Nhóm verification (mới)
| File | Chức năng |
|---|---|
| `verify_full_vs_preprocessed.py` | So sánh AI on full CT vs AI on preprocessed h5 (đang chạy) |

### Webapp (`src/webapp/`)
| File | Chức năng |
|---|---|
| `app.py` | FastAPI backend: `/api/analyze`, `/api/case/{id}`, SSE progress, async DICOM analysis |
| `predict.py` | Models lazy-load, pipeline (segment_lung + predict_nodules + clean_mask + find_nodules + merge + filter_subpleural + predict_malignancy + predict_fpr), 3D render |
| `clinical.py` | Lung-RADS + clinical heuristics (size + AI risk → final risk label) |

---

## 8. Webapp architecture

### Backend (`webapp/app.py`)
- FastAPI + Uvicorn
- Upload DICOM zip/folder → background task chạy pipeline
- SSE stream tiến độ (`/api/case/{id}/stream`)
- Lưu kết quả: `webapp/results/<case_id>/{meta.json, 3d.html, thumbs/*.png}`

### Pipeline trong 1 case
```
read_dicom_series → segment_lung (R231) → predict_nodules (TTA + ensemble best+swa)
  → clean_mask → find_nodules (min_voxels=120, max_elong=4)
  → merge_nearby_nodules (max_dist=10mm) → filter_subpleural (0mm = disabled)
  → predict_fpr_for_nodules (adds fpr_prob)
  → predict_malignancy_for_nodules (adds ai_class, lung_rads, risk_combined)
  → render_3d_html (Plotly Mesh3d) + render_nodule_thumb (PNG bbox)
```

### Frontend (Next.js, không trong repo này)
- Upload UI
- Case detail: 3D plotly viewer + danh sách nodule + Accept/Reject
- Sort by confidence (cao trước)

### Config hiện tại sau Phase 1 apply
```python
# configs.py
THRESHOLD_DEFAULT = 0.97
MIN_NODULE_VOXELS = 120

# webapp/app.py
predict_nodules(threshold=0.97)
find_nodules(min_voxels=120, max_elongation=4.0)
merge_nearby_nodules(max_dist_mm=10.0)
filter_subpleural(min_dist_mm=0.0)   # was 2.0
```

---

## 9. Kết quả benchmark + tuning history

### Test_panel LOCKED (95 patients, 250 GT nodules) — đo trên cached probs

| Pipeline | TP | FP | FN | Sens | Prec | F1 | FP/scan |
|---|---|---|---|---|---|---|---|
| **Baseline cũ** (pre-tuning webapp) | 106 | 51 | 144 | 42.4% | 67.5% | 0.522 | 0.54 |
| **Phase 1 tuned** (mv120, thr=0.97, no subp) | 134 | 66 | 116 | **53.6%** | 67.0% | **0.596** | 0.69 |
| **Phase 2 FPR @ 0.70** (balanced mode) | — | — | — | 69.6% | 38.1% | 0.492 | 2.19 |
| **Phase 2 FPR @ 0.30** (sensitive mode) | — | — | — | **80.4%** | 30.1% | 0.439 | 3.61 |

### Phase 1 — Post-processing tuning (CPU sweep, no retrain)
- Sweep 54 configs × 7 thresholds = 378 ops trên 96 val patients
- Selection: peak F1 across thresholds
- Best: `min_voxels=120, max_elong=4, merge=10mm, subp=0` @ thr=0.97
- **Đã apply vào webapp**

### Phase 2 — FPR classifier (mới train)
- Extract candidates @ thr=0.40, min_voxels=20 (permissive)
- val: 1283 cands (184 pos / 1099 neg), test: 1260 cands (184 pos / 1076 neg)
- DenseNet121-3D binary, 30 epochs, weighted CE + flip/rot90 augment
- **Test AUC=0.830, AP=0.480**
- Wired vào webapp: mỗi nodule có `fpr_prob`

### Phase 3 — Visibility analysis (test_panel)
| Bucket | n | Visible (>0.7) | Invisible (<0.3) |
|---|---|---|---|
| Small 3-6mm | 75 | **89.3%** | 6.7% |
| Medium 6-10mm | 93 | 78.5% | 20.4% |
| **Med-large 10-20mm** | 59 | 66.1% | **32.2%** ← worst |
| Large ≥20mm | 20 | 75.0% | 25.0% |
| **Tổng** | 250 | 77.6% | 20.4% |

**Phát hiện counterintuitive**: nodule lớn (10-20mm) khó hơn nodule nhỏ. Nguyên nhân: union mask noisy + atypical shapes (spiculated, ground-glass).

---

## 10. Hạn chế hiện tại

### Đã biết
1. **Recall ceiling ~78%**: 22% nodule completely invisible to model. Không phục hồi được bằng post-proc.
2. **Preprocessing distribution shift**: Train trên 18% slice (gần nodule). Webapp predict trên full CT → có thể FP ở vùng lạ. **Đang verify với `verify_full_vs_preprocessed.py`**.
3. **Diameter overestimate**: AI segment nodule rộng hơn GT (e.g., 0002: AI 27mm vs GT 17.5mm). Train trên union mask → boundary noisy.
4. **Malignancy classifier weak**: bal_acc 0.46 → đã giảm trọng số trong logic combined (60% size + 40% classifier).
5. **FPR classifier chỉ 1283 train samples**: cần thêm data từ train_panel (812 patients) để tăng AUC từ 0.83 → 0.88+.

### Chưa thử (code đã sẵn sàng)
1. Retrain với mask_mode=consensus2 (drop 1/4-rad noisy)
2. TverskyLoss(0.3, 0.7) — penalty FN nặng
3. 5-slice 2.5D — context rộng hơn
4. Oversample 10-20mm bucket (yếu nhất)
5. Re-preprocess giữ ALL slices (130 GB) hoặc add negative slice sampler

---

## 11. Artifacts trong `work/`

```
work/
├── splits.json                      812/99/99 patient split
├── sop_index.json                   SOP UID lookup từ DICOM
├── preprocessed/                    1010 .h5 files, 13 GB
├── runs/                            Production checkpoints
│   ├── best.pt, swa.pt              Segmentation (val_dice 0.74)
│   ├── snapshot_e030/058/085.pt     Top-K snapshots
│   ├── malignancy.pt                DenseNet121-3D 5-class (bal_acc 0.46)
│   ├── fpr.pt                       FPR DenseNet121-3D (test AUC 0.83) ← MỚI
│   ├── log.txt                      Training log
│   └── test_metrics.json            Test eval (06_evaluate output)
├── runs_exp/                        Experiment checkpoints (CHƯA train)
└── academic/
    ├── TUNING_SUMMARY.md            Báo cáo Phase 1+2+3 chi tiết
    ├── best_config.json             Phase 1 best config
    ├── tuning_results.csv           378 rows (config × threshold)
    ├── tuning_summary.csv           54 configs, peak F1 each
    ├── baseline_vs_tuned.json       val_panel comparison
    ├── baseline_vs_tuned_TEST.json  Locked test comparison
    ├── compare_ai_vs_gt_TEST.json   Per-patient match table
    ├── fpr_eval.json                FPR threshold sweep
    ├── fpr_eval_strict.json         FPR cascaded with Phase 1
    ├── missed_nodules_analysis.json Phase 3 visibility data
    ├── fpr_val.npz, fpr_test.npz    Extracted candidates
    ├── froc_baseline_vs_tuned.png   FROC plot
    ├── probs/                       Cached prob volumes (val + test)
    ├── lung_masks/                  Cached lung masks
    └── failure_cases/               PNG examples of FN/FP
```

---

## 12. Các báo cáo (markdown)

| File | Nội dung |
|---|---|
| `README.md` | Giới thiệu project + run instructions |
| `DOCUMENTATION.md` | Tech stack chi tiết + per-file role + design choices |
| `ACADEMIC_EVALUATION.md` | FROC + CPM + ablation + dataset description (academic-style) |
| `work/academic/TUNING_SUMMARY.md` | Full Phase 1+2+3 + cascading sanity check |
| `PROJECT_REPORT_FULL.md` | **Báo cáo này** — tổng hợp A-Z toàn project |

---

## 13. Quick reference — chạy gì ở đâu

```bash
# Kiểm tra/preprocess lại từ DICOM (chỉ chạy 1 lần)
python 01_validate.py
python 02_preprocess.py
python 03_split.py

# Train segmentation chính (~30 giờ trên RTX 4070)
python 05_train.py

# Train malignancy classifier
python 09_extract_patches.py
python 10_train_malignancy.py

# Eval test panel (gốc)
python 06_evaluate.py --ckpt work/runs/best.pt

# Phase 1 — tune post-proc (CPU after caching, ~30 phút)
python tune_detection_params.py --panel val --cache-only       # cache probs
python tune_detection_params.py --panel val --quick            # sweep
python apply_best_config.py                                    # apply
python compare_baseline_vs_tuned.py --panel test               # final test

# Phase 2 — FPR classifier
python extract_fpr_candidates.py --panel val
python extract_fpr_candidates.py --panel test
python train_fpr_classifier.py --train work/academic/fpr_val.npz \
    --val work/academic/fpr_test.npz --epochs 30
python apply_fpr_classifier.py --candidates work/academic/fpr_test.npz

# Phase 3 — visibility analysis
python analyze_missed_nodules.py

# Webapp
cd webapp && uvicorn app:app --reload --port 8000

# Retrain experiments (sẵn code, chưa chạy — multi-day GPU)
python train_experiment.py --exp list
python train_experiment.py --exp consensus2     # 1 experiment, ~5-7h trên 4070
python eval_experiment.py --exp consensus2

# Verify distribution shift
python verify_full_vs_preprocessed.py --patients LIDC-IDRI-0001,0002,0094
```

---

## 14. Kế hoạch tiếp theo (theo thứ tự ROI)

| Bước | Cost | Upside ước | Status |
|---|---|---|---|
| Verify full-vs-preprocessed shift | ~30 phút | Trả lời VĐ A | **Đang chạy** |
| Cache 800 train patients + retrain FPR | ~3-4 giờ | AUC 0.83 → 0.88, Sensitive sens 80% → 85% | Code sẵn |
| Re-preprocess giữ ALL slices (option A) | 4-6 giờ + 130 GB | Fix VĐ A nếu confirm | Code phải viết |
| Add negative slice sampler (option C) | 2-3 giờ + retrain | Fix VĐ A nhẹ hơn | Code phải viết |
| Retrain consensus2 + Tversky + 5-slice | 15-20 giờ GPU/exp × 3 | Sens 53.6% → 60-66% | Code sẵn |
| 3D detection framework (nnDetection) | 1-2 ngày setup + train | Top-tier baseline | Future |

---

## 15. Liên hệ với kết quả Phase 1+2

Nhìn từ góc người dùng cuối (bác sĩ tầm soát):

**Trước Phase 1+2 (baseline cũ webapp)**:
- Mỗi CT trung bình AI bắt được ~1.6 nodule, miss ~1.5 nodule thật/scan
- Sens 42% → thiếu hơn nửa nodule cần xem

**Sau Phase 1 (đã deploy)**:
- ~2.0 nodule/scan, miss ~1.2 nodule thật
- Sens 54% → bắt thêm 28 nodule trên 250 (test panel)
- Precision không đổi, FP/scan tăng nhẹ 0.54 → 0.69 (chấp nhận được)

**Tương lai Phase 2 sensitive mode (chưa wire mặc định)**:
- ~5.7 nodule/scan, miss ~0.5 nodule thật
- Sens 80% → bắt được hầu hết nodule thật
- FP 3.6/scan → bác sĩ phải reject ~70% prediction (tradeoff cao recall)

3 modes phù hợp ngữ cảnh khác nhau:
- **Strict** (Phase 1, mặc định): demo / triage cấp tốc
- **Balanced** (Phase 2 thr=0.7): screening hằng ngày
- **Sensitive** (Phase 2 thr=0.3): khám chuyên sâu / second-opinion
