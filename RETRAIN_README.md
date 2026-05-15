# RETRAIN README — V100 ops

Plan retrain chuẩn 9 bước (A→I), có stop condition + decision metric.

## Mục tiêu (3)
1. Giảm FP trên full CT — fix distribution shift
2. Tăng recall nhóm nodule invisible (22% ceiling, đặc biệt 10-20mm)
3. Protocol học thuật sạch — train trên train_panel, threshold trên val_panel, lock test_panel

## Files cần sync sang V100
- `src/` — toàn bộ code
- `work/runs/{best.pt, swa.pt, fpr.pt, malignancy.pt}` — current ckpts (init)
- `work/splits.json`, `sop_index.json`
- `manifest-1600709154662/` — 130 GB DICOM (chỉ cần nếu re-preprocess hoặc verify full CT)
- `tcia-lidc-xml/` — XML annotations (cần nếu re-preprocess v2)
- `work/preprocessed/` — 13 GB v1 data (dùng cho EXP02/EXP03 không cần v2)

---

## Phase A — Freeze baseline (~5 phút)

**Bắt buộc trước khi retrain.** Backup state hiện tại để rollback nếu retrain hỏng.

```bash
python freeze_baseline.py
# Output: work/baseline_frozen/<TIMESTAMP>/
#   - runs/{best,swa,fpr,malignancy,snapshot_e*}.pt
#   - src/configs.py + webapp/{app,predict}.py
#   - academic/{TUNING_SUMMARY.md, best_config.json, *.json, *.csv, *.png}
#   - PROJECT_REPORT_FULL.md, RETRAIN_README.md, ...
#   - MANIFEST.json (current metrics + restore instructions)
```

---

## Phase B — Verify distribution shift (~15-20 phút, GPU)

Đo bằng 10-15 patients, có FP-zone classification (apex/base/heart/mediastinum):

```bash
python verify_full_ct_drift.py --n-patients 15
# Output: work/academic/FULL_CT_DRIFT_VERIFY.md
```

**Decision rule** (script tự đánh giá):
- (a) ΔFP/scan_full vs preprocessed > 0.5 → trigger
- (b) >30% FP nằm trong apex/base/heart/mediastinum → trigger

**→ Nếu cả 2 không trigger**: SKIP negative slice sampler. Chỉ chạy EXP02/EXP03.
**→ Nếu trigger**: re-preprocess v2 + chạy EXP01/EXP04/EXP05.

---

## Phase C — Retrain FPR với protocol đúng (~3-5h V100)

**Đây là step ROI cao nhất** — fix protocol issue ngay với segmentation hiện tại, không chờ retrain seg.

```bash
python retrain_fpr_pipeline.py --seg-ckpt work/runs/best.pt --tag fpr_v2_baseline
# Tự động:
#   1. Cache probs train/val/test panels (~1-2h)
#   2. Extract candidates train_panel (812 patients ≈ 8x more data)
#   3. Train DenseNet121-3D 40 epochs
#   4. Pick threshold val_panel
#   5. Locked test_panel single eval
# Output: work/runs/fpr_v2_baseline.pt + work/academic/fpr_v2_baseline_eval_*.json
```

Mục tiêu: AUC ≥ 0.83 (cũ), Sensitive mode sens ≥ 80%, FP/scan ≤ 4.

Nếu FPR mới khá hơn → wire vào webapp + chốt 3 modes ngay (kể cả không retrain seg).

---

## Phase D — Re-preprocess v2 (CONDITIONAL, ~3-4h CPU)

**Chỉ chạy nếu Phase B trigger.**

```bash
python 02b_preprocess_with_negatives.py --neg-stride 4 --buffer 3
# Output: work/preprocessed_v2/ (~25 GB)
```

---

## Phase E — Pilot 6 experiments × 50 epochs (~25-30h V100)

Liệt kê:
```bash
python train_experiment.py --exp list
```

**Quan trọng**: KHÔNG gộp tất cả thay đổi vào EXP đầu — làm tuần tự để biết thành phần nào đóng góp.

```bash
python pilot_runner.py
# Auto train + eval + comparison table
# Output: work/academic/{PILOT_RESULTS.md, PILOT_PICK.json, experiments_comparison.csv}
```

Hoặc chạy lẻ, smoke test 5 epoch trước:
```bash
python pilot_runner.py --epochs 5 --only exp02_consensus2,exp03_cons2_tversky
```

---

## Phase F — Chọn winner (no GPU, manual)

Đọc `PILOT_RESULTS.md`. Chọn theo metric:

**Primary** (theo thứ tự ưu tiên):
1. CPM / Sens@1FP/scan / Sens@2FP / Sens@4FP
2. FP/scan trên full CT (`fullct_avg_extra_pred`)
3. Sensitivity nhóm 10-20mm (`sens_medium`)

**Secondary**:
- F1, Precision
- Invisible rate (`invisible_pct`)

**KHÔNG chọn bằng val_dice đơn thuần.**

Sau đó sửa `src/experiments.py::stage2_full` config để khớp pilot winner.

---

## Phase G — Train full winner (~15h V100)

```bash
python train_experiment.py --exp stage2_full --epochs 180
# stage2_full đã enable_swa=True → tự save best.pt + last.pt + swa.pt
python eval_experiment.py --exp stage2_full
```

(Optional) thử 5-slice biến thể nếu winner config dùng 3-slice và bạn còn budget:
```bash
# Sửa stage2_full_5slice config khớp winner trước
python train_experiment.py --exp stage2_full_5slice --epochs 180
python eval_experiment.py --exp stage2_full_5slice
```

---

## Phase H — Rebuild FPR với segmentation mới (~3-5h V100)

**Bắt buộc**: candidate distribution đã đổi vì seg model mới — KHÔNG dùng FPR cũ.

```bash
python retrain_fpr_pipeline.py \
    --seg-ckpt work/runs_exp/stage2_full/best.pt \
    --tag fpr_v3_stage2winner
```

---

## Phase I — Lock + finalize (~30 phút)

1. Chạy ĐÚNG 1 LẦN trên test_panel với (stage2_full + fpr_v3) — số cuối cùng để báo cáo.
2. Apply config mới vào webapp:
   ```bash
   # apply_best_config.py needs new config from stage2 eval
   # OR manually patch configs.py + webapp/app.py
   ```
3. Update `PROJECT_REPORT_FULL.md` + `TUNING_SUMMARY.md` với bảng:
   ```
   | Pipeline | Sens | Prec | F1 | FP/scan | sens 10-20mm | invisible% |
   |---|---|---|---|---|---|---|
   | Old baseline (frozen) | 42.4% | 67.5% | 0.522 | 0.54 | ? | ? |
   | Phase 1 strict (frozen) | 53.6% | 67.0% | 0.596 | 0.69 | 64.2% | 22.4% |
   | Phase 2 sensitive (frozen) | 80.4% | 30.1% | 0.439 | 3.61 | ? | ? |
   | Stage 2 + FPR v3 strict | ? | ? | ? | ≤1 | ? | ? |
   | Stage 2 + FPR v3 review | ~70-75% | ? | ? | ≤2.5 | ? | ? |
   | Stage 2 + FPR v3 screening | ≥80% | ? | ? | ≤4 | ? | ? |
   ```

---

## Operating modes mục tiêu sau retrain

| Mode | Sens | FP/scan | Use case |
|---|---|---|---|
| Precision | tự do | **≤ 1** | Demo / triage cấp tốc |
| Review | **70-75%** | ≤ 2.5 | Screening hằng ngày |
| Screening | **≥ 80-85%** | ≤ 4 | Khám chuyên sâu / second-opinion |

---

## Smoke test trước (~45 phút, V100)

Tránh phí GPU bằng cách verify pipeline trước:

```bash
# 1. Freeze (5 phút, không GPU)
python freeze_baseline.py

# 2. Verify (~15 phút, GPU)
python verify_full_ct_drift.py --n-patients 5

# 3. (Conditional) Re-preprocess subset (~15 phút)
python 02b_preprocess_with_negatives.py --workers 2  # Sẽ chạy lâu, dừng sau ~10 patients để test

# 4. Train smoke (5 epochs, ~25 phút)
python train_experiment.py --exp exp02_consensus2 --epochs 5
python eval_experiment.py --exp exp02_consensus2

# 5. FPR smoke (1 epoch FPR train, ~10 phút)
python retrain_fpr_pipeline.py --seg-ckpt work/runs/best.pt --tag fpr_smoke --epochs 1
```

Nếu pass mới chạy phases A-I.

---

## Time + cost budget tổng

| Phase | Time V100 | Disk delta |
|---|---|---|
| A. Freeze | 5 phút | +500 MB |
| B. Verify | 15-20 phút | +5 MB |
| C. FPR retrain (v2 baseline) | 3-5h | +5 GB |
| D. (Cond.) Re-preprocess v2 | 3-4h CPU | +25 GB |
| E. Pilot 6 exp × 50ep | 25-30h | +5 GB ckpts |
| F. Chọn winner | 0 (manual) | 0 |
| G. Stage 2 full | 15h | +1 GB |
| H. FPR rebuild | 3-5h | +5 GB |
| I. Finalize | 30 phút | +5 MB |
| **Tổng nếu Phase B trigger** | **~50-60h** | **+40 GB** |
| **Tổng nếu Phase B KHÔNG trigger** | **~25-30h** | **+12 GB** |

Wall-clock: **2-3 ngày** trên 1×V100, hoặc **1.5 ngày** trên 2×V100 (chạy 2 EXP song song trong Phase E).

## Máy thuê khuyến nghị

- **1×V100 32GB** (rqNRJBN3, ~5,500/h): kinh tế nhất, đủ
- **2×V100 32GB** (~11,000/h): nếu muốn pilot song song (rút Phase E xuống ~15h)
- **KHÔNG cần** RTX 5090/PRO 6000 — overkill cho project này

## Khi gặp lỗi

### OOM
- Giảm `batch_size` trong `experiments.py` (B5 mặc định 10 → 8 → 6)
- Tắt SWA cho stage2 nếu vẫn OOM (`enable_swa=False`)

### Loss nan
- Tversky đôi khi nan epoch đầu — giảm lr xuống 5e-5
- Hoặc ép init từ best.pt (mặc định đã làm)

### Disk full
- Xoá `work/runs_exp/<old_exp>/last.pt` (giữ best.pt)
- Snapshot ensemble chỉ cần Stage G

## Recommend chạy đúng thứ tự

```
A. Freeze (5 min)
B. Verify (20 min) → quyết định có cần D không
C. FPR retrain v2 baseline (3-5h)  [độc lập với D-G]
D. (cond) Re-preprocess v2 (3-4h CPU, có thể song song với C)
E. Pilot 6 exp (25-30h, để qua đêm × 1-2 đêm)
F. Inspect PILOT_RESULTS.md (manual)
G. Stage 2 full (15h, qua đêm × 1)
H. FPR rebuild (3-5h)
I. Lock + report (30 min)
```

Tổng wall-clock 2-3 ngày trên V100.
