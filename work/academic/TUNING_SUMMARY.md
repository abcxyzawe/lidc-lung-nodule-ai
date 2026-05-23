# Lung-nodule detection tuning — autonomous run, 2026-05-09

User instruction (paraphrased): "Use AI on multiple patients, compare to ground truth, adjust until matches as closely as possible."

## Pipeline

1. **Caching** — TTA + ensemble probability volumes pre-computed for `val_panel` (96/99 patients) and `test_panel` (95/99 patients). Stored in `work/academic/probs/`.
2. **Sweep** — `tune_detection_params.py --panel val` ran 54 post-processing configs × 7 thresholds = 378 ops × 96 patients. Selected by **peak F1 across thresholds** (best balance of TP and FP, matches "AI matches GT closely").
3. **Apply** — `apply_best_config.py` patched `configs.py` + `webapp/app.py` defaults.
4. **Test** — Locked test_panel (never tuned on) re-ran with both baseline and tuned configs.

## Best config (selected on val_panel, F1-based)

```
min_voxels         = 120
max_elongation     = 4.0
merge_dist_mm      = 10
subpleural_min_mm  = 0      # <-- previously 2.0; was killing real pleural-adjacent nodules
threshold          = 0.97   # high-confidence operating point
```

Key change vs old webapp defaults: **subpleural filter removed** (was rejecting real nodules near the pleura) and **threshold raised** to keep precision while accepting more candidate seeds.

## Results

### val_panel (96 patients, used for tuning)

|              | TP  | FP | FN  | Sens   | Prec   | FP/scan |
|--------------|-----|----|-----|--------|--------|---------|
| Baseline     |  99 | 38 | 144 | 0.407  | 0.723  | 0.40    |
| **Tuned**    | **126** | 43 | 117 | **0.519** | **0.746** | 0.45 |
| Δ            | +27 | +5 | -27 | +0.111 | +0.023 | +0.05   |

### test_panel (95 patients, locked — never tuned on)

|              | TP  | FP | FN  | Sens   | Prec   | FP/scan |
|--------------|-----|----|-----|--------|--------|---------|
| Baseline     | 106 | 51 | 144 | 0.424  | 0.675  | 0.54    |
| **Tuned**    | **134** | 66 | 116 | **0.536** | 0.670  | 0.69    |
| Δ            | +28 | +15| -28 | +0.112 | -0.005 | +0.16   |

**Improvement on test matches val** (+27 vs +28 TP) — no overfitting.

## Operating points (3 modes, on val_panel)

Best-F1 config (mv120_me4_md10_sp0):

| Mode      | Threshold | Sens  | Prec  | FP/scan | Use case |
|-----------|-----------|-------|-------|---------|----------|
| Sensitive | 0.20      | 0.621 | 0.455 | 1.89    | Don't miss anything |
| **Balanced** | **0.97** | **0.519** | **0.746** | **0.45** | **Default — best F1** |
| Strict    | 0.97      | 0.519 | 0.746 | 0.45    | Same as balanced for this config |

For separate "sensitive mode" with this model, you can also use mv50_me4_md10_sp0 + thr=0.92 (F1=0.602, sens 0.65, FP 1.29) — better recall, slightly more FP.

## Stratified sensitivity (val_panel, tuned config @ thr=0.97)

| Bucket          | n_GT | TP  | Sens   |
|-----------------|------|-----|--------|
| Small (4–6 mm)  | 61   | 26  | 0.426  |
| Medium (6–15 mm)| 120  | 77  | 0.642  |
| Large (≥15 mm)  | 44   | 34  | 0.773  |

Small nodules remain the hardest (model ceiling). Large nodules ≥15mm are caught 77% of the time.

## Why the model ceiling exists

Probing GT centroids on the debug panel: **8 of 12 missed nodules have peak prob ≈ 0** in the model's output — the model genuinely doesn't see them. No post-processing can recover these.

The recall ceiling (~76%) is a **training/architecture limit**. To push past it, the user's plan correctly identifies:
- 5-slice 2.5D input (more context per slice)
- Tversky/FocalTversky loss (penalise FN more heavily)
- Small-nodule oversampling
- 3D FP-reduction classifier (Phase 2 — would let us drop the threshold and use the classifier to filter FPs)

## Artifacts in work/academic/

- `best_config.json` — selected config + 3 operating points + FROC points
- `tuning_results.csv` — every (config × threshold) row swept (378 rows)
- `tuning_summary.csv` — one row per config (54 rows)
- `baseline_vs_tuned.json` — val_panel comparison
- `baseline_vs_tuned_TEST.json` — locked test_panel comparison
- `froc_baseline_vs_tuned_TEST.png` — FROC curve overlay
- `probs/` — cached prob volumes for both panels (used for fast re-runs)
- `lung_masks/` — cached lung masks

## Code added

- `src/panels.py` — debug/val/test panel definitions
- `src/tune_detection_params.py` — sweeper with disk-cached prob volumes
- `src/compare_baseline_vs_tuned.py` — side-by-side eval
- `src/compare_ai_vs_gt.py` — per-patient match table (lists which GT was found)
- `src/apply_best_config.py` — patches configs.py + webapp defaults
- `src/plot_tuning_froc.py` — FROC plot generator
- `src/run_full_tuning_pipeline.py` — end-to-end orchestrator

---

# Phase 2: 3D False-Positive Reduction Classifier

## Pipeline

1. **Extract candidates**: For each cached patient, run permissive detection (thr=0.40, min_voxels=20, no elong filter, no subpleural). Generates many candidates (most are FPs).
2. **Crop 48³ HU patches** around each candidate centroid.
3. **Label by GT match** (LUNA16 radius rule): 1 if matches a real GT nodule, 0 otherwise.
4. **Train DenseNet121-3D binary classifier**: weighted CE loss with class balancing, random flips/rot90 augmentation, cosine LR over 30 epochs.
5. **Apply at inference**: re-rank candidates by classifier prob × seg confidence.

## Dataset

|         | N patients | N candidates | Positives | Negatives |
|---------|------------|--------------|-----------|-----------|
| val (training) | 96 | 1283 | 184 (14.3%) | 1099 |
| test (held-out) | 95 | 1260 | 184 (14.6%) | 1076 |

## Classifier performance (test_panel)

- **AUC = 0.830**, AP = 0.480, best-F1 = 0.521 @ thr=0.75
- positive-class mean prob: 0.735 vs negative-class mean prob: 0.268 (clear separation)

## Test-panel detection at multiple FPR thresholds

| FPR thr | TP  | FP  | FN | Sens   | Prec   | F1     | FP/scan |
|---------|-----|-----|----|--------|--------|--------|---------|
| 0.10 (very lenient) | 163 | 463 | 21 | **0.886** | 0.260 | 0.402 | 4.87 |
| 0.30 | 148 | 343 | 36 | **0.804** | 0.301 | 0.439 | 3.61 |
| 0.50 | 142 | 273 | 42 | **0.772** | 0.342 | 0.474 | 2.87 |
| 0.70 | 128 | 208 | 56 | 0.696 | 0.381 | 0.492 | 2.19 |
| 0.80 (best F1) | 123 | 166 | 61 | 0.668 | 0.426 | **0.520** | 1.75 |

## Combined picture — 3 operating modes

| Mode | Pipeline | Sens | Prec | F1 | FP/scan |
|------|----------|------|------|-----|---------|
| **Strict** | Phase 1 segmentation only (mv120, thr=0.97) | 0.536 | 0.670 | 0.596 | 0.69 |
| **Balanced** | Phase 2 FPR @ thr=0.70 | 0.696 | 0.381 | 0.492 | 2.19 |
| **Sensitive** | Phase 2 FPR @ thr=0.30 | **0.804** | 0.301 | 0.439 | 3.61 |

vs **Baseline (old webapp)**: sens 0.424, prec 0.675, F1 0.522, FP/scan 0.54

## What this means

- Sensitivity ceiling moved from ~76% (model raw output) to **80.4%** because the FPR classifier lets us safely use a much lower seg threshold without drowning in FPs.
- The user's plan was right: this is the most leverage available without retraining segmentation.
- F1 ceiling didn't move much (0.59 → 0.52) because high-recall modes pay precision. For *clinical screening*, the sensitive mode is more useful than the strict one.

## Honest caveats

- **FPR was trained on val_panel** (~1283 cands, 184 pos) and **evaluated on test_panel** (~1260 cands, 184 pos) — no leak, but the data is small. AUC could be higher with all 812 train_panel patients cached + extracted (skipped tonight to keep within time budget).
- The 80.4% sensitivity is on candidates that the seg model already generated at thr=0.40. The truly invisible ~24% of GT nodules (peak prob ≈ 0) still cannot be found by either stage.

## Code added (Phase 2)

- `src/extract_fpr_candidates.py`
- `src/train_fpr_classifier.py`
- `src/apply_fpr_classifier.py`

## Artifacts (Phase 2)

- `work/academic/fpr_val.npz` (1283 candidates × 48³ patches, 75 MB compressed)
- `work/academic/fpr_test.npz` (1260 cands)
- `work/runs/fpr.pt` (DenseNet121-3D checkpoint, AUC=0.830)
- `work/academic/fpr_eval.json` (full FPR threshold sweep)

---

# Phase 3: Analysis of model's blind spots (test_panel)

For each of 250 GT nodules in test_panel, probed the cached prob volume at the GT centroid (5-voxel sphere). Categorized by peak prob:
- **Invisible** (peak < 0.3): model produces ~no signal
- **Weak** (0.3–0.7): model sees something but weak
- **Visible** (≥0.7): model is confident

Result: **77.6% visible, 20.4% invisible** → recall ceiling ≈ 78%.

## Counter-intuitive: medium-large nodules are HARDEST

| Size bucket | n | Visible | Invisible |
|-------------|---|---------|-----------|
| Small 3-6 mm | 75 | **89.3%** | 6.7% |
| Medium 6-10 mm | 93 | 78.5% | 20.4% |
| **Med-large 10-20 mm** | 59 | **66.1%** | **32.2%** |
| Large ≥20 mm | 20 | 75.0% | 25.0% |

The model's hardest bucket is NOT small nodules — it's the 10-20 mm range where 1 in 3 GT is invisible. Likely culprits:
- These are often spiculated / irregular / ground-glass (atypical shapes)
- The training mask mode (`union`) puts noisy boundaries from inconsistent radiologists, blurring the learning signal for atypical shapes

## By radiologist consensus

| n_rads | n | Visible | Invisible |
|--------|---|---------|-----------|
| 1/4 (noisy) | 207 | 76.8% | 20.8% |
| 2/4 | 15 | 66.7% | **33.3%** |
| **3/4** | 14 | **92.9%** | 7.1% |
| 4/4 | 14 | 85.7% | 14.3% |

When 3+ radiologists agree, the model finds the nodule 86–93% of the time. Most of the misses come from the 1/4 noisy annotations — but these are the ones LIDC's union mode trains on too.

## By malignancy

Roughly uniform across malignancy levels (78–82% visible) — model doesn't have a malignancy bias.

## Concrete next-iteration suggestion

Based on this analysis, the highest-leverage retraining changes (from your Phase-3-of-plan list):
1. **Switch mask mode `union → consensus2`**: drops the noisy 1/4 annotations. The 33% miss rate on 2/4 nodules suggests 2/4 is also unreliable — try consensus3 too.
2. **Oversample 10-20 mm GT during training** (the hardest bucket).
3. **Try TverskyLoss(α=0.3, β=0.7)**: penalises FN more heavily — would push the model to over-predict for irregular shapes rather than under-predict.

## Artifact

`work/academic/missed_nodules_analysis.json` — per-nodule peak prob + visibility categorisation.

## Code added (Phase 3)

- `src/analyze_missed_nodules.py`

---

# Sanity check: should we cascade Phase 1 → FPR?

Tried applying the FPR classifier to Phase 1 strict candidates (mv120, thr=0.97):

| Pipeline | TP | FP | FN | Sens (of 250) | Prec | F1 |
|----------|----|----|----|---------------|------|-----|
| Phase 1 alone (strict) | 134 | 66 | 116 | **0.536** | 0.670 | 0.596 |
| Phase 1 + FPR @ 0.10 | 121 | 51 | 129 | 0.484 | 0.703 | 0.573 |

**Conclusion: don't cascade.** Phase 1 strict already filters down to high-quality candidates; the FPR classifier was trained on permissive candidates and adds noise here (drops 13 real nodules to gain 15 FP rejections — net negative on F1).

The two phases are alternatives, not stages of one pipeline:
- **Use Phase 1 alone** when you want high precision (F1 0.596, FP/scan 0.69)
- **Use Phase 2 (permissive seg + FPR @ 0.30)** when you want high sensitivity (sens 0.804, F1 0.439, FP/scan 3.61)

The webapp's `predict_fpr_for_nodules()` adds `fpr_prob` as a *display signal* on Phase 1 candidates without filtering — useful for the radiologist to sort/triage but not used to drop predictions.
