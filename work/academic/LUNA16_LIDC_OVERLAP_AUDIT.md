# LUNA16 / LIDC test_panel Overlap Audit

**Date**: 2026-05-22  
**Purpose**: Verify that LUNA16 subset0-4 (fine-tune training candidates) do not leak into the locked LIDC test_panel used to measure F1=0.618.

---

## 1. Sources

| Source | File / Path | Count |
|--------|-------------|-------|
| LIDC test_panel PIDs | `work/splits.json` key "test" | 99 patients |
| LIDC SeriesInstanceUID mapping | `work/sop_index.json` (`series_to_patient`) | 1308 total series |
| LUNA16 local subset0-4 .mhd files | `datasetLuna16/subset{0-4}/subset{0-4}/*.mhd` | 445 series (89 per subset) |
| LUNA16 annotations.csv | `datasetLuna16/annotations.csv` | 601 unique seriesuid, 1186 nodule rows |

---

## 2. Series Counts Per LIDC Split

| Split | Patients | Series |
|-------|----------|--------|
| test | 99 | 129 |
| val | 99 | 131 |
| train | 812 | 1048 |

---

## 3. Overlap Results

| Comparison | Count | Percentage of LUNA16 subset0-4 |
|------------|-------|-------------------------------|
| LUNA16 subset0-4 ∩ LIDC **test** | **47** | **10.56%** |
| LUNA16 subset0-4 ∩ LIDC val | 44 | 9.89% |
| LUNA16 subset0-4 ∩ LIDC train | 354 | 79.55% |

**Interpretation**: LUNA16 is a curated subset of LIDC-IDRI. 47 of the 445 locally available LUNA16 series belong to patients assigned to the locked test_panel. If these series are used in fine-tune training, the model would have seen test-patient anatomy during training, invalidating the F1=0.618 benchmark.

---

## 4. First 10 Overlapping Series (LUNA16 ∩ LIDC test)

| Patient | SeriesInstanceUID (truncated) | LUNA16 Subset |
|---------|------------------------------|---------------|
| LIDC-IDRI-0830 | 1.3.6.1.4.1.14519.5.2.1.6279.6001.102133... | subset2 |
| LIDC-IDRI-0138 | 1.3.6.1.4.1.14519.5.2.1.6279.6001.104780... | subset4 |
| LIDC-IDRI-0447 | 1.3.6.1.4.1.14519.5.2.1.6279.6001.109002... | subset0 |
| LIDC-IDRI-0760 | 1.3.6.1.4.1.14519.5.2.1.6279.6001.111017... | subset1 |
| LIDC-IDRI-0828 | 1.3.6.1.4.1.14519.5.2.1.6279.6001.114195... | subset4 |
| LIDC-IDRI-0581 | 1.3.6.1.4.1.14519.5.2.1.6279.6001.116492... | subset2 |
| LIDC-IDRI-0260 | 1.3.6.1.4.1.14519.5.2.1.6279.6001.121993... | subset2 |
| LIDC-IDRI-0747 | 1.3.6.1.4.1.14519.5.2.1.6279.6001.126121... | subset0 |
| LIDC-IDRI-0286 | 1.3.6.1.4.1.14519.5.2.1.6279.6001.128059... | subset1 |
| LIDC-IDRI-0133 | 1.3.6.1.4.1.14519.5.2.1.6279.6001.133378... | subset2 |

Full list of all 47 series: `work/academic/LUNA16_EXCLUDE_FROM_TRAIN.json`

---

## 5. Decision

**Overlap = 47 series = 10.56%** → falls in the EXCLUDE band (5%–20%).

**Decision: EXCLUDE** — do not drop or restructure the LIDC split; instead, filter out the 47 conflicting series from any LUNA16 fine-tune training data loader.

After exclusion:
- LUNA16 fine-tune training pool: 445 - 47 = **398 series** from subset0-4
- LIDC test_panel: unchanged, 99 patients / 129 series, zero contamination

---

## 6. Action Required

1. Load `work/academic/LUNA16_EXCLUDE_FROM_TRAIN.json` in any LUNA16 fine-tune dataset class.
2. Filter: `if series_uid in excluded_set: skip`.
3. Also filter LUNA16 val series (subset7-8, not yet downloaded) against LIDC val_panel when those subsets are added — 44 series from subset0-4 already overlap with LIDC val.
4. Re-run this audit when subset5-9 are downloaded.

---

## 7. Notes on Methodology

- SeriesInstanceUID is the correct deduplication key: LUNA16 .mhd filenames are SeriesInstanceUIDs verbatim; LIDC sop_index maps these same UIDs to patient IDs.
- Coordinate frames not at issue here — this is an ID-level audit only.
- Raw DICOM data in `manifest-1600709154662/` not touched (read-only).
