"""Clinical decision support tools — published formulas, no ML training.

Implements:
  - Brock model (McWilliams NEJM 2013) — 4-year cancer probability per nodule
  - USPSTF 2021 lung cancer screening eligibility
  - NICE/BTS symptom concern level
  - Nodule type inference (solid / part-solid / non-solid) from CT HU
  - Upper-lobe location from z-position
"""
import math
from typing import Optional

import numpy as np


# ---------- Brock parsimonious model (NEJM 2013 Appendix Table S2) ----------

_BROCK = dict(
    intercept=-6.7892,
    age_coef=0.0287,           # per year, centered at 62
    age_center=62,
    female_coef=0.6011,
    family_hx_coef=0.2961,
    emphysema_coef=0.2953,
    size_coef=-5.3854,         # multiplied by ((size_mm/10)^-0.5 - 1.58113883)
    size_const=1.58113883,
    type_coef=-0.1466,         # multiplied by type_code (solid=0, non-solid=1, part-solid=2)
    upper_lobe_coef=0.6581,
    count_coef=-0.0824,
    spiculation_coef=0.7729,
)


def brock_probability(age: int, sex: str, family_hx: bool, emphysema: bool,
                      size_mm: float, nodule_type: str, upper_lobe: bool,
                      count: int, spiculated: bool) -> float:
    """Brock parsimonious model → P(nodule is malignant within ~4 years), 0..1.

    Validated AUC 0.970 in PanCan Early Detection Study.
    """
    type_code = {"solid": 0, "non-solid": 1, "part-solid": 2}.get(nodule_type, 0)
    size_term = (max(size_mm, 1.0) / 10) ** -0.5 - _BROCK["size_const"]
    log_odds = (
        _BROCK["intercept"]
        + _BROCK["age_coef"] * (age - _BROCK["age_center"])
        + (_BROCK["female_coef"] if sex == "female" else 0)
        + (_BROCK["family_hx_coef"] if family_hx else 0)
        + (_BROCK["emphysema_coef"] if emphysema else 0)
        + _BROCK["size_coef"] * size_term
        + _BROCK["type_coef"] * type_code
        + (_BROCK["upper_lobe_coef"] if upper_lobe else 0)
        + _BROCK["count_coef"] * max(count, 1)
        + (_BROCK["spiculation_coef"] if spiculated else 0)
    )
    # Numerically-stable sigmoid
    if log_odds >= 0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    e = math.exp(log_odds)
    return e / (1.0 + e)


def brock_band(prob: float) -> str:
    """Convert Brock probability to clinical action band (PanCan thresholds)."""
    if prob < 0.05: return "low"        # <5%   — annual LDCT
    if prob < 0.10: return "medium"     # 5-10% — 3-month follow-up
    return "high"                        # >10%  — PET-CT / biopsy


# ---------- USPSTF 2021 screening eligibility ----------

def uspstf_eligible(age: int, pack_years: float,
                    currently_smoking: bool, years_since_quit: int) -> dict:
    """USPSTF 2021 grade B: age 50-80, ≥20 pack-years, current or quit <15 yrs."""
    reasons_fail = []
    if not (50 <= age <= 80):
        reasons_fail.append(f"tuổi {age} ngoài khoảng 50-80")
    if pack_years < 20:
        reasons_fail.append(f"chỉ {pack_years:.0f} gói-năm (cần ≥20)")
    if not currently_smoking and years_since_quit > 15:
        reasons_fail.append(f"đã bỏ thuốc {years_since_quit} năm (giới hạn 15)")
    eligible = len(reasons_fail) == 0
    if eligible:
        msg = (f"Đủ điều kiện tầm soát LDCT hằng năm "
               f"({pack_years:.0f} gói-năm, "
               f"{'đang hút' if currently_smoking else f'bỏ {years_since_quit} năm'})")
    else:
        msg = "Không nằm trong nhóm tầm soát: " + ", ".join(reasons_fail)
    return {"eligible": eligible, "message": msg}


# ---------- Symptom-based concern (NICE NG12 + BTS) ----------

_SYMPTOM_LIST = [
    ("hemoptysis",  "Ho ra máu",                     "high"),
    ("weight_loss", "Sụt cân không rõ nguyên nhân",  "medium"),
    ("clubbing",    "Ngón tay dùi trống",            "medium"),
    ("hoarseness",  "Khàn tiếng kéo dài",            "medium"),
    ("cough",       "Ho mạn tính > 3 tuần",          "low"),
    ("chest_pain",  "Đau ngực",                      "low"),
    ("dyspnea",     "Khó thở mới / xấu đi",          "low"),
    ("recurrent_infection", "Nhiễm trùng ngực tái phát", "low"),
]


def symptom_concern(symptoms: dict) -> dict:
    """Aggregate symptom checklist into a concern level + active list."""
    active = []
    has_high = False; n_med = 0; n_low = 0
    for key, label, weight in _SYMPTOM_LIST:
        if symptoms.get(key):
            active.append({"key": key, "label": label, "weight": weight})
            if weight == "high": has_high = True
            elif weight == "medium": n_med += 1
            elif weight == "low": n_low += 1
    if has_high:
        level = "high"
        msg = "⚠ Triệu chứng red-flag — NICE khuyến cáo chuyển khám trong 2 tuần"
    elif n_med >= 1 or n_low >= 2:
        level = "medium"
        msg = "Có nhiều triệu chứng — nên khám chuyên khoa hô hấp"
    elif n_low >= 1:
        level = "low"
        msg = "Có triệu chứng nhẹ — theo dõi"
    else:
        level = "low"
        msg = "Không có triệu chứng đáng kể"
    return {"level": level, "message": msg, "active": active, "count": len(active)}


# ---------- Image-derived features used by Brock ----------

def detect_nodule_type_from_hu(volume_hu: np.ndarray, bbox_zyx: list,
                               labeled_pred: Optional[np.ndarray] = None,
                               nodule_id: Optional[int] = None) -> str:
    """Estimate nodule density type from mean HU inside the predicted mask.

    Solid:      mean > -100 HU (soft-tissue density)
    Part-solid: -600 to -100 HU
    Non-solid (ground-glass): mean < -600 HU
    """
    zmin, ymin, xmin, zmax, ymax, xmax = bbox_zyx
    crop = volume_hu[zmin:zmax, ymin:ymax, xmin:xmax].astype(np.float32)
    if labeled_pred is not None and nodule_id is not None:
        m = (labeled_pred[zmin:zmax, ymin:ymax, xmin:xmax] == nodule_id)
        if m.any():
            crop = crop[m]
    if crop.size == 0:
        return "solid"
    mean_hu = float(crop.mean())
    if mean_hu > -100: return "solid"
    if mean_hu > -600: return "part-solid"
    return "non-solid"


def is_upper_lobe(centroid_z: float, total_slices: int) -> bool:
    """Upper lobe if centroid in upper half of volume.

    LIDC preprocess sorts slices by ascending ImagePositionPatient[2] —
    higher z = more cranial = upper lobe.
    """
    return (centroid_z / max(total_slices, 1)) > 0.5
