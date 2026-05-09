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


# ---------- Final diagnosis & recommendation engine ----------

def diagnose(nodules: list, clinical: dict) -> dict:
    """Generate a rich, actionable clinical assessment from per-nodule + patient data.

    Combines size, Lung-RADS, AI malignancy probability, Brock 4-yr probability,
    patient risk factors, and symptom red flags into a single recommendation.

    Returns dict with:
        summary           one-line description ("3 nốt can thiệp, 2 nốt theo dõi")
        action            concrete recommendation text
        action_band       "urgent" | "soon" | "routine" | "none"
        action_title      short label used in banner ("Khám trong 2 tuần")
        nodule_buckets    {"actionable": int, "monitor": int, "incidental": int}
        risk_factors      list of {factor, weight}
        max_brock_pct     float (highest Brock % across all nodules)
        max_diameter_mm   float
        index_nodule_id   int | None
        index_reason      text explaining why the index nodule matters
    """
    p = (clinical or {}).get("patient", {}) or {}
    sym = (clinical or {}).get("symptoms", {}) or {}
    uspstf = (clinical or {}).get("uspstf", {}) or {}

    # ---- Bucket nodules clinically ----
    actionable, monitor, incidental = [], [], []
    for n in nodules:
        d = float(n.get("diameter_mm", 0))
        b = float(n.get("brock_prob", 0) or 0)
        lr = (n.get("lung_rads") or {}).get("category", "")
        if d >= 8 or b >= 0.10 or lr in ("4B", "4X"):
            actionable.append(n)
        elif d >= 6 or b >= 0.05:
            monitor.append(n)
        else:
            incidental.append(n)

    max_brock = max((float(n.get("brock_prob", 0) or 0) for n in nodules), default=0.0)
    max_diam = max((float(n.get("diameter_mm", 0)) for n in nodules), default=0.0)
    has_red_flag = sym.get("level") == "high"

    # ---- Risk factors ----
    risk_factors = []
    age = int(p.get("age", 0))
    py = float(p.get("pack_years", 0) or 0)
    if age >= 65:
        risk_factors.append({"factor": f"Tuổi {age} (>65)", "weight": "medium"})
    elif age >= 55:
        risk_factors.append({"factor": f"Tuổi {age} (>55)", "weight": "low"})
    if py >= 30:
        risk_factors.append({"factor": f"{py:.0f} gói-năm (>30)", "weight": "high"})
    elif py >= 20:
        risk_factors.append({"factor": f"{py:.0f} gói-năm (>20)", "weight": "medium"})
    elif py >= 10:
        risk_factors.append({"factor": f"{py:.0f} gói-năm", "weight": "low"})
    if p.get("currently_smoking"):
        risk_factors.append({"factor": "Đang hút thuốc", "weight": "high"})
    if p.get("family_hx"):
        risk_factors.append({"factor": "Tiền sử gia đình ung thư phổi", "weight": "medium"})
    if p.get("emphysema"):
        risk_factors.append({"factor": "COPD / khí phế thũng", "weight": "medium"})
    if has_red_flag:
        risk_factors.append({"factor": "Triệu chứng red-flag (ho ra máu)", "weight": "high"})
    elif sym.get("level") == "medium":
        risk_factors.append({"factor": f"{sym.get('count', 0)} triệu chứng đáng kể", "weight": "medium"})

    # ---- Recommendation ----
    if has_red_flag and len(actionable) > 0:
        band = "urgent"
        title = "Khám chuyên khoa TRONG 2 TUẦN"
        action = ("Bệnh nhân có triệu chứng red-flag (ho ra máu) cộng với "
                  f"{len(actionable)} nodule đáng can thiệp. Theo NICE NG12, "
                  "chỉ định khám chuyên khoa hô hấp/lồng ngực urgent trong 2 tuần.")
    elif max_diam >= 30 or max_brock >= 0.30:
        band = "urgent"
        title = "PET-CT + xem xét sinh thiết"
        action = (f"Có nodule đường kính {max_diam:.1f}mm hoặc Brock 4-yr "
                  f"{max_brock*100:.1f}% — khả năng ác tính cao. "
                  "Chỉ định PET-CT hoặc sinh thiết transthoracic.")
    elif max_diam >= 15 or max_brock >= 0.10:
        band = "soon"
        title = "PET-CT + chụp lại 3 tháng"
        action = (f"Nodule lớn nhất {max_diam:.1f}mm (Brock {max_brock*100:.1f}%). "
                  "Xem xét PET-CT, theo dõi CT phổi sau 3 tháng. "
                  "Tăng kích thước >1.5mm trong 3 tháng = đáng nghi ung thư.")
    elif max_diam >= 8 or max_brock >= 0.05:
        band = "soon"
        title = "Chụp lại CT phổi sau 3 tháng"
        action = (f"Nodule {max_diam:.1f}mm cần theo dõi tăng kích thước. "
                  "CT phổi liều thấp sau 3 tháng. "
                  "Brock 4-yr ở mức trung bình ({:.1f}%).".format(max_brock*100))
    elif max_diam >= 6:
        band = "routine"
        title = "Chụp lại CT sau 6-12 tháng"
        action = ("Nodule 6-8mm theo Lung-RADS 3 — cần theo dõi định kỳ. "
                  "Chụp CT phổi liều thấp sau 6-12 tháng.")
    elif len(nodules) > 0:
        band = "none"
        title = "Không cần follow-up"
        action = (f"Phát hiện {len(nodules)} nodule nhỏ (<6mm). "
                  "Lung-RADS phân loại bỏ qua — không cần can thiệp hay theo dõi đặc biệt.")
    else:
        band = "none"
        title = "Phổi sạch"
        action = "AI không tìm thấy nodule nào ≥ 4.5mm trong phổi."

    if uspstf.get("eligible"):
        action += " Bệnh nhân đủ điều kiện tầm soát ung thư phổi LDCT hằng năm theo USPSTF."

    # ---- Summary ----
    parts = []
    if actionable: parts.append(f"{len(actionable)} nốt cần can thiệp")
    if monitor:    parts.append(f"{len(monitor)} nốt theo dõi")
    if incidental: parts.append(f"{len(incidental)} nốt nhỏ (bỏ qua)")
    summary = ", ".join(parts) if parts else "Không phát hiện nodule"

    # ---- Index nodule (highest concern) ----
    if nodules:
        # Score each by combined risk: prefer high diameter + high Brock
        def score(n):
            return float(n.get("diameter_mm", 0)) + float(n.get("brock_prob", 0) or 0) * 100
        idx = max(nodules, key=score)
        reasons = [f"đường kính {idx['diameter_mm']:.1f}mm"]
        bp = float(idx.get("brock_prob", 0) or 0)
        if bp >= 0.05:
            reasons.append(f"Brock {bp*100:.1f}%")
        sp = float(idx.get("ai_susp_prob", 0) or 0)
        if sp >= 0.5:
            reasons.append(f"AI nghi ngờ {sp*100:.0f}%")
        if idx.get("upper_lobe"):
            reasons.append("ở thuỳ trên (vùng nguy cơ ung thư cao)")
        if (idx.get("nodule_type") or "solid") != "solid":
            reasons.append(f"loại {idx['nodule_type']}")
        index_reason = " · ".join(reasons)
        index_id = int(idx["id"])
    else:
        index_id = None
        index_reason = ""

    return {
        "summary": summary,
        "action": action,
        "action_band": band,
        "action_title": title,
        "nodule_buckets": {
            "actionable": len(actionable),
            "monitor": len(monitor),
            "incidental": len(incidental),
        },
        "risk_factors": risk_factors,
        "max_brock_pct": round(max_brock * 100, 2),
        "max_diameter_mm": round(max_diam, 1),
        "index_nodule_id": index_id,
        "index_reason": index_reason,
    }
