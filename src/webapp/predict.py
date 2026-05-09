"""Inference + lung segmentation + 3D rendering for the webapp.

- Reads a list of DICOM paths
- HU-threshold lung segmentation (no extra model)
- best.pt nodule segmentation (TTA)
- 3D Plotly Mesh3d: lung shell transparent + each nodule as a separate mesh
"""
from pathlib import Path
import sys

import numpy as np
import torch
import pydicom
from scipy.ndimage import (
    label as cc_label,
    binary_closing,
    binary_fill_holes,
    binary_erosion,
    distance_transform_edt,
)
from skimage import measure
from PIL import Image, ImageDraw
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import HU_LO, HU_HI, MIN_NODULE_VOXELS, RUNS_DIR
from importlib import import_module
make_model = import_module("05_train").make_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None
_MODEL_SWA = None
_MALIGNANCY_MODEL = None
_LUNG_INFERER = None
_MAL_HU = (-1024.0, 600.0)
_MAL_PATCH = 32


def get_model(ckpt_path: Path = None):
    global _MODEL
    if _MODEL is None:
        # Switched to last.pt (E200) per user request — testing late-epoch model
        ckpt_path = ckpt_path or (RUNS_DIR / "last.pt")
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        encoder = ck.get("encoder", "efficientnet-b5")
        m = make_model(encoder=encoder, pretrained=None).to(DEVICE)
        m.load_state_dict(ck["model"])
        m.train(False)
        _MODEL = m
        print(f"Seg model loaded ({encoder}) on {DEVICE}, ckpt={ckpt_path.name} "
              f"(epoch {ck.get('epoch','?')}, val_dice {ck.get('val_dice','?')})")
    return _MODEL


def get_swa_model():
    """Lazy-load SWA segmentation model for ensemble. Returns None if missing."""
    global _MODEL_SWA
    if _MODEL_SWA is False:
        return None
    if _MODEL_SWA is None:
        swa_path = RUNS_DIR / "swa.pt"
        if not swa_path.exists():
            _MODEL_SWA = False
            return None
        ck = torch.load(swa_path, map_location=DEVICE, weights_only=False)
        encoder = ck.get("encoder", "efficientnet-b5")
        m = make_model(encoder=encoder, pretrained=None).to(DEVICE)
        m.load_state_dict(ck["model"])
        m.train(False)
        _MODEL_SWA = m
        print(f"SWA seg model loaded for ensemble.")
    return _MODEL_SWA


def get_malignancy_model(ckpt_path: Path = None):
    """Lazy-load the 3D malignancy classifier (5-class). Returns None if not available."""
    global _MALIGNANCY_MODEL, _MAL_HU, _MAL_PATCH
    if _MALIGNANCY_MODEL is False:
        return None
    if _MALIGNANCY_MODEL is None:
        ckpt_path = ckpt_path or (RUNS_DIR / "malignancy.pt")
        if not ckpt_path.exists():
            print(f"Malignancy classifier not found at {ckpt_path}; skipping risk score.")
            _MALIGNANCY_MODEL = False
            return None
        from monai.networks.nets import DenseNet121
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        n_classes = int(ck.get("n_classes", 5))
        _MAL_PATCH = int(ck.get("patch_size", 32))
        _MAL_HU = (float(ck.get("hu_lo", -1024)), float(ck.get("hu_hi", 600)))
        m = DenseNet121(spatial_dims=3, in_channels=1, out_channels=n_classes).to(DEVICE)
        m.load_state_dict(ck["model"])
        m.train(False)
        _MALIGNANCY_MODEL = m
        print(f"Malignancy model loaded (DenseNet121-3D, {n_classes} classes), patch {_MAL_PATCH}^3")
    return _MALIGNANCY_MODEL


def normalize(x):
    x = np.clip(x.astype(np.float32), HU_LO, HU_HI)
    return (x - HU_LO) / (HU_HI - HU_LO)


def read_dicom_series(dcm_paths):
    """Read DICOMs, sort by z, return (volume_int16 [N,512,512], voxel_sp_zyx_mm)."""
    H = W = 512
    slices = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), force=True)
            z = float(ds.ImagePositionPatient[2])
            slices.append((z, ds))
        except Exception:
            continue
    if not slices:
        raise ValueError("Không đọc được DICOM nào hợp lệ.")
    slices.sort(key=lambda x: x[0])
    N = len(slices)
    vol = np.zeros((N, H, W), dtype=np.int16)
    for i, (_, ds) in enumerate(slices):
        try:
            arr = ds.pixel_array
        except Exception:
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            arr = ds.pixel_array
        if arr.shape != (H, W):
            pad = np.zeros((H, W), dtype=arr.dtype)
            pad[: min(arr.shape[0], H), : min(arr.shape[1], W)] = arr[:H, :W]
            arr = pad
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        vol[i] = (arr.astype(np.float32) * slope + inter).astype(np.int16)
    pix = list(map(float, getattr(slices[0][1], "PixelSpacing", [1.0, 1.0])))
    thk = float(getattr(slices[0][1], "SliceThickness", 1.0) or 1.0)
    if len(slices) > 1:
        z_gap = abs(slices[1][0] - slices[0][0])
        if z_gap > 0:
            thk = z_gap
    voxel_sp = (thk, pix[0], pix[1])
    return vol, voxel_sp


def _get_lung_inferer():
    """Lazy-load the pretrained lung segmentation U-Net (Hofmanninger 2020)."""
    global _LUNG_INFERER
    if _LUNG_INFERER is None:
        from lungmask import LMInferer
        _LUNG_INFERER = LMInferer(modelname="R231", force_cpu=False)
        print("Lungmask LMInferer (R231) loaded.")
    return _LUNG_INFERER


def segment_lung(volume_hu):
    """Pretrained U-Net lung segmentation (lungmask R231). Returns uint8 [N,H,W].

    Falls back to a HU-threshold heuristic if lungmask fails.
    Output mask: 1 = lung tissue (left + right merged), 0 = elsewhere.
    """
    try:
        inferer = _get_lung_inferer()
        mask = inferer.apply(volume_hu.astype(np.int16))  # returns 0/1/2 (bg / right / left)
        return (mask > 0).astype(np.uint8)
    except Exception as e:
        print(f"lungmask failed ({e!r}); falling back to HU heuristic")
        air = volume_hu < -320
        body = ~air
        body_filled = np.zeros_like(body)
        for i in range(body.shape[0]):
            body_filled[i] = binary_fill_holes(body[i])
        lung = air & body_filled
        lung = binary_closing(lung, structure=np.ones((1, 3, 3)))
        lab, n = cc_label(lung)
        if n == 0:
            return np.zeros_like(lung, dtype=np.uint8)
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        keep = np.zeros_like(lab, dtype=bool); kept = 0
        for cid in np.argsort(sizes)[::-1]:
            if sizes[cid] < 5000: break
            keep |= lab == cid
            kept += 1
            if kept >= 2: break
        return keep.astype(np.uint8)


def _infer_slice(models: list, x: torch.Tensor, tta: bool) -> torch.Tensor:
    """Average sigmoid probabilities across models × TTA augmentations."""
    probs = []
    for m in models:
        p1 = torch.sigmoid(m(x))
        if tta:
            p2 = torch.sigmoid(m(torch.flip(x, dims=[-1]))).flip(dims=[-1])
            p3 = torch.sigmoid(m(torch.flip(x, dims=[-2]))).flip(dims=[-2])
            p4 = torch.sigmoid(m(torch.flip(x, dims=[-1, -2]))).flip(dims=[-1, -2])
            probs.append((p1 + p2 + p3 + p4) / 4)
        else:
            probs.append(p1)
    return torch.stack(probs, 0).mean(0)


def predict_nodules(volume_hu, threshold=0.5, tta=True, ensemble=True, progress=None):
    """Run segmentation slice-by-slice with TTA + optional ensemble.

    ensemble=True: average best.pt + swa.pt (if available) for stability.
    """
    models = [get_model()]
    if ensemble:
        swa = get_swa_model()
        if swa is not None:
            models.append(swa)
    N, H, W = volume_hu.shape
    prob = np.zeros((N, H, W), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            ip = max(0, i - 1); ine = min(N - 1, i + 1)
            stk = np.stack(
                [normalize(volume_hu[ip]), normalize(volume_hu[i]), normalize(volume_hu[ine])],
                axis=0,
            )
            x = torch.from_numpy(stk).unsqueeze(0).float().to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                p = _infer_slice(models, x, tta)
            prob[i] = p[0, 0].float().cpu().numpy()
            if progress and i % 4 == 0:
                progress(i, N)
    mask = (prob > threshold).astype(np.uint8)
    return prob, mask


def clean_mask(mask_3d: np.ndarray, voxel_sp: tuple) -> np.ndarray:
    """Morphological cleanup of binary mask before connected components.

    1. closing (dilate then erode) → bridge small gaps from adjacent slices
       (same nodule split into 2 blobs by 1-voxel discontinuity)
    2. opening (erode then dilate) → remove single-voxel noise specks
    """
    # Closing radius 1 in xy + small in z (z spacing usually larger)
    structure_close = np.ones((1, 3, 3), dtype=np.uint8)
    closed = binary_closing(mask_3d.astype(bool), structure=structure_close, iterations=1)
    # Opening radius 1 → drops isolated <2 voxel artifacts
    structure_open = np.ones((1, 3, 3), dtype=np.uint8)
    opened = binary_erosion(closed, structure=structure_open, iterations=1)
    # Dilate back to recover boundary
    from scipy.ndimage import binary_dilation
    cleaned = binary_dilation(opened, structure=structure_open, iterations=1)
    return cleaned.astype(np.uint8)


def merge_nearby_nodules(nodules: list, voxel_sp: tuple, max_dist_mm: float = 8.0) -> list:
    """Merge nodules whose physical centroids are within max_dist_mm.

    Same physical nodule can split into 2 blobs because of slice discontinuity.
    Merged keeps the larger blob's stats, sums voxel count, expands bbox.
    """
    if len(nodules) < 2:
        return nodules
    nodules = sorted(nodules, key=lambda n: -n["volume_mm3"])
    used = [False] * len(nodules)
    merged = []
    for i, a in enumerate(nodules):
        if used[i]:
            continue
        used[i] = True
        cluster = [a]
        ca = np.array(a["centroid_zyx_voxel"]) * np.array(voxel_sp)
        for j in range(i + 1, len(nodules)):
            if used[j]:
                continue
            b = nodules[j]
            cb = np.array(b["centroid_zyx_voxel"]) * np.array(voxel_sp)
            if np.linalg.norm(ca - cb) <= max_dist_mm:
                cluster.append(b)
                used[j] = True
        if len(cluster) == 1:
            merged.append(a)
        else:
            # Merge: sum voxels, expand bbox, recompute diameter
            total_vox = sum(c["voxels"] for c in cluster)
            total_vol = sum(c["volume_mm3"] for c in cluster)
            new_diam = 2 * (3 * total_vol / (4 * np.pi)) ** (1 / 3)
            bbs = [c["bbox_zyx_voxel"] for c in cluster]
            bbox_min = [min(bb[k] for bb in bbs) for k in range(3)]
            bbox_max = [max(bb[k] for bb in bbs) for k in range(3, 6)]
            a2 = dict(a)
            a2["voxels"] = total_vox
            a2["volume_mm3"] = total_vol
            a2["diameter_mm"] = float(new_diam)
            a2["bbox_zyx_voxel"] = bbox_min + bbox_max
            a2["merged_from"] = [c["id"] for c in cluster]
            merged.append(a2)
    return merged


def render_nodule_thumb(volume_hu: np.ndarray, nodule: dict, out_path: Path,
                        size: int = 280, hu_window=(-1000, 200)) -> None:
    """Save a thumbnail PNG of the centroid axial slice with red bounding box.

    Crops a ~160-pixel ROI around the nodule, normalises to lung window,
    converts to RGB, draws the bbox in red, saves PNG.
    """
    H, W = volume_hu.shape[1], volume_hu.shape[2]
    cz, cy, cx = [int(round(c)) for c in nodule["centroid_zyx_voxel"]]
    cz = max(0, min(volume_hu.shape[0] - 1, cz))
    zmin, ymin, xmin, zmax, ymax, xmax = nodule["bbox_zyx_voxel"]

    # Crop a 160×160 ROI centered on the bbox
    crop_h = max(160, (ymax - ymin) + 60)
    crop_w = max(160, (xmax - xmin) + 60)
    cy_mid = (ymin + ymax) // 2
    cx_mid = (xmin + xmax) // 2
    y0 = max(0, cy_mid - crop_h // 2); y1 = min(H, y0 + crop_h)
    x0 = max(0, cx_mid - crop_w // 2); x1 = min(W, x0 + crop_w)
    if y1 == H: y0 = max(0, H - crop_h)
    if x1 == W: x0 = max(0, W - crop_w)

    sl = volume_hu[cz, y0:y1, x0:x1].astype(np.float32)
    img = np.clip(sl, hu_window[0], hu_window[1])
    img = ((img - hu_window[0]) / (hu_window[1] - hu_window[0]) * 255).astype(np.uint8)
    pim = Image.fromarray(img, mode="L").convert("RGB").resize((size, size))

    # Draw bbox
    sx = size / (x1 - x0); sy = size / (y1 - y0)
    bx0 = (xmin - x0) * sx; by0 = (ymin - y0) * sy
    bx1 = (xmax - x0) * sx; by1 = (ymax - y0) * sy
    draw = ImageDraw.Draw(pim)
    draw.rectangle([bx0, by0, bx1, by1], outline=(255, 64, 64), width=3)
    # Slice number text (lower-right)
    draw.text((size - 60, size - 18), f"z={cz}", fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pim.save(out_path, format="PNG", optimize=True)


def filter_subpleural(nodules: list, lung_mask: np.ndarray, voxel_sp: tuple,
                      min_dist_mm: float = 1.5) -> list:
    """Drop nodules whose centroid is too close to lung pleural surface.

    Subpleural blobs are often FP artifacts at the lung-chest-wall interface.
    Real pleural nodules exist but require special workup; for screening AI
    we conservatively drop them. Set min_dist_mm=0 to disable.
    """
    if min_dist_mm <= 0 or len(nodules) == 0:
        return nodules
    # Distance transform: for each lung voxel, distance to nearest non-lung voxel
    sampling = list(voxel_sp)  # (z, y, x) mm
    dist = distance_transform_edt(lung_mask, sampling=sampling)
    kept = []
    for n in nodules:
        z, y, x = [int(round(c)) for c in n["centroid_zyx_voxel"]]
        z = max(0, min(dist.shape[0] - 1, z))
        y = max(0, min(dist.shape[1] - 1, y))
        x = max(0, min(dist.shape[2] - 1, x))
        if dist[z, y, x] >= min_dist_mm:
            kept.append(n)
    return kept


def lung_rads(diameter_mm: float) -> dict:
    """Lung-RADS for solid nodules at baseline screening (simplified, size-only).

    Real Lung-RADS also considers solid/sub-solid type, growth, calcification.
    Here size is the only signal we have automatically.
    """
    d = diameter_mm
    if d < 6:    return {"category": "2",  "label": "Bỏ qua (< 6mm, không cần theo dõi)", "size_risk": "low"}
    if d < 8:    return {"category": "3",  "label": "Theo dõi 6 tháng",                   "size_risk": "low"}
    if d < 15:   return {"category": "4A", "label": "Đáng nghi (chụp lại 3 tháng)",       "size_risk": "medium"}
    if d < 30:   return {"category": "4B", "label": "Rất đáng nghi (PET-CT/sinh thiết)",  "size_risk": "high"}
    return       {"category": "4X", "label": "Cực kỳ đáng nghi (sinh thiết)",      "size_risk": "high"}


def predict_malignancy_for_nodules(volume_hu, nodules):
    """For each nodule, extract a 32^3 patch around centroid and classify.

    Adds keys to each nodule dict:
        ai_class:        argmax 1..5 (1=very benign, 5=very malignant)
        ai_expected:     expected value of malignancy (sum k * P(k))
        ai_susp_prob:    P(class >= 4)  in [0..1]
        ai_risk_label:   "low" | "medium" | "high"
        lung_rads:       size-based dict (cat, label, risk)
        risk_combined:   fused label: agree -> max(ai_risk, size_risk); else "review"
    """
    model = get_malignancy_model()
    P = _MAL_PATCH
    H = P // 2
    N, Hv, Wv = volume_hu.shape

    if model is not None and nodules:
        # Build patch tensor
        patches = []
        for n in nodules:
            cz, cy, cx = [int(round(c)) for c in n["centroid_zyx_voxel"]]
            zmin = max(0, cz - H); zmax = min(N, cz + H)
            ymin = max(0, cy - H); ymax = min(Hv, cy + H)
            xmin = max(0, cx - H); xmax = min(Wv, cx + H)
            crop = volume_hu[zmin:zmax, ymin:ymax, xmin:xmax].astype(np.float32)
            crop = np.clip(crop, _MAL_HU[0], _MAL_HU[1])
            crop = (crop - _MAL_HU[0]) / (_MAL_HU[1] - _MAL_HU[0])
            padded = np.zeros((P, P, P), dtype=np.float32)
            pz0 = H - (cz - zmin); py0 = H - (cy - ymin); px0 = H - (cx - xmin)
            padded[pz0:pz0+crop.shape[0], py0:py0+crop.shape[1], px0:px0+crop.shape[2]] = crop
            patches.append(padded)
        x = torch.from_numpy(np.stack(patches, 0)).unsqueeze(1).float().to(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    else:
        probs = None

    for i, n in enumerate(nodules):
        if probs is not None:
            p = probs[i]  # [5]
            cls = int(p.argmax()) + 1
            expected = float(sum((k + 1) * p[k] for k in range(len(p))))
            susp = float(p[3] + p[4]) if len(p) >= 5 else 0.0
            ai_risk = "high" if susp >= 0.5 else ("medium" if susp >= 0.25 else "low")
            n.update({
                "ai_class": cls,
                "ai_expected": expected,
                "ai_susp_prob": susp,
                "ai_risk_label": ai_risk,
            })
        size = lung_rads(n["diameter_mm"])
        n["lung_rads"] = size
        d = n["diameter_mm"]
        susp = n.get("ai_susp_prob", 0.0)
        ai_cls = n.get("ai_class", 3)

        # Clinically-tuned combined logic:
        # < 6mm = always LOW per Lung-RADS (incidental, no follow-up). AI can't override.
        # 6-8mm = LOW unless AI is clearly suspicious (class >= 4 AND P >= 0.4)
        # 8-15mm = MEDIUM, escalates to HIGH if AI strongly suspicious (P >= 0.5)
        # >= 15mm = HIGH (size-driven), de-escalate to MEDIUM only if AI strongly says benign (class <= 2 AND P < 0.15)
        if d < 6:
            final = "low"
        elif d < 8:
            final = "medium" if (ai_cls >= 4 and susp >= 0.4) else "low"
        elif d < 15:
            final = "high" if (ai_cls >= 4 and susp >= 0.5) else "medium"
        else:
            final = "medium" if (ai_cls <= 2 and susp < 0.15) else "high"
        n["risk_combined"] = final
    return nodules


def _elongation_ratio(coords: np.ndarray, voxel_sp: tuple) -> float:
    """PCA-based elongation: sqrt(λ_max / λ_min) of physical-coord covariance.

    Vessels and bronchi cut transversely give ratio > 4-5;
    real lung nodules are roughly spherical, ratio < 3.
    """
    if len(coords) < 10:
        return 1.0
    physical = coords.astype(np.float32) * np.array(voxel_sp, dtype=np.float32)
    centered = physical - physical.mean(axis=0)
    cov = np.cov(centered.T)
    try:
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    except np.linalg.LinAlgError:
        return 1.0
    if eigvals[2] < 1e-6:
        return 99.0
    return float(np.sqrt(max(eigvals[0], 0) / max(eigvals[2], 1e-6)))


def _bbox_extent(coords: np.ndarray) -> float:
    """Volume / bounding-box volume. 1.0 = perfect cube fill,
    sphere = ~0.524, vessel/strip = 0.1-0.3, irregular scar = 0.2-0.4.
    """
    bmin = coords.min(axis=0)
    bmax = coords.max(axis=0) + 1
    bbox_vol = float(np.prod(bmax - bmin))
    return float(len(coords)) / max(bbox_vol, 1.0)


def find_nodules(mask_3d, voxel_sp, min_voxels=MIN_NODULE_VOXELS,
                 max_elongation: float = 4.0):
    """Connected components 3D + filter shape (vessels) + size.

    A blob is dropped if:
      - voxels < min_voxels (too small / artifact), or
      - elongation ratio > max_elongation (tubular = vessel/bronchus).
    """
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    lab, n = cc_label(mask_3d, structure=structure)
    out = []
    for i in range(1, n + 1):
        coords = np.argwhere(lab == i)
        if len(coords) < min_voxels:
            continue
        elong = _elongation_ratio(coords, voxel_sp)
        if elong > max_elongation:
            continue
        c = coords.mean(0)
        bmin = coords.min(0)
        bmax = coords.max(0) + 1
        vol_mm3 = float(len(coords)) * float(np.prod(voxel_sp))
        diam = 2 * (3 * vol_mm3 / (4 * np.pi)) ** (1 / 3)
        out.append({
            "id": int(i),
            "voxels": int(len(coords)),
            "volume_mm3": vol_mm3,
            "diameter_mm": float(diam),
            "elongation": float(elong),
            "centroid_zyx_voxel": [float(v) for v in c],
            "bbox_zyx_voxel": [int(v) for v in bmin] + [int(v) for v in bmax],
        })
    out.sort(key=lambda n: n["volume_mm3"], reverse=True)
    return out, lab


def render_3d_html(lung_mask, pred_mask, voxel_sp, labeled_pred, nodules, title="",
                   max_nodule_meshes=30):
    """Plotly Mesh3d: lung shell (transparent) + per-nodule meshes (solid red).

    For speed, only the top-N largest nodules are rendered as meshes.
    Lung mesh uses coarse step_size=3 to keep triangle count manageable.
    """
    fig = go.Figure()

    if lung_mask.sum() > 1000:
        try:
            v, f, _, _ = measure.marching_cubes(
                lung_mask.astype(np.float32), level=0.5, spacing=voxel_sp, step_size=3,
            )
            fig.add_trace(go.Mesh3d(
                x=v[:, 2], y=v[:, 1], z=v[:, 0],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="#a5d8ff", opacity=0.10,
                name="Phổi (lung shell)", showlegend=True,
                flatshading=True, hoverinfo="skip",
            ))
        except Exception as e:
            print("lung mesh failed:", e)

    palette = ["#ff4d4d", "#ff8c42", "#ffd166", "#ef476f",
               "#f15bb5", "#9b5de5", "#ff5d8f", "#fb8500"]
    nodules_to_render = nodules[:max_nodule_meshes]
    for k, nod in enumerate(nodules_to_render):
        nid = nod["id"]
        zmin, ymin, xmin, zmax, ymax, xmax = nod["bbox_zyx_voxel"]
        # Pad bbox by 1 voxel for clean marching cubes boundary
        zmin = max(0, zmin - 1); ymin = max(0, ymin - 1); xmin = max(0, xmin - 1)
        zmax = min(labeled_pred.shape[0], zmax + 1)
        ymax = min(labeled_pred.shape[1], ymax + 1)
        xmax = min(labeled_pred.shape[2], xmax + 1)
        blob = (labeled_pred[zmin:zmax, ymin:ymax, xmin:xmax] == nid).astype(np.float32)
        if blob.sum() < MIN_NODULE_VOXELS:
            continue
        try:
            v, f, _, _ = measure.marching_cubes(
                blob, level=0.5, spacing=voxel_sp, step_size=1,
            )
            # Translate vertices back to full-volume coordinates (mm)
            v[:, 0] += zmin * voxel_sp[0]
            v[:, 1] += ymin * voxel_sp[1]
            v[:, 2] += xmin * voxel_sp[2]
        except Exception:
            continue
        fig.add_trace(go.Mesh3d(
            x=v[:, 2], y=v[:, 1], z=v[:, 0],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=palette[k % len(palette)], opacity=0.85,
            name=f"Nodule #{nid} (~{nod['diameter_mm']:.1f} mm)",
            showlegend=True, flatshading=True,
        ))
    if len(nodules) > max_nodule_meshes:
        title += f"  (hiển thị {max_nodule_meshes}/{len(nodules)} nodule lớn nhất)"

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=14)),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X (mm)", color="white", gridcolor="#333"),
            yaxis=dict(title="Y (mm)", color="white", gridcolor="#333"),
            zaxis=dict(title="Z (mm)", color="white", gridcolor="#333"),
            bgcolor="rgb(15,17,23)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgb(15,17,23)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.4)"),
        height=720,
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="plot3d")
