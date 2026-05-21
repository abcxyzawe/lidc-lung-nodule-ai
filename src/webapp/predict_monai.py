"""MONAI Lung Nodule Detection wrapper.

Uses pretrained lung_nodule_ct_detection bundle (RetinaNet 3D, LUNA16).
Wraps the TorchScript network with MONAI's RetinaNetDetector to decode
anchors into actual bounding boxes.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


BUNDLE_DIR = Path(__file__).resolve().parent.parent.parent / "bundles" / "lung_nodule_ct_detection"
MODEL_TS = BUNDLE_DIR / "models" / "model.ts"

# MONAI bundle spec — spacing is (x, y, z) in mm
MONAI_SPACING_XYZ = (0.703125, 0.703125, 1.25)
HU_LOW, HU_HIGH = -1024.0, 300.0
ROI_SIZE = [192, 192, 80]
SCORE_THRESH = 0.10
NMS_IOU = 0.22

_detector = None


def get_detector(device=None):
    global _detector
    if _detector is not None:
        return _detector
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not MODEL_TS.exists():
        raise FileNotFoundError(f"MONAI bundle not found at {MODEL_TS}")

    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape

    net = torch.jit.load(str(MODEL_TS), map_location=device)
    net.eval()

    anchor_gen = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[1, 2, 4],
        base_anchor_shapes=[[6, 8, 4], [8, 6, 5], [10, 10, 6]],
    )
    det = RetinaNetDetector(
        network=net,
        anchor_generator=anchor_gen,
        debug=False,
        spatial_dims=3,
        num_classes=1,
        size_divisible=[16, 16, 8],
    )
    det.set_target_keys(box_key="box", label_key="label")
    det.set_box_selector_parameters(
        score_thresh=SCORE_THRESH,
        topk_candidates_per_level=1000,
        nms_thresh=NMS_IOU,
        detections_per_img=300,
    )
    det.set_sliding_window_inferer(
        roi_size=ROI_SIZE,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device="cpu",
    )
    det = det.to(device)
    det.eval()
    _detector = det
    print(f"[MONAI] Loaded {MODEL_TS} + RetinaNetDetector on {device}")
    return _detector


def _resample_zyx_to_xyz(vol_zyx, src_sp_zyx, dst_sp_xyz):
    """Resample volume; returns (vol_xyz, scale_xyz) where scale=dst_size/src_size per axis."""
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0)).astype(np.float32)  # (X, Y, Z)
    src_sp_xyz = (src_sp_zyx[2], src_sp_zyx[1], src_sp_zyx[0])
    scale = tuple(s / d for s, d in zip(src_sp_xyz, dst_sp_xyz))
    new_size = tuple(max(1, int(round(vol_xyz.shape[i] * scale[i]))) for i in range(3))
    t = torch.from_numpy(vol_xyz).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=new_size, mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy(), scale


def predict_nodules_monai(volume_hu_zyx, voxel_sp_zyx, score_thresh=SCORE_THRESH):
    """Run MONAI RetinaNet on HU volume. Returns nodules in our pipeline format."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = get_detector(device)

    # Resample to MONAI spacing
    vol_xyz, scale_xyz = _resample_zyx_to_xyz(volume_hu_zyx, voxel_sp_zyx, MONAI_SPACING_XYZ)
    # Intensity scale [-1024, 300] -> [0, 1]
    vol_xyz = np.clip(vol_xyz, HU_LOW, HU_HIGH)
    vol_xyz = (vol_xyz - HU_LOW) / (HU_HIGH - HU_LOW)

    # (X, Y, Z) -> (N=1, C=1, X, Y, Z)
    inp = torch.from_numpy(vol_xyz).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = det(inp, use_inferer=True)

    if not isinstance(out, list) or len(out) == 0:
        return []
    boxes_t = out[0].get("box")
    scores_t = out[0].get("label_scores")
    if boxes_t is None or scores_t is None or boxes_t.numel() == 0:
        return []

    boxes = boxes_t.detach().cpu().numpy()      # (N, 6) xyzxyz in resampled voxel space
    scores = scores_t.detach().cpu().numpy()
    keep = scores >= score_thresh
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return []

    sx, sy, sz = scale_xyz  # resampled = original * scale
    nodules = []
    for i, (box, sc) in enumerate(zip(boxes, scores), 1):
        bx1, by1, bz1, bx2, by2, bz2 = box
        # Convert resampled XYZ voxel -> original ZYX voxel
        ox1, oy1, oz1 = bx1 / sx, by1 / sy, bz1 / sz
        ox2, oy2, oz2 = bx2 / sx, by2 / sy, bz2 / sz
        zmin, zmax = int(round(min(oz1, oz2))), int(round(max(oz1, oz2)))
        ymin, ymax = int(round(min(oy1, oy2))), int(round(max(oy1, oy2)))
        xmin, xmax = int(round(min(ox1, ox2))), int(round(max(ox1, ox2)))
        if zmax <= zmin: zmax = zmin + 1
        if ymax <= ymin: ymax = ymin + 1
        if xmax <= xmin: xmax = xmin + 1
        cz = (zmin + zmax) / 2
        cy = (ymin + ymax) / 2
        cx = (xmin + xmax) / 2
        dz_mm = (zmax - zmin) * voxel_sp_zyx[0]
        dy_mm = (ymax - ymin) * voxel_sp_zyx[1]
        dx_mm = (xmax - xmin) * voxel_sp_zyx[2]
        diam = max(dx_mm, dy_mm, dz_mm)
        vox = max(1, (zmax - zmin) * (ymax - ymin) * (xmax - xmin))
        vol_mm3 = vox * voxel_sp_zyx[0] * voxel_sp_zyx[1] * voxel_sp_zyx[2]
        nodules.append({
            "id": i,
            "voxels": int(vox),
            "core_voxels": int(vox),
            "volume_mm3": float(vol_mm3),
            "diameter_mm": float(diam),
            "diameter_full_mm": float(diam),
            "elongation": float(max(dx_mm, dy_mm, dz_mm) / max(1e-3, min(dx_mm, dy_mm, dz_mm))),
            "centroid_zyx_voxel": [float(cz), float(cy), float(cx)],
            "bbox_zyx_voxel": [zmin, ymin, xmin, zmax, ymax, xmax],
            "fpr_prob": float(sc),
            "confidence": float(sc),
            "nodule_type": "detected",
        })

    nodules.sort(key=lambda n: -n["confidence"])
    for new_id, n in enumerate(nodules, 1):
        n["original_id"] = n["id"]
        n["id"] = new_id
    return nodules
