"""Step 07 — Detect nodules in 3D for one or many patients.

For every (patient, series):
  1. Load full preprocessed CT volume (work/preprocessed/<pid>.h5)
  2. Run UNet++ slice-by-slice (with TTA optional)
  3. Stack 2D predictions → 3D mask volume
  4. 3D connected-components: each nodule = 1 connected blob
  5. Drop blobs smaller than MIN_NODULE_VOXELS
  6. Compute per-nodule stats: centroid, bbox, voxels, volume mm³
  7. Save:
       outputs/<pid>/<series_short>/pred_volume.npy   (uint8 3D mask)
       outputs/<pid>/<series_short>/gt_volume.npy     (uint8 3D mask, union of 4 rad)
       outputs/<pid>/<series_short>/images.npy        (int16 CT volume)
       outputs/<pid>/<series_short>/meta.json         (geometry + nodule list)

Run:  python 07_detect_3d.py --patients LIDC-IDRI-0837,LIDC-IDRI-0976
      python 07_detect_3d.py --split test --top 10   (top-N by GT nodule size)
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import label as cc_label

from configs import (
    PRE_DIR, SPLITS_JSON, OUTPUTS, RUNS_DIR,
    HU_LO, HU_HI, MIN_NODULE_VOXELS, THRESHOLD_DEFAULT, TTA_ENABLED,
)
from dataset import LIDCSeriesVolume
from importlib import import_module
make_model = import_module("05_train").make_model


def normalize(x):
    x = np.clip(x.astype(np.float32), HU_LO, HU_HI)
    return (x - HU_LO) / (HU_HI - HU_LO)


def predict_volume(images, model, device, threshold=THRESHOLD_DEFAULT, tta=True):
    """images: [N, H, W] int16. returns prob_volume [N, H, W] float32 + binary mask."""
    N, H, W = images.shape
    prob = np.zeros((N, H, W), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            ip, ine = max(0, i - 1), min(N - 1, i + 1)
            stk = np.stack([normalize(images[ip]),
                            normalize(images[i]),
                            normalize(images[ine])], axis=0)
            x = torch.from_numpy(stk).unsqueeze(0).float().to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                p1 = torch.sigmoid(model(x))
                if tta:
                    p2 = torch.sigmoid(model(torch.flip(x, dims=[-1]))).flip(dims=[-1])
                    p3 = torch.sigmoid(model(torch.flip(x, dims=[-2]))).flip(dims=[-2])
                    p4 = torch.sigmoid(model(torch.flip(x, dims=[-1, -2]))).flip(dims=[-1, -2])
                    p = (p1 + p2 + p3 + p4) / 4
                else:
                    p = p1
            prob[i] = p[0, 0].float().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    return prob, mask


def find_nodules(mask_3d, voxel_spacing_mm, min_voxels=MIN_NODULE_VOXELS):
    """Return list of nodule dicts with centroid, bbox, voxels, volume_mm3."""
    # 3D 26-connectivity
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, n = cc_label(mask_3d, structure=structure)
    nodules = []
    for blob_id in range(1, n + 1):
        coords = np.argwhere(labeled == blob_id)
        if len(coords) < min_voxels:
            continue
        centroid = coords.mean(axis=0)             # (z, y, x) voxel
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0) + 1
        z_sp, y_sp, x_sp = voxel_spacing_mm
        volume_mm3 = float(len(coords)) * z_sp * y_sp * x_sp
        # equivalent diameter of a sphere of same volume
        diam_mm = 2 * (3 * volume_mm3 / (4 * np.pi)) ** (1 / 3)
        nodules.append({
            "id": int(blob_id),
            "voxels": int(len(coords)),
            "volume_mm3": volume_mm3,
            "diameter_mm": float(diam_mm),
            "centroid_zyx_voxel": [float(c) for c in centroid],
            "bbox_zyx_voxel": [int(v) for v in bbox_min] + [int(v) for v in bbox_max],
        })
    nodules.sort(key=lambda n: n["volume_mm3"], reverse=True)
    return nodules


def select_patients(args, splits):
    PRE_DIR_ARG = Path(args.root)
    if args.patients:
        return [p.strip() for p in args.patients.split(",") if p.strip()]
    cands = splits[args.split]
    if args.top is None or args.top <= 0:
        return cands
    # Score by total GT mask voxels (biggest nodules first → nicer 3D)
    scored = []
    for pid in cands[:60]:
        h5p = PRE_DIR_ARG / f"{pid}.h5"
        if not h5p.exists(): continue
        with h5py.File(h5p) as f:
            tot = 0
            for k in f:
                m = (f[k]["mask_per_rad"][:].sum(axis=1) >= 1)
                tot += int(m.sum())
        scored.append((tot, pid))
    scored.sort(reverse=True)
    return [pid for _, pid in scored[:args.top]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUNS_DIR / "best.pt"))
    ap.add_argument("--root", default=str(PRE_DIR))
    ap.add_argument("--splits", default=str(SPLITS_JSON))
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    ap.add_argument("--tta", type=int, default=int(TTA_ENABLED))
    ap.add_argument("--min_voxels", type=int, default=MIN_NODULE_VOXELS)
    ap.add_argument("--patients", default="", help="Comma-separated; overrides --split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    splits = json.load(open(args.splits))
    pids = select_patients(args, splits)
    print(f"Will process {len(pids)} patients: {pids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder = ck.get("encoder", "efficientnet-b4")
    model = make_model(encoder=encoder, pretrained=None).to(device)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.train(False)
    print(f"Model loaded: encoder={encoder}, threshold={args.threshold}, tta={bool(args.tta)}")

    PRE_DIR_M = Path(args.root)
    for pid in pids:
        h5p = PRE_DIR_M / f"{pid}.h5"
        if not h5p.exists():
            print(f"[skip] {pid}: no h5"); continue
        with h5py.File(h5p) as f:
            keys = list(f.keys())
        for k in keys:
            try:
                vol = LIDCSeriesVolume(h5p, k)
            except Exception as e:
                print(f"[skip] {pid}/{k}: {e}"); continue
            short = k.split("_")[-1][:8]
            out_dir = OUTPUTS / pid / short
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{pid}/{short}] vol shape {vol.shape}, "
                  f"voxel_sp {tuple(round(s,2) for s in vol.voxel_spacing)} mm")

            prob, pred = predict_volume(vol.images, model, device,
                                        threshold=args.threshold, tta=bool(args.tta))
            gt = vol.gt_mask("union")
            nodules = find_nodules(pred, vol.voxel_spacing, args.min_voxels)

            np.save(out_dir / "images.npy", vol.images)
            np.save(out_dir / "pred_volume.npy", pred)
            np.save(out_dir / "gt_volume.npy", gt)
            meta = {
                "patient_id": pid,
                "series_uid": vol.h5_path.stem,
                "series_key": k,
                "shape_zyx": list(vol.shape),
                "voxel_spacing_zyx_mm": list(vol.voxel_spacing),
                "image_position_first": vol.image_position_first,
                "image_orientation": vol.image_orientation,
                "z_positions": vol.z_positions.tolist(),
                "threshold": args.threshold,
                "tta": bool(args.tta),
                "min_nodule_voxels": args.min_voxels,
                "n_predicted_nodules": len(nodules),
                "predicted_nodules": nodules,
                "gt_voxels_total": int(gt.sum()),
                "pred_voxels_total": int(pred.sum()),
                "nodule_meta_xml": vol.nodule_meta,
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"  GT vox {int(gt.sum())}, Pred vox {int(pred.sum())}, "
                  f"detected {len(nodules)} nodule(s)")
            for n in nodules:
                print(f"    nodule {n['id']}: "
                      f"{n['voxels']} vox, {n['volume_mm3']:.0f} mm³, "
                      f"~{n['diameter_mm']:.1f} mm diameter")

    print(f"\nDone. Artifacts in {OUTPUTS}/<patient>/<series>/")
    print("Next: python 08_render_3d.py")


if __name__ == "__main__":
    main()
