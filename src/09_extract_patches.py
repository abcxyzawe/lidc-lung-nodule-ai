"""Step 09 — Extract 3D patches around each annotated nodule with malignancy label.

For each (radiologist, nodule_id) in every patient's preprocessed h5:
  - Get all slices in which this radiologist drew a contour for this nodule
  - Compute 3D centroid of the union mask
  - Crop a 32x32x32 patch (HU values) around centroid, padding if near border
  - Record malignancy rating (1-5)

Outputs:
  work/malignancy_patches.npz  with arrays:
    patches  [N, 32, 32, 32] int16
    labels   [N] int8   (1..5; -1 if missing)
    pids     [N] str    patient IDs
    rad_ch   [N] int8   radiologist channel
    nod_id   [N] str    nodule ID

Use splits.json to keep train/val/test consistent with seg model.

Run:  python 09_extract_patches.py
"""
import json
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np

from configs import PRE_DIR, WORK


PATCH = 32
HALF = PATCH // 2


def extract_one(h5_path, pid):
    out = []
    try:
        f = h5py.File(h5_path, "r")
    except Exception:
        return out
    try:
        for series_key in f.keys():
            g = f[series_key]
            try:
                nodule_meta = json.loads(g.attrs["nodule_meta"])
            except Exception:
                continue
            images = g["images"][...]
            mask_per_rad = g["mask_per_rad"][...]
            N, H, W = images.shape

            # Group entries by (rad_channel, nodule_id) — one nodule per radiologist
            by_nodule = defaultdict(list)
            for e in nodule_meta:
                mal = e.get("malignancy", "")
                if not mal or not mal.strip().isdigit():
                    continue
                key = (int(e["rad_channel"]), str(e.get("nodule_id", "")))
                by_nodule[key].append(e)

            for (rad_ch, nod_id), entries in by_nodule.items():
                slice_idxs = [int(e["slice_idx"]) for e in entries
                              if 0 <= int(e["slice_idx"]) < N]
                if not slice_idxs:
                    continue
                mal_vals = [int(e["malignancy"]) for e in entries
                            if e["malignancy"].isdigit()]
                if not mal_vals:
                    continue
                mal = int(round(np.mean(mal_vals)))

                sub_mask = mask_per_rad[slice_idxs, rad_ch]
                if sub_mask.sum() < 5:
                    continue
                coords = np.argwhere(sub_mask > 0)
                cz_local = int(coords[:, 0].mean())
                cz_global = slice_idxs[min(cz_local, len(slice_idxs) - 1)]
                cy = int(coords[:, 1].mean())
                cx = int(coords[:, 2].mean())

                zmin = max(0, cz_global - HALF)
                zmax = min(N, cz_global + HALF)
                ymin = max(0, cy - HALF); ymax = min(H, cy + HALF)
                xmin = max(0, cx - HALF); xmax = min(W, cx + HALF)
                crop = images[zmin:zmax, ymin:ymax, xmin:xmax].astype(np.int16)

                # Pad to PATCH³ centered as best we can
                pad_z0 = HALF - (cz_global - zmin)
                pad_y0 = HALF - (cy - ymin)
                pad_x0 = HALF - (cx - xmin)
                padded = np.full((PATCH, PATCH, PATCH), -1024, dtype=np.int16)  # air
                padded[
                    pad_z0:pad_z0 + crop.shape[0],
                    pad_y0:pad_y0 + crop.shape[1],
                    pad_x0:pad_x0 + crop.shape[2],
                ] = crop

                out.append({
                    "patch": padded,
                    "label": np.int8(mal),
                    "pid": pid,
                    "rad_ch": np.int8(rad_ch),
                    "nod_id": nod_id,
                })
    finally:
        f.close()
    return out


def main():
    pids = sorted(p.stem for p in PRE_DIR.glob("*.h5"))
    print(f"Scanning {len(pids)} preprocessed patients ...")

    all_patches, all_labels, all_pids, all_rad, all_nod = [], [], [], [], []
    for i, pid in enumerate(pids):
        items = extract_one(PRE_DIR / f"{pid}.h5", pid)
        for it in items:
            all_patches.append(it["patch"])
            all_labels.append(it["label"])
            all_pids.append(it["pid"])
            all_rad.append(it["rad_ch"])
            all_nod.append(it["nod_id"])
        if (i + 1) % 50 == 0 or i + 1 == len(pids):
            print(f"  [{i+1}/{len(pids)}] running total: {len(all_patches)} patches", flush=True)

    if not all_patches:
        print("No patches extracted! Check that nodule_meta has malignancy values.")
        return

    patches = np.stack(all_patches, 0)
    labels = np.asarray(all_labels, dtype=np.int8)
    pids_arr = np.asarray(all_pids)
    rad_arr = np.asarray(all_rad, dtype=np.int8)
    nod_arr = np.asarray(all_nod)

    out = WORK / "malignancy_patches.npz"
    np.savez_compressed(
        out, patches=patches, labels=labels,
        pids=pids_arr, rad_ch=rad_arr, nod_id=nod_arr,
    )
    print(f"\nSaved {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"Patches: {patches.shape}  HU range [{patches.min()}, {patches.max()}]")
    print(f"Label distribution (1=benign…5=malignant):")
    for lab in range(1, 6):
        n = int((labels == lab).sum())
        print(f"  malignancy={lab}: {n} patches ({100*n/len(labels):.1f}%)")
    print(f"\nNext: upload {out.name} to GPU container, then run 10_train_malignancy.py")


if __name__ == "__main__":
    main()
