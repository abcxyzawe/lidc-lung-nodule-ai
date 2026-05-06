"""Step 04 — PyTorch Dataset + augmentation transforms.

Importable module (NOT a runnable step). 05_train, 06_evaluate, 07_detect import this.

Two flavors:
  LIDCSeg25D       — for training (1 sample = 3 stacked slices + center mask)
  LIDCSeriesVolume — for inference (returns whole CT volume of one series)
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from configs import (
    HU_LO, HU_HI, MASK_MODE_DEFAULT, MIN_MASK_PIXELS,
)


# ============================================================
# Slice-level Dataset for training
# ============================================================
def _normalize(img_int16):
    x = np.clip(img_int16.astype(np.float32), HU_LO, HU_HI)
    return (x - HU_LO) / (HU_HI - HU_LO)


def _build_mask(mask_per_rad, mode):
    if mode == "union":      return (mask_per_rad.sum(axis=1) >= 1).astype(np.uint8)
    if mode == "consensus2": return (mask_per_rad.sum(axis=1) >= 2).astype(np.uint8)
    if mode == "consensus3": return (mask_per_rad.sum(axis=1) >= 3).astype(np.uint8)
    if mode == "all4":       return (mask_per_rad.sum(axis=1) >= 4).astype(np.uint8)
    raise ValueError(mode)


class LIDCSeg25D(Dataset):
    """1 sample = (3 stacked slices, center-slice mask).

    Stacking 3 consecutive slices gives the model 3D context AND lets us use
    ImageNet-pretrained 3-channel encoders (efficientnet, resnet, ...).
    """
    def __init__(self, root, patient_ids, mask_mode=MASK_MODE_DEFAULT,
                 min_mask_pixels=MIN_MASK_PIXELS, augment=None,
                 only_with_mask=True):
        self.root = Path(root)
        self.mask_mode = mask_mode
        self.augment = augment
        self.entries = []   # (h5_path, key, center_idx, n_in_series)
        for pid in patient_ids:
            p = self.root / f"{pid}.h5"
            if not p.exists(): continue
            with h5py.File(p, "r") as f:
                for k in f.keys():
                    n = f[k]["images"].shape[0]
                    if only_with_mask:
                        m_rad = f[k]["mask_per_rad"][:]
                        m = _build_mask(m_rad, mask_mode)
                        for i in range(n):
                            if m[i].sum() >= min_mask_pixels:
                                self.entries.append((str(p), k, i, n))
                    else:
                        for i in range(n):
                            self.entries.append((str(p), k, i, n))
        print(f"LIDCSeg25D({mask_mode}, min_px={min_mask_pixels}, "
              f"aug={augment is not None}): {len(self.entries)} slices "
              f"from {len(patient_ids)} patients")

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        path, key, i, n = self.entries[idx]
        with h5py.File(path, "r") as f:
            ip, ine = max(0, i - 1), min(n - 1, i + 1)
            imgs = f[key]["images"][[ip, i, ine]]
            m_rad = f[key]["mask_per_rad"][i:i + 1]
        mask = _build_mask(m_rad, self.mask_mode)[0]
        imgs = np.stack([_normalize(imgs[c]) for c in range(3)], axis=0).astype(np.float32)
        mask = mask.astype(np.float32)[None]
        if self.augment is not None:
            imgs, mask = self.augment(imgs, mask)
        return torch.from_numpy(imgs), torch.from_numpy(mask)


# ============================================================
# Series-level loader for inference / 3D rendering
# ============================================================
class LIDCSeriesVolume:
    """Convenience reader for one (patient, series) volume + meta.

    NOT a torch Dataset — used directly by 07_detect_3d / 08_render_3d.
    """
    def __init__(self, h5_path, key):
        self.h5_path = Path(h5_path)
        self.key = key
        with h5py.File(self.h5_path) as f:
            g = f[key]
            self.images = g["images"][:].copy()             # [N, H, W] int16
            self.mask_per_rad = g["mask_per_rad"][:].copy() # [N, 4, H, W]
            self.sop_uids = [s.decode() for s in g["sop_uids"][:]]
            self.z_positions = g["z_positions"][:].copy()
            self.pixel_spacing = json.loads(g.attrs["pixel_spacing"])     # [y, x] mm
            self.slice_thickness = float(g.attrs["slice_thickness"])
            self.image_position_first = json.loads(g.attrs["image_position_first"])
            self.image_orientation = json.loads(g.attrs["image_orientation"])
            self.nodule_meta = json.loads(g.attrs["nodule_meta"])

        # Sort by z (should already be sorted, defensive)
        order = np.argsort(self.z_positions)
        self.images = self.images[order]
        self.mask_per_rad = self.mask_per_rad[order]
        self.sop_uids = [self.sop_uids[i] for i in order]
        self.z_positions = self.z_positions[order]

    @property
    def shape(self):
        return self.images.shape

    @property
    def z_spacing(self):
        if len(self.z_positions) > 1:
            diffs = np.diff(self.z_positions)
            return float(np.median(np.abs(diffs)))
        return self.slice_thickness

    @property
    def voxel_spacing(self):
        """Returns (z, y, x) in mm — needed for marching_cubes."""
        return (self.z_spacing, self.pixel_spacing[0], self.pixel_spacing[1])

    def gt_mask(self, mode="union"):
        return _build_mask(self.mask_per_rad, mode)


# ============================================================
# Augmentation pipeline (fast numpy)
# ============================================================
from scipy.ndimage import map_coordinates, gaussian_filter, rotate as nd_rotate


class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomFlip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img = img[:, :, ::-1].copy(); mask = mask[:, :, ::-1].copy()
        if np.random.rand() < self.p:
            img = img[:, ::-1, :].copy(); mask = mask[:, ::-1, :].copy()
        return img, mask


class RandomRotate:
    def __init__(self, max_deg=20, p=0.5): self.max = max_deg; self.p = p
    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            ang = np.random.uniform(-self.max, self.max)
            img = np.stack([nd_rotate(img[c], ang, reshape=False, order=1,
                                      mode="constant", cval=0)
                            for c in range(img.shape[0])])
            mask = np.stack([nd_rotate(mask[c], ang, reshape=False, order=0,
                                       mode="constant", cval=0)
                             for c in range(mask.shape[0])])
        return img, mask


class RandomIntensity:
    def __init__(self, shift=0.05, scale=0.1, p=0.5):
        self.shift = shift; self.scale = scale; self.p = p
    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            s = 1.0 + np.random.uniform(-self.scale, self.scale)
            b = np.random.uniform(-self.shift, self.shift)
            img = np.clip(img * s + b, 0, 1).astype(np.float32)
        return img, mask


class RandomNoise:
    def __init__(self, std=0.02, p=0.3):
        self.std = std; self.p = p
    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img = (img + np.random.randn(*img.shape).astype(np.float32) * self.std).clip(0, 1)
        return img, mask


class RandomElastic:
    def __init__(self, alpha=80, sigma=10, p=0.3):
        self.alpha = alpha; self.sigma = sigma; self.p = p
    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            H, W = img.shape[1], img.shape[2]
            dx = gaussian_filter(np.random.randn(H, W) * self.alpha, self.sigma)
            dy = gaussian_filter(np.random.randn(H, W) * self.alpha, self.sigma)
            ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            yi = (ys + dy).astype(np.float32); xi = (xs + dx).astype(np.float32)
            img = np.stack([map_coordinates(img[c], [yi, xi], order=1,
                                            mode="constant").reshape(H, W)
                            for c in range(img.shape[0])])
            mask = np.stack([map_coordinates(mask[c], [yi, xi], order=0,
                                             mode="constant").reshape(H, W)
                             for c in range(mask.shape[0])])
        return img, mask


def get_train_aug():
    return Compose([
        RandomFlip(p=0.5),
        RandomRotate(max_deg=20, p=0.5),
        RandomIntensity(shift=0.05, scale=0.1, p=0.5),
        RandomElastic(alpha=80, sigma=10, p=0.3),
        RandomNoise(std=0.02, p=0.3),
    ])
