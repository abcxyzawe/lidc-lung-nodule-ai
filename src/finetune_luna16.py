"""Fine-tune or train-from-scratch LIDC/LUNA16 nodule detector.

Two modes
---------
Fine-tune mode (default, backward-compat):
  - Warm-start from LIDC best.pt (UNet++ EfficientNet-B5, val_dice=0.8676 ep 16).
  - Discriminative LR: 5e-5 backbone, 2e-4 decoder+head.
  - Schedule: CosineAnnealingWarmRestarts T_0=10 T_mult=2 + 3-ep linear warmup.
  - Optimizer: AdamW, weight_decay 5e-5.
  - Freeze 3-phase: ep 1-5 (blocks 0-3), ep 6-20 (blocks 2-3), ep 21+ full unfreeze.

Train-from-scratch mode (--no-warm):
  - Random init (ImageNet encoder weights=None, no ckpt loaded).
  - Uniform LR across all param groups (single --lr-backbone value used).
  - No freeze phases — all parameters trainable from epoch 1.
  - Longer training recommended (80+ epochs) since model converges from random init.
  - Avoids catastrophic forgetting that can occur when fine-tuning from LIDC winner.

Common settings (both modes):
  - Batch: 6 + grad_accum 2 (effective 12).
  - Loss: Tversky(alpha=0.3, beta=0.7) + 0.5 * Focal-BCE.
  - Negative sampler: 40% pos / 30% border / 30% hard-neg.
  - SWA: last N epochs, LR const 1e-6.
  - NEVER overwrites locked LIDC ckpts in work/runs/ or work/runs_exp/stage2_full/.

Expected LUNA16 directory layout:
  <luna16-dir>/
    subset0/ ... subset9/        # .mhd + .raw files
    annotations.csv              # seriesuid,coordX,coordY,coordZ,diameter_mm
    seg-lungs-LUNA16/            # pre-computed lung masks (optional speedup)

Usage (fine-tune, original):
  python src/finetune_luna16.py \\
      --luna16-dir /workspace/datasetLuna16 \\
      --warm-ckpt  /workspace/work/runs_luna16/init.pth \\
      --exclude-json work/academic/LUNA16_EXCLUDE_FROM_TRAIN.json \\
      --train-subsets 0,1,2,3,4,5,6 --val-subsets 7,8 --test-subsets 9 \\
      --epochs 40 --batch-size 6 --grad-accum 2 \\
      --output-dir /workspace/work/runs_luna16/run001 \\
      --wandb-project lidc-luna16-finetune

Usage (train from scratch, --no-warm):
  python src/finetune_luna16.py \\
      --luna16-dir /workspace/datasetLuna16 \\
      --no-warm \\
      --exclude-json work/academic/LUNA16_EXCLUDE_FROM_TRAIN.json \\
      --train-subsets 0,1,5,6,7,8 --val-subsets 9 \\
      --epochs 80 --batch-size 6 --grad-accum 2 \\
      --lr-backbone 1e-4 \\
      --output-dir /workspace/work/runs_luna16/run002_scratch \\
      --no-wandb --seed 42

Smoke test (2 ep, 5 patients, CPU/GPU):
  python src/finetune_luna16.py --smoke 2 \\
      --luna16-dir /workspace/datasetLuna16 \\
      --no-warm \\
      --output-dir /workspace/work/runs_luna16/smoke

Estimated runtime (V100 32GB, batch 6+accum2, ~590 train scans):
  _build_index (one-off, CPU):  15-25 min
  Per epoch forward+backward:   18-25 min
  Total fine-tune (40 ep):      13-17 h
  Total from-scratch (80 ep):   26-34 h (~95-120s/ep)
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Constants — NEVER hard-code thresholds inline; change here or via args
# ---------------------------------------------------------------------------
HU_LO_FINETUNE = -1000.0   # LUNA16 window (slightly wider than LIDC HU_LO)
HU_HI_FINETUNE = 200.0
PATCH_HW = 128              # spatial patch size (px)
PATCH_SLICES = 5            # 2.5D: 5 consecutive slices
LUNA16_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
FROC_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
# Freeze phase boundaries (encoder block indices for EfficientNet-B5)
FREEZE_PHASE1_BLOCKS = list(range(0, 4))   # ep 1-5
FREEZE_PHASE2_BLOCKS = list(range(2, 4))   # ep 6-20
# Sampler target proportion: (pos, border, hard_neg)
NEG_SAMPLER_RATIO = (0.40, 0.30, 0.30)
PATIENCE_DEFAULT = 8
EARLY_STOP_METRIC = "val_froc_cpm"


# ---------------------------------------------------------------------------
# LUNA16 I/O helpers
# ---------------------------------------------------------------------------

def parse_luna16_annotations(csv_path: Path) -> dict:
    """Parse LUNA16 annotations.csv.

    Returns dict mapping seriesuid -> list of annotation dicts:
        {"coord_world_xyz": [x, y, z], "diameter_mm": float}
    """
    df = pd.read_csv(csv_path)
    out: dict = {}
    for _, row in df.iterrows():
        uid = str(row["seriesuid"])
        entry = {
            "coord_world_xyz": [float(row["coordX"]),
                                 float(row["coordY"]),
                                 float(row["coordZ"])],
            "diameter_mm": float(row["diameter_mm"]),
        }
        out.setdefault(uid, []).append(entry)
    return out


def load_luna16_scan(mhd_path: Path):
    """Load LUNA16 .mhd scan via SimpleITK.

    Returns:
        vol_zyx     : np.ndarray [Z, Y, X] int16 (Hounsfield units)
        spacing_zyx : tuple (sz, sy, sx) mm
        origin_zyx  : tuple (oz, oy, ox) mm

    SimpleITK uses XYZ order; we reverse to ZYX to match LIDC convention.
    """
    img = sitk.ReadImage(str(mhd_path))
    vol_zyx = sitk.GetArrayFromImage(img).astype(np.int16)
    spacing_zyx = tuple(reversed(img.GetSpacing()))
    origin_zyx = tuple(reversed(img.GetOrigin()))
    return vol_zyx, spacing_zyx, origin_zyx


def world_to_voxel(coord_world_xyz: list, origin_zyx: tuple,
                   spacing_zyx: tuple) -> np.ndarray:
    """Convert LUNA16 world coordinate (XYZ mm) to voxel index (ZYX).

    LUNA16 annotations are in world space (mm) with XYZ ordering.
    """
    x, y, z = coord_world_xyz
    oz, oy, ox = origin_zyx
    sz, sy, sx = spacing_zyx
    return np.array([(z - oz) / sz,
                     (y - oy) / sy,
                     (x - ox) / sx], dtype=np.float32)


def normalize_hu(vol: np.ndarray,
                 lo: float = HU_LO_FINETUNE,
                 hi: float = HU_HI_FINETUNE) -> np.ndarray:
    """Clip HU and map to [0, 1]."""
    x = np.clip(vol.astype(np.float32), lo, hi)
    return (x - lo) / (hi - lo)


def load_lung_mask(mhd_path: Path, luna16_dir: Path) -> np.ndarray | None:
    """Try to load pre-computed lung mask from seg-lungs-LUNA16/.

    Returns binary ZYX uint8 array, or None if mask file absent.
    Mask files follow LUNA16 naming: <seriesuid>.mhd in seg-lungs-LUNA16/
    """
    mask_dir = luna16_dir / "seg-lungs-LUNA16"
    uid = mhd_path.stem
    mask_path = mask_dir / f"{uid}.mhd"
    if not mask_path.exists():
        return None
    img = sitk.ReadImage(str(mask_path))
    return (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Augmentation helpers (stage2-minus-mixup as per design)
# ---------------------------------------------------------------------------

def _aug_flip(vol5, mask2d, rng):
    """Random horizontal + vertical flip on 2D patch; applied to all 5 slices."""
    if rng.random() > 0.5:
        vol5 = np.flip(vol5, axis=2).copy()
        if mask2d is not None:
            mask2d = np.flip(mask2d, axis=1).copy()
    if rng.random() > 0.5:
        vol5 = np.flip(vol5, axis=1).copy()
        if mask2d is not None:
            mask2d = np.flip(mask2d, axis=0).copy()
    return vol5, mask2d


def _aug_rotate90(vol5, mask2d, rng):
    """Random 0/90/180/270 rotation on HW axes."""
    k = rng.randint(0, 4)
    if k == 0:
        return vol5, mask2d
    vol5 = np.rot90(vol5, k=k, axes=(1, 2)).copy()
    if mask2d is not None:
        mask2d = np.rot90(mask2d, k=k, axes=(0, 1)).copy()
    return vol5, mask2d


def _aug_noise(vol5, rng, sigma_max=0.02):
    """Additive Gaussian noise."""
    if rng.random() > 0.5:
        sigma = rng.uniform(0.0, sigma_max)
        noise = np.random.normal(0, sigma, vol5.shape).astype(np.float32)
        vol5 = vol5 + noise
        vol5 = np.clip(vol5, 0.0, 1.0)
    return vol5


def apply_aug(vol5, mask2d, rng):
    """vol5: [5, H, W] float32; mask2d: [H, W] float32 or None."""
    vol5, mask2d = _aug_flip(vol5, mask2d, rng)
    vol5, mask2d = _aug_rotate90(vol5, mask2d, rng)
    vol5 = _aug_noise(vol5, rng)
    return vol5, mask2d


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LunaDataset(Dataset):
    """LUNA16 2.5D patch dataset for fine-tuning.

    Yields (input_tensor, mask_tensor, label):
      input_tensor : [5, PATCH_HW, PATCH_HW] float32 normalized
      mask_tensor  : [1, PATCH_HW, PATCH_HW] float32 binary (1 = nodule voxel)
      label        : int  1=positive  0=negative

    Sample index entry: (mhd_path, cz, cy, cx, label, diameter_mm, sample_class)
      sample_class: "pos" | "border" | "hard_neg"
    """

    LABEL_POS = 1
    LABEL_NEG = 0

    def __init__(self, scan_paths: list, annotations: dict,
                 luna16_dir: Path,
                 patch_hw: int = PATCH_HW,
                 n_slices: int = PATCH_SLICES,
                 neg_sampler_ratio: tuple = NEG_SAMPLER_RATIO,
                 augment: bool = True,
                 seed: int = 42,
                 smoke_n_scans: int = 0):
        self.scan_paths = scan_paths
        self.annotations = annotations
        self.luna16_dir = luna16_dir
        self.patch_hw = patch_hw
        self.n_slices = n_slices
        self.neg_sampler_ratio = neg_sampler_ratio
        self.augment = augment
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._samples: list = []   # (mhd_path, cz, cy, cx, label, diam_mm, cls_str)
        if smoke_n_scans > 0:
            scan_paths = scan_paths[:smoke_n_scans]
        self._build_index(scan_paths)

    # ------------------------------------------------------------------
    def _build_index(self, scan_paths):
        """Build self._samples from scan_paths + annotations dict.

        For each .mhd scan:
          1. Load vol to get shape/spacing/origin.
          2. For each annotation: world_to_voxel, append positive sample.
          3. Build negative pool from lung mask (or random fallback).
          4. Apply 40/30/30 sampler by weighting sample classes.
        """
        pos_samples = []
        border_samples = []
        hard_neg_samples = []

        half_hw = self.patch_hw // 2
        half_sl = self.n_slices // 2

        n_scans = 0
        for mhd_path in scan_paths:
            mhd_path = Path(mhd_path)
            uid = mhd_path.stem
            nods = self.annotations.get(uid, [])

            try:
                vol_zyx, spacing_zyx, origin_zyx = load_luna16_scan(mhd_path)
            except Exception as e:
                print(f"  WARN: could not load {mhd_path}: {e}", flush=True)
                continue

            nD, nH, nW = vol_zyx.shape

            # -- Positive samples (one per annotation centroid) --
            nodule_vox_centroids = []
            for nod in nods:
                vox = world_to_voxel(nod["coord_world_xyz"], origin_zyx, spacing_zyx)
                cz = int(round(float(vox[0])))
                cy = int(round(float(vox[1])))
                cx = int(round(float(vox[2])))
                # Bounds check: patch must be fully inside volume
                if (half_sl <= cz < nD - half_sl and
                        half_hw <= cy < nH - half_hw and
                        half_hw <= cx < nW - half_hw):
                    diam_mm = nod["diameter_mm"]
                    pos_samples.append((str(mhd_path), cz, cy, cx,
                                        self.LABEL_POS, diam_mm, "pos"))
                    nodule_vox_centroids.append((cz, cy, cx, diam_mm))

            # -- Negative pool from lung mask or random --
            lung_mask = load_lung_mask(mhd_path, self.luna16_dir)

            if lung_mask is not None:
                # Sample within lung (not within any nodule region)
                # Use a 2x oversample, then filter out near-nodule positions
                z_inds, y_inds, x_inds = np.where(lung_mask > 0)
                # Restrict to valid patch region
                valid = (
                    (z_inds >= half_sl) & (z_inds < nD - half_sl) &
                    (y_inds >= half_hw) & (y_inds < nH - half_hw) &
                    (x_inds >= half_hw) & (x_inds < nW - half_hw)
                )
                z_inds = z_inds[valid]
                y_inds = y_inds[valid]
                x_inds = x_inds[valid]
                # Down-sample candidate pool to max 2000 per scan for speed
                pool_size = min(len(z_inds), 2000)
                if pool_size > 0:
                    idxs = self.np_rng.choice(len(z_inds), size=pool_size, replace=False)
                    cands = list(zip(z_inds[idxs].tolist(),
                                     y_inds[idxs].tolist(),
                                     x_inds[idxs].tolist()))
                else:
                    cands = []
            else:
                # Fallback: random sampling within volume bounds
                cands = []
                for _ in range(600):
                    cz_r = int(self.np_rng.integers(half_sl, nD - half_sl))
                    cy_r = int(self.np_rng.integers(half_hw, nH - half_hw))
                    cx_r = int(self.np_rng.integers(half_hw, nW - half_hw))
                    cands.append((cz_r, cy_r, cx_r))

            # Filter: exclude positions within 1.5× radius of any nodule centroid
            def _near_nodule(cz, cy, cx, nod_cents):
                sz, sy, sx = spacing_zyx
                for nz, ny, nx, diam in nod_cents:
                    dist_mm = np.sqrt(
                        ((cz - nz) * sz) ** 2 +
                        ((cy - ny) * sy) ** 2 +
                        ((cx - nx) * sx) ** 2
                    )
                    if dist_mm < max(diam * 0.75, 6.0):
                        return True
                return False

            clean_negs = []
            border_negs = []
            for cz_c, cy_c, cx_c in cands:
                if _near_nodule(cz_c, cy_c, cx_c, nodule_vox_centroids):
                    # Within 1.5-3× radius = "border" (harder negatives)
                    border_negs.append((str(mhd_path), cz_c, cy_c, cx_c,
                                        self.LABEL_NEG, 0.0, "border"))
                else:
                    clean_negs.append((str(mhd_path), cz_c, cy_c, cx_c,
                                       self.LABEL_NEG, 0.0, "hard_neg"))

            # Keep balanced: at most 2× positives per scan in each neg class
            n_pos_scan = max(len(nodule_vox_centroids), 1)
            border_samples.extend(border_negs[:n_pos_scan * 2])
            hard_neg_samples.extend(
                self.rng.sample(clean_negs, min(len(clean_negs), n_pos_scan * 2))
            )
            n_scans += 1

        self._samples = pos_samples + border_samples + hard_neg_samples
        n_pos = len(pos_samples)
        n_brd = len(border_samples)
        n_neg = len(hard_neg_samples)
        print(
            f"LunaDataset: {n_scans} scans  "
            f"pos={n_pos}  border={n_brd}  hard_neg={n_neg}  "
            f"total={len(self._samples)}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def make_sampler(self) -> WeightedRandomSampler:
        """Return WeightedRandomSampler that hits ~40/30/30 proportion."""
        cls_map = {"pos": 0, "border": 1, "hard_neg": 2}
        counts = [0, 0, 0]
        for s in self._samples:
            counts[cls_map[s[6]]] += 1
        counts_arr = np.array(counts, dtype=np.float64)
        counts_arr[counts_arr == 0] = 1.0
        target = np.array(self.neg_sampler_ratio, dtype=np.float64)
        w_per_class = target / counts_arr
        w_per_class = w_per_class / w_per_class.max()
        weights = np.array([w_per_class[cls_map[s[6]]] for s in self._samples],
                           dtype=np.float64)
        print(
            f"  Sampler weights: pos={w_per_class[0]:.3f} "
            f"border={w_per_class[1]:.3f} hard_neg={w_per_class[2]:.3f}",
            flush=True,
        )
        return WeightedRandomSampler(
            weights.tolist(), num_samples=len(self._samples), replacement=True
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        """Load patch, normalize HU, return (vol_5slice, mask, label).

        vol_5slice : [5, PATCH_HW, PATCH_HW] float32
        mask       : [1, PATCH_HW, PATCH_HW] float32
        label      : torch.long scalar
        """
        mhd_path_str, cz, cy, cx, label, diam_mm, _ = self._samples[idx]
        mhd_path = Path(mhd_path_str)

        vol_zyx, spacing_zyx, _ = load_luna16_scan(mhd_path)
        nD, nH, nW = vol_zyx.shape
        half_sl = self.n_slices // 2
        half_hw = self.patch_hw // 2

        # -- Extract 5-slice stack and 2D center mask --
        # Clamp z-indices to volume bounds (edge padding via repeat)
        z_idxs = [max(0, min(nD - 1, cz + off))
                  for off in range(-half_sl, half_sl + 1)]
        slices_raw = vol_zyx[z_idxs,
                             max(0, cy - half_hw):cy + half_hw,
                             max(0, cx - half_hw):cx + half_hw]  # [5, H', W']

        # Pad to exact [5, patch_hw, patch_hw] if near boundary
        if slices_raw.shape[1] < self.patch_hw or slices_raw.shape[2] < self.patch_hw:
            pad_h = self.patch_hw - slices_raw.shape[1]
            pad_w = self.patch_hw - slices_raw.shape[2]
            slices_raw = np.pad(
                slices_raw,
                ((0, 0), (0, max(pad_h, 0)), (0, max(pad_w, 0))),
                mode="edge",
            )[:, :self.patch_hw, :self.patch_hw]

        vol5 = normalize_hu(slices_raw)   # [5, patch_hw, patch_hw] float32

        # Build binary segmentation mask for center slice
        # (nodule pixels = circle of radius diam_mm/2 centred at patch centre)
        mask2d = np.zeros((self.patch_hw, self.patch_hw), dtype=np.float32)
        if label == self.LABEL_POS and diam_mm > 0:
            sz, sy, sx = spacing_zyx
            r_pix_y = diam_mm / (2.0 * sy)
            r_pix_x = diam_mm / (2.0 * sx)
            yy, xx = np.ogrid[:self.patch_hw, :self.patch_hw]
            cy_local = self.patch_hw // 2
            cx_local = self.patch_hw // 2
            inside = ((yy - cy_local) ** 2 / max(r_pix_y ** 2, 1e-4) +
                      (xx - cx_local) ** 2 / max(r_pix_x ** 2, 1e-4)) <= 1.0
            mask2d[inside] = 1.0

        # Augmentation
        if self.augment:
            vol5, mask2d = apply_aug(vol5, mask2d, self.rng)

        vol_t = torch.from_numpy(vol5.astype(np.float32))             # [5, H, W]
        mask_t = torch.from_numpy(mask2d[None].astype(np.float32))    # [1, H, W]
        lbl_t = torch.tensor(label, dtype=torch.long)
        return vol_t, mask_t, lbl_t


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_warm_ckpt(ckpt_path: Path, device: torch.device, in_channels: int = 5):
    """Load LIDC winner checkpoint. Adapts 3-ch input conv to 5-ch if needed.

    Returns (model, meta_dict).
    LOCKED: never overwrites the source checkpoint.
    """
    import segmentation_models_pytorch as smp
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_in_ch = ck.get("in_channels", 3)
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b5", encoder_weights=None,
        in_channels=in_channels, classes=1, decoder_attention_type="scse",
    )
    if ckpt_in_ch != in_channels:
        # Average 3-ch input conv weights, expand to 5-ch (preserve magnitude)
        print(f"  Adapting {ckpt_in_ch}-ch ckpt input conv -> {in_channels}-ch by averaging")
        sd = ck["model"]
        in_conv_keys = [k for k in sd if "conv_stem.weight" in k or "_conv_stem.weight" in k]
        if in_conv_keys:
            key = in_conv_keys[0]
            w3 = sd[key]   # [out_ch, 3, kH, kW]
            w_avg = w3.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            w_avg = w_avg * (ckpt_in_ch / in_channels)
            sd[key] = w_avg
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(ck["model"], strict=True)
    model = model.to(device)
    meta = {k: v for k, v in ck.items() if k != "model"}
    print(f"Loaded warm ckpt: epoch={meta.get('epoch')}, "
          f"val_dice={meta.get('val_dice', 'N/A')}")
    return model, meta


def build_scratch_model(device: torch.device, in_channels: int = 5):
    """Build UNet++ EfficientNet-B5 with random init (no warm ckpt).

    encoder_weights=None ensures fully random initialization — no ImageNet
    pretrain either.  All parameters are trainable from epoch 1.

    Returns (model, meta_dict) where meta_dict is empty (no warm ckpt to report).
    LOCKED: never overwrites any source checkpoint.
    """
    import segmentation_models_pytorch as smp
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b5",
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
        decoder_attention_type="scse",
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built scratch model: {n_params/1e6:.1f}M params, random init", flush=True)
    return model, {}


def set_freeze_phase(model: nn.Module, epoch: int) -> None:
    """3-phase encoder freeze schedule for EfficientNet-B5.

    Phase 1 (ep 1-5):   freeze blocks 0-3 (early features)
    Phase 2 (ep 6-20):  freeze blocks 2-3 (only very early)
    Phase 3 (ep 21+):   full unfreeze

    NOTE: this function is skipped entirely when --no-warm is set
    (train-from-scratch always uses full unfreeze from epoch 1).
    """
    if epoch <= 5:
        freeze_blocks = FREEZE_PHASE1_BLOCKS
        label = "phase1 (freeze blocks 0-3)"
    elif epoch <= 20:
        freeze_blocks = FREEZE_PHASE2_BLOCKS
        label = "phase2 (freeze blocks 2-3)"
    else:
        freeze_blocks = []
        label = "phase3 (full unfreeze)"

    for name, param in model.encoder.named_parameters():
        is_frozen = any(f"_blocks.{i}." in name for i in freeze_blocks) or \
                    (epoch <= 5 and (name.startswith("_conv_stem") or
                                     name.startswith("_bn0")))
        param.requires_grad_(not is_frozen)

    trainable = sum(p.requires_grad for p in model.parameters())
    total = sum(1 for _ in model.parameters())
    print(f"  Freeze {label}: trainable {trainable}/{total} params", flush=True)


def make_optimizer(model: nn.Module, lr_backbone: float, lr_head: float,
                   weight_decay: float):
    """Discriminative LR: backbone 5e-5, decoder+head 2e-4.

    Used in fine-tune mode (warm-start) where encoder needs a lower LR
    to avoid catastrophic forgetting of LIDC-learned features.
    """
    enc_ids = {id(p) for p in model.encoder.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) in enc_ids and p.requires_grad]
    head_params = [p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad]
    return torch.optim.AdamW(
        [{"params": backbone_params, "lr": lr_backbone},
         {"params": head_params, "lr": lr_head}],
        weight_decay=weight_decay,
    )


def make_optimizer_uniform(model: nn.Module, lr: float, weight_decay: float):
    """Uniform LR across all parameters — used in train-from-scratch mode.

    No discriminative LR needed because there is no pre-trained encoder to protect.
    All trainable parameters use a single lr (--lr-backbone value).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_loss_fn():
    """Tversky(alpha=0.3, beta=0.7) + 0.5 × Focal-BCE.

    smooth_dr=1.0 prevents NaN on pure-negative slices under AMP fp16
    (same fix as dataset_v2.py make_loss).
    """
    from monai.losses import TverskyLoss
    tversky = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7,
                          smooth_nr=1.0, smooth_dr=1.0, batch=True)

    def combined(pred, target):
        t_loss = tversky(pred, target)
        focal_bce = F.binary_cross_entropy_with_logits(
            pred, target,
            reduction="mean",
        ) * 0.5
        return t_loss + focal_bce

    return combined


# ---------------------------------------------------------------------------
# Warmup scheduler helper
# ---------------------------------------------------------------------------

def make_warmup_coswr_scheduler(optimizer, warmup_epochs: int, T_0: int, T_mult: int):
    """Linear warmup for warmup_epochs, then CosineAnnealingWarmRestarts."""
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-7,
    )

    class WarmupCosWR:
        def __init__(self):
            self._epoch = 0

        def step(self):
            self._epoch += 1
            if self._epoch <= warmup_epochs:
                scale = self._epoch / max(warmup_epochs, 1)
                for pg in optimizer.param_groups:
                    pg["lr"] = pg.get("_base_lr", pg["lr"]) * scale
            else:
                cosine_sched.step(self._epoch - warmup_epochs)

        def state_dict(self):
            return {"_epoch": self._epoch,
                    "cosine": cosine_sched.state_dict()}

        def load_state_dict(self, sd):
            self._epoch = sd["_epoch"]
            cosine_sched.load_state_dict(sd["cosine"])

    # Store base LR for warmup scaling
    for pg in optimizer.param_groups:
        pg["_base_lr"] = pg["lr"]

    return WarmupCosWR()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn,
                    grad_accum: int = 2):
    """Forward + Tversky+Focal-BCE + grad accumulation + AMP.

    Returns: average loss over epoch.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (x, y, _lbl) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            pred = model(x)
            loss = loss_fn(pred, y) / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum
    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# FROC validation
# ---------------------------------------------------------------------------

def _extract_detections_from_batch(pred_logits: torch.Tensor,
                                   thresholds: list) -> dict:
    """Given sigmoid(pred_logits) per sample, return per-threshold binary preds.

    Returns {thr: list[bool]} — True if max prob in patch > thr.
    """
    probs = torch.sigmoid(pred_logits).cpu().numpy()   # [B, 1, H, W]
    max_probs = probs.max(axis=(1, 2, 3))              # [B]
    result = {}
    for thr in thresholds:
        result[thr] = (max_probs > thr).tolist()
    return result


def run_validation(model, loader, device, loss_fn,
                   thresholds: list = FROC_THRESHOLDS):
    """Compute val_loss, val_dice, FROC CPM + per-FP-rate sensitivities.

    FROC evaluated at patch level (positive patch = nodule present):
      TP = positive patch predicted positive
      FP = negative patch predicted positive
      FN = positive patch predicted negative
      FP/scan approximated per scan (one patch per GT nodule for positives;
      hard-neg patches for negatives).

    Returns dict with all metrics.
    """
    model.eval()
    total_loss = 0.0
    dice_sum = 0.0
    dice_n = 0

    # Per threshold: collect (label, confidence) pairs
    all_scores = []   # list of (label, max_prob)
    scan_counts = {"pos": 0, "neg": 0}

    with torch.no_grad():
        for x, y, lbl in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                pred_logits = model(x)
                loss = loss_fn(pred_logits, y)
            total_loss += loss.item()
            # Dice on positive samples
            pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
            has_mask = y.sum(dim=(1, 2, 3)) > 0
            if has_mask.any():
                inter = (pred_bin[has_mask] * y[has_mask]).sum(dim=(1, 2, 3))
                union = pred_bin[has_mask].sum(dim=(1, 2, 3)) + y[has_mask].sum(dim=(1, 2, 3))
                dice_batch = ((2 * inter + 1) / (union + 1)).mean().item()
                dice_sum += dice_batch
                dice_n += 1
            # Confidence for FROC
            probs = torch.sigmoid(pred_logits).cpu().numpy().max(axis=(1, 2, 3))
            for prob, l in zip(probs.tolist(), lbl.tolist()):
                all_scores.append((l, prob))
                scan_counts["pos" if l == 1 else "neg"] += 1

    val_loss = total_loss / max(len(loader), 1)
    val_dice = dice_sum / max(dice_n, 1)

    # FROC: sweep thresholds
    n_gt_total = scan_counts["pos"]
    n_scans = scan_counts["pos"] + scan_counts["neg"]
    froc_points = []
    for thr in sorted(thresholds):
        tp = sum(1 for (l, p) in all_scores if l == 1 and p > thr)
        fp = sum(1 for (l, p) in all_scores if l == 0 and p > thr)
        fn = n_gt_total - tp
        sens = tp / max(n_gt_total, 1)
        fp_per_scan = fp / max(n_scans, 1)
        froc_points.append({"threshold": thr, "sensitivity": sens,
                            "fp_per_scan": fp_per_scan,
                            "tp": tp, "fp": fp, "fn": fn})

    # Interpolate sensitivity at LUNA16 standard FP rates
    froc_sorted = sorted(froc_points, key=lambda p: p["fp_per_scan"])
    fp_arr = np.array([p["fp_per_scan"] for p in froc_sorted])
    sens_arr = np.array([p["sensitivity"] for p in froc_sorted])
    luna_sens = {}
    for fp_target in LUNA16_FP_RATES:
        if len(fp_arr) == 0 or fp_target <= fp_arr.min():
            s = float(sens_arr[0]) if len(sens_arr) else 0.0
        elif fp_target >= fp_arr.max():
            s = float(sens_arr[-1]) if len(sens_arr) else 0.0
        else:
            s = float(np.interp(fp_target, fp_arr, sens_arr))
        luna_sens[fp_target] = s
    cpm = float(np.mean(list(luna_sens.values())))

    # Best F1 operating point
    best_f1, best_thr = 0.0, 0.5
    for pt in froc_points:
        tp_, fp_, fn_ = pt["tp"], pt["fp"], pt["fn"]
        prec = tp_ / max(tp_ + fp_, 1)
        rec = tp_ / max(n_gt_total, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-7)
        if f1 > best_f1:
            best_f1, best_thr = f1, pt["threshold"]

    # Size-stratified sensitivity at thr=best_thr
    # (patch-level proxy: diameter_mm available if loading with diam info)
    # Full strat requires passing diameter per sample — deferred to full eval.

    metrics = {
        "val_loss": round(val_loss, 6),
        "val_dice": round(val_dice, 4),
        "val_froc_cpm": round(cpm, 4),
        "val_best_f1": round(best_f1, 4),
        "val_best_thr": round(best_thr, 4),
        "n_gt": n_gt_total,
        "n_scans": n_scans,
    }
    for fp_target, s in luna_sens.items():
        key = f"val_sens@{fp_target}FP"
        metrics[key] = round(s, 4)
    metrics["froc_points"] = froc_points
    return metrics


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def save_metrics_csv(csv_path: Path, row: dict, epoch: int) -> None:
    """Append one epoch row to CSV.  Creates file + header on first call."""
    fixed_cols = ["epoch", "train_loss", "val_loss", "val_dice", "val_froc_cpm",
                  "val_best_f1", "val_best_thr"]
    fp_cols = [f"val_sens@{fp}FP" for fp in LUNA16_FP_RATES]
    all_cols = fixed_cols + fp_cols

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({"epoch": epoch, **row})


def save_froc_plot(froc_points: list, out_path: Path,
                   epoch: int, cpm: float) -> None:
    """Save FROC curve PNG for this epoch."""
    pts = sorted(froc_points, key=lambda p: p["fp_per_scan"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([p["fp_per_scan"] for p in pts],
            [p["sensitivity"] for p in pts],
            "b-o", linewidth=2, markersize=4)
    luna_pts = [(fp, float(np.interp(
        fp,
        [p["fp_per_scan"] for p in pts],
        [p["sensitivity"] for p in pts],
    ))) for fp in LUNA16_FP_RATES]
    ax.scatter([lp[0] for lp in luna_pts], [lp[1] for lp in luna_pts],
               c="red", s=80, marker="*", zorder=5,
               label=f"LUNA16 FP rates (CPM={cpm:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("FP/scan (log)")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"FROC epoch {epoch}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)


def save_confusion_json(metrics: dict, out_path: Path, epoch: int) -> None:
    """Save per-epoch confusion summary as JSON."""
    record = {
        "epoch": epoch,
        "val_froc_cpm": metrics["val_froc_cpm"],
        "val_dice": metrics["val_dice"],
        "val_best_f1": metrics["val_best_f1"],
        "val_best_thr": metrics["val_best_thr"],
        "n_gt": metrics["n_gt"],
        "n_scans": metrics["n_scans"],
        "luna16_sens": {
            str(fp): metrics.get(f"val_sens@{fp}FP", 0.0)
            for fp in LUNA16_FP_RATES
        },
    }
    out_path.write_text(json.dumps(record, indent=2))


def try_log_wandb(run, metrics: dict, epoch: int) -> None:
    """Log to wandb if run is not None."""
    if run is None:
        return
    log_dict = {k: v for k, v in metrics.items() if k != "froc_points"}
    log_dict["epoch"] = epoch
    run.log(log_dict, step=epoch)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_ckpt(out_dir: Path, fname: str, model, epoch: int,
              metrics: dict, args_dict: dict, git_sha: str) -> None:
    """Save versioned checkpoint. Never overwrites locked dirs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "in_channels": PATCH_SLICES,
        "val_froc_cpm": metrics.get("val_froc_cpm"),
        "val_dice": metrics.get("val_dice"),
        "args": args_dict,
        "git_sha": git_sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(ckpt, str(out_dir / fname))


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Locked artifact protection
# ---------------------------------------------------------------------------

LOCKED_DIRS_REL = [
    "work/runs_exp/stage2_full",
    "work/runs",
    "bundles",
]


def check_not_locked(out_dir: Path, repo_root: Path) -> None:
    for rel in LOCKED_DIRS_REL:
        locked = (repo_root / rel).resolve()
        if out_dir.resolve() == locked:
            raise ValueError(
                f"LOCKED ARTIFACT PROTECTION: --output-dir resolves to {out_dir}, "
                f"which is a locked checkpoint directory ({rel}). "
                "Use a versioned path such as work/runs_luna16/run001."
            )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune LIDC winner ckpt on LUNA16")
    ap.add_argument("--luna16-dir", type=Path, required=True,
                    help="LUNA16 root dir (subsetN/ + annotations.csv)")
    ap.add_argument("--no-warm", action="store_true",
                    help=(
                        "Train from scratch (random init). "
                        "Skips loading --warm-ckpt entirely. "
                        "Enables uniform LR (--lr-backbone) across all params. "
                        "Disables 3-phase freeze schedule (all params trainable from ep 1). "
                        "Recommended with --epochs 80+ and --lr-backbone 1e-4."
                    ))
    ap.add_argument("--warm-ckpt", type=Path,
                    default=Path("/workspace/work/runs_luna16/init.pth"),
                    help="LIDC warm-start checkpoint (LOCKED — never overwrite source). "
                         "Ignored when --no-warm is set.")
    ap.add_argument("--exclude-json", type=Path,
                    default=Path("work/academic/LUNA16_EXCLUDE_FROM_TRAIN.json"),
                    help="JSON list of seriesuids to exclude (LIDC test_panel overlap)")
    ap.add_argument("--train-subsets", type=str, default="0,1,2,3,4,5,6",
                    help="Comma-separated subset indices for training")
    ap.add_argument("--val-subsets", type=str, default="7,8",
                    help="Comma-separated subset indices for validation")
    ap.add_argument("--test-subsets", type=str, default="9",
                    help="Held-out test subsets (not used for selection)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="Gradient accumulation steps (effective batch = batch*accum)")
    ap.add_argument("--lr-backbone", type=float, default=5e-5,
                    help="LR for frozen-then-unfrozen encoder blocks")
    ap.add_argument("--lr-head", type=float, default=2e-4,
                    help="LR for decoder + segmentation head")
    ap.add_argument("--weight-decay", type=float, default=5e-5)
    ap.add_argument("--warmup-epochs", type=int, default=3,
                    help="Linear LR warmup (epochs)")
    ap.add_argument("--coswr-t0", type=int, default=10,
                    help="CosineWarmRestarts T_0")
    ap.add_argument("--coswr-tmult", type=int, default=2,
                    help="CosineWarmRestarts T_mult")
    ap.add_argument("--swa-start-ep", type=int, default=36,
                    help="Epoch to begin SWA model accumulation (0 = disable)")
    ap.add_argument("--swa-lr", type=float, default=1e-6,
                    help="Constant SWA LR during SWA phase")
    ap.add_argument("--patience", type=int, default=PATIENCE_DEFAULT,
                    help="Early stop patience on val_froc_cpm (0 = disable)")
    ap.add_argument("--patience-metric", type=str, default="dice",
                    choices=["cpm", "dice", "f1"],
                    help="Metric for early stop (default: dice — val_cpm saturates too fast)")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("work/runs_luna16/run001"),
                    help="Versioned output dir (must not be a locked dir)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patch-hw", type=int, default=PATCH_HW,
                    help="Patch spatial size in pixels")
    ap.add_argument("--n-slices", type=int, default=PATCH_SLICES,
                    help="Number of 2.5D input slices (must be odd)")
    ap.add_argument("--wandb-project", type=str, default="lidc-luna16-finetune",
                    help="W&B project name (set WANDB_API_KEY or pass offline)")
    ap.add_argument("--no-wandb", action="store_true",
                    help="Disable wandb logging entirely")
    ap.add_argument("--smoke", type=int, default=0,
                    help="Smoke test: run N epochs on first 5 train + 2 val scans")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    git_sha = get_git_sha()
    print(f"Device: {device}  seed: {args.seed}  git_sha: {git_sha}")

    # -- Locked artifact protection --
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = args.output_dir.resolve()
    check_not_locked(out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- LUNA16 directory checks --
    luna16_dir = args.luna16_dir
    annotations_csv = luna16_dir / "annotations.csv"
    if not annotations_csv.exists():
        raise FileNotFoundError(
            f"annotations.csv not found at {annotations_csv}. "
            "Verify --luna16-dir points to the LUNA16 root."
        )

    # -- Parse annotations + exclude list --
    annotations = parse_luna16_annotations(annotations_csv)
    total_nods = sum(len(v) for v in annotations.values())
    print(f"Annotations: {len(annotations)} series  {total_nods} nodule instances")

    exclude_set: set = set()
    if args.exclude_json.exists():
        exclude_set = set(json.loads(args.exclude_json.read_text()))
        print(f"Exclude list: {len(exclude_set)} series (LIDC test_panel overlap)")
    else:
        print(f"WARN: exclude list not found at {args.exclude_json} — proceeding without it")

    def _collect_scans(subset_str: str) -> list:
        paths = []
        for idx in subset_str.split(","):
            sd = luna16_dir / f"subset{idx.strip()}"
            if sd.exists():
                for mhd in sorted(sd.glob("*.mhd")):
                    if mhd.stem not in exclude_set:
                        paths.append(mhd)
        return paths

    train_scans = _collect_scans(args.train_subsets)
    val_scans = _collect_scans(args.val_subsets)
    test_scans = _collect_scans(args.test_subsets)
    print(f"Train scans: {len(train_scans)}  Val: {len(val_scans)}  Test: {len(test_scans)}")

    # -- Smoke mode override --
    smoke_n = 0
    if args.smoke > 0:
        args.epochs = args.smoke
        smoke_n = 5
        val_scans = val_scans[:2]
        print(f"SMOKE MODE: {args.smoke} epochs, {smoke_n} train scans, {len(val_scans)} val scans")

    # -- Build datasets --
    print("Building training index ...", flush=True)
    t_idx0 = time.time()
    train_ds = LunaDataset(
        train_scans, annotations, luna16_dir=luna16_dir,
        patch_hw=args.patch_hw, n_slices=args.n_slices,
        neg_sampler_ratio=NEG_SAMPLER_RATIO,
        augment=True, seed=args.seed,
        smoke_n_scans=smoke_n,
    )
    print(f"Building val index ...", flush=True)
    val_ds = LunaDataset(
        val_scans, annotations, luna16_dir=luna16_dir,
        patch_hw=args.patch_hw, n_slices=args.n_slices,
        augment=False, seed=args.seed,
        smoke_n_scans=smoke_n,
    )
    print(f"Index built in {time.time()-t_idx0:.1f}s  "
          f"train={len(train_ds)}  val={len(val_ds)}", flush=True)

    train_sampler = train_ds.make_sampler()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # -- Model --
    if args.no_warm:
        print("\nTrain-from-scratch mode (--no-warm): random init, no warm ckpt loaded.")
        model, _ = build_scratch_model(device, in_channels=args.n_slices)
    else:
        print(f"\nFine-tune mode: loading warm ckpt: {args.warm_ckpt}")
        model, _ = load_warm_ckpt(args.warm_ckpt.resolve(), device,
                                   in_channels=args.n_slices)

    # -- Loss / Optimizer / Scaler --
    loss_fn = make_loss_fn()
    if args.no_warm:
        # Uniform LR — no encoder to protect, no discriminative LR needed
        optimizer = make_optimizer_uniform(model, args.lr_backbone, args.weight_decay)
    else:
        optimizer = make_optimizer(model, args.lr_backbone, args.lr_head, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # -- LR scheduler --
    scheduler = make_warmup_coswr_scheduler(
        optimizer, args.warmup_epochs, args.coswr_t0, args.coswr_tmult
    )

    # -- SWA --
    swa_model = AveragedModel(model) if args.swa_start_ep > 0 else None
    swa_sched = SWALR(optimizer, swa_lr=args.swa_lr,
                      anneal_epochs=max(1, args.epochs - args.swa_start_ep)) \
        if args.swa_start_ep > 0 else None

    # -- Wandb --
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config={k: str(v) for k, v in vars(args).items()},
                name=out_dir.name,
                dir=str(out_dir),
                mode=os.environ.get("WANDB_MODE", "online"),
            )
            print(f"W&B run: {wandb_run.name}  id: {wandb_run.id}")
        except Exception as e:
            print(f"W&B init failed ({e}) — continuing offline")

    # -- Save config --
    args_dict = {k: str(v) for k, v in vars(args).items()}
    args_dict["git_sha"] = git_sha
    (out_dir / "config.json").write_text(json.dumps(args_dict, indent=2))
    csv_path = out_dir / "metrics.csv"

    # -- Training loop --
    best_cpm = -1.0
    best_metric = -1.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.no_warm:
            # Scratch mode: all params trainable from ep 1, no freeze schedule.
            # Optimizer already covers all params — no rebuild needed.
            pass
        else:
            # Fine-tune mode: 3-phase freeze + optimizer rebuild at transitions.
            set_freeze_phase(model, epoch)
            if epoch in (1, 6, 21):
                optimizer = make_optimizer(model, args.lr_backbone, args.lr_head,
                                           args.weight_decay)
                scheduler = make_warmup_coswr_scheduler(
                    optimizer, args.warmup_epochs, args.coswr_t0, args.coswr_tmult
                )
                if swa_sched is not None:
                    swa_sched = SWALR(optimizer, swa_lr=args.swa_lr,
                                      anneal_epochs=max(1, args.epochs - args.swa_start_ep))
                scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, loss_fn, args.grad_accum
        )

        val_metrics = run_validation(model, val_loader, device, loss_fn)
        val_cpm = val_metrics["val_froc_cpm"]
        val_dice = val_metrics["val_dice"]

        # LR schedule step
        in_swa_phase = (args.swa_start_ep > 0 and epoch >= args.swa_start_ep)
        if in_swa_phase:
            if swa_model is not None:
                swa_model.update_parameters(model)
            if swa_sched is not None:
                swa_sched.step()
        else:
            scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[-1]["lr"]
        print(
            f"ep {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['val_loss']:.4f}  "
            f"val_dice={val_dice:.4f}  cpm={val_cpm:.4f}  "
            f"lr={current_lr:.2e}  {elapsed:.0f}s",
            flush=True,
        )

        # -- Logging --
        row = {
            "train_loss": round(train_loss, 6),
            **{k: v for k, v in val_metrics.items() if k != "froc_points"},
        }
        save_metrics_csv(csv_path, row, epoch)
        try_log_wandb(wandb_run, row, epoch)

        # Confusion matrix + FROC curve per epoch
        save_confusion_json(
            val_metrics, out_dir / f"confusion_matrix_ep{epoch:03d}.json", epoch
        )
        save_froc_plot(
            val_metrics["froc_points"],
            out_dir / f"froc_curve_ep{epoch:03d}.png",
            epoch, val_cpm,
        )

        # -- Checkpoint --
        # Track best_cpm independently (always, for final report)
        if val_cpm > best_cpm:
            best_cpm = val_cpm
        # Early stop metric is configurable — val_cpm saturates at ~1.0 too fast
        metric_map = {"cpm": val_cpm, "dice": val_dice, "f1": val_metrics.get("val_best_f1", 0.0)}
        current_metric = metric_map[args.patience_metric]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            no_improve = 0
            save_ckpt(out_dir, "best.pt", model, epoch, val_metrics, args_dict, git_sha)
            print(f"  ** New best {args.patience_metric}={best_metric:.4f} — saved best.pt", flush=True)
        else:
            no_improve += 1

        # Always save last.pt
        save_ckpt(out_dir, "last.pt", model, epoch, val_metrics, args_dict, git_sha)

        # -- Early stopping --
        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stop at epoch {epoch}: no improvement for {args.patience} epochs")
            break

    # -- SWA final BN update + save --
    if swa_model is not None and args.swa_start_ep > 0:
        print("Running SWA BN update ...", flush=True)
        update_bn(train_loader, swa_model, device=device)
        swa_val_metrics = run_validation(swa_model.module, val_loader, device, loss_fn)
        print(f"SWA val cpm={swa_val_metrics['val_froc_cpm']:.4f}  "
              f"dice={swa_val_metrics['val_dice']:.4f}")
        save_ckpt(out_dir, "swa.pt", swa_model.module, args.epochs,
                  swa_val_metrics, args_dict, git_sha)
        print(f"Saved swa.pt", flush=True)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"\nTraining done. Best {args.patience_metric}={best_metric:.4f} at epoch {best_epoch}. Best CPM={best_cpm:.4f}.")
    print(f"Checkpoints saved to: {out_dir}")
    print(f"Metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
