"""Extended Dataset variants for retraining experiments.

v1 (LIDCSeg25DMulti): reads work/preprocessed/  (current default — nodule + buffer only)
v2 (LIDCSeg25DV2):    reads work/preprocessed_v2/ (nodule + buffer + negative slices,
                       per-slice class label in `slice_classes` dataset)

Plus:
  - N-slice 2.5D input (3, 5, 7 ...)
  - Class-balanced sampler (40/30/30 nodule/buffer/negative)
  - Oversample 10-20mm nodule slices
  - 3→5 channel weight init
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from configs import HU_LO, HU_HI, MIN_MASK_PIXELS, WORK
from dataset import _normalize, _build_mask, get_train_aug


PRE_DIR_V2 = WORK / "preprocessed_v2"

SLICE_NEGATIVE = 0
SLICE_BUFFER = 1
SLICE_NODULE = 2
SLICE_CLASS_NAMES = {0: "negative", 1: "buffer", 2: "nodule"}


def _stack_slices(center_idx, n_in_series, n_channels):
    """Stack n_channels slices around center, padding with edge slices when out of range."""
    half = n_channels // 2
    return [max(0, min(n_in_series - 1, center_idx + off))
            for off in range(-half, half + 1)]


# ---------------------------------------------------------------------------
# v1 — same as original but with N-slice support (reads work/preprocessed/)
# ---------------------------------------------------------------------------
class LIDCSeg25DMulti(Dataset):
    """Multi-slice 2.5D dataset reading the ORIGINAL preprocessed h5 (nodule + buffer)."""
    def __init__(self, root, patient_ids, mask_mode="union",
                 n_channels=3, min_mask_pixels=MIN_MASK_PIXELS, augment=None,
                 only_with_mask=True):
        assert n_channels % 2 == 1
        self.root = Path(root)
        self.mask_mode = mask_mode
        self.n_channels = n_channels
        self.augment = augment
        self.entries = []   # (h5_path, key, center_idx, n_in_series, gt_diam_mm)
        for pid in patient_ids:
            p = self.root / f"{pid}.h5"
            if not p.exists(): continue
            with h5py.File(p, "r") as f:
                for k in f.keys():
                    n = f[k]["images"].shape[0]
                    pix_sp = json.loads(f[k].attrs["pixel_spacing"])
                    if only_with_mask:
                        m_rad = f[k]["mask_per_rad"][:]
                        m = _build_mask(m_rad, mask_mode)
                        for i in range(n):
                            n_pix = int(m[i].sum())
                            if n_pix < min_mask_pixels: continue
                            area_mm2 = n_pix * pix_sp[0] * pix_sp[1]
                            diam = 2.0 * np.sqrt(area_mm2 / np.pi)
                            self.entries.append((str(p), k, i, n, float(diam)))
                    else:
                        for i in range(n):
                            self.entries.append((str(p), k, i, n, 0.0))
        print(f"LIDCSeg25DMulti(v1, {mask_mode}, ch={n_channels}): "
              f"{len(self.entries)} slices from {len(patient_ids)} patients")

    def __len__(self): return len(self.entries)

    @property
    def slice_diameters(self):
        return np.array([e[4] for e in self.entries], dtype=np.float32)

    def __getitem__(self, idx):
        path, key, i, n, _ = self.entries[idx]
        idxs = _stack_slices(i, n, self.n_channels)
        with h5py.File(path, "r") as f:
            # Read one-at-a-time to avoid h5py fancy-indexing constraint on edges
            unique_sorted = sorted(set(idxs))
            data = f[key]["images"][unique_sorted]
            map_ = {u: i for i, u in enumerate(unique_sorted)}
            imgs = np.stack([data[map_[j]] for j in idxs])
            m_rad = f[key]["mask_per_rad"][i:i + 1]
        mask = _build_mask(m_rad, self.mask_mode)[0]
        imgs = np.stack([_normalize(imgs[c]) for c in range(self.n_channels)],
                         axis=0).astype(np.float32)
        mask = mask.astype(np.float32)[None]
        if self.augment is not None:
            imgs, mask = self.augment(imgs, mask)
        return torch.from_numpy(imgs), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# v2 — reads work/preprocessed_v2/ with slice_classes (nodule/buffer/negative)
# ---------------------------------------------------------------------------
class LIDCSeg25DV2(Dataset):
    """v2: includes negative background slices.

    Each entry: (h5_path, key, center_idx, n_in_series, slice_class, gt_diam_mm)
    Negative slices have empty mask (zeros).
    """
    def __init__(self, root, patient_ids, mask_mode="union",
                 n_channels=3, augment=None,
                 include_classes=(SLICE_NODULE, SLICE_BUFFER, SLICE_NEGATIVE)):
        assert n_channels % 2 == 1
        self.root = Path(root)
        self.mask_mode = mask_mode
        self.n_channels = n_channels
        self.augment = augment
        self.entries = []
        n_pat = 0
        for pid in patient_ids:
            p = self.root / f"{pid}.h5"
            if not p.exists(): continue
            with h5py.File(p, "r") as f:
                for k in f.keys():
                    n = f[k]["images"].shape[0]
                    if "slice_classes" not in f[k]:
                        # fallback: treat all as nodule
                        slice_classes = np.full(n, SLICE_NODULE, dtype=np.uint8)
                    else:
                        slice_classes = f[k]["slice_classes"][:]
                    pix_sp = json.loads(f[k].attrs["pixel_spacing"])
                    m_rad = f[k]["mask_per_rad"][:]
                    m = _build_mask(m_rad, mask_mode)
                    for i in range(n):
                        sc = int(slice_classes[i])
                        if sc not in include_classes: continue
                        # gt diameter on this slice (0 if no mask)
                        n_pix = int(m[i].sum())
                        if n_pix > 0:
                            area_mm2 = n_pix * pix_sp[0] * pix_sp[1]
                            diam = 2.0 * np.sqrt(area_mm2 / np.pi)
                        else:
                            diam = 0.0
                        # For consensus modes, a slice that was originally a
                        # nodule slice may have empty mask under consensus2/3;
                        # still keep it but mark as negative-equivalent for sampling.
                        effective_class = sc
                        if sc == SLICE_NODULE and n_pix == 0:
                            effective_class = SLICE_BUFFER
                        self.entries.append((str(p), k, i, n, effective_class, float(diam)))
            n_pat += 1
        cls_counts = {c: sum(1 for e in self.entries if e[4] == c) for c in include_classes}
        names = ", ".join(f"{SLICE_CLASS_NAMES[c]}={cls_counts[c]}" for c in include_classes)
        print(f"LIDCSeg25DV2({mask_mode}, ch={n_channels}, classes={include_classes}): "
              f"{len(self.entries)} slices from {n_pat} patients ({names})")

    def __len__(self): return len(self.entries)

    @property
    def slice_classes_arr(self):
        return np.array([e[4] for e in self.entries], dtype=np.int8)

    @property
    def slice_diameters(self):
        return np.array([e[5] for e in self.entries], dtype=np.float32)

    def __getitem__(self, idx):
        path, key, i, n, sc, _ = self.entries[idx]
        idxs = _stack_slices(i, n, self.n_channels)
        with h5py.File(path, "r") as f:
            unique_sorted = sorted(set(idxs))
            data = f[key]["images"][unique_sorted]
            map_ = {u: i for i, u in enumerate(unique_sorted)}
            imgs = np.stack([data[map_[j]] for j in idxs])
            m_rad = f[key]["mask_per_rad"][i:i + 1]
        mask = _build_mask(m_rad, self.mask_mode)[0]
        imgs = np.stack([_normalize(imgs[c]) for c in range(self.n_channels)],
                         axis=0).astype(np.float32)
        mask = mask.astype(np.float32)[None]
        if self.augment is not None:
            imgs, mask = self.augment(imgs, mask)
        return torch.from_numpy(imgs), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
def make_class_balanced_sampler(dataset: LIDCSeg25DV2,
                                  class_weights=(1.0, 0.75, 0.75),
                                  oversample_10_20mm: bool = False,
                                  oversample_ratio: float = 3.0,
                                  bucket_lo_mm: float = 10.0,
                                  bucket_hi_mm: float = 20.0,
                                  num_samples: int = None) -> WeightedRandomSampler:
    """Pick slices with weights:
      class_weights = (nodule_w, buffer_w, negative_w)
    Then if oversample_10_20mm, multiply nodule slices in [bucket_lo, bucket_hi] by oversample_ratio.

    With class_weights=(1.0, 0.75, 0.75) and typical counts (e.g. 18K nodule + 26K buffer +
    50K negative), expected sampling proportion roughly:
      nodule ≈ 1·18K / (1·18K + 0.75·26K + 0.75·50K) ≈ 23%

    For exactly 40/30/30, set class_weights such that
      nodule_w / nodule_count : buffer_w / buffer_count : negative_w / negative_count
    matches 40:30:30. The runner computes this from the dataset stats automatically.
    """
    classes = dataset.slice_classes_arr
    weights = np.zeros(len(dataset), dtype=np.float64)
    for c, w in enumerate(class_weights):
        mask = classes == c
        weights[mask] = w
    if oversample_10_20mm:
        diams = dataset.slice_diameters
        in_bucket = (diams >= bucket_lo_mm) & (diams < bucket_hi_mm)
        weights[in_bucket] *= oversample_ratio
        n_b = int(in_bucket.sum())
        print(f"  oversample_10_20mm: {n_b} slices boosted ×{oversample_ratio}")
    weights = weights / weights.sum() * len(weights)
    n = num_samples if num_samples is not None else len(weights)
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)


def make_target_proportion_sampler(dataset: LIDCSeg25DV2,
                                     target_proportions=(0.4, 0.3, 0.3),
                                     oversample_10_20mm: bool = False,
                                     oversample_ratio: float = 3.0):
    """Convenience: compute per-class weight to hit exact target proportion in expectation.

    target_proportions = (nodule_pct, buffer_pct, negative_pct)
    """
    classes = dataset.slice_classes_arr
    counts = np.array([(classes == c).sum() for c in range(3)], dtype=np.float64)
    counts[counts == 0] = 1.0
    target = np.array(target_proportions, dtype=np.float64)
    # weight per class so that sum(weight × count) gives target proportion
    w_per_class = target / counts
    w_per_class = w_per_class / w_per_class.max()  # normalise so largest class weight = 1
    print(f"  target_proportions sampler: nodule={target_proportions[0]:.0%} "
          f"buffer={target_proportions[1]:.0%} negative={target_proportions[2]:.0%}")
    print(f"    per-slice weights: nodule={w_per_class[2]:.3f} buffer={w_per_class[1]:.3f} "
          f"negative={w_per_class[0]:.3f}")
    return make_class_balanced_sampler(
        dataset, class_weights=(w_per_class[0], w_per_class[1], w_per_class[2]),
        oversample_10_20mm=oversample_10_20mm, oversample_ratio=oversample_ratio,
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def make_loss(loss_name: str, alpha: float = 0.5, beta: float = 0.5,
              gamma: float = 1.0):
    from monai.losses import DiceFocalLoss, TverskyLoss
    if loss_name == "dice_focal":
        return DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=0.5)
    if loss_name == "tversky":
        return TverskyLoss(sigmoid=True, alpha=alpha, beta=beta, include_background=False)
    if loss_name == "focal_tversky":
        base = TverskyLoss(sigmoid=True, alpha=alpha, beta=beta,
                           include_background=False, reduction="none")
        gamma_const = float(gamma)
        class _FocalTversky:
            def __call__(self, y_pred, y_true):
                t = base(y_pred, y_true)
                return (t.clamp(min=1e-7) ** gamma_const).mean()
        return _FocalTversky()
    raise ValueError(f"Unknown loss: {loss_name}")


def init_5ch_encoder_from_3ch(model_5ch, ckpt_path):
    """Load a 3-channel checkpoint into a 5-channel model by averaging input conv weights."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_3ch = ck["model"]
    in_conv_keys = [k for k in sd_3ch.keys()
                    if (k.endswith("encoder._conv_stem.weight")
                        or (k.endswith("conv_stem.weight") and sd_3ch[k].shape[1] == 3))]
    if not in_conv_keys:
        print("  WARN: could not find 3-ch input conv; loading non-strict")
        model_5ch.load_state_dict(sd_3ch, strict=False)
        return model_5ch
    in_conv_key = in_conv_keys[0]
    w_3ch = sd_3ch[in_conv_key]
    out_ch, _, kh, kw = w_3ch.shape
    w_5ch = w_3ch.mean(dim=1, keepdim=True).repeat(1, 5, 1, 1) * (3.0 / 5.0)
    sd_3ch[in_conv_key] = w_5ch
    model_5ch.load_state_dict(sd_3ch, strict=False)
    print(f"  Initialised 5-ch from 3-ch checkpoint {ckpt_path}")
    return model_5ch
