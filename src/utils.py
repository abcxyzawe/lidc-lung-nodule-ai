"""Shared utility functions used across multiple scripts."""
import time
import functools
from pathlib import Path
from contextlib import contextmanager

import numpy as np

from configs import HU_LO, HU_HI


def normalize_hu(img_int16: np.ndarray) -> np.ndarray:
    """Apply lung window to int16 HU array → float32 in [0, 1]."""
    x = np.clip(img_int16.astype(np.float32), HU_LO, HU_HI)
    return (x - HU_LO) / (HU_HI - HU_LO)


def build_mask_from_per_rad(mask_per_rad: np.ndarray, mode: str = "union") -> np.ndarray:
    """Combine 4-radiologist masks into a single binary mask.

    mask_per_rad: [N, 4, H, W] uint8
    returns:       [N, H, W]    uint8
    """
    if mode == "union":
        return (mask_per_rad.sum(axis=1) >= 1).astype(np.uint8)
    if mode == "consensus2":
        return (mask_per_rad.sum(axis=1) >= 2).astype(np.uint8)
    if mode == "consensus3":
        return (mask_per_rad.sum(axis=1) >= 3).astype(np.uint8)
    if mode == "all4":
        return (mask_per_rad.sum(axis=1) >= 4).astype(np.uint8)
    raise ValueError(f"Unknown mask mode: {mode}")


@contextmanager
def timer(label: str):
    """Context manager printing elapsed time on exit."""
    t0 = time.time()
    print(f"[{label}] start", flush=True)
    yield
    print(f"[{label}] done in {time.time() - t0:.1f}s", flush=True)


def patient_id_from_path(p: Path) -> str:
    """Extract LIDC-IDRI-XXXX from any path inside a patient folder."""
    for part in p.parts:
        if part.startswith("LIDC-IDRI-"):
            return part
    return "UNKNOWN"
