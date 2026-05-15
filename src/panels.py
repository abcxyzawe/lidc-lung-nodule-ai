"""Panel definitions for academic evaluation.

Three panels:
  - debug_panel  (12 patients) — small, for fast debugging only
  - val_panel    (99 patients) — for hyperparameter tuning
  - test_panel   (99 patients) — LOCKED — only for final reporting

val_panel and test_panel are the canonical val/test splits from splits.json.
NEVER tune on test_panel.
"""
import json
from pathlib import Path

from configs import SPLITS_JSON


# Hand-picked diverse cases for fast debug runs (~3 min total)
DEBUG_PANEL = [
    "LIDC-IDRI-0001",  # easy 1-nodule
    "LIDC-IDRI-0002",  # 17.5mm + 5mm
    "LIDC-IDRI-0003",  # multi-nodule
    "LIDC-IDRI-0094",  # easy
    "LIDC-IDRI-0220",  # multi
    "LIDC-IDRI-0303",  # 28mm large
    "LIDC-IDRI-0447",  # multi
    "LIDC-IDRI-0595",  # easy
    "LIDC-IDRI-0651",  # multi
    "LIDC-IDRI-0751",  # 9 weak-consensus + thick slice
    "LIDC-IDRI-0936",  # 17mm
    "LIDC-IDRI-0940",  # multi
]


def load_panel(name: str) -> list[str]:
    """Return list of patient IDs for the named panel.

    name: 'debug' | 'train' | 'val' | 'test'
    """
    if name == "debug":
        return DEBUG_PANEL
    splits = json.loads(Path(SPLITS_JSON).read_text())
    if name == "train":
        return splits["train"]
    if name == "val":
        return splits["val"]
    if name == "test":
        return splits["test"]
    raise ValueError(f"Unknown panel: {name}. Use debug/train/val/test.")
