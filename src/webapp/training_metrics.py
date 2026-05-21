"""Parse training logs and checkpoint metadata for the /api/training-metrics endpoint.

All parsing is done once and cached in memory (logs are static artifacts).
"""
import json
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path resolution — work/ lives two levels above this file (src/webapp/)
# ---------------------------------------------------------------------------
_WEBAPP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _WEBAPP_DIR.parent.parent  # E:/Phan Tich Ung Thu

_STAGE2_LOG = _PROJECT_ROOT / "work" / "runs_exp" / "stage2_full" / "log.txt"
_FPR_LOG = _PROJECT_ROOT / "work" / "runs" / "mal_train.log"
_FPR_CKPT = _PROJECT_ROOT / "work" / "runs" / "fpr.pt"
_MAL_METRICS = _PROJECT_ROOT / "work" / "runs" / "malignancy_metrics.json"

# Regex patterns
# loss field may be "0.????" (sampled milestone gap), so match any non-space token
_STAGE2_RE = re.compile(
    r"ep\s+(\d+)/(\d+)\s+loss=(\S+)\s+val_dice=([\d.]+)"
)
# F-35: use \S+ for all metric fields to tolerate nan/inf values in log
_FPR_RE = re.compile(
    r"epoch\s+(\d+)/(\d+)"
    r"\s+loss=(\S+)"
    r"\s+val_acc=(\S+)"
    r"\s+val_bal_acc=(\S+)"
    r"\s+val_susp_f1=(\S+)"
)

# Known constant — test_panel F1 from winner ckpt commit cd48359
_MINE_F1_TEST_PANEL: float = 0.618


@lru_cache(maxsize=1)
def parse_stage2_log() -> list[dict[str, Any]]:
    """Parse work/runs_exp/stage2_full/log.txt.

    Returns list of dicts with keys: epoch, loss, val_dice, is_best.
    Loss is None when the log records '0.????' (sampled milestone gap).
    """
    text = _STAGE2_LOG.read_text(encoding="utf-8")
    best_so_far: float = -1.0
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _STAGE2_RE.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        try:
            loss: float | None = float(m.group(3))
        except ValueError:
            loss = None  # "0.????" placeholder in sampled log
        val_dice = float(m.group(4))
        is_best = val_dice > best_so_far
        if is_best:
            best_so_far = val_dice
        records.append(
            {
                "epoch": epoch,
                "loss": loss,
                "val_dice": val_dice,
                "is_best": is_best,
            }
        )
    return records


@lru_cache(maxsize=1)
def parse_fpr_log() -> list[dict[str, Any]]:
    """Parse work/runs/mal_train.log.

    Returns list of dicts with keys:
      epoch, loss, val_acc, val_bal_acc, val_susp_f1, is_best.
    Any metric that logged as nan/inf is returned as None.
    """
    text = _FPR_LOG.read_text(encoding="utf-8")
    best_so_far: float = -1.0
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _FPR_RE.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        try:
            loss: float | None = float(m.group(3))
        except ValueError:
            loss = None
        try:
            val_acc: float | None = float(m.group(4))
        except ValueError:
            val_acc = None
        try:
            val_bal_acc: float | None = float(m.group(5))
        except ValueError:
            val_bal_acc = None
        try:
            val_susp_f1: float | None = float(m.group(6))
        except ValueError:
            val_susp_f1 = None
        # F-35: guard against None val_bal_acc before comparison
        is_best = val_bal_acc is not None and val_bal_acc > best_so_far
        if is_best:
            best_so_far = val_bal_acc
        records.append(
            {
                "epoch": epoch,
                "loss": loss,
                "val_acc": val_acc,
                "val_bal_acc": val_bal_acc,
                "val_susp_f1": val_susp_f1,
                "is_best": is_best,
            }
        )
    return records


@lru_cache(maxsize=1)
def get_fpr_ckpt_summary() -> dict[str, float]:
    """Load fpr.pt and extract best_thr + auc scalar fields.

    F-33: attempts weights_only=True first to avoid arbitrary code execution
    via pickle. Falls back to weights_only=False only for legacy checkpoints
    that embed non-tensor objects, with an explicit warning logged.
    Only scalar values are extracted from the checkpoint — no model weights
    are returned or executed.
    """
    import torch  # deferred — not needed until endpoint is hit

    try:
        ckpt = torch.load(str(_FPR_CKPT), map_location="cpu", weights_only=True)
    except Exception:
        warnings.warn(
            "fpr.pt loaded with weights_only=False (legacy checkpoint with "
            "non-tensor objects). Recommend re-saving scalars only to "
            "eliminate pickle deserialization risk.",
            stacklevel=2,
        )
        ckpt = torch.load(str(_FPR_CKPT), map_location="cpu", weights_only=False)
    # Extract only scalar values — no model weights or arbitrary objects returned
    return {
        "best_thr": float(ckpt.get("best_thr", 0.5)),
        "auc": float(ckpt.get("auc", 0.0)),
    }


@lru_cache(maxsize=1)
def get_summary() -> dict[str, Any]:
    """Aggregate summary across stage2 + FPR + test metrics."""
    stage2 = parse_stage2_log()
    fpr_records = parse_fpr_log()

    # Stage2 best
    best_stage2 = max(stage2, key=lambda r: r["val_dice"])
    stage2_best_epoch: int = best_stage2["epoch"]
    stage2_best_val_dice: float = best_stage2["val_dice"]
    stage2_total_epochs: int = max(r["epoch"] for r in stage2)

    # FPR ckpt scalars
    fpr_ckpt = get_fpr_ckpt_summary()
    fpr_total_epochs: int = max(r["epoch"] for r in fpr_records) if fpr_records else 0

    # F-31a: fpr_best_epoch — epoch of last is_best record in FPR log
    best_fpr_records = [r for r in fpr_records if r["is_best"]]
    fpr_best_epoch: int = (
        max(best_fpr_records, key=lambda r: r["epoch"])["epoch"]
        if best_fpr_records
        else 0
    )

    # Test metrics from JSON
    test_metrics: dict[str, Any] = {}
    if _MAL_METRICS.exists():
        test_metrics = json.loads(_MAL_METRICS.read_text(encoding="utf-8"))

    return {
        "stage2_best_epoch": stage2_best_epoch,
        "stage2_best_val_dice": stage2_best_val_dice,
        "stage2_total_epochs": stage2_total_epochs,
        "fpr_auc": fpr_ckpt["auc"],
        "fpr_best_thr": fpr_ckpt["best_thr"],
        "fpr_best_epoch": fpr_best_epoch,
        "fpr_total_epochs": fpr_total_epochs,
        "test_acc": test_metrics.get("test_acc"),
        "test_bal_acc": test_metrics.get("test_bal_acc"),
        "test_susp_f1": test_metrics.get("test_susp_f1"),
        "mine_f1_test_panel": _MINE_F1_TEST_PANEL,
    }
