"""Project-wide configuration: paths, data constants, model defaults.

Single source of truth — every other script imports from here.
"""
from pathlib import Path

# ============ Paths ============
# ROOT can be overridden by env var LIDC_ROOT (e.g. /workspace on remote).
# Default: Windows local path.
import os as _os
ROOT       = Path(_os.environ.get("LIDC_ROOT", r"E:/Phan Tich Ung Thu"))
DICOM_ROOT = ROOT / "manifest-1600709154662" / "LIDC-IDRI"
XML_ROOT   = ROOT / "tcia-lidc-xml"
WORK       = ROOT / "work"
OUTPUTS    = ROOT / "outputs"
# Don't mkdir on import — let scripts that need them create on demand.

# Generated artifacts
SOP_INDEX_JSON = WORK / "sop_index.json"
CLEAN_CSV      = WORK / "clean_dataset.csv"
SUMMARY_JSON   = WORK / "summary.json"
PRE_DIR        = WORK / "preprocessed"
SPLITS_JSON    = WORK / "splits.json"
RUNS_DIR       = WORK / "runs"

# Remote container (override via env)
REMOTE_HOST = _os.environ.get("LIDC_VPS_HOST", "")
REMOTE_PORT = int(_os.environ.get("LIDC_VPS_PORT", "22"))
REMOTE_USER = _os.environ.get("LIDC_VPS_USER", "root")
REMOTE_KEY  = Path.home() / ".ssh" / "lidc_remote"
REMOTE_WORK = "/workspace"

# ============ Data preprocessing ============
BUFFER_SLICES   = 3        # keep slice ±N around each nodule slice
TARGET_HW       = 512      # all images resized/padded to (H, W)

# CT lung window (Hounsfield Units)
HU_LO = -1350.0
HU_HI = 150.0

# Mask building from 4 radiologist contours
MASK_MODE_DEFAULT = "union"   # union | consensus2 | consensus3 | all4
MIN_MASK_PIXELS   = 10        # drop sub-10px point markers (nonNodule)

# ============ Train / model defaults (TOP-1 settings) ============
ENCODER     = "efficientnet-b5"        # bigger backbone (28M -> 30M params)
ENCODER_PRE = "imagenet"
EPOCHS      = 200                       # longer training
BATCH_SIZE  = 10                        # B5 a bit heavier than B4
LR          = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_FRAC = 0.05
NUM_WORKERS = 8

# Loss weights
DICE_W   = 1.0
FOCAL_W  = 0.5
FOCAL_GAMMA = 2.0

# Stochastic Weight Averaging (last K epochs)
SWA_ENABLED      = True
SWA_START_FRAC   = 0.75    # start SWA at 75% of training
SWA_LR           = 1e-4

# Snapshot ensemble - keep top-K best checkpoints by val_dice
SNAPSHOT_K = 3

# Inference
TTA_ENABLED       = True
THRESHOLD_DEFAULT = 0.97
ENSEMBLE_INFER    = True   # average over snapshot ensemble at inference

# ============ 3D detection / rendering ============
MIN_NODULE_VOXELS = 120          # drop predicted blobs smaller than N voxels
DEFAULT_SPACING_XY = 0.7        # mm/pixel fallback if DICOM missing
RENDER_LUNG_SHELL  = True       # show transparent lung context in 3D
