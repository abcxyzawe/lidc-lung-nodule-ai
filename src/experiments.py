"""Registry of segmentation retraining experiments — pilot plan.

Stage 1 — pilot 50 epochs each, init from current best.pt:
  EXP00: baseline (re-eval current best.pt on test_panel; epochs=0)
  EXP01: baseline + negative slice sampler  (40/30/30 mix)
  EXP02: consensus2 only
  EXP03: consensus2 + TverskyLoss(0.3, 0.7)
  EXP04: consensus2 + TverskyLoss + negative sampler
  EXP05: consensus2 + TverskyLoss + negative sampler + oversample 10-20mm

Stage 2 — full train (150-200 epochs) of best Stage 1 cfg, with SWA + snapshot ensemble.

Run pilot:
  python pilot_runner.py
Run one experiment:
  python train_experiment.py --exp exp03_cons2_tversky --epochs 50
List:
  python train_experiment.py --exp list
"""
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    name: str
    mask_mode: str = "union"               # union | consensus2 | consensus3
    loss: str = "dice_focal"               # dice_focal | tversky | focal_tversky
    tversky_alpha: float = 0.5
    tversky_beta: float = 0.5
    focal_tversky_gamma: float = 1.0

    # Sampling — uses preprocessed_v2 h5 with slice_classes
    use_v2_data: bool = False              # True = read work/preprocessed_v2/, False = work/preprocessed/
    sampler: str = "uniform"               # uniform | class_balanced
    class_weights: tuple = (1.0, 1.0, 1.0) # (nodule, buffer, negative)  — used iff sampler=class_balanced
    oversample_10_20mm: bool = False       # extra weight on 10-20mm nodule slices
    oversample_ratio: float = 3.0

    # Architecture
    in_channels: int = 3                   # 3 | 5 | 7
    encoder: str = "efficientnet-b5"

    # Training
    epochs: int = 50
    lr: float = 1e-4                       # fine-tune from best.pt
    batch_size: int = 10
    init_from: str = "best.pt"             # transfer learning starting point ("" = scratch)
    enable_swa: bool = False               # only for full-stage 2 runs
    note: str = ""


# ---------------------------------------------------------------------------
# STAGE 1 — pilot 50-epoch experiments matching the user's plan verbatim
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    # EXP00 — baseline. epochs=0 means "skip training, eval only"
    "exp00_baseline": ExperimentConfig(
        name="exp00_baseline",
        epochs=0,
        note="Re-eval current production best.pt (no retrain). Reference column.",
    ),

    # EXP01 — baseline mask/loss + negative slice sampler
    "exp01_negsampler": ExperimentConfig(
        name="exp01_negsampler",
        use_v2_data=True,
        sampler="class_balanced",
        class_weights=(1.0, 0.75, 0.75),   # nodule:buffer:negative ≈ 40:30:30 in samples
        note="Add negative/background slices to training mix. Tests whether "
             "distribution shift on full CT improves WITHOUT changing mask/loss.",
    ),

    # EXP02 — consensus2 mask only (cleaner labels)
    "exp02_consensus2": ExperimentConfig(
        name="exp02_consensus2",
        mask_mode="consensus2",
        note="Drop noisy 1/4-radiologist annotations. Cleaner training signal "
             "but less data. Hypothesis: 22% invisible-nodule problem partly "
             "from noisy labels.",
    ),

    # EXP03 — consensus2 + Tversky (penalty FN)
    "exp03_cons2_tversky": ExperimentConfig(
        name="exp03_cons2_tversky",
        mask_mode="consensus2",
        loss="tversky",
        tversky_alpha=0.3,
        tversky_beta=0.7,
        note="Add Tversky(α=0.3, β=0.7) — 7×stronger FN penalty. Targets the "
             "10-20mm bucket (32% invisible).",
    ),

    # EXP04 — full combo (excluding 5-slice + oversample)
    "exp04_cons2_tversky_negs": ExperimentConfig(
        name="exp04_cons2_tversky_negs",
        mask_mode="consensus2",
        loss="tversky",
        tversky_alpha=0.3,
        tversky_beta=0.7,
        use_v2_data=True,
        sampler="class_balanced",
        class_weights=(1.0, 0.75, 0.75),
        note="EXP03 + negative sampler. Both data fix and loss fix together. "
             "If EXP03 helps and EXP01 helps, this should help most.",
    ),

    # EXP05 — full pilot stack
    "exp05_full_pilot": ExperimentConfig(
        name="exp05_full_pilot",
        mask_mode="consensus2",
        loss="tversky",
        tversky_alpha=0.3,
        tversky_beta=0.7,
        use_v2_data=True,
        sampler="class_balanced",
        class_weights=(1.0, 0.75, 0.75),
        oversample_10_20mm=True,
        oversample_ratio=3.0,
        note="EXP04 + oversample 10-20mm bucket 3×. Targets the worst stratified "
             "bucket per Phase 3 analysis. Most aggressive pilot config.",
    ),

    # EXP06 — best combo + 5-slice (RUN ONLY AFTER pilot identifies winner)
    # Edit this config to mirror pilot winner BEFORE running.
    "exp06_winner_5slice": ExperimentConfig(
        name="exp06_winner_5slice",
        mask_mode="consensus2",          # placeholder — copy from pilot winner
        loss="tversky",                  # placeholder — copy from pilot winner
        tversky_alpha=0.3,
        tversky_beta=0.7,
        use_v2_data=True,                # placeholder — copy from pilot winner
        sampler="class_balanced",
        class_weights=(1.0, 0.75, 0.75),
        in_channels=5,                   # the only forced change vs winner
        epochs=50,
        note="ONLY RUN after pilot. Take winner config + bump to 5-slice 2.5D. "
             "Tests whether wider axial context adds on top of best Stage 1 config.",
    ),

    # ---------------------------------------------------------------------------
    # STAGE 2 — long full train of the best Stage 1 cfg (placeholder; copy from winner)
    # ---------------------------------------------------------------------------
    "stage2_full": ExperimentConfig(
        name="stage2_full",
        # WINNER (re-creation 2026-05-15): cons2+tversky on v1 data. F1=0.535 seg-only,
        # F1=0.606 with FPR v3 — confirmed on test_panel previously. v2 data + class_balanced
        # sampler caused NaN under AMP fp16 (3/3 v2 exps failed). Big Swing fix never beat v1.
        mask_mode="consensus2",
        loss="tversky",
        tversky_alpha=0.3,
        tversky_beta=0.7,
        use_v2_data=False,
        sampler="uniform",
        class_weights=(1.0, 1.0, 1.0),
        epochs=180,
        enable_swa=True,
        init_from="best.pt",
        note="Winner config re-creation. v1 data, uniform sampler, tversky α=0.3 β=0.7, 180ep+SWA.",
    ),

    # Optional: 5-slice variant if Stage 1 confirms direction
    "stage2_full_5slice": ExperimentConfig(
        name="stage2_full_5slice",
        mask_mode="consensus2",
        loss="tversky",
        tversky_alpha=0.3,
        tversky_beta=0.7,
        use_v2_data=True,
        sampler="class_balanced",
        class_weights=(1.0, 0.75, 0.75),
        in_channels=5,
        epochs=180,
        enable_swa=True,
        init_from="best.pt",
        note="5-slice 2.5D version of stage2_full. Run only if Stage 1 + 3-slice "
             "stage2 are still leaving recall on the table.",
    ),
}


def get_experiment(name: str) -> ExperimentConfig:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{name}'. Choices: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name]


def list_experiments():
    print(f"{len(EXPERIMENTS)} experiments registered:")
    for name, cfg in EXPERIMENTS.items():
        v2 = "V2" if cfg.use_v2_data else "v1"
        ovr = "+over10-20" if cfg.oversample_10_20mm else ""
        print(f"  {name:<28} mask={cfg.mask_mode:<11} loss={cfg.loss:<14} "
              f"in_ch={cfg.in_channels} ep={cfg.epochs:>3} "
              f"sampler={cfg.sampler:<15} {v2} {ovr}")


PILOT_ORDER = [
    "exp00_baseline",
    "exp01_negsampler",
    "exp02_consensus2",
    "exp03_cons2_tversky",
    "exp04_cons2_tversky_negs",
    "exp05_full_pilot",
]


if __name__ == "__main__":
    list_experiments()
