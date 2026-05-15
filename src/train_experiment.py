"""Train one segmentation experiment by name (see experiments.py).

Run:
  python train_experiment.py --exp exp00_baseline                  # eval only
  python train_experiment.py --exp exp03_cons2_tversky              # 50ep pilot
  python train_experiment.py --exp stage2_full --epochs 180         # full train + SWA
  python train_experiment.py --exp list                             # list all

Saves to work/runs_exp/<name>/:
  best.pt, last.pt, swa.pt (if enable_swa), log.txt, config.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import segmentation_models_pytorch as smp
from monai.metrics import DiceMetric, MeanIoU

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import (
    PRE_DIR, SPLITS_JSON, RUNS_DIR, NUM_WORKERS, WEIGHT_DECAY, WARMUP_FRAC,
    SWA_LR,
)
from experiments import EXPERIMENTS, get_experiment, ExperimentConfig
from dataset import get_train_aug
from dataset_v2 import (
    LIDCSeg25DMulti, LIDCSeg25DV2, PRE_DIR_V2,
    make_target_proportion_sampler, make_loss, init_5ch_encoder_from_3ch,
)

EXP_RUNS = Path(__file__).resolve().parent.parent / "work" / "runs_exp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def warmup_cosine(opt, n_warm, n_total):
    def lr_lambda(step):
        if step < n_warm: return step / max(1, n_warm)
        progress = (step - n_warm) / max(1, n_total - n_warm)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(opt, lr_lambda)


def make_model(encoder: str, in_channels: int, pretrained: str = "imagenet"):
    return smp.UnetPlusPlus(
        encoder_name=encoder, encoder_weights=pretrained,
        in_channels=in_channels, classes=1, decoder_attention_type="scse",
    )


def evaluate_dice(model, dl, dice_m, iou_m):
    model.train(False)
    dice_m.reset(); iou_m.reset()
    n_seen = 0
    with torch.no_grad():
        for img, mask in dl:
            img = img.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                pred = (torch.sigmoid(model(img)) > 0.5).float()
            # Skip empty-mask batches in metric to avoid skew
            has_mask = mask.sum(dim=(1, 2, 3)) > 0
            if has_mask.any():
                dice_m(y_pred=pred[has_mask], y=mask[has_mask])
                iou_m(y_pred=pred[has_mask], y=mask[has_mask])
                n_seen += int(has_mask.sum().item())
    if n_seen == 0:
        return 0.0, 0.0
    return float(dice_m.aggregate().item()), float(iou_m.aggregate().item())


def build_datasets(exp: ExperimentConfig):
    """Returns (train_ds, val_ds, train_sampler) according to exp.use_v2_data."""
    splits = json.loads(Path(SPLITS_JSON).read_text())
    if exp.use_v2_data:
        if not PRE_DIR_V2.exists() or not any(PRE_DIR_V2.glob("*.h5")):
            raise FileNotFoundError(
                f"{PRE_DIR_V2} is empty. Run 02b_preprocess_with_negatives.py first."
            )
        train_ds = LIDCSeg25DV2(
            PRE_DIR_V2, splits["train"], mask_mode=exp.mask_mode,
            n_channels=exp.in_channels, augment=get_train_aug(),
        )
        val_ds = LIDCSeg25DV2(
            PRE_DIR_V2, splits["val"], mask_mode=exp.mask_mode,
            n_channels=exp.in_channels, augment=None,
            include_classes=(2, 1),  # val: only nodule + buffer (skip neg for fair Dice)
        )
        if exp.sampler == "class_balanced":
            sampler = make_target_proportion_sampler(
                train_ds, target_proportions=(0.4, 0.3, 0.3),
                oversample_10_20mm=exp.oversample_10_20mm,
                oversample_ratio=exp.oversample_ratio,
            )
            # Cap sampling to ~20K per epoch (similar to v1) so pilot doesn't blow up.
            sampler.num_samples = 20000
        else:
            sampler = None
    else:
        train_ds = LIDCSeg25DMulti(
            PRE_DIR, splits["train"], mask_mode=exp.mask_mode,
            n_channels=exp.in_channels, augment=get_train_aug(),
        )
        val_ds = LIDCSeg25DMulti(
            PRE_DIR, splits["val"], mask_mode=exp.mask_mode,
            n_channels=exp.in_channels, augment=None,
        )
        sampler = None
    return train_ds, val_ds, sampler


def train_one(exp: ExperimentConfig, epochs_override: int = None):
    out_dir = EXP_RUNS / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(exp), indent=2))

    epochs = epochs_override if epochs_override is not None else exp.epochs
    if epochs == 0:
        print(f"[{exp.name}] epochs=0 → skipping training (eval-only experiment)")
        return None

    # --- Datasets ---
    train_ds, val_ds, sampler = build_datasets(exp)
    if len(train_ds) == 0:
        print(f"[{exp.name}] EMPTY train set with mask_mode={exp.mask_mode}. Skipping.")
        return None

    shuffle = sampler is None
    train_dl = DataLoader(train_ds, batch_size=exp.batch_size, shuffle=shuffle,
                          sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=exp.batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True)

    # --- Model ---
    print(f"[{exp.name}] Building model ({exp.encoder}, in_ch={exp.in_channels})")
    model = make_model(exp.encoder, exp.in_channels).to(DEVICE)
    if exp.init_from:
        ckpt_path = RUNS_DIR / exp.init_from
        if ckpt_path.exists():
            if exp.in_channels == 3:
                ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(ck["model"], strict=False)
                print(f"  Loaded init from {ckpt_path}")
            else:
                init_5ch_encoder_from_3ch(model, ckpt_path)
        else:
            print(f"  WARN: init_from={ckpt_path} not found; using ImageNet weights")

    # --- Loss / Opt / Sched ---
    loss_fn = make_loss(exp.loss, alpha=exp.tversky_alpha,
                        beta=exp.tversky_beta, gamma=exp.focal_tversky_gamma)
    # Use 1/2 lr if dice_focal + class_balanced sampler (NaN-prone with empty masks)
    effective_lr = exp.lr
    if exp.loss == "dice_focal" and exp.sampler == "class_balanced":
        effective_lr = exp.lr * 0.5
        print(f"  Reduced lr to {effective_lr} (dice_focal + class_balanced is NaN-prone)")
    opt = AdamW(model.parameters(), lr=effective_lr, weight_decay=WEIGHT_DECAY)
    n_steps = epochs * len(train_dl)
    sched = warmup_cosine(opt, n_warm=int(WARMUP_FRAC * n_steps), n_total=n_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    dice_m = DiceMetric(include_background=False, reduction="mean")
    iou_m = MeanIoU(include_background=False, reduction="mean")

    # --- SWA setup (only for stage2 long runs) ---
    swa_enabled = exp.enable_swa
    swa_start_epoch = int(0.75 * epochs) if swa_enabled else epochs + 1
    swa_model = None; swa_sched = None
    if swa_enabled:
        print(f"  SWA enabled, will start at epoch {swa_start_epoch+1}")

    log_path = out_dir / "log.txt"
    best_dice = 0.0
    print(f"[{exp.name}] Training {epochs} epochs, "
          f"loss={exp.loss}(α={exp.tversky_alpha},β={exp.tversky_beta}) "
          f"sampler={exp.sampler} on {len(train_ds)} train slices")
    for ep in range(epochs):
        model.train(True)
        t0 = time.time(); loss_sum = 0.0; n = 0
        for img, mask in train_dl:
            img = img.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                pred = model(img)
                loss = loss_fn(pred, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt); scaler.update()
            if ep < swa_start_epoch:
                sched.step()
            loss_sum += float(loss.item()) * img.size(0); n += img.size(0)
        train_loss = loss_sum / max(n, 1)

        # Switch to SWA after warmup phase
        if swa_enabled and ep == swa_start_epoch:
            swa_model = AveragedModel(model).to(DEVICE)
            swa_sched = SWALR(opt, swa_lr=SWA_LR, anneal_epochs=3)
            print(f"  --> Switched to SWA at epoch {ep+1}, lr={SWA_LR}")
        if swa_model is not None:
            swa_model.update_parameters(model)
            swa_sched.step()

        val_dice, val_iou = evaluate_dice(model, val_dl, dice_m, iou_m)
        dt = time.time() - t0
        msg = (f"ep {ep+1:>3}/{epochs}  loss={train_loss:.4f}  "
               f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f}  "
               f"lr={opt.param_groups[0]['lr']:.5f}  ({dt:.0f}s)")
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(msg + "\n")
        ck = {"model": model.state_dict(), "epoch": ep+1, "val_dice": val_dice,
              "encoder": exp.encoder, "in_channels": exp.in_channels,
              "experiment": exp.name}
        torch.save(ck, out_dir / "last.pt")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(ck, out_dir / "best.pt")
            print(f"  -> saved best.pt (dice={best_dice:.4f})")

    # SWA finalisation
    if swa_model is not None:
        print(f"\n  Updating SWA BN statistics ...")
        update_bn(train_dl, swa_model, device=DEVICE)
        swa_dice, swa_iou = evaluate_dice(swa_model, val_dl, dice_m, iou_m)
        print(f"  SWA val_dice={swa_dice:.4f}  val_iou={swa_iou:.4f}")
        torch.save({"model": swa_model.module.state_dict(), "epoch": epochs,
                    "val_dice": swa_dice, "encoder": exp.encoder,
                    "in_channels": exp.in_channels, "experiment": exp.name,
                    "swa": True}, out_dir / "swa.pt")
        if swa_dice > best_dice:
            best_dice = swa_dice
    print(f"[{exp.name}] DONE. best_val_dice={best_dice:.4f}")
    return best_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    help="Experiment name (or 'all' / 'pilot' / 'list')")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override exp.epochs (useful for smoke test)")
    args = ap.parse_args()

    if args.exp == "list":
        from experiments import list_experiments
        list_experiments(); return
    if args.exp in ("all", "pilot"):
        from experiments import PILOT_ORDER
        names = PILOT_ORDER if args.exp == "pilot" else list(EXPERIMENTS.keys())
        results = {}
        for name in names:
            print(f"\n{'='*70}\n=== {name} ===\n{'='*70}")
            try:
                results[name] = train_one(EXPERIMENTS[name], args.epochs)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[{name}] FAILED: {type(e).__name__}: {e}")
                results[name] = None
        print(f"\n=== SUMMARY ===")
        for name, dice in results.items():
            print(f"  {name:<28} best_val_dice={dice}")
    else:
        train_one(get_experiment(args.exp), args.epochs)


if __name__ == "__main__":
    main()
