"""Step 05 - Train UNet++ on LIDC-IDRI (TOP-1 settings).

Architecture:  UNet++ + EfficientNet-B5 (ImageNet pretrained) + SCSE attention
Input:         3 stacked CT slices (2.5D)
Loss:          DiceFocalLoss
Optimizer:     AdamW + warmup + cosine decay -> SWA in last quarter
Mixed precision (fp16) on V100.

Saves:
  best.pt           highest val_dice
  snapshot_<i>.pt   top-K best by val_dice (for ensemble)
  swa.pt            SWA-averaged model weights (BN-recalibrated)
  last.pt           latest epoch
  log.txt           per-epoch log

Resume: pass --resume <ckpt>
"""
import argparse
import json
import math
import time
import heapq
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import segmentation_models_pytorch as smp
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU

from configs import (
    PRE_DIR, SPLITS_JSON, RUNS_DIR,
    ENCODER, ENCODER_PRE, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, WARMUP_FRAC,
    NUM_WORKERS, DICE_W, FOCAL_W, FOCAL_GAMMA,
    SWA_ENABLED, SWA_START_FRAC, SWA_LR, SNAPSHOT_K,
)
from dataset import LIDCSeg25D, get_train_aug


def warmup_cosine(opt, n_warm, n_total):
    def lr_lambda(step):
        if step < n_warm:
            return step / max(1, n_warm)
        progress = (step - n_warm) / max(1, n_total - n_warm)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(opt, lr_lambda)


def make_model(encoder=ENCODER, in_ch=3, classes=1, pretrained=ENCODER_PRE):
    return smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=pretrained,
        in_channels=in_ch,
        classes=classes,
        decoder_attention_type="scse",
    )


def evaluate(model, dl, device, dice_m, iou_m):
    model.train(False)
    dice_m.reset(); iou_m.reset()
    with torch.no_grad():
        for img, mask in dl:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                pred = (torch.sigmoid(model(img)) > 0.5).float()
            dice_m(y_pred=pred, y=mask)
            iou_m(y_pred=pred, y=mask)
    return float(dice_m.aggregate().item()), float(iou_m.aggregate().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PRE_DIR))
    ap.add_argument("--splits", default=str(SPLITS_JSON))
    ap.add_argument("--out", default=str(RUNS_DIR))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--bs", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--no_swa", action="store_true")
    ap.add_argument("--snapshot_k", type=int, default=SNAPSHOT_K)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.load(open(args.splits))
    train_ds = LIDCSeg25D(args.root, splits["train"], augment=get_train_aug())
    val_ds   = LIDCSeg25D(args.root, splits["val"], augment=None)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                          num_workers=args.workers, pin_memory=True,
                          persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(encoder=args.encoder).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Device: {device}, GPU: "
          f"{torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")
    print(f"Model: UNet++ / {args.encoder} pretrained ImageNet / SCSE  ({n_params:.1f}M params)")
    print(f"Train slices: {len(train_ds)}, Val slices: {len(val_ds)}")

    loss_fn = DiceFocalLoss(sigmoid=True, gamma=FOCAL_GAMMA,
                            lambda_dice=DICE_W, lambda_focal=FOCAL_W)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    n_steps = args.epochs * len(train_dl)
    sched = warmup_cosine(opt, n_warm=int(WARMUP_FRAC * n_steps), n_total=n_steps)
    scaler = torch.amp.GradScaler("cuda")
    dice_m = DiceMetric(include_background=False, reduction="mean")
    iou_m  = MeanIoU(include_background=False, reduction="mean")

    swa_enabled = SWA_ENABLED and not args.no_swa
    swa_start_epoch = int(SWA_START_FRAC * args.epochs) if swa_enabled else args.epochs + 1
    swa_model = None
    swa_sched = None
    print(f"SWA: {'enabled, start at epoch ' + str(swa_start_epoch+1) if swa_enabled else 'disabled'}")
    print(f"Snapshot ensemble: top-{args.snapshot_k}")

    start_epoch = 0
    best_dice = 0.0
    snapshot_heap = []   # (val_dice, epoch, file_path) — small heap of size K
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck: sched.load_state_dict(ck["scheduler"])
        start_epoch = ck.get("epoch", 0)
        best_dice = ck.get("best_dice", 0.0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}, best_dice {best_dice:.4f}")

    log_path = out_dir / "log.txt"
    for epoch in range(start_epoch, args.epochs):
        model.train(True)
        t0 = time.time(); tot = 0.0; n = 0
        for img, mask in train_dl:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                pred = model(img)
                loss = loss_fn(pred, mask)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if epoch < swa_start_epoch:
                sched.step()
            tot += loss.item() * img.size(0); n += img.size(0)
        train_loss = tot / max(n, 1)

        # Switch to SWA after warmup phase
        if swa_enabled and epoch == swa_start_epoch:
            swa_model = AveragedModel(model).to(device)
            swa_sched = SWALR(opt, swa_lr=SWA_LR, anneal_epochs=3)
            print(f"--> Switched to SWA at epoch {epoch+1}, lr={SWA_LR}")
        if swa_model is not None:
            swa_model.update_parameters(model)
            swa_sched.step()

        val_dice, val_iou = evaluate(model, val_dl, device, dice_m, iou_m)
        dt = time.time() - t0

        msg = (f"epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  "
               f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f}  "
               f"lr={opt.param_groups[0]['lr']:.2e}  ({dt:.0f}s)")
        print(msg, flush=True)
        with open(log_path, "a") as f: f.write(msg + "\n")

        ck = {"model": model.state_dict(),
              "optimizer": opt.state_dict(),
              "scheduler": sched.state_dict(),
              "epoch": epoch + 1,
              "val_dice": val_dice,
              "best_dice": max(best_dice, val_dice),
              "encoder": args.encoder}
        torch.save(ck, out_dir / "last.pt")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(ck, out_dir / "best.pt")
            print(f"  -> saved best.pt (dice={best_dice:.4f})")

        # Snapshot ensemble: keep top-K by val_dice
        snap_path = out_dir / f"snapshot_e{epoch+1:03d}.pt"
        if len(snapshot_heap) < args.snapshot_k:
            torch.save({"model": model.state_dict(), "epoch": epoch+1,
                        "val_dice": val_dice, "encoder": args.encoder}, snap_path)
            heapq.heappush(snapshot_heap, (val_dice, epoch+1, str(snap_path)))
        elif val_dice > snapshot_heap[0][0]:
            _, _, old = heapq.heapreplace(snapshot_heap,
                                          (val_dice, epoch+1, str(snap_path)))
            torch.save({"model": model.state_dict(), "epoch": epoch+1,
                        "val_dice": val_dice, "encoder": args.encoder}, snap_path)
            Path(old).unlink(missing_ok=True)

    # Finalize SWA: recompute BN statistics
    if swa_model is not None:
        print("\nUpdating SWA BN statistics ...")
        update_bn(train_dl, swa_model, device=device)
        swa_dice, swa_iou = evaluate(swa_model, val_dl, device, dice_m, iou_m)
        print(f"SWA val_dice={swa_dice:.4f}  val_iou={swa_iou:.4f}")
        torch.save({"model": swa_model.module.state_dict(), "epoch": args.epochs,
                    "val_dice": swa_dice, "encoder": args.encoder, "swa": True},
                   out_dir / "swa.pt")
        if swa_dice > best_dice:
            best_dice = swa_dice

    snapshots = sorted(snapshot_heap, reverse=True)
    print(f"\nDone. best_dice={best_dice:.4f}")
    print(f"Snapshots kept: {[(round(d,4), e) for d,e,_ in snapshots]}")
    print(f"Next: python 06_evaluate.py --ckpts {','.join(p for _,_,p in snapshots)}")


if __name__ == "__main__":
    main()
