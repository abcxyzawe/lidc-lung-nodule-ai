"""Cache probability volumes for all patients in a panel using a given segmentation ckpt.

Used by retrain_fpr_pipeline.py to populate probs for train_panel before
extracting FPR candidates with a new (stage2_full) segmentation model.

Run:
  python cache_panel_probs.py --panel train --ckpt work/runs_exp/stage2_full/best.pt \
      --out-dir work/academic/probs_exp/stage2_full/
  python cache_panel_probs.py --panel train  # uses default best.pt + writes default probs/
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from configs import WORK, RUNS_DIR, HU_LO, HU_HI
from panels import load_panel
from benchmark import load_patient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(x):
    x = np.clip(x.astype(np.float32), HU_LO, HU_HI)
    return (x - HU_LO) / (HU_HI - HU_LO)


def predict_volume(model, vol, n_channels, tta=True):
    N, H, W = vol.shape
    prob = np.zeros((N, H, W), dtype=np.float32)
    half = n_channels // 2
    with torch.no_grad():
        for i in range(N):
            idxs = [max(0, min(N-1, i + off)) for off in range(-half, half + 1)]
            stk = np.stack([normalize(vol[j]) for j in idxs], axis=0)
            x = torch.from_numpy(stk).unsqueeze(0).float().to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                p1 = torch.sigmoid(model(x))
                if tta:
                    p2 = torch.sigmoid(model(torch.flip(x, dims=[-1]))).flip(dims=[-1])
                    p = (p1 + p2) / 2
                else:
                    p = p1
            prob[i] = p[0, 0].float().cpu().numpy()
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["debug", "train", "val", "test"], required=True)
    ap.add_argument("--ckpt", default=str(RUNS_DIR / "best.pt"),
                    help="Segmentation checkpoint .pt")
    ap.add_argument("--encoder", default="efficientnet-b5")
    ap.add_argument("--out-dir", default="",
                    help="Output dir for prob.npy files (default: work/academic/probs/)")
    ap.add_argument("--in-channels", type=int, default=3,
                    help="Override input channels (auto-detected from ckpt if available)")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (WORK / "academic" / "probs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    in_ch = ck.get("in_channels", args.in_channels)
    encoder = ck.get("encoder", args.encoder)
    model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None,
                              in_channels=in_ch, classes=1,
                              decoder_attention_type="scse").to(DEVICE)
    model.load_state_dict(ck["model"])
    model.train(False)
    print(f"Loaded {args.ckpt} (encoder={encoder}, in_ch={in_ch})")
    print(f"Output dir: {out_dir}")

    pids = load_panel(args.panel)
    print(f"Caching {len(pids)} patients from panel '{args.panel}'")
    skipped = cached = 0
    t0 = time.time()
    for i, pid in enumerate(pids):
        out_path = out_dir / f"{pid}_tta{int(not args.no_tta)}.npy"
        if out_path.exists():
            skipped += 1; continue
        d = load_patient(pid)
        if d is None:
            print(f"  [{i+1}/{len(pids)}] {pid}: SKIP (no h5)"); continue
        ts = time.time()
        prob = predict_volume(model, d["vol"], in_ch, tta=not args.no_tta)
        np.save(out_path, prob.astype(np.float16))
        cached += 1
        print(f"  [{i+1}/{len(pids)}] {pid}: {time.time()-ts:.1f}s  ({d['vol'].shape[0]} slices)",
              flush=True)
    print(f"\nDone. {cached} new, {skipped} skipped.  Total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
