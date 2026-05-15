"""Extract candidate patches + labels for false-positive reduction training.

For each patient with a cached prob volume:
  1. Generate candidates with permissive threshold (default 0.40)
  2. For each candidate, crop 48^3 HU patch around centroid
  3. Label = 1 if matches a GT nodule (LUNA16 radius rule), 0 if not
  4. Save as npz file: patches [N,48,48,48] uint16, labels [N], conf [N], pids [N]

Run:
  python extract_fpr_candidates.py --panel val --out work/academic/fpr_train.npz
  python extract_fpr_candidates.py --panel test --out work/academic/fpr_test.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))
from configs import WORK  # noqa: E402
from panels import load_panel  # noqa: E402
from benchmark import load_patient  # noqa: E402
from tune_detection_params import (  # noqa: E402
    PROBS_DIR, cache_lung_mask, precompute_threshold_blobs, filter_and_match,
)

ACADEMIC = WORK / "academic"
DEFAULT_PATCH = 48      # 48^3 patch -> ~33mm at 0.7mm xy / 1.25mm z
DEFAULT_THR = 0.40      # permissive — catch most candidates


def extract_patch(vol: np.ndarray, centroid_voxel, patch_size: int = 48) -> np.ndarray:
    """Crop patch_size^3 HU patch around centroid, padding with -1000 (air) if needed."""
    H = patch_size // 2
    cz, cy, cx = [int(round(c)) for c in centroid_voxel]
    N, Hv, Wv = vol.shape
    z0, z1 = max(0, cz - H), min(N, cz + H)
    y0, y1 = max(0, cy - H), min(Hv, cy + H)
    x0, x1 = max(0, cx - H), min(Wv, cx + H)
    crop = vol[z0:z1, y0:y1, x0:x1]
    out = np.full((patch_size, patch_size, patch_size), -1000, dtype=np.int16)
    pz0 = H - (cz - z0)
    py0 = H - (cy - y0)
    px0 = H - (cx - x0)
    out[pz0:pz0+crop.shape[0], py0:py0+crop.shape[1], px0:px0+crop.shape[2]] = crop
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["debug", "train", "val", "test"], default="val")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THR,
                    help="Detection threshold for candidate generation")
    ap.add_argument("--patch-size", type=int, default=DEFAULT_PATCH)
    ap.add_argument("--min-voxels", type=int, default=20,
                    help="Permissive — let small candidates through for FPR to filter")
    ap.add_argument("--max-elong", type=float, default=99.0)
    ap.add_argument("--probs-dir", default="",
                    help="Custom probs cache dir (default: work/academic/probs/). "
                         "Use this when extracting from a non-default segmentation model "
                         "(e.g., work/academic/probs_exp/stage2_full/)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pids = load_panel(args.panel)
    # Honour custom probs dir if specified
    global PROBS_DIR
    if args.probs_dir:
        from pathlib import Path as _Path
        PROBS_DIR = _Path(args.probs_dir)
        print(f"Using custom probs dir: {PROBS_DIR}")
    out_path = Path(args.out) if args.out else ACADEMIC / f"fpr_{args.panel}.npz"

    patches, labels, confs, diam_pred, diam_gt, sources = [], [], [], [], [], []
    n_pat = 0
    t0 = time.time()
    for pid in pids:
        cp = PROBS_DIR / f"{pid}_tta1.npy"
        if not cp.exists(): continue
        d = load_patient(pid)
        if d is None: continue
        prob = np.load(cp).astype(np.float32)
        lung = cache_lung_mask(pid, d["vol"])
        sp = d["voxel_sp"]
        ldist = distance_transform_edt(lung, sampling=list(sp)).astype(np.float32)
        blobs = precompute_threshold_blobs(prob, lung, sp, args.threshold)
        # Use minimal post-proc to get many candidates
        res = filter_and_match(
            blobs, ldist, sp, d["gt_nodules"],
            min_voxels=args.min_voxels, max_elong=args.max_elong,
            merge_dist_mm=10.0, subpleural_min_mm=0.0,
        )
        matched_pi = {m[1] for m in res["matched"]}
        gt_for_pi = {m[1]: m[0] for m in res["matched"]}
        for pi, p in enumerate(res["preds"]):
            patch = extract_patch(d["vol"], p["centroid_voxel"], args.patch_size)
            patches.append(patch)
            label = 1 if pi in matched_pi else 0
            labels.append(label)
            confs.append(p["confidence"])
            diam_pred.append(p["diameter_mm"])
            if pi in matched_pi:
                gi = gt_for_pi[pi]
                diam_gt.append(d["gt_nodules"][gi]["diam_mm"])
            else:
                diam_gt.append(0.0)
            sources.append(pid)
        n_pat += 1
        n_pos = sum(labels[-len(res["preds"]):])
        n_neg = len(res["preds"]) - n_pos
        if n_pat % 10 == 0 or n_pat <= 5:
            print(f"  [{n_pat}/{len(pids)}] {pid}: +{len(res['preds'])} cands "
                  f"({n_pos} pos, {n_neg} neg)  total={len(patches)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    patches_arr = np.stack(patches, 0).astype(np.int16)
    labels_arr = np.array(labels, dtype=np.uint8)
    confs_arr = np.array(confs, dtype=np.float32)
    print(f"\n=== {args.panel} candidates ===")
    print(f"  N patients: {n_pat}")
    print(f"  N candidates: {len(patches)}")
    print(f"  Positives:  {labels_arr.sum()} ({100*labels_arr.mean():.1f}%)")
    print(f"  Negatives:  {(labels_arr==0).sum()} ({100*(labels_arr==0).mean():.1f}%)")
    print(f"  Patch shape: {patches_arr.shape} ({patches_arr.nbytes / 1e6:.0f} MB)")

    np.savez_compressed(out_path,
                         patches=patches_arr, labels=labels_arr,
                         confs=confs_arr,
                         diam_pred=np.array(diam_pred, dtype=np.float32),
                         diam_gt=np.array(diam_gt, dtype=np.float32),
                         sources=np.array(sources))
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
