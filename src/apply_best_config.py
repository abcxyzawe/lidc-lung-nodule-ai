"""Apply work/academic/best_config.json to webapp defaults + configs.py.

Reads the tuner output and patches:
  src/configs.py       -> MIN_NODULE_VOXELS, THRESHOLD_DEFAULT
  src/webapp/app.py    -> threshold, min_voxels, max_elongation,
                          merge_nearby_nodules max_dist_mm,
                          filter_subpleural min_dist_mm

Run: python apply_best_config.py
     python apply_best_config.py --threshold 0.40   # override op threshold
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEST_JSON = ROOT / "work" / "academic" / "best_config.json"
CONFIGS_PY = ROOT / "src" / "configs.py"
APP_PY = ROOT / "src" / "webapp" / "app.py"


def patch_file(path: Path, patches: list[tuple[str, str]]) -> int:
    """Apply (regex, replacement) patches. Returns number of changes."""
    txt = path.read_text(encoding="utf-8")
    n = 0
    for pat, rep in patches:
        new_txt, k = re.subn(pat, rep, txt, count=1)
        if k:
            n += 1
            txt = new_txt
            print(f"  [patched] {path.name}: {pat[:50]} -> {rep[:60]}")
        else:
            print(f"  [SKIP   ] {path.name}: pattern not found {pat[:50]}")
    path.write_text(txt, encoding="utf-8")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override operating threshold (else use best_threshold_op_at_1fp)")
    ap.add_argument("--config-json", default=str(BEST_JSON))
    args = ap.parse_args()

    cfg = json.loads(Path(args.config_json).read_text())
    bc = cfg["best_config"]
    thr = args.threshold if args.threshold is not None else cfg["best_threshold_op_at_1fp"]
    min_vox = int(bc["min_voxels"])
    max_elong = float(bc["max_elong"])
    merge = float(bc["merge_dist_mm"])
    subp = float(bc["subpleural_min_mm"])
    print(f"Applying best config: thr={thr:.2f}  min_vox={min_vox}  "
          f"max_elong={max_elong}  merge={merge}mm  subp={subp}mm")
    print(f"  CPM={cfg['cpm']:.3f}  S@1FP={cfg['sens_at_1fp']:.3f}  "
          f"S@2FP={cfg['sens_at_2fp']:.3f}\n")

    print("[1/2] Patching configs.py")
    patch_file(CONFIGS_PY, [
        (r"MIN_NODULE_VOXELS\s*=\s*\d+", f"MIN_NODULE_VOXELS = {min_vox}"),
        (r"THRESHOLD_DEFAULT\s*=\s*[\d.]+", f"THRESHOLD_DEFAULT = {thr:.2f}"),
    ])
    print("\n[2/2] Patching webapp/app.py")
    patch_file(APP_PY, [
        (r"predict_nodules\(vol,\s*threshold=[\d.]+",
            f"predict_nodules(vol, threshold={thr:.2f}"),
        (r"find_nodules\(pred,\s*voxel_sp,\s*min_voxels=\d+,\s*max_elongation=[\d.]+",
            f"find_nodules(pred, voxel_sp, min_voxels={min_vox}, max_elongation={max_elong}"),
        (r"merge_nearby_nodules\(nodules,\s*voxel_sp,\s*max_dist_mm=[\d.]+\)",
            f"merge_nearby_nodules(nodules, voxel_sp, max_dist_mm={merge})"),
        (r"filter_subpleural\(nodules,\s*lung,\s*voxel_sp,\s*min_dist_mm=[\d.]+\)",
            f"filter_subpleural(nodules, lung, voxel_sp, min_dist_mm={subp})"),
    ])
    print("\nDone. Restart webapp backend to pick up changes.")


if __name__ == "__main__":
    main()
