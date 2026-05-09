"""Show LIDC-IDRI ground truth nodules for a given patient.

Reads the preprocessed h5 file (work/preprocessed/<PID>.h5) and prints the
4-radiologist consensus annotations: nodule count, diameter, malignancy,
spiculation, texture, margin, calcification, subtlety, location.

Use this to compare against the AI prediction from the webapp.

Run:
  python show_gt.py LIDC-IDRI-0002
  python show_gt.py LIDC-IDRI-0751 --merge-dist 15
  python show_gt.py LIDC-IDRI-0001 --json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from configs import PRE_DIR


# Lung-RADS bands for solid nodules (simplified)
def lung_rads(diameter_mm: float) -> str:
    d = diameter_mm
    if d < 6:   return "LR2 (bỏ qua, < 6mm)"
    if d < 8:   return "LR3 (theo dõi 6 tháng)"
    if d < 15:  return "LR4A (suspicious — 3 tháng)"
    if d < 30:  return "LR4B (very suspicious — PET/biopsy)"
    return "LR4X (highly suspicious — biopsy)"


def malignancy_band(mean_score):
    if mean_score is None: return "không đánh giá"
    if mean_score < 2: return "lành tính"
    if mean_score < 3: return "có thể lành"
    if mean_score < 4: return "không xác định"
    if mean_score < 4.5: return "có thể ác"
    return "ác tính cao"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patient_id", help="ví dụ LIDC-IDRI-0002")
    ap.add_argument("--merge-dist", type=float, default=12.0,
                    help="Khoảng cách centroid (mm) để gộp cùng 1 nodule giữa các radiologist")
    ap.add_argument("--min-voxels", type=int, default=5,
                    help="Min voxel để giữ annotation (mặc định 5; đặt 0 để xem cả point markers)")
    ap.add_argument("--all", action="store_true",
                    help="Hiện cả point markers (nodule < 3mm + non-nodule)")
    ap.add_argument("--json", action="store_true",
                    help="Xuất JSON thay vì human-readable")
    args = ap.parse_args()
    if args.all:
        args.min_voxels = 0

    h5p = PRE_DIR / f"{args.patient_id}.h5"
    if not h5p.exists():
        print(f"Không tìm thấy {h5p}.", file=sys.stderr)
        print("Chạy 02_preprocess.py trước hoặc kiểm tra patient ID.", file=sys.stderr)
        sys.exit(1)

    output = {"patient_id": args.patient_id, "series": []}

    with h5py.File(h5p, "r") as f:
        for series_key in f.keys():
            g = f[series_key]
            images = g["images"][:]
            masks = g["mask_per_rad"][:]
            nodule_meta = json.loads(g.attrs["nodule_meta"])
            pixel_sp = json.loads(g.attrs["pixel_spacing"])
            slice_thk = float(g.attrs["slice_thickness"])
            voxel_vol = pixel_sp[0] * pixel_sp[1] * slice_thk

            series_info = {
                "series_uid": g.attrs["series_uid"],
                "shape": list(images.shape),
                "voxel_spacing_zyx_mm": [slice_thk, pixel_sp[0], pixel_sp[1]],
                "n_radiologists_total": len(json.loads(g.attrs["radiologists"])),
            }

            # Group annotations by (radiologist, nodule_id)
            by = defaultdict(list)
            for e in nodule_meta:
                by[(e["rad_channel"], e["nodule_id"])].append(e)

            per_rad_nodules = []
            for (rad_ch, nod_id), entries in by.items():
                slice_idxs = [int(e["slice_idx"]) for e in entries
                              if 0 <= int(e["slice_idx"]) < masks.shape[0]]
                if not slice_idxs: continue
                sub = masks[slice_idxs, rad_ch]
                if sub.sum() < args.min_voxels: continue
                coords = np.argwhere(sub > 0)
                if len(coords) == 0: continue
                cz_local = int(coords[:, 0].mean())
                cz = slice_idxs[min(cz_local, len(slice_idxs) - 1)]
                cy = int(coords[:, 1].mean())
                cx = int(coords[:, 2].mean())
                voxels = int(sub.sum())
                vol = voxels * voxel_vol
                diam = 2 * (3 * vol / (4 * np.pi)) ** (1 / 3)
                first = entries[0]
                per_rad_nodules.append({
                    "rad": rad_ch, "nodule_id": nod_id,
                    "cz": cz, "cy": cy, "cx": cx,
                    "voxels": voxels, "vol_mm3": float(vol),
                    "diam_mm": float(diam),
                    "malignancy": first.get("malignancy", ""),
                    "spiculation": first.get("spiculation", ""),
                    "calcification": first.get("calcification", ""),
                    "texture": first.get("texture", ""),
                    "margin": first.get("margin", ""),
                    "subtlety": first.get("subtlety", ""),
                    "lobulation": first.get("lobulation", ""),
                    "sphericity": first.get("sphericity", ""),
                })

            # Cluster across radiologists by physical centroid distance
            per_rad_nodules.sort(key=lambda n: -n["vol_mm3"])
            merged = []
            used = [False] * len(per_rad_nodules)
            for i, n in enumerate(per_rad_nodules):
                if used[i]: continue
                cluster = [n]; used[i] = True
                for j in range(i + 1, len(per_rad_nodules)):
                    if used[j]: continue
                    m = per_rad_nodules[j]
                    d = np.sqrt(
                        ((n["cz"] - m["cz"]) * slice_thk) ** 2 +
                        ((n["cy"] - m["cy"]) * pixel_sp[0]) ** 2 +
                        ((n["cx"] - m["cx"]) * pixel_sp[1]) ** 2
                    )
                    if d < args.merge_dist:
                        cluster.append(m); used[j] = True

                # Aggregate stats
                rads = sorted(set(c["rad"] for c in cluster))
                mals = [int(c["malignancy"]) for c in cluster if str(c["malignancy"]).isdigit()]
                spics = [int(c["spiculation"]) for c in cluster if str(c["spiculation"]).isdigit()]
                texs = [int(c["texture"]) for c in cluster if str(c["texture"]).isdigit()]
                margs = [int(c["margin"]) for c in cluster if str(c["margin"]).isdigit()]
                subs = [int(c["subtlety"]) for c in cluster if str(c["subtlety"]).isdigit()]
                diams = [c["diam_mm"] for c in cluster]
                merged.append({
                    "rads_marked": rads,
                    "n_radiologists": len(rads),
                    "malignancy_ratings": mals,
                    "malignancy_mean": round(float(np.mean(mals)), 2) if mals else None,
                    "spiculation_mean": round(float(np.mean(spics)), 2) if spics else None,
                    "texture_mean": round(float(np.mean(texs)), 2) if texs else None,
                    "margin_mean": round(float(np.mean(margs)), 2) if margs else None,
                    "subtlety_mean": round(float(np.mean(subs)), 2) if subs else None,
                    "diam_mm_mean": round(float(np.mean(diams)), 2),
                    "diam_mm_max": round(float(np.max(diams)), 2),
                    "centroid_zyx": [n["cz"], n["cy"], n["cx"]],
                    "calcification": n["calcification"],
                    "lung_rads": lung_rads(float(np.mean(diams))),
                    "verdict": malignancy_band(float(np.mean(mals)) if mals else None),
                })

            series_info["nodules"] = merged
            output["series"].append(series_info)

    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    # Print raw annotation counts so user knows what was filtered
    print("=" * 70)
    print(f"GROUND TRUTH — {args.patient_id}")
    print("(Annotations từ 4 radiologist trong dataset LIDC-IDRI gốc)")
    print("=" * 70)
    with h5py.File(h5p, "r") as f:
        for series_key in f.keys():
            g = f[series_key]
            meta_raw = json.loads(g.attrs["nodule_meta"])
            masks_raw = g["mask_per_rad"][:]
            from collections import defaultdict as _dd
            by_raw = _dd(list)
            for e in meta_raw:
                by_raw[(e["rad_channel"], e["nodule_id"])].append(e)
            n_total = len(by_raw)
            n_polygon = 0
            for (r, nid), es in by_raw.items():
                idxs = [int(e["slice_idx"]) for e in es if 0 <= int(e["slice_idx"]) < masks_raw.shape[0]]
                if idxs and masks_raw[idxs, r].sum() >= 5:
                    n_polygon += 1
            n_point = n_total - n_polygon
            print(f"  Tổng annotations (4 rad):  {n_total}")
            print(f"    - Có polygon (≥ 3mm):    {n_polygon}")
            print(f"    - Point markers:         {n_point}  (LIDC dot: nodule < 3mm hoặc non-nodule)")
            if not args.all and n_point > 0:
                print(f"  → đang lọc point markers; dùng --all để xem hết")

    for s in output["series"]:
        print()
        print(f"Series UID: ...{s['series_uid'][-30:]}")
        print(f"Volume:     {s['shape']}, voxel ({s['voxel_spacing_zyx_mm'][0]:.2f}, "
              f"{s['voxel_spacing_zyx_mm'][1]:.3f}, {s['voxel_spacing_zyx_mm'][2]:.3f}) mm")
        print(f"Radiologist tổng: {s['n_radiologists_total']}/4")
        print()
        print(f">>> {len(s['nodules'])} unique 3D nodule(s) <<<")
        print()
        for i, m in enumerate(s["nodules"], 1):
            print(f"┌─ Nodule #{i} " + "─" * 50)
            print(f"│  Đường kính (mean):     {m['diam_mm_mean']:.1f} mm "
                  f"(max {m['diam_mm_max']:.1f}mm)")
            print(f"│  Lung-RADS:              {m['lung_rads']}")
            print(f"│  Phát hiện bởi:          {m['n_radiologists']}/4 radiologist "
                  f"(channels {m['rads_marked']})")
            print(f"│  Malignancy 1-5:         {m['malignancy_ratings']}  "
                  f"→ mean {m['malignancy_mean']} ({m['verdict']})")
            if m["spiculation_mean"] is not None:
                print(f"│  Spiculation 1-5:        mean {m['spiculation_mean']}  "
                      f"(1=none, 5=marked)")
            if m["texture_mean"] is not None:
                print(f"│  Texture 1-5:            mean {m['texture_mean']}  "
                      f"(1=non-solid/GGN, 5=solid)")
            if m["margin_mean"] is not None:
                print(f"│  Margin 1-5:             mean {m['margin_mean']}  "
                      f"(1=poorly defined, 5=sharp)")
            if m["subtlety_mean"] is not None:
                print(f"│  Subtlety 1-5:           mean {m['subtlety_mean']}  "
                      f"(1=khó thấy, 5=rõ)")
            calc = m["calcification"]
            if str(calc).isdigit():
                calc_label = {1: "popcorn", 2: "laminated", 3: "solid",
                              4: "non-central", 5: "central", 6: "absent"}.get(int(calc), "?")
                print(f"│  Calcification:          {calc} ({calc_label})")
            print(f"│  Vị trí (z, y, x):       {m['centroid_zyx']}")
            print("└" + "─" * 60)
            print()

    print()
    print("=" * 70)
    print("So sánh với AI:")
    print(f"  - Upload {args.patient_id} vào webapp Nodura")
    print(f"  - AI nên phát hiện ~{sum(len(s['nodules']) for s in output['series'])} nodule trên")
    print( "  - Đối chiếu diameter, vị trí, risk_combined với GT trên")
    print("=" * 70)


if __name__ == "__main__":
    main()
