"""Step 02b - DICOM → h5 with NODULE + BUFFER + NEGATIVE background slices.

Like 02_preprocess.py BUT also keeps a stride-sampled subset of slices that are
far from any nodule. This fixes the distribution shift problem confirmed by
verify_full_vs_preprocessed.py: model never sees apex/base/heart anatomy.

Output: work/preprocessed_v2/<pid>.h5  (separate from work/preprocessed/ — original
preserved for backwards compatibility with current best.pt).

Each h5 group has:
  images          [N, H, W] int16
  mask_per_rad    [N, 4, H, W] uint8
  sop_uids        [N] string
  z_positions     [N] float32
  slice_classes   [N] uint8     (0 = negative, 1 = buffer, 2 = nodule)
  attrs: pixel_spacing, slice_thickness, image_position_first, image_orientation,
         radiologists, nodule_meta

Negative slice selection:
  - Every Nth slice OUTSIDE the nodule±buffer range, with N = NEG_STRIDE
  - Ensures coverage of apex (z=0), base (z=N-1), mid-lung
  - Default NEG_STRIDE=4 → keep ~25% of background slices

Run:
  python 02b_preprocess_with_negatives.py
  python 02b_preprocess_with_negatives.py --neg-stride 5    # fewer negatives
  python 02b_preprocess_with_negatives.py --buffer 3        # match original
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pydicom
from PIL import Image, ImageDraw

from configs import (
    DICOM_ROOT, XML_ROOT, ROOT, SOP_INDEX_JSON, WORK,
    BUFFER_SLICES, TARGET_HW,
)
from utils import patient_id_from_path

warnings.filterwarnings("ignore", category=UserWarning)

# Output dir is SEPARATE from work/preprocessed/ to preserve current model+benchmark
PRE_DIR_V2 = WORK / "preprocessed_v2"

# Per-slice class encoding
SLICE_NEGATIVE = 0   # background, far from any nodule
SLICE_BUFFER = 1     # within ±BUFFER_SLICES of a nodule but no polygon on this slice
SLICE_NODULE = 2     # this slice has at least one polygon


# ---------- XML helpers (copied from 02_preprocess.py) ----------
def xml_text(root, tag):
    for e in root.iter():
        if e.tag.endswith("}" + tag) or e.tag == tag:
            return (e.text or "").strip()
    return None


def xml_findall(root, tag):
    return [e for e in root.iter() if e.tag.endswith("}" + tag) or e.tag == tag]


def parse_polygon(roi_elem):
    pts = []
    for em in xml_findall(roi_elem, "edgeMap"):
        x = xml_text(em, "xCoord"); y = xml_text(em, "yCoord")
        try: pts.append((int(x), int(y)))
        except (TypeError, ValueError): pass
    return pts


def to_hu(ds, arr):
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return (arr.astype(np.float32) * slope + inter).astype(np.int16)


def safe_pixel_array(ds):
    try:
        return ds.pixel_array
    except Exception:
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        return ds.pixel_array


def select_negative_slice_indices(n_total: int, nodule_buffer_idx: set,
                                  neg_stride: int) -> list:
    """Select stride-sampled background slices guaranteed-not in nodule±buffer.

    Coverage guarantees:
      - First slice (z=0, apex) included if not already in nodule_buffer_idx
      - Last slice (z=N-1, base) included if not already
      - Every neg_stride-th slice in between
    """
    available = [i for i in range(n_total) if i not in nodule_buffer_idx]
    if not available:
        return []
    selected = set()
    # Force apex + base
    if 0 not in nodule_buffer_idx and 0 in available:
        selected.add(0)
    if (n_total - 1) not in nodule_buffer_idx and (n_total - 1) in available:
        selected.add(n_total - 1)
    # Stride sample
    for i, idx in enumerate(available):
        if i % neg_stride == 0:
            selected.add(idx)
    return sorted(selected)


def process_patient(args):
    patient_id, xmls_for_patient, neg_stride, buffer_slices = args
    out_path = PRE_DIR_V2 / f"{patient_id}.h5"
    if out_path.exists():
        return f"{patient_id}: skip (exists)"

    # 1. Group XMLs by series
    xml_by_series = defaultdict(list)
    for xp in xmls_for_patient:
        try:
            root = ET.parse(xp).getroot()
        except Exception:
            continue
        series = xml_text(root, "SeriesInstanceUid") or xml_text(root, "SeriesInstanceUID")
        if not series:
            continue
        xml_by_series[series].append((str(xp), root))

    if not xml_by_series:
        return f"{patient_id}: no XML"

    # 2. Find DICOM folder for this patient
    p_dir = DICOM_ROOT / patient_id
    if not p_dir.exists():
        return f"{patient_id}: no DICOM folder"

    dcm_by_series = defaultdict(list)
    for dcm_path in p_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(
                str(dcm_path), stop_before_pixels=True, force=True,
                specific_tags=["SeriesInstanceUID"],
            )
            dcm_by_series[str(ds.SeriesInstanceUID)].append(str(dcm_path))
        except Exception:
            continue

    h5 = h5py.File(out_path, "w")
    try:
        for series_uid, xml_list in xml_by_series.items():
            if series_uid not in dcm_by_series:
                continue

            # 3. Load + sort all slices in series by z
            slices = []
            for dp in dcm_by_series[series_uid]:
                try:
                    ds = pydicom.dcmread(str(dp), force=True)
                    z = float(ds.ImagePositionPatient[2])
                    slices.append((z, ds, dp))
                except Exception:
                    continue
            if not slices: continue
            slices.sort(key=lambda x: x[0])
            sop_to_idx = {str(s[1].SOPInstanceUID): i for i, s in enumerate(slices)}
            n_total_orig = len(slices)

            # 4. Build per-radiologist polygons + nodule meta + mark nodule slices
            seen_session_keys = set()
            rad_to_ch = {}
            nodule_meta = []
            polygons_per_slice = defaultdict(list)
            nodule_slice_idx = set()
            for xp_str, root in xml_list:
                for session in xml_findall(root, "readingSession"):
                    rad_id = xml_text(session, "servicingRadiologistID") or "anon"
                    if rad_id not in rad_to_ch and len(rad_to_ch) < 4:
                        rad_to_ch[rad_id] = len(rad_to_ch)
                    ch = rad_to_ch.get(rad_id, 3)
                    sess_id = xml_text(session, "annotationVersion") or ""
                    for nod in xml_findall(session, "unblindedReadNodule") + \
                              xml_findall(session, "blindedReadNodule"):
                        nodule_id = xml_text(nod, "noduleID") or ""
                        for roi in xml_findall(nod, "roi"):
                            sop = xml_text(roi, "imageSOP_UID")
                            if not sop or sop not in sop_to_idx: continue
                            idx = sop_to_idx[sop]
                            pts = parse_polygon(roi)
                            key = (rad_id, sess_id, nodule_id, sop, len(pts))
                            if key in seen_session_keys: continue
                            seen_session_keys.add(key)
                            polygons_per_slice[idx].append((ch, pts))
                            nodule_slice_idx.add(idx)
                            nodule_meta.append({
                                "slice_idx_full": idx,
                                "radiologist": rad_id,
                                "rad_channel": ch,
                                "nodule_id": nodule_id,
                                "nodule_type": nod.tag.split("}")[-1],
                                "malignancy": xml_text(nod, "malignancy") or "",
                                "subtlety": xml_text(nod, "subtlety") or "",
                                "calcification": xml_text(nod, "calcification") or "",
                                "sphericity": xml_text(nod, "sphericity") or "",
                                "margin": xml_text(nod, "margin") or "",
                                "lobulation": xml_text(nod, "lobulation") or "",
                                "spiculation": xml_text(nod, "spiculation") or "",
                                "texture": xml_text(nod, "texture") or "",
                                "internal_structure": xml_text(nod, "internalStructure") or "",
                                "n_edge_points": len(pts),
                            })

            if not nodule_slice_idx: continue

            # 5. Build kept-slice list with class labels
            buffer_idx = set()
            for i in nodule_slice_idx:
                for j in range(max(0, i - buffer_slices),
                               min(n_total_orig, i + buffer_slices + 1)):
                    if j not in nodule_slice_idx:
                        buffer_idx.add(j)
            nodule_buffer_idx = nodule_slice_idx | buffer_idx
            negative_idx = set(select_negative_slice_indices(
                n_total_orig, nodule_buffer_idx, neg_stride
            ))

            keep_idx = sorted(nodule_buffer_idx | negative_idx)
            old_to_new = {old: new for new, old in enumerate(keep_idx)}

            # Per-slice class
            slice_classes = np.zeros(len(keep_idx), dtype=np.uint8)
            for old_i in keep_idx:
                new_i = old_to_new[old_i]
                if old_i in nodule_slice_idx:
                    slice_classes[new_i] = SLICE_NODULE
                elif old_i in buffer_idx:
                    slice_classes[new_i] = SLICE_BUFFER
                else:
                    slice_classes[new_i] = SLICE_NEGATIVE

            # 6. Build arrays
            H = W = TARGET_HW
            N = len(keep_idx)
            imgs = np.zeros((N, H, W), dtype=np.int16)
            m_rad = np.zeros((N, 4, H, W), dtype=np.uint8)
            sop_uids = []
            zpos = np.zeros(N, dtype=np.float32)

            first_ds = slices[keep_idx[0]][1]
            pix_sp = list(map(float, getattr(first_ds, "PixelSpacing", [1.0, 1.0])))
            slice_thk = float(getattr(first_ds, "SliceThickness", 1.0) or 1.0)
            ipos0 = list(map(float, getattr(first_ds, "ImagePositionPatient", [0, 0, 0])))
            iorient = list(map(float, getattr(first_ds,
                                              "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])))

            for new_i, old_i in enumerate(keep_idx):
                z, ds, _ = slices[old_i]
                arr = safe_pixel_array(ds)
                if arr.shape != (H, W):
                    pad = np.zeros((H, W), dtype=arr.dtype)
                    pad[:min(arr.shape[0], H), :min(arr.shape[1], W)] = arr[:H, :W]
                    arr = pad
                imgs[new_i] = to_hu(ds, arr)
                sop_uids.append(str(ds.SOPInstanceUID))
                zpos[new_i] = z
                # Render polygons of this slice into mask channels (only if nodule slice)
                if old_i in nodule_slice_idx:
                    for ch, pts in polygons_per_slice.get(old_i, []):
                        if len(pts) >= 3:
                            img = Image.new("L", (W, H), 0)
                            ImageDraw.Draw(img).polygon(pts, outline=1, fill=1)
                            m_rad[new_i, ch] |= np.array(img, dtype=np.uint8)
                        elif pts:
                            for x, y in pts:
                                if 0 <= x < W and 0 <= y < H:
                                    m_rad[new_i, ch, y, x] = 1

            # Re-map slice_idx
            for m in nodule_meta:
                m["slice_idx"] = old_to_new.get(m["slice_idx_full"], -1)

            g = h5.create_group(series_uid.replace(".", "_"))
            g.create_dataset("images", data=imgs, compression="gzip", compression_opts=4)
            g.create_dataset("mask_per_rad", data=m_rad, compression="gzip", compression_opts=4)
            g.create_dataset("sop_uids", data=np.array(sop_uids, dtype="S64"))
            g.create_dataset("z_positions", data=zpos)
            g.create_dataset("slice_classes", data=slice_classes)
            g.attrs["series_uid"] = series_uid
            g.attrs["pixel_spacing"] = json.dumps(pix_sp)
            g.attrs["slice_thickness"] = slice_thk
            g.attrs["image_position_first"] = json.dumps(ipos0)
            g.attrs["image_orientation"] = json.dumps(iorient)
            g.attrs["radiologists"] = json.dumps(rad_to_ch)
            g.attrs["nodule_meta"] = json.dumps(nodule_meta)
            g.attrs["n_total_orig"] = n_total_orig
            g.attrs["n_kept"] = N
            g.attrs["n_nodule"] = int((slice_classes == SLICE_NODULE).sum())
            g.attrs["n_buffer"] = int((slice_classes == SLICE_BUFFER).sum())
            g.attrs["n_negative"] = int((slice_classes == SLICE_NEGATIVE).sum())
            g.attrs["neg_stride"] = neg_stride
            g.attrs["buffer_slices"] = buffer_slices
        h5.close()
        size_mb = out_path.stat().st_size / 1e6
        return f"{patient_id}: ok ({size_mb:.1f} MB)"
    except Exception as e:
        h5.close()
        out_path.unlink(missing_ok=True)
        return f"{patient_id}: ERROR {e}"


def collect_xmls_per_patient(series_to_patient):
    xml_files = (list(DICOM_ROOT.rglob("*.xml")) +
                 (list(XML_ROOT.rglob("*.xml")) if XML_ROOT.exists() else []) +
                 list(ROOT.glob("*.xml")))
    xml_files = list(set(xml_files))
    by_patient = defaultdict(list)
    for xp in xml_files:
        pid = patient_id_from_path(xp)
        if pid == "UNKNOWN":
            try:
                root = ET.parse(xp).getroot()
                series = xml_text(root, "SeriesInstanceUid") or xml_text(root, "SeriesInstanceUID")
                pid = series_to_patient.get(series, "UNKNOWN")
            except Exception:
                continue
        if pid != "UNKNOWN":
            by_patient[pid].append(xp)
    return dict(by_patient)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-stride", type=int, default=4,
                    help="Keep 1 of every N background slices (default 4 = 25%%)")
    ap.add_argument("--buffer", type=int, default=BUFFER_SLICES,
                    help="Buffer slices around each nodule (default 3)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    PRE_DIR_V2.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {PRE_DIR_V2}")
    print(f"Settings:   buffer=±{args.buffer}, negative stride={args.neg_stride} "
          f"(=> ~{100/args.neg_stride:.0f}% of background slices kept)")

    print("Loading SOP index ...")
    idx = json.loads(SOP_INDEX_JSON.read_text())
    series_to_patient = idx["series_to_patient"]

    print("Grouping XMLs by patient ...")
    xmls_per_patient = collect_xmls_per_patient(series_to_patient)
    print(f"  {len(xmls_per_patient)} patients have XMLs")

    items = sorted(xmls_per_patient.items())
    args_list = [(pid, xmls, args.neg_stride, args.buffer) for pid, xmls in items]
    print(f"Processing {len(args_list)} patients with {args.workers} workers")

    done = 0
    errors = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_patient, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            msg = fut.result()
            done += 1
            if "ERROR" in msg:
                errors.append(msg)
            if done % 25 == 0 or done == len(args_list) or "ERROR" in msg:
                print(f"  [{done}/{len(args_list)}] {msg}", flush=True)

    files = list(PRE_DIR_V2.glob("*.h5"))
    total = sum(f.stat().st_size for f in files)

    # Per-class slice counts
    n_nod = n_buf = n_neg = n_total = 0
    for f in files:
        with h5py.File(f) as h:
            for k in h.keys():
                n_nod += int(h[k].attrs.get("n_nodule", 0))
                n_buf += int(h[k].attrs.get("n_buffer", 0))
                n_neg += int(h[k].attrs.get("n_negative", 0))
                n_total += int(h[k].attrs.get("n_kept", 0))

    print()
    print("=" * 70)
    print(f"Done. {len(files)} h5 files, total {total/1e9:.2f} GB, "
          f"{time.time()-t0:.0f}s, {len(errors)} errors")
    print(f"Slice composition: {n_total:,} kept slices")
    print(f"  nodule:   {n_nod:,} ({100*n_nod/max(n_total,1):.1f}%)")
    print(f"  buffer:   {n_buf:,} ({100*n_buf/max(n_total,1):.1f}%)")
    print(f"  negative: {n_neg:,} ({100*n_neg/max(n_total,1):.1f}%)")
    print(f"\nNext: train_experiment.py --exp exp01_negsampler --epochs 50")


if __name__ == "__main__":
    main()
