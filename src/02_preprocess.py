"""Step 02 — DICOM series → one HDF5 per patient.

For every series of every patient:
  - Read every DICOM, sort by ImagePositionPatient z
  - Render every per-radiologist polygon onto a [N, 4, H, W] mask
  - Identify slices that contain a nodule contour
  - Keep ONLY those slices (+ buffer ±BUFFER_SLICES around each)
  - Convert image to int16 Hounsfield Units
  - Capture full geometry: pixel_spacing, slice_thickness, z_positions,
    image_position_first, image_orientation
  - Save to work/preprocessed/<PatientID>.h5

Each series is a group inside the .h5 with datasets:
    images          [N, H, W] int16  (Hounsfield)
    mask_per_rad    [N, 4, H, W] uint8
    sop_uids        [N] string
    z_positions     [N] float32  (mm)

…and attrs:
    series_uid, pixel_spacing (yx mm), slice_thickness, image_position_first,
    image_orientation, radiologists (json: id->channel),
    nodule_meta (json: list of per-roi info)

Resume: if <PatientID>.h5 already exists, the patient is skipped.

Run:  python 02_preprocess.py
"""
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
    DICOM_ROOT, XML_ROOT, ROOT, SOP_INDEX_JSON, PRE_DIR,
    BUFFER_SLICES, TARGET_HW, CLEAN_CSV,
)
from utils import patient_id_from_path

warnings.filterwarnings("ignore", category=UserWarning)


# ---------- XML helpers ----------
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


# ---------- DICOM helpers ----------
def to_hu(ds, arr):
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return (arr.astype(np.float32) * slope + inter).astype(np.int16)


def safe_pixel_array(ds):
    try:    return ds.pixel_array
    except Exception:
        # Some files need an explicit transfer syntax workaround
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        return ds.pixel_array


# ---------- Per-patient worker ----------
def process_patient(args):
    patient_id, xmls_for_patient = args
    out_path = PRE_DIR / f"{patient_id}.h5"
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

    # All .dcm grouped by SeriesInstanceUID
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

            # 4. Build per-radiologist polygons + nodule meta + mark nodule slices
            #    Dedup: an XML may appear in two locations (LIDC-IDRI/ and tcia-lidc-xml/).
            seen_session_keys = set()
            rad_to_ch = {}      # radiologist_id -> 0..3
            nodule_meta = []
            polygons_per_slice = defaultdict(list)  # idx -> [(rad_ch, polygon_pts)]
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
                        # dedup: same (rad, session, nodule)
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

            # 5. Determine slices to keep (nodule + buffer)
            keep = set()
            for i in nodule_slice_idx:
                for j in range(max(0, i - BUFFER_SLICES),
                               min(len(slices), i + BUFFER_SLICES + 1)):
                    keep.add(j)
            keep_idx = sorted(keep)
            old_to_new = {old: new for new, old in enumerate(keep_idx)}

            # 6. Build arrays
            H = W = TARGET_HW
            N = len(keep_idx)
            imgs = np.zeros((N, H, W), dtype=np.int16)
            m_rad = np.zeros((N, 4, H, W), dtype=np.uint8)
            sop_uids = []
            zpos = np.zeros(N, dtype=np.float32)

            # Geometry from first slice
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
                # Render polygons of this slice into mask channels
                for ch, pts in polygons_per_slice.get(old_i, []):
                    if len(pts) >= 3:
                        img = Image.new("L", (W, H), 0)
                        ImageDraw.Draw(img).polygon(pts, outline=1, fill=1)
                        m_rad[new_i, ch] |= np.array(img, dtype=np.uint8)
                    elif pts:
                        for x, y in pts:
                            if 0 <= x < W and 0 <= y < H:
                                m_rad[new_i, ch, y, x] = 1

            # Re-map slice_idx_full → slice_idx_in_h5
            for m in nodule_meta:
                m["slice_idx"] = old_to_new.get(m["slice_idx_full"], -1)

            g = h5.create_group(series_uid.replace(".", "_"))
            g.create_dataset("images", data=imgs, compression="gzip", compression_opts=4)
            g.create_dataset("mask_per_rad", data=m_rad, compression="gzip", compression_opts=4)
            g.create_dataset("sop_uids", data=np.array(sop_uids, dtype="S64"))
            g.create_dataset("z_positions", data=zpos)
            g.attrs["series_uid"] = series_uid
            g.attrs["pixel_spacing"] = json.dumps(pix_sp)
            g.attrs["slice_thickness"] = slice_thk
            g.attrs["image_position_first"] = json.dumps(ipos0)
            g.attrs["image_orientation"] = json.dumps(iorient)
            g.attrs["radiologists"] = json.dumps(rad_to_ch)
            g.attrs["nodule_meta"] = json.dumps(nodule_meta)
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
        # Try patient from path; fallback: read XML's series_uid → look up patient
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
    PRE_DIR.mkdir(exist_ok=True)
    print("Loading SOP index ...")
    idx = json.loads(SOP_INDEX_JSON.read_text())
    series_to_patient = idx["series_to_patient"]

    print("Grouping XMLs by patient ...")
    xmls_per_patient = collect_xmls_per_patient(series_to_patient)
    print(f"  {len(xmls_per_patient)} patients have XMLs")

    items = sorted(xmls_per_patient.items())
    workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Processing {len(items)} patients with {workers} workers")

    done = 0
    errors = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_patient, it): it[0] for it in items}
        for fut in as_completed(futures):
            msg = fut.result()
            done += 1
            if "ERROR" in msg:
                errors.append(msg)
            if done % 25 == 0 or done == len(items) or "ERROR" in msg:
                print(f"  [{done}/{len(items)}] {msg}", flush=True)

    files = list(PRE_DIR.glob("*.h5"))
    total = sum(f.stat().st_size for f in files)
    print()
    print("=" * 70)
    print(f"Done. {len(files)} h5 files, total {total/1e9:.2f} GB, "
          f"{time.time()-t0:.0f}s, {len(errors)} errors")
    if errors:
        print("First errors:")
        for e in errors[:5]: print("  " + e)
    print(f"\nNext: python 03_split.py")


if __name__ == "__main__":
    main()
