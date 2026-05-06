"""Step 01 - Validate that XML annotations map cleanly to DICOM data.

For every .dcm file: read SOPInstanceUID + SeriesInstanceUID + PatientID.
For every .xml file: parse SeriesInstanceUid + every roi.imageSOP_UID.
Report:
  - DICOM read errors
  - XML parse errors
  - XMLs whose Series is not in DICOM
  - ROIs whose imageSOP_UID is not in DICOM
  - List of patients with any mismatch (will be excluded later)

Caches the SOP index to work/sop_index.json so later steps can re-use it.

Run:  python 01_validate.py
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import pydicom
import xml.etree.ElementTree as ET

from configs import (
    DICOM_ROOT, XML_ROOT, ROOT, SOP_INDEX_JSON, SUMMARY_JSON, WORK,
)
from utils import timer, patient_id_from_path


def _xml_text(root, tag):
    for e in root.iter():
        if e.tag.endswith("}" + tag) or e.tag == tag:
            return (e.text or "").strip()
    return None


def _xml_findall(root, tag):
    return [e for e in root.iter() if e.tag.endswith("}" + tag) or e.tag == tag]


def index_dicom():
    """Walk DICOM root once, return:
        sop_to_path:       SOPInstanceUID  -> .dcm path
        series_to_sops:    SeriesInstanceUID -> set(SOPInstanceUID)
        series_to_patient: SeriesInstanceUID -> PatientID
    """
    sop_to_path = {}
    series_to_sops = defaultdict(set)
    series_to_patient = {}
    errors = 0

    files = list(DICOM_ROOT.rglob("*.dcm"))
    print(f"  found {len(files)} .dcm files")
    t0 = time.time()
    for i, p in enumerate(files):
        try:
            ds = pydicom.dcmread(
                str(p), stop_before_pixels=True, force=True,
                specific_tags=["SOPInstanceUID", "SeriesInstanceUID", "PatientID"],
            )
            sop = str(ds.SOPInstanceUID)
            series = str(ds.SeriesInstanceUID)
            sop_to_path[sop] = str(p)
            series_to_sops[series].add(sop)
            series_to_patient.setdefault(series, str(getattr(ds, "PatientID", "?")))
        except Exception:
            errors += 1
        if (i + 1) % 20000 == 0:
            print(f"  ... {i+1}/{len(files)}  elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"  indexed {len(sop_to_path)} slices, {errors} errors, {time.time()-t0:.0f}s")
    return sop_to_path, dict(series_to_sops), series_to_patient


def collect_xmls():
    files = list(DICOM_ROOT.rglob("*.xml"))
    files += list(XML_ROOT.rglob("*.xml")) if XML_ROOT.exists() else []
    files += list(ROOT.glob("*.xml"))
    files = list(set(files))
    print(f"  found {len(files)} .xml files")
    return files


def main():
    print("=" * 70)
    print("Step 01: validate DICOM <-> XML")
    print("=" * 70)

    with timer("index DICOM"):
        sop_to_path, series_to_sops, series_to_patient = index_dicom()

    print("\nSaving SOP index cache ...")
    SOP_INDEX_JSON.write_text(json.dumps({
        "sop_to_path":       sop_to_path,
        "series_to_sops":    {k: sorted(v) for k, v in series_to_sops.items()},
        "series_to_patient": series_to_patient,
    }))
    print(f"  wrote {SOP_INDEX_JSON}  ({SOP_INDEX_JSON.stat().st_size/1e6:.1f} MB)")

    with timer("scan XMLs"):
        xml_files = collect_xmls()

    xml_ok = xml_no_series = xml_series_missing = xml_parse_err = 0
    roi_total = roi_ok = roi_missing = 0
    bad_patients_per_roi = defaultdict(int)
    xml_bad_per_roi = defaultdict(int)

    for xp in xml_files:
        try:
            root = ET.parse(xp).getroot()
        except Exception:
            xml_parse_err += 1
            continue
        series = _xml_text(root, "SeriesInstanceUid") or _xml_text(root, "SeriesInstanceUID")
        if not series:
            xml_no_series += 1
            continue
        if series not in series_to_sops:
            xml_series_missing += 1
            continue
        xml_ok += 1
        ok_sops = series_to_sops[series]
        patient = series_to_patient.get(series, patient_id_from_path(xp))
        for roi in _xml_findall(root, "roi"):
            sop = _xml_text(roi, "imageSOP_UID")
            if not sop:
                continue
            roi_total += 1
            if sop in ok_sops:
                roi_ok += 1
            else:
                roi_missing += 1
                bad_patients_per_roi[patient] += 1
                xml_bad_per_roi[str(xp)] += 1

    bad_patients = sorted(bad_patients_per_roi.keys())
    summary = {
        "total_dicom_slices":   len(sop_to_path),
        "total_series":         len(series_to_sops),
        "total_patients":       len(set(series_to_patient.values())),
        "xml_files_total":      len(xml_files),
        "xml_parse_errors":     xml_parse_err,
        "xml_no_series_uid":    xml_no_series,
        "xml_series_missing":   xml_series_missing,
        "xml_ok":               xml_ok,
        "roi_total":            roi_total,
        "roi_ok":               roi_ok,
        "roi_missing":          roi_missing,
        "roi_match_rate":       roi_ok / max(roi_total, 1),
        "patients_with_mismatch":  bad_patients,
        "patients_with_mismatch_count": len(bad_patients),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    for k, v in summary.items():
        if isinstance(v, list):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print(f"\nWrote {SUMMARY_JSON}")
    print(f"\nNext: python 02_preprocess.py")


if __name__ == "__main__":
    main()
