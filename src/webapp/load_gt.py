"""Ground truth loader from LIDC XML annotations.

Robust strategy:
  1. Read uploaded DICOMs → pick the series with the MOST slices (skip localizer)
  2. Build (SOP UID → slice_index) for THAT series only (sorted by IPP[2])
  3. Scan tcia-lidc-xml/**/*.xml; only keep LidcReadMessage files (skip Idri schema)
  4. Match XMLs by SOP-UID overlap with upload (bulletproof — no substring false positives)
  5. Parse unblindedReadNodule per reader → cluster same-nodule annotations across readers
     by centroid distance < 5mm to avoid 4x duplication
"""
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parent.parent.parent
SOP_INDEX_P = ROOT / "work" / "sop_index.json"
XML_DIR = ROOT / "tcia-lidc-xml"
LIDC_NS = "http://www.nih.gov"


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _read_all_dicoms(dcm_paths: List[Path]) -> List[Dict]:
    out = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            out.append({
                "path": p,
                "series_uid": str(ds.SeriesInstanceUID),
                "sop_uid": str(ds.SOPInstanceUID),
                "z": float(ds.ImagePositionPatient[2]),
            })
        except Exception:
            continue
    return out


def _pick_main_series(dicoms: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
    """Pick the series with the most slices (likely diagnostic CT, not localizer)."""
    if not dicoms:
        return None, []
    counter = Counter(d["series_uid"] for d in dicoms)
    main_series = counter.most_common(1)[0][0]
    filtered = [d for d in dicoms if d["series_uid"] == main_series]
    return main_series, filtered


def _build_sop_to_slice(series_dicoms: List[Dict]) -> Dict[str, int]:
    """Sort by ImagePositionPatient[2] and return {SOP_UID: slice_index}."""
    sorted_d = sorted(series_dicoms, key=lambda d: d["z"])
    return {d["sop_uid"]: i for i, d in enumerate(sorted_d)}


def _xml_is_lidc(xml_path: Path) -> bool:
    """Quick check: only parse LidcReadMessage files (skip IdriReadMessage)."""
    try:
        with open(xml_path, encoding="utf-8", errors="replace") as f:
            head = f.read(2048)
        return "LidcReadMessage" in head and "nih.gov/idri" not in head
    except Exception:
        return False


def _xml_sop_uids(xml_path: Path) -> set:
    """Extract all imageSOP_UID references from XML (cheap regex)."""
    try:
        txt = xml_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return set()
    return set(re.findall(r"<imageSOP_UID[^>]*>([^<]+)</imageSOP_UID>", txt))


def _find_xml_by_sop_overlap(upload_sops: set) -> List[Path]:
    """Find XMLs whose imageSOP_UIDs overlap with upload's SOPs.

    This is bulletproof — substring search on series UID can match wrong files
    when an XML mentions multiple series in cross-references.
    """
    matched = []
    for xml_path in XML_DIR.rglob("*.xml"):
        if not _xml_is_lidc(xml_path):
            continue
        sops = _xml_sop_uids(xml_path)
        if sops & upload_sops:
            matched.append(xml_path)
    return matched


def _parse_xml_nodules(xml_path: Path) -> List[Dict]:
    """Parse LidcReadMessage XML → list of nodules with edges per slice.

    Returns: [{nodule_id, reader_idx, edges: [(sop_uid, x, y)], char: {malignancy,...}}]
    """
    try:
        tree = ET.parse(str(xml_path))
    except Exception as e:
        print(f"[GT] XML parse failed {xml_path.name}: {e}")
        return []
    root = tree.getroot()
    nodules = []
    for ri, session in enumerate(root.findall(f".//{{{LIDC_NS}}}readingSession")):
        for nod in session.findall(f"{{{LIDC_NS}}}unblindedReadNodule"):
            nid_el = nod.find(f"{{{LIDC_NS}}}noduleID")
            nid = nid_el.text if nid_el is not None else "?"
            char = {}
            char_el = nod.find(f"{{{LIDC_NS}}}characteristics")
            if char_el is not None:
                for c in char_el:
                    char[_strip_ns(c.tag)] = (c.text or "").strip()
            edges = []
            for roi in nod.findall(f"{{{LIDC_NS}}}roi"):
                sop_el = roi.find(f"{{{LIDC_NS}}}imageSOP_UID")
                sop = (sop_el.text or "").strip() if sop_el is not None else None
                if not sop:
                    continue
                for em in roi.findall(f"{{{LIDC_NS}}}edgeMap"):
                    x_el = em.find(f"{{{LIDC_NS}}}xCoord")
                    y_el = em.find(f"{{{LIDC_NS}}}yCoord")
                    if x_el is None or y_el is None:
                        continue
                    try:
                        x = int(x_el.text); y = int(y_el.text)
                    except Exception:
                        continue
                    edges.append((sop, x, y))
            if edges:
                nodules.append({"id": nid, "reader_idx": ri, "edges": edges, "char": char})
    return nodules


def _centroid_zyx(rn: Dict, sop_to_slice: Dict[str, int]) -> Optional[Tuple[float, float, float]]:
    pts = []
    for sop, x, y in rn["edges"]:
        si = sop_to_slice.get(sop)
        if si is None:
            continue
        pts.append((si, y, x))
    if not pts:
        return None
    zs = [p[0] for p in pts]; ys = [p[1] for p in pts]; xs = [p[2] for p in pts]
    return (sum(zs) / len(zs), sum(ys) / len(ys), sum(xs) / len(xs))


def _cluster_readers(raw_nodules: List[Dict], sop_to_slice: Dict[str, int],
                     voxel_sp_zyx: tuple, cluster_mm: float = 8.0) -> List[List[Dict]]:
    """Group raw_nodules by centroid proximity (different readers, same nodule)."""
    items = []
    for rn in raw_nodules:
        cen = _centroid_zyx(rn, sop_to_slice)
        if cen is None:
            continue
        items.append((cen, rn))
    if not items:
        return []
    # voxel→mm
    sp = np.array(voxel_sp_zyx)
    clusters: List[List[Tuple[Tuple[float, float, float], Dict]]] = []
    for cen, rn in items:
        cen_mm = np.array(cen) * sp
        placed = False
        for cl in clusters:
            cl_cen_mm = np.mean([np.array(c) * sp for c, _ in cl], axis=0)
            if np.linalg.norm(cen_mm - cl_cen_mm) < cluster_mm:
                cl.append((cen, rn))
                placed = True
                break
        if not placed:
            clusters.append([(cen, rn)])
    return [[rn for _, rn in cl] for cl in clusters]


def load_gt_for_upload(dcm_paths: List[Path], vol_shape_zyx: tuple,
                       voxel_sp_zyx: tuple) -> List[Dict]:
    """Return GT nodules in same format as predict_nodules() output."""
    dicoms = _read_all_dicoms(dcm_paths)
    if not dicoms:
        print("[GT] No readable DICOMs")
        return []

    main_series, series_dicoms = _pick_main_series(dicoms)
    print(f"[GT] Main series: {main_series[-25:]} ({len(series_dicoms)} slices)")

    sop_to_slice = _build_sop_to_slice(series_dicoms)
    upload_sops = set(sop_to_slice.keys())

    xml_files = _find_xml_by_sop_overlap(upload_sops)
    if not xml_files:
        print(f"[GT] No matching XML annotations for this series")
        return []
    print(f"[GT] Matched {len(xml_files)} XML file(s) by SOP-UID overlap: "
          f"{[x.relative_to(XML_DIR).as_posix() for x in xml_files]}")

    raw_nodules = []
    for xp in xml_files:
        raw_nodules.extend(_parse_xml_nodules(xp))
    print(f"[GT] Total reader-annotations: {len(raw_nodules)} (across 4 readers)")

    clusters = _cluster_readers(raw_nodules, sop_to_slice, voxel_sp_zyx)
    print(f"[GT] Clustered into {len(clusters)} unique nodules")

    nodules = []
    for k, cluster in enumerate(clusters, 1):
        # Aggregate all edges from all readers in this cluster
        all_pts = []
        mals = []
        nod_ids = []
        for rn in cluster:
            try:
                m = int(rn["char"].get("malignancy", "0"))
                if m > 0:
                    mals.append(m)
            except Exception:
                pass
            nod_ids.append(rn["id"])
            for sop, x, y in rn["edges"]:
                si = sop_to_slice.get(sop)
                if si is not None:
                    all_pts.append((si, y, x))
        if not all_pts:
            continue
        zs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        xs = [p[2] for p in all_pts]
        zmin, zmax = min(zs), max(zs) + 1
        ymin, ymax = min(ys), max(ys) + 1
        xmin, xmax = min(xs), max(xs) + 1
        cz = sum(zs) / len(zs)
        cy = sum(ys) / len(ys)
        cx = sum(xs) / len(xs)
        dz_mm = (zmax - zmin) * voxel_sp_zyx[0]
        dy_mm = (ymax - ymin) * voxel_sp_zyx[1]
        dx_mm = (xmax - xmin) * voxel_sp_zyx[2]
        diam = max(dx_mm, dy_mm, dz_mm)
        vox = max(1, (zmax - zmin) * (ymax - ymin) * (xmax - xmin))
        # LIDC convention: malignancy>=4=likely malignant, >=3 readers=high-confidence
        mal_avg = sum(mals) / len(mals) if mals else None
        mal_avg_display = mal_avg if mal_avg is not None else 0
        n_readers = len(cluster)
        nodules.append({
            "id": k,
            "voxels": int(vox),
            "core_voxels": int(vox),
            "volume_mm3": float(vox * voxel_sp_zyx[0] * voxel_sp_zyx[1] * voxel_sp_zyx[2]),
            "diameter_mm": float(diam),
            "diameter_full_mm": float(diam),
            "elongation": 1.0,
            "centroid_zyx_voxel": [float(cz), float(cy), float(cx)],
            "bbox_zyx_voxel": [int(zmin), int(ymin), int(xmin), int(zmax), int(ymax), int(xmax)],
            "fpr_prob": 1.0,
            "confidence": 1.0,
            "nodule_type": f"GT ({n_readers} reader{'s' if n_readers > 1 else ''}"
                           f"{f', malignancy={mal_avg_display:.1f}/5' if mal_avg is not None else ''})",
            "upper_lobe": False,
            "original_id": k,
            "_is_gt": True,
            "lidc_nodule_ids": list(set(nod_ids)),
            "n_readers": n_readers,
            "confidence_tier": "high" if n_readers >= 3 else "low",
            "malignancy_score": float(mal_avg) if mal_avg is not None else None,
            "malignant": bool(mal_avg >= 4.0) if mal_avg is not None else None,
        })

    nodules.sort(key=lambda n: -n["diameter_mm"])
    for new_id, n in enumerate(nodules, 1):
        n["id"] = new_id
    return nodules
