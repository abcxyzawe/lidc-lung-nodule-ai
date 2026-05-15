"""Aggregate doctor verdicts from src/webapp/results/*/verdicts.json into a feedback dataset.

Output: work/academic/feedback_dataset.json with one row per nodule that has a verdict.
Each row: case_id, case_name, ts, nodule_id, voxels, diameter_mm, fpr_prob,
          confidence, lung_rads, verdict, reason.

Use this dataset later for FPR retrain (hard-neg / hard-pos mining).

Run:
  python src/aggregate_verdicts.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "src" / "webapp" / "results"
OUT = ROOT / "work" / "academic" / "feedback_dataset.json"


def main():
    rows = []
    for cdir in sorted(RESULTS_DIR.glob("*")):
        if not cdir.is_dir():
            continue
        meta_p = cdir / "meta.json"
        verdicts_p = cdir / "verdicts.json"
        if not (meta_p.exists() and verdicts_p.exists()):
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        verdicts = json.loads(verdicts_p.read_text(encoding="utf-8"))
        nodules_by_id = {n["id"]: n for n in meta.get("nodules", [])}
        for nid_str, vd in verdicts.items():
            nid = int(nid_str)
            n = nodules_by_id.get(nid)
            if n is None:
                continue
            rows.append({
                "case_id": meta["id"],
                "case_name": meta.get("name"),
                "case_ts": meta.get("timestamp"),
                "verdict_ts": vd.get("ts"),
                "nodule_id": nid,
                "centroid_zyx_voxel": n.get("centroid_zyx_voxel"),
                "voxels": n.get("voxels"),
                "diameter_mm": n.get("diameter_mm"),
                "fpr_prob": n.get("fpr_prob"),
                "confidence": n.get("confidence"),
                "lung_rads": (n.get("lung_rads") or {}).get("category"),
                "nodule_type": n.get("nodule_type"),
                "verdict": vd.get("verdict"),
                "reason": vd.get("reason", ""),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    n_total = len(rows)
    n_acc = sum(1 for r in rows if r["verdict"] == "accepted")
    n_rej = sum(1 for r in rows if r["verdict"] == "rejected")
    print(f"Total verdicts: {n_total} ({n_acc} accepted / {n_rej} rejected)")

    if n_rej:
        rej_high_fpr = [r for r in rows
                        if r["verdict"] == "rejected" and (r.get("fpr_prob") or 0) > 0.5]
        print(f"  Hard FPs (rejected by doctor but FPR>0.5): {len(rej_high_fpr)}")
    if n_acc:
        acc_low_fpr = [r for r in rows
                       if r["verdict"] == "accepted" and (r.get("fpr_prob") or 0) < 0.5]
        print(f"  Hard TPs (accepted by doctor but FPR<0.5): {len(acc_low_fpr)}")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
