"""QA smoke test for /api/analyze — 3 modes."""
import json
import pathlib
import sys
import time
import requests

SERIES = pathlib.Path(
    "E:/Phan Tich Ung Thu/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0014"
    "/01-01-2000-NA-NA-38612/3000562.000000-NA-07402"
)
DICOM_FILES = sorted(SERIES.glob("*.dcm"))
API = "http://127.0.0.1:8081"

def post_analyze(mode, files):
    file_handles = []
    try:
        for f in files:
            file_handles.append(("files", (f.name, open(f, "rb"), "application/octet-stream")))
        data = {"model": mode}
        t0 = time.time()
        r = requests.post(f"{API}/api/analyze", files=file_handles, data=data, timeout=600)
        elapsed = time.time() - t0
        return r.json(), elapsed
    finally:
        for _, (_, fh, _) in file_handles:
            fh.close()

def poll_case(case_id, timeout=300):
    results_dir = pathlib.Path("E:/Phan Tich Ung Thu/src/webapp/results") / case_id
    meta_path = results_dir / "meta.json"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        time.sleep(3)
    raise TimeoutError(f"meta.json not found after {timeout}s for {case_id}")

if __name__ == "__main__":
    modes = sys.argv[1:] if sys.argv[1:] else ["mine", "monai", "gt"]
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"MODE: {mode} | files: {len(DICOM_FILES)}")
        resp, t_post = post_analyze(mode, DICOM_FILES)
        print(f"POST response ({t_post:.1f}s): {json.dumps(resp)}")
        case_id = resp.get("case_id")
        if not case_id:
            print(f"FAIL: no case_id in response")
            continue
        print(f"Polling case {case_id}...")
        t_poll0 = time.time()
        try:
            meta = poll_case(case_id)
            t_total = time.time() - t_poll0 + t_post
            print(f"meta.json ready in {t_total:.0f}s total")
            print(f"n_nodules: {meta.get('n_nodules')}")
            nodules = meta.get("nodules", [])
            if nodules:
                n = nodules[0]
                print(f"First nodule keys: {list(n.keys())}")
                print(f"  upper_lobe: {n.get('upper_lobe')}")
                print(f"  nodule_type: {n.get('nodule_type')}")
                print(f"  confidence: {n.get('confidence')}")
                if mode == "gt":
                    print(f"  n_readers: {n.get('n_readers')}")
                    print(f"  confidence_tier: {n.get('confidence_tier')}")
                    print(f"  malignancy_score: {n.get('malignancy_score')}")
                    print(f"  malignant: {n.get('malignant')}")
                if mode == "monai":
                    print(f"  nodule_type: {n.get('nodule_type')}")
                print(f"SNIPPET: {json.dumps(n, indent=2)[:600]}")
            else:
                print("No nodules found")
        except TimeoutError as e:
            print(f"FAIL: {e}")
