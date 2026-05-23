"""Pull run001 artifacts from V100 to local work/runs_luna16/run001/.

Idempotent: skips files that already exist with matching MD5.
Verifies byte-exact MD5 after download.

Run:
  python src/pull_run001.py

Env vars (same as vps_helper.py):
  LIDC_VPS_HOST  LIDC_VPS_PORT  LIDC_VPS_USER  LIDC_VPS_PASS
"""
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vps_helper import run as vps_run, get as vps_get

REMOTE_ROOT = "/workspace/work/runs_luna16/run001"
LOCAL_ROOT = Path(__file__).resolve().parent.parent / "work" / "runs_luna16" / "run001"

# All expected files in run001
FILES = [
    "best.pt",
    "swa.pt",
    "last.pt",
    "metrics.csv",
    "config.json",
    "manifest.json",
    "train.log",
] + [f"confusion_matrix_ep{i:03d}.json" for i in range(1, 10)] \
  + [f"froc_curve_ep{i:03d}.png" for i in range(1, 10)]


def md5_local(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_remote(remote_path: str) -> str:
    rc, out, err = vps_run(f"md5sum {remote_path}", timeout=60)
    if rc != 0:
        raise RuntimeError(f"md5sum failed for {remote_path}: {err.strip()}")
    return out.strip().split()[0]


def pull_file(fname: str) -> dict:
    remote = f"{REMOTE_ROOT}/{fname}"
    local = LOCAL_ROOT / fname

    # Get remote MD5 first
    try:
        remote_md5 = md5_remote(remote)
    except Exception as e:
        return {"file": fname, "status": "REMOTE_MISS", "error": str(e)}

    # If local exists and MD5 matches, skip
    if local.exists():
        local_md5 = md5_local(local)
        if local_md5 == remote_md5:
            size = local.stat().st_size
            print(f"  SKIP {fname} (already verified, {size/1e6:.1f} MB)")
            return {"file": fname, "status": "SKIP", "md5": remote_md5,
                    "size_bytes": size}

    # Pull via vps_get (SFTP with base64 fallback)
    t0 = time.time()
    try:
        vps_get(remote, str(local))
    except Exception as e:
        return {"file": fname, "status": "PULL_FAIL", "error": str(e)}

    elapsed = time.time() - t0

    # Verify
    if not local.exists():
        return {"file": fname, "status": "VERIFY_FAIL",
                "error": "local file missing after pull"}
    local_md5 = md5_local(local)
    size = local.stat().st_size
    if local_md5 != remote_md5:
        local.unlink()  # delete corrupted file
        return {"file": fname, "status": "MD5_MISMATCH",
                "remote_md5": remote_md5, "local_md5": local_md5}

    speed_mb = size / 1e6 / max(elapsed, 0.001)
    print(f"  OK  {fname}  {size/1e6:.1f} MB  {elapsed:.0f}s  ({speed_mb:.1f} MB/s)  "
          f"md5={remote_md5[:12]}...")
    return {"file": fname, "status": "OK", "md5": remote_md5,
            "size_bytes": size, "elapsed_s": round(elapsed, 1)}


def main():
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Pulling run001 -> {LOCAL_ROOT}")
    print(f"Remote: {REMOTE_ROOT}\n")

    results = []
    for fname in FILES:
        r = pull_file(fname)
        results.append(r)

    print("\n=== Summary ===")
    total_bytes = 0
    ok = skip = fail = 0
    for r in results:
        st = r["status"]
        if st == "OK":
            ok += 1
            total_bytes += r.get("size_bytes", 0)
        elif st == "SKIP":
            skip += 1
            total_bytes += r.get("size_bytes", 0)
        else:
            fail += 1
            print(f"  FAIL {r['file']}: {st} {r.get('error', r.get('remote_md5', ''))}")
    print(f"  {ok} pulled  {skip} skipped (already OK)  {fail} failed")
    print(f"  Total local size: {total_bytes/1e6:.1f} MB")
    if fail > 0:
        sys.exit(1)
    print("\nAll artifacts verified byte-exact.")


if __name__ == "__main__":
    main()
