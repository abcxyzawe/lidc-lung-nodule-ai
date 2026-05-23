"""Pull run003_scratch_full artifacts from VPS to local work/runs_luna16/run003_scratch_full/.

Syncs:
  best.pt, last.pt, swa.pt (124MB each)
  metrics.csv, config.json, manifest.json
  confusion_matrix_ep001-011.json (11 files)
  froc_curve_ep001-011.png (11 files)
  train.log

Verifies md5 after each file. Skips if local exists + md5 matches.

Usage:
  python src/pull_run003.py [--force]
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vps_helper as vps

REMOTE_BASE = "/workspace/work/runs_luna16/run003_scratch_full"
LOCAL_BASE = Path(__file__).resolve().parent.parent / "work" / "runs_luna16" / "run003_scratch_full"

SMALL_FILES = (
    ["config.json", "manifest.json", "metrics.csv", "train.log"]
    + [f"confusion_matrix_ep{i:03d}.json" for i in range(1, 12)]
    + [f"froc_curve_ep{i:03d}.png" for i in range(1, 12)]
)
LARGE_FILES = ["best.pt", "last.pt", "swa.pt"]


def md5_local(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_remote(remote_path: str) -> str:
    rc, out, err = vps.run(f"md5sum {remote_path}", timeout=120)
    if rc != 0:
        raise RuntimeError(f"md5sum failed on {remote_path}: {err}")
    return out.strip().split()[0]


def pull_file(name: str, force: bool = False) -> dict:
    local = LOCAL_BASE / name
    remote = f"{REMOTE_BASE}/{name}"
    LOCAL_BASE.mkdir(parents=True, exist_ok=True)

    # Check if already synced
    if local.exists() and not force:
        local_md5 = md5_local(local)
        remote_md5 = md5_remote(remote)
        if local_md5 == remote_md5:
            size = local.stat().st_size
            print(f"  SKIP {name} ({size/1024/1024:.1f}MB) md5={local_md5[:8]}.. already synced")
            return {"file": name, "status": "skipped", "md5": local_md5, "size": size}

    t0 = time.time()
    print(f"  PULL {name} ...", flush=True)
    vps.get(remote, str(local))
    elapsed = time.time() - t0

    local_md5 = md5_local(local)
    remote_md5 = md5_remote(remote)
    size = local.stat().st_size

    if local_md5 != remote_md5:
        raise RuntimeError(
            f"MD5 MISMATCH {name}: local={local_md5} remote={remote_md5}"
        )
    rate = size / max(elapsed, 0.1) / 1024 / 1024
    print(f"  OK {name} ({size/1024/1024:.1f}MB, {elapsed:.1f}s, {rate:.1f}MB/s) md5={local_md5[:8]}..")
    return {"file": name, "status": "pulled", "md5": local_md5, "size": size, "elapsed": elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-pull even if md5 matches")
    args = ap.parse_args()

    LOCAL_BASE.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"\n=== Pulling small files ({len(SMALL_FILES)} files) ===")
    for name in SMALL_FILES:
        try:
            r = pull_file(name, force=args.force)
            results.append(r)
        except Exception as e:
            print(f"  ERROR {name}: {e}", file=sys.stderr)
            results.append({"file": name, "status": "error", "error": str(e)})

    print(f"\n=== Pulling large checkpoints ({len(LARGE_FILES)} files) ===")
    for name in LARGE_FILES:
        try:
            r = pull_file(name, force=args.force)
            results.append(r)
        except Exception as e:
            print(f"  ERROR {name}: {e}", file=sys.stderr)
            results.append({"file": name, "status": "error", "error": str(e)})

    print("\n=== Pull summary ===")
    ok = [r for r in results if r["status"] in ("pulled", "skipped")]
    err = [r for r in results if r["status"] == "error"]
    total_bytes = sum(r.get("size", 0) for r in ok)
    print(f"  Files OK: {len(ok)}/{len(results)}  ({total_bytes/1024/1024:.1f}MB total)")
    if err:
        print(f"  ERRORS ({len(err)}):")
        for e in err:
            print(f"    {e['file']}: {e.get('error')}")
        sys.exit(1)
    else:
        print("  All files verified OK.")

    # Print md5 manifest
    print("\n=== MD5 Manifest ===")
    for r in results:
        if r.get("md5"):
            print(f"  {r['md5']}  {r['file']}")


if __name__ == "__main__":
    main()
