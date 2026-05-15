"""Upload DICOM data 130 GB to VPS in patient-folder chunks (resume-able).

For each LIDC-IDRI-XXXX folder under manifest-1600709154662/LIDC-IDRI/:
  1. tar to /tmp (~130 MB per patient)
  2. scp to VPS
  3. untar on VPS into /workspace/manifest-1600709154662/LIDC-IDRI/<pid>/
  4. delete local tar
  5. mark done in state file
If interrupted, re-running skips already-done patients.

Run: python upload_dicom_chunked.py
     python upload_dicom_chunked.py --start 100 --end 200   # subset
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DICOM_LOCAL = ROOT / "manifest-1600709154662" / "LIDC-IDRI"
DICOM_REMOTE = "/workspace/manifest-1600709154662/LIDC-IDRI"
STATE_FILE = ROOT / "work" / "dicom_upload_state.json"
SSH_KEY = Path(os.environ.get("LIDC_VPS_KEY", str(Path.home() / ".ssh" / "lidc_remote")))
HOST = os.environ.get("LIDC_VPS_HOST", "")
PORT = int(os.environ.get("LIDC_VPS_PORT", "22"))
USER = os.environ.get("LIDC_VPS_USER", "root")


def load_state():
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()).get("done", []))
    return set()


def save_state(done):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"done": sorted(done)}, indent=0))


def ssh_run(cmd):
    full = ["ssh", "-i", str(SSH_KEY), "-p", str(PORT),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            f"{USER}@{HOST}", cmd]
    return subprocess.run(full, capture_output=True, text=True)


def scp_put(local, remote):
    full = ["scp", "-i", str(SSH_KEY), "-P", str(PORT),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-q", local, f"{USER}@{HOST}:{remote}"]
    return subprocess.run(full, capture_output=True, text=True)


def upload_one(pid):
    local_dir = DICOM_LOCAL / pid
    if not local_dir.exists():
        return False, "no local"
    tmp_tar = Path(tempfile.gettempdir()) / f"{pid}.tar"
    try:
        r = subprocess.run(
            ["tar", "-cf", str(tmp_tar), "-C", str(DICOM_LOCAL), pid],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return False, f"tar failed: {r.stderr}"
        size_mb = tmp_tar.stat().st_size / 1e6
        r = scp_put(str(tmp_tar), f"/tmp/{pid}.tar")
        if r.returncode != 0:
            return False, f"scp failed: {r.stderr}"
        r = ssh_run(
            f"mkdir -p {DICOM_REMOTE} && "
            f"tar -xf /tmp/{pid}.tar -C {DICOM_REMOTE} && "
            f"rm /tmp/{pid}.tar"
        )
        if r.returncode != 0:
            return False, f"untar failed: {r.stderr}"
        return True, f"{size_mb:.0f} MB"
    finally:
        tmp_tar.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    if not DICOM_LOCAL.exists():
        sys.exit(f"DICOM dir not found: {DICOM_LOCAL}")

    pids = sorted(p.name for p in DICOM_LOCAL.iterdir() if p.is_dir())
    print(f"Total {len(pids)} patient folders")

    done = load_state()
    todo = [p for p in pids if p not in done]
    print(f"Already done: {len(done)}, todo: {len(todo)}")

    if args.end:
        todo = todo[args.start:args.end]
    else:
        todo = todo[args.start:]
    print(f"Will upload {len(todo)} patients in this run")

    t0 = time.time()
    n_ok = 0; n_fail = 0
    fails = []
    for i, pid in enumerate(todo):
        ts = time.time()
        ok, msg = upload_one(pid)
        dt = time.time() - ts
        if ok:
            done.add(pid); n_ok += 1
            elapsed = time.time() - t0
            avg_per = elapsed / max(n_ok, 1)
            eta_min = avg_per * (len(todo) - i - 1) / 60
            print(f"  [{i+1}/{len(todo)}] {pid} OK {msg} ({dt:.1f}s, ETA {eta_min:.0f} min)",
                  flush=True)
            if (i + 1) % 10 == 0:
                save_state(done)
        else:
            n_fail += 1; fails.append((pid, msg))
            print(f"  [{i+1}/{len(todo)}] {pid} FAIL: {msg}", flush=True)
    save_state(done)
    print(f"\nDone. {n_ok} ok, {n_fail} fail. Total {(time.time()-t0)/60:.1f} min")
    if fails:
        print(f"Failed: {[f[0] for f in fails[:5]]}{'...' if len(fails)>5 else ''}")


if __name__ == "__main__":
    main()
