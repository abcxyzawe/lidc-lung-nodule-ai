"""Pull run005_scratch_full80 artifacts from V100 to local with md5 verify.

Files pulled:
  - best.pt, last.pt, swa.pt  (3x 124MB)
  - metrics.csv, train.log, config.json, manifest.json
  - 80x confusion_matrix_ep*.json
  - 80x froc_curve_ep*.png

Run: python src/pull_run005.py
"""
import hashlib
import sys
import time
from pathlib import Path

import paramiko

HOST = "159.48.242.1"
PORT = 25006
USER = "root"
KEY_PATH = Path.home() / ".ssh" / "lidc_remote"
PASS = "rri_SKgNcT52PGmZnnHc"

REMOTE_DIR = "/workspace/work/runs_luna16/run005_scratch_full80"
LOCAL_DIR = Path("E:/Phan Tich Ung Thu/work/runs_luna16/run005_scratch_full80")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)


def get_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if KEY_PATH.exists():
        try:
            c.connect(HOST, port=PORT, username=USER,
                      key_filename=str(KEY_PATH), timeout=30,
                      look_for_keys=False, allow_agent=False)
            return c
        except paramiko.AuthenticationException:
            pass
    c.connect(HOST, port=PORT, username=USER, password=PASS,
              timeout=30, look_for_keys=False, allow_agent=False)
    return c


def run_cmd(cmd, timeout=60):
    c = get_client()
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    c.close()
    return rc, out, err


def md5_local(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sftp_get(sftp, remote_path: str, local_path: Path) -> bool:
    """Download via SFTP, return True on success."""
    try:
        stat = sftp.stat(remote_path)
        size_mb = stat.st_size / 1024 / 1024
    except Exception:
        size_mb = 0
    t0 = time.time()
    sftp.get(remote_path, str(local_path))
    elapsed = time.time() - t0
    actual_size = local_path.stat().st_size
    print(f"  {local_path.name:50s} {size_mb:7.1f}MB  {elapsed:.1f}s", flush=True)
    return actual_size > 0


# Get list of all files in remote dir
rc, out, _ = run_cmd(f"ls {REMOTE_DIR}")
remote_files = [f.strip() for f in out.strip().splitlines() if f.strip()]
print(f"[pull_run005] {len(remote_files)} remote files found")

# Get remote md5 for large files in one batch
large_files = [f for f in remote_files if f.endswith(".pt")]
print(f"[pull_run005] Computing remote md5 for {len(large_files)} .pt files...")
rc, out, _ = run_cmd(
    f"for f in {REMOTE_DIR}/*.pt; do md5sum $f; done",
    timeout=120
)
remote_md5 = {}
for line in out.strip().splitlines():
    parts = line.strip().split()
    if len(parts) == 2:
        remote_md5[Path(parts[1]).name] = parts[0]
print(f"[pull_run005] Remote md5: {remote_md5}")

# Open persistent SFTP connection for batch download
c = get_client()
sftp = c.open_sftp()

ok_count = 0
fail_count = 0
verified_count = 0

for fname in sorted(remote_files):
    remote_path = f"{REMOTE_DIR}/{fname}"
    local_path = LOCAL_DIR / fname

    # Skip if already exists with same size
    if local_path.exists() and local_path.stat().st_size > 0:
        # For .pt files, verify md5
        if fname.endswith(".pt") and fname in remote_md5:
            local_md5 = md5_local(local_path)
            if local_md5 == remote_md5[fname]:
                print(f"  {fname:50s} SKIP (md5 match)", flush=True)
                ok_count += 1
                verified_count += 1
                continue
            else:
                print(f"  {fname:50s} RE-DOWNLOAD (md5 mismatch)", flush=True)
        else:
            print(f"  {fname:50s} SKIP (exists, {local_path.stat().st_size} bytes)", flush=True)
            ok_count += 1
            continue

    try:
        success = sftp_get(sftp, remote_path, local_path)
        if success:
            ok_count += 1
        else:
            fail_count += 1
            print(f"  {fname}: EMPTY after download", flush=True)
    except Exception as e:
        fail_count += 1
        print(f"  {fname}: FAILED - {e}", flush=True)

sftp.close()
c.close()

# Verify .pt md5 checksums
print(f"\n[pull_run005] Verifying .pt checksums...")
for fname, expected_md5 in remote_md5.items():
    local_path = LOCAL_DIR / fname
    if local_path.exists():
        actual_md5 = md5_local(local_path)
        status = "OK" if actual_md5 == expected_md5 else "MISMATCH"
        if actual_md5 == expected_md5:
            verified_count += 1
        print(f"  {fname}: {status} (remote={expected_md5[:8]}... local={actual_md5[:8]}...)")
    else:
        print(f"  {fname}: MISSING")

print(f"\n[pull_run005] SUMMARY:")
print(f"  Total remote: {len(remote_files)}")
print(f"  OK: {ok_count}")
print(f"  Failed: {fail_count}")
print(f"  Verified (md5): {verified_count}")
print(f"  Local dir: {LOCAL_DIR}")
