"""Upload checkpoints to V100 using paramiko SFTP with mkdir fix.

Usage:
    python src/upload_ckpts.py

Uploads swa.pt, fpr.pt, malignancy.pt to /workspace/work/runs/
Also uploads annotations.csv and candidates.csv to /workspace/datasetLuna16/
"""
import os
import sys
import hashlib
import time
from pathlib import Path
import paramiko

HOST = "159.48.242.1"
PORT = 25011
USER = "root"
PASS = "rri_GgkLpKRZ7soZCut4"

LOCAL_RUNS = Path("work/runs")
REMOTE_RUNS = "/workspace/work/runs"

LOCAL_LUNA = Path("datasetLuna16")
REMOTE_LUNA = "/workspace/datasetLuna16"

FILES_CKPT = [
    ("swa.pt", f"{REMOTE_RUNS}/swa.pt"),
    ("fpr.pt", f"{REMOTE_RUNS}/fpr.pt"),
    ("malignancy.pt", f"{REMOTE_RUNS}/malignancy.pt"),
]

FILES_ANNO = [
    ("annotations.csv", f"{REMOTE_LUNA}/annotations.csv"),
    ("candidates.csv", f"{REMOTE_LUNA}/candidates.csv"),
]


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_path = Path.home() / ".ssh" / "lidc_remote"
    if key_path.exists():
        try:
            c.connect(HOST, port=PORT, username=USER, key_filename=str(key_path),
                      timeout=30, look_for_keys=False, allow_agent=False)
            return c
        except paramiko.AuthenticationException:
            pass
    c.connect(HOST, port=PORT, username=USER, password=PASS,
              timeout=30, look_for_keys=False, allow_agent=False)
    return c


def sftp_mkdir_p(sftp, remote_dir):
    """Create directory tree on remote, ignoring if already exists."""
    parts = remote_dir.strip("/").split("/")
    path = ""
    for part in parts:
        path = path + "/" + part
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)


def upload_file(local_path: str, remote_path: str, label: str):
    local_size = os.path.getsize(local_path)
    print(f"[{label}] Uploading {local_path} ({local_size // 1024 // 1024}MB) -> {remote_path}")
    t0 = time.time()
    c = get_client()
    try:
        sftp = c.open_sftp()
        # Ensure remote dir exists via SFTP mkdir
        remote_dir = "/".join(remote_path.split("/")[:-1])
        sftp_mkdir_p(sftp, remote_dir)

        last_print = [0]
        def progress(transferred, total):
            pct = transferred * 100 // total
            if pct - last_print[0] >= 10:
                elapsed = time.time() - t0
                speed_mb = transferred / elapsed / 1024 / 1024 if elapsed > 0 else 0
                print(f"  {pct}%  {transferred // 1024 // 1024}MB / {total // 1024 // 1024}MB  {speed_mb:.1f}MB/s")
                last_print[0] = pct

        sftp.put(local_path, remote_path, callback=progress)
        sftp.close()
        elapsed = time.time() - t0
        speed = local_size / elapsed / 1024 / 1024
        print(f"[{label}] SFTP upload done in {elapsed:.1f}s  ({speed:.1f}MB/s)")
    except Exception as e:
        print(f"[{label}] SFTP failed: {e}", file=sys.stderr)
        c.close()
        return False
    finally:
        try: c.close()
        except: pass
    return True


def verify_remote_size(remote_path: str, expected_size: int) -> bool:
    c = get_client()
    try:
        _, stdout, _ = c.exec_command(f"wc -c < {remote_path}", timeout=30)
        result = stdout.read().decode().strip()
        remote_size = int(result)
        if remote_size == expected_size:
            print(f"  Size verify OK: {remote_size} bytes")
            return True
        else:
            print(f"  Size MISMATCH: remote={remote_size} expected={expected_size}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"  Verify failed: {e}", file=sys.stderr)
        return False
    finally:
        c.close()


def main():
    # Force stdout utf-8
    if hasattr(sys.stdout, "reconfigure"):
        try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except: pass

    all_files = []
    # Check which ckpts actually need uploading (skip if remote already correct size)
    for fname, remote in FILES_CKPT:
        local = str(LOCAL_RUNS / fname)
        if not os.path.exists(local):
            print(f"SKIP {fname}: not found locally")
            continue
        all_files.append((local, remote, fname))

    for fname, remote in FILES_ANNO:
        local = str(LOCAL_LUNA / fname)
        if not os.path.exists(local):
            print(f"SKIP {fname}: not found locally")
            continue
        all_files.append((local, remote, fname))

    results = {}
    for local, remote, label in all_files:
        ok = upload_file(local, remote, label)
        if ok:
            expected = os.path.getsize(local)
            ok = verify_remote_size(remote, expected)
        results[label] = "OK" if ok else "FAIL"

    print("\n=== Upload Summary ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    failures = [k for k, v in results.items() if v != "OK"]
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    else:
        print("All uploads complete.")


if __name__ == "__main__":
    main()
