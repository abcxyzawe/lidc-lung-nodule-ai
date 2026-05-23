"""Poll V100 eval logs until both jobs complete.

Prints progress every 5 minutes, exits when both have FROC results.
"""
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import paramiko

HOST = "159.48.242.1"
PORT = 25006
USER = "root"
KEY_PATH = Path.home() / ".ssh" / "lidc_remote"
PASS = "rri_SKgNcT52PGmZnnHc"

LOG_BEST = "/workspace/work/runs_luna16/eval_run005_best.log"
LOG_MONAI = "/workspace/work/runs_luna16/eval_monai.log"
DONE_TOKEN = "=== FROC Results"


def get_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if KEY_PATH.exists():
        try:
            c.connect(HOST, port=PORT, username=USER,
                      key_filename=str(KEY_PATH), timeout=20,
                      look_for_keys=False, allow_agent=False)
            return c
        except Exception:
            pass
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20,
              look_for_keys=False, allow_agent=False)
    return c


def run_cmd(cmd, timeout=30):
    c = get_client()
    _, stdout, stderr = c.exec_command(
        f"export PATH=/opt/conda/bin:$PATH; {cmd}", timeout=timeout
    )
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    c.close()
    return out.strip()


def scan_count(log):
    out = run_cmd(f"grep -c 'candidates=' {log} 2>/dev/null || echo 0")
    try:
        return int(out.strip().splitlines()[-1])
    except:
        return 0


def is_done(log):
    out = run_cmd(f"grep -c '{DONE_TOKEN}' {log} 2>/dev/null || echo 0")
    try:
        return int(out.strip().splitlines()[-1]) > 0
    except Exception:
        return False


def get_results(log):
    return run_cmd(f"cat {log}", timeout=60)


t_start = time.time()
poll_interval = 120  # 2 minutes

print(f"[poll] Waiting for both eval jobs to complete... (checking every {poll_interval}s)")

while True:
    best_done = is_done(LOG_BEST)
    monai_done = is_done(LOG_MONAI)
    best_scans = scan_count(LOG_BEST)
    monai_scans = scan_count(LOG_MONAI)
    elapsed = (time.time() - t_start) / 60

    print(f"[{elapsed:.0f}m] run005={best_scans}/88 {'DONE' if best_done else '...'} | MONAI={monai_scans}/88 {'DONE' if monai_done else '...'}")
    sys.stdout.flush()

    if best_done and monai_done:
        print("\n=== BOTH DONE ===")
        print("\n--- run005 best.pt full log ---")
        print(get_results(LOG_BEST))
        print("\n--- MONAI full log ---")
        print(get_results(LOG_MONAI))
        break

    time.sleep(poll_interval)
