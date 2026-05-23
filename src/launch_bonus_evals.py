"""Launch bonus evals (last.pt + swa.pt) once best.pt eval is done.

Run after poll shows best.pt DONE.
"""
import sys
from pathlib import Path
import paramiko

HOST = "159.48.242.1"
PORT = 25006
USER = "root"
KEY_PATH = Path.home() / ".ssh" / "lidc_remote"
PASS = "rri_SKgNcT52PGmZnnHc"


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
    prefix = "export PATH=/opt/conda/bin:$PATH; export PYTHONIOENCODING=utf-8; "
    c = get_client()
    _, stdout, stderr = c.exec_command(prefix + cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    c.close()
    return rc, out, err


BASE = "/workspace/work/runs_luna16"
CKPT_DIR = f"{BASE}/run005_scratch_full80"
LUNA_DIR = "/workspace/datasetLuna16"
EVAL_SCRIPT = "/workspace/src/eval_on_luna16_test.py"

jobs = [
    ("last", f"{CKPT_DIR}/last.pt", f"{BASE}/eval_run005_last_subset9.json", f"{BASE}/eval_run005_last.log"),
    ("swa",  f"{CKPT_DIR}/swa.pt",  f"{BASE}/eval_run005_swa_subset9.json",  f"{BASE}/eval_run005_swa.log"),
]

for name, ckpt, out_json, log in jobs:
    cmd = (
        f"nohup python {EVAL_SCRIPT} "
        f"--ckpt {ckpt} "
        f"--luna16-dir {LUNA_DIR} "
        f"--subset 9 "
        f"--model-type finetune "
        f"--output-json {out_json} "
        f"> {log} 2>&1 & echo PID_{name}=$!"
    )
    rc, out, err = run_cmd(cmd, timeout=15)
    print(f"Launched {name}: {out.strip()}")
