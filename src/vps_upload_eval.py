"""Upload eval script + MONAI bundle to V100, then launch eval jobs.

Run once:
  python src/vps_upload_eval.py
"""
import os
import sys
import time
import hashlib
from pathlib import Path
import paramiko

HOST = "159.48.242.1"
PORT = 25006
USER = "root"
KEY_PATH = Path.home() / ".ssh" / "lidc_remote"
PASS = "rri_SKgNcT52PGmZnnHc"

LOCAL_ROOT = Path("E:/Phan Tich Ung Thu")


def get_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if KEY_PATH.exists():
        try:
            c.connect(HOST, port=PORT, username=USER,
                      key_filename=str(KEY_PATH), timeout=20,
                      look_for_keys=False, allow_agent=False)
            return c
        except paramiko.AuthenticationException:
            pass
    c.connect(HOST, port=PORT, username=USER, password=PASS,
              timeout=20, look_for_keys=False, allow_agent=False)
    return c


def run_cmd(cmd, timeout=60):
    c = get_client()
    env = "export PATH=/opt/conda/bin:$PATH; export PYTHONIOENCODING=utf-8; "
    _, stdout, stderr = c.exec_command(env + cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    c.close()
    return rc, out, err


def sftp_put(local_path: Path, remote_path: str):
    """Upload file via paramiko SFTP."""
    c = get_client()
    sftp = c.open_sftp()
    print(f"  uploading {local_path.name} ({local_path.stat().st_size/1024/1024:.1f}MB) -> {remote_path}")
    sftp.put(str(local_path), remote_path)
    sftp.close()
    c.close()
    print(f"  done: {remote_path}")


def sftp_get(remote_path: str, local_path: Path):
    """Download file via paramiko SFTP."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    c = get_client()
    sftp = c.open_sftp()
    try:
        stat = sftp.stat(remote_path)
        print(f"  downloading {Path(remote_path).name} ({stat.st_size/1024/1024:.1f}MB)")
    except Exception:
        print(f"  downloading {remote_path}")
    sftp.get(remote_path, str(local_path))
    sftp.close()
    c.close()
    print(f"  saved: {local_path}")


def md5_local(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_remote(remote_path: str) -> str:
    rc, out, err = run_cmd(f"md5sum {remote_path} | awk '{{print $1}}'")
    return out.strip()


# ===========================================================================
# STEP 1: Upload eval script + predict_monai.py
# ===========================================================================
print("\n=== STEP 1: Upload scripts ===")
rc, out, _ = run_cmd("mkdir -p /workspace/src/webapp /workspace/bundles/lung_nodule_ct_detection/models /workspace/bundles/lung_nodule_ct_detection/configs && echo OK")
print("mkdir:", out.strip())

sftp_put(LOCAL_ROOT / "src/eval_on_luna16_test.py",
         "/workspace/src/eval_on_luna16_test.py")
sftp_put(LOCAL_ROOT / "src/webapp/predict_monai.py",
         "/workspace/src/webapp/predict_monai.py")

# ===========================================================================
# STEP 2: Upload MONAI bundle (model.ts + configs)
# ===========================================================================
print("\n=== STEP 2: Upload MONAI bundle ===")
bundle_local = LOCAL_ROOT / "bundles/lung_nodule_ct_detection"

# model.ts (80MB)
sftp_put(bundle_local / "models/model.ts",
         "/workspace/bundles/lung_nodule_ct_detection/models/model.ts")

# configs directory
configs_local = bundle_local / "configs"
rc, _, _ = run_cmd("mkdir -p /workspace/bundles/lung_nodule_ct_detection/configs")
for f in configs_local.iterdir():
    if f.is_file():
        sftp_put(f, f"/workspace/bundles/lung_nodule_ct_detection/configs/{f.name}")

print("\n=== Upload complete ===")

# ===========================================================================
# STEP 3: Verify script uploaded OK
# ===========================================================================
print("\n=== STEP 3: Verify uploads ===")
rc, out, err = run_cmd("wc -l /workspace/src/eval_on_luna16_test.py && wc -l /workspace/src/webapp/predict_monai.py && ls -lh /workspace/bundles/lung_nodule_ct_detection/models/model.ts")
print(out.strip())

# ===========================================================================
# STEP 4: Check deps on V100
# ===========================================================================
print("\n=== STEP 4: Check deps ===")
rc, out, err = run_cmd(
    "python -c 'import segmentation_models_pytorch, SimpleITK, monai, pandas; print(\"deps OK\")' 2>&1",
    timeout=30
)
print(out.strip() or err.strip())

# ===========================================================================
# STEP 5: Launch run005 best.pt eval (nohup background)
# ===========================================================================
print("\n=== STEP 5: Launch run005 best.pt eval ===")
cmd_best = (
    "nohup python /workspace/src/eval_on_luna16_test.py "
    "--ckpt /workspace/work/runs_luna16/run005_scratch_full80/best.pt "
    "--luna16-dir /workspace/datasetLuna16 "
    "--subset 9 "
    "--model-type finetune "
    "--output-json /workspace/work/runs_luna16/eval_run005_best_subset9.json "
    "> /workspace/work/runs_luna16/eval_run005_best.log 2>&1 & echo PID_BEST=$!"
)
rc, out, err = run_cmd(cmd_best, timeout=30)
print("Launch result:", out.strip(), err.strip())

# ===========================================================================
# STEP 6: Launch MONAI eval (nohup background, sequential to avoid OOM)
# ===========================================================================
print("\n=== STEP 6: Launch MONAI eval ===")
cmd_monai = (
    "nohup python /workspace/src/eval_on_luna16_test.py "
    "--model-type monai "
    "--luna16-dir /workspace/datasetLuna16 "
    "--subset 9 "
    "--output-json /workspace/work/runs_luna16/eval_monai_subset9.json "
    "> /workspace/work/runs_luna16/eval_monai.log 2>&1 & echo PID_MONAI=$!"
)
rc, out, err = run_cmd(cmd_monai, timeout=30)
print("Launch result:", out.strip(), err.strip())

# Wait a moment then check processes started
import time
time.sleep(5)
rc, out, err = run_cmd("ps aux | grep eval_on_luna16 | grep -v grep", timeout=15)
print("\nActive eval processes:")
print(out.strip() or "(none visible yet)")

print("\n=== All jobs launched. Check logs with: ===")
print("  tail /workspace/work/runs_luna16/eval_run005_best.log")
print("  tail /workspace/work/runs_luna16/eval_monai.log")
