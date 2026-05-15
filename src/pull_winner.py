"""Pull all critical winner artifacts from VPS to local. Idempotent — safe to re-run.

Run: python pull_winner.py
"""
import subprocess
import sys
import os
from pathlib import Path

HOST = os.environ.get("LIDC_VPS_HOST", "")
PORT = int(os.environ.get("LIDC_VPS_PORT", "22"))
USER = os.environ.get("LIDC_VPS_USER", "root")
KEY = os.environ.get("LIDC_VPS_KEY", str(Path.home() / ".ssh" / "lidc_remote"))
LOCAL_ROOT = Path(__file__).resolve().parent.parent

# (remote_path, local_path) tuples. local_path is relative to LOCAL_ROOT.
TARGETS = [
    # Reports + configs (small, always pull)
    ("/workspace/work/academic/PROJECT_REPORT_FULL.md", "work/academic/PROJECT_REPORT_FULL.md"),
    ("/workspace/work/academic/TUNING_SUMMARY.md", "work/academic/TUNING_SUMMARY.md"),
    ("/workspace/work/academic/PILOT_RESULTS.md", "work/academic/PILOT_RESULTS.md"),
    ("/workspace/work/academic/RETRAIN_PROGRESS.md", "work/academic/RETRAIN_PROGRESS.md"),
    ("/workspace/work/academic/best_config.json", "work/academic/best_config.json"),
    ("/workspace/work/academic/best_config_stage2_retuned.json", "work/academic/best_config_stage2_retuned.json"),
    # Winner ckpts (big, only after train done)
    ("/workspace/work/runs_exp/stage2_full/best.pt", "work/runs_exp/stage2_full/best.pt"),
    ("/workspace/work/runs_exp/stage2_full/last.pt", "work/runs_exp/stage2_full/last.pt"),
    ("/workspace/work/runs_exp/stage2_full/swa.pt", "work/runs_exp/stage2_full/swa.pt"),
    ("/workspace/work/runs_exp/stage2_full/eval.json", "work/runs_exp/stage2_full/eval.json"),
    # FPR winner
    ("/workspace/work/runs/fpr_v3_stage2winner.pt", "work/runs/fpr_v3_stage2winner.pt"),
    ("/workspace/work/academic/fpr_v3_stage2winner_eval_test.json",
     "work/academic/fpr_v3_stage2winner_eval_test.json"),
    ("/workspace/work/academic/fpr_v3_stage2winner_eval_val.json",
     "work/academic/fpr_v3_stage2winner_eval_val.json"),
    # FPR candidates (could regen but cheap to keep)
    ("/workspace/work/academic/fpr_v3_stage2winner_test.npz",
     "work/academic/fpr_v3_stage2winner_test.npz"),
    ("/workspace/work/academic/fpr_v3_stage2winner_val.npz",
     "work/academic/fpr_v3_stage2winner_val.npz"),
]


def pull(remote, local):
    local_path = LOCAL_ROOT / local
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", "-i", KEY, "-P", str(PORT),
           "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
           f"{USER}@{HOST}:{remote}", str(local_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        size = local_path.stat().st_size if local_path.exists() else 0
        print(f"  OK {remote} -> {local} ({size/1e6:.1f} MB)")
        return True
    else:
        print(f"  MISS {remote}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'fail'}")
        return False


def main():
    n_ok = n_miss = 0
    for r, l in TARGETS:
        if pull(r, l):
            n_ok += 1
        else:
            n_miss += 1
    print(f"\n{n_ok} pulled, {n_miss} missing (some are produced later in pipeline)")


if __name__ == "__main__":
    main()
