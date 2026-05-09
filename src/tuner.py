"""Auto-tuner: grid-search webapp pipeline parameters against LIDC GT.

State persisted in work/tuner_state.json so each run picks up where the last left off.
Each invocation runs ONE benchmark (single param combo across panel) and updates state.

Run:
  python tuner.py                     # do next param combo
  python tuner.py --status            # show progress
  python tuner.py --apply-best        # write best params into webapp config
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from configs import WORK


STATE_PATH = WORK / "tuner_state.json"
LOG_PATH = WORK / "tuner_log.txt"


# Parameter grid — most impactful knobs
PARAM_GRID = [
    # threshold, min_voxels, max_elong, merge_dist, subpleural_dist, ensemble
    # Round 1: tune detection sensitivity
    (0.55, 200, 4.0, 10.0, 2.0, 1),
    (0.60, 200, 4.0, 10.0, 2.0, 1),
    (0.65, 200, 4.0, 10.0, 2.0, 1),  # current
    (0.70, 200, 4.0, 10.0, 2.0, 1),
    (0.75, 200, 4.0, 10.0, 2.0, 1),
    # Round 2: tune size cutoff
    (0.65, 100, 4.0, 10.0, 2.0, 1),
    (0.65, 150, 4.0, 10.0, 2.0, 1),
    (0.65, 300, 4.0, 10.0, 2.0, 1),
    (0.65, 500, 4.0, 10.0, 2.0, 1),
    # Round 3: elongation
    (0.65, 200, 3.0, 10.0, 2.0, 1),
    (0.65, 200, 5.0, 10.0, 2.0, 1),
    (0.65, 200, 99.0, 10.0, 2.0, 1),  # disable
    # Round 4: merge distance
    (0.65, 200, 4.0, 6.0, 2.0, 1),
    (0.65, 200, 4.0, 14.0, 2.0, 1),
    (0.65, 200, 4.0, 0.0, 2.0, 1),  # disable
    # Round 5: subpleural
    (0.65, 200, 4.0, 10.0, 0.0, 1),  # disable
    (0.65, 200, 4.0, 10.0, 1.0, 1),
    (0.65, 200, 4.0, 10.0, 4.0, 1),
    # Round 6: ensemble vs single
    (0.65, 200, 4.0, 10.0, 2.0, 0),  # no swa
    # Round 7: combo of best findings
    (0.60, 150, 4.0, 8.0, 1.5, 1),
    (0.70, 250, 4.0, 12.0, 2.0, 1),
    (0.65, 200, 3.5, 10.0, 1.5, 1),
]


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"results": [], "next_idx": 0}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def log(msg):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def best_so_far(results: list) -> dict:
    """Score = F1_micro × (1 - 0.3 × size_err)."""
    if not results:
        return None
    scored = []
    for r in results:
        f1 = r.get("f1_micro", 0)
        se = r.get("mean_size_err") or 0.5
        score = f1 * (1 - 0.3 * min(se, 1.0))
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return {"score": scored[0][0], **scored[0][1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--apply-best", action="store_true")
    ap.add_argument("--patients", default="")
    args = ap.parse_args()

    state = load_state()

    if args.status:
        print(f"Tried: {len(state['results'])}/{len(PARAM_GRID)} combos")
        b = best_so_far(state["results"])
        if b:
            print(f"\nBest so far (score={b['score']:.4f}):")
            print(json.dumps(b, indent=2))
        return

    if args.apply_best:
        b = best_so_far(state["results"])
        if not b:
            log("No results yet, cannot apply.")
            return
        log(f"Best params (score={b['score']:.4f}): {b['params']}")
        # TODO: patch webapp/app.py with these params
        return

    # Run next combo
    if state["next_idx"] >= len(PARAM_GRID):
        log(f"All {len(PARAM_GRID)} combos done. Use --status to see best.")
        return

    combo = PARAM_GRID[state["next_idx"]]
    threshold, min_voxels, max_elong, merge_dist, subpleural, ensemble = combo
    log(f"=== Combo {state['next_idx']+1}/{len(PARAM_GRID)}: "
        f"thr={threshold} min_vox={min_voxels} elong={max_elong} "
        f"merge={merge_dist} subpl={subpleural} ens={ensemble}")

    out_json = WORK / f"bench_{state['next_idx']:02d}.json"
    cmd = [
        sys.executable, str(Path(__file__).parent / "benchmark.py"),
        "--threshold", str(threshold),
        "--min-voxels", str(min_voxels),
        "--max-elong", str(max_elong),
        "--merge-dist", str(merge_dist),
        "--subpleural-dist", str(subpleural),
        "--ensemble", str(ensemble),
        "--out", str(out_json),
    ]
    if args.patients:
        cmd += ["--patients", args.patients]

    log(f"$ {' '.join(cmd)}")
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0

    if proc.returncode != 0:
        log(f"FAILED ({dt:.0f}s): {proc.stderr[-500:]}")
        # Don't advance index; will retry next iteration
        return

    if not out_json.exists():
        log(f"FAILED: no output file ({dt:.0f}s)")
        return

    result = json.loads(out_json.read_text())
    log(f"  P={result['precision_micro']:.3f}  R={result['recall_micro']:.3f}  "
        f"F1={result['f1_micro']:.3f}  size_err={(result.get('mean_size_err') or 0)*100:.1f}%  "
        f"({dt:.0f}s)")

    state["results"].append(result)
    state["next_idx"] += 1
    save_state(state)

    b = best_so_far(state["results"])
    if b:
        log(f"  Best so far: F1={b['f1_micro']:.3f} score={b['score']:.4f} "
            f"params={b['params']}")


if __name__ == "__main__":
    main()
