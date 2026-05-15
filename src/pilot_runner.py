"""Pilot orchestrator — run EXP00..EXP05 sequentially on remote GPU.

Sequence per experiment:
  1. Train (50 epochs from best.pt) — skipped if epochs=0
  2. Eval: cache probs val+test, threshold sweep, locked test eval, visibility,
     full-CT FP rate verification
  3. Append row to work/academic/experiments_comparison.csv

After all experiments done, dumps:
  work/academic/PILOT_RESULTS.md   — markdown comparison table
  work/academic/PILOT_PICK.json    — recommended best config for Stage 2

Run on remote GPU:
  python pilot_runner.py
  python pilot_runner.py --epochs 30          # shorter pilot
  python pilot_runner.py --resume-eval        # skip training, only run eval on existing ckpts
  python pilot_runner.py --only exp03,exp04   # run subset

Time estimate (V100, ~5 min/epoch): ~50 epochs × 5 = 4-5 hours per experiment.
EXP00 + 5 trains × ~4.5h + eval = ~25-30 hours total for full pilot.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import WORK
from experiments import EXPERIMENTS, PILOT_ORDER

PILOT_RESULTS_MD = WORK / "academic" / "PILOT_RESULTS.md"
PILOT_PICK_JSON = WORK / "academic" / "PILOT_PICK.json"
COMPARISON_CSV = WORK / "academic" / "experiments_comparison.csv"


def run_one_experiment(name, epochs_override=None, skip_train=False):
    from train_experiment import train_one
    from eval_experiment import evaluate_one
    cfg = EXPERIMENTS[name]
    print(f"\n{'='*78}\n>>> {name}\n{'='*78}")
    print(f"  {cfg.note}")
    print()
    if not skip_train and cfg.epochs > 0:
        try:
            train_one(cfg, epochs_override)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{name}] TRAIN FAILED: {e}")
            return None
    try:
        eval_out = evaluate_one(name)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[{name}] EVAL FAILED: {e}")
        return None
    return eval_out


def write_results_md():
    """Read comparison.csv and write PILOT_RESULTS.md table."""
    if not COMPARISON_CSV.exists():
        print("No comparison.csv yet"); return
    rows = list(csv.DictReader(open(COMPARISON_CSV)))
    if not rows: return
    # dedup: keep latest row per experiment
    by_exp = {}
    for r in rows: by_exp[r["experiment"]] = r
    rows_dedup = list(by_exp.values())

    md = ["# Pilot Stage 1 Results\n", "", "## Test panel (locked) — by experiment\n", ""]
    md.append("| Experiment | Sens | Prec | F1 | FP/scan | sm | md | lg | invis% | full-CT extra/scan |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows_dedup:
        md.append(
            f"| {r['experiment']} "
            f"| {float(r.get('test_sens',0)):.3f} "
            f"| {float(r.get('test_prec',0)):.3f} "
            f"| {float(r.get('test_f1',0)):.3f} "
            f"| {float(r.get('test_fp_per_scan',0)):.2f} "
            f"| {float(r.get('sens_small',0)):.2f} "
            f"| {float(r.get('sens_medium',0)):.2f} "
            f"| {float(r.get('sens_large',0)):.2f} "
            f"| {float(r.get('invisible_pct',0)):.1f}% "
            f"| {r.get('fullct_avg_extra_pred','-')} |"
        )
    md.append("")
    md.append("## Best by F1 (excluding stage2)")
    md.append("")
    pilot_rows = [r for r in rows_dedup if r["experiment"].startswith("exp")]
    if pilot_rows:
        pilot_rows.sort(key=lambda r: -float(r.get("test_f1", 0)))
        winner = pilot_rows[0]
        md.append(f"**Winner: `{winner['experiment']}` (F1={float(winner['test_f1']):.3f})**")
        md.append("")
        md.append(f"Recommend Stage 2: copy this config to `experiments.py::stage2_full` "
                  "and train 150-200 epochs with SWA + snapshot ensemble.")
        # save pick
        PILOT_PICK_JSON.write_text(json.dumps({
            "winner_experiment": winner["experiment"],
            "metrics": {k: winner[k] for k in winner if k != "experiment"},
            "recommendation": "Copy config to stage2_full and run long train.",
        }, indent=2))
    PILOT_RESULTS_MD.write_text("\n".join(md))
    print(f"Wrote {PILOT_RESULTS_MD}")
    if pilot_rows:
        print(f"Wrote {PILOT_PICK_JSON}: winner = {pilot_rows[0]['experiment']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override per-experiment epochs (default uses ExperimentConfig)")
    ap.add_argument("--resume-eval", action="store_true",
                    help="Skip training, only run evaluation on existing checkpoints")
    ap.add_argument("--only", default="",
                    help="Comma-separated experiment names to run (default: full PILOT_ORDER)")
    args = ap.parse_args()

    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        for n in names:
            if n not in EXPERIMENTS:
                sys.exit(f"Unknown experiment '{n}'. Available: {list(EXPERIMENTS.keys())}")
    else:
        names = PILOT_ORDER

    print(f"Running pilot: {names}")
    print(f"Epochs override: {args.epochs}")
    print(f"Resume eval only: {args.resume_eval}")
    print()
    t0 = time.time()
    for name in names:
        run_one_experiment(name, epochs_override=args.epochs,
                            skip_train=args.resume_eval)
    print(f"\n=== PILOT COMPLETE ({time.time()-t0:.0f}s) ===\n")
    write_results_md()


if __name__ == "__main__":
    main()
