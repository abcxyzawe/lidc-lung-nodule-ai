"""Step 03 — Patient-level train/val/test split (8 : 1 : 1).

Stratify by average malignancy across all annotated nodules of that patient:
    avg < 2.5  → "benign"
    2.5..3.5   → "indeterminate"
    avg > 3.5  → "malignant"
    no rating  → "unknown"

Patient-level split → no slice from the same patient appears in multiple sets.

Run:  python 03_split.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import h5py

from configs import PRE_DIR, SPLITS_JSON

random.seed(42)


def avg_malignancy(h5_path: Path):
    vals = []
    with h5py.File(h5_path) as f:
        for k in f:
            meta = json.loads(f[k].attrs["nodule_meta"])
            for m in meta:
                try:
                    v = int(m["malignancy"])
                    if 1 <= v <= 5:
                        vals.append(v)
                except (ValueError, TypeError):
                    pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def main():
    files = sorted(PRE_DIR.glob("*.h5"))
    print(f"{len(files)} preprocessed patients")

    buckets = defaultdict(list)
    for fp in files:
        avg = avg_malignancy(fp)
        if avg is None: bucket = "unknown"
        elif avg < 2.5: bucket = "benign"
        elif avg <= 3.5: bucket = "indeterminate"
        else: bucket = "malignant"
        buckets[bucket].append(fp.stem)

    for k, v in buckets.items():
        print(f"  {k}: {len(v)}")

    train, val, test = [], [], []
    for bucket, lst in buckets.items():
        random.shuffle(lst)
        n = len(lst)
        n_test = max(1, n // 10)
        n_val  = max(1, n // 10)
        test  += lst[:n_test]
        val   += lst[n_test:n_test + n_val]
        train += lst[n_test + n_val:]

    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    splits = {"train": train, "val": val, "test": test}
    SPLITS_JSON.write_text(json.dumps(splits, indent=2))
    print(f"\nSplit sizes: train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Wrote {SPLITS_JSON}")
    print(f"\nNext: python 05_train.py")


if __name__ == "__main__":
    main()
