#!/usr/bin/env python3
"""
audit_uavdt_classes.py — count every annotated class across the whole UAVDT mirror.

The mirror declares four classes and the conversion dropped the
fourth, `vehicle`, after a 400-file scan found no object carrying it. Four hundred files
out of roughly forty thousand is a one percent sample. If the class appears anywhere in the
remainder, boxes were silently discarded and the evaluation is missing objects.

This reads every annotation file, so it settles the question rather than sampling it.
No GPU, a few minutes.

    python scripts/audit_uavdt_classes.py
"""
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "RAODpp"
RAW = ROOT / "dataset" / "uavdt_raw"
OUT = ROOT / "output"
KEPT = ["car", "truck", "bus"]      # the three classes the conversion instantiated


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_root():
    for cand in [RAW] + [p for p in RAW.iterdir() if p.is_dir()]:
        if (cand / "train" / "ann").is_dir() and (cand / "test" / "ann").is_dir():
            return cand
    raise SystemExit(f"could not find train/ann and test/ann under {RAW}")


def main():
    base = find_root()
    by_id = {c["id"]: c["title"]
             for c in json.loads((base / "meta.json").read_text())["classes"]}
    log(f"classes declared in meta.json: {sorted(by_id.values())}")

    rows = []
    for split in ["train", "test"]:
        files = sorted((base / split / "ann").glob("*.json"))
        log(f"{split}: scanning {len(files):,} annotation files")
        counts, files_with = Counter(), Counter()
        empty = 0
        for i, f in enumerate(files, 1):
            objs = json.loads(f.read_text()).get("objects", [])
            if not objs:
                empty += 1
            seen = set()
            for o in objs:
                t = by_id.get(o.get("classId"), f"<unknown id {o.get('classId')}>")
                counts[t] += 1
                seen.add(t)
            for t in seen:
                files_with[t] += 1
            if i % 10000 == 0:
                log(f"   {i:,}/{len(files):,}")
        log(f"{split}: {sum(counts.values()):,} objects, {empty:,} files with none")
        for t, n in counts.most_common():
            rows.append({"split": split, "class": t, "objects": n,
                         "files_containing": files_with[t]})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "uavdt_class_audit.csv", index=False)
    print("\n" + df.to_string(index=False))

    total = df.groupby("class").objects.sum()
    dropped = total.drop(labels=[c for c in KEPT if c in total.index], errors="ignore")
    print("\n=== verdict ===")
    if dropped.empty or dropped.sum() == 0:
        print("no object outside the three instantiated classes: nothing was discarded,")
        print("and the decision taken from the 400-file sample is confirmed on the whole set.")
    else:
        print("objects OUTSIDE the three instantiated classes were found and were dropped")
        print("by the conversion:")
        print(dropped.to_string())
        print(f"\nshare of all annotated objects: "
              f"{dropped.sum() / total.sum() * 100:.3f}%")
        print("the conversion must be re-run with these classes included, or their "
              "exclusion justified and reported.")
    print(f"\nwritten: {OUT / 'uavdt_class_audit.csv'}")


if __name__ == "__main__":
    main()
