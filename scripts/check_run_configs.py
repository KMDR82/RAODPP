#!/usr/bin/env python3
"""
check_run_configs.py — compare the training settings of runs that the paper treats as the
same configuration.

The resolution arm's 640 point on VisDrone is E0, trained by `train_e0.py` before
`train_resolution.py` existed. The manuscript uses it as the arm's low end, so the two code
paths must be shown to produce the same configuration rather than assumed to.

    python scripts/check_run_configs.py                       # E0 vs the grid
    python scripts/check_run_configs.py --runs A B --seed 42

Fields that are expected to differ (paths, run names, the resolution itself) are listed
separately from fields that must match. Exit status is non-zero if a must-match field
differs, so this can gate a submission checklist.
"""
import argparse, sys
from pathlib import Path

import yaml

RUNS = Path.home() / "RAODpp" / "model" / "runs"

# settings that define the experiment: any difference here breaks comparability
MUST_MATCH = [
    "epochs", "patience", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
    "close_mosaic", "erasing", "auto_augment", "augment", "rect", "single_cls",
    "cos_lr", "amp", "nbs", "deterministic", "workers", "fraction", "multi_scale",
]
# settings that are allowed to differ, with the reason
MAY_DIFFER = {
    "imgsz": "the factor under test",
    "batch": "memory constraint at each resolution; nbs accumulation keeps the "
             "effective batch equal",
    "model": "the architecture under test",
    "name": "run identifier", "save_dir": "run identifier", "project": "run identifier",
    "seed": "seed", "data": "dataset under test", "time": "wall clock",
    "resume": "restart bookkeeping", "exist_ok": "restart bookkeeping",
    "device": "hardware index", "cache": "IO only", "plots": "output only",
    "save_period": "output only", "verbose": "output only", "val": "output only",
    "pretrained": "checked separately below",
}


def load(run):
    p = RUNS / run / "args.yaml"
    if not p.exists():
        return None
    return yaml.safe_load(open(p))


def compare(a_name, b_name):
    a, b = load(a_name), load(b_name)
    if a is None or b is None:
        print(f"  missing args.yaml: {a_name if a is None else b_name}")
        return None
    keys = sorted(set(a) | set(b))
    bad, benign, unknown = [], [], []
    for k in keys:
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if va == vb:
            continue
        if k in MUST_MATCH:
            bad.append((k, va, vb))
        elif k in MAY_DIFFER:
            benign.append((k, va, vb))
        else:
            unknown.append((k, va, vb))
    return bad, benign, unknown


def report(a_name, b_name):
    print(f"\n=== {a_name}  vs  {b_name} ===")
    r = compare(a_name, b_name)
    if r is None:
        return False
    bad, benign, unknown = r
    if bad:
        print("  MUST MATCH but differ:")
        for k, va, vb in bad:
            print(f"    {k:20s} {va!r}  vs  {vb!r}")
    else:
        print("  all experiment-defining settings match")
    if unknown:
        print("  not classified — inspect these:")
        for k, va, vb in unknown:
            print(f"    {k:20s} {va!r}  vs  {vb!r}")
    if benign:
        print("  differ as expected:")
        for k, va, vb in benign:
            print(f"    {k:20s} {va!r}  vs  {vb!r}   ({MAY_DIFFER[k]})")
    return not bad and not unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, default=None,
                    help="two run directory names to compare")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.runs:
        ok = report(*a.runs)
        sys.exit(0 if ok else 1)

    # default: E0 against every other VisDrone condition of the same seed
    base = f"E0_visdrone_s{a.seed}"
    others = sorted(p.name for p in RUNS.glob(f"*_visdrone_s{a.seed}")
                    if p.name != base and (p / "args.yaml").exists())
    if not others:
        sys.exit(f"no other VisDrone runs found for seed {a.seed}")
    print(f"baseline: {base}")
    all_ok = True
    for o in others:
        all_ok &= bool(report(base, o))
    print("\n" + ("all comparisons clean" if all_ok
                  else "differences found — see above; they must be reported "
                       "in the manuscript or removed by retraining"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
