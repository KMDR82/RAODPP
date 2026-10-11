#!/usr/bin/env python3
"""
finalize_runs.py — evaluate runs whose training finished but whose result row is missing.

Training and evaluation were sharing one process, and at 1280/1536 the evaluation step
kept failing with CUDA OOM on memory the trainer had already fragmented. Weights survive
those failures, so evaluation is simply done here, later, in a clean process.

    python scripts/finalize_runs.py                    # every unfinished run
    python scripts/finalize_runs.py --runs S1280_visdrone_s42
    python scripts/finalize_runs.py --dry-run

For each run it validates on the frozen val split, captures the figure data, and appends
one row to runs.csv. It never trains and never touches weights.
"""
import argparse, datetime, platform, sys, time
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT = Path.home() / "RAODpp"
OUT, RUNS_DIR = ROOT / "output", ROOT / "model" / "runs"
RUNS_CSV = OUT / "runs.csv"
DATA_YAML = ROOT / "dataset" / "visdrone_raodpp.yaml"
sys.path.insert(0, str(ROOT / "scripts"))
import figdata

# imgsz cannot be guessed from the weights; it is part of the condition's definition
IMGSZ = {"E0": 640, "E1": 640, "E2": 640, "E3": 640, "E4": 640, "E5": 640,
         "F3": 640, "F5": 640, "F5b15": 640, "F5b30": 640,
         "R960": 960, "R1280": 1280, "R1536": 1536,
         "S640": 640, "S960": 960, "S1280": 1280,
         "P640": 640, "P960": 960}
VAL_BATCH = {640: 8, 960: 4, 1280: 2, 1536: 2}
MIN_EPOCHS = 20
STALE_MINUTES = 15   # results.csv untouched for this long => training is over


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def recorded():
    if not RUNS_CSV.exists():
        return set()
    return set(pd.read_csv(RUNS_CSV).run_id.astype(str))


def pending():
    done = recorded()
    out = []
    for d in sorted(RUNS_DIR.glob("*_visdrone_s*")):
        if d.name in done:
            continue
        if not (d / "weights" / "best.pt").exists():
            continue
        res = d / "results.csv"
        if not res.exists():
            continue
        try:
            n = len(pd.read_csv(res))
        except Exception:
            continue
        if n < MIN_EPOCHS:
            log(f"{d.name}: only {n} epochs, skipped")
            continue
        # a run still being trained keeps touching results.csv; evaluating it would
        # record a half-trained checkpoint as if it were final
        age_min = (time.time() - res.stat().st_mtime) / 60
        if age_min < STALE_MINUTES:
            log(f"{d.name}: results.csv modified {age_min:.1f} min ago "
                f"— looks active, skipped")
            continue
        exp = d.name.split("_")[0]
        if exp not in IMGSZ:
            log(f"{d.name}: unknown condition '{exp}', skipped")
            continue
        out.append((d, exp, int(d.name.split("_s")[-1]), n))
    return out


def append_row(row):
    df = pd.DataFrame([row])
    if RUNS_CSV.exists():
        old = pd.read_csv(RUNS_CSV)
        old = old[old.run_id.astype(str) != row["run_id"]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RUNS_CSV, index=False)


def finalize(run_dir, exp, seed, n_epochs):
    rid = run_dir.name
    imgsz = IMGSZ[exp]
    best = run_dir / "weights" / "best.pt"
    names = yaml.safe_load(open(DATA_YAML))["names"]

    log(f"{rid}: evaluating at imgsz={imgsz}, batch={VAL_BATCH[imgsz]}")
    # plots=True makes ultralytics allocate very large intermediates at 1280 with the
    # larger backbone and reproducibly OOMs; the figure pipeline collects its own data
    res = YOLO(str(best)).val(data=str(DATA_YAML), split="val", imgsz=imgsz,
                              batch=1, device=0, workers=0,
                              plots=False, verbose=False)
    met = {"mAP50": float(res.box.map50), "mAP50_95": float(res.box.map),
           "precision": float(res.box.mp), "recall": float(res.box.mr)}

    figdata.capture_training_curve(run_dir, rid, exp, seed, OUT)
    figdata.capture_val_artifacts(res, names, rid, exp, seed, OUT)
    figdata.capture_model_info(best, rid, exp, seed, OUT, imgsz=imgsz)
    vy = yaml.safe_load(open(DATA_YAML))
    figdata.capture_predictions(best, Path(vy["path"]) / vy["val"], names, rid, exp,
                                seed, OUT, imgsz=imgsz, device=0)

    append_row({"run_id": rid, "exp": exp, "dataset": "visdrone",
                "model": "yolov8", "seed": seed, "epochs": n_epochs, "imgsz": imgsz,
                **met,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "host": platform.node(), "torch": torch.__version__,
                "script": "finalize_runs.py"})
    log(f"{rid}: mAP50={met['mAP50']:.4f}  mAP50-95={met['mAP50_95']:.4f}  "
        f"({n_epochs} epochs)")
    log(f"{rid}: cost still needs measure_cost_only.py on an idle GPU")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA unavailable"

    todo = pending()
    if a.runs:
        todo = [t for t in todo if t[0].name in a.runs]
    if not todo:
        log("nothing pending"); return

    log(f"{len(todo)} run(s) pending:")
    for d, exp, seed, n in todo:
        log(f"   {d.name}  ({exp}, seed {seed}, {n} epochs)")
    if a.dry_run:
        return

    for d, exp, seed, n in todo:
        try:
            finalize(d, exp, seed, n)
        except Exception as e:
            log(f"{d.name} FAILED: {type(e).__name__}: {e}")
    log("DONE")


if __name__ == "__main__":
    main()
