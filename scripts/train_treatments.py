#!/usr/bin/env python3
"""
train_treatments.py — RAOD++ control matrix E1-E5 (protocol v1.0-v1.4).

Every condition starts from the SAME E0 checkpoint of the SAME seed and trains for the
SAME +20 epochs with the SAME hyperparameters. Only the training data differs. That is
what makes the comparison a control experiment rather than a demonstration.

    nohup bash scripts/run_treatments.sh > /dev/null 2>&1 &
    tail -f output/logs/treat_latest.log

Resumable: a run whose weights and runs.csv row already exist is skipped, so a power cut
costs at most one run.
"""
import argparse, datetime, json, platform, sys, time
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT = Path.home() / "RAODpp"
OUT, RUNS_DIR = ROOT / "output", ROOT / "model" / "runs"
ENR = ROOT / "dataset" / "enriched"
RUNS_CSV = OUT / "runs.csv"
sys.path.insert(0, str(ROOT / "scripts"))
import figdata

CONFIG = {
    "SEEDS": [42, 123, 2024, 7, 3407, 11, 55, 101, 777, 12345],
    "CONDITIONS": ["E1", "E2", "E3", "E4", "E5"],
    "EPOCHS": 20,                 # equal budget for every condition (protocol v1.0)
    "IMGSZ": 640, "BATCH": 64, "WORKERS": 8, "DEVICE": 0,
    "DATASET": "visdrone",
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def data_yaml_for(cond, seed):
    if cond == "E1":
        return ROOT / "dataset" / "visdrone_raodpp.yaml"      # untouched data
    p = ENR / f"{cond}_E0_{CONFIG['DATASET']}_s{seed}" / "data.yaml"
    assert p.exists(), f"missing enriched set: {p}"
    return p


def already_done(run_id):
    if not (RUNS_DIR / run_id / "weights" / "best.pt").exists():
        return False
    if not RUNS_CSV.exists():
        return False
    return run_id in set(pd.read_csv(RUNS_CSV).run_id.astype(str))


def append_row(row):
    """Append one row, never a duplicate (defect fixed after the E0 run)."""
    df = pd.DataFrame([row])
    if RUNS_CSV.exists():
        old = pd.read_csv(RUNS_CSV)
        old = old[old.run_id.astype(str) != row["run_id"]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RUNS_CSV, index=False)


def val_yaml_names(dy):
    d = yaml.safe_load(open(dy))
    return d["names"], d


def run_one(cond, seed):
    run_id = f"{cond}_{CONFIG['DATASET']}_s{seed}"
    if already_done(run_id):
        log(f"{run_id}: already complete, skipping")
        return

    base = RUNS_DIR / f"E0_{CONFIG['DATASET']}_s{seed}" / "weights" / "best.pt"
    assert base.exists(), f"missing E0 checkpoint {base}"
    dy = data_yaml_for(cond, seed)
    names, dcfg = val_yaml_names(dy)

    run_dir = RUNS_DIR / run_id
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"

    t0 = time.time()
    if last.exists() and not best.exists():
        log(f"{run_id}: resuming interrupted run")
        YOLO(str(last)).train(resume=True)
    elif not best.exists():
        log(f"{run_id}: fine-tuning from E0 seed {seed} on {dy.parent.name}")
        YOLO(str(base)).train(
            data=str(dy), epochs=CONFIG["EPOCHS"], imgsz=CONFIG["IMGSZ"],
            batch=CONFIG["BATCH"], workers=CONFIG["WORKERS"], device=CONFIG["DEVICE"],
            seed=seed, deterministic=True, cache=False, patience=0,
            project=str(RUNS_DIR), name=run_id, exist_ok=True, verbose=True, plots=True)
    train_h = (time.time() - t0) / 3600

    # ---- evaluation: always the frozen val split, never the enriched images ----
    model = YOLO(str(best))
    res = model.val(data=str(ROOT / "dataset" / "visdrone_raodpp.yaml"), split="val",
                    imgsz=CONFIG["IMGSZ"], batch=CONFIG["BATCH"],
                    device=CONFIG["DEVICE"], plots=True, verbose=False)
    met = {"mAP50": float(res.box.map50), "mAP50_95": float(res.box.map),
           "precision": float(res.box.mp), "recall": float(res.box.mr),
           "fitness": float(res.fitness)}

    # ---- figure data ----
    figdata.capture_training_curve(run_dir, run_id, cond, seed, OUT)
    figdata.capture_val_artifacts(res, names, run_id, cond, seed, OUT)
    figdata.capture_model_info(best, run_id, cond, seed, OUT, imgsz=CONFIG["IMGSZ"])
    vy = yaml.safe_load(open(ROOT / "dataset" / "visdrone_raodpp.yaml"))
    val_imgs = Path(vy["path"]) / vy["val"]
    figdata.capture_predictions(best, val_imgs, names, run_id, cond, seed, OUT,
                                imgsz=CONFIG["IMGSZ"], device=CONFIG["DEVICE"])

    # ---- enrichment audit carried into the run record ----
    extra = {}
    if cond != "E1":
        aud = ENR / f"{cond}_E0_{CONFIG['DATASET']}_s{seed}" / "done.json"
        if aud.exists():
            a = json.load(open(aud))
            extra = {"added_instances": a.get("pasted", 0) or a.get("pseudo_added", 0),
                     "paste_attempted": a.get("attempted", 0),
                     "mean_sim": a.get("mean_sim")}

    append_row({"run_id": run_id, "exp": cond, "dataset": CONFIG["DATASET"],
                "model": "yolov8n_ft", "seed": seed, "epochs": CONFIG["EPOCHS"],
                "imgsz": CONFIG["IMGSZ"], "batch": CONFIG["BATCH"], **met, **extra,
                "train_hours": round(train_h, 3),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "host": platform.node(), "torch": torch.__version__,
                "script": "train_treatments.py"})
    log(f"{run_id}: mAP50={met['mAP50']:.4f} mAP50-95={met['mAP50_95']:.4f} "
        f"({train_h:.2f} h)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=CONFIG["CONDITIONS"])
    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["SEEDS"])
    a = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable"
    log(f"GPU: {torch.cuda.get_device_name(0)} | conditions={a.conditions} "
        f"| seeds={len(a.seeds)}")

    for cond in a.conditions:
        for seed in a.seeds:
            try:
                run_one(cond, seed)
            except Exception as e:
                log(f"{cond} s{seed} FAILED: {type(e).__name__}: {e}")
                continue

    if RUNS_CSV.exists():
        df = pd.read_csv(RUNS_CSV)
        s = (df.groupby("exp")[["mAP50", "mAP50_95"]]
             .agg(["mean", "std", "size"]).round(4))
        log("summary so far\n" + s.to_string())
    log("ALL DONE")


if __name__ == "__main__":
    main()
