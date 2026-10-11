#!/usr/bin/env python3
"""
train_scratch.py — RAOD++ from-scratch evaluation (protocol amendment v1.6).

Why this exists: the E1-E5 fine-tuning matrix showed every treatment *below* E0
(E2 -0.0025, E3 -0.0018, E4 -0.0018, E5 -0.0014 mAP50-95, all p < 0.005), while every
treatment was *above* E1 (+0.0006 to +0.0008). The fine-tuning regime itself costs more
than any data intervention can return, because restarting the LR schedule perturbs an
already-converged model. Data-centric augmentation work (X-Paste, Simple Copy-Paste,
AutoAugment) is evaluated from scratch for exactly this reason.

Conditions, each trained from the pretrained COCO checkpoint for the full schedule:
    F0  original VisDrone data        (== E0; reused, not retrained)
    F3  size-matched random pasting   (artefact control)
    F5  full RAOD++ enrichment

    nohup bash scripts/run_scratch.sh > /dev/null 2>&1 &
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
    "SEEDS": [42, 123, 2024],          # extend to 10 only if the effect is real
    "CONDITIONS": ["F3", "F5"],        # F0 is E0, already trained
    "SOURCE": {"F3": "E3", "F5": "E5"},
    "MODEL": "yolov8n.pt",
    "EPOCHS": 200, "PATIENCE": 30,     # identical to E0 (protocol v1.0)
    "IMGSZ": 640, "BATCH": 64, "WORKERS": 8, "DEVICE": 0,
    "DATASET": "visdrone",
}
VAL_YAML = ROOT / "dataset" / "visdrone_raodpp.yaml"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def already_done(run_id):
    if not (RUNS_DIR / run_id / "weights" / "best.pt").exists():
        return False
    return RUNS_CSV.exists() and run_id in set(pd.read_csv(RUNS_CSV).run_id.astype(str))


def append_row(row):
    df = pd.DataFrame([row])
    if RUNS_CSV.exists():
        old = pd.read_csv(RUNS_CSV)
        old = old[old.run_id.astype(str) != row["run_id"]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RUNS_CSV, index=False)


def run_one(cond, seed):
    run_id = f"{cond}_{CONFIG['DATASET']}_s{seed}"
    if already_done(run_id):
        log(f"{run_id}: already complete, skipping")
        return

    src = CONFIG["SOURCE"][cond]
    dy = ENR / f"{src}_E0_{CONFIG['DATASET']}_s{seed}" / "data.yaml"
    assert dy.exists(), f"missing enriched set {dy}"
    names = yaml.safe_load(open(dy))["names"]

    run_dir = RUNS_DIR / run_id
    last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"

    t0 = time.time()
    if last.exists() and not best.exists():
        log(f"{run_id}: resuming")
        YOLO(str(last)).train(resume=True)
    elif not best.exists():
        log(f"{run_id}: training from scratch on {dy.parent.name}")
        YOLO(CONFIG["MODEL"]).train(
            data=str(dy), epochs=CONFIG["EPOCHS"], patience=CONFIG["PATIENCE"],
            imgsz=CONFIG["IMGSZ"], batch=CONFIG["BATCH"], workers=CONFIG["WORKERS"],
            device=CONFIG["DEVICE"], seed=seed, deterministic=True, cache=False,
            project=str(RUNS_DIR), name=run_id, exist_ok=True, verbose=True, plots=True)
    hours = (time.time() - t0) / 3600

    # evaluation always on the frozen val split, never on enriched images
    res = YOLO(str(best)).val(data=str(VAL_YAML), split="val", imgsz=CONFIG["IMGSZ"],
                              batch=CONFIG["BATCH"], device=CONFIG["DEVICE"],
                              plots=True, verbose=False)
    met = {"mAP50": float(res.box.map50), "mAP50_95": float(res.box.map),
           "precision": float(res.box.mp), "recall": float(res.box.mr),
           "fitness": float(res.fitness)}

    figdata.capture_training_curve(run_dir, run_id, cond, seed, OUT)
    figdata.capture_val_artifacts(res, names, run_id, cond, seed, OUT)
    figdata.capture_model_info(best, run_id, cond, seed, OUT, imgsz=CONFIG["IMGSZ"])
    vy = yaml.safe_load(open(VAL_YAML))
    figdata.capture_predictions(best, Path(vy["path"]) / vy["val"], names, run_id,
                                cond, seed, OUT, imgsz=CONFIG["IMGSZ"],
                                device=CONFIG["DEVICE"])

    extra = {}
    aud = ENR / f"{src}_E0_{CONFIG['DATASET']}_s{seed}" / "done.json"
    if aud.exists():
        a = json.load(open(aud))
        extra = {"added_instances": a.get("pasted", 0), "mean_sim": a.get("mean_sim")}

    append_row({"run_id": run_id, "exp": cond, "dataset": CONFIG["DATASET"],
                "model": "yolov8n_scratch", "seed": seed, "epochs": CONFIG["EPOCHS"],
                "imgsz": CONFIG["IMGSZ"], "batch": CONFIG["BATCH"], **met, **extra,
                "train_hours": round(hours, 3),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "host": platform.node(), "torch": torch.__version__,
                "script": "train_scratch.py"})
    log(f"{run_id}: mAP50={met['mAP50']:.4f} mAP50-95={met['mAP50_95']:.4f} ({hours:.2f} h)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=CONFIG["CONDITIONS"])
    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["SEEDS"])
    ap.add_argument("--source-map", default="",
                    help="extra condition:source pairs, e.g. F5b15:E5b15")
    a = ap.parse_args()
    for pair in filter(None, a.source_map.split(",")):
        k, v = pair.split(":"); CONFIG["SOURCE"][k] = v
    assert torch.cuda.is_available(), "CUDA unavailable"
    log(f"GPU: {torch.cuda.get_device_name(0)} | {a.conditions} | seeds {a.seeds}")

    # interleave conditions so a partial run still yields a paired comparison
    for seed in a.seeds:
        for cond in a.conditions:
            try:
                run_one(cond, seed)
            except Exception as e:
                log(f"{cond} s{seed} FAILED: {type(e).__name__}: {e}")

    if RUNS_CSV.exists():
        d = pd.read_csv(RUNS_CSV)
        d = d[d.exp.isin(["E0"] + a.conditions)]
        log("scratch comparison so far\n" +
            d.groupby("exp")[["mAP50", "mAP50_95"]]
             .agg(["mean", "std", "size"]).round(4).to_string())
    log("ALL DONE")


if __name__ == "__main__":
    main()
