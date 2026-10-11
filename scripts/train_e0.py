#!/usr/bin/env python3
"""
train_e0.py — RAOD++ experiment E0 (baseline), crash- and power-cut-resilient.

Run it detached so it survives VS Code, SSH drops and terminal closes:
    cd ~/RAODpp && nohup bash scripts/run_resilient.sh > /dev/null 2>&1 &

Resilience model:
  * per-seed: if best.pt exists          -> skip (already finished)
  * per-seed: if only last.pt exists     -> resume from the interrupted epoch
  * per-run : every metric, curve and prediction dump is written to output/
              immediately after that seed finishes, so a later crash never
              destroys completed work
  * runs.csv: one traceable row per finished seed

Protocol v1.0 values are hard requirements; change them only with a written
protocol amendment.
"""
import argparse, datetime, hashlib, json, platform, sys, time
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata

# ----------------------------------------------------------------------------- config
CONFIG = {
    "PROJECT_ROOT": Path.home() / "RAODpp",
    "EXP_CODE": "E0",
    "DATASET": "visdrone",
    "MODEL": "yolov8n.pt",
    "EPOCHS": 200,
    "PATIENCE": 30,
    "IMGSZ": 640,
    "BATCH": 64,
    "WORKERS": 8,
    "SEEDS": [42, 123, 2024, 7, 3407, 11, 55, 101, 777, 12345],
    "DEVICE": 0,
    "CACHE": False,
}

ROOT = CONFIG["PROJECT_ROOT"]
OUT = ROOT / "output"
RUNS_DIR = ROOT / "model" / "runs"
DATA_YAML = ROOT / "dataset" / "visdrone_raodpp.yaml"
RUNS_CSV = OUT / "runs.csv"
STATE = OUT / "e0_state.json"


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def split_gate():
    """Abort if the dataset drifted since notebook 01 froze the split."""
    frozen = json.load(open(OUT / "visdrone_split_hashes.json"))
    dcfg = yaml.safe_load(open(DATA_YAML))
    base = Path(dcfg["path"])
    for split, rel in [("train", dcfg["train"]), ("val", dcfg["val"])]:
        files = sorted(p.name for p in (base / rel).glob("*.jpg"))
        h = hashlib.sha256("\n".join(files).encode()).hexdigest()
        assert h == frozen[split]["sha256"], f"SPLIT DRIFT in {split}"
    log(f"split gate PASS (train={frozen['train']['n_files']}, val={frozen['val']['n_files']})")
    return dcfg["names"], frozen


def append_row(row):
    df = pd.DataFrame([row])
    if RUNS_CSV.exists():
        df = pd.concat([pd.read_csv(RUNS_CSV), df], ignore_index=True)
    df.to_csv(RUNS_CSV, index=False)


def train_one(seed, class_names, frozen):
    name = f"{CONFIG['EXP_CODE']}_{CONFIG['DATASET']}_s{seed}"
    run_dir = RUNS_DIR / name
    best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"

    if best.exists() and (OUT / "figdata" / f"predictions_{name}.csv").exists():
        log(f"{name}: already complete, skipping")
        return name, run_dir, best

    t0 = time.time()
    if last.exists() and not best.exists():
        log(f"{name}: RESUMING from interrupted run")
        YOLO(str(last)).train(resume=True)
    elif not best.exists():
        log(f"{name}: training from scratch")
        YOLO(CONFIG["MODEL"]).train(
            data=str(DATA_YAML), epochs=CONFIG["EPOCHS"], patience=CONFIG["PATIENCE"],
            imgsz=CONFIG["IMGSZ"], batch=CONFIG["BATCH"], workers=CONFIG["WORKERS"],
            device=CONFIG["DEVICE"], seed=seed, deterministic=True, cache=CONFIG["CACHE"],
            project=str(RUNS_DIR), name=name, exist_ok=True, verbose=True, plots=True)
    log(f"{name}: training phase done in {(time.time()-t0)/3600:.2f} h")
    return name, run_dir, best


def evaluate_and_capture(name, run_dir, best, seed, class_names, frozen):
    """Validate on the frozen val split and store every figure source for this run."""
    model = YOLO(str(best))
    res = model.val(data=str(DATA_YAML), split="val", imgsz=CONFIG["IMGSZ"],
                    batch=CONFIG["BATCH"], device=CONFIG["DEVICE"],
                    plots=True, verbose=False)

    met = {"mAP50": float(res.box.map50), "mAP50_95": float(res.box.map),
           "precision": float(res.box.mp), "recall": float(res.box.mr),
           "fitness": float(res.fitness), "n_val_expected": frozen["val"]["n_files"]}

    n_curve = figdata.capture_training_curve(run_dir, name, CONFIG["EXP_CODE"], seed, OUT)
    n_val = figdata.capture_val_artifacts(res, class_names, name, CONFIG["EXP_CODE"], seed, OUT)
    dcfg = yaml.safe_load(open(DATA_YAML))
    val_imgs = Path(dcfg["path"]) / dcfg["val"]
    n_pred = figdata.capture_predictions(best, val_imgs, class_names, name,
                                         CONFIG["EXP_CODE"], seed, OUT,
                                         imgsz=CONFIG["IMGSZ"], device=CONFIG["DEVICE"])
    log(f"{name}: figdata captured (curve={n_curve}, {n_val}, predictions={n_pred})")

    append_row({"run_id": name, "exp": CONFIG["EXP_CODE"], "dataset": CONFIG["DATASET"],
                "model": CONFIG["MODEL"], "seed": seed, "epochs": CONFIG["EPOCHS"],
                "imgsz": CONFIG["IMGSZ"], "batch": CONFIG["BATCH"], **met,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "host": platform.node(), "torch": torch.__version__,
                "script": "train_e0.py"})
    log(f"{name}: mAP50={met['mAP50']:.4f}  mAP50-95={met['mAP50_95']:.4f}")
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["SEEDS"])
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable"
    log(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    class_names, frozen = split_gate()

    done = json.load(open(STATE))["done"] if STATE.exists() else []
    for seed in args.seeds:
        try:
            name, run_dir, best = train_one(seed, class_names, frozen)
            evaluate_and_capture(name, run_dir, best, seed, class_names, frozen)
            done = sorted(set(done + [seed]))
            json.dump({"done": done, "updated": datetime.datetime.now().isoformat()},
                      open(STATE, "w"), indent=2)
        except Exception as e:                    # one bad seed must not kill the batch
            log(f"seed {seed} FAILED: {type(e).__name__}: {e}")
            continue

    log(f"E0 finished. completed seeds: {done}")


if __name__ == "__main__":
    main()
