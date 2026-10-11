#!/usr/bin/env python3
"""
train_resolution.py — RAOD++ protocol v2.0, phases A and B.

Trains the resolution ladder and the architecture arms on identical data, splits and
schedules, then measures accuracy AND cost so the comparison can be made at equal latency.

    nohup bash scripts/run_resolution.sh > /dev/null 2>&1 &
    tail -f ~/RAODpp/output/logs/res_latest.log

Resumable: a condition/seed whose weights and runs.csv row exist is skipped.
"""
import argparse, datetime, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT = Path.home() / "RAODpp"
OUT, RUNS_DIR = ROOT / "output", ROOT / "model" / "runs"
RUNS_CSV, FIG = OUT / "runs.csv", OUT / "figdata"
DATASETS = {"visdrone": "visdrone_raodpp.yaml", "uavdt": "uavdt_raodpp.yaml"}
DATA_YAML = ROOT / "dataset" / DATASETS["visdrone"]   # rebound in main() from --dataset
sys.path.insert(0, str(ROOT / "scripts"))
import figdata

# condition -> (model, imgsz, per-image batch that fits in 24 GB)
GRID = {
    # R640 is the resolution arm's 640 point; on VisDrone that role was filled by
    # E0, trained by train_e0.py before this script existed
    "R640":  ("yolov8n.pt",     640, 64),
    "R960":  ("yolov8n.pt",     960, 16),
    "R1280": ("yolov8n.pt",    1280,  8),
    "S640":  ("yolov8s.pt",     640, 32),
    # capacity-only path at fixed 640 px, so the capacity axis is isolated
    "M640":  ("yolov8m.pt",     640, 16),
    "L640":  ("yolov8l.pt",     640,  8),
    "S960":  ("yolov8s.pt",     960, 12),
    "P640":  ("yolov8-p2.yaml", 640, 24),
    "P960":  ("yolov8-p2.yaml", 960,  8),
    # protocol v2.1 C4.1: widen the cost overlap so the two arms span a common range
    # self-review R2: a third P2 point so the arm spans a usable cost range
    "P1280": ("yolov8-p2.yaml", 1280,  2),
    "S1280": ("yolov8s.pt",    1280,  8),
    "R1536": ("yolov8n.pt",    1536,  4),
}
MIN_EPOCHS_DONE = 20
CONFIG = {"DETERMINISTIC": True, "SEEDS": [42, 123, 2024], "EPOCHS": 200, "PATIENCE": 30,
          "WORKERS": 8, "DEVICE": 0, "DATASET": "visdrone",
          "LAT_FRAMES": 1000, "LAT_WARMUP": 100}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def already_done(rid):
    """Done only if the weights, an epoch log of reasonable length, and a row all exist.

    A run killed during epoch 1 leaves a best.pt that is essentially random; treating it
    as complete silently poisons the study.
    """
    best = RUNS_DIR / rid / "weights" / "best.pt"
    res = RUNS_DIR / rid / "results.csv"
    if not (best.exists() and res.exists()):
        return False
    try:
        n_ep = len(pd.read_csv(res))
    except Exception:
        return False
    if n_ep < MIN_EPOCHS_DONE:
        log(f"{rid}: only {n_ep} epochs on disk -> treating as incomplete")
        return False
    return RUNS_CSV.exists() and rid in set(pd.read_csv(RUNS_CSV).run_id.astype(str))


def append_row(row):
    df = pd.DataFrame([row])
    if RUNS_CSV.exists():
        old = pd.read_csv(RUNS_CSV)
        old = old[old.run_id.astype(str) != row["run_id"]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RUNS_CSV, index=False)


def measure_cost(weights, imgsz, rid, cond, seed, val_speed=None):
    """Batch-1 latency with frames preloaded into RAM, so disk I/O is not timed.

    val_speed carries ultralytics' own preprocess/inference/postprocess split from the
    validation pass; both are stored because the field quotes the inference-only number.
    """
    import cv2
    model = YOLO(str(weights))
    dcfg = yaml.safe_load(open(DATA_YAML))
    paths = sorted((Path(dcfg["path"]) / dcfg["val"]).glob("*.jpg"))[:50]
    frames = [f for f in (cv2.imread(str(p)) for p in paths) if f is not None]
    assert frames, "could not preload any frame"

    for i in range(CONFIG["LAT_WARMUP"]):
        model.predict(frames[i % len(frames)], imgsz=imgsz,
                      device=CONFIG["DEVICE"], verbose=False)
    torch.cuda.synchronize()

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        h = None

    lat, pw = [], []
    for i in range(CONFIG["LAT_FRAMES"]):
        t0 = time.perf_counter()
        model.predict(frames[i % len(frames)], imgsz=imgsz,
                      device=CONFIG["DEVICE"], verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        if h is not None and i % 10 == 0:
            pw.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)

    lat = np.array(lat)
    vs = val_speed or {}
    row = {"run_id": rid, "exp": cond, "seed": seed, "imgsz": imgsz,
           "ms_mean": float(lat.mean()), "ms_p95": float(np.percentile(lat, 95)),
           "fps": float(1000 / lat.mean()),
           "ms_pre": float(vs["preprocess"]) if "preprocess" in vs else None,
           "ms_infer_only": float(vs["inference"]) if "inference" in vs else None,
           "ms_post": float(vs["postprocess"]) if "postprocess" in vs else None,
           "power_w": float(np.mean(pw)) if pw else None,
           "j_per_frame": float(np.mean(pw) * lat.mean() / 1000) if pw else None}
    f = FIG / "efficiency.csv"
    df = pd.DataFrame([row])
    if f.exists():
        prev = pd.read_csv(f)
        df = pd.concat([prev[prev.run_id != rid], df], ignore_index=True)
    df.to_csv(f, index=False)
    return row


def run_one(cond, seed):
    arch, imgsz, batch = GRID[cond]
    rid = f"{cond}_{CONFIG['DATASET']}_s{seed}"
    if already_done(rid):
        log(f"{rid}: already complete"); return

    run_dir = RUNS_DIR / rid
    last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"
    t0 = time.time()
    # an interrupted run leaves BOTH last.pt and best.pt; the old condition
    # (last and not best) skipped training entirely and jumped to evaluation
    if last.exists():
        log(f"{rid}: resuming from last.pt")
        try:
            YOLO(str(last)).train(resume=True)
        except Exception as e:
            log(f"{rid}: resume reported '{type(e).__name__}: {e}'; "
                f"treating training as finished")
    elif not best.exists():
        log(f"{rid}: training {arch} @ {imgsz} (batch {batch})")
        YOLO(arch).train(data=str(DATA_YAML), epochs=CONFIG["EPOCHS"],
                         patience=CONFIG["PATIENCE"], imgsz=imgsz, batch=batch,
                         workers=CONFIG["WORKERS"], device=CONFIG["DEVICE"], seed=seed,
                         deterministic=CONFIG["DETERMINISTIC"], cache=False,
                         project=str(RUNS_DIR),
                         name=rid, exist_ok=True, verbose=True, plots=True)
    hours = (time.time() - t0) / 3600

    names = yaml.safe_load(open(DATA_YAML))["names"]
    # training leaves a large fragmented cache; evaluation at high resolution then fails
    import gc
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    val_batch = 2 if imgsz >= 1280 else max(batch // 2, 4)
    res = YOLO(str(best)).val(data=str(DATA_YAML), split="val", imgsz=imgsz,
                              batch=val_batch, device=CONFIG["DEVICE"], workers=0,
                              plots=True, verbose=False)
    met = {"mAP50": float(res.box.map50), "mAP50_95": float(res.box.map),
           "precision": float(res.box.mp), "recall": float(res.box.mr)}

    figdata.capture_training_curve(run_dir, rid, cond, seed, OUT)
    figdata.capture_val_artifacts(res, names, rid, cond, seed, OUT)
    figdata.capture_model_info(best, rid, cond, seed, OUT, imgsz=imgsz)
    vy = yaml.safe_load(open(DATA_YAML))
    figdata.capture_predictions(best, Path(vy["path"]) / vy["val"], names, rid, cond,
                                seed, OUT, imgsz=imgsz, device=CONFIG["DEVICE"])
    cost = measure_cost(best, imgsz, rid, cond, seed, val_speed=res.speed)

    append_row({"run_id": rid, "exp": cond, "dataset": CONFIG["DATASET"],
                "model": arch.replace(".yaml", ""), "seed": seed,
                "epochs": CONFIG["EPOCHS"], "imgsz": imgsz, "batch": batch, **met,
                "ms_mean": cost["ms_mean"], "fps": cost["fps"],
                "power_w": cost["power_w"], "train_hours": round(hours, 2),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "host": platform.node(), "torch": torch.__version__,
                "script": "train_resolution.py"})
    log(f"{rid}: mAP50={met['mAP50']:.4f} mAP50-95={met['mAP50_95']:.4f} "
        f"{cost['ms_mean']:.1f} ms ({hours:.1f} h)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=list(GRID))
    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["SEEDS"])
    ap.add_argument("--dataset", choices=list(DATASETS), default="visdrone")
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader workers; 0 restores the seed effect that "
                         "multi-worker loading suppresses on UAVDT")
    ap.add_argument("--deterministic", choices=["true", "false"], default="true",
                    help="false lifts the seed suppression seen on UAVDT with yolov8s")
    a = ap.parse_args()

    global DATA_YAML
    CONFIG["DATASET"] = a.dataset
    CONFIG["DETERMINISTIC"] = (a.deterministic == "true")
    if a.workers is not None:
        CONFIG["WORKERS"] = a.workers
    DATA_YAML = ROOT / "dataset" / DATASETS[a.dataset]
    assert DATA_YAML.exists(), f"missing {DATA_YAML}"
    assert torch.cuda.is_available(), "CUDA unavailable"
    log(f"GPU: {torch.cuda.get_device_name(0)} | {a.conditions} | seeds {a.seeds}")

    # seed-outer so an interrupted study still yields complete paired comparisons
    for seed in a.seeds:
        for cond in a.conditions:
            try:
                run_one(cond, seed)
            except Exception as e:
                log(f"{cond} s{seed} FAILED: {type(e).__name__}: {e}")

    if RUNS_CSV.exists():
        d = pd.read_csv(RUNS_CSV)
        d = d[d.exp.isin(["E0"] + list(GRID))]
        cols = [c for c in ["mAP50", "mAP50_95", "ms_mean"] if c in d.columns]
        log("study so far\n" + d.groupby("exp")[cols].agg(["mean", "size"]).round(4).to_string())
    log("ALL DONE")


if __name__ == "__main__":
    main()
