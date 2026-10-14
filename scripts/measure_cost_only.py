#!/usr/bin/env python3
"""
measure_cost_only.py — backfill the cost axis for models trained before cost was measured.

Runs no training. Loads existing weights and measures batch-1 latency and NVML power with
frames preloaded into RAM, exactly as train_resolution.py does, so the numbers are
comparable across the whole study.

    python scripts/measure_cost_only.py --runs E0_visdrone_s42 --imgsz 640
    python scripts/measure_cost_only.py --exp E0 --imgsz 640            # all seeds of E0
"""
import argparse, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT = Path.home() / "RAODpp"
RUNS_DIR, FIG = ROOT / "model" / "runs", ROOT / "output" / "figdata"
RUNS_CSV = ROOT / "output" / "runs.csv"
DATASETS = {"visdrone": "visdrone_raodpp.yaml", "uavdt": "uavdt_raodpp.yaml"}
DATA_YAML = ROOT / "dataset" / DATASETS["visdrone"]   # rebound in main() from --dataset
LAT_FRAMES, LAT_WARMUP, N_PRELOAD = 1000, 100, 50
REPEATS = 3          # protocol: report the median of three independent passes
IDLE_MB = 2000       # refuse to measure if the GPU is busier than this


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def idle_power_w(seconds=20, device=0):
    """Board power with nothing running, sampled over a window.

    NVML reports total board draw, which includes the idle floor. Energy per frame
    computed from it therefore mixes the cost of inference with the cost of the card
    merely being powered on, and the two arms are affected unequally: a light model sits
    close to the floor while a heavy one does not. Both the total and the marginal figure
    are reported so a reader can see which one a claim rests on.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None
    log(f"sampling idle power for {seconds}s (leave the GPU alone)")
    vals = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        vals.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
        time.sleep(0.25)
    v = float(np.median(vals))
    log(f"idle power: {v:.1f} W (median of {len(vals)} samples, "
        f"min {min(vals):.1f}, max {max(vals):.1f})")
    return v


def gpu_busy_mb():
    out = subprocess.run(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
        shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    try:
        return int(out.splitlines()[0])
    except Exception:
        return -1


def measure_once(model, frames, imgsz, device):
    for i in range(LAT_WARMUP):
        model.predict(frames[i % len(frames)], imgsz=imgsz, device=device, verbose=False)
    torch.cuda.synchronize()

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        h = None

    lat, pw = [], []
    for i in range(LAT_FRAMES):
        t0 = time.perf_counter()
        model.predict(frames[i % len(frames)], imgsz=imgsz, device=device, verbose=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
        if h is not None and i % 10 == 0:
            pw.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
    lat = np.array(lat)
    return {"ms_mean": float(lat.mean()), "ms_p95": float(np.percentile(lat, 95)),
            "power_w": float(np.mean(pw)) if pw else np.nan}


def measure(weights, imgsz, device=0, idle_w=None):
    """Median of REPEATS independent passes; frames preloaded so disk I/O is not timed."""
    import cv2
    model = YOLO(str(weights))
    dcfg = yaml.safe_load(open(DATA_YAML))
    paths = sorted((Path(dcfg["path"]) / dcfg["val"]).glob("*.jpg"))[:N_PRELOAD]
    frames = [f for f in (cv2.imread(str(p)) for p in paths) if f is not None]
    assert frames, "could not preload any frame"

    runs = [measure_once(model, frames, imgsz, device) for _ in range(REPEATS)]
    med = {k: float(np.median([r[k] for r in runs])) for k in runs[0]}
    spread = max(r["ms_mean"] for r in runs) - min(r["ms_mean"] for r in runs)
    med["ms_spread"] = float(spread)
    med["fps"] = float(1000 / med["ms_mean"])
    has_power = med["power_w"] == med["power_w"]
    med["j_per_frame"] = (float(med["power_w"] * med["ms_mean"] / 1000)
                          if has_power else None)
    # marginal energy: what inference costs above the idle floor
    if has_power and idle_w is not None:
        med["idle_w"] = float(idle_w)
        med["power_marginal_w"] = float(med["power_w"] - idle_w)
        med["j_marginal"] = float(med["power_marginal_w"] * med["ms_mean"] / 1000)
    else:
        med["idle_w"] = None
        med["power_marginal_w"] = None
        med["j_marginal"] = None
    if not has_power:
        med["power_w"] = None
    return med


def upsert(path, row, key="run_id"):
    df = pd.DataFrame([row])
    if Path(path).exists():
        prev = pd.read_csv(path)
        keep = prev[prev[key].astype(str) != str(row[key])]
        df = pd.concat([keep, df], ignore_index=True)
    df.to_csv(path, index=False)
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--exp", default=None)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dataset", choices=list(DATASETS), default="visdrone")
    ap.add_argument("--idle-seconds", type=int, default=20,
                    help="window for the idle-power measurement taken before the runs")
    ap.add_argument("--no-idle", action="store_true",
                    help="skip the idle measurement; marginal energy is then unavailable")
    a = ap.parse_args()

    global DATA_YAML
    DATA_YAML = ROOT / "dataset" / DATASETS[a.dataset]
    assert DATA_YAML.exists(), f"missing {DATA_YAML}"

    if a.runs:
        run_ids = a.runs
    else:
        assert a.exp, "either --runs or --exp is required"
        run_ids = sorted(p.name for p in RUNS_DIR.glob(f"{a.exp}_{a.dataset}_s*")
                         if (p / "weights" / "best.pt").exists())
    assert run_ids, "no matching runs found"
    busy = gpu_busy_mb()
    if busy > IDLE_MB:
        raise SystemExit(f"GPU is busy ({busy} MiB in use). Latency must be measured on an "
                         f"idle GPU; stop other runs first.")
    log(f"{len(run_ids)} run(s) at imgsz={a.imgsz}")

    idle_w = None if a.no_idle else idle_power_w(a.idle_seconds, a.device)
    runs = pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame()
    for rid in run_ids:
        w = RUNS_DIR / rid / "weights" / "best.pt"
        if not w.exists():
            log(f"{rid}: no weights, skipped"); continue
        c = measure(w, a.imgsz, a.device, idle_w=idle_w)
        seed = int(rid.split("_s")[-1])
        exp = rid.split("_")[0]
        upsert(FIG / "efficiency.csv",
               {"run_id": rid, "exp": exp, "seed": seed, "imgsz": a.imgsz, **c})
        if len(runs) and rid in set(runs.run_id.astype(str)):
            for k, v in c.items():
                runs.loc[runs.run_id.astype(str) == rid, k] = v
        log(f"{rid}: {c['ms_mean']:.2f} ms (spread {c['ms_spread']:.2f})  "
            f"{c['fps']:.1f} FPS  "
            f"{c['power_w'] if c['power_w'] is None else round(c['power_w'],1)} W"
            + (f"  ({c['power_marginal_w']:.1f} W above idle)"
               if c.get("power_marginal_w") is not None else ""))

    if len(runs):
        runs.to_csv(RUNS_CSV, index=False)
        log(f"runs.csv updated ({len(runs)} rows)")
    log("DONE")


if __name__ == "__main__":
    main()
