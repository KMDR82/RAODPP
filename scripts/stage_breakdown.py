#!/usr/bin/env python3
"""
stage_breakdown.py — where the latency of a checkpoint is actually spent.

One checkpoint (R1280 on UAVDT, seed 123) is reproducibly slower than
its siblings, 7.96 ms against 6.2, and the draft attributed this to a longer non-maximum
suppression stage because that model trained for more epochs and therefore proposes more
boxes. That was a conjecture. This measures it.

For each run it reports the preprocess / inference / postprocess split that ultralytics
records, and the number of detections surviving a low confidence threshold, which is the
quantity NMS cost scales with. If the slow checkpoint is slow in postprocess and also
returns more boxes, the conjecture holds; if it is slow in inference, it does not and the
text must say so.

    python scripts/stage_breakdown.py --exp R1280 --imgsz 1280 --dataset uavdt

Requires an idle GPU. Writes output/figdata/stage_breakdown.csv.
"""
import argparse, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT = Path.home() / "RAODpp"
RUNS, FIGDATA = ROOT / "model" / "runs", ROOT / "output" / "figdata"
DATASETS = {"visdrone": "visdrone_raodpp.yaml", "uavdt": "uavdt_raodpp.yaml"}
N_FRAMES, N_WARMUP, IDLE_MB = 300, 50, 2000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gpu_busy_mb():
    out = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                         shell=True, capture_output=True, text=True, timeout=20).stdout
    try:
        return int(out.split()[0])
    except Exception:
        return -1


def measure(weights, frames, imgsz, conf):
    model = YOLO(str(weights))
    for i in range(N_WARMUP):
        model.predict(frames[i % len(frames)], imgsz=imgsz, conf=conf,
                      device=0, verbose=False)
    torch.cuda.synchronize()

    pre, inf, post, ndet = [], [], [], []
    for i in range(N_FRAMES):
        r = model.predict(frames[i % len(frames)], imgsz=imgsz, conf=conf,
                          device=0, verbose=False)[0]
        s = r.speed
        pre.append(float(s.get("preprocess", np.nan)))
        inf.append(float(s.get("inference", np.nan)))
        post.append(float(s.get("postprocess", np.nan)))
        ndet.append(0 if r.boxes is None else len(r.boxes))
    torch.cuda.synchronize()
    return {"ms_pre": float(np.mean(pre)), "ms_inference": float(np.mean(inf)),
            "ms_post": float(np.mean(post)),
            "ms_total": float(np.mean(pre) + np.mean(inf) + np.mean(post)),
            "detections_per_frame": float(np.mean(ndet))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--dataset", choices=list(DATASETS), default="visdrone")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="low threshold: NMS cost scales with the candidates it must sort")
    a = ap.parse_args()

    busy = gpu_busy_mb()
    if busy > IDLE_MB:
        raise SystemExit(f"GPU busy ({busy} MiB). Stage timings need an idle device.")

    import cv2
    cfg = yaml.safe_load(open(ROOT / "dataset" / DATASETS[a.dataset]))
    paths = sorted((Path(cfg["path"]) / cfg["val"]).glob("*.jpg"))[:50]
    frames = [f for f in (cv2.imread(str(p)) for p in paths) if f is not None]
    assert frames, "could not preload frames"

    runs = sorted(p.name for p in RUNS.glob(f"{a.exp}_{a.dataset}_s*")
                  if (p / "weights" / "best.pt").exists())
    assert runs, f"no runs for {a.exp} on {a.dataset}"
    log(f"{len(runs)} run(s), conf >= {a.conf}")

    rows = []
    for rid in runs:
        w = RUNS / rid / "weights" / "best.pt"
        epochs = None
        rp = RUNS / rid / "results.csv"
        if rp.exists():
            try:
                epochs = len(pd.read_csv(rp))
            except Exception:
                pass
        m = measure(w, frames, a.imgsz, a.conf)
        rows.append({"run_id": rid, "exp": a.exp, "dataset": a.dataset,
                     "seed": int(rid.split("_s")[-1]), "epochs": epochs, **m})
        log(f"{rid}: total {m['ms_total']:.2f} ms = pre {m['ms_pre']:.2f} + "
            f"inference {m['ms_inference']:.2f} + post {m['ms_post']:.2f}  |  "
            f"{m['detections_per_frame']:.0f} detections/frame  |  {epochs} epochs")

    df = pd.DataFrame(rows)
    f = FIGDATA / "stage_breakdown.csv"
    if f.exists():
        old = pd.read_csv(f)
        df = pd.concat([old[~old.run_id.isin(df.run_id)], df], ignore_index=True)
    df.to_csv(f, index=False)

    cur = df[(df.exp == a.exp) & (df.dataset == a.dataset)]
    print("\n" + cur[["run_id", "epochs", "ms_pre", "ms_inference", "ms_post",
                      "ms_total", "detections_per_frame"]].round(2).to_string(index=False))
    if len(cur) >= 3:
        slow = cur.loc[cur.ms_total.idxmax()]
        rest = cur[cur.run_id != slow.run_id]
        print(f"\nslowest run: {slow.run_id}")
        for col in ["ms_pre", "ms_inference", "ms_post", "detections_per_frame"]:
            d = slow[col] - rest[col].mean()
            print(f"  {col:22s} {slow[col]:8.2f} vs {rest[col].mean():8.2f} "
                  f"for the others  ({d:+.2f})")
        print("\nThe conjecture that the extra time is spent in postprocess holds only if "
              "ms_post carries most of the difference and the detection count is higher.")
    print(f"\nwritten: {f}")


if __name__ == "__main__":
    main()
