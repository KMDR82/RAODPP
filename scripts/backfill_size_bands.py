#!/usr/bin/env python3
"""
backfill_size_bands.py — compute size-band AP for runs that have prediction dumps but no
entry in ap_size_bands.csv.

`ap_size_bands.csv` was written only for the baseline, because the size-band evaluator was
added after those runs. The prediction dumps for every later run are on disk, so nothing
needs re-inferring: the COCO evaluator is run over the stored predictions instead. No GPU
is used.

    python scripts/backfill_size_bands.py --dry-run
    python scripts/backfill_size_bands.py --dataset visdrone

Size bands follow the COCO convention on the ground-truth box: small < 32^2,
32^2 <= medium < 96^2, large >= 96^2. AP is averaged over IoU 0.50:0.95.
"""
import argparse, contextlib, io, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path.home() / "RAODpp"
OUT, FIGDATA = ROOT / "output", ROOT / "output" / "figdata"
DATASETS = {"visdrone": "visdrone_raodpp.yaml", "uavdt": "uavdt_raodpp.yaml"}
TARGET = FIGDATA / "ap_size_bands.csv"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_gt(dataset):
    """Ground truth in COCO form, from the frozen val split of this dataset."""
    cfg = yaml.safe_load(open(ROOT / "dataset" / DATASETS[dataset]))
    img_dir = Path(cfg["path"]) / cfg["val"]
    lbl_dir = Path(str(img_dir).replace("/images", "/labels"))
    names = cfg["names"]

    images, anns, ann_id = [], [], 1
    index = {}
    for i, p in enumerate(sorted(img_dir.glob("*.jpg")), start=1):
        W, H = Image.open(p).size
        images.append({"id": i, "file_name": p.name, "width": W, "height": H})
        index[p.name] = (i, W, H)
        lp = lbl_dir / f"{p.stem}.txt"
        if not lp.exists():
            continue
        for line in open(lp):
            f = line.split()
            if len(f) != 5:
                continue
            c = int(f[0]); cx, cy, bw, bh = map(float, f[1:])
            x, y, w, h = (cx - bw / 2) * W, (cy - bh / 2) * H, bw * W, bh * H
            anns.append({"id": ann_id, "image_id": i, "category_id": c,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": k, "name": v} for k, v in sorted(names.items())]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
    log(f"{dataset}: {len(images)} val images, {len(anns)} boxes")
    return coco, index


def evaluate(coco, index, pred_csv):
    d = pd.read_csv(pred_csv)
    dets = []
    for r in d.itertuples():
        if r.image not in index:
            continue
        img_id, W, H = index[r.image]
        w, h = r.w * W, r.h * H
        dets.append({"image_id": img_id, "category_id": int(r.class_id),
                     "bbox": [(r.cx - r.w / 2) * W, (r.cy - r.h / 2) * H, w, h],
                     "score": float(r.conf)})
    if not dets:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(dets)
        e = COCOeval(coco, dt, "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    return {k: (float(v) if v >= 0 else np.nan) for k, v in zip(keys, s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="visdrone")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    done = set()
    if TARGET.exists():
        done = set(pd.read_csv(TARGET).run_id.astype(str))

    preds = sorted(FIGDATA.glob(f"predictions_*_{a.dataset}_s*.csv"))
    todo = [p for p in preds if p.stem.replace("predictions_", "") not in done]
    log(f"{len(preds)} prediction dumps, {len(todo)} without size bands")
    if a.dry_run:
        for p in todo[:20]:
            print("   ", p.stem.replace("predictions_", ""))
        return
    if not todo:
        return

    coco, index = build_gt(a.dataset)
    rows = []
    for i, p in enumerate(todo, 1):
        rid = p.stem.replace("predictions_", "")
        try:
            r = evaluate(coco, index, p)
        except Exception as e:
            log(f"{rid}: FAILED {type(e).__name__}: {e}")
            continue
        if r is None:
            log(f"{rid}: no detections, skipped"); continue
        rows.append({"run_id": rid, "exp": rid.split("_")[0], "dataset": a.dataset,
                     "seed": int(rid.split("_s")[-1]), **r})
        if i % 5 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)}  last: {rid} AP_small={r['AP_small']:.4f}")

    if not rows:
        return
    new = pd.DataFrame(rows)
    if TARGET.exists():
        old = pd.read_csv(TARGET)
        new = pd.concat([old[~old.run_id.isin(new.run_id)], new], ignore_index=True)
    new.to_csv(TARGET, index=False)
    log(f"{TARGET.name}: {len(new)} rows")
    print(new.groupby("exp")[["AP_small", "AP_medium", "AP_large"]]
          .mean().round(4).to_string())


if __name__ == "__main__":
    main()
