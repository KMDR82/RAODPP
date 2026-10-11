#!/usr/bin/env python3
"""
backfill_figdata.py — fills the five data gaps left by the training loop.

Run it AFTER an experiment finishes. It re-reads what was already saved on disk
(weights, results.csv, label files) and never retrains anything.

    cd ~/RAODpp && source .venv/bin/activate
    python scripts/backfill_figdata.py --exp E0

Fills:
  1. AP-small / medium / large  (pycocotools; ultralytics does not report these)
  2. actual epoch count + best epoch   (early stopping makes the planned 200 wrong)
  3. wall-clock training time per run
  4. parameter count and GFLOPs
  5. per-image resolutions of the evaluation split

Writes:
  output/figdata/ap_size_bands.csv     one row per run
  output/figdata/val_image_sizes.csv   one row per evaluation image (written once)
  output/runs_enriched.csv             runs.csv + corrected epochs, time, params

Re-running is safe: existing rows for a run are replaced, not duplicated.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path.home() / "RAODpp"
OUT = ROOT / "output"
FIGDATA = OUT / "figdata"
RUNS_DIR = ROOT / "model" / "runs"
FIGDATA.mkdir(parents=True, exist_ok=True)

DATASET_YAML = {"visdrone": ROOT / "dataset" / "visdrone_raodpp.yaml",
                "coco10":   ROOT / "dataset" / "coco10_raodpp.yaml",
                "exdark":   ROOT / "dataset" / "exdark_raodpp.yaml"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def upsert(df_new, path, key_cols):
    """Replace rows with the same key instead of appending duplicates."""
    if path.exists():
        old = pd.read_csv(path)
        mask = ~old.set_index(key_cols).index.isin(df_new.set_index(key_cols).index)
        df_new = pd.concat([old[mask], df_new], ignore_index=True)
    df_new.to_csv(path, index=False)
    return len(df_new)


# ---------------------------------------------------------------- gap 5: image sizes
def dump_val_image_sizes(val_dir):
    path = FIGDATA / "val_image_sizes.csv"
    if path.exists():
        log(f"val_image_sizes.csv already present ({len(pd.read_csv(path))} rows)")
        return pd.read_csv(path)
    rows = []
    for p in sorted(Path(val_dir).glob("*.jpg")):
        w, h = Image.open(p).size
        rows.append({"image": p.name, "width": w, "height": h})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log(f"wrote val_image_sizes.csv ({len(df)} images)")
    return df


# ------------------------------------------------- gap 1: COCO-style size-band AP
def build_coco_gt(val_img_dir, val_lbl_dir, class_names, sizes_df, cache):
    """Convert the frozen val split into a COCO ground-truth dict (cached on disk)."""
    if cache.exists():
        return json.load(open(cache))

    size_map = dict(zip(sizes_df.image, zip(sizes_df.width, sizes_df.height)))
    images, annotations = [], []
    ann_id = 1
    for img_id, p in enumerate(sorted(Path(val_img_dir).glob("*.jpg")), start=1):
        w, h = size_map[p.name]
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lp = Path(val_lbl_dir) / (p.stem + ".txt")
        if not lp.exists():
            continue
        for line in open(lp):
            f = line.split()
            if len(f) != 5:
                continue
            c = int(f[0]); cx, cy, bw, bh = map(float, f[1:])
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            aw, ah = bw * w, bh * h
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": c + 1,
                                "bbox": [x, y, aw, ah], "area": aw * ah, "iscrowd": 0})
            ann_id += 1
    gt = {"images": images, "annotations": annotations,
          "categories": [{"id": i + 1, "name": n} for i, n in sorted(class_names.items())]}
    json.dump(gt, open(cache, "w"))
    log(f"built COCO GT: {len(images)} images, {len(annotations)} annotations")
    return gt


def size_band_ap(weights, gt, val_img_dir, imgsz, device, conf=0.001):
    """Run inference and evaluate with COCOeval to obtain AP-S/M/L."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO

    name_to_id = {im["file_name"]: im["id"] for im in gt["images"]}
    model = YOLO(str(weights))
    dets = []
    imgs = [str(p) for p in sorted(Path(val_img_dir).glob("*.jpg"))]
    for r in model.predict(source=imgs, imgsz=imgsz, conf=conf, device=device,
                           stream=True, verbose=False):
        img_id = name_to_id[Path(r.path).name]
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy(); cls = b.cls.cpu().numpy(); cf = b.conf.cpu().numpy()
        for k in range(len(cf)):
            x1, y1, x2, y2 = xyxy[k]
            dets.append({"image_id": int(img_id), "category_id": int(cls[k]) + 1,
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "score": float(cf[k])})
    if not dets:
        return None

    tmp_gt, tmp_dt = FIGDATA / "_tmp_gt.json", FIGDATA / "_tmp_dt.json"
    json.dump(gt, open(tmp_gt, "w")); json.dump(dets, open(tmp_dt, "w"))
    coco_gt = COCO(str(tmp_gt))
    coco_dt = coco_gt.loadRes(str(tmp_dt))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    tmp_gt.unlink(missing_ok=True); tmp_dt.unlink(missing_ok=True)
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    return {k: float(v) for k, v in zip(keys, s)}


# ------------------------------------------- gaps 2-4: epochs, time, params, GFLOPs
def run_meta(run_dir, weights):
    meta = {}
    res_csv = Path(run_dir) / "results.csv"
    if res_csv.exists():
        df = pd.read_csv(res_csv)
        df.columns = [c.strip() for c in df.columns]
        meta["epochs_actual"] = int(len(df))
        fit_col = next((c for c in df.columns if "mAP50-95" in c or "metrics/mAP50-95" in c), None)
        if fit_col:
            meta["best_epoch"] = int(df[fit_col].idxmax() + 1)
            meta["best_mAP50_95_train_log"] = float(df[fit_col].max())
        tcol = next((c for c in df.columns if c.lower() == "time"), None)
        if tcol:
            meta["train_seconds"] = float(df[tcol].max())
            meta["sec_per_epoch"] = round(float(df[tcol].max()) / max(1, len(df)), 2)
    try:
        from ultralytics import YOLO
        m = YOLO(str(weights))
        info = m.info(verbose=False)          # (layers, params, gradients, flops)
        if info and len(info) >= 4:
            meta.update({"layers": int(info[0]), "params": int(info[1]),
                         "gflops": float(info[3])})
    except Exception as e:
        log(f"  model info skipped: {e}")
    return meta


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E0")
    ap.add_argument("--dataset", default="visdrone")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--skip-cocoeval", action="store_true",
                    help="fill only epochs/time/params (fast, no GPU)")
    args = ap.parse_args()

    yml = DATASET_YAML[args.dataset]
    assert yml.exists(), f"missing {yml}"
    dcfg = yaml.safe_load(open(yml))
    base = Path(dcfg["path"])
    val_img = base / dcfg["val"]
    val_lbl = Path(str(val_img).replace("/images", "/labels"))
    class_names = dcfg["names"]

    runs = sorted(RUNS_DIR.glob(f"{args.exp}_{args.dataset}_s*"))
    assert runs, f"no runs found matching {args.exp}_{args.dataset}_s*"
    log(f"found {len(runs)} runs: {[r.name for r in runs]}")

    sizes_df = dump_val_image_sizes(val_img)

    gt = None
    if not args.skip_cocoeval:
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            log("pycocotools missing -> run: pip install pycocotools")
            sys.exit(1)
        gt = build_coco_gt(val_img, val_lbl, class_names, sizes_df,
                           FIGDATA / f"cocogt_{args.dataset}_val.json")

    band_rows, meta_rows = [], []
    for rd in runs:
        w = rd / "weights" / "best.pt"
        if not w.exists():
            log(f"{rd.name}: no best.pt, skipping")
            continue
        seed = int(rd.name.split("_s")[-1])
        log(f"{rd.name}: collecting metadata")
        meta = run_meta(rd, w)
        meta_rows.append({"run_id": rd.name, "exp": args.exp,
                          "dataset": args.dataset, "seed": seed, **meta})

        if gt is not None:
            log(f"{rd.name}: COCOeval (size bands)")
            bands = size_band_ap(w, gt, val_img, args.imgsz, args.device)
            if bands:
                band_rows.append({"run_id": rd.name, "exp": args.exp,
                                  "dataset": args.dataset, "seed": seed, **bands})

    if band_rows:
        n = upsert(pd.DataFrame(band_rows), FIGDATA / "ap_size_bands.csv",
                   ["run_id"])
        log(f"ap_size_bands.csv -> {n} rows")
        print(pd.DataFrame(band_rows)[["seed", "AP", "AP50", "AP_small",
                                       "AP_medium", "AP_large"]].round(4).to_string(index=False))

    meta_df = pd.DataFrame(meta_rows)
    if len(meta_df):
        runs_csv = OUT / "runs.csv"
        if runs_csv.exists():
            base_df = pd.read_csv(runs_csv)
            enriched = base_df.merge(meta_df.drop(columns=["exp", "dataset", "seed"]),
                                     on="run_id", how="left")
        else:
            enriched = meta_df
        enriched.to_csv(OUT / "runs_enriched.csv", index=False)
        log(f"runs_enriched.csv -> {len(enriched)} rows")
        cols = [c for c in ["run_id", "seed", "epochs_actual", "best_epoch",
                            "sec_per_epoch", "params", "gflops"] if c in meta_df.columns]
        print(meta_df[cols].to_string(index=False))

    log("backfill complete")


if __name__ == "__main__":
    main()
