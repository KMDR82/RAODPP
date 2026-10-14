#!/usr/bin/env python3
"""
make_qualitative.py — side-by-side detections for the qualitative figure.

Runs the saved weights of three configurations on a few validation frames at the
deployment threshold and writes one panel per frame: ground truth, baseline, and the best
configuration on each cost axis. Uses the GPU, so run it when nothing else is training.

    python scripts/make_qualitative.py --dry-run
    python scripts/make_qualitative.py --n 4

Frames are chosen deterministically as the validation images with the most annotated
boxes, so the panel shows the dense small-object case the paper is about rather than a
hand-picked success.
"""
import argparse, time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path.home() / "RAODpp"
RUNS, FIG = ROOT / "model" / "runs", ROOT / "fig"
DATA_YAML = ROOT / "dataset" / "visdrone_raodpp.yaml"

# condition -> (run directory prefix, input size, panel caption)
PANELS = [("E0",    640,  "baseline  n@640"),
          ("S960",  960,  "capacity  s@960"),
          ("R1536", 1536, "resolution  n@1536")]
SEED = 42
CONF = 0.25          # a plausible deployment threshold, stated in the caption
BOX_W = 2


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_gt(img_dir, lbl_dir, name):
    W, H = Image.open(img_dir / name).size
    p = lbl_dir / f"{Path(name).stem}.txt"
    out = []
    if p.exists():
        for line in open(p):
            f = line.split()
            if len(f) != 5:
                continue
            cx, cy, bw, bh = map(float, f[1:])
            out.append([(cx - bw / 2) * W, (cy - bh / 2) * H,
                        (cx + bw / 2) * W, (cy + bh / 2) * H])
    return out, (W, H)


def draw(img, boxes, colour):
    d = ImageDraw.Draw(img)
    for b in boxes:
        d.rectangle([b[0], b[1], b[2], b[3]], outline=colour, width=BOX_W)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of frames")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(DATA_YAML))
    img_dir = Path(cfg["path"]) / cfg["val"]
    lbl_dir = Path(str(img_dir).replace("/images", "/labels"))

    # densest frames, chosen by label count so the selection is reproducible
    counts = []
    for p in sorted(img_dir.glob("*.jpg")):
        lp = lbl_dir / f"{p.stem}.txt"
        n = sum(1 for _ in open(lp)) if lp.exists() else 0
        counts.append((n, p.name))
    counts.sort(reverse=True)
    frames = [n for _, n in counts[:a.n]]
    log(f"frames chosen (densest {a.n}): {frames}")

    missing = [c for c, _, _ in PANELS
               if not (RUNS / f"{c}_visdrone_s{SEED}" / "weights" / "best.pt").exists()]
    if missing:
        log(f"missing weights for {missing}; cannot build the figure")
        return
    if a.dry_run:
        log("dry run: weights present, frames selected, nothing rendered")
        return

    from ultralytics import YOLO
    FIG.mkdir(parents=True, exist_ok=True)

    for name in frames:
        gt, (W, H) = load_gt(img_dir, lbl_dir, name)
        tiles, captions = [], []

        base = Image.open(img_dir / name).convert("RGB")
        tiles.append(draw(base.copy(), gt, (60, 200, 60)))
        captions.append(f"ground truth  ({len(gt)} boxes)")

        for cond, imgsz, cap in PANELS:
            w = RUNS / f"{cond}_visdrone_s{SEED}" / "weights" / "best.pt"
            r = YOLO(str(w)).predict(str(img_dir / name), imgsz=imgsz, conf=a.conf,
                                     device=0, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            tiles.append(draw(base.copy(), b, (220, 60, 60)))
            captions.append(f"{cap}  ({len(b)} detections)")
            log(f"{name}  {cond}: {len(b)} detections at conf >= {a.conf}")

        # stack 2x2 with a caption strip under each tile
        strip = 22
        tw, th = W, H + strip
        sheet = Image.new("RGB", (tw * 2, th * 2), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for i, (t, c) in enumerate(zip(tiles, captions)):
            x, y = (i % 2) * tw, (i // 2) * th
            sheet.paste(t, (x, y))
            d.text((x + 6, y + H + 5), c, fill=(20, 20, 20))
        out = FIG / f"qualitative_{Path(name).stem}.jpg"
        sheet.save(out, quality=90)
        log(f"wrote {out}")

    log("assemble the chosen panel into fig/qualitative.pdf, or reference one "
        "of the JPEGs directly from the manuscript")


if __name__ == "__main__":
    main()
