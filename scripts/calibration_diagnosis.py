#!/usr/bin/env python3
"""
calibration_diagnosis.py — RAOD++ / phase G1: is the detector's confidence biased by
object size, and what does that cost at a deployment threshold?

Uses only data already on disk (val prediction dumps + frozen val labels). No GPU.

    python scripts/calibration_diagnosis.py --exp E0

Produces, in output/figdata/:
  calib_size_bins.csv        per size bin: mean confidence vs empirical precision, ECE, gap
  calib_reliability.csv      reliability-curve points, per size bin (for the figure)
  calib_operating_point.csv  precision/recall per size bin at deployment thresholds
  calib_summary.json         headline numbers for the manuscript

Definitions
  A detection is a true positive if it matches an unused ground-truth box of the same
  class at IoU >= 0.5, matching greedily in descending confidence (standard protocol).
  Size bins follow the COCO convention on the *predicted* box, because that is all a
  deployed system knows at inference time.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path.home() / "RAODpp"
FIG = ROOT / "output" / "figdata"
SIZE_EDGES = [0, 16**2, 32**2, 96**2, np.inf]
SIZE_NAMES = ["tiny", "small", "medium", "large"]
CONF_BINS = np.linspace(0, 1, 16)
OP_THRESHOLDS = [0.10, 0.25, 0.40, 0.50]


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def load_gt():
    d = yaml.safe_load(open(ROOT / "dataset" / "visdrone_raodpp.yaml"))
    img_dir = Path(d["path"]) / d["val"]
    lbl_dir = Path(str(img_dir).replace("/images", "/labels"))
    sizes = pd.read_csv(FIG / "val_image_sizes.csv").set_index("image")
    gt = {}
    for name, row in sizes.iterrows():
        W, H = row.width, row.height
        p = lbl_dir / (Path(name).stem + ".txt")
        boxes, cls = [], []
        if p.exists():
            for line in open(p):
                f = line.split()
                if len(f) != 5:
                    continue
                c = int(f[0]); cx, cy, bw, bh = map(float, f[1:])
                boxes.append([(cx - bw / 2) * W, (cy - bh / 2) * H,
                              (cx + bw / 2) * W, (cy + bh / 2) * H])
                cls.append(c)
        gt[name] = (np.array(boxes) if boxes else np.zeros((0, 4)),
                    np.array(cls, dtype=int), W, H)
    return gt, d["names"]


def score_run(path, gt):
    """Return a detection table with tp flag and predicted area, plus GT counts."""
    det = pd.read_csv(path)
    det = det.sort_values("conf", ascending=False)
    sizes = {k: (v[2], v[3]) for k, v in gt.items()}
    out = []
    n_gt_by_bin = np.zeros(len(SIZE_NAMES))

    for name, (gb, gc, W, H) in gt.items():
        if len(gb):
            ga = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            n_gt_by_bin += np.histogram(ga, bins=SIZE_EDGES)[0]

    for name, d in det.groupby("image", sort=False):
        if name not in gt:
            continue
        gb, gc, W, H = gt[name]
        px = np.stack([(d.cx - d.w / 2) * W, (d.cy - d.h / 2) * H,
                       (d.cx + d.w / 2) * W, (d.cy + d.h / 2) * H], axis=1)
        area = (px[:, 2] - px[:, 0]) * (px[:, 3] - px[:, 1])
        cls = d.class_id.values; conf = d.conf.values
        tp = np.zeros(len(d), dtype=bool)
        if len(gb):
            M = iou_mat(px, gb)
            used = np.zeros(len(gb), dtype=bool)
            for i in range(len(d)):
                cand = np.where((gc == cls[i]) & (~used) & (M[i] >= 0.5))[0]
                if len(cand):
                    j = cand[np.argmax(M[i, cand])]
                    used[j] = True; tp[i] = True
        out.append(pd.DataFrame({"conf": conf, "tp": tp, "area": area,
                                 "class_id": cls}))
    df = pd.concat(out, ignore_index=True)
    df["size_bin"] = pd.cut(df.area, bins=SIZE_EDGES, labels=SIZE_NAMES)
    return df, n_gt_by_bin


def ece(conf, tp, bins=CONF_BINS):
    idx = np.digitize(conf, bins) - 1
    e, n = 0.0, len(conf)
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(conf[m].mean() - tp[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E0")
    ap.add_argument("--min-conf", type=float, default=0.05)
    a = ap.parse_args()

    gt, names = load_gt()
    files = sorted(FIG.glob(f"predictions_{a.exp}_visdrone_s*.csv"))
    assert files, f"no prediction dumps for {a.exp}"
    print(f"runs: {len(files)}")

    rows, rel, op = [], [], []
    for f in files:
        seed = int(f.stem.split("_s")[-1])
        df, n_gt_bin = score_run(f, gt)
        df = df[df.conf >= a.min_conf]

        for k, sb in enumerate(SIZE_NAMES):
            m = df.size_bin == sb
            if m.sum() < 50:
                continue
            c, t = df.conf[m].values, df.tp[m].values.astype(float)
            rows.append({"seed": seed, "size_bin": sb, "n_det": int(m.sum()),
                         "mean_conf": float(c.mean()), "precision": float(t.mean()),
                         "gap": float(c.mean() - t.mean()), "ece": ece(c, t),
                         "n_gt": float(n_gt_bin[k])})
            # reliability curve
            idx = np.digitize(c, CONF_BINS) - 1
            for b in range(len(CONF_BINS) - 1):
                mm = idx == b
                if mm.sum() < 20:
                    continue
                rel.append({"seed": seed, "size_bin": sb,
                            "conf_bin": float((CONF_BINS[b] + CONF_BINS[b + 1]) / 2),
                            "mean_conf": float(c[mm].mean()),
                            "precision": float(t[mm].mean()), "n": int(mm.sum())})
            # operating point
            for th in OP_THRESHOLDS:
                s = c >= th
                op.append({"seed": seed, "size_bin": sb, "threshold": th,
                           "n_kept": int(s.sum()),
                           "precision": float(t[s].mean()) if s.sum() else np.nan,
                           "recall": float(t[s].sum() / max(n_gt_bin[k], 1))})

    sb_df = pd.DataFrame(rows); sb_df.to_csv(FIG / "calib_size_bins.csv", index=False)
    pd.DataFrame(rel).to_csv(FIG / "calib_reliability.csv", index=False)
    op_df = pd.DataFrame(op); op_df.to_csv(FIG / "calib_operating_point.csv", index=False)

    print("\n=== confidence vs empirical precision, by predicted size ===")
    t1 = (sb_df.groupby("size_bin", observed=True)
          [["n_det", "mean_conf", "precision", "gap", "ece"]]
          .agg(["mean", "std"]).round(4))
    print(t1.to_string())

    print("\n=== operating point: recall by size bin ===")
    t2 = (op_df.pivot_table(index="size_bin", columns="threshold",
                            values="recall", observed=True).round(4))
    print(t2.to_string())
    print("\n=== operating point: precision by size bin ===")
    t3 = (op_df.pivot_table(index="size_bin", columns="threshold",
                            values="precision", observed=True).round(4))
    print(t3.to_string())

    g = sb_df.groupby("size_bin", observed=True).gap.mean()
    summary = {
        "exp": a.exp, "n_seeds": len(files),
        "gap_by_size": {k: round(float(v), 4) for k, v in g.items()},
        "gap_spread": round(float(g.max() - g.min()), 4),
        "recall_tiny_at_0.5": float(op_df.query("size_bin=='tiny' and threshold==0.5").recall.mean()),
        "recall_tiny_at_0.1": float(op_df.query("size_bin=='tiny' and threshold==0.1").recall.mean()),
        "recall_large_at_0.5": float(op_df.query("size_bin=='large' and threshold==0.5").recall.mean()),
    }
    json.dump(summary, open(FIG / "calib_summary.json", "w"), indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
