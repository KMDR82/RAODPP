#!/usr/bin/env python3
"""
gate_calibration.py — choose the cross-modal gate rule from evidence.

The equality rule (accept only if CLIP's text-predicted class equals the proposer's)
rejects 96% of motor crops and 88% of bicycle crops. Before accepting or discarding the
gate, we measure what it actually does to label quality.

Validation set: query crops from the training split, where ground truth IS available.
A crop's proposer label is 'correct' if the crop's box matches a GT box of the same class
at IoU >= 0.5. We then ask, for each candidate rule: how many crops survive, and is the
surviving set cleaner than the unfiltered one?

Caveat recorded in the output: query crops come from the uncertainty band while gallery
crops come from conf >= 0.25, so this is a proxy distribution, not the gallery itself.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

R = Path.home() / "RAODpp"
EMB, CROPS, FIGDATA = R / "dataset" / "embeddings", R / "dataset" / "crops", R / "output" / "figdata"
SEED = 42
PROMPTS = ["an aerial photo of a {}", "a drone image of a {}", "a small {} seen from above"]
MARGINS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def iou_1_to_n(box, arr):
    if len(arr) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], arr[:, 0]); y1 = np.maximum(box[1], arr[:, 1])
    x2 = np.minimum(box[2], arr[:, 2]); y2 = np.minimum(box[3], arr[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
    return inter / (a + b - inter + 1e-9)


d = yaml.safe_load(open(R / "dataset" / "visdrone_raodpp.yaml"))
base = Path(d["path"]); names = d["names"]
train_img = base / d["train"]
train_lbl = Path(str(train_img).replace("/images", "/labels"))

rid = f"E0_visdrone_s{SEED}"
man = pd.read_csv(CROPS / f"manifest_queries_{rid}.csv")
Q = np.asarray(np.load(EMB / f"queries_{rid}.npy", mmap_mode="r"), dtype=np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
log(f"query crops: {len(man):,}")

# ---- ground-truth match: is the proposer label correct? ----
correct = np.full(len(man), -1, dtype=np.int8)     # -1 = no GT overlap (background)
for img, g in man.groupby("source_image"):
    lp = train_lbl / (Path(img).stem + ".txt")
    if not lp.exists():
        continue
    W, H = Image.open(train_img / img).size
    gt, gc = [], []
    for line in open(lp):
        f = line.split()
        if len(f) != 5:
            continue
        c = int(f[0]); cx, cy, bw, bh = map(float, f[1:])
        gt.append([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H])
        gc.append(c)
    if not gt:
        continue
    gt = np.array(gt); gc = np.array(gc)
    for r in g.itertuples():
        v = iou_1_to_n(np.array([r.x1, r.y1, r.x2, r.y2]), gt)
        j = int(v.argmax())
        if v[j] >= 0.5:
            correct[r.Index] = int(gc[j] == r.class_id)

man["correct"] = correct
val = man[man.correct >= 0].copy()
log(f"crops with a GT match: {len(val):,}  |  proposer accuracy: {val.correct.mean():.4f}")

# ---- CLIP text embeddings ----
import open_clip
model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
model = model.to("cuda").eval()
tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
T = []
with torch.no_grad():
    for cid in sorted(names):
        e = model.encode_text(tok([p.format(names[cid]) for p in PROMPTS]).to("cuda")).float()
        e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
        T.append((e / e.norm()).cpu().numpy())
T = np.stack(T)

S = Q[val.index.values] @ T.T
order = np.argsort(-S, axis=1)
val["text_class"] = order[:, 0]
val["text_margin"] = S[np.arange(len(S)), order[:, 0]] - S[np.arange(len(S)), order[:, 1]]
val["agree"] = val.text_class == val.class_id

# ---- rule comparison ----
rows = [{"rule": "no gate", "kept_frac": 1.0, "precision_kept": val.correct.mean(),
         "gain": 0.0, "n_kept": len(val)}]
k = val.agree
rows.append({"rule": "equality", "kept_frac": k.mean(),
             "precision_kept": val.correct[k].mean(),
             "gain": val.correct[k].mean() - val.correct.mean(), "n_kept": int(k.sum())})
for m in MARGINS:
    k = val.agree | (val.text_margin < m)      # reject only a confident contradiction
    rows.append({"rule": f"contradiction m>={m}", "kept_frac": k.mean(),
                 "precision_kept": val.correct[k].mean(),
                 "gain": val.correct[k].mean() - val.correct.mean(), "n_kept": int(k.sum())})

res = pd.DataFrame(rows).round(4)
res.to_csv(FIGDATA / "gate_rule_calibration.csv", index=False)
log("gate rule comparison (higher precision_kept with high kept_frac is better)\n" +
    res.to_string(index=False))

# ---- where does CLIP disagree, and what does it say instead? ----
dis = val[~val.agree]
conf = (dis.groupby([dis.class_id.map(names), dis.text_class.map(names)]).size()
        .rename("n").reset_index().sort_values("n", ascending=False).head(12))
conf.columns = ["proposer", "clip_text", "n"]
conf.to_csv(FIGDATA / "gate_disagreement_top.csv", index=False)
log("most common disagreements\n" + conf.to_string(index=False))

per = (val.groupby(val.class_id.map(names))
       .agg(n=("correct", "size"), proposer_acc=("correct", "mean"),
            agree_rate=("agree", "mean")).round(4))
per.to_csv(FIGDATA / "gate_per_class_validation.csv")
log("per class\n" + per.to_string())
log("DONE")
