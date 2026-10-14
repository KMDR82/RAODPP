#!/usr/bin/env python3
"""RAOD++ F6 / protocol v1.3: gallery-mean centering + re-retrieval + quality tables."""
import time
from pathlib import Path
import numpy as np, pandas as pd, faiss, umap

R = Path.home()/"RAODpp"
E, C, FIG = R/"dataset"/"embeddings", R/"dataset"/"crops", R/"output"/"figdata"
SEEDS = [42, 123, 2024, 7, 3407, 11, 55, 101, 777, 12345]
KMAX, K = 10, 5

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def unit(a):
    return a/np.linalg.norm(a, axis=1, keepdims=True)

def check(a, tag):                       # protocol v1.3 A14
    n = np.linalg.norm(a[:2000], axis=1)
    assert n.mean() > 0.5, f"{tag}: embeddings look empty (norm={n.mean():.4f})"

gal = pd.read_csv(C/"manifest_gallery_visdrone.csv")
gc = gal.class_id.values
G = np.load(E/"gallery_visdrone.npy").astype(np.float32)
check(G, "gallery")

# ---- frozen center vector, gallery only ----
mu = G.mean(0, keepdims=True)
np.save(E/"center_vector.npy", mu)
Gc = unit(G - mu)
index = faiss.IndexFlatIP(Gc.shape[1]); index.add(Gc)
faiss.write_index(index, str(E/"faiss_gallery_centered.index"))
log(f"center vector saved; index rebuilt ({index.ntotal} vectors)")

# ---- retrieval per seed ----
rows = []
for s in SEEDS:
    rid = f"E0_visdrone_s{s}"
    man = pd.read_csv(C/f"manifest_queries_{rid}.csv")
    Q = np.asarray(np.load(E/f"queries_{rid}.npy", mmap_mode="r"), dtype=np.float32)
    check(Q, rid)
    Qc = unit(Q - mu)
    D, I = [], []
    for i in range(0, len(Qc), 50_000):
        d, ii = index.search(Qc[i:i+50_000], KMAX)
        D.append(d.astype(np.float16)); I.append(ii.astype(np.int32))
    D, I = np.concatenate(D), np.concatenate(I)
    np.savez_compressed(E/f"retrieval_{rid}_centered.npz", D=D, I=I)

    q = man.class_id.values
    for k in [1, 3, 5, 10]:
        nb = gc[I[:, :k]]
        rows.append({"seed": s, "k": k,
                     "consistency_any": float((nb == q[:, None]).any(1).mean()),
                     "consistency_mean": float((nb == q[:, None]).mean()),
                     "mean_sim": float(D[:, :k].astype(np.float32).mean())})
    log(f"{rid}: cons@5={rows[-2]['consistency_any']:.4f}")

cons = pd.DataFrame(rows); cons.to_csv(FIG/"retrieval_quality_centered.csv", index=False)
log("consistency@k (centered)\n" +
    cons.groupby("k")[["consistency_any","consistency_mean","mean_sim"]]
        .agg(["mean","std"]).round(4).to_string())

# ---- per-class table with the corrected chance baseline (v1.3 A12) ----
share = gal.class_name.value_counts(normalize=True)
per = []
for s in SEEDS:
    rid = f"E0_visdrone_s{s}"
    man = pd.read_csv(C/f"manifest_queries_{rid}.csv")
    I = np.load(E/f"retrieval_{rid}_centered.npz")["I"]
    man["hit"] = (gc[I[:, :K]] == man.class_id.values[:, None]).any(1)
    g = man.groupby("class_name").hit.agg(["size","mean"]).rename(
        columns={"size":"queries","mean":"consistency"})
    g["seed"] = s
    per.append(g.reset_index())

pc = (pd.concat(per).groupby("class_name")
      .agg(queries=("queries","mean"), consistency=("consistency","mean"),
           consistency_std=("consistency","std")).round(4))
pc["gallery_crops"] = pc.index.map(gal.class_name.value_counts())
pc["gallery_share"] = pc.index.map(share).round(4)
pc["chance_at_5"] = (1 - (1 - pc.gallery_share)**K).round(4)
pc["ratio_vs_chance"] = (pc.consistency/pc.chance_at_5).round(2)
pc = pc.sort_values("gallery_share")
pc.to_csv(FIG/"retrieval_quality_per_class_centered.csv")
log("per-class @k=5, mean over 10 seeds\n" + pc.to_string())
log(f"macro consistency = {pc.consistency.mean():.4f}")

# ---- UMAP coordinates for figure F10 ----
try:
    rid = "E0_visdrone_s42"
    man = pd.read_csv(C/f"manifest_queries_{rid}.csv")
    Q = np.load(E/f"queries_{rid}.npy", mmap_mode="r")
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(Q), min(30_000, len(Q)), replace=False))
    sub = unit(np.asarray(Q[idx], dtype=np.float32) - mu)
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                   random_state=42).fit_transform(sub)
    pd.DataFrame({"x": xy[:,0], "y": xy[:,1],
                  "class_id": man.class_id.values[idx],
                  "class_name": man.class_name.values[idx],
                  "conf": man.conf.values[idx],
                  "box_area": man.box_area.values[idx]}
                 ).to_csv(FIG/"umap_coords.csv", index=False)
    log(f"umap_coords.csv written ({len(idx)} points)")
except Exception as e:
    log(f"UMAP skipped: {type(e).__name__}: {e}")

log("DONE")
