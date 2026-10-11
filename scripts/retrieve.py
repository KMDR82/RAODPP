#!/usr/bin/env python3
"""RAOD++ F6: query embedding + retrieval + quality. Resumable, memory-safe."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, faiss, open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

R = Path.home()/"RAODpp"
CROPS, EMB, FIG = R/"dataset"/"crops", R/"dataset"/"embeddings", R/"output"/"figdata"
SEEDS = [42, 123, 2024, 7, 3407, 11, 55, 101, 777, 12345]
BLOCK, BATCH, WORKERS, KMAX = 25_000, 256, 0, 10

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

model, _, pp = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
model = model.to("cuda").eval().half()

class CS(Dataset):
    def __init__(s, f, r): s.f, s.r = f, Path(r)
    def __len__(s): return len(s.f)
    def __getitem__(s, i): return pp(Image.open(s.r/s.f[i]).convert("RGB"))

@torch.no_grad()
def embed(files, root, out, tag):
    out = Path(out); prog = out.with_suffix(".prog"); n = len(files)
    if out.exists() and not prog.exists():
        log(f"{tag}: cached"); return out
    start = json.load(open(prog))["done"] if prog.exists() else 0
    arr = (np.load(out, mmap_mode="r+") if start else
           np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(n, 512)))
    json.dump({"done": start}, open(prog, "w"))
    t0 = time.time()
    for b0 in range(start, n, BLOCK):
        b1 = min(b0+BLOCK, n); pos = b0
        dl = DataLoader(CS(files[b0:b1], root), batch_size=BATCH, num_workers=WORKERS)
        for x in dl:
            f = model.encode_image(x.to("cuda").half())
            f = f/f.norm(dim=-1, keepdim=True)
            arr[pos:pos+len(x)] = f.cpu().numpy().astype(np.float16); pos += len(x)
        arr.flush(); json.dump({"done": b1}, open(prog, "w"))
        r = (b1-start)/max(1e-9, time.time()-t0)
        log(f"{tag}: {b1:,}/{n:,} {r:.0f} img/s eta {(n-b1)/max(r,1)/60:.1f}min")
    del arr; prog.unlink(missing_ok=True); return out

gal = pd.read_csv(CROPS/"manifest_gallery_visdrone.csv")
index = faiss.read_index(str(EMB/"faiss_gallery_visdrone.index"))
log(f"index: {index.ntotal} vectors")

for s in SEEDS:
    rid = f"E0_visdrone_s{s}"
    man = pd.read_csv(CROPS/f"manifest_queries_{rid}.csv")
    npy = embed(man.crop.tolist(), CROPS/f"queries_{rid}", EMB/f"queries_{rid}.npy", f"s{s}")
    ret = EMB/f"retrieval_{rid}.npz"
    if not ret.exists():
        Q = np.load(npy, mmap_mode="r"); D, I = [], []
        for i in range(0, len(Q), 50_000):
            d, ii = index.search(np.asarray(Q[i:i+50_000], dtype=np.float32), KMAX)
            D.append(d.astype(np.float16)); I.append(ii.astype(np.int32))
        np.savez_compressed(ret, D=np.concatenate(D), I=np.concatenate(I))
        log(f"{rid}: retrieval saved")

gc = gal.class_id.values
base = float((gal.class_id.value_counts(normalize=True)**2).sum())
rows = []
for s in SEEDS:
    rid = f"E0_visdrone_s{s}"
    man = pd.read_csv(CROPS/f"manifest_queries_{rid}.csv")
    z = np.load(EMB/f"retrieval_{rid}.npz"); I, D = z["I"], z["D"]
    q = man.class_id.values
    for k in [1, 3, 5, 10]:
        nb = gc[I[:, :k]]
        rows.append({"seed": s, "k": k,
                     "consistency_any": float((nb == q[:, None]).any(1).mean()),
                     "consistency_mean": float((nb == q[:, None]).mean()),
                     "mean_sim": float(D[:, :k].astype(np.float32).mean())})
cons = pd.DataFrame(rows); cons.to_csv(FIG/"retrieval_quality.csv", index=False)
log("consistency@k\n" + cons.groupby("k")[["consistency_any","consistency_mean","mean_sim"]]
    .agg(["mean","std"]).round(4).to_string())
log(f"random baseline: {base:.4f}")

rid = "E0_visdrone_s42"
man = pd.read_csv(CROPS/f"manifest_queries_{rid}.csv")
I = np.load(EMB/f"retrieval_{rid}.npz")["I"]
man["hit"] = (gc[I[:, :5]] == man.class_id.values[:, None]).any(1)
pc = (man.groupby("class_name").agg(queries=("hit","size"), consistency=("hit","mean"))
      .round(4).sort_values("consistency"))
pc["gallery_crops"] = pc.index.map(gal.class_name.value_counts())
pc.to_csv(FIG/"retrieval_quality_per_class.csv")
log("per-class @k=5\n" + pc.to_string())
log("DONE")
