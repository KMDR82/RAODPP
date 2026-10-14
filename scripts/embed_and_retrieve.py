#!/usr/bin/env python3
"""
embed_and_retrieve.py — RAOD++ F6 retrieval stage, crash-resilient.

Replaces notebook 08 cells 7-17. Run detached so a VS Code disconnect cannot kill it:

    cd ~/RAODpp && source .venv/bin/activate
    nohup python scripts/embed_and_retrieve.py > output/logs/embed.log 2>&1 &
    tail -f output/logs/embed.log

Resumable at every level:
  * embeddings are written to a memory-mapped .npy in blocks, with a .progress file,
    so an interrupted run resumes from the last completed block
  * a finished .npy is never recomputed
  * retrieval results are cached per seed

Memory-safe: never holds more than one block of embeddings in RAM.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from PIL import Image
from torch.utils.data import Dataset, DataLoader

CONFIG = {
    "PROJECT_ROOT": Path.home() / "RAODpp",
    "SEEDS": [42, 123, 2024, 7, 3407, 11, 55, 101, 777, 12345],
    "EXP": "E0", "DATASET": "visdrone",
    "CLIP_ARCH": "ViT-B-32-quickgelu",     # openai weights use QuickGELU
    "CLIP_PRETRAIN": "openai",
    "BATCH": 256,
    "WORKERS": 4,
    "FP16": True,
    "BLOCK": 50_000,                       # embeddings flushed to disk every N crops
    "K_EVAL": [1, 3, 5, 10],
    "K": 5,
    "UMAP_SAMPLE": 30_000,
    "DIM": 512,
}

ROOT = CONFIG["PROJECT_ROOT"]
OUT = ROOT / "output"; FIGDATA = OUT / "figdata"; FIGDATA.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)
CROPS = ROOT / "dataset" / "crops"
EMB = ROOT / "dataset" / "embeddings"; EMB.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- model
def build_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        CONFIG["CLIP_ARCH"], pretrained=CONFIG["CLIP_PRETRAIN"])
    model = model.to("cuda").eval()
    if CONFIG["FP16"]:
        model = model.half()
    return model, preprocess


class CropSet(Dataset):
    def __init__(self, files, root, preprocess):
        self.files, self.root, self.pp = files, Path(root), preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.pp(Image.open(self.root / self.files[i]).convert("RGB"))


@torch.no_grad()
def embed(files, root, out_npy, tag, model, preprocess):
    """Block-wise, resumable embedding into a memory-mapped array."""
    out_npy = Path(out_npy)
    prog_file = out_npy.with_suffix(".progress")
    n = len(files)

    if out_npy.exists() and not prog_file.exists():
        log(f"{tag}: cached ({np.load(out_npy, mmap_mode='r').shape})")
        return out_npy

    start = json.load(open(prog_file))["done"] if prog_file.exists() else 0
    if start:
        log(f"{tag}: resuming from {start:,}/{n:,}")
        arr = np.load(out_npy, mmap_mode="r+")
    else:
        arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float16,
                                        shape=(n, CONFIG["DIM"]))

    t0 = time.time()
    for b0 in range(start, n, CONFIG["BLOCK"]):
        b1 = min(b0 + CONFIG["BLOCK"], n)
        ds = CropSet(files[b0:b1], root, preprocess)
        try:
            dl = DataLoader(ds, batch_size=CONFIG["BATCH"],
                            num_workers=CONFIG["WORKERS"], pin_memory=True)
            it = iter(dl); first = next(it)
        except Exception as e:
            log(f"{tag}: workers failed ({type(e).__name__}), using num_workers=0")
            dl = DataLoader(ds, batch_size=CONFIG["BATCH"], num_workers=0)
            it = iter(dl); first = next(it)

        pos = b0
        for x in [first] + list(it):
            x = x.to("cuda", non_blocking=True)
            if CONFIG["FP16"]:
                x = x.half()
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            arr[pos:pos + len(x)] = f.cpu().numpy().astype(np.float16)
            pos += len(x)

        arr.flush()
        json.dump({"done": b1, "total": n}, open(prog_file, "w"))
        rate = (b1 - start) / max(1e-9, time.time() - t0)
        log(f"{tag}: {b1:,}/{n:,}  {rate:.0f} img/s  eta {(n-b1)/max(rate,1)/60:.1f} min")

    del arr
    prog_file.unlink(missing_ok=True)
    log(f"{tag}: done in {(time.time()-t0)/60:.1f} min")
    return out_npy


# --------------------------------------------------------------------------- main
def main():
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    assert torch.cuda.is_available(), "CUDA unavailable"
    log(f"GPU: {torch.cuda.get_device_name(0)}")

    model, preprocess = build_model()
    log(f"model: {CONFIG['CLIP_ARCH']} / {CONFIG['CLIP_PRETRAIN']}")

    # 1) gallery
    gal_man = pd.read_csv(CROPS / "manifest_gallery_visdrone.csv")
    gal_npy = embed(gal_man.crop.tolist(), CROPS / "gallery_visdrone",
                    EMB / "gallery_visdrone.npy", "gallery", model, preprocess)
    G = np.load(gal_npy).astype(np.float32)
    log(f"gallery embeddings {G.shape}, norm check {np.linalg.norm(G[0]):.4f}")

    # 2) index
    import faiss
    index = faiss.IndexFlatIP(G.shape[1])
    index.add(G)
    faiss.write_index(index, str(EMB / "faiss_gallery_visdrone.index"))
    log(f"FAISS IndexFlatIP built: {index.ntotal} vectors (exact search)")

    # 3) queries + retrieval
    KMAX = max(CONFIG["K_EVAL"])
    summary = []
    for seed in CONFIG["SEEDS"]:
        rid = f"{CONFIG['EXP']}_{CONFIG['DATASET']}_s{seed}"
        man = pd.read_csv(CROPS / f"manifest_queries_{rid}.csv")
        npy = embed(man.crop.tolist(), CROPS / f"queries_{rid}",
                    EMB / f"queries_{rid}.npy", f"q_s{seed}", model, preprocess)

        ret = EMB / f"retrieval_{rid}.npz"
        if ret.exists():
            z = np.load(ret); D, I = z["D"], z["I"]
            log(f"{rid}: retrieval cached {I.shape}")
        else:
            Q = np.load(npy).astype(np.float32)
            D_all, I_all = [], []
            for i in range(0, len(Q), 100_000):          # search in chunks
                d, ii = index.search(Q[i:i+100_000], KMAX)
                D_all.append(d.astype(np.float16)); I_all.append(ii.astype(np.int32))
            D = np.concatenate(D_all); I = np.concatenate(I_all)
            np.savez_compressed(ret, D=D, I=I)
            del Q, D_all, I_all
            log(f"{rid}: retrieved {I.shape}, mean top1 sim {D[:,0].mean():.4f}")
        summary.append({"seed": seed, "n_queries": len(man),
                        "mean_sim_top1": float(np.mean(D[:, 0].astype(np.float32))),
                        "mean_sim_topk": float(np.mean(D[:, :CONFIG["K"]].astype(np.float32)))})

    pd.DataFrame(summary).to_csv(FIGDATA / "retrieval_similarity_summary.csv", index=False)
    log("similarity summary written")

    # 4) class-consistency@k
    gal_cls = gal_man.class_id.values
    p_gal = gal_man.class_id.value_counts(normalize=True)
    baseline = float((p_gal ** 2).sum())

    rows = []
    for seed in CONFIG["SEEDS"]:
        rid = f"{CONFIG['EXP']}_{CONFIG['DATASET']}_s{seed}"
        man = pd.read_csv(CROPS / f"manifest_queries_{rid}.csv")
        I = np.load(EMB / f"retrieval_{rid}.npz")["I"]
        q = man.class_id.values
        for k in CONFIG["K_EVAL"]:
            nb = gal_cls[I[:, :k]]
            rows.append({"seed": seed, "k": k,
                         "consistency_any": float((nb == q[:, None]).any(axis=1).mean()),
                         "consistency_mean": float((nb == q[:, None]).mean())})
    cons = pd.DataFrame(rows)
    cons.to_csv(FIGDATA / "retrieval_quality.csv", index=False)
    log("class-consistency@k over 10 seeds\n" +
        cons.groupby("k")[["consistency_any", "consistency_mean"]]
            .agg(["mean", "std"]).round(4).to_string())
    log(f"random-match baseline (gallery prior): {baseline:.4f}")

    # per-class breakdown, reference seed
    rid = f"{CONFIG['EXP']}_{CONFIG['DATASET']}_s42"
    man = pd.read_csv(CROPS / f"manifest_queries_{rid}.csv")
    I = np.load(EMB / f"retrieval_{rid}.npz")["I"]
    nb = gal_cls[I[:, :CONFIG["K"]]]
    man["hit"] = (nb == man.class_id.values[:, None]).any(axis=1)
    per_cls = (man.groupby("class_name")
               .agg(queries=("hit", "size"), consistency=("hit", "mean")).round(4)
               .sort_values("consistency"))
    per_cls["gallery_crops"] = per_cls.index.map(gal_man.class_name.value_counts())
    per_cls.to_csv(FIGDATA / "retrieval_quality_per_class.csv")
    log("per-class consistency @k=5 (seed 42)\n" + per_cls.to_string())

    # 5) UMAP coordinates for figure F10
    try:
        import umap
        Q = np.load(EMB / f"queries_{rid}.npy", mmap_mode="r")
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(Q), size=min(CONFIG["UMAP_SAMPLE"], len(Q)),
                                 replace=False))
        sub = np.asarray(Q[idx], dtype=np.float32)
        xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                       random_state=42).fit_transform(sub)
        pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                      "class_id": man.class_id.values[idx],
                      "class_name": man.class_name.values[idx],
                      "conf": man.conf.values[idx],
                      "box_area": man.box_area.values[idx]}
                     ).to_csv(FIGDATA / "umap_coords.csv", index=False)
        log(f"umap_coords.csv written ({len(idx)} points)")
    except Exception as e:
        log(f"UMAP skipped: {type(e).__name__}: {e}")

    log("retrieval stage COMPLETE")


if __name__ == "__main__":
    main()
