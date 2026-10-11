#!/usr/bin/env python3
"""
upaste.py — RAOD++ enrichment stage (design spec approved 2026-08-15).

Builds the enriched training sets for E2-E5. Nothing is trained here.

    python scripts/upaste.py --mode gate            # once: score gallery crops
    python scripts/upaste.py --mode E5 --seeds 42   # build one enriched set
    python scripts/upaste.py --mode E5              # all 10 seeds

Modes
  gate : CLIP text-image agreement over gallery crops -> gallery_gate.csv
  E2   : self-training pseudo-labels only (sigma>=0.60), images untouched
  E3   : U-Paste placement with randomly chosen gallery crops   (control)
  E4   : U-Paste with kNN retrieval, no gate, uniform selection (basic RAOD)
  E5   : full RAOD++  (kNN + gate + similarity-weighted sampling)

Output: dataset/enriched/<mode>_<run_id>/{images,labels} + a data yaml + an audit csv.
Image count is unchanged, so the +20-epoch budget stays comparable across conditions.
"""
import argparse, json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFilter

R = Path.home() / "RAODpp"
EMB, CROPS = R / "dataset" / "embeddings", R / "dataset" / "crops"
FIG, FIGDATA = R / "fig", R / "output" / "figdata"
ENR = R / "dataset" / "enriched"
SEEDS_ALL = [42, 123, 2024, 7, 3407, 11, 55, 101, 777, 12345]

P_PER_IMAGE = 5          # paste budget per image (overridable: --budget)
TAG = ""                  # suffix for the enriched set name (--tag)
ALPHA = 0.20             # context margin used when crops were written
SCALE_JITTER = 0.15
GAMMA = 2.0              # similarity weighting exponent
MAX_TRIES = 30
IOU_MAX = 0.05
FEATHER = 3
SIGMA_PSEUDO = 0.60      # E2 pseudo-label acceptance
GATE_MARGIN = 0.01       # protocol v1.4 A16: reject only a confident contradiction

# --- context-matching constraints (visual audit fix, 2026-08-15) ---
RADIUS_MIN, RADIUS_MAX = 0.8, 2.0    # annulus around the anchor, in anchor diagonals
MAX_UPSCALE = 1.5                    # never blow a crop up more than this
SURFACE_MEAN_TOL = 25.0              # per-channel mean difference vs the anchor surface
SURFACE_STD_TOL = 20.0               # texture difference vs the anchor surface

# --- compositing constraints (visual audit round 2, 2026-08-15) ---
TIGHT_PASTE = True                   # paste the object region only, not the context margin
FEATHER_FRAC = 0.08                  # feather radius as a fraction of the paste size
BORDER_TOL = 35.0
MIN_SRC_PX = 16.0                    # source object smaller than this is too blurry to paste                    # crop border vs target background, per-channel mean

PROMPTS = ["an aerial photo of a {}", "a drone image of a {}",
           "a small {} seen from above"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_data_cfg():
    d = yaml.safe_load(open(R / "dataset" / "visdrone_raodpp.yaml"))
    base = Path(d["path"])
    return base / d["train"], Path(str(base / d["train"]).replace("/images", "/labels")), d["names"]


# ------------------------------------------------------------------ gate
def run_gate(names):
    import torch, open_clip
    gal = pd.read_csv(CROPS / "manifest_gallery_visdrone.csv")
    G = np.load(EMB / "gallery_visdrone.npy").astype(np.float32)
    G /= np.linalg.norm(G, axis=1, keepdims=True)

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu",
                                                        pretrained="openai")
    model = model.to("cuda").eval()
    tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    T = []
    with torch.no_grad():
        for cid in sorted(names):
            txt = tok([p.format(names[cid]) for p in PROMPTS]).to("cuda")
            e = model.encode_text(txt).float()
            e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
            T.append((e / e.norm()).cpu().numpy())
    T = np.stack(T)

    S = G @ T.T
    order = np.argsort(-S, axis=1)
    gal["text_class"] = order[:, 0]
    gal["text_margin"] = (S[np.arange(len(S)), order[:, 0]] -
                          S[np.arange(len(S)), order[:, 1]])
    gal["gate_pass"] = gal.text_class == gal.class_id
    gal.to_csv(FIGDATA / "gallery_gate.csv", index=False)

    rate = gal.gate_pass.mean()
    log(f"gate acceptance: {rate:.4f} ({gal.gate_pass.sum():,}/{len(gal):,})")
    log("per class\n" + gal.groupby("class_name").gate_pass
        .agg(["size", "mean"]).round(4).to_string())
    if rate < 0.40:
        log("WARNING: acceptance below 0.40 — margin rule should replace equality (spec §3)")
    return gal


# ------------------------------------------------------------------ geometry
def iou_1_to_n(box, arr):
    if len(arr) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], arr[:, 0]); y1 = np.maximum(box[1], arr[:, 1])
    x2 = np.minimum(box[2], arr[:, 2]); y2 = np.minimum(box[3], arr[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
    return inter / (a + b - inter + 1e-9)


def footprint(gbox, gsize):
    """Where the object sits inside its 224x224 crop, in normalised crop coords."""
    gx1, gy1, gx2, gy2 = gbox
    W, H = gsize
    gw, gh = gx2 - gx1, gy2 - gy1
    ex1, ey1 = max(0, gx1 - ALPHA * gw), max(0, gy1 - ALPHA * gh)
    ex2, ey2 = min(W, gx2 + ALPHA * gw), min(H, gy2 + ALPHA * gh)
    cw, ch = max(ex2 - ex1, 1e-6), max(ey2 - ey1, 1e-6)
    return ((gx1 - ex1) / cw, (gy1 - ey1) / ch, (gx2 - ex1) / cw, (gy2 - ey1) / ch)


def surface_ok(arr, px, py, Wp, Hp, ax, ay):
    """Reject a location whose surface does not look like the anchor's surface.

    Without this, 'no overlap with any object' pushes pastes onto roofs, trees and
    walls, because those are the only empty regions in a dense aerial scene.
    """
    H, W = arr.shape[:2]
    def stats(x, y):
        x0, y0 = int(np.clip(x, 0, W - Wp)), int(np.clip(y, 0, H - Hp))
        p = arr[y0:y0 + Hp, x0:x0 + Wp]
        if p.size == 0:
            return None
        return p.reshape(-1, 3).mean(0), p.reshape(-1, 3).mean(1).std()
    a = stats(px, py)
    if a is None:
        return False
    # the reference must be the anchor's BACKGROUND, not the anchor object itself:
    # sample four patches offset around it and take the median statistics
    off = 1.5 * max(Wp, Hp)
    refs = [stats(ax + dx * off - Wp / 2, ay + dy * off - Hp / 2)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    refs = [r for r in refs if r is not None]
    if not refs:
        return False
    m = np.median(np.stack([r[0] for r in refs]), axis=0)
    sd = float(np.median([r[1] for r in refs]))
    return (np.abs(a[0] - m).mean() <= SURFACE_MEAN_TOL and
            abs(a[1] - sd) <= SURFACE_STD_TOL)


def paste_one(img, arr, crop, obj_uv, anchor, gal_box, occupied, rng, blend=True):
    """Place one crop near a same-class anchor, at the anchor's scale.

    Returns (object_box, reason); reason is None on success.

    The context margin that helps retrieval hurts compositing: pasting it drags the
    gallery image's background along and leaves a visible rectangle. With TIGHT_PASTE
    only the object region of the crop is transferred.
    """
    u1, v1, u2, v2 = obj_uv
    aw, ah = anchor[2] - anchor[0], anchor[3] - anchor[1]
    j = 1.0 + rng.uniform(-SCALE_JITTER, SCALE_JITTER)
    tw, th = max(aw * j, 4), max(ah * j, 4)

    gw = max(gal_box[2] - gal_box[0], 1.0)
    gh = max(gal_box[3] - gal_box[1], 1.0)
    if min(gw, gh) < MIN_SRC_PX:                    # low-resolution source
        return None, "blurry"
    if tw > MAX_UPSCALE * gw:                       # would be an upscaled blur
        return None, "upscale"

    if TIGHT_PASTE:
        cw, ch = crop.size
        box = (int(u1 * cw), int(v1 * ch), int(np.ceil(u2 * cw)), int(np.ceil(v2 * ch)))
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            return None, "size"
        crop = crop.crop(box)
        Wp, Hp = int(round(tw)), int(round(th))     # crop == object, so paste == object box
        ru1 = rv1 = 0.0; ru2 = rv2 = 1.0
    else:
        ow, oh = max(u2 - u1, 1e-3), max(v2 - v1, 1e-3)
        Wp, Hp = int(round(tw / ow)), int(round(th / oh))
        ru1, rv1, ru2, rv2 = u1, v1, u2, v2

    if Wp < 6 or Hp < 6 or Wp > img.width // 2 or Hp > img.height // 2:
        return None, "size"

    # border statistics of the incoming crop, used for the seam check below
    ca = np.asarray(crop.resize((Wp, Hp), Image.BILINEAR), dtype=np.float32)
    ring = np.concatenate([ca[0], ca[-1], ca[:, 0], ca[:, -1]]).reshape(-1, 3).mean(0)

    ax, ay = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    diag = float(np.hypot(aw, ah))
    seam_hits = 0
    for _ in range(MAX_TRIES):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(RADIUS_MIN, RADIUS_MAX) * diag
        px = int(ax + rad * np.cos(ang) - Wp / 2)
        py = int(ay + rad * np.sin(ang) - Hp / 2)
        if px < 0 or py < 0 or px + Wp > img.width or py + Hp > img.height:
            continue
        obox = np.array([px + ru1 * Wp, py + rv1 * Hp, px + ru2 * Wp, py + rv2 * Hp])
        if len(occupied) and iou_1_to_n(obox, occupied).max() >= IOU_MAX:
            continue
        if not surface_ok(arr, px, py, Wp, Hp, ax, ay):
            continue

        bg = arr[py:py + Hp, px:px + Wp]
        if bg.size == 0:
            continue
        if np.abs(ring - bg.reshape(-1, 3).mean(0)).mean() > BORDER_TOL:
            seam_hits += 1                          # position-dependent: try elsewhere
            continue

        patch = Image.fromarray(np.clip(
            ca + (bg.reshape(-1, 3).mean(0) - ca.reshape(-1, 3).mean(0)), 0, 255
        ).astype(np.uint8)) if blend else crop.resize((Wp, Hp), Image.BILINEAR)

        if blend:
            f = max(2, int(FEATHER_FRAC * min(Wp, Hp)))
            mask = Image.new("L", (Wp, Hp), 255).filter(ImageFilter.GaussianBlur(f))
            img.paste(patch, (px, py), mask)
        else:
            img.paste(patch, (px, py))
        return obox, None
    return None, ("seam" if seam_hits else "no_space")


# ------------------------------------------------------------------ build
def build(mode, seed, names, train_img, train_lbl, gate_df, blend=True):
    rid = f"E0_visdrone_s{seed}"
    out = ENR / f"{mode}{TAG}_{rid}"
    if (out / "done.json").exists():
        log(f"{mode} s{seed}: already built"); return
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed); nprng = np.random.default_rng(seed)

    det = pd.read_csv(FIGDATA / f"train_predictions_{rid}.csv")
    band = pd.read_csv(FIGDATA / "tau_band_per_seed.csv")
    band = band[band.seed == seed].set_index("class_id")
    det["tau_high"] = det.class_id.map(band.tau_high).fillna(0.7)
    conf_det = det[det.conf >= det.tau_high]
    conf_by_img = {k: (v[["x1", "y1", "x2", "y2"]].values, v["class_id"].values)
                   for k, v in conf_det.groupby("image")}

    audit = {"attempted": 0, "pasted": 0, "no_space": 0, "gate_blocked": 0,
             "no_anchor": 0, "upscale": 0, "size": 0, "seam": 0, "blurry": 0, "pseudo_added": 0}
    per_class = {c: 0 for c in names}
    sims = []

    if mode == "E2":
        ps = det[det.conf >= SIGMA_PSEUDO]
        ps_by_img = {k: v for k, v in ps.groupby("image")}
    else:
        man = pd.read_csv(CROPS / f"manifest_queries_{rid}.csv")
        z = np.load(EMB / f"retrieval_{rid}_centered.npz")
        D, I = z["D"].astype(np.float32), z["I"]
        gal = pd.read_csv(CROPS / "manifest_gallery_visdrone.csv")
        # E3 control: random choice is restricted to gallery crops that could pass the
        # same size/blur filters as a kNN neighbour, so E3 and E4/E5 differ only in
        # HOW the neighbour is chosen, not in how many survive compositing.
        _gw = (gal.x2 - gal.x1).values
        _gh = (gal.y2 - gal.y1).values
        e3_pool = np.where(np.minimum(_gw, _gh) >= MIN_SRC_PX)[0]
        gsize = {}
        gal_dir = R / "dataset" / "visdrone_gallery" / "VisDrone2019-DET-test-dev" / "images"
        for im in gal.source_image.unique():
            gsize[im] = Image.open(gal_dir / im).size
        gate_ok = set(gate_df[gate_df.gate_pass |
                              (gate_df.text_margin < GATE_MARGIN)].index) if mode == "E5" else None
        man["row"] = np.arange(len(man))
        q_by_img = {k: v for k, v in man.groupby("source_image")}

    imgs = sorted(train_img.glob("*.jpg"))
    for n, p in enumerate(imgs, 1):
        lines = []
        lp = train_lbl / (p.stem + ".txt")
        if lp.exists():
            lines = [l.strip() for l in open(lp) if l.strip()]
        img = Image.open(p).convert("RGB")
        W, H = img.size
        _sp = [l.split() for l in lines]
        gt = np.array([[(float(f[1]) - float(f[3]) / 2) * W, (float(f[2]) - float(f[4]) / 2) * H,
                        (float(f[1]) + float(f[3]) / 2) * W, (float(f[2]) + float(f[4]) / 2) * H]
                       for f in _sp]) if lines else np.zeros((0, 4))
        gt_cls = np.array([int(f[0]) for f in _sp]) if lines else np.zeros(0, dtype=int)

        if mode == "E2":
            d = ps_by_img.get(p.name)
            if d is not None:
                for r in d.itertuples():
                    b = np.array([r.x1, r.y1, r.x2, r.y2])
                    if len(gt) and iou_1_to_n(b, gt).max() >= 0.5:
                        continue
                    lines.append(f"{r.class_id} {(r.x1+r.x2)/2/W:.6f} {(r.y1+r.y2)/2/H:.6f} "
                                 f"{(r.x2-r.x1)/W:.6f} {(r.y2-r.y1)/H:.6f}")
                    audit["pseudo_added"] += 1
            (out / "labels" / f"{p.stem}.txt").write_text("\n".join(lines))
            dst = out / "images" / p.name
            if not dst.exists():
                dst.symlink_to(p)
        else:
            q = q_by_img.get(p.name)
            occupied = [gt] if len(gt) else []
            cd_box, cd_cls = conf_by_img.get(p.name, (np.zeros((0, 4)), np.zeros(0, dtype=int)))
            if len(cd_box):
                occupied.append(cd_box)
            occ = np.concatenate(occupied) if occupied else np.zeros((0, 4))
            arr = np.asarray(img, dtype=np.float32)

            if q is not None and len(q):
                cand = []
                for r in q.itertuples():
                    if mode == "E3":
                        gi = int(e3_pool[nprng.integers(0, len(e3_pool))])
                        sim = float(D[r.row, 0])
                    else:
                        gi, sim = int(I[r.row, 0]), float(D[r.row, 0])
                        if mode == "E5" and gi not in gate_ok:
                            hit = False
                            for j in range(1, I.shape[1]):
                                if int(I[r.row, j]) in gate_ok:
                                    gi, sim, hit = int(I[r.row, j]), float(D[r.row, j]), True
                                    break
                            if not hit:
                                audit["gate_blocked"] += 1; continue
                    cand.append((r.x1, r.y1, r.x2, r.y2, gi, sim, int(r.class_id)))

                if len(cand) > P_PER_IMAGE:
                    if mode == "E5":
                        w = np.array([max(c[5], 1e-6) ** GAMMA for c in cand])
                        pick = nprng.choice(len(cand), P_PER_IMAGE, replace=False, p=w / w.sum())
                    else:
                        pick = nprng.choice(len(cand), P_PER_IMAGE, replace=False)
                    cand = [cand[i] for i in pick]

                for (x1, y1, x2, y2, gi, sim, qcls) in cand:
                    audit["attempted"] += 1
                    g = gal.iloc[gi]
                    c = int(g.class_id)

                    # context anchor: an existing instance of the same class in this image
                    anchors = []
                    if len(gt) and (gt_cls == c).any():
                        anchors.append(gt[gt_cls == c])
                    if len(cd_box) and (cd_cls == c).any():
                        anchors.append(cd_box[cd_cls == c])
                    if int(qcls) == c:
                        anchors.append(np.array([[x1, y1, x2, y2]]))
                    if not anchors:
                        audit["no_anchor"] += 1; continue
                    A = np.concatenate(anchors)
                    anchor = A[rng.randrange(len(A))]

                    uv = footprint((g.x1, g.y1, g.x2, g.y2), gsize[g.source_image])
                    crop = Image.open(CROPS / "gallery_visdrone" / g.crop).convert("RGB")
                    ob, why = paste_one(img, arr, crop, uv, anchor,
                                        (g.x1, g.y1, g.x2, g.y2), occ, rng, blend)
                    if ob is None:
                        audit[why] = audit.get(why, 0) + 1; continue
                    occ = np.vstack([occ, ob[None, :]])
                    lines.append(f"{int(g.class_id)} {(ob[0]+ob[2])/2/W:.6f} "
                                 f"{(ob[1]+ob[3])/2/H:.6f} {(ob[2]-ob[0])/W:.6f} "
                                 f"{(ob[3]-ob[1])/H:.6f}")
                    audit["pasted"] += 1
                    per_class[int(g.class_id)] = per_class.get(int(g.class_id), 0) + 1
                    sims.append(sim)

            img.save(out / "images" / p.name, quality=92)
            (out / "labels" / f"{p.stem}.txt").write_text("\n".join(lines))

        if n % 500 == 0:
            log(f"  {mode} s{seed}: {n}/{len(imgs)} pasted={audit['pasted']:,}")

    (out / "data.yaml").write_text(
        f"# RAOD++ enriched set: {mode}, seed {seed}\npath: {out}\n"
        f"train: images\nval: {R/'dataset'/'visdrone'/'VisDrone2019-DET-val'/'images'}\n"
        "names:\n" + "\n".join(f"  {i}: {n_}" for i, n_ in sorted(names.items())) + "\n")

    audit.update({"mode": mode, "seed": seed, "images": len(imgs),
                  "mean_sim": float(np.mean(sims)) if sims else None,
                  "per_class": {names[c]: v for c, v in per_class.items() if v}})
    json.dump(audit, open(out / "done.json", "w"), indent=2)
    row = {k: v for k, v in audit.items() if k != "per_class"}
    f = FIGDATA / "upaste_audit.csv"
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
    log(f"{mode} s{seed}: pasted={audit['pasted']:,} attempted={audit['attempted']:,} "
        f"no_space={audit['no_space']:,} gate_blocked={audit['gate_blocked']:,} "
        f"pseudo={audit['pseudo_added']:,}")


def save_samples(mode, seed, k=20):
    """Visual audit: copy k enriched images with boxes drawn, for inspection."""
    from PIL import ImageDraw
    src = ENR / f"{mode}_E0_visdrone_s{seed}"
    dst = FIG / f"upaste_samples_{mode}_s{seed}"; dst.mkdir(parents=True, exist_ok=True)
    for p in sorted((src / "images").glob("*.jpg"))[:k]:
        im = Image.open(p).convert("RGB"); d = ImageDraw.Draw(im)
        W, H = im.size
        for l in open(src / "labels" / f"{p.stem}.txt"):
            f = l.split()
            if len(f) != 5: continue
            cx, cy, w, h = map(float, f[1:])
            d.rectangle([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H], outline=(255, 0, 0), width=2)
        im.save(dst / p.name, quality=88)
    log(f"visual audit written: {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gate", "E2", "E3", "E4", "E5"])
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS_ALL)
    ap.add_argument("--no-blend", action="store_true")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--budget", type=int, default=P_PER_IMAGE)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    P_PER_IMAGE = a.budget
    TAG = a.tag
    train_img, train_lbl, names = load_data_cfg()
    if a.mode == "gate":
        run_gate(names); raise SystemExit(0)

    gate_df = None
    if a.mode == "E5":
        gp = FIGDATA / "gallery_gate.csv"
        assert gp.exists(), "run --mode gate first"
        gate_df = pd.read_csv(gp)

    for s in a.seeds:
        build(a.mode, s, names, train_img, train_lbl, gate_df, blend=not a.no_blend)
    if a.samples:
        save_samples(a.mode, a.seeds[0], a.samples)
    log("DONE")
