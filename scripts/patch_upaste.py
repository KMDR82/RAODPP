#!/usr/bin/env python3
"""
patch_upaste.py — applies the visual-audit fixes to scripts/upaste.py in place.

Run once:   python scripts/patch_upaste.py
Idempotent: running it twice reports "already applied" instead of double-patching.

Fixes (visual audit 2026-08-15 + protocol v1.4 A16):
  1. gate rule -> reject only a confident contradiction (margin 0.01)
  2. placement anchored on a same-class instance already present in the image
  3. surface check: candidate location must look like the anchor's surroundings
  4. no upscaling a gallery crop beyond 1.5x its original box
"""
from pathlib import Path
import sys

P = Path.home() / "RAODpp" / "scripts" / "upaste.py"
s = P.read_text()
orig_len = len(s)
applied, skipped = [], []


def sub(tag, old, new):
    global s
    if new.strip()[:40] in s:
        skipped.append(tag); return
    if old not in s:
        print(f"!! ANCHOR NOT FOUND for '{tag}' — patch aborted, file untouched")
        sys.exit(1)
    s = s.replace(old, new, 1)
    applied.append(tag)


# ---------------------------------------------------------------- 1. constants
sub("constants",
"""SIGMA_PSEUDO = 0.60      # E2 pseudo-label acceptance""",
"""SIGMA_PSEUDO = 0.60      # E2 pseudo-label acceptance
GATE_MARGIN = 0.01       # protocol v1.4 A16: reject only a confident contradiction

# --- context-matching constraints (visual audit fix, 2026-08-15) ---
RADIUS_MIN, RADIUS_MAX = 0.8, 2.0    # annulus around the anchor, in anchor diagonals
MAX_UPSCALE = 1.5                    # never blow a crop up more than this
SURFACE_MEAN_TOL = 25.0              # per-channel mean difference vs the anchor surface
SURFACE_STD_TOL = 20.0               # texture difference vs the anchor surface""")

s = s.replace("MAX_TRIES = 20", "MAX_TRIES = 30", 1)

# ---------------------------------------------------------------- 2. paste_one
i = s.index("def paste_one(")
j = s.index("# ------------------------------------------------------------------ build")
old_paste = s[i:j]
new_paste = '''def surface_ok(arr, px, py, Wp, Hp, ax, ay):
    """Reject a location whose surface does not look like the anchor's surface.

    Without this, 'no overlap with any object' pushes pastes onto roofs, trees and
    walls, because those are the only empty regions in a dense aerial scene.
    """
    H, W = arr.shape[:2]
    def stats(x, y):
        x0, y0 = int(np.clip(x, 0, max(W - Wp, 0))), int(np.clip(y, 0, max(H - Hp, 0)))
        p = arr[y0:y0 + Hp, x0:x0 + Wp]
        if p.size == 0:
            return None
        return p.reshape(-1, 3).mean(0), p.reshape(-1, 3).mean(1).std()
    a, b = stats(px, py), stats(ax - Wp / 2, ay - Hp / 2)
    if a is None or b is None:
        return False
    return (np.abs(a[0] - b[0]).mean() <= SURFACE_MEAN_TOL and
            abs(a[1] - b[1]) <= SURFACE_STD_TOL)


def paste_one(img, arr, crop, obj_uv, anchor, gal_box, occupied, rng, blend=True):
    """Place one crop near a same-class anchor, at the anchor's scale.

    Returns (object_box, reason); reason is None on success.
    """
    u1, v1, u2, v2 = obj_uv
    ow, oh = max(u2 - u1, 1e-3), max(v2 - v1, 1e-3)
    aw, ah = anchor[2] - anchor[0], anchor[3] - anchor[1]
    j = 1.0 + rng.uniform(-SCALE_JITTER, SCALE_JITTER)
    tw, th = max(aw * j, 4), max(ah * j, 4)

    gw = max(gal_box[2] - gal_box[0], 1.0)
    if tw > MAX_UPSCALE * gw:                 # would be an upscaled blur
        return None, "upscale"

    Wp, Hp = int(round(tw / ow)), int(round(th / oh))
    if Wp < 6 or Hp < 6 or Wp > img.width // 2 or Hp > img.height // 2:
        return None, "size"

    ax, ay = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    diag = float(np.hypot(aw, ah))
    for _ in range(MAX_TRIES):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(RADIUS_MIN, RADIUS_MAX) * diag
        px = int(ax + rad * np.cos(ang) - Wp / 2)
        py = int(ay + rad * np.sin(ang) - Hp / 2)
        if px < 0 or py < 0 or px + Wp > img.width or py + Hp > img.height:
            continue
        obox = np.array([px + u1 * Wp, py + v1 * Hp, px + u2 * Wp, py + v2 * Hp])
        if len(occupied) and iou_1_to_n(obox, occupied).max() >= IOU_MAX:
            continue
        if not surface_ok(arr, px, py, Wp, Hp, ax, ay):
            continue

        patch = crop.resize((Wp, Hp), Image.BILINEAR)
        if blend:
            bg = np.asarray(img.crop((px, py, px + Wp, py + Hp)), dtype=np.float32)
            pa = np.asarray(patch, dtype=np.float32)
            pa = np.clip(pa + (bg.mean((0, 1)) - pa.mean((0, 1))), 0, 255)
            patch = Image.fromarray(pa.astype(np.uint8))
            mask = Image.new("L", (Wp, Hp), 255).filter(ImageFilter.GaussianBlur(FEATHER))
            img.paste(patch, (px, py), mask)
        else:
            img.paste(patch, (px, py))
        return obox, None
    return None, "no_space"


'''
if "def surface_ok(" in s:
    skipped.append("paste_one")
else:
    s = s.replace(old_paste, new_paste, 1); applied.append("paste_one")

# ---------------------------------------------------------------- 3. build()
sub("audit counters",
"""    audit = {"attempted": 0, "pasted": 0, "no_space": 0, "gate_blocked": 0,
             "too_small": 0, "pseudo_added": 0}""",
"""    audit = {"attempted": 0, "pasted": 0, "no_space": 0, "gate_blocked": 0,
             "no_anchor": 0, "upscale": 0, "size": 0, "pseudo_added": 0}""")

sub("confident detection classes",
"""    conf_by_img = {k: v[["x1", "y1", "x2", "y2"]].values
                   for k, v in conf_det.groupby("image")}""",
"""    conf_by_img = {k: (v[["x1", "y1", "x2", "y2"]].values, v["class_id"].values)
                   for k, v in conf_det.groupby("image")}""")

sub("gate rule",
"""        gate_ok = set(gate_df[gate_df.gate_pass].index) if mode == "E5" else None""",
"""        gate_ok = set(gate_df[gate_df.gate_pass |
                              (gate_df.text_margin < GATE_MARGIN)].index) if mode == "E5" else None""")

sub("gt classes",
"""        gt = np.array([[(float(f[1]) - float(f[3]) / 2) * W, (float(f[2]) - float(f[4]) / 2) * H,
                        (float(f[1]) + float(f[3]) / 2) * W, (float(f[2]) + float(f[4]) / 2) * H]
                       for f in (l.split() for l in lines)]) if lines else np.zeros((0, 4))""",
"""        _sp = [l.split() for l in lines]
        gt = np.array([[(float(f[1]) - float(f[3]) / 2) * W, (float(f[2]) - float(f[4]) / 2) * H,
                        (float(f[1]) + float(f[3]) / 2) * W, (float(f[2]) + float(f[4]) / 2) * H]
                       for f in _sp]) if lines else np.zeros((0, 4))
        gt_cls = np.array([int(f[0]) for f in _sp]) if lines else np.zeros(0, dtype=int)""")

sub("occupied setup",
"""            occupied = [gt] if len(gt) else []
            if p.name in conf_by_img:
                occupied.append(conf_by_img[p.name])
            occ = np.concatenate(occupied) if occupied else np.zeros((0, 4))""",
"""            occupied = [gt] if len(gt) else []
            cd_box, cd_cls = conf_by_img.get(p.name, (np.zeros((0, 4)), np.zeros(0, dtype=int)))
            if len(cd_box):
                occupied.append(cd_box)
            occ = np.concatenate(occupied) if occupied else np.zeros((0, 4))
            arr = np.asarray(img, dtype=np.float32)""")

sub("candidate tuple",
"""                        cand.append((r.x1, r.y1, r.x2, r.y2, gi, sim))""",
"""                        cand.append((r.x1, r.y1, r.x2, r.y2, gi, sim, int(r.class_id)))""")

sub("paste loop",
"""                for (x1, y1, x2, y2, gi, sim) in cand:
                    audit["attempted"] += 1
                    g = gal.iloc[gi]
                    uv = footprint((g.x1, g.y1, g.x2, g.y2), gsize[g.source_image])
                    crop = Image.open(CROPS / "gallery_visdrone" / g.crop).convert("RGB")
                    ob = paste_one(img, crop, uv, (x1, y1, x2, y2), occ, rng, blend)
                    if ob is None:
                        audit["no_space"] += 1; continue""",
"""                for (x1, y1, x2, y2, gi, sim, qcls) in cand:
                    audit["attempted"] += 1
                    g = gal.iloc[gi]
                    c = int(g.class_id)

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
                        audit[why] = audit.get(why, 0) + 1; continue""")

# ---------------------------------------------------------------- verify
import ast
try:
    ast.parse(s)
except SyntaxError as e:
    print("!! patched file does not parse — original left untouched:", e)
    sys.exit(1)

P.write_text(s)
print("applied :", applied or "nothing")
print("skipped :", skipped or "nothing")
print(f"size    : {orig_len} -> {len(s)} chars")
for tok in ["GATE_MARGIN", "surface_ok", "no_anchor", "MAX_UPSCALE", "gt_cls"]:
    print(f"  {tok:14s} {s.count(tok)} occurrences")
print("SYNTAX OK")
