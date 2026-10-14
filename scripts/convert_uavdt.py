#!/usr/bin/env python3
"""
convert_uavdt.py — turn the extracted Supervisely mirror of UAVDT into a YOLO dataset.

UAVDT ships two subsets under one archive. Sequences named `M####` are the detection
subset, annotated with the three vehicle categories. Sequences named `S####` are the
tracking subset: one box per frame carrying a single `vehicle` label, because the tracking
task does not distinguish categories. Only the detection subset is usable for a
three-class detection benchmark, so this script keeps `M####` and reports how many frames
of the other kind it declined, rather than dropping them silently.

The mirror ships one JSON per image and an official sequence split: 30 train sequences,
20 test sequences, no sequence in both. That split is used as-is rather than re-drawn,
which keeps the setup comparable to published UAVDT results and rules out the frame-level
leakage a random split would cause (consecutive video frames are near duplicates).

Frames are then thinned per sequence so the training set lands near VisDrone's size.

    python scripts/convert_uavdt.py --scan     # class counts and geometry check only
    python scripts/convert_uavdt.py            # convert
"""
import argparse, hashlib, json, re, time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "RAODpp"
RAW = ROOT / "dataset" / "uavdt_raw"
OUT = ROOT / "dataset" / "uavdt"
OUTPUT = ROOT / "output"

CLASSES = ["car", "truck", "bus"]     # official UAVDT categories
EXTRA_CLASS = "vehicle"               # present in the mirror; counted, then decided on
TARGET_TRAIN = 6000
TARGET_VAL = 2000
MIN_BOX_PX = 2


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_root():
    """Locate the directory that holds train/ and test/."""
    for cand in [RAW] + [p for p in RAW.iterdir() if p.is_dir()]:
        if (cand / "train" / "ann").is_dir() and (cand / "test" / "ann").is_dir():
            return cand
    raise SystemExit(f"could not find train/ann and test/ann under {RAW}")


def class_map(base):
    meta = json.loads((base / "meta.json").read_text())
    by_id = {c["id"]: c["title"] for c in meta["classes"]}
    log(f"classes in meta.json: {sorted(by_id.values())}")
    return by_id


def seq_of(name):
    m = re.search(r"(M\d{4})", name)
    return m.group(1) if m else None


def scan(base, by_id, n=400):
    """Check the geometry field and count classes before converting anything."""
    cls_count, geom, missing_pts = Counter(), Counter(), 0
    for split in ["train", "test"]:
        files = sorted((base / split / "ann").glob("*.json"))[:n]
        for f in files:
            d = json.loads(f.read_text())
            for o in d.get("objects", []):
                cls_count[by_id.get(o.get("classId"), "?")] += 1
                geom[o.get("geometryType", "?")] += 1
                pts = (o.get("points") or {}).get("exterior")
                if not pts or len(pts) < 2:
                    missing_pts += 1
    log(f"sampled objects: {sum(cls_count.values()):,}")
    log("class counts: " + ", ".join(f"{k}={v:,}" for k, v in cls_count.most_common()))
    log(f"geometry types: {dict(geom)}")
    log(f"objects without a usable 'points.exterior': {missing_pts}")
    return cls_count


def frame_table(base, split):
    rows = []
    for f in sorted((base / split / "ann").glob("*.json")):
        stem = f.name[:-5]                      # strip .json -> M0101_img000001.jpg
        rows.append({"ann": f, "img": base / split / "img" / stem,
                     "stem": Path(stem).stem, "seq": seq_of(stem), "name": f.name})
    df = pd.DataFrame(rows)

    # frames outside the detection subset are declined explicitly, with a count, so the
    # decision is visible in the log and in the data card rather than implied by a regex
    kept = df.seq.notna()
    if (~kept).any():
        other = df[~kept]
        tracking = other.name.str.match(r"^S\d+_")
        log(f"{split}: {len(other):,} frames are not in the M#### detection subset "
            f"({int(tracking.sum()):,} of them S#### tracking frames) and are excluded; "
            f"the tracking subset carries a single class and cannot be evaluated as "
            f"three-class detection")
    df = df[kept].sort_values(["seq", "stem"]).reset_index(drop=True)
    df["idx_in_seq"] = df.groupby("seq").cumcount()
    return df


def convert_split(df, by_id, split_name, keep_extra):
    img_dir, lbl_dir = OUT / split_name / "images", OUT / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    names = CLASSES + ([EXTRA_CLASS] if keep_extra else [])
    idx_of = {c: i for i, c in enumerate(names)}

    n_box, n_drop, n_img, n_noimg = 0, 0, 0, 0
    for r in df.itertuples():
        if not r.img.exists():
            n_noimg += 1
            continue
        d = json.loads(r.ann.read_text())
        W, H = d["size"]["width"], d["size"]["height"]
        lines = []
        for o in d.get("objects", []):
            title = by_id.get(o.get("classId"))
            if title not in idx_of:
                n_drop += 1
                continue
            pts = (o.get("points") or {}).get("exterior") or []
            if len(pts) < 2:
                n_drop += 1
                continue
            (x1, y1), (x2, y2) = pts[0], pts[1]
            x1, x2 = sorted((float(x1), float(x2)))
            y1, y2 = sorted((float(y1), float(y2)))
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, W), min(y2, H)
            bw, bh = x2 - x1, y2 - y1
            if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
                n_drop += 1
                continue
            lines.append(f"{idx_of[title]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                         f"{bw/W:.6f} {bh/H:.6f}")
            n_box += 1
        dst = img_dir / r.img.name
        if not dst.exists():
            dst.symlink_to(r.img)
        (lbl_dir / f"{r.stem}.txt").write_text("\n".join(lines))
        n_img += 1
    log(f"{split_name}: {n_img:,} images, {n_box:,} boxes, {n_drop:,} objects dropped, "
        f"{n_noimg:,} annotations without an image")
    return n_img, n_box, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--keep-vehicle", action="store_true",
                    help="keep the mirror's extra 'vehicle' class as a 4th category")
    a = ap.parse_args()

    base = find_root()
    log(f"dataset root: {base}")
    by_id = class_map(base)

    counts = scan(base, by_id)
    if a.scan:
        log("scan only; re-run without --scan to convert")
        return
    if not a.keep_vehicle and counts.get(EXTRA_CLASS, 0):
        log(f"NOTE: {counts[EXTRA_CLASS]:,} sampled '{EXTRA_CLASS}' objects will be "
            f"dropped (use --keep-vehicle to include them)")

    tr, te = frame_table(base, "train"), frame_table(base, "test")
    log(f"sequences: {tr.seq.nunique()} train / {te.seq.nunique()} test  "
        f"(overlap {len(set(tr.seq) & set(te.seq))})")
    assert not (set(tr.seq) & set(te.seq)), "official split shares a sequence"

    s_tr = max(1, -(-len(tr) // TARGET_TRAIN))
    s_te = max(1, -(-len(te) // TARGET_VAL))
    tr_k, te_k = tr[tr.idx_in_seq % s_tr == 0], te[te.idx_in_seq % s_te == 0]
    log(f"frames: train {len(tr):,} -> {len(tr_k):,} (stride {s_tr}), "
        f"test {len(te):,} -> {len(te_k):,} (stride {s_te})")

    n_tr, b_tr, names = convert_split(tr_k, by_id, "train", a.keep_vehicle)
    n_va, b_va, _ = convert_split(te_k, by_id, "val", a.keep_vehicle)

    hashes = {"stride_train": s_tr, "stride_val": s_te,
              "train_sequences": sorted(tr.seq.unique().tolist()),
              "val_sequences": sorted(te.seq.unique().tolist())}
    for split in ["train", "val"]:
        files = sorted(p.name for p in (OUT / split / "images").glob("*.jpg"))
        txt = "\n".join(files)
        (OUTPUT / f"uavdt_{split}_files.txt").write_text(txt)
        hashes[split] = {"n_files": len(files),
                         "sha256": hashlib.sha256(txt.encode()).hexdigest()}
        log(f"{split}: n={len(files):,} sha256={hashes[split]['sha256'][:16]}...")
    json.dump(hashes, open(OUTPUT / "uavdt_split_hashes.json", "w"), indent=2)

    a_tr = {p.name for p in (OUT / "train" / "images").glob("*.jpg")}
    a_va = {p.name for p in (OUT / "val" / "images").glob("*.jpg")}
    assert not (a_tr & a_va), "filename overlap between splits"
    log("leakage audit: PASS (no shared sequences, no shared filenames)")

    yml = ROOT / "dataset" / "uavdt_raodpp.yaml"
    yml.write_text(f"# RAOD++ UAVDT (second dataset) — convert_uavdt.py\n"
                   f"path: {OUT}\ntrain: train/images\nval: val/images\nnames:\n"
                   + "\n".join(f"  {i}: {n}" for i, n in enumerate(names)) + "\n")
    log(f"wrote {yml}")

    card = pd.DataFrame([
        ["Dataset", "UAVDT (Supervisely mirror, Kaggle)"],
        ["Role in study", "Second dataset, H2 confirmation (protocol v2.1 C4.3)"],
        ["Split", "official sequence split, used as-is"],
        ["Subset", "M#### detection subset only; S#### tracking frames excluded"],
        ["Sequences", f"{tr.seq.nunique()} train / {te.seq.nunique()} val"],
        ["Frame stride", f"train {s_tr}, val {s_te}"],
        ["Train images", n_tr], ["Val images", n_va],
        ["Train boxes", b_tr], ["Val boxes", b_va],
        ["Classes", ", ".join(names)],
        ["Split hash (train)", hashes["train"]["sha256"][:32] + "..."],
        ["Split hash (val)", hashes["val"]["sha256"][:32] + "..."],
    ], columns=["field", "value"])
    card.to_csv(OUTPUT / "datacard_uavdt.csv", index=False)
    print("\n" + card.to_string(index=False))


if __name__ == "__main__":
    main()
