#!/usr/bin/env python3
"""
prep_uavdt.py — step 1 of the second dataset: download UAVDT and report what is inside.

Nothing is converted or split here. Mirror layouts differ from the official release, so the
archive is inspected first and the conversion is written afterwards against what is actually
there.

    python scripts/prep_uavdt.py            # download (if needed) and inspect
    python scripts/prep_uavdt.py --extract  # also extract
"""
import argparse, subprocess, sys, time, zipfile
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "RAODpp"
RAW = ROOT / "dataset" / "uavdt_raw"
SLUG = "shakaibkaggle/uavdt-dataset"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    zips = sorted(RAW.glob("*.zip"))
    if zips:
        log(f"archive already present: {[z.name for z in zips]}")
        return zips
    log(f"downloading {SLUG} — this can take a while")
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                        "-d", SLUG, "-p", str(RAW)], text=True)
    if r.returncode != 0:
        log("kaggle download failed; check ~/.config/kaggle/kaggle.json and the network")
        sys.exit(1)
    return sorted(RAW.glob("*.zip"))


def inspect(zips):
    for zp in zips:
        log(f"{zp.name}: {zp.stat().st_size/1e9:.2f} GB")
    z = zipfile.ZipFile(zips[0])
    names = z.namelist()
    print(f"\nentries: {len(names):,}")

    print("\n--- first 20 entries ---")
    for n in names[:20]:
        print("   ", n)

    print("\n--- extensions ---")
    for e, c in Counter(Path(n).suffix.lower() for n in names).most_common(10):
        print(f"    {e or '(none)':8s} {c:>8,}")

    print("\n--- top level ---")
    for k, c in Counter(n.split("/")[0] for n in names if "/" in n).most_common(10):
        print(f"    {k:40s} {c:>8,}")

    print("\n--- second level (sample) ---")
    for k, c in Counter("/".join(n.split("/")[:2])
                        for n in names if n.count("/") >= 2).most_common(12):
        print(f"    {k:50s} {c:>8,}")

    print("\n--- third level (sample) ---")
    for k, c in Counter("/".join(n.split("/")[:3])
                        for n in names if n.count("/") >= 3).most_common(8):
        print(f"    {k:60s} {c:>8,}")

    # show a couple of annotation files verbatim: the conversion depends on their format
    ann = [n for n in names if n.lower().endswith((".txt", ".xml", ".json"))
           and "readme" not in n.lower()]
    print(f"\n--- annotation-like files: {len(ann):,} ---")
    for n in ann[:5]:
        print("   ", n)
    for n in ann[:2]:
        try:
            head = z.read(n).decode("utf-8", "replace").splitlines()[:6]
        except Exception as e:
            head = [f"(could not read: {e})"]
        print(f"\n--- {n} — first lines ---")
        for line in head:
            print("   ", line)

    imgs = [n for n in names if n.lower().endswith(".jpg")]
    print(f"\n--- images: {len(imgs):,} ---")
    for n in imgs[:5]:
        print("   ", n)
    return z, names


def extract(z, names):
    marker = RAW / ".extracted"
    if marker.exists():
        log("already extracted")
        return
    t0 = time.time()
    for i, n in enumerate(names, 1):
        z.extract(n, RAW)
        if i % 5000 == 0:
            log(f"   {i:,}/{len(names):,}  {time.time()-t0:.0f}s")
    marker.write_text("done")
    log(f"extracted in {(time.time()-t0)/60:.1f} min")
    log(f"images on disk: {len(list(RAW.rglob('*.jpg'))):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()
    zips = download()
    z, names = inspect(zips)
    if a.extract:
        extract(z, names)
    else:
        print("\nrun again with --extract once the layout above looks right")
