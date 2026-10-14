#!/usr/bin/env python3
"""
check_repo.py — does the repository contain what the manuscript says it contains?

The paper cites files by path: table captions name the CSV each table was computed from,
figure captions name the figure's data, and the data-availability statement promises the
code, the run table and the per-seed data behind every figure. Each of those is a claim a
reader can check in about ten seconds, and a path that resolves to nothing is worse than
no path at all.

This compares the two. It reads every \\texttt{...} path out of the manuscript, looks for
it in the repository, and lists what is missing. It also flags files that should not be
in a released repository: editor and OS droppings, byte-compiled Python, the .bak files
editor backups, and superseded copies.

    python check_repo.py --tex path/to/main.tex --repo path/to/repo

Exits non-zero if anything the manuscript cites is missing.
"""
import argparse
import re
import sys
from pathlib import Path

# paths the manuscript may cite that are produced at run time rather than committed
RUNTIME_ONLY = set()

JUNK = [
    ("*Zone.Identifier", "Windows alternate-data-stream droppings from the WSL copy"),
    ("__pycache__/*", "byte-compiled Python"),
    ("*.pyc", "byte-compiled Python"),
    ("*.bak", "patch-script backups"),
    ("*.bak-*", "patch-script backups"),
    ("*eski*", "superseded copy, judging by the name"),
    ("*_broken*", "superseded copy, judging by the name"),
    ("* (1).*", "duplicate from a browser download"),
    ("* (2).*", "duplicate from a browser download"),
]


def cited_paths(tex):
    """Every \\texttt{...} that looks like a repository path."""
    out = set()
    for m in re.finditer(r"\\texttt\{([^}]*)\}", tex):
        raw = m.group(1).replace("\\_", "_").replace("\\%", "%").strip()
        if "/" not in raw or raw.startswith("FILL"):
            continue
        if not re.match(r"^[A-Za-z0-9_./-]+$", raw):
            continue
        out.add(raw)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()

    tex = Path(a.tex).read_text(encoding="utf-8")
    repo = Path(a.repo)
    if not repo.is_dir():
        sys.exit(f"{repo} is not a directory")

    print(f"manuscript: {a.tex}")
    print(f"repository: {a.repo}\n")

    paths = cited_paths(tex)
    missing, present = [], []
    for p in paths:
        (present if (repo / p).exists() else missing).append(p)

    print(f"--- paths the manuscript cites: {len(paths)} ---")
    for p in present:
        print(f"  ok      {p}")
    for p in missing:
        print(f"  MISSING {p}")

    # figures the manuscript includes
    figs = sorted({m.group(1) for m in
                   re.finditer(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", tex)})
    fig_missing = [f for f in figs if not (repo / f).exists()]
    print(f"\n--- figures the manuscript includes: {len(figs)} ---")
    for f in figs:
        print(f"  {'ok     ' if f not in fig_missing else 'MISSING'} {f}")

    # things that should not ship
    print("\n--- files that should not be in a released repository ---")
    junk_found = []
    for pattern, why in JUNK:
        hits = [p for p in repo.rglob(pattern.replace("__pycache__/*", "__pycache__/**"))
                if ".git/" not in str(p) and p.is_file()]
        if hits:
            junk_found += hits
            print(f"  {len(hits):>3}  {pattern:<18} {why}")
    if not junk_found:
        print("  none")

    print()
    bad = len(missing) + len(fig_missing)
    if bad:
        print(f"FAIL: {len(missing)} cited path(s) and {len(fig_missing)} figure(s) are "
              f"not in the repository.")
        print("A reader who follows a caption to one of these finds nothing.")
        sys.exit(1)
    print("Every path and figure the manuscript cites is present."
          + (f" {len(junk_found)} junk file(s) to remove." if junk_found else ""))


if __name__ == "__main__":
    main()
