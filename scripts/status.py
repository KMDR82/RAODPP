#!/usr/bin/env python3
"""
status.py — one command that answers: is anything running, what is it doing, and what do
we have so far. Every call also writes a consolidated snapshot to
output/STUDY_SUMMARY.md and output/STUDY_SUMMARY.csv, so results never live only in a
terminal scrollback.

    python scripts/status.py
"""
import re, subprocess, time
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "RAODpp"
OUT, RUNS_DIR, FIG = ROOT / "output", ROOT / "model" / "runs", ROOT / "output" / "figdata"

PHASES = {
    "E0": "baseline @640 (v1)",
    "E1": "control: +20 ep, same data",
    "E2": "control: self-training pseudo-labels",
    "E3": "control: random-neighbour paste",
    "E4": "treatment: kNN retrieval paste",
    "E5": "treatment: full RAOD++",
    "F3": "from scratch: random paste",
    "F5": "from scratch: RAOD++",
    "F5b15": "from scratch: RAOD++ budget 15",
    "F5b30": "from scratch: RAOD++ budget 30",
    "R960": "resolution 960",
    "R1280": "resolution 1280",
    "S640": "capacity yolov8s @640",
    "S960": "capacity yolov8s @960",
    "P640": "P2 head @640",
    "P960": "P2 head @960",
}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def running():
    ps = sh("ps -eo pid,etime,cmd | grep '[t]rain_\\|[u]paste.py\\|[e]mbed_' "
            "| grep -v forkserver")
    return [l for l in ps.splitlines() if l.strip()]


def gpu():
    return sh("nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw "
              "--format=csv,noheader")


def active_run():
    """The most recently modified run directory that has a results.csv."""
    cands = [(p.stat().st_mtime, p) for p in RUNS_DIR.glob("*")
             if (p / "results.csv").exists()]
    if not cands:
        return None
    return max(cands)[1]


def progress(run_dir):
    try:
        d = pd.read_csv(run_dir / "results.csv")
    except Exception:
        return None
    d.columns = [c.strip() for c in d.columns]
    col = next((c for c in d.columns if "mAP50-95" in c), None)
    tcol = next((c for c in d.columns if c.lower() == "time"), None)
    if col is None:
        return None
    return {"run": run_dir.name, "epochs": len(d),
            "last": round(float(d[col].iloc[-1]), 4),
            "best": round(float(d[col].max()), 4),
            "best_epoch": int(d[col].idxmax()) + 1,
            "hours": round(float(d[tcol].iloc[-1]) / 3600, 2) if tcol else None}


def summary_table():
    f = OUT / "runs.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    d = d[d.get("dataset", "visdrone") == "visdrone"] if "dataset" in d else d
    cols = {"mAP50": "mean", "mAP50_95": "mean"}
    for c in ["ms_mean", "fps", "power_w", "imgsz"]:
        if c in d.columns:
            cols[c] = "mean"
    g = d.groupby("exp").agg({**cols, "seed": "size"}).rename(columns={"seed": "n"})
    g = g.round(4)
    g["note"] = [PHASES.get(i, "") for i in g.index]
    order = [k for k in PHASES if k in g.index] + [i for i in g.index if i not in PHASES]
    return g.loc[order]


def main():
    print("=" * 78)
    print(f"RAOD++ status — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    procs = running()
    if procs:
        print(f"\nRUNNING ({len(procs)} process):")
        for l in procs:
            print("  ", re.sub(r"\s+", " ", l)[:150])
    else:
        print("\nRUNNING: nothing")
    print("GPU     :", gpu() or "n/a")

    ar = active_run()
    if ar:
        p = progress(ar)
        if p:
            print(f"\nMOST RECENT RUN DIR: {p['run']}")
            print(f"  epochs={p['epochs']}  last={p['last']}  best={p['best']} "
                  f"(epoch {p['best_epoch']})  elapsed={p['hours']} h")

    g = summary_table()
    if len(g):
        print("\nRESULTS SO FAR (mean over seeds):")
        print(g.to_string())
        OUT.mkdir(exist_ok=True)
        g.to_csv(OUT / "STUDY_SUMMARY.csv")
        md = ["# RAOD++ — study summary",
              f"_generated {time.strftime('%Y-%m-%d %H:%M')}_", "",
              g.to_markdown(), "",
              "Every number here comes from `output/runs.csv`; nothing is typed by hand.",
              "Per-run detail: `output/runs.csv`. Figure sources: `output/figdata/`."]
        (OUT / "STUDY_SUMMARY.md").write_text("\n".join(md))
        print(f"\nwritten: {OUT/'STUDY_SUMMARY.csv'}")
        print(f"written: {OUT/'STUDY_SUMMARY.md'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
