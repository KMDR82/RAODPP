#!/usr/bin/env python3
"""
patch_dataset_arg.py — add a --dataset switch to the four scripts that were written
against VisDrone only, so the same code can run the UAVDT confirmation.

Applied in place, idempotent, and it refuses to write anything if an anchor is missing or
the result does not parse.

    python scripts/patch_dataset_arg.py
"""
import ast
from pathlib import Path

S = Path.home() / "RAODpp" / "scripts"
DATASETS_BLOCK = '''DATASETS = {"visdrone": "visdrone_raodpp.yaml", "uavdt": "uavdt_raodpp.yaml"}
DATA_YAML = ROOT / "dataset" / DATASETS["visdrone"]   # rebound in main() from --dataset'''
OLD_YAML = 'DATA_YAML = ROOT / "dataset" / "visdrone_raodpp.yaml"'

report = []


def edit(name, edits, check):
    p = S / name
    if not p.exists():
        report.append((name, "MISSING")); return
    s = orig = p.read_text()
    if check in s:
        report.append((name, "already patched")); return
    for old, new in edits:
        if old not in s:
            report.append((name, f"ANCHOR NOT FOUND: {old.splitlines()[0][:60]}"))
            return
        s = s.replace(old, new, 1)
    try:
        ast.parse(s)
    except SyntaxError as e:
        report.append((name, f"WOULD NOT PARSE: {e}")); return
    p.write_text(s)
    report.append((name, f"patched ({len(orig)} -> {len(s)} chars)"))


# ---------------------------------------------------------------- train_resolution.py
edit("train_resolution.py", [
    (OLD_YAML, DATASETS_BLOCK),
    ('''    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["SEEDS"])
    a = ap.parse_args()''',
     '''    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["SEEDS"])
    ap.add_argument("--dataset", choices=list(DATASETS), default="visdrone")
    a = ap.parse_args()

    global DATA_YAML
    CONFIG["DATASET"] = a.dataset
    DATA_YAML = ROOT / "dataset" / DATASETS[a.dataset]
    assert DATA_YAML.exists(), f"missing {DATA_YAML}"'''),
], check="--dataset")

# ---------------------------------------------------------------- finalize_runs.py
edit("finalize_runs.py", [
    (OLD_YAML, DATASETS_BLOCK + '\nDATASET = "visdrone"'),
    ('for d in sorted(RUNS_DIR.glob("*_visdrone_s*")):',
     'for d in sorted(RUNS_DIR.glob(f"*_{DATASET}_s*")):'),
    ('"exp": exp, "dataset": "visdrone",', '"exp": exp, "dataset": DATASET,'),
    ('''    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()''',
     '''    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dataset", choices=list(DATASETS), default="visdrone")
    a = ap.parse_args()

    global DATA_YAML, DATASET
    DATASET = a.dataset
    DATA_YAML = ROOT / "dataset" / DATASETS[a.dataset]
    assert DATA_YAML.exists(), f"missing {DATA_YAML}"'''),
], check="--dataset")

# ---------------------------------------------------------------- measure_cost_only.py
edit("measure_cost_only.py", [
    (OLD_YAML, DATASETS_BLOCK),
    ('''    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()''',
     '''    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dataset", choices=list(DATASETS), default="visdrone")
    a = ap.parse_args()

    global DATA_YAML
    DATA_YAML = ROOT / "dataset" / DATASETS[a.dataset]
    assert DATA_YAML.exists(), f"missing {DATA_YAML}"'''),
    ('run_ids = sorted(p.name for p in RUNS_DIR.glob(f"{a.exp}_visdrone_s*")',
     'run_ids = sorted(p.name for p in RUNS_DIR.glob(f"{a.exp}_{a.dataset}_s*")'),
], check="--dataset")

# ---------------------------------------------------------------- iso_analysis.py
edit("iso_analysis.py", [
    ('''def load(metric):
    d = pd.read_csv(OUT / "runs.csv")''',
     '''def load(metric, dataset):
    d = pd.read_csv(OUT / "runs.csv")
    if "dataset" in d.columns:
        d = d[d.dataset == dataset]'''),
    ('    ap.add_argument("--metric", default="mAP50_95")',
     '    ap.add_argument("--metric", default="mAP50_95")\n'
     '    ap.add_argument("--dataset", default="visdrone")'),
    ('    d = load(a.metric)', '    d = load(a.metric, a.dataset)'),
    ('FIG / f"iso_{a.cost}_curve.csv"', 'FIG / f"iso_{a.dataset}_{a.cost}_curve.csv"'),
    ('FIG / f"iso_{a.cost}_conditions.csv"',
     'FIG / f"iso_{a.dataset}_{a.cost}_conditions.csv"'),
    ("{FIG/f'iso_{a.cost}_curve.csv'}", "{FIG/f'iso_{a.dataset}_{a.cost}_curve.csv'}"),
], check="a.dataset")

print("\n".join(f"{n:26s} {r}" for n, r in report))
bad = [n for n, r in report if r.startswith(("ANCHOR", "WOULD", "MISSING"))]
raise SystemExit(1 if bad else 0)
