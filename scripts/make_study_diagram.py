#!/usr/bin/env python3
"""Generate fig/study_design.drawio — the experimental design of the paper."""
from pathlib import Path
from xml.sax.saxutils import escape

W = 1560
cells, edges = [], []

PAL = {
    "data":   "fillColor=#dae8fc;strokeColor=#6c8ebf;",
    "ctrl":   "fillColor=#ffe6cc;strokeColor=#d79b00;",
    "iso":    "fillColor=#d5e8d4;strokeColor=#82b366;",
    "meas":   "fillColor=#e1d5e7;strokeColor=#9673a6;",
    "neg":    "fillColor=#f8cecc;strokeColor=#b85450;",
    "pos":    "fillColor=#d5e8d4;strokeColor=#82b366;strokeWidth=2;",
    "out":    "fillColor=#f5f5f5;strokeColor=#999999;",
    "group":  "fillColor=none;strokeColor=#b3b3b3;dashed=1;verticalAlign=top;"
              "align=left;spacingLeft=10;spacingTop=4;fontStyle=1;fontSize=13;"
              "fontColor=#666666;",
}
BASE = ("rounded=1;whiteSpace=wrap;html=1;fontSize=12;fontFamily=Helvetica;"
        "verticalAlign=middle;arcSize=8;")
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;"
        "strokeColor=#555555;strokeWidth=1.5;")


def box(i, x, y, w, h, title, sub="", pal="data", style_extra=""):
    html = f"<b>{title}</b>"
    if sub:
        html += ("<br><font style=\"font-size:10px;color:#444444\">"
                 + sub.replace("\n", "<br>") + "</font>")
    label = escape(html, {'"': "&quot;"})
    cells.append(f'<mxCell id="{i}" value="{label}" style="{BASE}{PAL[pal]}{style_extra}" '
                 f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
                 f'height="{h}" as="geometry"/></mxCell>')


def frame(i, x, y, w, h, title):
    cells.append(f'<mxCell id="{i}" value="{escape(title)}" '
                 f'style="rounded=1;whiteSpace=wrap;html=1;{PAL["group"]}" '
                 f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
                 f'height="{h}" as="geometry"/></mxCell>')


def arrow(i, s, t, label="", extra=""):
    val = f' value="{escape(label)}"' if label else ""
    edges.append(f'<mxCell id="{i}"{val} style="{EDGE}{extra}" edge="1" parent="1" '
                 f'source="{s}" target="{t}"><mxGeometry relative="1" as="geometry"/>'
                 f'</mxCell>')


# ----------------------------------------------------------------- title
box("t", 40, 20, 1480, 44,
    "Controlled study: where should a fixed inference budget be spent?",
    "Every reported number is produced by a script and written to output/ ; "
    "no value is transcribed by hand.",
    "out", "fillColor=none;strokeColor=none;fontSize=15;")

# ----------------------------------------------------------------- data
frame("f_data", 40, 80, 1480, 130, "1 — Datasets and frozen splits")
box("d1", 70, 115, 440, 80, "VisDrone2019-DET  (primary)",
    "6,471 / 548 images · 343,197 / 38,759 boxes · 10 classes\n"
    "ignored regions removed · boxes clipped and normalised", "data")
box("d2", 550, 115, 440, 80, "UAVDT  (second dataset)",
    "official split by video sequence, 30 / 20, no overlap\n"
    "thinned by stride · 4,840 / 1,854 images · 3 classes", "data")
box("d3", 1030, 115, 460, 80, "Frozen before any training",
    "sorted file list hashed with SHA-256 · re-checked by every run\n"
    "leakage audit: shared names, and on UAVDT shared sequences", "data",
    "dashed=1;")
arrow("e_d1", "d1", "d3")
arrow("e_d2", "d2", "d3")

# ----------------------------------------------------------------- control study
frame("f_ctrl", 40, 235, 700, 640, "2 — Data-centric control study (elimination)")
box("c0", 70, 275, 640, 56, "Baseline E0 — YOLOv8n @640, 10 seeds",
    "mAP50-95 = 0.1863 ± 0.0015 · converged (early stop 133–178 epochs)", "ctrl")
box("c1", 70, 351, 640, 62, "Low-confidence mining",
    "per class, the band where empirical precision runs 0.50 → 0.90\n"
    "replaces a percentile threshold, which landed in the noise floor", "ctrl")
box("c2", 70, 433, 640, 62, "Retrieval — CLIP ViT-B/32 + exact inner-product search",
    "embeddings centred on the gallery mean, then renormalised\n"
    "gallery: VisDrone test-dev, images only, labels deleted", "ctrl")
box("c3", 70, 515, 640, 62, "Cross-modal gate",
    "a crop is rejected only when the text classifier both disagrees\n"
    "with the proposer and is confident about it", "ctrl")
box("c4", 70, 597, 640, 62, "U-Paste placement",
    "anchored on a same-class instance already in the frame, at its scale\n"
    "surface, seam, blur and overlap checks · seven rounds of visual audit", "ctrl")
box("c5", 70, 679, 640, 56, "Conditions E1–E5 · equal +20 epochs · 10 seeds",
    "60 runs · same checkpoint, same budget, only the data differs", "ctrl")
box("c6", 70, 755, 640, 90, "Result: no gain",
    "every condition within ±0.0025 of the baseline, all below it\n"
    "budget sweep 2.7% → 9.5% of boxes: flat, then harmful\n"
    "pre-registered criterion not met against any control", "neg")
for i, (a, b) in enumerate(zip(["c0", "c1", "c2", "c3", "c4", "c5"],
                               ["c1", "c2", "c3", "c4", "c5", "c6"])):
    arrow(f"e_c{i}", a, b)

# ----------------------------------------------------------------- iso-cost study
frame("f_iso", 780, 235, 740, 640, "3 — Iso-cost study")
box("i1", 810, 275, 680, 74, "Three allocation arms",
    "resolution   YOLOv8n @ 640 / 960 / 1280 / 1536\n"
    "capacity     YOLOv8s @ 640 / 960 / 1280\n"
    "P2 head      YOLOv8n-P2 @ 640 / 960", "iso")
box("i2", 810, 369, 680, 56, "Training — identical apart from the factor under test",
    "200-epoch ceiling, patience 30, 3 seeds · nominal batch fixed at 64", "iso")
box("i3", 810, 445, 320, 74, "Accuracy",
    "mAP50-95 on the frozen\nvalidation split only", "iso")
box("i4", 1170, 445, 320, 74, "Cost — measured, not derived",
    "batch-1 latency, frames in RAM\nNVML power · median of 3 passes\n"
    "refused unless the GPU is idle", "meas")
box("i5", 810, 539, 680, 74, "Arm curves and overlap",
    "piecewise-linear accuracy-vs-cost curve per arm and per seed\n"
    "compared only where both arms are measured — no extrapolation", "meas")
box("i6", 810, 633, 680, 62, "Statistics, fixed in advance",
    "paired t-test on seed-wise differences · Wilcoxon confirmation\n"
    "Holm over 4 comparisons · bootstrap CI · floor 0.005", "meas")
box("i7", 810, 715, 680, 130, "Result: the scarce resource decides",
    "latency budget  →  capacity leads by 0.0216   (p_Holm = 0.0008)\n"
    "energy budget   →  resolution leads by 0.0371 (p_Holm = 0.0008)\n"
    "the P2 head loses on both axes\n"
    "resolution alone: 0.1863 → 0.3249 mAP50-95, +74% relative", "pos")
arrow("e_i1", "i1", "i2")
arrow("e_i2", "i2", "i3")
arrow("e_i3", "i2", "i4")
arrow("e_i4", "i3", "i5")
arrow("e_i5", "i4", "i5")
arrow("e_i6", "i5", "i6")
arrow("e_i7", "i6", "i7")

arrow("e_bridge", "c6", "i1", "motivates",
      "dashed=1;strokeColor=#b85450;exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

# ----------------------------------------------------------------- outputs
frame("f_out", 40, 900, 1480, 110, "4 — Everything is written to a file")
box("o1", 70, 940, 340, 56, "output/runs.csv",
    "one row per run: condition, seed,\nresolution, accuracy, latency, power", "out")
box("o2", 430, 940, 340, 56, "output/figdata/",
    "one file per figure source;\nfigures redrawn without any GPU hour", "out")
box("o3", 790, 940, 340, 56, "protocol amendments",
    "each change dated, with its reason\nand whether results were seen", "out")
box("o4", 1150, 940, 340, 56, "output/PROJECT_LOG.md",
    "decisions, defects found and fixed,\nfile-by-file map of results", "out")

xml = f"""<mxfile host="app.diagrams.net" type="device">
  <diagram name="Study design" id="study-design">
    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1"
                  connect="1" arrows="1" fold="1" page="1" pageScale="1"
                  pageWidth="{W}" pageHeight="1080" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {chr(10).join("        " + c for c in cells)}
        {chr(10).join("        " + e for e in edges)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

out = Path.home() / "RAODpp" / "fig" / "study_design.drawio"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(xml)
print(f"wrote {out}  ({len(cells)} nodes, {len(edges)} edges, {len(xml)} chars)")
