"""
figdata.py — RAOD++ figure-data capture layer.

Every figure in the manuscript must be reproducible from a CSV in output/figdata/.
No plotting happens here; this module only extracts and stores raw numbers so that
F9 can redraw any figure without re-running a single GPU hour.

Captured per run:
  training_curves.csv   per-epoch losses + metrics        (Fig: convergence)
  perclass_ap.csv       per-class AP50 / AP50-95          (Fig: class-wise gains)
  pr_curves.csv         precision-recall curve points     (Fig: PR)
  confusion_matrix.csv  long-format confusion matrix      (Fig: confusion)
  predictions.csv       per-detection dump on val         (Fig: qualitative, error analysis)
  val_conf_hist.csv     confidence histogram              (Fig: low-confidence motivation)
"""
from pathlib import Path
import numpy as np
import pandas as pd


def _fig_dir(output_dir: Path) -> Path:
    d = Path(output_dir) / "figdata"
    d.mkdir(parents=True, exist_ok=True)
    return d


def capture_training_curve(run_dir, run_id, exp, seed, output_dir):
    """Ultralytics writes results.csv per run; store it in long format, appendable."""
    src = Path(run_dir) / "results.csv"
    if not src.exists():
        return None
    df = pd.read_csv(src)
    df.columns = [c.strip() for c in df.columns]
    long = df.melt(id_vars=[df.columns[0]], var_name="metric", value_name="value")
    long = long.rename(columns={df.columns[0]: "epoch"})
    long.insert(0, "run_id", run_id); long.insert(1, "exp", exp); long.insert(2, "seed", seed)
    out = _fig_dir(output_dir) / "training_curves.csv"
    long.to_csv(out, mode="a", header=not out.exists(), index=False)
    return len(long)


def capture_val_artifacts(res, class_names, run_id, exp, seed, output_dir):
    """Per-class AP, PR curves and the confusion matrix from a validation result."""
    fd = _fig_dir(output_dir)
    n = {}

    # --- per-class AP ---
    idx = list(getattr(res.box, "ap_class_index", []))
    if idx:
        rows = []
        ap50 = np.atleast_1d(res.box.ap50)
        ap = np.atleast_1d(res.box.ap)
        for i, c in enumerate(idx):
            rows.append({"run_id": run_id, "exp": exp, "seed": seed,
                         "class_id": int(c), "class_name": class_names.get(int(c), str(c)),
                         "AP50": float(ap50[i]), "AP50_95": float(ap[i])})
        out = fd / "perclass_ap.csv"
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        n["perclass_ap"] = len(rows)

    # --- PR / F1 / P / R curves ---
    try:
        rows = []
        for (xs, ys, xlab, ylab) in res.curves_results:
            ys = np.atleast_2d(ys)
            for ci in range(ys.shape[0]):
                cname = class_names.get(int(idx[ci]), str(ci)) if ci < len(idx) else "all"
                for x, y in zip(np.asarray(xs).ravel(), ys[ci].ravel()):
                    rows.append({"run_id": run_id, "exp": exp, "seed": seed,
                                 "curve": f"{ylab}_vs_{xlab}", "class_name": cname,
                                 "x": float(x), "y": float(y)})
        if rows:
            out = fd / "pr_curves.csv"
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
            n["pr_curves"] = len(rows)
    except Exception as e:                                   # never lose a run over a figure
        print(f"  [figdata] curve capture skipped: {e}")

    # --- confusion matrix (long format) ---
    try:
        cm = res.confusion_matrix.matrix
        labels = [class_names[i] for i in sorted(class_names)] + ["background"]
        rows = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    rows.append({"run_id": run_id, "exp": exp, "seed": seed,
                                 "true": labels[j] if j < len(labels) else str(j),
                                 "pred": labels[i] if i < len(labels) else str(i),
                                 "count": float(cm[i, j])})
        out = fd / "confusion_matrix.csv"
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        n["confusion"] = len(rows)
    except Exception as e:
        print(f"  [figdata] confusion capture skipped: {e}")

    return n


def capture_predictions(weights, img_dir, class_names, run_id, exp, seed,
                        output_dir, imgsz=640, device=0, conf=0.001, max_images=None):
    """Per-detection dump on the evaluation split.

    Feeds three things: qualitative figure selection, the confidence-distribution
    figure that motivates the method, and error analysis in the discussion.
    """
    from ultralytics import YOLO
    imgs = sorted(Path(img_dir).glob("*.jpg"))
    if max_images:
        imgs = imgs[:max_images]
    model = YOLO(str(weights))
    rows = []
    for r in model.predict(source=[str(p) for p in imgs], imgsz=imgsz, conf=conf,
                           device=device, stream=True, verbose=False):
        name = Path(r.path).name
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xywhn = b.xywhn.cpu().numpy(); cls = b.cls.cpu().numpy(); cf = b.conf.cpu().numpy()
        for k in range(len(cf)):
            rows.append({"run_id": run_id, "exp": exp, "seed": seed, "image": name,
                         "class_id": int(cls[k]),
                         "class_name": class_names.get(int(cls[k]), str(int(cls[k]))),
                         "conf": float(cf[k]),
                         "cx": float(xywhn[k][0]), "cy": float(xywhn[k][1]),
                         "w": float(xywhn[k][2]), "h": float(xywhn[k][3]),
                         "area_norm": float(xywhn[k][2] * xywhn[k][3])})
    df = pd.DataFrame(rows)
    fd = _fig_dir(output_dir)
    df.to_csv(fd / f"predictions_{run_id}.csv", index=False)

    # confidence histogram (compact, appendable across runs)
    if len(df):
        hist, edges = np.histogram(df["conf"], bins=np.linspace(0, 1, 51))
        h = pd.DataFrame({"run_id": run_id, "exp": exp, "seed": seed,
                          "bin_left": edges[:-1], "bin_right": edges[1:], "count": hist})
        out = fd / "val_conf_hist.csv"
        h.to_csv(out, mode="a", header=not out.exists(), index=False)
    return len(df)


def capture_model_info(weights, run_id, exp, seed, output_dir, imgsz=640):
    """Parameter count, layer count and GFLOPs for the cost table (T6).

    ultralytics' `model.info()` return signature varies across versions, so values
    are computed from the module directly instead of trusting that tuple.
    """
    from ultralytics import YOLO
    m = YOLO(str(weights))
    net = m.model
    params = int(sum(p.numel() for p in net.parameters()))
    layers = int(len(list(net.modules())))
    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = float(get_flops(net, imgsz))
    except Exception:
        try:
            info = m.info(verbose=False)
            if info and len(info) >= 4:
                gflops = float(info[3])
        except Exception:
            pass

    row = {"run_id": run_id, "exp": exp, "seed": seed, "imgsz": imgsz,
           "params": params, "layers": layers, "gflops": gflops}
    out = _fig_dir(output_dir) / "model_info.csv"
    df = pd.DataFrame([row])
    if out.exists():
        old = pd.read_csv(out)
        old = old[old.run_id != run_id]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    return row
