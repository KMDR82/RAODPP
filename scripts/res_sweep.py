from pathlib import Path
from ultralytics import YOLO

R = Path.home() / "RAODpp"
w = R / "model" / "runs" / "E0_visdrone_s42" / "weights" / "best.pt"
y = R / "dataset" / "visdrone_raodpp.yaml"

print(f"{'imgsz':>6} {'mAP50':>8} {'mAP50-95':>9} {'ms/img':>8}")
for sz in [640, 960, 1280, 1536]:
    r = YOLO(str(w)).val(data=str(y), split="val", imgsz=sz, batch=4,
                         device=0, workers=0, verbose=False, plots=False)
    ms = sum(r.speed.values())
    print(f"{sz:>6} {r.box.map50:>8.4f} {r.box.map:>9.4f} {ms:>8.1f}", flush=True)
