import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)   # 3.14's forkserver cannot start our workers
from ultralytics import YOLO
from pathlib import Path

R = Path.home() / "RAODpp"
for seed in [42, 123]:
    print(f"=== workers=8 seed {seed} ===", flush=True)
    YOLO("yolov8s.pt").train(
        data=str(R / "dataset" / "uavdt_raodpp.yaml"), epochs=5, imgsz=640, batch=32,
        workers=8, device=0, seed=seed, deterministic=False, cache=False,
        project=str(R / "model" / "runs"), name=f"DIAG_w8_s{seed}",
        exist_ok=True, verbose=False, plots=False)
