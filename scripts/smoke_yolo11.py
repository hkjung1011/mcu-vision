from __future__ import annotations

import json
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "processed" / "micropcb_rpi_phash_v2" / "dataset.yaml"


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    matrix = torch.rand((512, 512), device="cuda")
    checksum = float((matrix @ matrix).mean().cpu())

    model = YOLO("yolo11n.yaml")
    result = model.train(
        data=str(DATASET),
        epochs=1,
        imgsz=320,
        batch=2,
        workers=0,
        device=0,
        amp=False,
        fraction=0.002,
        val=False,
        plots=False,
        save=False,
        cache=False,
        project=str(PROJECT_ROOT / "runs" / "smoke"),
        name="yolo11n_cuda",
        exist_ok=True,
        verbose=False,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "gpu": torch.cuda.get_device_name(0),
                "cuda_matrix_checksum": checksum,
                "dataset": str(DATASET),
                "save_dir": str(result.save_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
