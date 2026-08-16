from __future__ import annotations

import json

import torch
import yolox
from yolox.exp import get_exp


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    experiment = get_exp(None, "yolox-s")
    model = experiment.get_model().eval().cuda()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 640, 640), device="cuda"))
    print(
        json.dumps(
            {
                "status": "PASS",
                "yolox": yolox.__version__,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "output_shape": list(output.shape),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
