from __future__ import annotations

import json
from pathlib import Path

from yolox.exp import get_exp


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    experiment = get_exp(str(PROJECT_ROOT / "configs" / "yolox_s_micropcb.py"), None)
    train_dataset = experiment.get_dataset(cache=False)
    val_dataset = experiment.get_eval_dataset()
    train_image, train_target, train_info, train_id = train_dataset.pull_item(0)
    print(
        json.dumps(
            {
                "status": "PASS",
                "num_classes": experiment.num_classes,
                "train_images": len(train_dataset),
                "val_images": len(val_dataset),
                "first_image_shape": list(train_image.shape),
                "first_target_shape": list(train_target.shape),
                "first_image_info": list(train_info),
                "first_image_id": int(train_id[0]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
