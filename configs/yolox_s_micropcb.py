from __future__ import annotations

import os
from pathlib import Path

from yolox.exp import Exp as YOLOXBaseExp


def _environment_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _environment_path(name: str, default: Path) -> str:
    return str(Path(os.environ.get(name, str(default))).resolve())


class Exp(YOLOXBaseExp):
    def __init__(self) -> None:
        super().__init__()
        project_root = Path(__file__).resolve().parents[1]
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = Path(__file__).stem
        self.data_dir = _environment_path(
            "MCU_COCO_DATA_DIR",
            project_root / "data" / "processed" / "micropcb_rpi_phash_v2_coco",
        )
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.num_classes = _environment_int("MCU_NUM_CLASSES", 1)
        image_size = _environment_int("MCU_IMAGE_SIZE", 640)
        self.input_size = (image_size, image_size)
        self.test_size = (image_size, image_size)
        # YOLOX paper section 2.2 reports that S/Tiny/Nano work better without MixUp
        # and with a narrower Mosaic scale range than the base L/X recipe.
        self.enable_mixup = False
        self.mosaic_scale = (0.5, 1.5)
        # Fixed 640 makes accuracy/latency comparison with YOLO11 reproducible.
        self.multiscale_range = _environment_int("MCU_MULTISCALE_RANGE", 0)
        self.max_epoch = _environment_int("MCU_EPOCHS", 100)
        self.data_num_workers = _environment_int("MCU_WORKERS", 0)
        self.eval_interval = _environment_int("MCU_EVAL_INTERVAL", 1)
        self.print_interval = _environment_int("MCU_PRINT_INTERVAL", 10)
        self.no_aug_epochs = min(_environment_int("MCU_NO_AUG_EPOCHS", 10), self.max_epoch)
        self.seed = _environment_int("MCU_SEED", 42)
        self.test_conf = float(os.environ.get("MCU_PREDICTION_FLOOR", "0.001"))
        self.nmsthre = float(os.environ.get("MCU_NMS_IOU", "0.65"))
        self.output_dir = os.environ.get(
            "MCU_OUTPUT_ROOT", str(project_root / "runs" / "benchmarks")
        )
        # Per-epoch history would exceed 7 GB for 100 epochs. best/latest/last are retained.
        self.save_history_ckpt = False

    def get_trainer(self, args):
        from mcu_data.yolox_metrics import MetricsTrainer

        return MetricsTrainer(self, args)

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from mcu_data.yolox_metrics import AuditedCOCOEvaluator

        return AuditedCOCOEvaluator(
            dataloader=self.get_eval_loader(
                batch_size, is_distributed, testdev=testdev, legacy=legacy
            ),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            per_class_AP=True,
            per_class_AR=True,
        )
