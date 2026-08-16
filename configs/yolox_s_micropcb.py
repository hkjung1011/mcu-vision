from __future__ import annotations

from pathlib import Path

from yolox.exp import Exp as YOLOXBaseExp


class Exp(YOLOXBaseExp):
    def __init__(self) -> None:
        super().__init__()
        project_root = Path(__file__).resolve().parents[1]
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = Path(__file__).stem
        self.data_dir = str(project_root / "data" / "processed" / "micropcb_rpi_coco")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.num_classes = 1
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.max_epoch = 100
        self.data_num_workers = 2
        self.eval_interval = 1
        self.print_interval = 10
        self.no_aug_epochs = 10
        self.seed = 42
