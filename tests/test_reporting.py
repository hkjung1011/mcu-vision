from __future__ import annotations

import csv
import json
from pathlib import Path

from mcu_data.reporting import evaluate_predictions, normalize_yolo11_results


def test_normalize_yolo11_results(tmp_path: Path) -> None:
    source = tmp_path / "results.csv"
    source.write_text(
        "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
        "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
        "metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0\n"
        "0,2.5,1.0,2.0,3.0,0.6,0.7,0.8,0.5,1.1,2.1,3.1,0.01\n",
        encoding="utf-8",
    )
    destination = tmp_path / "epoch_metrics.csv"

    rows = normalize_yolo11_results(source, destination)

    assert rows[0]["epoch"] == 1
    assert rows[0]["map50_95"] == 0.5
    with destination.open("r", encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["train_dfl_loss"] == "3.0"


def test_common_evaluator_handles_no_predictions(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground_truth.json"
    predictions = tmp_path / "predictions.json"
    ground_truth.write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "images": [{"id": 1, "file_name": "one.jpg", "width": 100, "height": 100}],
                "categories": [{"id": 1, "name": "chip", "supercategory": "component"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text("[]", encoding="utf-8")

    result = evaluate_predictions(ground_truth, predictions, tmp_path / "report")

    assert result["metrics"]["tp"] == 0
    assert result["metrics"]["fn"] == 1
    assert result["metrics"]["ap50_95"] == 0.0
    assert result["metrics"]["ap_small"] == 0.0
    assert result["metrics"]["tp"] == 0
    assert result["pseudo_label_calibration"]["status"] == "NOT_AVAILABLE"
    assert (tmp_path / "report" / "confidence_curve.png").exists()
    assert (tmp_path / "report" / "autolabel_thresholds.csv").exists()
    assert (tmp_path / "report" / "confusion_counts.csv").read_text(encoding="utf-8").splitlines()[1].endswith(",0,1")
    assert (tmp_path / "report" / "confusion_normalized.png").exists()
