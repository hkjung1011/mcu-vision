from __future__ import annotations

import csv
import json
from pathlib import Path

from mcu_data.reporting import _protocol_compatibility, evaluate_predictions, normalize_yolo11_results


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


def test_formal_release_gate_requires_full_seed_matrix_and_dataset_evidence() -> None:
    evidence_fields = [
        "canonical_dataset_manifest_sha256",
        "class_map_sha256",
        "train_image_list_sha256",
        "val_image_list_sha256",
        "canonical_train_records_sha256",
        "canonical_val_records_sha256",
    ]
    expected = {
        "common": {
            "epochs": 100,
            "batch_size": 8,
            "image_size": 640,
            "amp": True,
            "seeds": [42, 43, 44],
            "prediction_floor": 0.001,
            "nms_iou": 0.65,
            "class_agnostic_nms": False,
            "operating_confidence": 0.25,
            "operating_match_iou": 0.5,
        },
        "comparison_rules": {
            "required_models": ["yolo11m", "YOLOX-S"],
            "required_dataset_evidence": evidence_fields,
        },
    }

    def run(model: str, seed: int) -> dict[str, object]:
        run_id = f"{model}_{seed}"
        dataset = {
            "train_annotation_sha256": "train",
            "val_annotation_sha256": "val",
            **{field: f"shared-{field}" for field in evidence_fields},
        }
        return {
            "run_id": run_id,
            "model": model,
            "metadata": {
                "status": "complete",
                "stage": "full_finetune",
                "dataset": dataset,
                "protocol_config": {"sha256": "protocol"},
                "protocol": {
                    "seed": seed,
                    "epochs": 100,
                    "batch": 8,
                    "imgsz": 640,
                    "amp": True,
                    "fraction": 1.0,
                    "multiscale_range": 0,
                    "prediction_floor": 0.001,
                    "nms_iou": 0.65,
                    "class_agnostic_nms": False,
                    "common_operating_confidence": 0.25,
                    "common_match_iou": 0.5,
                },
            },
            "epochs": [{"epoch": index + 1} for index in range(100)],
            "metrics": {
                "ap50_95": 0.5,
                "ap50": 0.7,
                "ap75": 0.4,
                "ar100": 0.6,
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.7466666667,
                "tp": 70,
                "fp": 20,
                "fn": 30,
            },
            "latency": {"e2e_p50_ms": 10.0, "e2e_p95_ms": 12.0, "sustained_fps": 90.0},
            "gpu": {"peak_memory_used_mib": 4000.0},
            "evidence_status": {
                "epoch_metrics_exists": True,
                "final_metrics_exists": True,
                "latency_exists": True,
                "gpu_summary_exists": True,
                "checkpoint_exists": True,
                "checkpoint_hash_matches": True,
            },
        }

    runs = [run(model, seed) for model in ("yolo11m", "YOLOX-S") for seed in (42, 43, 44)]
    result = _protocol_compatibility(runs, expected)
    assert result["comparable"] is True
    assert result["release_ready"] is True

    missing_seed = _protocol_compatibility(runs[:-1], expected)
    assert missing_seed["release_ready"] is False
    assert any(
        item["field"] == "complete_model_seed_matrix"
        for item in missing_seed["release_blockers"]
    )

    runs[0]["metadata"]["protocol"]["epochs"] = 1
    runs[0]["epochs"] = [{"epoch": 1}]
    one_epoch = _protocol_compatibility(runs, expected)
    assert one_epoch["release_ready"] is False
    assert any(item["field"] == "expected:epochs" for item in one_epoch["release_blockers"])
    runs[0]["metadata"]["protocol"]["epochs"] = 100
    runs[0]["epochs"] = [{"epoch": index + 1} for index in range(100)]

    expected["_loaded_source_sha256"] = "different-protocol"
    changed_protocol = _protocol_compatibility(runs, expected)
    assert changed_protocol["release_ready"] is False
    assert any(
        item["field"] == "protocol_snapshot_integrity"
        for item in changed_protocol["release_blockers"]
    )
    expected.pop("_loaded_source_sha256")

    runs[0]["metadata"]["dataset"].pop("class_map_sha256")
    blocked = _protocol_compatibility(runs, expected)
    assert blocked["comparable"] is False
    assert blocked["release_ready"] is False
    assert any(item["field"] == "dataset_evidence:class_map_sha256" for item in blocked["release_blockers"])

    runs[0]["metadata"]["dataset"]["class_map_sha256"] = "different"
    mismatched = _protocol_compatibility(runs, expected)
    assert mismatched["comparable"] is False
    assert any(item["field"] == "class_map_sha256" for item in mismatched["critical_mismatches"])
