from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import pytest

import mcu_data.reporting as reporting_module
from mcu_data.common import sha256_file
from mcu_data.methodology import load_protocol
from mcu_data.reporting import (
    _comparison_terminal_text,
    _per_run_label,
    _plot_aggregate_comparison,
    _protocol_compatibility,
    _write_comparison_markdown,
    compare_runs,
    evaluate_predictions,
    normalize_yolo11_results,
)


def _display_row(seed: int, run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "model": "yolo11m",
        "seed": seed,
        "status": "complete",
        "ap50_95": 0.5,
        "ap50": 0.7,
        "ap75": 0.4,
        "ap_small": 0.2,
        "ar100": 0.6,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.746,
        "tp": 70,
        "fp": 20,
        "fn": 30,
        "best_f1_confidence": 0.25,
        "params": 1_000_000,
        "checkpoint_mib": 10.0,
        "latency_p50_ms": 10.0,
        "latency_p95_ms": 12.0,
        "fps": 90.0,
        "peak_gpu_memory_mib": 4000.0,
        "train_elapsed_s": 600.0,
    }


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


def test_all_per_run_text_labels_include_model_seed_and_run_id(tmp_path: Path) -> None:
    rows = [_display_row(42, "yolo11_seed42"), _display_row(43, "yolo11_seed43")]
    labels = [
        _per_run_label(row["model"], row["seed"], row["run_id"]) for row in rows
    ]
    assert len(set(labels)) == 2
    terminal = _comparison_terminal_text(
        rows,
        {"comparable": True, "release_ready": False, "release_blockers": []},
        [],
    )
    assert all(label in terminal for label in labels)

    report = tmp_path / "experiment_report.md"
    _write_comparison_markdown(
        report,
        rows,
        {"comparable": True, "release_ready": False, "release_blockers": []},
        [],
    )
    text = report.read_text(encoding="utf-8")
    assert "| yolo11m | 42 | yolo11_seed42 |" in text
    assert "| yolo11m | 43 | yolo11_seed43 |" in text


def test_aggregate_plot_has_explicit_sample_sd_error_bars(tmp_path: Path) -> None:
    output = tmp_path / "aggregate_comparison.png"
    _plot_aggregate_comparison(
        [
            {
                "model": "yolo11m",
                "runs": 3,
                "ap50_95_mean": 0.5,
                "ap50_95_std": 0.02,
                "latency_p50_ms_mean": 10.0,
                "latency_p50_ms_std": 0.5,
            },
            {
                "model": "YOLOX-S",
                "runs": 3,
                "ap50_95_mean": 0.45,
                "ap50_95_std": 0.03,
                "latency_p50_ms_mean": 12.0,
                "latency_p50_ms_std": 0.7,
            },
        ],
        output,
        "SOURCE: fixture",
    )
    assert output.is_file() and output.stat().st_size > 0


def test_formal_comparison_refuses_nonempty_output_before_any_write(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    sentinel = output / "stale.txt"
    sentinel.write_text("do-not-touch", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(ValueError, match="must be empty before writing"):
        compare_runs([], output, formal=True)

    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert after == before


def test_formal_comparison_generates_hash_bound_execution_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    protocol_source = repository / "configs" / "experiments" / "baseline_v1.yaml"
    protocol = load_protocol(protocol_source)
    evidence = protocol["dataset"]["evidence"]
    runs = []
    for model in ("yolo11m", "YOLOX-S"):
        for seed in (42, 43, 44):
            run_id = ("yolo11" if model == "yolo11m" else "yolox") + f"_seed{seed}"
            run = tmp_path / "runs" / run_id
            run.mkdir(parents=True)
            snapshot = run / "protocol_snapshot.yaml"
            snapshot.write_bytes(protocol_source.read_bytes())
            checkpoint = run / "best.pt"
            checkpoint.write_bytes(f"checkpoint-{run_id}".encode())
            manifest = {
                "run_id": run_id,
                "model": model,
                "status": "complete",
                "stage": "fine_tune_candidate",
                "best_checkpoint": {
                    "path": "best.pt",
                    "sha256": sha256_file(checkpoint),
                    "mib": checkpoint.stat().st_size / (1024**2),
                },
                "protocol_config": {
                    "path": str(snapshot),
                    "sha256": sha256_file(snapshot),
                },
                "dataset": {
                    "train_annotation_sha256": protocol["dataset"]["coco_annotation_sha256"]["train"],
                    "val_annotation_sha256": protocol["dataset"]["coco_annotation_sha256"]["val"],
                    **evidence,
                },
                "protocol": {
                    "seed": seed,
                    "epochs": 100,
                    "batch": 8,
                    "imgsz": 640,
                    "workers": 0,
                    "amp": True,
                    "fraction": 1.0,
                    "multiscale_range": 0,
                    "prediction_floor": 0.001,
                    "nms_iou": 0.65,
                    "class_agnostic_nms": False,
                    "common_operating_confidence": 0.25,
                    "common_match_iou": 0.5,
                },
                "model_details": {"parameters": 1_000_000},
            }
            (run / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with (run / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "epoch",
                        "elapsed_s",
                        "map50_95",
                        "map50",
                        "train_total_loss",
                        "train_box_loss",
                        "train_peak_allocated_mib",
                    ],
                )
                writer.writeheader()
                for epoch in range(1, 101):
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "elapsed_s": epoch * 10,
                            "map50_95": 0.4 + seed / 1000,
                            "map50": 0.7,
                            "train_total_loss": 1.0 / epoch,
                            "train_box_loss": 0.5 / epoch,
                            "train_peak_allocated_mib": 4000,
                        }
                    )
            metrics = {
                "ap50_95": 0.4 + seed / 1000,
                "ap50": 0.7,
                "ap75": 0.4,
                "ap_small": 0.2,
                "ap_medium": 0.5,
                "ap_large": 0.6,
                "ar100": 0.6,
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.746,
                "tp": 70,
                "fp": 20,
                "fn": 30,
                "best_f1": 0.75,
                "best_f1_confidence": 0.25,
            }
            (run / "final_metrics.json").write_text(
                json.dumps({"metrics": metrics}), encoding="utf-8"
            )
            (run / "latency.json").write_text(
                json.dumps(
                    {"e2e_p50_ms": 10.0, "e2e_p95_ms": 12.0, "sustained_fps": 90.0}
                ),
                encoding="utf-8",
            )
            (run / "gpu_summary.json").write_text(
                json.dumps({"peak_memory_used_mib": 4000.0}), encoding="utf-8"
            )
            (run / "terminal.log").write_text(f"{run_id}\n", encoding="utf-8")
            runs.append(run)

    monkeypatch.setattr(
        reporting_module,
        "verify_run_provenance",
        lambda *args, **kwargs: {
            "schema_version": 2,
            "status": "PASS",
            "mixed_commits": False,
            "blockers": [],
        },
    )
    output = tmp_path / "formal-comparison"
    compare_runs(runs, output, formal=True)

    validation = json.loads((output / "formal_validation.json").read_text(encoding="utf-8"))
    execution = json.loads(
        (output / "formal_execution_status.json").read_text(encoding="utf-8")
    )
    assert validation["status"] == "PASS"
    assert validation["formal_release"] is True
    assert validation["evidence_manifest_sha256"] == sha256_file(
        output / "evidence_manifest.json"
    )
    assert execution["summary"] == "2 models × 3 seeds × 100 epochs"
    assert len(execution["runs"]) == 6
    assert (output / "aggregate_comparison.png").is_file()
    assert (output / "ubuntu_handoff.md").is_file()
    assert (output / "protocol_snapshot.yaml").read_bytes() == protocol_source.read_bytes().replace(
        b"\r\n", b"\n"
    ).replace(b"\r", b"\n")


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


def test_common_evaluator_marks_test_thresholds_diagnostic_only(tmp_path: Path) -> None:
    ground_truth = tmp_path / "test.json"
    predictions = tmp_path / "predictions.json"
    ground_truth.write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "images": [{"id": 1, "file_name": "one.jpg", "width": 32, "height": 32}],
                "categories": [{"id": 1, "name": "chip", "supercategory": "component"}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text("[]", encoding="utf-8")
    output = tmp_path / "test-report"

    result = evaluate_predictions(
        ground_truth,
        predictions,
        output,
        evaluation_set="test",
    )

    assert result["threshold_selection"]["status"] == "NOT_FOR_THRESHOLD_SELECTION"
    assert result["pseudo_label_calibration"]["status"] == "NOT_FOR_THRESHOLD_SELECTION"
    assert all(
        row["status"] == "NOT_FOR_THRESHOLD_SELECTION"
        for row in result["pseudo_label_calibration_by_class"]
    )
    assert "best_f1_confidence" not in result["metrics"]
    assert "pseudo_label_target_precision" not in result["protocol"]
    assert not (output / "autolabel_thresholds.csv").exists()


def test_actual_completed_single_run_comparison_is_diagnostic(tmp_path: Path) -> None:
    raw_run_dir = os.environ.get("MCU_COMPLETED_RUN_DIR")
    if not raw_run_dir:
        pytest.skip("Set MCU_COMPLETED_RUN_DIR to exercise a real completed run artifact")
    run_dir = Path(raw_run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        pytest.fail(f"MCU_COMPLETED_RUN_DIR has no run_manifest.json: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert manifest.get("status") == "complete"

    output_dir = tmp_path / "single-run-summary"
    rows = compare_runs([run_dir], output_dir)

    assert len(rows) == 1
    assert rows[0]["run_id"] == manifest["run_id"]
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "protocol_compatibility.json").is_file()
    assert (output_dir / "experiment_report.md").is_file()
    assert (output_dir / "training_curves.png").is_file()
    assert (output_dir / "comparison_dashboard.png").is_file()
    assert (output_dir / "terminal_summary.png").is_file()
    assert (output_dir / "sources_manifest.json").is_file()
    assert (output_dir / "evidence_manifest.json").is_file()
    assert not (output_dir / "formal_validation.json").exists()


@pytest.mark.parametrize(
    ("status", "stage"),
    (("running", "fine_tune_candidate"), ("complete", "smoke_not_comparable")),
)
def test_incomplete_and_smoke_single_run_comparisons_remain_diagnostic(
    tmp_path: Path,
    status: str,
    stage: str,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    run_dir = tmp_path / f"{stage}-{status}"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "model": "yolo11m",
                "status": status,
                "stage": stage,
                "git": {"commit": commit, "dirty": False, "changed_paths": []},
                "protocol": {"seed": 42},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / f"summary-{stage}-{status}"

    rows = compare_runs([run_dir], output_dir)

    assert rows[0]["status"] == status
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "protocol_compatibility.json").is_file()
    assert (output_dir / "experiment_report.md").is_file()
    assert (output_dir / "evidence_manifest.json").is_file()
    assert not (output_dir / "formal_validation.json").exists()


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
        "dataset": {
            "evidence": {field: f"shared-{field}" for field in evidence_fields},
        },
        "common": {
            "epochs": 100,
            "batch_size": 8,
            "image_size": 640,
            "workers": 0,
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
                    "workers": 0,
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

    runs[0]["metadata"]["protocol"]["workers"] = 1
    worker_mismatch = _protocol_compatibility(runs, expected)
    assert worker_mismatch["comparable"] is False
    assert worker_mismatch["release_ready"] is False
    assert any(item["field"] == "workers" for item in worker_mismatch["critical_mismatches"])
    runs[0]["metadata"]["protocol"]["workers"] = 0

    runs[0]["metadata"]["dataset"].pop("class_map_sha256")
    blocked = _protocol_compatibility(runs, expected)
    assert blocked["comparable"] is False
    assert blocked["release_ready"] is False
    assert any(item["field"] == "dataset_evidence:class_map_sha256" for item in blocked["release_blockers"])

    runs[0]["metadata"]["dataset"]["class_map_sha256"] = "different"
    mismatched = _protocol_compatibility(runs, expected)
    assert mismatched["comparable"] is False
    assert any(item["field"] == "class_map_sha256" for item in mismatched["critical_mismatches"])

    for candidate in runs:
        candidate["metadata"]["dataset"]["class_map_sha256"] = "same-but-not-protocol"
    wrong_declared_dataset = _protocol_compatibility(runs, expected)
    assert wrong_declared_dataset["comparable"] is True
    assert wrong_declared_dataset["release_ready"] is False
    assert any(
        item["field"] == "expected_dataset_evidence:class_map_sha256"
        for item in wrong_declared_dataset["release_blockers"]
    )
