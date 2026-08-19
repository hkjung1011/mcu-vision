from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import mcu_data.reporting as reporting_module
from mcu_data.common import sha256_file
from mcu_data.methodology import load_protocol
from mcu_data.publishing import (
    validate_comparison_for_run,
    validate_formal_comparison,
    validated_formal_publication_plan,
)
from mcu_data.release_policy import (
    load_formal_release_policy,
    public_policy_binding,
)
from mcu_data.reporting import _protocol_compatibility, compare_runs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PROTOCOL = PROJECT_ROOT / "configs" / "experiments" / "baseline_v1.yaml"
POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "rpi_bootstrap_paired_2seed_release_v1.yaml"
)
ATTESTATION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "mixed_commit_rpi_paired_2seed_v1_attestation.json"
)


def _policy() -> dict[str, Any]:
    return load_formal_release_policy(POLICY_PATH, base_protocol_path=BASE_PROTOCOL)


def _memory_run(model: str, seed: int, *, epochs: int = 100, run_id: str | None = None) -> dict[str, Any]:
    protocol = load_protocol(BASE_PROTOCOL)
    evidence = protocol["dataset"]["evidence"]
    return {
        "run_id": run_id or f"{model}_{seed}",
        "model": model,
        "metadata": {
            "status": "complete",
            "stage": "full_finetune",
            "dataset": {
                "train_annotation_sha256": protocol["dataset"]["coco_annotation_sha256"]["train"],
                "val_annotation_sha256": protocol["dataset"]["coco_annotation_sha256"]["val"],
                **evidence,
            },
            "protocol_config": {"sha256": _policy()["base_protocol_sha256"]},
            "protocol": {
                "seed": seed,
                "epochs": epochs,
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
        "epochs": [{"epoch": value} for value in range(1, epochs + 1)],
        "metrics": {
            "ap50_95": 0.5,
            "ap50": 0.7,
            "ap75": 0.4,
            "ar100": 0.6,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.746,
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


def _expected_protocol() -> dict[str, Any]:
    document = load_protocol(BASE_PROTOCOL)
    document["_loaded_source_sha256"] = _policy()["base_protocol_sha256"]
    document["_loaded_source_path"] = str(BASE_PROTOCOL)
    return document


def test_policy_is_exactly_bound_to_immutable_base_protocol() -> None:
    policy = _policy()
    assert policy["expected_runs"] == 4
    assert policy["seeds"] == [42, 43]
    assert policy["degrees_of_freedom"] == 1
    assert policy["interpretation"] == "descriptive_only"
    attestation = json.loads(ATTESTATION_PATH.read_text(encoding="utf-8"))
    assert attestation["formal_release_policy"] == {
        key: public_policy_binding(policy)[key]
        for key in (
            "policy_id",
            "policy_sha256",
            "base_protocol_id",
            "base_protocol_sha256",
            "evidence_tier",
            "models",
            "seeds",
            "expected_runs",
        )
    }


def test_policy_tamper_and_base_hash_mismatch_fail_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "policy.yaml"
    text = POLICY_PATH.read_text(encoding="utf-8")
    tampered.write_text(text.replace("seeds: [42, 43]", "seeds: [42, 44]"), encoding="utf-8")
    with pytest.raises(ValueError, match="approved frozen policy"):
        load_formal_release_policy(tampered, base_protocol_path=BASE_PROTOCOL)

    wrong_base = tmp_path / "wrong-base.yaml"
    wrong_base.write_text(
        BASE_PROTOCOL.read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base protocol SHA-256 mismatch"):
        load_formal_release_policy(POLICY_PATH, base_protocol_path=wrong_base)


def test_paired_matrix_is_fail_closed_for_99_missing_extra_duplicate_and_seed44() -> None:
    policy = _policy()
    expected = _expected_protocol()
    runs = [
        _memory_run(model, seed)
        for model in ("yolo11m", "YOLOX-S")
        for seed in (42, 43)
    ]
    assert _protocol_compatibility(runs, expected, policy)["release_ready"] is True

    ninety_nine = [dict(run) for run in runs]
    ninety_nine[0] = _memory_run("yolo11m", 42, epochs=99)
    assert _protocol_compatibility(ninety_nine, expected, policy)["release_ready"] is False

    assert _protocol_compatibility(runs[:-1], expected, policy)["release_ready"] is False

    seed44 = runs + [_memory_run("yolo11m", 44)]
    result = _protocol_compatibility(seed44, expected, policy)
    assert result["release_ready"] is False
    assert any(item["field"] == "complete_model_seed_matrix" for item in result["release_blockers"])

    duplicate = runs + [_memory_run("yolo11m", 42, run_id="duplicate_yolo11m_42")]
    result = _protocol_compatibility(duplicate, expected, policy)
    assert result["release_ready"] is False
    assert any(item["field"] == "duplicate_model_seed_pair" for item in result["release_blockers"])

    unpaired_fifth = runs + [_memory_run("YOLOX-S", 44)]
    assert _protocol_compatibility(unpaired_fifth, expected, policy)["release_ready"] is False


def _write_disk_run(root: Path, model: str, seed: int) -> Path:
    protocol = load_protocol(BASE_PROTOCOL)
    evidence = protocol["dataset"]["evidence"]
    run_id = ("yolo11" if model == "yolo11m" else "yolox") + f"_seed{seed}"
    run = root / run_id
    run.mkdir(parents=True)
    snapshot = run / "protocol_snapshot.yaml"
    snapshot.write_text(
        BASE_PROTOCOL.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n"),
        encoding="utf-8",
        newline="\n",
    )
    checkpoint = run / ("best.pt" if model == "yolo11m" else "best.pth")
    checkpoint.write_bytes(f"checkpoint-{run_id}".encode())
    manifest = {
        "run_id": run_id,
        "model": model,
        "status": "complete",
        "stage": "fine_tune_candidate",
        "git": {"commit": "a" * 40, "dirty": False, "changed_paths": []},
        "best_checkpoint": {
            "path": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "mib": checkpoint.stat().st_size / (1024**2),
        },
        "protocol_config": {"path": str(snapshot), "sha256": sha256_file(snapshot)},
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
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
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
    model_offset = 0.05 if model == "yolo11m" else 0.0
    metrics = {
        "ap50_95": 0.4 + model_offset + seed / 1000,
        "ap50": 0.7 + model_offset,
        "ap75": 0.4 + model_offset,
        "ap_small": 0.2,
        "ap_medium": 0.5,
        "ap_large": 0.6,
        "ar100": 0.6 + model_offset,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.746,
        "tp": 70,
        "fp": 20,
        "fn": 30,
        "best_f1": 0.75,
        "best_f1_confidence": 0.25,
    }
    (run / "final_metrics.json").write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    (run / "latency.json").write_text(
        json.dumps({"e2e_p50_ms": 10.0, "e2e_p95_ms": 12.0, "sustained_fps": 90.0}),
        encoding="utf-8",
    )
    (run / "gpu_summary.json").write_text(
        json.dumps({"peak_memory_used_mib": 4000.0}), encoding="utf-8"
    )
    (run / "terminal.log").write_text(run_id + "\n", encoding="utf-8")
    return run


def test_paired_formal_output_binds_policy_statistics_and_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = [
        _write_disk_run(tmp_path / "runs", model, seed)
        for model in ("yolo11m", "YOLOX-S")
        for seed in (42, 43)
    ]
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
    output = tmp_path / "formal"
    compare_runs(
        runs,
        output,
        provenance_attestation=ATTESTATION_PATH,
        formal_release_policy=POLICY_PATH,
        formal=True,
    )
    validation = json.loads((output / "formal_validation.json").read_text(encoding="utf-8"))
    execution = json.loads(
        (output / "formal_execution_status.json").read_text(encoding="utf-8")
    )
    assert validation["run_count"] == 4
    assert validation["evidence_tier"] == "paired_2seed_descriptive"
    assert validation["paired_n"] == 2
    assert validation["degrees_of_freedom"] == 1
    assert validation["paired_seed_deltas"] == execution["paired_seed_deltas"]
    assert validate_formal_comparison(output) == validation
    terminal = (output / "comparison_terminal.txt").read_text(encoding="utf-8")
    assert "mean ± sample standard deviation" in terminal
    assert "+/-" not in terminal
    publication_plan = validated_formal_publication_plan(
        output,
        require_local_originals=True,
    )
    assert publication_plan["scan"]["weight_files"] == []
    assert publication_plan["scan"]["raw_image_files"] == []
    assert (output / "formal_release_policy.yaml").is_file()
    run_manifest = runs[0] / "run_manifest.json"
    deployment_bridge = validate_comparison_for_run(
        output,
        run_id="yolo11_seed42",
        run_manifest_path=run_manifest,
    )
    assert deployment_bridge["policy_id"] == validation["policy_id"]
    assert deployment_bridge["policy_sha256"] == validation["policy_sha256"]
    assert deployment_bridge["evidence_tier"] == "paired_2seed_descriptive"

    policy_tamper = tmp_path / "policy-tamper"
    shutil.copytree(output, policy_tamper)
    policy_file = policy_tamper / "formal_release_policy.yaml"
    policy_file.write_text(
        policy_file.read_text(encoding="utf-8").replace(
            "paired_2seed_descriptive", "paired_2seed_descriptive_tampered", 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="approved frozen policy|evidence_tier|binding"):
        validate_formal_comparison(policy_tamper)

    attestation_tamper = tmp_path / "attestation-tamper"
    shutil.copytree(output, attestation_tamper)
    attestation_path = attestation_tamper / "run_provenance_attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["formal_release_policy"]["policy_sha256"] = "0" * 64
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    with pytest.raises(ValueError, match="attestation policy binding"):
        validate_formal_comparison(attestation_tamper)
