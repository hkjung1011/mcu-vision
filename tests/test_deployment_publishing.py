from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcu_data.common import sha256_file
from mcu_data.publishing import create_formal_validation, validate_formal_comparison
from mcu_data.deployment_publishing import (
    promote_deployment_release,
    validate_promoted_deployment_for_runtime,
)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _record(path: Path, project: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(project.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _build_release_fixture(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".gitattributes").write_text(
        "*.pt filter=lfs diff=lfs merge=lfs -text\n"
        "*.pth filter=lfs diff=lfs merge=lfs -text\n"
        "*.onnx filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    release = "release_yolo11_seed42"
    run_id = "yolo11_seed42"
    release_weights = project / "weights" / "trained" / release
    checkpoint = release_weights / "best.pt"
    onnx = release_weights / "yolo11m" / "model.onnx"
    checkpoint.parent.mkdir(parents=True)
    onnx.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trained-native-checkpoint")
    onnx.write_bytes(b"verified-onnx-graph")
    source_original_sha = "1" * 64
    comparison = project / "runs" / "comparisons" / "campaign_full"
    provenance_document = {"status": "PASS", "schema_version": 2, "mixed_commits": False}
    provenance_path = _write_json(comparison / "run_provenance.json", provenance_document)
    attestation_path = _write_json(
        comparison / "run_provenance_attestation.json",
        {"schema_version": 2, "required_scope_version": "formal-training-provenance-v2"},
    )
    compatibility_path = _write_json(
        comparison / "protocol_compatibility.json",
        {
            "release_ready": True,
            "comparable": True,
            "critical_mismatches": [],
            "release_blockers": [],
            "run_count": 6,
            "release_expectations": {
                "models": ["yolo11m", "YOLOX-S"],
                "seeds": [42, 43, 44],
                "runs": 6,
            },
            "run_provenance": provenance_document,
        },
    )
    dataset = {
        "train_annotation_sha256": "2" * 64,
        "val_annotation_sha256": "3" * 64,
        **{field: "4" * 64 for field in (
            "canonical_dataset_manifest_sha256", "class_map_sha256",
            "train_image_list_sha256", "val_image_list_sha256",
            "canonical_train_records_sha256", "canonical_val_records_sha256",
        )},
    }
    metrics_document = {"metrics": {
        "ap50_95": 0.5, "ap50": 0.7, "ap75": 0.4, "ar100": 0.6,
        "precision": 0.8, "recall": 0.7, "f1": 0.746, "tp": 70, "fp": 20, "fn": 30,
    }}
    rows = []
    source_records = []
    local_bindings = []
    selected: dict[str, Path] = {}
    for model in ("yolo11m", "YOLOX-S"):
        for seed in (42, 43, 44):
            candidate_id = ("yolo11" if model == "yolo11m" else "yolox") + f"_seed{seed}"
            source_dir = project / "runs" / "benchmarks" / candidate_id
            manifest = _write_json(source_dir / "run_manifest.json", {
                "run_id": candidate_id, "status": "complete", "stage": "fine_tune_candidate",
                "model": model, "best_checkpoint": {"sha256": source_original_sha if candidate_id == run_id else "5" * 64},
                "protocol": {
                    "seed": seed, "epochs": 100, "batch": 8, "imgsz": 640, "workers": 0,
                    "amp": True, "fraction": 1.0, "multiscale_range": 0, "prediction_floor": 0.001,
                    "nms_iou": 0.65, "class_agnostic_nms": False,
                    "common_operating_confidence": 0.25, "common_match_iou": 0.5,
                },
                "dataset": dataset, "protocol_config": {"sha256": "6" * 64},
            })
            final_metrics = _write_json(
                (project / "runs" / candidate_id / "final_metrics.json") if candidate_id == run_id else source_dir / "final_metrics.json",
                metrics_document,
            )
            epoch_metrics = source_dir / "epoch_metrics.csv"
            epoch_metrics.write_text("epoch,map50\n" + "".join(f"{i},0.5\n" for i in range(1, 101)), encoding="utf-8")
            latency = _write_json(source_dir / "latency.json", {"e2e_p50_ms": 10.0, "e2e_p95_ms": 12.0, "sustained_fps": 90.0})
            gpu = _write_json(source_dir / "gpu_summary.json", {"peak_memory_used_mib": 4000.0})
            rows.append({
                "run_id": candidate_id, "model": model, "seed": seed, "status": "complete",
                **metrics_document["metrics"],
            })
            for source in (manifest, epoch_metrics, final_metrics, latency, gpu):
                destination = comparison / "sources" / candidate_id / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
                source_records.append({
                    "run_id": candidate_id,
                    "path": destination.relative_to(comparison).as_posix(),
                    "source_original_sha256": sha256_file(source),
                    "published_sha256": sha256_file(destination),
                    "bytes": destination.stat().st_size,
                })
                local_bindings.append({
                    "run_id": candidate_id,
                    "filename": source.name,
                    "source_path": str(source.resolve()),
                    "source_original_sha256": sha256_file(source),
                })
            if candidate_id == run_id:
                selected = {"manifest": manifest, "metrics": final_metrics}
    run_manifest = selected["manifest"]
    native_metrics = selected["metrics"]
    run_manifest_sha = sha256_file(run_manifest)
    comparison_path = _write_json(comparison / "comparison.json", rows)
    sources_path = _write_json(comparison / "sources_manifest.json", {"files": source_records})
    _write_json(
        comparison / "local_source_bindings.json",
        {"schema_version": 1, "private_local_only": True, "files": local_bindings},
    )
    formal_validation_path = comparison / "formal_validation.json"
    create_formal_validation(comparison)
    comparison_evidence = {
        "comparison_id": comparison.name,
        "protocol_compatibility_sha256": sha256_file(compatibility_path),
        "comparison_sha256": sha256_file(comparison_path),
        "sources_manifest_sha256": sha256_file(sources_path),
        "run_provenance_sha256": sha256_file(provenance_path),
        "run_provenance_attestation_sha256": sha256_file(attestation_path),
        "formal_validation_sha256": sha256_file(formal_validation_path),
        "run_manifest_sha256": run_manifest_sha,
        "native_final_metrics_sha256": sha256_file(native_metrics),
        "run_id": run_id,
    }

    native_artifact = project / "reports" / "runs" / release / "artifact_manifest.json"
    checkpoint_publication = {
        "source_original_sha256": source_original_sha,
        "metadata_sanitized": True,
        "state_dict_bitwise_equal": True,
        "forward_max_abs_difference": 0.0,
        "source_forward_captured_before_scrub": True,
        "ultralytics_load": "PASS",
    }
    _write_json(
        native_artifact,
        {
            "release_name": release,
            "source_run_id": run_id,
            "source_run_manifest_sha256": run_manifest_sha,
            "validated_by_comparison": comparison_evidence,
            "local_source_path_included": False,
            "model": "yolo11m",
            "stage": "fine_tune_candidate",
            "checkpoint": _record(checkpoint, project) | checkpoint_publication,
        },
    )

    deployment_metadata = release_weights / "yolo11m" / "model.deployment.json"
    _write_json(
        deployment_metadata,
        {
            "schema_version": 1,
            "status": "PASS",
            "profile": "fixed_batch1_fp32_onnx",
            "framework": "yolo11",
            "model_input": {
                "batch": 1,
                "height": 640,
                "width": 640,
                "dtype": "float32",
            },
            "classes": {"names": ["raspberry_pi_sbc"]},
            "postprocessing": {"confidence": 0.25, "nms_iou": 0.65, "max_detections": 100},
            "training_run": {
                "run_id": run_id,
                "status": "complete",
                "stage": "fine_tune_candidate",
                "model": "yolo11m",
            },
            "release_validation": {
                "status": "PASS",
                "formal_release": True,
                **comparison_evidence,
            },
            "checkpoint_publication": {
                **checkpoint_publication,
                "published_sha256": sha256_file(checkpoint),
            },
            "verification": {"status": "PASS", "numeric": {"status": "PASS"}},
            "artifacts": {
                "checkpoint": _record(checkpoint, project),
                "run_manifest": _record(run_manifest, project),
                "onnx": {**_record(onnx, project), "file_name": onnx.name},
            },
        },
    )

    protocol = _write_json(project / "evidence" / "protocol.json", {"id": "protocol_v2"})
    split_manifest = project / "data" / "split.csv"
    split_manifest.parent.mkdir(parents=True)
    split_manifest.write_text("split,name\n", encoding="utf-8")
    split_summary = _write_json(project / "data" / "split.summary.json", {"status": "PASS"})
    coco_val = _write_json(project / "data" / "val.json", {"images": [], "annotations": []})
    coco_test = _write_json(project / "data" / "test.json", {"images": [], "annotations": []})
    def evaluation(split: str, coco: Path, native: bool) -> Path:
        directory = project / "runs" / "deployment_eval" / run_id / split
        predictions = _write_json(directory / "predictions.coco.json", [])
        metrics = {
            "ap50_95": 0.81 if split == "val" else 0.79,
            "ap50": 0.95,
            "precision": 0.91,
            "recall": 0.89,
        }
        final_metrics = _write_json(
            directory / "final_metrics.json",
            {
                "evaluation_set": split,
                "ground_truth": {"sha256": sha256_file(coco)},
                "predictions": {"sha256": sha256_file(predictions)},
                "metrics": metrics,
            },
        )
        image_manifest = _write_json(
            directory / "image_manifest.json",
            {
                "split": split,
                "image_count": 1,
                "images": [
                    {
                        "file_name": f"{split}.jpg",
                        "sha256": "2" * 64,
                        "synthetic_or_generated": False,
                        "path": str(project / "data" / f"{split}.jpg"),
                    }
                ],
            },
        )
        artifacts: dict[str, Any] = {
            "deployment_metadata": _record(deployment_metadata, project),
            "onnx": _record(onnx, project),
            "coco_annotations": _record(coco, project),
            "image_manifest": _record(image_manifest, project),
            "predictions": _record(predictions, project),
            "final_metrics": _record(final_metrics, project),
        }
        if native:
            artifacts["native_final_metrics"] = _record(native_metrics, project)
        summary = {
            "schema_version": 1,
            "status": "PASS",
            "mode": "formal",
            "split": split,
            "framework": "yolo11",
            "profile": "fixed_batch1_fp32_onnxruntime",
            "protocol": {
                "batch": 1,
                "image_size": 640,
                "prediction_floor_for_coco_ap": 0.001,
                "nms_iou": 0.65,
                "max_detections_per_image": 100,
                "operating_confidence": 0.25,
                "operating_match_iou": 0.5,
            },
            "inputs": {
                "image_count": 1,
                "category_names": ["raspberry_pi_sbc"],
                "protocol_binding": {
                    "status": "PASS",
                    "protocol_id": "protocol_v2",
                    "artifact": _record(protocol, project),
                },
                "split_binding": {
                    "status": "PASS",
                    "assignment_sha256": "3" * 64,
                    "artifacts": {
                        "split_manifest": _record(split_manifest, project),
                        "split_summary": _record(split_summary, project),
                    },
                },
            },
            "artifacts": artifacts,
            "metrics": metrics,
            "native_metric_equivalence": {
                "status": "PASS" if native else "NOT_REQUESTED"
            },
            "native_reference_binding": (
                {
                    "status": "PASS",
                    "expected_sha256": sha256_file(native_metrics),
                    "actual_sha256": sha256_file(native_metrics),
                    "binding": "deployment.release_validation.native_final_metrics_sha256",
                }
                if native
                else {"status": "NOT_APPLICABLE"}
            ),
        }
        return _write_json(directory / "onnx_split_evaluation.json", summary)

    val_summary = evaluation("val", coco_val, native=True)
    test_summary = evaluation("test", coco_test, native=False)
    return {
        "project": project,
        "native_artifact": native_artifact,
        "deployment_metadata": deployment_metadata,
        "onnx": onnx,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "comparison": comparison,
        "native_metrics": native_metrics,
    }


def _promote(paths: dict[str, Path]) -> dict[str, Any]:
    return promote_deployment_release(
        project_root=paths["project"],
        native_artifact_path=paths["native_artifact"],
        deployment_metadata_path=paths["deployment_metadata"],
        onnx_path=paths["onnx"],
        val_summary_path=paths["val_summary"],
        test_summary_path=paths["test_summary"],
        comparison_dir=paths["comparison"],
    )


def test_promote_deployment_creates_sanitized_hash_bound_release(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)

    manifest = _promote(paths)

    destination = paths["project"] / "reports" / "deployments" / manifest["release_name"]
    assert manifest["status"] == "PASS"
    assert manifest["gates"]["validation_native_metric_equivalence"] == "PASS"
    assert (destination / "deployment_release_manifest.json").is_file()
    assert (destination / "deployment_metadata.json").is_file()
    assert (destination / "val" / "onnx_split_evaluation.json").is_file()
    assert (destination / "test" / "final_metrics.json").is_file()
    assert not (destination / "val" / "predictions.coco.json").exists()
    published_image_manifest = (destination / "val" / "image_manifest.json").read_text(
        encoding="utf-8"
    )
    assert str(paths["project"]) not in published_image_manifest
    assert "<PROJECT_ROOT>" in published_image_manifest
    runtime_evidence = validate_promoted_deployment_for_runtime(
        project_root=paths["project"],
        release_manifest_path=destination / "deployment_release_manifest.json",
        deployment_metadata_path=paths["deployment_metadata"],
        onnx_path=paths["onnx"],
    )
    assert runtime_evidence["status"] == "PASS"


def test_promote_deployment_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    paths["onnx"].parent.parent.joinpath("best.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _promote(paths)


def test_promote_deployment_rejects_missing_yolo11_sanitizer_proof(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    native = json.loads(paths["native_artifact"].read_text(encoding="utf-8"))
    native["checkpoint"]["metadata_sanitized"] = False
    _write_json(paths["native_artifact"], native)

    with pytest.raises(ValueError, match="sanitizer evidence"):
        _promote(paths)


def test_promote_deployment_rejects_checkpoint_publication_bridge_tamper(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    deployment = json.loads(paths["deployment_metadata"].read_text(encoding="utf-8"))
    deployment["checkpoint_publication"]["source_original_sha256"] = "f" * 64
    _write_json(paths["deployment_metadata"], deployment)

    with pytest.raises(ValueError, match="publication bridge"):
        _promote(paths)


def test_promote_deployment_rejects_local_path_in_tracked_metadata(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    metadata = json.loads(paths["deployment_metadata"].read_text(encoding="utf-8"))
    metadata["local_debug_path"] = str(paths["project"].resolve())
    _write_json(paths["deployment_metadata"], metadata)

    with pytest.raises(ValueError, match="absolute path"):
        _promote(paths)


def test_promote_deployment_rejects_provenance_file_tamper(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    provenance = paths["comparison"] / "run_provenance.json"
    document = json.loads(provenance.read_text(encoding="utf-8"))
    document["status"] = "FAIL"
    _write_json(provenance, document)

    with pytest.raises(ValueError, match="provenance"):
        _promote(paths)


def test_promote_deployment_rejects_missing_formal_validation(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    (paths["comparison"] / "formal_validation.json").unlink()

    with pytest.raises(FileNotFoundError, match="formal_validation.json"):
        _promote(paths)


def test_promote_deployment_rejects_duplicate_source_bundle_record(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    sources = paths["comparison"] / "sources_manifest.json"
    document = json.loads(sources.read_text(encoding="utf-8"))
    document["files"].append(dict(document["files"][0]))
    _write_json(sources, document)

    with pytest.raises(ValueError, match="paths must be unique"):
        _promote(paths)


def test_promote_deployment_rejects_comparison_metric_forgery(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    comparison = paths["comparison"] / "comparison.json"
    rows = json.loads(comparison.read_text(encoding="utf-8"))
    rows[0]["ap50_95"] = 0.999
    _write_json(comparison, rows)

    with pytest.raises(ValueError, match="row metric differs"):
        _promote(paths)


def test_promote_deployment_rejects_row_seed_swap(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    comparison = paths["comparison"] / "comparison.json"
    rows = json.loads(comparison.read_text(encoding="utf-8"))
    rows[0]["seed"], rows[1]["seed"] = rows[1]["seed"], rows[0]["seed"]
    _write_json(comparison, rows)

    with pytest.raises(ValueError, match="row differs"):
        _promote(paths)


def test_promote_deployment_rejects_coherent_published_numeric_forgery(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    comparison_dir = paths["comparison"]
    comparison_path = comparison_dir / "comparison.json"
    rows = json.loads(comparison_path.read_text(encoding="utf-8"))
    target_id = rows[0]["run_id"]
    rows[0]["ap50_95"] = 0.999
    _write_json(comparison_path, rows)
    published_metrics = comparison_dir / "sources" / target_id / "final_metrics.json"
    metrics = json.loads(published_metrics.read_text(encoding="utf-8"))
    metrics["metrics"]["ap50_95"] = 0.999
    _write_json(published_metrics, metrics)
    sources_path = comparison_dir / "sources_manifest.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    record = next(
        item for item in sources["files"]
        if item["run_id"] == target_id and item["path"].endswith("final_metrics.json")
    )
    record["published_sha256"] = sha256_file(published_metrics)
    record["bytes"] = published_metrics.stat().st_size
    _write_json(sources_path, sources)

    with pytest.raises(ValueError, match="numeric evidence differs from local original"):
        _promote(paths)


def test_promoted_comparison_chain_verifies_without_private_local_bindings(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    (paths["comparison"] / "local_source_bindings.json").unlink()
    result = validate_formal_comparison(paths["comparison"])
    assert result["status"] == "PASS"


def test_promoted_comparison_chain_rejects_formal_record_tamper(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    (paths["comparison"] / "local_source_bindings.json").unlink()
    formal = paths["comparison"] / "formal_validation.json"
    document = json.loads(formal.read_text(encoding="utf-8"))
    document["source_chain_sha256"] = "0" * 64
    _write_json(formal, document)
    with pytest.raises(ValueError, match="digest chain differs"):
        validate_formal_comparison(paths["comparison"])


def test_promote_deployment_rejects_missing_provenance_file(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    (paths["comparison"] / "run_provenance.json").unlink()

    with pytest.raises(FileNotFoundError):
        _promote(paths)


def test_promote_deployment_rejects_missing_mixed_commit_attestation(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    provenance_path = paths["comparison"] / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["mixed_commits"] = True
    provenance["attestation"] = {"allowed_commits": ["a" * 40, "b" * 40]}
    _write_json(provenance_path, provenance)
    compatibility_path = paths["comparison"] / "protocol_compatibility.json"
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    compatibility["run_provenance"] = provenance
    _write_json(compatibility_path, compatibility)
    (paths["comparison"] / "run_provenance_attestation.json").unlink()

    with pytest.raises(ValueError, match="attestation is missing"):
        _promote(paths)


def test_promote_deployment_requires_formal_test_binding(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    summary = json.loads(paths["test_summary"].read_text(encoding="utf-8"))
    summary["inputs"]["split_binding"]["status"] = "FAIL"
    _write_json(paths["test_summary"], summary)

    with pytest.raises(ValueError, match="test split_binding is not PASS"):
        _promote(paths)


def test_promote_deployment_requires_validation_native_equivalence(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    summary = json.loads(paths["val_summary"].read_text(encoding="utf-8"))
    summary["native_metric_equivalence"]["status"] = "FAIL"
    _write_json(paths["val_summary"], summary)

    with pytest.raises(ValueError, match="Validation native_metric_equivalence is not PASS"):
        _promote(paths)


def test_promote_deployment_binds_validation_to_comparison_native_metrics(
    tmp_path: Path,
) -> None:
    paths = _build_release_fixture(tmp_path)
    alternative = _write_json(
        paths["project"] / "runs" / "alternative" / "final_metrics.json",
        {"metrics": {"ap50_95": 0.01}},
    )
    summary = json.loads(paths["val_summary"].read_text(encoding="utf-8"))
    summary["artifacts"]["native_final_metrics"] = _record(alternative, paths["project"])
    summary["native_reference_binding"]["expected_sha256"] = sha256_file(alternative)
    summary["native_reference_binding"]["actual_sha256"] = sha256_file(alternative)
    _write_json(paths["val_summary"], summary)

    with pytest.raises(ValueError, match="differ from the metrics frozen"):
        _promote(paths)


def test_promote_deployment_rejects_tampered_metric_file(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    final_metrics = paths["val_summary"].parent / "final_metrics.json"
    final_metrics.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="val final_metrics artifact SHA-256 mismatch"):
        _promote(paths)


def test_promote_deployment_refuses_overwrite(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    _promote(paths)

    with pytest.raises(FileExistsError, match="already exists"):
        _promote(paths)


def test_runtime_gate_rejects_tampered_published_formal_report(tmp_path: Path) -> None:
    paths = _build_release_fixture(tmp_path)
    manifest = _promote(paths)
    destination = paths["project"] / "reports" / "deployments" / manifest["release_name"]
    (destination / "test" / "onnx_split_evaluation.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Published deployment report SHA-256 mismatch"):
        validate_promoted_deployment_for_runtime(
            project_root=paths["project"],
            release_manifest_path=destination / "deployment_release_manifest.json",
            deployment_metadata_path=paths["deployment_metadata"],
            onnx_path=paths["onnx"],
        )
