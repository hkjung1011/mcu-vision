from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mcu_data.common import sha256_file
from mcu_data.onnx_split_evaluation import (
    compare_metric_documents,
    detections_to_coco,
    load_coco_split,
    load_deployment_metadata,
    resolve_verified_onnx,
    verify_formal_native_reference,
    verify_protocol_binding,
    verify_native_evaluation_context,
    verify_split_binding,
)


def _write_coco(path: Path, file_name: str = "image.jpg") -> None:
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": file_name, "width": 20, "height": 10}],
                "annotations": [
                    {"id": 1, "image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 4]}
                ],
                "categories": [{"id": 1, "name": "chip"}],
            }
        ),
        encoding="utf-8",
    )


def test_load_coco_split_and_image_evidence_are_hash_bound(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "image.jpg").write_bytes(b"real-camera-bytes")
    annotation = tmp_path / "instances_val2017.json"
    _write_coco(annotation)

    document, records, category_ids, category_names = load_coco_split(annotation, image_root)

    assert len(document["annotations"]) == 1
    assert category_ids == [1]
    assert category_names == ["chip"]
    assert records[0].image_id == 7
    assert records[0].evidence()["sha256"] == sha256_file(image_root / "image.jpg")
    assert records[0].evidence()["synthetic_or_generated"] is False


def test_load_coco_split_rejects_path_escape(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    (tmp_path / "outside.jpg").write_bytes(b"outside")
    annotation = tmp_path / "instances.json"
    _write_coco(annotation, "../outside.jpg")

    with pytest.raises(ValueError, match="escapes image root"):
        load_coco_split(annotation, image_root)


def test_resolve_verified_onnx_prefers_portable_sibling_and_checks_hash(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    metadata_path = tmp_path / "model.deployment.json"
    metadata = {
        "artifacts": {
            "onnx": {
                "file_name": model.name,
                "path": "C:/Users/someone/old/model.onnx",
                "sha256": sha256_file(model),
            }
        }
    }

    assert resolve_verified_onnx(metadata_path, metadata) == model.resolve()
    metadata["artifacts"]["onnx"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_verified_onnx(metadata_path, metadata)


def test_formal_metadata_and_native_reference_are_exactly_hash_bound(tmp_path: Path) -> None:
    native_metrics = tmp_path / "final_metrics.json"
    native_metrics.write_text('{"metrics": {"ap50_95": 0.5}}', encoding="utf-8")
    metadata_path = tmp_path / "model.deployment.json"
    metadata = {
        "schema_version": 1,
        "status": "PASS",
        "framework": "yolo11",
        "release_validation": {
            "status": "PASS",
            "formal_release": True,
            "native_final_metrics_sha256": sha256_file(native_metrics),
        },
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = load_deployment_metadata(metadata_path, require_formal_release=True)
    binding = verify_formal_native_reference(loaded, native_metrics)

    assert binding["status"] == "PASS"
    native_metrics.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="reference frozen"):
        verify_formal_native_reference(loaded, native_metrics)


def test_formal_metadata_rejects_missing_frozen_native_metric_hash(tmp_path: Path) -> None:
    metadata_path = tmp_path / "model.deployment.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "framework": "yolo11",
                "release_validation": {"status": "PASS", "formal_release": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="native_final_metrics_sha256"):
        load_deployment_metadata(metadata_path, require_formal_release=True)


def test_formal_evaluator_rejects_diagnostic_export_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "model.deployment.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "framework": "yolo11",
                "release_validation": {
                    "status": "DIAGNOSTIC_ONLY",
                    "formal_release": False,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="release-ready comparison"):
        load_deployment_metadata(metadata_path, require_formal_release=True)


def test_detections_to_coco_maps_class_and_xyxy_to_xywh() -> None:
    detections = np.array([[2.0, 3.0, 12.0, 8.0, 0.75, 0.0]], dtype=np.float32)

    rows = detections_to_coco(9, detections, [4])

    assert rows == [
        {
            "image_id": 9,
            "category_id": 4,
            "bbox": [2.0, 3.0, 10.0, 5.0],
            "score": 0.75,
        }
    ]


def test_compare_metric_documents_applies_absolute_and_relative_tolerance() -> None:
    native = {"metrics": {"ap50_95": 0.80, "precision": 0.50}}
    close = {"metrics": {"ap50_95": 0.791, "precision": 0.504}}
    far = {"metrics": {"ap50_95": 0.77, "precision": 0.50}}

    passing = compare_metric_documents(
        close,
        native,
        metric_names=("ap50_95", "precision"),
        absolute_tolerance=0.005,
        relative_tolerance=0.01,
    )
    failing = compare_metric_documents(
        far,
        native,
        metric_names=("ap50_95", "precision"),
        absolute_tolerance=0.005,
        relative_tolerance=0.01,
    )

    assert passing["status"] == "PASS"
    assert passing["metrics_passed"] == 2
    assert failing["status"] == "FAIL"
    assert failing["rows"][0]["reason"] == "tolerance_exceeded"


def test_compare_metric_documents_fails_closed_on_missing_metric() -> None:
    result = compare_metric_documents(
        {"metrics": {}},
        {"metrics": {"ap50": 0.5}},
        metric_names=("ap50",),
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )

    assert result["status"] == "FAIL"
    assert result["rows"][0]["reason"] == "missing_or_non_numeric"


def test_verify_split_binding_checks_assignment_and_image_bytes(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "image.jpg"
    image.write_bytes(b"real-camera-bytes")
    annotation = tmp_path / "instances_val2017.json"
    _write_coco(annotation)
    _, records, _, _ = load_coco_split(annotation, image_root)
    evidence = [records[0].evidence()]
    split_manifest = tmp_path / "split.csv"
    split_manifest.write_text(
        "yolo_split,source_relative_path,sha256\n"
        f"val,source/image.jpg,{sha256_file(image)}\n",
        encoding="utf-8",
    )
    split_summary = tmp_path / "split.summary.json"
    split_summary.write_text(
        json.dumps(
            {
                "policy": "test_policy",
                "assignment_sha256": "a" * 64,
                "output_manifest_sha256": sha256_file(split_manifest),
                "split_image_counts": {"val": 1},
            }
        ),
        encoding="utf-8",
    )

    result = verify_split_binding(
        split="val",
        images=records,
        image_evidence=evidence,
        split_manifest_path=split_manifest,
        split_summary_path=split_summary,
    )

    assert result["status"] == "PASS"
    assert result["image_set_exact_match"] is True
    assert result["all_image_sha256_match"] is True

    image.write_bytes(b"changed")
    changed_evidence = [records[0].evidence()]
    with pytest.raises(ValueError, match="differs from split manifest"):
        verify_split_binding(
            split="val",
            images=records,
            image_evidence=changed_evidence,
            split_manifest_path=split_manifest,
            split_summary_path=split_summary,
        )


def test_verify_protocol_binding_enforces_split_annotation_hash(tmp_path: Path) -> None:
    annotation = tmp_path / "instances_test2017.json"
    _write_coco(annotation)
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "\n".join(
            [
                "protocol_id: fixture",
                "dataset:",
                "  coco_annotation_sha256:",
                f"    test: {sha256_file(annotation)}",
                "common:",
                "  condition_held_out_test_images: 1",
                "  image_size: 640",
                "  prediction_floor: 0.001",
                "  nms_iou: 0.65",
                "  max_detections_for_coco_ap: 100",
                "  operating_confidence: 0.25",
                "  operating_match_iou: 0.5",
                "  class_agnostic_nms: false",
            ]
        ),
        encoding="utf-8",
    )

    result = verify_protocol_binding(
        protocol_path=protocol,
        split="test",
        annotation_path=annotation,
        image_count=1,
        image_size=640,
        prediction_floor=0.001,
        nms_iou=0.65,
        max_detections=100,
        operating_confidence=0.25,
        match_iou=0.5,
    )

    assert result["status"] == "PASS"
    annotation.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="annotation SHA-256 mismatch"):
        verify_protocol_binding(
            protocol_path=protocol,
            split="test",
            annotation_path=annotation,
            image_count=1,
            image_size=640,
            prediction_floor=0.001,
            nms_iou=0.65,
            max_detections=100,
            operating_confidence=0.25,
            match_iou=0.5,
        )


def test_verify_native_context_binds_ground_truth_and_operating_protocol(tmp_path: Path) -> None:
    annotation = tmp_path / "instances_val2017.json"
    _write_coco(annotation)
    native = {
        "ground_truth": {"sha256": sha256_file(annotation)},
        "protocol": {
            "coco_ap_max_dets": 100,
            "operating_max_dets_per_image": 100,
            "operating_confidence": 0.25,
            "operating_match_iou": 0.5,
        },
    }

    result = verify_native_evaluation_context(
        native,
        annotation_path=annotation,
        operating_confidence=0.25,
        match_iou=0.5,
    )

    assert result["status"] == "PASS"
    native["protocol"]["operating_confidence"] = 0.3
    with pytest.raises(ValueError, match="operating_confidence mismatch"):
        verify_native_evaluation_context(
            native,
            annotation_path=annotation,
            operating_confidence=0.25,
            match_iou=0.5,
        )
