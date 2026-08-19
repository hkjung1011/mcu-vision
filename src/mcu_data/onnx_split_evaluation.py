from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from mcu_data.common import load_yaml, portable_path, sha256_file, write_json
from mcu_data.deployment import (
    Framework,
    artifact_record,
    decode_detections,
    preprocess_image,
    restore_boxes,
)


DEFAULT_METRICS = (
    "ap50_95",
    "ap50",
    "ap75",
    "ar1",
    "ar10",
    "ar100",
    "precision",
    "recall",
    "f1",
)
FORMAL_METRIC_ATOL = 0.005
FORMAL_METRIC_RTOL = 0.0


@dataclass(frozen=True)
class CocoImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    path: Path

    def evidence(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "file_name": self.file_name,
            "path": portable_path(self.path),
            "bytes": self.path.stat().st_size,
            "sha256": sha256_file(self.path),
            "width": self.width,
            "height": self.height,
            "synthetic_or_generated": False,
        }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_deployment_metadata(path: Path, *, require_formal_release: bool = False) -> dict[str, Any]:
    path = path.resolve()
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Deployment metadata must be a JSON object: {path}")
    if value.get("schema_version") != 1:
        raise ValueError(f"Unsupported deployment schema: {value.get('schema_version')}")
    if value.get("status") != "PASS":
        raise ValueError(f"Deployment metadata status is not PASS: {path}")
    if value.get("framework") not in {"yolo11", "yolox"}:
        raise ValueError(f"Unsupported framework: {value.get('framework')}")
    if require_formal_release:
        release_validation = value.get("release_validation")
        if (
            not isinstance(release_validation, dict)
            or release_validation.get("status") != "PASS"
            or release_validation.get("formal_release") is not True
        ):
            raise ValueError(
                "Formal split evaluation requires deployment metadata validated by a release-ready comparison"
            )
        native_metrics_sha = str(
            release_validation.get("native_final_metrics_sha256", "")
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", native_metrics_sha):
            raise ValueError(
                "Formal split evaluation requires release_validation."
                "native_final_metrics_sha256"
            )
    return value


def verify_formal_native_reference(
    metadata: dict[str, Any], native_final_metrics_path: Path
) -> dict[str, Any]:
    release_validation = metadata.get("release_validation")
    if not isinstance(release_validation, dict):
        raise ValueError("Deployment metadata is missing release_validation")
    expected_sha = str(release_validation.get("native_final_metrics_sha256", "")).lower()
    actual_sha = sha256_file(native_final_metrics_path.resolve())
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or actual_sha.lower() != expected_sha:
        raise ValueError(
            "Native final metrics SHA-256 differs from the reference frozen by the release-ready "
            f"comparison: expected={expected_sha or 'MISSING'}, actual={actual_sha}"
        )
    return {
        "status": "PASS",
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "binding": "deployment.release_validation.native_final_metrics_sha256",
    }


def resolve_verified_onnx(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    """Resolve the ONNX next to its metadata first and enforce its recorded SHA-256."""

    record = metadata.get("artifacts", {}).get("onnx")
    if not isinstance(record, dict):
        raise ValueError("Deployment metadata has no verified artifacts.onnx record")
    file_name = record.get("file_name")
    recorded_path = record.get("path")
    candidates: list[Path] = []
    if isinstance(file_name, str) and file_name:
        candidates.append(metadata_path.resolve().parent / file_name)
    if isinstance(recorded_path, str) and recorded_path:
        candidates.append(Path(recorded_path))
    onnx_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if onnx_path is None:
        rendered = ", ".join(portable_path(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Recorded ONNX artifact was not found; tried: {rendered}")
    expected_sha = str(record.get("sha256", "")).lower()
    actual_sha = sha256_file(onnx_path)
    if not expected_sha or actual_sha.lower() != expected_sha:
        raise ValueError(
            f"ONNX SHA-256 mismatch: expected={expected_sha or 'MISSING'}, "
            f"actual={actual_sha}, path={portable_path(onnx_path)}"
        )
    return onnx_path


def load_coco_split(
    annotation_path: Path,
    image_root: Path,
) -> tuple[dict[str, Any], list[CocoImageRecord], list[int], list[str]]:
    annotation_path = annotation_path.resolve()
    image_root = image_root.resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)
    document = _read_json(annotation_path)
    if not isinstance(document, dict):
        raise ValueError(f"COCO annotations must be a JSON object: {annotation_path}")
    images = document.get("images")
    annotations = document.get("annotations")
    categories = document.get("categories")
    if not isinstance(images, list) or not images:
        raise ValueError("COCO split must contain at least one image")
    if not isinstance(annotations, list):
        raise ValueError("COCO annotations must be a list")
    if not isinstance(categories, list) or not categories:
        raise ValueError("COCO categories must be a non-empty list")

    sorted_categories = sorted(categories, key=lambda item: int(item["id"]))
    category_ids = [int(item["id"]) for item in sorted_categories]
    if category_ids != list(range(1, len(category_ids) + 1)):
        raise ValueError(
            "ONNX deployment evaluation requires consecutive COCO category IDs starting at 1; "
            f"found {category_ids}"
        )
    category_names = [str(item["name"]) for item in sorted_categories]

    records: list[CocoImageRecord] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for item in sorted(images, key=lambda value: int(value["id"])):
        image_id = int(item["id"])
        file_name = str(item["file_name"])
        if image_id in seen_ids:
            raise ValueError(f"Duplicate COCO image ID: {image_id}")
        if file_name in seen_names:
            raise ValueError(f"Duplicate COCO file_name: {file_name}")
        seen_ids.add(image_id)
        seen_names.add(file_name)
        image_path = (image_root / file_name).resolve()
        try:
            image_path.relative_to(image_root)
        except ValueError as error:
            raise ValueError(f"COCO file_name escapes image root: {file_name}") from error
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        width = int(item["width"])
        height = int(item["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid COCO dimensions for image_id={image_id}: {width}x{height}")
        records.append(CocoImageRecord(image_id, file_name, width, height, image_path))
    return document, records, category_ids, category_names


def verify_split_binding(
    *,
    split: str,
    images: Sequence[CocoImageRecord],
    image_evidence: Sequence[dict[str, Any]],
    split_manifest_path: Path,
    split_summary_path: Path,
) -> dict[str, Any]:
    """Bind the requested split to the audited CSV assignment and source image hashes."""

    split_manifest_path = split_manifest_path.resolve()
    split_summary_path = split_summary_path.resolve()
    summary = _read_json(split_summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"Split summary must be a JSON object: {split_summary_path}")
    expected_manifest_sha = str(summary.get("output_manifest_sha256", "")).lower()
    actual_manifest_sha = sha256_file(split_manifest_path)
    if not expected_manifest_sha or actual_manifest_sha.lower() != expected_manifest_sha:
        raise ValueError(
            "Split manifest/summary SHA-256 mismatch: "
            f"expected={expected_manifest_sha or 'MISSING'}, actual={actual_manifest_sha}"
        )
    with split_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"yolo_split", "source_relative_path", "sha256"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Split manifest is missing columns: {sorted(missing_columns)}")
        rows = [row for row in reader if row.get("yolo_split") == split]
    expected_count = summary.get("split_image_counts", {}).get(split)
    if int(expected_count if expected_count is not None else -1) != len(rows):
        raise ValueError(
            f"Split summary/manifest count mismatch for {split}: summary={expected_count}, rows={len(rows)}"
        )
    manifest_by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        file_name = Path(str(row["source_relative_path"]).replace("\\", "/")).name
        if file_name in manifest_by_name:
            raise ValueError(f"Duplicate split-manifest basename in {split}: {file_name}")
        manifest_by_name[file_name] = row
    coco_names = {record.file_name for record in images}
    manifest_names = set(manifest_by_name)
    if coco_names != manifest_names:
        missing = sorted(manifest_names - coco_names)[:10]
        unexpected = sorted(coco_names - manifest_names)[:10]
        raise ValueError(
            f"COCO/{split} manifest image-set mismatch: missing={missing}, unexpected={unexpected}"
        )
    evidence_by_name = {str(item["file_name"]): item for item in image_evidence}
    hash_mismatches = [
        file_name
        for file_name, row in manifest_by_name.items()
        if str(row["sha256"]).lower() != str(evidence_by_name[file_name]["sha256"]).lower()
    ]
    if hash_mismatches:
        raise ValueError(
            f"Processed image SHA-256 differs from split manifest: {hash_mismatches[:10]}"
        )
    return {
        "status": "PASS",
        "policy": summary.get("policy"),
        "assignment_sha256": summary.get("assignment_sha256"),
        "split": split,
        "image_count": len(images),
        "image_set_exact_match": True,
        "all_image_sha256_match": True,
        "artifacts": {
            "split_manifest": artifact_record(split_manifest_path),
            "split_summary": artifact_record(split_summary_path),
        },
    }


def verify_protocol_binding(
    *,
    protocol_path: Path,
    split: str,
    annotation_path: Path,
    image_count: int,
    image_size: int,
    prediction_floor: float,
    nms_iou: float,
    max_detections: int,
    operating_confidence: float,
    match_iou: float,
) -> dict[str, Any]:
    """Verify the exact COCO annotation and evaluation settings frozen in a protocol YAML."""

    protocol_path = protocol_path.resolve()
    protocol = load_yaml(protocol_path)
    dataset = protocol.get("dataset")
    common = protocol.get("common")
    if not isinstance(dataset, dict) or not isinstance(common, dict):
        raise ValueError("Protocol must contain dataset and common mappings")
    annotation_hashes = dataset.get("coco_annotation_sha256")
    if not isinstance(annotation_hashes, dict):
        raise ValueError("Protocol dataset.coco_annotation_sha256 must be a split->SHA mapping")
    expected_annotation_sha = str(annotation_hashes.get(split, "")).lower()
    actual_annotation_sha = sha256_file(annotation_path)
    if not expected_annotation_sha or expected_annotation_sha != actual_annotation_sha.lower():
        raise ValueError(
            f"Protocol/{split} COCO annotation SHA-256 mismatch: "
            f"expected={expected_annotation_sha or 'MISSING'}, actual={actual_annotation_sha}"
        )
    count_key = "validation_images" if split == "val" else "condition_held_out_test_images"
    expected_image_count = int(common.get(count_key, -1))
    if expected_image_count != image_count:
        raise ValueError(
            f"Protocol/{split} image count mismatch: expected={expected_image_count}, actual={image_count}"
        )
    expected_values: dict[str, int | float] = {
        "image_size": image_size,
        "prediction_floor": prediction_floor,
        "nms_iou": nms_iou,
        "max_detections_for_coco_ap": max_detections,
        "operating_confidence": operating_confidence,
        "operating_match_iou": match_iou,
    }
    checks: dict[str, dict[str, Any]] = {}
    for key, actual in expected_values.items():
        frozen = common.get(key)
        if not isinstance(frozen, (int, float)) or not math.isclose(
            float(frozen), float(actual), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Protocol common.{key} mismatch: expected={frozen!r}, actual={actual!r}"
            )
        checks[key] = {"status": "PASS", "frozen": frozen, "actual": actual}
    if common.get("class_agnostic_nms") is not False:
        raise ValueError("Protocol common.class_agnostic_nms must be false")
    return {
        "status": "PASS",
        "protocol_id": protocol.get("protocol_id"),
        "split": split,
        "coco_annotation_sha256": actual_annotation_sha,
        "image_count": image_count,
        "checks": checks,
        "artifact": artifact_record(protocol_path),
    }


def detections_to_coco(
    image_id: int,
    detections: np.ndarray,
    category_ids: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for x1, y1, x2, y2, score, class_id_value in np.asarray(detections, dtype=np.float32):
        class_id = int(class_id_value)
        if class_id < 0 or class_id >= len(category_ids):
            raise ValueError(f"Decoded class index {class_id} is outside 0..{len(category_ids) - 1}")
        width = max(0.0, float(x2 - x1))
        height = max(0.0, float(y2 - y1))
        if width == 0.0 or height == 0.0:
            continue
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_ids[class_id]),
                "bbox": [float(x1), float(y1), width, height],
                "score": float(score),
            }
        )
    return rows


def compare_metric_documents(
    evaluated: dict[str, Any],
    native: dict[str, Any],
    *,
    metric_names: Iterable[str] = DEFAULT_METRICS,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("Metric tolerances must be non-negative")
    evaluated_metrics = evaluated.get("metrics")
    native_metrics = native.get("metrics")
    if not isinstance(evaluated_metrics, dict) or not isinstance(native_metrics, dict):
        raise ValueError("Both documents must contain a metrics JSON object")

    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        exported_value = evaluated_metrics.get(metric)
        native_value = native_metrics.get(metric)
        if not isinstance(exported_value, (int, float)) or not isinstance(native_value, (int, float)):
            rows.append(
                {
                    "metric": metric,
                    "status": "FAIL",
                    "reason": "missing_or_non_numeric",
                    "onnx": exported_value,
                    "native": native_value,
                }
            )
            continue
        exported_float = float(exported_value)
        native_float = float(native_value)
        if not math.isfinite(exported_float) or not math.isfinite(native_float):
            rows.append(
                {
                    "metric": metric,
                    "status": "FAIL",
                    "reason": "non_finite",
                    "onnx": exported_float,
                    "native": native_float,
                }
            )
            continue
        difference = abs(exported_float - native_float)
        limit = absolute_tolerance + relative_tolerance * abs(native_float)
        passed = difference <= limit
        rows.append(
            {
                "metric": metric,
                "status": "PASS" if passed else "FAIL",
                "reason": "within_tolerance" if passed else "tolerance_exceeded",
                "onnx": exported_float,
                "native": native_float,
                "absolute_difference": difference,
                "allowed_difference": limit,
            }
        )
    passed_count = sum(row["status"] == "PASS" for row in rows)
    return {
        "status": "PASS" if passed_count == len(rows) else "FAIL",
        "gate": "abs(onnx-native) <= atol + rtol*abs(native) for every selected metric",
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "metrics_requested": len(rows),
        "metrics_passed": passed_count,
        "metrics_failed": len(rows) - passed_count,
        "rows": rows,
    }


def verify_native_evaluation_context(
    native: dict[str, Any],
    *,
    annotation_path: Path,
    operating_confidence: float,
    match_iou: float,
) -> dict[str, Any]:
    ground_truth = native.get("ground_truth")
    expected_ground_truth_sha = sha256_file(annotation_path)
    if (
        not isinstance(ground_truth, dict)
        or str(ground_truth.get("sha256", "")).lower() != expected_ground_truth_sha
    ):
        raise ValueError(
            "Native final metrics were not produced from this COCO split: "
            f"expected ground_truth.sha256={expected_ground_truth_sha}"
        )
    protocol = native.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Native final metrics are missing the common evaluator protocol")
    expected: dict[str, int | float] = {
        "coco_ap_max_dets": 100,
        "operating_max_dets_per_image": 100,
        "operating_confidence": operating_confidence,
        "operating_match_iou": match_iou,
    }
    for key, actual in expected.items():
        frozen = protocol.get(key)
        if not isinstance(frozen, (int, float)) or not math.isclose(
            float(frozen), float(actual), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Native final metrics protocol.{key} mismatch: expected={actual}, found={frozen!r}"
            )
    return {
        "status": "PASS",
        "ground_truth_sha256": expected_ground_truth_sha,
        "protocol": expected,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _validate_probability(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _split_warning(split: str) -> str:
    if split == "test":
        return (
            "This is the held-out phash_v2 bootstrap test split, not a new physical-item/session "
            "conveyor-camera acceptance set."
        )
    return (
        "This is the phash_v2 bootstrap validation split, not an independent conveyor-camera "
        "acceptance set."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a verified batch-1 ONNX deployment artifact over a complete COCO split, "
            "using the repository's common evaluator."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True, help="PASS *.deployment.json")
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--coco-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument(
        "--mode",
        choices=("formal", "diagnostic"),
        default="formal",
        help="formal is fail-closed; diagnostic must be selected explicitly to omit release evidence",
    )
    parser.add_argument("--protocol", type=Path, help="Frozen experiment protocol YAML")
    parser.add_argument(
        "--require-protocol-binding",
        action="store_true",
        help="Fail unless the protocol binds this split's exact COCO annotation SHA and settings",
    )
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument(
        "--prediction-floor",
        type=float,
        default=0.001,
        help="Low score floor used before COCO AP; operating metrics use --operating-confidence",
    )
    parser.add_argument("--nms-iou", type=float, help="Default: deployment metadata value")
    parser.add_argument("--max-detections", type=int, help="Default: deployment metadata value")
    parser.add_argument(
        "--operating-confidence", type=float, help="Default: deployment metadata value"
    )
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--native-final-metrics", type=Path)
    parser.add_argument(
        "--require-native-equivalence",
        action="store_true",
        help="Fail unless --native-final-metrics is supplied and every selected metric passes",
    )
    parser.add_argument("--split-manifest", type=Path, help="Audited split assignment CSV")
    parser.add_argument("--split-summary", type=Path, help="Summary binding the split CSV SHA-256")
    parser.add_argument(
        "--require-split-evidence",
        action="store_true",
        help="Fail unless split CSV + summary exactly bind image names and SHA-256 values",
    )
    parser.add_argument("--metric-atol", type=float, default=FORMAL_METRIC_ATOL)
    parser.add_argument("--metric-rtol", type=float, default=FORMAL_METRIC_RTOL)
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric key to compare; repeatable. Default: AP/AR/P/R/F1 core metrics",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import onnxruntime as ort

    from mcu_data.reporting import evaluate_predictions

    metadata_path = args.metadata.resolve()
    annotation_path = args.coco_annotations.resolve()
    image_root = args.coco_images.resolve()
    output_dir = args.output_dir.resolve()
    formal = args.mode == "formal"
    metadata = load_deployment_metadata(metadata_path, require_formal_release=formal)
    onnx_path = resolve_verified_onnx(metadata_path, metadata)
    coco_document, images, category_ids, category_names = load_coco_split(
        annotation_path, image_root
    )

    metadata_names = [str(value) for value in metadata.get("classes", {}).get("names", [])]
    if metadata_names != category_names:
        raise ValueError(
            f"Deployment/COCO class map mismatch: deployment={metadata_names}, COCO={category_names}"
        )
    model_input = metadata.get("model_input", {})
    batch = int(model_input.get("batch", 0))
    height = int(model_input.get("height", 0))
    width = int(model_input.get("width", 0))
    if batch != 1 or height <= 0 or height != width:
        raise ValueError(
            f"Split evaluator requires fixed square batch-1 metadata, got batch={batch}, {width}x{height}"
        )
    image_size = height
    framework: Framework = metadata["framework"]
    postprocessing = metadata.get("postprocessing", {})
    prediction_floor = float(args.prediction_floor)
    nms_iou = float(args.nms_iou if args.nms_iou is not None else postprocessing["nms_iou"])
    max_detections = int(
        args.max_detections
        if args.max_detections is not None
        else postprocessing["max_detections"]
    )
    operating_confidence = float(
        args.operating_confidence
        if args.operating_confidence is not None
        else postprocessing["confidence"]
    )
    _validate_probability("--prediction-floor", prediction_floor)
    _validate_probability("--nms-iou", nms_iou)
    _validate_probability("--operating-confidence", operating_confidence)
    _validate_probability("--match-iou", float(args.match_iou))
    if max_detections <= 0:
        raise ValueError("--max-detections must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if formal:
        if args.force:
            raise ValueError("Formal evaluation refuses --force; use a new immutable output directory")
        if args.metrics is not None:
            raise ValueError("Formal evaluation uses the complete fixed metric set; --metric is diagnostic-only")
        if not math.isclose(float(args.metric_atol), FORMAL_METRIC_ATOL, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Formal metric atol is fixed to {FORMAL_METRIC_ATOL}")
        if not math.isclose(float(args.metric_rtol), FORMAL_METRIC_RTOL, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Formal metric rtol is fixed to {FORMAL_METRIC_RTOL}")
    require_protocol = formal or args.require_protocol_binding
    require_split_evidence = formal or args.require_split_evidence
    require_native_equivalence = (
        (formal and args.split == "val") or args.require_native_equivalence
    )
    if require_protocol and not args.protocol:
        raise ValueError("Formal/protocol-bound evaluation requires --protocol")
    if bool(args.split_manifest) != bool(args.split_summary):
        raise ValueError("--split-manifest and --split-summary must be supplied together")
    if require_split_evidence and not args.split_manifest:
        raise ValueError("Formal/split-bound evaluation requires --split-manifest and --split-summary")
    if require_native_equivalence and not args.native_final_metrics:
        raise ValueError("Formal validation/native-bound evaluation requires --native-final-metrics")
    native_reference_binding: dict[str, Any] = {
        "status": "NOT_APPLICABLE",
        "reason": "only formal validation is bound to the comparison-frozen native metrics",
    }
    if formal and args.split == "val":
        assert args.native_final_metrics is not None
        native_reference_binding = verify_formal_native_reference(
            metadata, args.native_final_metrics
        )

    protocol_binding: dict[str, Any] = {
        "status": "NOT_PROVIDED",
        "reason": "protocol was not supplied; diagnostic result is not a release gate",
    }
    if args.protocol:
        protocol_binding = verify_protocol_binding(
            protocol_path=args.protocol,
            split=args.split,
            annotation_path=annotation_path,
            image_count=len(images),
            image_size=image_size,
            prediction_floor=prediction_floor,
            nms_iou=nms_iou,
            max_detections=max_detections,
            operating_confidence=operating_confidence,
            match_iou=float(args.match_iou),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "image_manifest.json",
        "predictions.coco.json",
        "final_metrics.json",
        "onnx_split_evaluation.json",
    )
    existing = [output_dir / name for name in artifact_names if (output_dir / name).exists()]
    if existing and not args.force:
        rendered = "\n".join(f"  {portable_path(path)}" for path in existing)
        raise FileExistsError(f"Refusing to overwrite evaluation artifacts:\n{rendered}\nUse --force.")

    image_manifest_path = output_dir / "image_manifest.json"
    image_evidence = [record.evidence() for record in images]
    image_manifest = {
        "schema_version": 1,
        "split": args.split,
        "annotation": artifact_record(annotation_path),
        "image_root": portable_path(image_root),
        "image_count": len(images),
        "images": image_evidence,
    }
    write_json(image_manifest_path, image_manifest)

    split_binding: dict[str, Any] = {
        "status": "NOT_PROVIDED",
        "reason": "split manifest and summary were not supplied",
    }
    if args.split_manifest and args.split_summary:
        split_binding = verify_split_binding(
            split=args.split,
            images=images,
            image_evidence=image_evidence,
            split_manifest_path=args.split_manifest,
            split_summary_path=args.split_summary,
        )

    available_providers = ort.get_available_providers()
    if args.provider not in available_providers:
        raise ValueError(
            f"Requested provider {args.provider!r} is unavailable; available={available_providers}"
        )
    session = ort.InferenceSession(str(onnx_path), providers=[args.provider])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Expected an ONNX model with exactly one input and one output")
    input_node = session.get_inputs()[0]
    output_node = session.get_outputs()[0]
    expected_shape = [1, 3, image_size, image_size]
    if list(input_node.shape) != expected_shape:
        raise ValueError(f"ONNX input shape mismatch: expected={expected_shape}, actual={input_node.shape}")

    predictions: list[dict[str, Any]] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    e2e_ms: list[float] = []
    started = time.perf_counter()
    for index, record in enumerate(images, start=1):
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode COCO image: {portable_path(record.path)}")
        if (image.shape[1], image.shape[0]) != (record.width, record.height):
            raise ValueError(
                f"COCO/decode dimension mismatch for image_id={record.image_id}: "
                f"record={record.width}x{record.height}, decoded={image.shape[1]}x{image.shape[0]}"
            )
        if index == 1 and args.warmup:
            warmup_input, _ = preprocess_image(image, framework, image_size)
            for _ in range(args.warmup):
                session.run([output_node.name], {input_node.name: warmup_input})
        e2e_start = time.perf_counter()
        preprocess_start = e2e_start
        input_array, transform = preprocess_image(image, framework, image_size)
        preprocess_end = time.perf_counter()
        inference_start = time.perf_counter()
        output = session.run([output_node.name], {input_node.name: input_array})[0]
        inference_end = time.perf_counter()
        detections = decode_detections(
            output,
            framework=framework,
            class_count=len(category_ids),
            confidence=prediction_floor,
            nms_iou=nms_iou,
            max_detections=max_detections,
        )
        detections = restore_boxes(detections, transform)
        predictions.extend(detections_to_coco(record.image_id, detections, category_ids))
        postprocess_end = time.perf_counter()
        preprocess_ms.append((preprocess_end - preprocess_start) * 1000)
        inference_ms.append((inference_end - inference_start) * 1000)
        postprocess_ms.append((postprocess_end - inference_end) * 1000)
        e2e_ms.append((postprocess_end - e2e_start) * 1000)
        if index % args.progress_every == 0 or index == len(images):
            print(
                f"[ONNX {index:04d}/{len(images):04d}] predictions={len(predictions):06d} "
                f"inference_ms={inference_ms[-1]:.2f}",
                flush=True,
            )
    inference_wall_seconds = time.perf_counter() - started

    predictions_path = output_dir / "predictions.coco.json"
    write_json(predictions_path, predictions)
    evaluated = evaluate_predictions(
        annotation_path,
        predictions_path,
        output_dir,
        confidence=operating_confidence,
        match_iou=float(args.match_iou),
        evaluation_set=args.split,
    )
    evaluated["warning"] = _split_warning(args.split)
    final_metrics_path = output_dir / "final_metrics.json"
    write_json(final_metrics_path, evaluated)

    comparison: dict[str, Any] = {
        "status": "NOT_REQUESTED",
        "reason": "--native-final-metrics was not supplied",
    }
    native_record: dict[str, Any] | None = None
    if args.native_final_metrics:
        native_path = args.native_final_metrics.resolve()
        native = _read_json(native_path)
        if not isinstance(native, dict):
            raise ValueError(f"Native final metrics must be a JSON object: {native_path}")
        native_context = verify_native_evaluation_context(
            native,
            annotation_path=annotation_path,
            operating_confidence=operating_confidence,
            match_iou=float(args.match_iou),
        )
        comparison = compare_metric_documents(
            evaluated,
            native,
            metric_names=args.metrics or DEFAULT_METRICS,
            absolute_tolerance=float(args.metric_atol),
            relative_tolerance=float(args.metric_rtol),
        )
        comparison["reference_context"] = native_context
        native_record = artifact_record(native_path)

    status = "FAIL" if comparison.get("status") == "FAIL" else (
        "PASS" if formal else "DIAGNOSTIC_ONLY"
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        "mode": args.mode,
        "warning": _split_warning(args.split),
        "profile": "fixed_batch1_fp32_onnxruntime",
        "framework": framework,
        "provider_requested": args.provider,
        "provider_used": session.get_providers()[0],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "onnxruntime": ort.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "protocol": {
            "batch": 1,
            "image_size": image_size,
            "prediction_floor_for_coco_ap": prediction_floor,
            "nms_iou": nms_iou,
            "class_agnostic_nms": False,
            "max_detections_per_image": max_detections,
            "operating_confidence": operating_confidence,
            "operating_match_iou": float(args.match_iou),
            "warmup_runs_excluded": args.warmup,
            "latency_scope": "preprocess + ONNX Runtime inference + decode/NMS/restore; image I/O and SHA-256 excluded",
        },
        "inputs": {
            "image_root": portable_path(image_root),
            "image_count": len(images),
            "annotation_count": len(coco_document.get("annotations", [])),
            "category_ids": category_ids,
            "category_names": category_names,
            "protocol_binding": protocol_binding,
            "split_binding": split_binding,
        },
        "artifacts": {
            "deployment_metadata": artifact_record(metadata_path),
            "onnx": artifact_record(onnx_path),
            "coco_annotations": artifact_record(annotation_path),
            "image_manifest": artifact_record(image_manifest_path),
            "predictions": artifact_record(predictions_path),
            "final_metrics": artifact_record(final_metrics_path),
        },
        "inference": {
            "images": len(images),
            "predictions": len(predictions),
            "loop_wall_seconds": inference_wall_seconds,
            "images_per_second": (
                len(images) / inference_wall_seconds if inference_wall_seconds > 0 else None
            ),
            "preprocess_ms_mean": float(np.mean(preprocess_ms)),
            "inference_ms_mean": float(np.mean(inference_ms)),
            "inference_ms_p50": _percentile(inference_ms, 50),
            "inference_ms_p95": _percentile(inference_ms, 95),
            "postprocess_ms_mean": float(np.mean(postprocess_ms)),
            "e2e_ms_mean": float(np.mean(e2e_ms)),
            "e2e_ms_p50": _percentile(e2e_ms, 50),
            "e2e_ms_p95": _percentile(e2e_ms, 95),
        },
        "metrics": evaluated["metrics"],
        "threshold_selection": evaluated["threshold_selection"],
        "native_metric_equivalence": comparison,
        "native_reference_binding": native_reference_binding,
    }
    if native_record is not None:
        summary["artifacts"]["native_final_metrics"] = native_record
    write_json(output_dir / "onnx_split_evaluation.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print("\nONNX SPLIT EVALUATION")
    print(f"status          {summary['status']}")
    print(f"framework       {summary['framework']}")
    print(f"split/images    {summary['split']} / {summary['inputs']['image_count']}")
    print(f"AP50-95         {summary['metrics']['ap50_95']:.6f}")
    print(f"AP50            {summary['metrics']['ap50']:.6f}")
    print(f"precision       {summary['metrics']['precision']:.6f}")
    print(f"recall          {summary['metrics']['recall']:.6f}")
    print(f"inference p50   {summary['inference']['inference_ms_p50']:.3f} ms")
    print(f"native gate     {summary['native_metric_equivalence']['status']}")
    print(f"output          {portable_path(args.output_dir.resolve())}")
    return 0 if summary["status"] in {"PASS", "DIAGNOSTIC_ONLY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
