from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import portable_path, safe_stem, sha256_file, write_json
from .checkpoint_publishing import assert_binary_has_no_local_paths
from .publishing import publish_evidence_file, validate_formal_comparison


EVALUATION_REPORT_FILES = (
    "onnx_split_evaluation.json",
    "final_metrics.json",
    "image_manifest.json",
    "per_class_metrics.csv",
    "confidence_curve.csv",
    "autolabel_thresholds.csv",
    "confusion_counts.csv",
    "confusion_normalized.csv",
    "confidence_curve.png",
    "per_class_ap.png",
    "confusion_counts.png",
    "confusion_normalized.png",
)

COMPARISON_REPORT_FILES = (
    "protocol_compatibility.json",
    "run_provenance.json",
    "run_provenance_attestation.json",
    "formal_validation.json",
    "comparison.json",
    "sources_manifest.json",
)
REQUIRED_RELEASE_GATES = {
    "native_release",
    "deployment_release_validation",
    "native_onnx_numeric_equivalence",
    "validation_formal_split",
    "validation_native_metric_equivalence",
    "validation_native_reference_binding",
    "test_formal_split",
    "protocol_binding",
    "split_binding",
    "artifact_hashes_and_sizes",
    "publication_path_privacy",
    "git_lfs_rules",
}

TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml"}
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[\s'\"=(])(?:[a-z]:[\\/]|\\\\)")
POSIX_ABSOLUTE = re.compile(r"(?:^|[\s'\"=(])/(?!/)[A-Za-z0-9_.-]+/")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _safe_release_name(value: Any) -> str:
    release = str(value or "")
    if not release or not re.fullmatch(r"[A-Za-z0-9._-]+", release):
        raise ValueError(f"Unsafe or missing release_name: {release!r}")
    return release


def _relative_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must stay under {root.resolve()}: {resolved}") from error


def _record_sha(record: Mapping[str, Any], label: str) -> str:
    value = str(record.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label}.sha256 is missing or invalid")
    return value


def _verify_record_file(path: Path, record: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file is missing: {path}")
    expected_sha = _record_sha(record, label)
    actual_sha = sha256_file(path)
    if actual_sha.lower() != expected_sha:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected_sha}, actual={actual_sha}, "
            f"path={portable_path(path)}"
        )
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and int(expected_bytes) != path.stat().st_size:
        raise ValueError(
            f"{label} byte-size mismatch: expected={expected_bytes}, actual={path.stat().st_size}"
        )
    return {
        "path": portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": actual_sha,
    }


def _resolve_record_path(record: Mapping[str, Any], project_root: Path, label: str) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label}.path is missing")
    normalized = raw.replace("\\", "/")
    if normalized == "<PROJECT_ROOT>":
        candidate = project_root
    elif normalized.startswith("<PROJECT_ROOT>/"):
        candidate = project_root / normalized.removeprefix("<PROJECT_ROOT>/")
    else:
        path = Path(raw)
        candidate = path if path.is_absolute() else project_root / path
    return candidate.resolve()


def _verify_comparison_source_copy(
    comparison_dir: Path, row: Mapping[str, Any], label: str
) -> Path:
    raw_path = row.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path is missing")
    source_path = (comparison_dir / raw_path).resolve()
    _relative_under(source_path, comparison_dir, label)
    if not source_path.is_file():
        raise FileNotFoundError(f"{label} is missing: {source_path}")
    expected_sha = str(row.get("published_sha256", "")).lower()
    actual_sha = sha256_file(source_path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or actual_sha != expected_sha:
        raise ValueError(f"{label} published SHA-256 mismatch")
    expected_bytes = row.get("bytes")
    if expected_bytes is not None and int(expected_bytes) != source_path.stat().st_size:
        raise ValueError(f"{label} published byte-size mismatch")
    return source_path


def _verify_comparison(
    comparison_dir: Path,
    *,
    run_id: str,
    run_manifest_sha256: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_dir = comparison_dir.resolve()
    validate_formal_comparison(comparison_dir)
    compatibility_path = comparison_dir / "protocol_compatibility.json"
    comparison_path = comparison_dir / "comparison.json"
    sources_path = comparison_dir / "sources_manifest.json"
    provenance_path = comparison_dir / "run_provenance.json"
    compatibility = _read_object(compatibility_path, "comparison protocol compatibility")
    if compatibility.get("release_ready") is not True:
        raise ValueError("Original comparison is not release_ready")
    provenance = _read_object(provenance_path, "comparison run provenance")
    if provenance.get("status") != "PASS" or compatibility.get("run_provenance") != provenance:
        raise ValueError("Original comparison provenance is not an exact PASS")
    attestation_path = comparison_dir / "run_provenance_attestation.json"
    if provenance.get("mixed_commits") is True and not attestation_path.is_file():
        raise FileNotFoundError("Mixed comparison provenance attestation is missing")
    with comparison_path.open("r", encoding="utf-8") as handle:
        comparison_rows = json.load(handle)
    if not isinstance(comparison_rows, list):
        raise ValueError("comparison.json must be a JSON array")
    matching_rows = [
        row
        for row in comparison_rows
        if isinstance(row, dict) and str(row.get("run_id")) == run_id
    ]
    if len(matching_rows) != 1 or matching_rows[0].get("status") != "complete":
        raise ValueError(f"Original comparison must contain one complete row for run_id={run_id}")
    sources = _read_object(sources_path, "comparison sources manifest")
    source_rows = [
        row
        for row in sources.get("files", [])
        if isinstance(row, dict)
        and str(row.get("run_id")) == run_id
        and str(row.get("path", "")).replace("\\", "/")
        == f"sources/{safe_stem(run_id)}/run_manifest.json"
    ]
    if len(source_rows) != 1:
        raise ValueError(f"Comparison sources do not contain the exact run manifest: {run_id}")
    if str(source_rows[0].get("source_original_sha256", "")).lower() != run_manifest_sha256:
        raise ValueError("Comparison source run manifest SHA-256 differs from the native release")
    _verify_comparison_source_copy(
        comparison_dir, source_rows[0], "comparison source run manifest"
    )
    metric_rows = [
        row
        for row in sources.get("files", [])
        if isinstance(row, dict)
        and str(row.get("run_id")) == run_id
        and str(row.get("path", "")).replace("\\", "/")
        == f"sources/{safe_stem(run_id)}/final_metrics.json"
    ]
    if len(metric_rows) != 1:
        raise ValueError(f"Comparison sources do not contain the run's final metrics: {run_id}")
    native_final_metrics_sha256 = str(
        metric_rows[0].get("source_original_sha256", "")
    ).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", native_final_metrics_sha256):
        raise ValueError("Comparison native final metrics SHA-256 is invalid")
    _verify_comparison_source_copy(
        comparison_dir, metric_rows[0], "comparison source native final metrics"
    )

    actual = {
        "comparison_id": comparison_dir.name,
        "protocol_compatibility_sha256": sha256_file(compatibility_path),
        "comparison_sha256": sha256_file(comparison_path),
        "sources_manifest_sha256": sha256_file(sources_path),
        "run_provenance_sha256": sha256_file(provenance_path),
        "run_provenance_attestation_sha256": (
            sha256_file(attestation_path) if attestation_path.is_file() else None
        ),
        "formal_validation_sha256": sha256_file(comparison_dir / "formal_validation.json"),
        "run_manifest_sha256": run_manifest_sha256,
        "native_final_metrics_sha256": native_final_metrics_sha256,
        "run_id": run_id,
    }
    for key, value in actual.items():
        if str(expected.get(key, "")) != str(value):
            raise ValueError(
                f"Native/deployment comparison evidence mismatch for {key}: "
                f"expected={expected.get(key)!r}, actual={value!r}"
            )
    actual["release_ready"] = True
    actual["paths"] = {
        "protocol_compatibility": compatibility_path,
        "comparison": comparison_path,
        "sources_manifest": sources_path,
        "run_provenance": provenance_path,
        "run_provenance_attestation": attestation_path if attestation_path.is_file() else None,
    }
    return actual


def _verify_final_metrics(
    final_metrics_path: Path,
    predictions_path: Path,
    coco_path: Path,
    *,
    split: str,
    summary_metrics: Mapping[str, Any],
) -> None:
    final_metrics = _read_object(final_metrics_path, f"{split} final metrics")
    if final_metrics.get("evaluation_set") != split:
        raise ValueError(
            f"{split} final_metrics evaluation_set mismatch: {final_metrics.get('evaluation_set')!r}"
        )
    if final_metrics.get("metrics") != dict(summary_metrics):
        raise ValueError(f"{split} summary metrics differ from final_metrics.json")
    ground_truth = _mapping(final_metrics.get("ground_truth"), f"{split} final_metrics.ground_truth")
    if str(ground_truth.get("sha256", "")).lower() != sha256_file(coco_path):
        raise ValueError(f"{split} final metrics ground-truth SHA-256 mismatch")
    predictions = _mapping(final_metrics.get("predictions"), f"{split} final_metrics.predictions")
    if str(predictions.get("sha256", "")).lower() != sha256_file(predictions_path):
        raise ValueError(f"{split} final metrics prediction SHA-256 mismatch")


def _verify_evaluation(
    summary_path: Path,
    *,
    split: str,
    project_root: Path,
    deployment_metadata_path: Path,
    deployment_metadata_sha256: str,
    onnx_path: Path,
    onnx_sha256: str,
    deployment: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    summary = _read_object(summary_path, f"formal {split} ONNX evaluation")
    if summary.get("schema_version") != 1:
        raise ValueError(f"Unsupported {split} ONNX evaluation schema")
    if summary.get("status") != "PASS" or summary.get("mode") != "formal":
        raise ValueError(
            f"{split} ONNX evaluation must be status=PASS and mode=formal, got "
            f"{summary.get('status')!r}/{summary.get('mode')!r}"
        )
    if summary.get("split") != split:
        raise ValueError(f"Expected {split} evaluation, got split={summary.get('split')!r}")
    if summary.get("framework") != deployment.get("framework"):
        raise ValueError(f"{split} evaluation framework differs from deployment metadata")
    if summary.get("profile") != "fixed_batch1_fp32_onnxruntime":
        raise ValueError(f"{split} evaluation used an unsupported deployment profile")

    inputs = _mapping(summary.get("inputs"), f"{split} evaluation inputs")
    protocol_binding = _mapping(inputs.get("protocol_binding"), f"{split} protocol binding")
    split_binding = _mapping(inputs.get("split_binding"), f"{split} split binding")
    if protocol_binding.get("status") != "PASS":
        raise ValueError(f"{split} protocol_binding is not PASS")
    if split_binding.get("status") != "PASS":
        raise ValueError(f"{split} split_binding is not PASS")
    protocol_id = protocol_binding.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError(f"{split} protocol_binding is missing protocol_id")
    assignment_sha256 = str(split_binding.get("assignment_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", assignment_sha256):
        raise ValueError(f"{split} split_binding assignment_sha256 is invalid")
    deployment_names = [
        str(value)
        for value in _mapping(deployment.get("classes"), "deployment classes").get(
            "names", []
        )
    ]
    if inputs.get("category_names") != deployment_names:
        raise ValueError(f"{split} evaluation class names differ from deployment metadata")
    if split == "val":
        native_equivalence = _mapping(
            summary.get("native_metric_equivalence"), "validation native metric equivalence"
        )
        if native_equivalence.get("status") != "PASS":
            raise ValueError("Validation native_metric_equivalence is not PASS")
        native_reference_binding = _mapping(
            summary.get("native_reference_binding"), "validation native reference binding"
        )
        if native_reference_binding.get("status") != "PASS":
            raise ValueError("Validation native_reference_binding is not PASS")

    protocol = _mapping(summary.get("protocol"), f"{split} evaluation protocol")
    model_input = _mapping(deployment.get("model_input"), "deployment model_input")
    postprocess = _mapping(deployment.get("postprocessing"), "deployment postprocessing")
    expected_protocol = {
        "batch": int(model_input.get("batch", -1)),
        "image_size": int(model_input.get("height", -1)),
        "nms_iou": float(postprocess.get("nms_iou", -1)),
        "max_detections_per_image": int(postprocess.get("max_detections", -1)),
        "operating_confidence": float(postprocess.get("confidence", -1)),
    }
    for key, expected_value in expected_protocol.items():
        if protocol.get(key) != expected_value:
            raise ValueError(
                f"{split} evaluation protocol {key} differs from deployment metadata: "
                f"expected={expected_value!r}, actual={protocol.get(key)!r}"
            )

    artifacts = _mapping(summary.get("artifacts"), f"{split} evaluation artifacts")
    _verify_record_file(
        deployment_metadata_path,
        _mapping(artifacts.get("deployment_metadata"), f"{split} deployment metadata artifact"),
        f"{split} deployment metadata artifact",
    )
    if _record_sha(artifacts["deployment_metadata"], "deployment metadata") != deployment_metadata_sha256:
        raise ValueError(f"{split} evaluation used a different deployment metadata file")
    _verify_record_file(
        onnx_path,
        _mapping(artifacts.get("onnx"), f"{split} ONNX artifact"),
        f"{split} ONNX artifact",
    )
    if _record_sha(artifacts["onnx"], "ONNX") != onnx_sha256:
        raise ValueError(f"{split} evaluation used a different ONNX file")

    evaluation_dir = summary_path.parent
    known_paths = {
        "image_manifest": evaluation_dir / "image_manifest.json",
        "predictions": evaluation_dir / "predictions.coco.json",
        "final_metrics": evaluation_dir / "final_metrics.json",
    }
    verified_paths: dict[str, Path] = {}
    for name, path in known_paths.items():
        record = _mapping(artifacts.get(name), f"{split} {name} artifact")
        _verify_record_file(path, record, f"{split} {name} artifact")
        verified_paths[name] = path.resolve()

    coco_record = _mapping(artifacts.get("coco_annotations"), f"{split} COCO artifact")
    coco_path = _resolve_record_path(coco_record, project_root, f"{split} COCO artifact")
    _verify_record_file(coco_path, coco_record, f"{split} COCO artifact")
    verified_paths["coco_annotations"] = coco_path
    protocol_artifact = _mapping(protocol_binding.get("artifact"), f"{split} protocol artifact")
    protocol_path = _resolve_record_path(protocol_artifact, project_root, f"{split} protocol artifact")
    _verify_record_file(protocol_path, protocol_artifact, f"{split} protocol artifact")
    verified_paths["protocol"] = protocol_path
    split_artifacts = _mapping(split_binding.get("artifacts"), f"{split} split artifacts")
    for name in ("split_manifest", "split_summary"):
        record = _mapping(split_artifacts.get(name), f"{split} {name} artifact")
        path = _resolve_record_path(record, project_root, f"{split} {name} artifact")
        _verify_record_file(path, record, f"{split} {name} artifact")
        verified_paths[name] = path
    if split == "val":
        native_record = _mapping(artifacts.get("native_final_metrics"), "validation native metrics artifact")
        native_path = _resolve_record_path(native_record, project_root, "validation native metrics artifact")
        _verify_record_file(native_path, native_record, "validation native metrics artifact")
        verified_paths["native_final_metrics"] = native_path
        native_sha = sha256_file(native_path)
        if (
            str(native_reference_binding.get("expected_sha256", "")).lower() != native_sha
            or str(native_reference_binding.get("actual_sha256", "")).lower() != native_sha
        ):
            raise ValueError(
                "Validation native_reference_binding differs from its native metrics artifact"
            )

    metrics = _mapping(summary.get("metrics"), f"{split} summary metrics")
    _verify_final_metrics(
        verified_paths["final_metrics"],
        verified_paths["predictions"],
        verified_paths["coco_annotations"],
        split=split,
        summary_metrics=metrics,
    )
    image_manifest = _read_object(verified_paths["image_manifest"], f"{split} image manifest")
    image_count = int(inputs.get("image_count", -2))
    images = image_manifest.get("images")
    if (
        image_manifest.get("split") != split
        or int(image_manifest.get("image_count", -1)) != image_count
        or not isinstance(images, list)
        or len(images) != image_count
    ):
        raise ValueError(f"{split} image manifest does not match the evaluation summary")
    if any(
        not isinstance(item, dict) or item.get("synthetic_or_generated") is not False
        for item in images
    ):
        raise ValueError(f"{split} image manifest contains unconfirmed/generated image evidence")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "verified_paths": verified_paths,
        "metrics": metrics,
        "protocol": dict(protocol),
        "protocol_id": protocol_id,
        "protocol_sha256": sha256_file(verified_paths["protocol"]),
        "assignment_sha256": assignment_sha256,
        "split_manifest_sha256": sha256_file(verified_paths["split_manifest"]),
        "split_summary_sha256": sha256_file(verified_paths["split_summary"]),
        "native_final_metrics_sha256": (
            sha256_file(verified_paths["native_final_metrics"]) if split == "val" else None
        ),
    }


def _assert_no_local_paths(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if WINDOWS_ABSOLUTE.search(text) or POSIX_ABSOLUTE.search(text):
        raise ValueError(f"Published report contains an absolute path: {path}")
    sensitive_values = {
        str(Path.home()),
        Path.home().name,
        os.environ.get("USERNAME", ""),
        os.environ.get("USER", ""),
    }
    lowered = text.lower()
    for value in sensitive_values:
        if not value or len(value) < 3:
            continue
        lowered_value = value.lower()
        path_like = any(separator in lowered_value for separator in ("/", "\\"))
        found = (
            lowered_value in lowered
            if path_like
            else re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(lowered_value)}(?![A-Za-z0-9_])",
                lowered,
            )
            is not None
        )
        if found:
            raise ValueError(f"Published report contains a local user identifier: {path}")


def _verify_lfs_rules(project_root: Path, suffixes: Iterable[str]) -> dict[str, Any]:
    attributes_path = project_root / ".gitattributes"
    if not attributes_path.is_file():
        raise FileNotFoundError(attributes_path)
    lines = [line.strip() for line in attributes_path.read_text(encoding="utf-8").splitlines()]
    patterns: list[str] = []
    for suffix in suffixes:
        pattern = f"*{suffix}"
        if not any(
            line.split()[:2] == [pattern, "filter=lfs"]
            for line in lines
            if line and not line.startswith("#")
        ):
            raise ValueError(f"Git LFS rule is missing for {pattern}")
        patterns.append(pattern)
    return {"status": "PASS", "attributes": ".gitattributes", "patterns": patterns}


def _publish(
    source: Path,
    destination: Path,
    *,
    project_root: Path,
    relative_path: str,
) -> dict[str, Any]:
    record = publish_evidence_file(source, destination, project_root=project_root)
    _assert_no_local_paths(destination)
    return {"path": relative_path, **record}


def validate_promoted_deployment_for_runtime(
    *,
    project_root: Path,
    release_manifest_path: Path,
    deployment_metadata_path: Path,
    onnx_path: Path,
) -> dict[str, Any]:
    """Fail closed unless a fresh clone contains one intact promoted deployment chain."""

    project_root = project_root.resolve()
    release_manifest_path = release_manifest_path.resolve()
    deployment_metadata_path = deployment_metadata_path.resolve()
    onnx_path = onnx_path.resolve()
    relative = _relative_under(
        release_manifest_path,
        project_root / "reports" / "deployments",
        "Deployment release manifest",
    )
    if relative.name != "deployment_release_manifest.json" or len(relative.parts) != 2:
        raise ValueError(
            "Release manifest must be reports/deployments/<release>/"
            "deployment_release_manifest.json"
        )
    manifest = _read_object(release_manifest_path, "deployment release manifest")
    _assert_no_local_paths(release_manifest_path)
    release_name = _safe_release_name(manifest.get("release_name"))
    if relative.parts[0] != release_name:
        raise ValueError("Deployment release_name differs from its reports/deployments directory")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
        raise ValueError("Deployment release manifest must be schema_version=1 and status=PASS")
    gates = _mapping(manifest.get("gates"), "deployment release gates")
    missing_gates = sorted(REQUIRED_RELEASE_GATES.difference(gates))
    failing_gates = sorted(key for key, value in gates.items() if value != "PASS")
    if missing_gates or failing_gates:
        raise ValueError(
            "Deployment release gates are incomplete or not PASS: "
            f"missing={missing_gates}, failing={failing_gates}"
        )

    metadata_record = _mapping(
        manifest.get("deployment_metadata"), "deployment release metadata record"
    )
    expected_metadata_sha = str(metadata_record.get("source_sha256", "")).lower()
    actual_metadata_sha = sha256_file(deployment_metadata_path)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_metadata_sha)
        or actual_metadata_sha.lower() != expected_metadata_sha
    ):
        raise ValueError(
            "Deployment metadata SHA-256 differs from promoted release: "
            f"expected={expected_metadata_sha or 'MISSING'}, actual={actual_metadata_sha}"
        )
    _assert_no_local_paths(deployment_metadata_path)

    onnx_record = _mapping(manifest.get("onnx"), "deployment release ONNX record")
    manifest_onnx_path = _resolve_record_path(
        onnx_record, project_root, "deployment release ONNX record"
    )
    if manifest_onnx_path != onnx_path:
        raise ValueError("Camera ONNX path differs from the promoted deployment release")
    verified_onnx = _verify_record_file(onnx_path, onnx_record, "promoted deployment ONNX")

    release_dir = release_manifest_path.parent
    published_rows = manifest.get("published_files")
    if not isinstance(published_rows, list):
        raise ValueError("Deployment release manifest is missing published_files")
    rows_by_path: dict[str, Mapping[str, Any]] = {}
    for row in published_rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("Deployment release published_files contains an invalid row")
        relative_path = str(row["path"]).replace("\\", "/")
        if relative_path in rows_by_path:
            raise ValueError(f"Duplicate deployment release published path: {relative_path}")
        rows_by_path[relative_path] = row
    required_reports = {
        "deployment_metadata.json": (None, None),
        "val/onnx_split_evaluation.json": ("val", "formal"),
        "test/onnx_split_evaluation.json": ("test", "formal"),
    }
    formal_evaluations = _mapping(
        manifest.get("formal_evaluations"), "deployment formal evaluations"
    )
    for relative_path, (expected_split, expected_mode) in required_reports.items():
        row = rows_by_path.get(relative_path)
        if row is None:
            raise ValueError(f"Deployment release is missing published report: {relative_path}")
        expected_source_sha = (
            expected_metadata_sha
            if expected_split is None
            else str(
                _mapping(
                    formal_evaluations.get(expected_split),
                    f"deployment {expected_split} evaluation record",
                ).get("source_sha256", "")
            ).lower()
        )
        if str(row.get("source_original_sha256", "")).lower() != expected_source_sha:
            raise ValueError(
                f"Deployment release source hash linkage differs: {relative_path}"
            )
        report_path = (release_dir / relative_path).resolve()
        _relative_under(report_path, release_dir, f"Published report {relative_path}")
        expected_published_sha = str(row.get("published_sha256", "")).lower()
        actual_published_sha = sha256_file(report_path)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_published_sha)
            or actual_published_sha.lower() != expected_published_sha
        ):
            raise ValueError(f"Published deployment report SHA-256 mismatch: {relative_path}")
        _assert_no_local_paths(report_path)
        if expected_split is not None:
            summary = _read_object(report_path, f"published {expected_split} evaluation")
            if (
                summary.get("status") != "PASS"
                or summary.get("mode") != expected_mode
                or summary.get("split") != expected_split
            ):
                raise ValueError(
                    f"Published {expected_split} evaluation is not a formal PASS"
                )

    return {
        "status": "PASS",
        "release_name": release_name,
        "release_manifest_sha256": sha256_file(release_manifest_path),
        "deployment_metadata_sha256": actual_metadata_sha,
        "onnx_sha256": verified_onnx["sha256"],
    }


def promote_deployment_release(
    *,
    project_root: Path,
    native_artifact_path: Path,
    deployment_metadata_path: Path,
    onnx_path: Path,
    val_summary_path: Path,
    test_summary_path: Path,
    comparison_dir: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    native_artifact_path = native_artifact_path.resolve()
    deployment_metadata_path = deployment_metadata_path.resolve()
    onnx_path = onnx_path.resolve()
    val_summary_path = val_summary_path.resolve()
    test_summary_path = test_summary_path.resolve()
    comparison_dir = comparison_dir.resolve()

    reports_runs_root = project_root / "reports" / "runs"
    native_relative = _relative_under(native_artifact_path, reports_runs_root, "Native artifact manifest")
    if native_relative.name != "artifact_manifest.json" or len(native_relative.parts) != 2:
        raise ValueError(
            "Native artifact must be reports/runs/<release>/artifact_manifest.json"
        )
    native = _read_object(native_artifact_path, "native artifact manifest")
    release_name = _safe_release_name(native.get("release_name"))
    if native_relative.parts[0] != release_name:
        raise ValueError("Native artifact release_name differs from its reports/runs directory")
    run_id = str(native.get("source_run_id") or "")
    if not run_id:
        raise ValueError("Native artifact is missing source_run_id")
    native_stage = str(native.get("stage") or "")
    if not native_stage or native_stage == "smoke_not_comparable":
        raise ValueError("Native artifact is not a promotable full training release")
    run_manifest_sha = str(native.get("source_run_manifest_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", run_manifest_sha):
        raise ValueError("Native artifact source_run_manifest_sha256 is invalid")
    _relative_under(comparison_dir, project_root / "runs" / "comparisons", "Original comparison")
    _relative_under(val_summary_path, project_root, "Validation evaluation")
    _relative_under(test_summary_path, project_root, "Test evaluation")
    native_comparison = _mapping(native.get("validated_by_comparison"), "native comparison evidence")
    comparison = _verify_comparison(
        comparison_dir,
        run_id=run_id,
        run_manifest_sha256=run_manifest_sha,
        expected=native_comparison,
    )

    release_weight_root = (project_root / "weights" / "trained" / release_name).resolve()
    checkpoint_record = _mapping(native.get("checkpoint"), "native checkpoint record")
    checkpoint_path = _resolve_record_path(checkpoint_record, project_root, "native checkpoint")
    _relative_under(checkpoint_path, release_weight_root, "Promoted native checkpoint")
    checkpoint = _verify_record_file(checkpoint_path, checkpoint_record, "promoted native checkpoint")
    checkpoint["path"] = checkpoint_path.relative_to(project_root).as_posix()
    is_yolo11 = str(native.get("model", "")).lower().startswith("yolo11")
    source_original_sha = str(checkpoint_record.get("source_original_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_original_sha):
        raise ValueError("Native checkpoint record is missing source_original_sha256")
    checkpoint["source_original_sha256"] = source_original_sha
    if is_yolo11 and not (
        checkpoint_record.get("metadata_sanitized") is True
        and checkpoint_record.get("state_dict_bitwise_equal") is True
        and float(checkpoint_record.get("forward_max_abs_difference", -1)) == 0.0
        and checkpoint_record.get("source_forward_captured_before_scrub") is True
        and checkpoint_record.get("ultralytics_load") == "PASS"
    ):
        raise ValueError("YOLO11 native checkpoint sanitizer evidence is not a formal PASS")
    assert_binary_has_no_local_paths(checkpoint_path, project_root)

    _relative_under(deployment_metadata_path, release_weight_root, "Deployment metadata")
    _relative_under(onnx_path, release_weight_root, "Deployment ONNX")
    deployment = _read_object(deployment_metadata_path, "deployment metadata")
    _assert_no_local_paths(native_artifact_path)
    _assert_no_local_paths(deployment_metadata_path)
    if deployment.get("schema_version") != 1 or deployment.get("status") != "PASS":
        raise ValueError("Deployment metadata must be schema_version=1 and status=PASS")
    if deployment.get("profile") != "fixed_batch1_fp32_onnx":
        raise ValueError("Deployment metadata is not the audited fixed batch-1 FP32 profile")
    release_validation = _mapping(deployment.get("release_validation"), "deployment release_validation")
    if (
        release_validation.get("status") != "PASS"
        or release_validation.get("formal_release") is not True
    ):
        raise ValueError("Deployment release_validation is not a formal PASS")
    verification = _mapping(deployment.get("verification"), "deployment verification")
    numeric = _mapping(verification.get("numeric"), "deployment numerical verification")
    if verification.get("status") != "PASS" or numeric.get("status") != "PASS":
        raise ValueError("Deployment native/ONNX numerical verification is not PASS")
    training_run = _mapping(deployment.get("training_run"), "deployment training_run")
    if (
        str(training_run.get("run_id")) != run_id
        or training_run.get("status") != "complete"
        or str(training_run.get("stage")) != native_stage
        or str(training_run.get("model")) != str(native.get("model"))
    ):
        raise ValueError("Deployment training_run does not match the promoted full run")
    model_input = _mapping(deployment.get("model_input"), "deployment model_input")
    if (
        int(model_input.get("batch", -1)) != 1
        or int(model_input.get("height", -1)) <= 0
        or int(model_input.get("width", -2)) != int(model_input.get("height", -1))
        or model_input.get("dtype") != "float32"
    ):
        raise ValueError("Deployment model input must be fixed square batch-1 float32")
    deployment_artifacts = _mapping(deployment.get("artifacts"), "deployment artifacts")
    _verify_record_file(
        checkpoint_path,
        _mapping(deployment_artifacts.get("checkpoint"), "deployment checkpoint artifact"),
        "deployment checkpoint artifact",
    )
    if _record_sha(deployment_artifacts["checkpoint"], "deployment checkpoint") != checkpoint["sha256"]:
        raise ValueError("Deployment checkpoint differs from the promoted native checkpoint")
    publication = _mapping(deployment.get("checkpoint_publication"), "deployment checkpoint_publication")
    if (
        str(publication.get("source_original_sha256", "")).lower() != source_original_sha
        or str(publication.get("published_sha256", "")).lower() != checkpoint["sha256"]
    ):
        raise ValueError("Deployment checkpoint publication bridge differs from native artifact")
    if is_yolo11 and not (
        publication.get("metadata_sanitized") is True
        and publication.get("state_dict_bitwise_equal") is True
        and float(publication.get("forward_max_abs_difference", -1)) == 0.0
        and publication.get("source_forward_captured_before_scrub") is True
        and publication.get("ultralytics_load") == "PASS"
    ):
        raise ValueError("Deployment did not retain YOLO11 sanitizer evidence")
    run_manifest_record = _mapping(
        deployment_artifacts.get("run_manifest"), "deployment run manifest artifact"
    )
    if _record_sha(run_manifest_record, "deployment run manifest") != run_manifest_sha:
        raise ValueError("Deployment run manifest differs from the promoted native release")
    run_manifest_path = _resolve_record_path(
        run_manifest_record, project_root, "deployment run manifest artifact"
    )
    _verify_record_file(
        run_manifest_path, run_manifest_record, "deployment run manifest artifact"
    )
    run_manifest_document = _read_object(run_manifest_path, "deployment run manifest")
    original_record = _mapping(
        run_manifest_document.get("best_checkpoint"), "run manifest best_checkpoint"
    )
    if str(original_record.get("sha256", "")).lower() != source_original_sha:
        raise ValueError("Run manifest original checkpoint differs from publication bridge")
    onnx_record = _mapping(deployment_artifacts.get("onnx"), "deployment ONNX artifact")
    onnx_file_name = onnx_record.get("file_name")
    if (
        not isinstance(onnx_file_name, str)
        or Path(onnx_file_name).name != onnx_file_name
        or onnx_path.parent != deployment_metadata_path.parent
        or onnx_path.name != onnx_file_name
    ):
        raise ValueError("Deployment ONNX must be the recorded file next to its metadata")
    onnx = _verify_record_file(onnx_path, onnx_record, "deployment ONNX artifact")
    onnx["path"] = onnx_path.relative_to(project_root).as_posix()
    assert_binary_has_no_local_paths(onnx_path, project_root)
    metadata_sha = sha256_file(deployment_metadata_path)

    for key in (
        "comparison_id",
        "protocol_compatibility_sha256",
        "comparison_sha256",
        "sources_manifest_sha256",
        "run_provenance_sha256",
        "run_provenance_attestation_sha256",
        "formal_validation_sha256",
        "run_manifest_sha256",
        "native_final_metrics_sha256",
        "run_id",
    ):
        if str(release_validation.get(key, "")) != str(comparison.get(key, "")):
            raise ValueError(f"Deployment release_validation differs from original comparison: {key}")

    val = _verify_evaluation(
        val_summary_path,
        split="val",
        project_root=project_root,
        deployment_metadata_path=deployment_metadata_path,
        deployment_metadata_sha256=metadata_sha,
        onnx_path=onnx_path,
        onnx_sha256=onnx["sha256"],
        deployment=deployment,
    )
    test = _verify_evaluation(
        test_summary_path,
        split="test",
        project_root=project_root,
        deployment_metadata_path=deployment_metadata_path,
        deployment_metadata_sha256=metadata_sha,
        onnx_path=onnx_path,
        onnx_sha256=onnx["sha256"],
        deployment=deployment,
    )
    if val["protocol_id"] != test["protocol_id"]:
        raise ValueError("Validation/test protocol IDs differ")
    if val["protocol_sha256"] != test["protocol_sha256"]:
        raise ValueError("Validation/test protocol artifacts differ")
    if val["protocol"] != test["protocol"]:
        raise ValueError("Validation/test evaluation protocols differ")
    if val["assignment_sha256"] != test["assignment_sha256"]:
        raise ValueError("Validation/test split assignment hashes differ")
    if (
        val["split_manifest_sha256"] != test["split_manifest_sha256"]
        or val["split_summary_sha256"] != test["split_summary_sha256"]
    ):
        raise ValueError("Validation/test split evidence artifacts differ")
    if val["native_final_metrics_sha256"] != comparison["native_final_metrics_sha256"]:
        raise ValueError(
            "Validation native final metrics differ from the metrics frozen in the comparison"
        )

    lfs = _verify_lfs_rules(project_root, {checkpoint_path.suffix.lower(), ".onnx"})
    destination_root = project_root / "reports" / "deployments" / release_name
    if destination_root.exists():
        raise FileExistsError(f"Deployment release already exists: {destination_root}")
    destination_parent = destination_root.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    staging = destination_parent / f".{release_name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(staging)
    published: list[dict[str, Any]] = []
    try:
        staging.mkdir()
        published.append(
            _publish(
                native_artifact_path,
                staging / "native_artifact_manifest.json",
                project_root=project_root,
                relative_path="native_artifact_manifest.json",
            )
        )
        published.append(
            _publish(
                deployment_metadata_path,
                staging / "deployment_metadata.json",
                project_root=project_root,
                relative_path="deployment_metadata.json",
            )
        )
        for split, evidence in (("val", val), ("test", test)):
            source_dir = evidence["summary_path"].parent
            for name in EVALUATION_REPORT_FILES:
                source = source_dir / name
                if not source.is_file():
                    if name in {"onnx_split_evaluation.json", "final_metrics.json", "image_manifest.json"}:
                        raise FileNotFoundError(f"Required {split} report is missing: {source}")
                    continue
                relative = f"{split}/{name}"
                published.append(
                    _publish(
                        source,
                        staging / relative,
                        project_root=project_root,
                        relative_path=relative,
                    )
                )
        for name in COMPARISON_REPORT_FILES:
            source = comparison_dir / name
            if name == "run_provenance_attestation.json" and not source.is_file():
                continue
            relative = f"comparison/{name}"
            published.append(
                _publish(
                    source,
                    staging / relative,
                    project_root=project_root,
                    relative_path=relative,
                )
            )

        manifest = {
            "schema_version": 1,
            "status": "PASS",
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "release_name": release_name,
            "source_run_id": run_id,
            "model": native.get("model"),
            "protocol_id": val["protocol_id"],
            "gates": {
                "native_release": "PASS",
                "deployment_release_validation": "PASS",
                "native_onnx_numeric_equivalence": "PASS",
                "validation_formal_split": "PASS",
                "validation_native_metric_equivalence": "PASS",
                "validation_native_reference_binding": "PASS",
                "test_formal_split": "PASS",
                "protocol_binding": "PASS",
                "split_binding": "PASS",
                "artifact_hashes_and_sizes": "PASS",
                "publication_path_privacy": "PASS",
                "git_lfs_rules": "PASS",
            },
            "native_checkpoint": checkpoint,
            "onnx": onnx,
            "deployment_metadata": {
                "source_sha256": metadata_sha,
                "published_path": "deployment_metadata.json",
            },
            "comparison": {
                key: value
                for key, value in comparison.items()
                if key not in {"paths", "release_ready"}
            }
            | {"release_ready": True},
            "formal_evaluations": {
                "val": {
                    "source_sha256": sha256_file(val_summary_path),
                    "published_path": "val/onnx_split_evaluation.json",
                    "metrics": val["metrics"],
                    "native_metric_equivalence": "PASS",
                    "native_reference_sha256": val["native_final_metrics_sha256"],
                },
                "test": {
                    "source_sha256": sha256_file(test_summary_path),
                    "published_path": "test/onnx_split_evaluation.json",
                    "metrics": test["metrics"],
                },
            },
            "git_lfs": lfs,
            "published_files": sorted(published, key=lambda item: str(item["path"])),
            "local_source_path_included": False,
            "raw_images_included": False,
            "predictions_included": False,
            "generated_visual_policy": "Only evaluator-generated numeric plots are included; no AI-generated images.",
        }
        manifest_path = staging / "deployment_release_manifest.json"
        write_json(manifest_path, manifest)
        _assert_no_local_paths(manifest_path)
        for path in staging.rglob("*"):
            if path.is_file():
                _assert_no_local_paths(path)
        staging.replace(destination_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a native release plus formally verified ONNX val/test evidence"
    )
    parser.add_argument("--native-artifact", type=Path, required=True)
    parser.add_argument("--deployment-metadata", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--val-evaluation", type=Path, required=True)
    parser.add_argument("--test-evaluation", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    manifest = promote_deployment_release(
        project_root=project_root,
        native_artifact_path=args.native_artifact,
        deployment_metadata_path=args.deployment_metadata,
        onnx_path=args.onnx,
        val_summary_path=args.val_evaluation,
        test_summary_path=args.test_evaluation,
        comparison_dir=args.comparison_dir,
    )
    print("DEPLOYMENT PROMOTION: PASS")
    print(f"release          {manifest['release_name']}")
    print(f"model            {manifest['model']}")
    print(f"checkpoint sha   {manifest['native_checkpoint']['sha256']}")
    print(f"onnx sha         {manifest['onnx']['sha256']}")
    print(
        "output           "
        + portable_path(
            project_root / "reports" / "deployments" / str(manifest["release_name"])
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
