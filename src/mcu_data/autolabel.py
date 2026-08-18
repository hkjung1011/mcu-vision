from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .common import load_yaml, sha256_file, write_json
from .contracts import (
    ContractError,
    Ontology,
    canonical_sha256,
    load_json_object,
    load_ontology,
    require_sha256,
    safe_relative_path,
)
from .runlog import (
    GpuSampler,
    collect_system_environment,
    collect_torch_environment,
    configure_utf8_output,
    print_section,
    tee_console,
    utc_now_precise,
    write_pip_freeze,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
AUTOLABEL_SOURCE_SCHEMA = "mcu.autolabel-source.v1"
TRUST_REGISTRY_SCHEMA = "mcu.data-trust-registry.v1"
LOCKED_SPLIT_SCHEMA = "mcu.locked-split-evidence.v1"
ALLOWED_TEACHER_ANNOTATION_STATES = {"manual_seed", "reviewed_train"}
LOCKED_SPLIT_ROLES = {"unlabeled_train", "gold_validation_locked", "test_locked"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source.resolve()]
    if not source.is_dir():
        raise FileNotFoundError(f"Image source does not exist: {source}")
    return sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _ontology_sha(document: dict[str, Any], *, field: str) -> str:
    direct = document.get("ontology_sha256")
    if direct is None and isinstance(document.get("ontology"), dict):
        direct = document["ontology"].get("sha256")
    return require_sha256(direct, field=field)


def _manifest_class_map(value: Any, *, field: str) -> dict[int, str]:
    if not isinstance(value, dict) or not value:
        raise ContractError(f"{field} must be a non-empty id-to-name mapping")
    result: dict[int, str] = {}
    for raw_id, raw_name in value.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{field} contains a non-integer class id: {raw_id!r}") from exc
        if not isinstance(raw_name, str) or not raw_name:
            raise ContractError(f"{field}[{class_id}] must be a non-empty class name")
        if class_id in result:
            raise ContractError(f"{field} contains duplicate class id {class_id}")
        result[class_id] = raw_name
    if sorted(result) != list(range(len(result))):
        raise ContractError(f"{field} ids must be contiguous from zero")
    return result


def _normalized_image_binding(
    row: Any, *, index: int, role: str, field_prefix: str
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ContractError(f"{field_prefix}[{index}] must be an object")
    image_id = row.get("image_id")
    if not isinstance(image_id, str) or not image_id.strip():
        raise ContractError(f"{field_prefix}[{index}].image_id must be a non-empty string")
    relative = safe_relative_path(row.get("path"), field=f"{field_prefix}[{index}].path")
    if row.get("role", role) != role:
        raise ContractError(f"{field_prefix}[{index}] role must be {role}")
    try:
        width = int(row["width"])
        height = int(row["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"{field_prefix}[{index}] dimensions are invalid") from exc
    if width <= 0 or height <= 0:
        raise ContractError(f"{field_prefix}[{index}] dimensions are invalid")
    return {
        "image_id": image_id.strip(),
        "path": relative,
        "sha256": require_sha256(
            row.get("sha256"), field=f"{field_prefix}[{index}].sha256"
        ),
        "width": width,
        "height": height,
        "role": role,
    }


def _load_trusted_source_approval(
    *,
    trusted_registry_path: Path,
    dataset_id: str,
    ontology: Ontology,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    registry_path = trusted_registry_path.resolve()
    registry = load_yaml(registry_path)
    if registry.get("schema_version") != TRUST_REGISTRY_SCHEMA:
        raise ContractError(f"Trusted registry schema must be {TRUST_REGISTRY_SCHEMA}")
    registry_id = registry.get("registry_id")
    if not isinstance(registry_id, str) or not registry_id:
        raise ContractError("Trusted registry registry_id must be a non-empty string")
    actual_registry_sha = sha256_file(registry_path)
    if source_manifest.get("trusted_registry_id") != registry_id:
        raise ContractError("Source manifest trusted_registry_id differs from project registry")
    if require_sha256(
        source_manifest.get("trusted_registry_sha256"),
        field="source manifest trusted_registry_sha256",
    ) != actual_registry_sha:
        raise ContractError("Source manifest is bound to a different trusted registry revision")
    datasets = registry.get("datasets")
    if not isinstance(datasets, dict) or dataset_id not in datasets:
        raise ContractError(
            f"Dataset {dataset_id!r} is not approved in the project trusted registry"
        )
    entry = datasets[dataset_id]
    if not isinstance(entry, dict) or entry.get("status") != "APPROVED":
        raise ContractError(f"Trusted dataset {dataset_id!r} is not status=APPROVED")
    entry_sha = canonical_sha256(entry)
    if require_sha256(
        source_manifest.get("trusted_dataset_record_sha256"),
        field="source manifest trusted_dataset_record_sha256",
    ) != entry_sha:
        raise ContractError("Source manifest trusted dataset record hash differs from registry")
    if require_sha256(
        entry.get("ontology_sha256"), field="trusted dataset ontology_sha256"
    ) != ontology.sha256:
        raise ContractError("Trusted dataset ontology differs from selected ontology")
    evidence_record = entry.get("locked_split_evidence")
    if not isinstance(evidence_record, dict):
        raise ContractError("Trusted dataset must contain locked_split_evidence")
    evidence_relative = safe_relative_path(
        evidence_record.get("path"), field="trusted dataset locked_split_evidence.path"
    )
    project_root_path = registry_path.parent.parent.resolve()
    evidence_path = (project_root_path / evidence_relative).resolve()
    try:
        evidence_path.relative_to(project_root_path)
    except ValueError as exc:
        raise ContractError("Locked split evidence escapes the project root") from exc
    expected_evidence_sha = require_sha256(
        evidence_record.get("sha256"), field="trusted dataset locked_split_evidence.sha256"
    )
    if not evidence_path.is_file() or sha256_file(evidence_path) != expected_evidence_sha:
        raise ContractError("Locked split evidence file is missing or differs from registry")
    if require_sha256(
        source_manifest.get("locked_split_evidence_sha256"),
        field="source manifest locked_split_evidence_sha256",
    ) != expected_evidence_sha:
        raise ContractError("Source manifest is bound to different locked split evidence")
    evidence = load_json_object(evidence_path, label="locked split evidence")
    if evidence.get("schema_version") != LOCKED_SPLIT_SCHEMA:
        raise ContractError(f"Locked split evidence schema must be {LOCKED_SPLIT_SCHEMA}")
    if evidence.get("dataset_id") != dataset_id:
        raise ContractError("Locked split evidence dataset_id differs from source manifest")
    if _ontology_sha(evidence, field="locked split evidence ontology_sha256") != ontology.sha256:
        raise ContractError("Locked split evidence ontology differs from selected ontology")
    splits = evidence.get("splits")
    if not isinstance(splits, dict) or set(splits) != LOCKED_SPLIT_ROLES:
        raise ContractError(
            "Locked split evidence must explicitly contain unlabeled_train, "
            "gold_validation_locked, and test_locked"
        )
    normalized_splits: dict[str, list[dict[str, Any]]] = {}
    sha_roles: dict[str, str] = {}
    id_roles: dict[str, str] = {}
    path_roles: dict[str, str] = {}
    for role in sorted(LOCKED_SPLIT_ROLES):
        split = splits[role]
        if not isinstance(split, dict) or not isinstance(split.get("images"), list):
            raise ContractError(f"Locked split {role} must contain an images list")
        rows = [
            _normalized_image_binding(
                row, index=index, role=role, field_prefix=f"locked splits.{role}.images"
            )
            for index, row in enumerate(split["images"], start=1)
        ]
        rows.sort(key=lambda row: (str(row["path"]), str(row["image_id"])))
        if len({row["image_id"] for row in rows}) != len(rows):
            raise ContractError(f"Locked split {role} contains duplicate image_id")
        if len({row["path"] for row in rows}) != len(rows):
            raise ContractError(f"Locked split {role} contains duplicate path")
        expected_list_sha = canonical_sha256(rows)
        if require_sha256(
            split.get("image_list_sha256"),
            field=f"locked splits.{role}.image_list_sha256",
        ) != expected_list_sha:
            raise ContractError(f"Locked split {role} image list hash is invalid")
        for row in rows:
            stable_id = str(row["image_id"])
            relative_path = str(row["path"])
            previous_id_role = id_roles.get(stable_id)
            if previous_id_role is not None:
                raise ContractError(
                    f"Locked split evidence reuses image_id {stable_id!r} in "
                    f"{previous_id_role} and {role}"
                )
            previous_path_role = path_roles.get(relative_path)
            if previous_path_role is not None:
                raise ContractError(
                    f"Locked split evidence reuses image path {relative_path!r} in "
                    f"{previous_path_role} and {role}"
                )
            previous = sha_roles.get(str(row["sha256"]))
            if previous is not None:
                raise ContractError(
                    f"Locked split evidence reuses an image SHA in {previous} and {role}"
                )
            id_roles[stable_id] = role
            path_roles[relative_path] = role
            sha_roles[str(row["sha256"])] = role
        normalized_splits[role] = rows
    approved = normalized_splits.get("unlabeled_train", [])
    if not approved:
        raise ContractError("Trusted locked split has no approved unlabeled_train images")
    return {
        "registry_id": registry_id,
        "registry_sha256": actual_registry_sha,
        "dataset_record_sha256": entry_sha,
        "locked_split_evidence_path": evidence_relative,
        "locked_split_evidence_sha256": expected_evidence_sha,
        "approved_unlabeled_train_images": approved,
        "protected_image_sha_roles": {
            sha: role for sha, role in sha_roles.items() if role != "unlabeled_train"
        },
    }


def validate_source_binding(
    *,
    source_manifest_path: Path,
    source: Path,
    ontology: Ontology,
    trusted_registry_path: Path,
) -> dict[str, Any]:
    manifest_path = source_manifest_path.resolve()
    document = load_json_object(manifest_path, label="autolabel source manifest")
    if document.get("schema_version") != AUTOLABEL_SOURCE_SCHEMA:
        raise ContractError(
            f"Autolabel source manifest schema must be {AUTOLABEL_SOURCE_SCHEMA}"
        )
    if document.get("role") != "unlabeled_train":
        raise ContractError(
            "Autolabel source role must be unlabeled_train; validation/test/gold inputs are forbidden"
        )
    if _ontology_sha(document, field="source manifest ontology_sha256") != ontology.sha256:
        raise ContractError("Autolabel source manifest ontology hash differs from the selected ontology")
    dataset_id = document.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ContractError("Source manifest dataset_id must be a non-empty string")
    trust = _load_trusted_source_approval(
        trusted_registry_path=trusted_registry_path,
        dataset_id=dataset_id,
        ontology=ontology,
        source_manifest=document,
    )
    raw_images = document.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ContractError("Autolabel source manifest images must be a non-empty list")
    declared: dict[str, dict[str, Any]] = {}
    binding_rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_images, start=1):
        normalized = _normalized_image_binding(
            row,
            index=index,
            role="unlabeled_train",
            field_prefix="images",
        )
        relative = str(normalized["path"])
        if relative in declared:
            raise ContractError(f"Duplicate source manifest image path: {relative}")
        declared[relative] = normalized
        binding_rows.append(normalized)
    if len({row["image_id"] for row in binding_rows}) != len(binding_rows):
        raise ContractError("Source manifest contains duplicate image_id")
    binding_rows.sort(key=lambda row: (str(row["path"]), str(row["image_id"])))
    expected_image_list_sha = canonical_sha256(binding_rows)
    declared_image_list_sha = require_sha256(
        document.get("image_list_sha256"), field="source manifest image_list_sha256"
    )
    if declared_image_list_sha != expected_image_list_sha:
        raise ContractError("Source manifest image_list_sha256 does not match its image records")
    protected = trust["protected_image_sha_roles"]
    redeclared = [
        {"image_id": row["image_id"], "sha256": row["sha256"], "locked_role": protected[row["sha256"]]}
        for row in binding_rows
        if row["sha256"] in protected
    ]
    if redeclared:
        raise ContractError(
            "Source manifest attempts to redeclare locked validation/test image SHA values: "
            f"{redeclared[:10]}"
        )
    if binding_rows != trust["approved_unlabeled_train_images"]:
        raise ContractError(
            "Source manifest image IDs/paths/dimensions/SHA values do not exactly match the "
            "approved unlabeled_train set in locked split evidence"
        )

    actual_paths = iter_images(source)
    actual: dict[str, Path] = {}
    for path in actual_paths:
        relative = path.name if source.is_file() else path.relative_to(source.resolve()).as_posix()
        if relative in actual:
            raise ContractError(f"Duplicate runtime source image path: {relative}")
        actual[relative] = path
    if set(actual) != set(declared):
        missing = sorted(set(declared) - set(actual))[:10]
        unexpected = sorted(set(actual) - set(declared))[:10]
        raise ContractError(
            f"Runtime source image set differs from manifest: missing={missing}, unexpected={unexpected}"
        )
    for relative, path in actual.items():
        expected = declared[relative]
        if sha256_file(path) != expected["sha256"]:
            raise ContractError(f"Runtime source image hash differs from manifest: {relative}")
        with Image.open(path) as opened:
            opened.load()
            dimensions = ImageOps.exif_transpose(opened).size
        if dimensions != (expected["width"], expected["height"]):
            raise ContractError(
                f"Runtime source image dimensions differ from manifest: {relative}"
            )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "schema_version": document["schema_version"],
        "dataset_id": dataset_id,
        "role": "unlabeled_train",
        "ontology_sha256": ontology.sha256,
        "image_list_sha256": declared_image_list_sha,
        "image_count": len(binding_rows),
        "images": binding_rows,
        "trust": {
            "registry_id": trust["registry_id"],
            "registry_sha256": trust["registry_sha256"],
            "dataset_record_sha256": trust["dataset_record_sha256"],
            "locked_split_evidence_path": trust["locked_split_evidence_path"],
            "locked_split_evidence_sha256": trust["locked_split_evidence_sha256"],
        },
    }


def validate_teacher_binding(
    *, teacher_manifest_path: Path, model_path: Path, ontology: Ontology
) -> dict[str, Any]:
    manifest_path = teacher_manifest_path.resolve()
    document = load_json_object(manifest_path, label="teacher manifest")
    if document.get("training_annotation_state") not in ALLOWED_TEACHER_ANNOTATION_STATES:
        raise ContractError(
            "Teacher training_annotation_state must be manual_seed or reviewed_train"
        )
    if _ontology_sha(document, field="teacher ontology_sha256") != ontology.sha256:
        raise ContractError("Teacher ontology hash differs from the selected ontology")
    expected_checkpoint = require_sha256(
        document.get("checkpoint_sha256"), field="teacher checkpoint_sha256"
    )
    actual_checkpoint = sha256_file(model_path)
    if expected_checkpoint != actual_checkpoint:
        raise ContractError("Teacher checkpoint hash differs from teacher manifest")
    class_map = _manifest_class_map(document.get("class_map"), field="teacher class_map")
    if class_map != ontology.classes_by_id:
        raise ContractError(
            "Teacher class map must exactly match the selected frozen ontology: "
            f"teacher={class_map}, ontology={ontology.classes_by_id}"
        )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": actual_checkpoint,
        "ontology_sha256": ontology.sha256,
        "training_annotation_state": document["training_annotation_state"],
        "training_dataset_id": document.get("training_dataset_id"),
        "class_map": class_map,
    }


def validate_calibration_binding(
    *, calibration_path: Path, teacher_sha256: str, ontology: Ontology
) -> dict[str, Any]:
    document = load_json_object(calibration_path.resolve(), label="calibration metrics")
    binding = document.get("calibration_binding")
    if not isinstance(binding, dict):
        raise ContractError("Calibration metrics must contain calibration_binding")
    if binding.get("source_role") != "gold_validation_locked":
        raise ContractError("Calibration source_role must be gold_validation_locked")
    if require_sha256(binding.get("teacher_sha256"), field="calibration teacher_sha256") != teacher_sha256:
        raise ContractError("Calibration teacher hash differs from the selected teacher")
    if require_sha256(binding.get("ontology_sha256"), field="calibration ontology_sha256") != ontology.sha256:
        raise ContractError("Calibration ontology hash differs from the selected ontology")
    image_list_sha = require_sha256(
        binding.get("image_list_sha256"), field="calibration image_list_sha256"
    )
    return {
        "source_role": "gold_validation_locked",
        "teacher_sha256": teacher_sha256,
        "ontology_sha256": ontology.sha256,
        "image_list_sha256": image_list_sha,
    }


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= 0 or tile_size <= 0:
        raise ValueError("length and tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("tile overlap must be in [0, 1)")
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1 - overlap))))
    last = length - tile_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def image_tiles(
    image: Image.Image, tile_size: int, overlap: float
) -> list[tuple[Image.Image, int, int]]:
    if tile_size <= 0:
        return [(image.copy(), 0, 0)]
    x_starts = tile_starts(image.width, tile_size, overlap)
    y_starts = tile_starts(image.height, tile_size, overlap)
    tiles = []
    for top in y_starts:
        for left in x_starts:
            right = min(left + tile_size, image.width)
            bottom = min(top + tile_size, image.height)
            tiles.append((image.crop((left, top, right, bottom)), left, top))
    return tiles


def box_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_nms(
    predictions: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("NMS IoU threshold must be in [0, 1]")
    kept: list[dict[str, Any]] = []
    for candidate in sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True):
        duplicate = any(
            int(candidate["class_id"]) == int(selected["class_id"])
            and box_iou(candidate["xyxy"], selected["xyxy"]) >= iou_threshold
            for selected in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pending_coco_document(
    *,
    source_binding: dict[str, Any],
    class_names: dict[int, str],
    prediction_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    bindings = [
        {
            "image_id": row["image_id"],
            "path": row["path"],
            "sha256": row["sha256"],
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        for row in source_binding["images"]
    ]
    bindings.sort(key=lambda row: (str(row["path"]), str(row["image_id"])))
    image_id_by_path = {
        str(row["path"]): index for index, row in enumerate(bindings, start=1)
    }
    images = [
        {
            "id": image_id_by_path[str(row["path"])],
            "file_name": row["path"],
            "width": row["width"],
            "height": row["height"],
            "mcu_image_id": row["image_id"],
            "sha256": row["sha256"],
        }
        for row in bindings
    ]
    annotations: list[dict[str, Any]] = []
    for annotation_id, row in enumerate(prediction_rows, start=1):
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id_by_path[str(row["image"])],
                "category_id": int(row["class_id"]) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "score": float(row["confidence"]),
                "attributes": {"occluded": False, "truncated": False},
            }
        )
    document = {
        "info": {
            "description": "Pending autolabel reference; every annotation requires human review",
            "dataset_id": source_binding["dataset_id"],
            "ontology_sha256": source_binding["ontology_sha256"],
            "image_bindings_sha256": canonical_sha256(bindings),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id + 1, "name": name, "ontology_class_id": class_id}
            for class_id, name in sorted(class_names.items())
        ],
    }
    return document, canonical_sha256(bindings)


def _calibration_confidence(value: Any, *, field: str) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"{field} must be finite and in [0, 1] (numeric value required)"
        ) from exc
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ContractError(f"{field} must be finite and in [0, 1]")
    return confidence


def _calibration_thresholds(path: Path | None) -> tuple[dict[str, float], dict[str, Any]]:
    if path is None:
        return {}, {"status": "NOT_PROVIDED"}
    document = json.loads(path.read_text(encoding="utf-8"))
    thresholds: dict[str, float] = {}
    rows = document.get("pseudo_label_calibration_by_class", [])
    if not isinstance(rows, list):
        raise ContractError("pseudo_label_calibration_by_class must be a list")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ContractError(f"Calibration row {index} must be an object")
        if row.get("confidence") is not None:
            category = row.get("category_name")
            if not isinstance(category, str) or not category:
                raise ContractError(f"Calibration row {index} category_name is invalid")
            if category in thresholds:
                raise ContractError(f"Calibration contains duplicate category {category!r}")
            confidence = _calibration_confidence(
                row["confidence"], field=f"Calibration confidence for {category!r}"
            )
            thresholds[category] = confidence
    global_record = document.get("pseudo_label_calibration", {})
    if not isinstance(global_record, dict):
        raise ContractError("pseudo_label_calibration must be an object")
    global_value = global_record.get("confidence")
    if global_value is not None:
        confidence = _calibration_confidence(
            global_value, field="Global calibration confidence"
        )
        thresholds["__global__"] = confidence
    return thresholds, {
        "status": "VALIDATION_DERIVED" if thresholds else "NO_VALID_THRESHOLD",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "thresholds": thresholds,
    }


def _class_names(model: Any) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def _threshold_for(
    class_id: int,
    class_name: str,
    calibration: dict[str, float],
    manual_threshold: float | None,
) -> float | None:
    if manual_threshold is not None:
        return manual_threshold
    if class_name in calibration:
        return calibration[class_name]
    if str(class_id) in calibration:
        return calibration[str(class_id)]
    return calibration.get("__global__")


def _relative_image_path(path: Path, source: Path) -> Path:
    return Path(path.name) if source.is_file() else path.relative_to(source.resolve())


def _draw_preview(
    image: Image.Image,
    predictions: list[dict[str, Any]],
    output: Path,
) -> None:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    width = max(2, int(round(max(preview.size) / 500)))
    for prediction in predictions:
        class_id = int(prediction["class_id"])
        color = (
            64 + (class_id * 97) % 192,
            64 + (class_id * 57) % 192,
            64 + (class_id * 137) % 192,
        )
        xyxy = tuple(float(value) for value in prediction["xyxy"])
        draw.rectangle(xyxy, outline=color, width=width)
        label = (
            f"{prediction['class_name']} {prediction['confidence']:.3f} "
            f"[{prediction['review_status']}]"
        )
        text_box = draw.textbbox((xyxy[0], xyxy[1]), label)
        text_height = text_box[3] - text_box[1]
        text_width = text_box[2] - text_box[0]
        top = max(0.0, xyxy[1] - text_height - 4)
        draw.rectangle((xyxy[0], top, xyxy[0] + text_width + 4, top + text_height + 4), fill=color)
        draw.text((xyxy[0] + 2, top + 2), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.convert("RGB").save(output, quality=92)


def _write_yolo_labels(
    path: Path,
    predictions: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> None:
    lines = []
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["xyxy"]
        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        lines.append(
            f"{int(prediction['class_id'])} {center_x:.8f} {center_y:.8f} "
            f"{width:.8f} {height:.8f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _predict_image(
    model: Any,
    image: Image.Image,
    args: argparse.Namespace,
    class_names: dict[int, str],
    selected_classes: set[int] | None,
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], int]:
    tiles = image_tiles(image, args.tile_size, args.tile_overlap)
    tile_images = [tile for tile, _, _ in tiles]
    results = model.predict(
        source=tile_images,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.prediction_floor,
        iou=args.tile_nms_iou,
        max_det=args.max_detections_per_tile,
        quantize=(
            16
            if not args.fp32 and str(args.device).lower() != "cpu"
            else None
        ),
        verbose=False,
        save=False,
        stream=False,
    )
    predictions: list[dict[str, Any]] = []
    for result, (_, offset_x, offset_y) in zip(results, tiles, strict=True):
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()
        classes = [int(value) for value in result.boxes.cls.detach().cpu().tolist()]
        for box, confidence, class_id in zip(boxes, confidences, classes, strict=True):
            if selected_classes is not None and class_id not in selected_classes:
                continue
            x1 = max(0.0, min(float(box[0]) + offset_x, float(image.width)))
            y1 = max(0.0, min(float(box[1]) + offset_y, float(image.height)))
            x2 = max(0.0, min(float(box[2]) + offset_x, float(image.width)))
            y2 = max(0.0, min(float(box[3]) + offset_y, float(image.height)))
            if min(x2 - x1, y2 - y1) < args.minimum_box_pixels:
                continue
            class_name = class_names.get(class_id, str(class_id))
            high_threshold = _threshold_for(
                class_id, class_name, thresholds, args.high_confidence
            )
            review_status = (
                "high_confidence_candidate"
                if high_threshold is not None and float(confidence) >= high_threshold
                else "review_required"
            )
            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "xyxy": [x1, y1, x2, y2],
                    "width_px": x2 - x1,
                    "height_px": y2 - y1,
                    "high_confidence_threshold": high_threshold,
                    "review_status": review_status,
                }
            )
    return class_aware_nms(predictions, args.merge_nms_iou), len(tiles)


def _write_review_instructions(path: Path) -> None:
    path.write_text(
        "# Pending pseudo-label review\n\n"
        "이 폴더의 라벨은 ground truth가 아닙니다. `high_confidence_candidate`도 자동 승인하지 않습니다.\n\n"
        "1. 모든 box의 위치와 class를 확인합니다.\n"
        "2. 모델이 놓친 목표 객체를 추가합니다. 특히 `empty_prediction_review_required`를 우선 봅니다.\n"
        "3. 잘못된 box와 중복 box를 삭제합니다.\n"
        "4. 승인한 결과만 canonical train annotation으로 별도 export합니다.\n"
        "5. validation/test 이미지는 이 pending pool에 넣지 않습니다.\n\n"
        "`predictions.csv`가 정확한 score/좌표 원본이고, `previews/`는 검수 보조 이미지입니다.\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Create review-only YOLO pseudo-label proposals with optional tiled inference"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="Hash-bound mcu.autolabel-source.v1 manifest with role=unlabeled_train",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        required=True,
        help="Teacher checkpoint/class/ontology provenance manifest",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=root / "configs" / "classes.smd_v1.yaml",
        help="Frozen class ontology used by source, teacher, and calibration",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=root / "runs" / "autolabel")
    parser.add_argument("--calibration", type=Path, help="final_metrics.json from a manual gold validation set")
    parser.add_argument("--high-confidence", type=float, help="manual override; prefer --calibration")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=640, help="0 disables tiling")
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--prediction-floor", type=float, default=0.10)
    parser.add_argument("--tile-nms-iou", type=float, default=0.65)
    parser.add_argument("--merge-nms-iou", type=float, default=0.50)
    parser.add_argument("--minimum-box-pixels", type=float, default=2.0)
    parser.add_argument("--max-detections-per-tile", type=int, default=1000)
    parser.add_argument("--classes", type=int, nargs="+")
    parser.add_argument("--device", default="0")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--gpu-sample-seconds", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    if not 0 <= args.prediction_floor <= 1:
        raise ValueError("prediction floor must be in [0, 1]")
    if args.high_confidence is not None and not 0 <= args.high_confidence <= 1:
        raise ValueError("high confidence must be in [0, 1]")
    if args.tile_size < 0:
        raise ValueError("tile size must be zero or positive")
    if args.tile_size and not 0 <= args.tile_overlap < 1:
        raise ValueError("tile overlap must be in [0, 1)")
    source = args.source.resolve()
    source_manifest_path = args.source_manifest.resolve()
    model_path = args.model.resolve()
    teacher_manifest_path = args.teacher_manifest.resolve()
    ontology = load_ontology(args.ontology.resolve())
    calibration_path = args.calibration.resolve() if args.calibration else None
    images = iter_images(source)
    if not images:
        raise ValueError(f"No supported images found: {source}")
    if not model_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {model_path}")
    source_binding = validate_source_binding(
        source_manifest_path=source_manifest_path,
        source=source,
        ontology=ontology,
        trusted_registry_path=project_root() / "configs" / "data_trust_registry.yaml",
    )
    teacher_binding = validate_teacher_binding(
        teacher_manifest_path=teacher_manifest_path,
        model_path=model_path,
        ontology=ontology,
    )
    calibration_binding = None
    if calibration_path is not None:
        calibration_binding = validate_calibration_binding(
            calibration_path=calibration_path,
            teacher_sha256=teacher_binding["checkpoint_sha256"],
            ontology=ontology,
        )
    if args.high_confidence is not None and calibration_binding is None:
        raise ContractError(
            "--high-confidence requires a hash-bound gold_validation_locked calibration"
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Run this command from .venv-yolo11; Ultralytics is intentionally not required by "
            "the collection environment."
        ) from exc

    thresholds, calibration_record = _calibration_thresholds(calibration_path)
    if calibration_binding is not None:
        calibration_record["binding"] = calibration_binding
    model = YOLO(str(model_path))
    class_names = _class_names(model)
    if class_names != teacher_binding["class_map"]:
        raise ContractError(
            f"Loaded teacher class map differs from teacher manifest: "
            f"loaded={class_names}, manifest={teacher_binding['class_map']}"
        )
    allowed_threshold_keys = {"__global__", *class_names.values(), *(str(key) for key in class_names)}
    unknown_threshold_keys = sorted(set(thresholds) - allowed_threshold_keys)
    if unknown_threshold_keys:
        raise ContractError(
            f"Calibration contains classes absent from teacher manifest: {unknown_threshold_keys}"
        )
    selected_classes = set(args.classes) if args.classes else None
    if selected_classes and not selected_classes.issubset(class_names):
        unknown = sorted(selected_classes - set(class_names))
        raise ValueError(f"Requested class IDs are not in teacher model: {unknown}")
    run_id = args.run_id or "autolabel_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Autolabel run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_record = {
        "path": str(model_path),
        "sha256": teacher_binding["checkpoint_sha256"],
        "bytes": model_path.stat().st_size,
        "mib": model_path.stat().st_size / 1024**2,
        "classes": class_names,
        "teacher_manifest_sha256": teacher_binding["manifest_sha256"],
        "training_annotation_state": teacher_binding["training_annotation_state"],
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "annotation_state": "PENDING_HUMAN_REVIEW",
        "start_utc": utc_now_precise(),
        "source": str(source),
        "source_images": len(images),
        "source_binding": source_binding,
        "ontology": ontology.record(),
        "teacher": model_record,
        "calibration": calibration_record,
        "manual_high_confidence_override": args.high_confidence,
        "protocol": {
            "imgsz": args.imgsz,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "prediction_floor": args.prediction_floor,
            "tile_nms_iou": args.tile_nms_iou,
            "merge_nms_iou": args.merge_nms_iou,
            "minimum_box_pixels": args.minimum_box_pixels,
            "selected_classes": sorted(selected_classes) if selected_classes else "all",
            "human_review_required": True,
            "automatic_promotion_to_training": False,
        },
        "environment": collect_system_environment() | collect_torch_environment(),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_pip_freeze(run_dir / "pip-freeze.txt")

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    gpu_sampler = GpuSampler(run_dir / "gpu_samples.csv", args.gpu_sample_seconds)
    with tee_console(run_dir / "terminal.log"):
        print("\nMCU/SMD REVIEW-ONLY AUTO-LABELING")
        print("=" * 88)
        print_section("TEACHER MODEL", model_record)
        print_section("CALIBRATION", calibration_record)
        print_section("PENDING LABEL PROTOCOL", manifest["protocol"])
        if len(class_names) >= 20:
            print(
                "WARNING: teacher has many generic classes. A COCO-pretrained model is not a reliable "
                "MCU/SMD labeler; train a domain seed model first."
            )
        if not thresholds and args.high_confidence is None:
            print(
                "NOTICE: no gold-validation calibration was provided. Every proposal is marked "
                "review_required and no high-confidence candidate is claimed."
            )
        gpu_sampler.start()
        try:
            for index, image_path in enumerate(images, start=1):
                relative = _relative_image_path(image_path, source)
                with Image.open(image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                predictions, tile_count = _predict_image(
                    model,
                    image,
                    args,
                    class_names,
                    selected_classes,
                    thresholds,
                )
                label_path = run_dir / "labels_pending" / relative.with_suffix(".txt")
                preview_path = run_dir / "previews" / relative.with_suffix(".jpg")
                _write_yolo_labels(label_path, predictions, image.width, image.height)
                _draw_preview(image, predictions, preview_path)
                high_count = sum(
                    prediction["review_status"] == "high_confidence_candidate"
                    for prediction in predictions
                )
                review_count = len(predictions) - high_count
                image_status = (
                    "empty_prediction_review_required"
                    if not predictions
                    else "proposal_review_required"
                    if review_count
                    else "high_confidence_candidates_still_require_review"
                )
                confidences = [float(prediction["confidence"]) for prediction in predictions]
                summary_rows.append(
                    {
                        "image": relative.as_posix(),
                        "source_sha256": sha256_file(image_path),
                        "width": image.width,
                        "height": image.height,
                        "tiles": tile_count,
                        "proposals": len(predictions),
                        "high_confidence_candidates": high_count,
                        "review_required_proposals": review_count,
                        "minimum_confidence": min(confidences) if confidences else "",
                        "maximum_confidence": max(confidences) if confidences else "",
                        "image_status": image_status,
                        "pending_label": str(label_path.resolve()),
                        "preview": str(preview_path.resolve()),
                    }
                )
                for proposal_index, prediction in enumerate(predictions, start=1):
                    x1, y1, x2, y2 = prediction["xyxy"]
                    prediction_rows.append(
                        {
                            "image": relative.as_posix(),
                            "proposal_index": proposal_index,
                            "class_id": prediction["class_id"],
                            "class_name": prediction["class_name"],
                            "confidence": prediction["confidence"],
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width_px": prediction["width_px"],
                            "height_px": prediction["height_px"],
                            "high_confidence_threshold": prediction["high_confidence_threshold"],
                            "review_status": prediction["review_status"],
                        }
                    )
                if index == 1 or index % args.print_every == 0 or index == len(images):
                    print(
                        f"[AUTO-LABEL {index:05d}/{len(images):05d}] {relative.as_posix()} | "
                        f"tiles={tile_count} proposals={len(predictions)} high={high_count} "
                        f"review={review_count} status={image_status}"
                    )
        finally:
            gpu_summary = gpu_sampler.stop()
            write_json(run_dir / "gpu_summary.json", gpu_summary)

        prediction_fields = [
            "image",
            "proposal_index",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "width_px",
            "height_px",
            "high_confidence_threshold",
            "review_status",
        ]
        summary_fields = [
            "image",
            "source_sha256",
            "width",
            "height",
            "tiles",
            "proposals",
            "high_confidence_candidates",
            "review_required_proposals",
            "minimum_confidence",
            "maximum_confidence",
            "image_status",
            "pending_label",
            "preview",
        ]
        priority = {
            "empty_prediction_review_required": 0,
            "proposal_review_required": 1,
            "high_confidence_candidates_still_require_review": 2,
        }
        summary_rows.sort(
            key=lambda row: (
                priority[row["image_status"]],
                float(row["minimum_confidence"]) if row["minimum_confidence"] != "" else -1.0,
            )
        )
        _write_csv(run_dir / "predictions.csv", prediction_rows, prediction_fields)
        _write_csv(run_dir / "review_queue.csv", summary_rows, summary_fields)
        (run_dir / "classes.txt").write_text(
            "\n".join(class_names[index] for index in sorted(class_names)) + "\n",
            encoding="utf-8",
        )
        _write_review_instructions(run_dir / "README_REVIEW.md")

        pending_coco, pending_image_bindings_sha = _pending_coco_document(
            source_binding=source_binding,
            class_names=class_names,
            prediction_rows=prediction_rows,
        )
        pending_reference_path = run_dir / "pending_reference.coco.json"
        write_json(pending_reference_path, pending_coco)

        total_high = sum(row["high_confidence_candidates"] for row in summary_rows)
        empty_images = sum(
            row["image_status"] == "empty_prediction_review_required" for row in summary_rows
        )
        manifest.update(
            {
                "status": "complete",
                "end_utc": utc_now_precise(),
                "results": {
                    "images": len(summary_rows),
                    "proposals": len(prediction_rows),
                    "high_confidence_candidates": total_high,
                    "other_review_required_proposals": len(prediction_rows) - total_high,
                    "empty_prediction_images": empty_images,
                    "all_labels_pending_human_review": True,
                },
                "gpu_summary": gpu_summary,
                "pending_reference": {
                    "path": "pending_reference.coco.json",
                    "sha256": sha256_file(pending_reference_path),
                    "image_bindings_sha256": pending_image_bindings_sha,
                    "class_map_sha256": canonical_sha256(
                        {str(key): value for key, value in sorted(class_names.items())}
                    ),
                },
            }
        )
        write_json(run_dir / "run_manifest.json", manifest)
        print("\nAUTO-LABEL SUMMARY — NOT GROUND TRUTH")
        print("=" * 88)
        print(f"images                       : {len(summary_rows)}")
        print(f"proposals                    : {len(prediction_rows)}")
        print(f"high-confidence candidates   : {total_high} (still requires review)")
        print(f"other proposals              : {len(prediction_rows) - total_high}")
        print(f"empty-prediction images      : {empty_images} (inspect for missed chips)")
        print(f"pending labels / previews    : {run_dir.resolve()}")
        print("PROMOTION TO TRAINING         : BLOCKED UNTIL HUMAN APPROVAL")


if __name__ == "__main__":
    main()
