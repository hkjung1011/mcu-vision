from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image

from .common import IMAGE_SUFFIXES, load_yaml, sha256_file


SCHEMA_VERSION = "mcu.canonical-detection-dataset.v1"
EVIDENCE_SCHEMA_VERSION = "mcu.dataset-equivalence-evidence.v1"
REQUIRED_EVIDENCE_FIELDS = (
    "canonical_dataset_manifest_sha256",
    "class_map_sha256",
    "train_image_list_sha256",
    "val_image_list_sha256",
    "canonical_train_records_sha256",
    "canonical_val_records_sha256",
)
ARTIFACT_FILENAMES = {
    "class_map": "class_map.json",
    "train_image_list": "train_image_list.json",
    "val_image_list": "val_image_list.json",
    "canonical_train_records": "canonical_train_records.jsonl",
    "canonical_val_records": "canonical_val_records.jsonl",
    "canonical_dataset_manifest": "canonical_dataset_manifest.json",
    "dataset_evidence": "dataset_evidence.json",
    "equivalence_report": "dataset_equivalence_report.json",
}


class DatasetEvidenceError(ValueError):
    """Raised when an input cannot be represented without ambiguity."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _jsonl_bytes(values: Iterable[dict[str, Any]]) -> bytes:
    return "".join(_canonical_json(value) + "\n" for value in values).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_envelopes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=_canonical_json)
    return [
        {
            "record_sha256": _sha256_bytes(_canonical_json(record).encode("utf-8")),
            "record": record,
        }
        for record in ordered
    ]


def _normalized_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetEvidenceError(f"{field} must be a non-empty string")
    return unicodedata.normalize("NFC", value)


def _image_key(value: Any) -> str:
    normalized = _normalized_text(value, field="image file_name").replace("\\", "/")
    key = PurePosixPath(normalized).name
    if key in {"", ".", ".."}:
        raise DatasetEvidenceError(f"Invalid image file_name: {value!r}")
    return key


def _decimal(value: Any, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DatasetEvidenceError(f"{field} is not a decimal number: {value!r}") from exc
    if not parsed.is_finite():
        raise DatasetEvidenceError(f"{field} must be finite: {value!r}")
    return parsed


def _quantized(value: Decimal, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    result = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    if result == 0:
        result = abs(result)
    return f"{result:.{decimals}f}"


def _normalized_bbox(
    values: Iterable[Any],
    *,
    decimals: int,
    field: str,
) -> list[Decimal]:
    parsed = [_decimal(value, field=field) for value in values]
    if len(parsed) != 4:
        raise DatasetEvidenceError(f"{field} must contain exactly four coordinates")
    cx, cy, width, height = parsed
    if width <= 0 or height <= 0:
        raise DatasetEvidenceError(f"{field} width and height must be positive")
    if not all(Decimal("0") <= value <= Decimal("1") for value in parsed):
        raise DatasetEvidenceError(f"{field} normalized coordinates must be in [0, 1]")
    half = Decimal("0.5")
    # YOLO text is commonly serialized to eight decimals. This tolerance only
    # admits serialization noise; canonical pixel precision is configured separately.
    tolerance = Decimal("0.0000001")
    if cx - width * half < -tolerance or cx + width * half > Decimal("1") + tolerance:
        raise DatasetEvidenceError(f"{field} extends beyond the image width")
    if cy - height * half < -tolerance or cy + height * half > Decimal("1") + tolerance:
        raise DatasetEvidenceError(f"{field} extends beyond the image height")
    return parsed


def _canonical_bbox_from_yolo(
    values: Iterable[Any],
    *,
    image_width: int,
    image_height: int,
    decimals: int,
    field: str,
) -> list[str]:
    cx, cy, width, height = _normalized_bbox(values, decimals=decimals, field=field)
    pixel_width = Decimal(image_width)
    pixel_height = Decimal(image_height)
    xywh = (
        (cx - width / Decimal(2)) * pixel_width,
        (cy - height / Decimal(2)) * pixel_height,
        width * pixel_width,
        height * pixel_height,
    )
    return [_quantized(value, decimals) for value in xywh]


def _canonical_bbox_from_coco(
    values: Iterable[Any],
    *,
    image_width: int,
    image_height: int,
    decimals: int,
    field: str,
) -> list[str]:
    parsed = [_decimal(value, field=field) for value in values]
    if len(parsed) != 4:
        raise DatasetEvidenceError(f"{field} must contain exactly four coordinates")
    x, y, width, height = parsed
    if width <= 0 or height <= 0:
        raise DatasetEvidenceError(f"{field} width and height must be positive")
    tolerance = Decimal(1).scaleb(-decimals)
    if x < -tolerance or y < -tolerance:
        raise DatasetEvidenceError(f"{field} begins outside the image")
    if x + width > Decimal(image_width) + tolerance:
        raise DatasetEvidenceError(f"{field} extends beyond the image width")
    if y + height > Decimal(image_height) + tolerance:
        raise DatasetEvidenceError(f"{field} extends beyond the image height")
    return [_quantized(value, decimals) for value in parsed]


def _parse_yolo_class_map(document: dict[str, Any]) -> list[dict[str, Any]]:
    names = document.get("names")
    indexed: dict[int, str] = {}
    if isinstance(names, list):
        indexed = {
            index: _normalized_text(name, field=f"YOLO class name {index}")
            for index, name in enumerate(names)
        }
    elif isinstance(names, dict):
        for raw_index, name in names.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise DatasetEvidenceError(f"YOLO class index is not an integer: {raw_index!r}") from exc
            if index in indexed:
                raise DatasetEvidenceError(f"Duplicate YOLO class index: {index}")
            indexed[index] = _normalized_text(name, field=f"YOLO class name {index}")
    else:
        raise DatasetEvidenceError("dataset.yaml names must be a list or mapping")
    expected = list(range(len(indexed)))
    if sorted(indexed) != expected:
        raise DatasetEvidenceError(
            f"YOLO class indices must be contiguous from zero; found {sorted(indexed)}"
        )
    names_in_order = [indexed[index] for index in expected]
    if len(set(names_in_order)) != len(names_in_order):
        raise DatasetEvidenceError("YOLO class names must be unique")
    return [{"index": index, "name": indexed[index]} for index in expected]


def _dataset_root(dataset_yaml: Path, document: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    raw = document.get("path", ".")
    if not isinstance(raw, str):
        raise DatasetEvidenceError("dataset.yaml path must be a string")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = dataset_yaml.parent / candidate
    return candidate.resolve()


def _resolve_listed_image(raw: str, *, list_path: Path, dataset_root: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    candidates = [list_path.parent / candidate, dataset_root / candidate]
    for value in candidates:
        if value.is_file():
            return value.resolve()
    return candidates[0].resolve()


def _expand_yolo_reference(reference: Any, *, dataset_root: Path) -> list[Path]:
    if isinstance(reference, list):
        images: list[Path] = []
        for item in reference:
            images.extend(_expand_yolo_reference(item, dataset_root=dataset_root))
        return images
    if not isinstance(reference, str) or not reference:
        raise DatasetEvidenceError("YOLO split reference must be a string or list of strings")
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = dataset_root / path
    path = path.resolve()
    if path.is_dir():
        return sorted(
            (item.resolve() for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda item: item.as_posix(),
        )
    if path.is_file() and path.suffix.lower() == ".txt":
        images = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            value = line.strip()
            if not value:
                continue
            image = _resolve_listed_image(value, list_path=path, dataset_root=dataset_root)
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                raise DatasetEvidenceError(
                    f"Unsupported image suffix in {path.name}:{line_number}: {image.suffix}"
                )
            images.append(image)
        return images
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    raise DatasetEvidenceError(f"YOLO split path does not exist or is unsupported: {path}")


def _yolo_label_path(image_path: Path, dataset_root: Path) -> Path:
    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        relative = Path(*image_path.parts)
    parts = list(relative.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indices:
        raise DatasetEvidenceError(
            f"Cannot derive YOLO label path because 'images' is absent: {image_path}"
        )
    parts[image_indices[-1]] = "labels"
    if image_path.is_relative_to(dataset_root):
        return (dataset_root / Path(*parts)).with_suffix(".txt")
    return Path(*parts).with_suffix(".txt")


def _image_properties(path: Path) -> tuple[int, int, str]:
    if not path.is_file():
        raise DatasetEvidenceError(f"Image file is missing: {path}")
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:
        raise DatasetEvidenceError(f"Image cannot be decoded: {path}") from exc
    if width <= 0 or height <= 0:
        raise DatasetEvidenceError(f"Image dimensions must be positive: {path}")
    return width, height, sha256_file(path)


def _canonicalize_yolo_split(
    images: list[Path],
    *,
    dataset_root: Path,
    class_map: list[dict[str, Any]],
    decimals: int,
) -> dict[str, Any]:
    image_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for image_path in images:
        key = _image_key(image_path.name)
        if key in seen_keys:
            raise DatasetEvidenceError(
                f"Duplicate basename in one YOLO split is ambiguous: {key}"
            )
        seen_keys.add(key)
        width, height, image_sha256 = _image_properties(image_path)
        image_rows.append(
            {
                "image_key": key,
                "image_sha256": image_sha256,
                "width": width,
                "height": height,
            }
        )
        label_path = _yolo_label_path(image_path, dataset_root)
        if not label_path.exists():
            continue
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise DatasetEvidenceError(
                    f"Only YOLO detection labels with 5 fields are supported: "
                    f"{label_path.name}:{line_number}"
                )
            try:
                class_index = int(fields[0])
            except ValueError as exc:
                raise DatasetEvidenceError(
                    f"YOLO class index is not an integer: {label_path.name}:{line_number}"
                ) from exc
            if not 0 <= class_index < len(class_map):
                raise DatasetEvidenceError(
                    f"YOLO class index {class_index} is outside the class map: "
                    f"{label_path.name}:{line_number}"
                )
            records.append(
                {
                    "image_key": key,
                    "image_sha256": image_sha256,
                    "class_index": class_index,
                    "class_name": class_map[class_index]["name"],
                    "bbox_xywh_pixels": _canonical_bbox_from_yolo(
                        fields[1:],
                        image_width=width,
                        image_height=height,
                        decimals=decimals,
                        field=f"{label_path.name}:{line_number}",
                    ),
                }
            )
    return {
        "images": sorted(image_rows, key=_canonical_json),
        "records": sorted(records, key=_canonical_json),
    }


def canonicalize_yolo_dataset(
    dataset_yaml: Path,
    *,
    dataset_root: Path | None = None,
    bbox_decimals: int = 3,
) -> dict[str, Any]:
    dataset_yaml = dataset_yaml.resolve()
    document = load_yaml(dataset_yaml)
    root = _dataset_root(dataset_yaml, document, dataset_root)
    class_map = _parse_yolo_class_map(document)
    splits = {}
    for split in ("train", "val"):
        if split not in document:
            raise DatasetEvidenceError(f"dataset.yaml is missing the {split!r} split")
        images = _expand_yolo_reference(document[split], dataset_root=root)
        splits[split] = _canonicalize_yolo_split(
            images,
            dataset_root=root,
            class_map=class_map,
            decimals=bbox_decimals,
        )
    return {"class_map": class_map, "splits": splits}


def _coco_class_lookup(
    document: dict[str, Any], class_map: list[dict[str, Any]]
) -> dict[Any, dict[str, Any]]:
    canonical_by_name = {str(row["name"]): row for row in class_map}
    categories = document.get("categories")
    if not isinstance(categories, list):
        raise DatasetEvidenceError("COCO categories must be a list")
    lookup: dict[Any, dict[str, Any]] = {}
    seen_names: set[str] = set()
    for category in categories:
        if not isinstance(category, dict) or "id" not in category:
            raise DatasetEvidenceError("Every COCO category must contain id and name")
        category_id = category["id"]
        if category_id in lookup:
            raise DatasetEvidenceError(f"Duplicate COCO category id: {category_id!r}")
        name = _normalized_text(category.get("name"), field="COCO category name")
        if name in seen_names:
            raise DatasetEvidenceError(f"Duplicate COCO category name: {name}")
        seen_names.add(name)
        if name not in canonical_by_name:
            raise DatasetEvidenceError(f"COCO category is absent from YOLO class names: {name}")
        lookup[category_id] = canonical_by_name[name]
    missing = sorted(set(canonical_by_name) - seen_names)
    if missing:
        raise DatasetEvidenceError(f"COCO categories are missing YOLO classes: {missing}")
    return lookup


def _inferred_coco_image_root(annotation_path: Path) -> Path:
    stem = annotation_path.stem
    prefix = "instances_"
    split_directory = stem[len(prefix) :] if stem.startswith(prefix) else stem
    return annotation_path.parent.parent / split_directory


def _coco_image_path(image_root: Path, file_name: str) -> Path:
    relative = PurePosixPath(file_name.replace("\\", "/"))
    path = image_root.joinpath(*relative.parts)
    if path.is_file():
        return path
    basename_path = image_root / relative.name
    if basename_path.is_file():
        return basename_path
    return path


def _canonicalize_coco_split(
    annotation_path: Path,
    *,
    image_root: Path | None,
    class_map: list[dict[str, Any]],
    decimals: int,
) -> dict[str, Any]:
    annotation_path = annotation_path.resolve()
    try:
        document = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetEvidenceError(f"Invalid COCO JSON: {annotation_path}") from exc
    if not isinstance(document, dict):
        raise DatasetEvidenceError(f"COCO root must be a JSON object: {annotation_path}")
    category_lookup = _coco_class_lookup(document, class_map)
    resolved_image_root = (
        image_root.resolve() if image_root is not None else _inferred_coco_image_root(annotation_path)
    )
    images = document.get("images")
    annotations = document.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise DatasetEvidenceError("COCO images and annotations must be lists")
    image_rows: list[dict[str, Any]] = []
    image_lookup: dict[Any, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    for image in images:
        if not isinstance(image, dict) or "id" not in image:
            raise DatasetEvidenceError("Every COCO image must contain id and file_name")
        image_id = image["id"]
        if image_id in image_lookup:
            raise DatasetEvidenceError(f"Duplicate COCO image id: {image_id!r}")
        file_name = _normalized_text(image.get("file_name"), field="COCO image file_name")
        key = _image_key(file_name)
        if key in seen_keys:
            raise DatasetEvidenceError(
                f"Duplicate basename in one COCO split is ambiguous: {key}"
            )
        seen_keys.add(key)
        image_path = _coco_image_path(resolved_image_root, file_name)
        width, height, image_sha256 = _image_properties(image_path)
        try:
            declared_width = int(image["width"])
            declared_height = int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetEvidenceError(f"Invalid COCO dimensions for image {key}") from exc
        if (declared_width, declared_height) != (width, height):
            raise DatasetEvidenceError(
                f"COCO dimensions differ from decoded image {key}: "
                f"JSON={(declared_width, declared_height)}, decoded={(width, height)}"
            )
        row = {
            "image_key": key,
            "image_sha256": image_sha256,
            "width": width,
            "height": height,
        }
        image_rows.append(row)
        image_lookup[image_id] = row
    records: list[dict[str, Any]] = []
    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict):
            raise DatasetEvidenceError(f"COCO annotation {index} must be an object")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id not in image_lookup:
            raise DatasetEvidenceError(f"COCO annotation references unknown image_id: {image_id!r}")
        if category_id not in category_lookup:
            raise DatasetEvidenceError(
                f"COCO annotation references unknown category_id: {category_id!r}"
            )
        if annotation.get("iscrowd", 0) not in (0, False):
            raise DatasetEvidenceError("COCO crowd annotations cannot be represented by YOLO txt")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise DatasetEvidenceError(f"COCO annotation {index} bbox must have four values")
        x, y, box_width, box_height = [
            _decimal(value, field=f"COCO annotation {index} bbox") for value in bbox
        ]
        image_row = image_lookup[image_id]
        image_width = Decimal(int(image_row["width"]))
        image_height = Decimal(int(image_row["height"]))
        canonical_class = category_lookup[category_id]
        records.append(
            {
                "image_key": image_row["image_key"],
                "image_sha256": image_row["image_sha256"],
                "class_index": canonical_class["index"],
                "class_name": canonical_class["name"],
                "bbox_xywh_pixels": _canonical_bbox_from_coco(
                    (x, y, box_width, box_height),
                    image_width=int(image_width),
                    image_height=int(image_height),
                    decimals=decimals,
                    field=f"COCO annotation {index}",
                ),
            }
        )
    return {
        "images": sorted(image_rows, key=_canonical_json),
        "records": sorted(records, key=_canonical_json),
    }


def canonicalize_coco_dataset(
    train_annotations: Path,
    val_annotations: Path,
    *,
    class_map: list[dict[str, Any]],
    train_image_root: Path | None = None,
    val_image_root: Path | None = None,
    bbox_decimals: int = 3,
) -> dict[str, Any]:
    return {
        "class_map": class_map,
        "splits": {
            "train": _canonicalize_coco_split(
                train_annotations,
                image_root=train_image_root,
                class_map=class_map,
                decimals=bbox_decimals,
            ),
            "val": _canonicalize_coco_split(
                val_annotations,
                image_root=val_image_root,
                class_map=class_map,
                decimals=bbox_decimals,
            ),
        },
    }


def _counter_differences(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    left_counter = Counter(_canonical_json(value) for value in left)
    right_counter = Counter(_canonical_json(value) for value in right)
    left_only = left_counter - right_counter
    right_only = right_counter - left_counter

    def examples(values: Counter[str]) -> list[dict[str, Any]]:
        rows = []
        for encoded, count in sorted(values.items())[:limit]:
            rows.append({"count": count, "value": json.loads(encoded)})
        return rows

    return {
        "yolo_only_count": sum(left_only.values()),
        "coco_only_count": sum(right_only.values()),
        "yolo_only_examples": examples(left_only),
        "coco_only_examples": examples(right_only),
    }


def _side_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        split: {
            "images": len(dataset["splits"][split]["images"]),
            "annotations": len(dataset["splits"][split]["records"]),
            "image_payload_sha256": _sha256_bytes(
                _json_bytes(dataset["splits"][split]["images"])
            ),
            "record_payload_sha256": _sha256_bytes(
                _jsonl_bytes(_record_envelopes(dataset["splits"][split]["records"]))
            ),
        }
        for split in ("train", "val")
    }


def _artifact_payloads(
    dataset: dict[str, Any], *, bbox_decimals: int
) -> tuple[dict[str, bytes], dict[str, str], dict[str, Any]]:
    payloads: dict[str, bytes] = {
        "class_map": _json_bytes(dataset["class_map"]),
    }
    split_manifest: dict[str, Any] = {}
    evidence: dict[str, str] = {}
    for split in ("train", "val"):
        image_key = f"{split}_image_list"
        record_key = f"canonical_{split}_records"
        payloads[image_key] = _json_bytes(dataset["splits"][split]["images"])
        envelopes = _record_envelopes(dataset["splits"][split]["records"])
        payloads[record_key] = _jsonl_bytes(envelopes)
        image_sha = _sha256_bytes(payloads[image_key])
        record_sha = _sha256_bytes(payloads[record_key])
        evidence[f"{split}_image_list_sha256"] = image_sha
        evidence[f"canonical_{split}_records_sha256"] = record_sha
        split_manifest[split] = {
            "image_count": len(dataset["splits"][split]["images"]),
            "annotation_count": len(dataset["splits"][split]["records"]),
            "image_list_sha256": image_sha,
            "canonical_records_sha256": record_sha,
        }
    evidence["class_map_sha256"] = _sha256_bytes(payloads["class_map"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "canonicalization": {
            "image_identity": "Unicode-NFC basename + encoded-file SHA-256 + decoded dimensions",
            "record_order": "ascending canonical UTF-8 JSON; duplicates retained",
            "class_identity": "YOLO zero-based index and Unicode-NFC class name; COCO id mapped by name",
            "bbox_format": "x,y,width,height in decoded-image pixels",
            "bbox_decimal_places": bbox_decimals,
            "bbox_rounding": "decimal ROUND_HALF_EVEN",
            "conversion_note": (
                "YOLO cxcywh is converted to pixels before fixed-decimal normalization; "
                "this removes expected normalized-text round-trip noise"
            ),
            "ignored_coco_fields": ["annotation id", "area", "segmentation", "supercategory"],
        },
        "class_count": len(dataset["class_map"]),
        "class_map_sha256": evidence["class_map_sha256"],
        "splits": split_manifest,
    }
    payloads["canonical_dataset_manifest"] = _json_bytes(manifest)
    evidence["canonical_dataset_manifest_sha256"] = _sha256_bytes(
        payloads["canonical_dataset_manifest"]
    )
    ordered_evidence = {field: evidence[field] for field in REQUIRED_EVIDENCE_FIELDS}
    return payloads, ordered_evidence, manifest


def build_dataset_equivalence_evidence(
    *,
    yolo_dataset_yaml: Path,
    coco_train_annotations: Path,
    coco_val_annotations: Path,
    output_dir: Path | None = None,
    yolo_dataset_root: Path | None = None,
    coco_train_image_root: Path | None = None,
    coco_val_image_root: Path | None = None,
    bbox_decimals: int = 3,
    max_difference_examples: int = 20,
) -> dict[str, Any]:
    """Verify YOLO txt and COCO JSON as the same canonical detection dataset.

    Canonical evidence contains no absolute path, source annotation id, or source list order.
    Images are keyed by basename and encoded-file SHA-256; duplicate basenames are rejected.
    """
    if not 0 <= bbox_decimals <= 12:
        raise DatasetEvidenceError("bbox_decimals must be between 0 and 12")
    if max_difference_examples < 0:
        raise DatasetEvidenceError("max_difference_examples cannot be negative")
    yolo = canonicalize_yolo_dataset(
        yolo_dataset_yaml,
        dataset_root=yolo_dataset_root,
        bbox_decimals=bbox_decimals,
    )
    coco = canonicalize_coco_dataset(
        coco_train_annotations,
        coco_val_annotations,
        class_map=yolo["class_map"],
        train_image_root=coco_train_image_root,
        val_image_root=coco_val_image_root,
        bbox_decimals=bbox_decimals,
    )
    split_differences: dict[str, Any] = {}
    equivalent = yolo["class_map"] == coco["class_map"]
    for split in ("train", "val"):
        image_difference = _counter_differences(
            yolo["splits"][split]["images"],
            coco["splits"][split]["images"],
            limit=max_difference_examples,
        )
        record_difference = _counter_differences(
            yolo["splits"][split]["records"],
            coco["splits"][split]["records"],
            limit=max_difference_examples,
        )
        split_equal = not any(
            (
                image_difference["yolo_only_count"],
                image_difference["coco_only_count"],
                record_difference["yolo_only_count"],
                record_difference["coco_only_count"],
            )
        )
        equivalent = equivalent and split_equal
        split_differences[split] = {
            "equivalent": split_equal,
            "images": image_difference,
            "records": record_difference,
        }

    report: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "PASS" if equivalent else "FAIL",
        "equivalent": equivalent,
        "bbox_decimal_places": bbox_decimals,
        "match_key_policy": "Unicode-NFC basename; duplicate basenames are rejected",
        "sides": {"yolo": _side_summary(yolo), "coco": _side_summary(coco)},
        "differences": split_differences,
        "evidence": {},
    }
    payloads: dict[str, bytes] = {}
    if equivalent:
        payloads, evidence, manifest = _artifact_payloads(
            yolo, bbox_decimals=bbox_decimals
        )
        report["evidence"] = evidence
        report["canonical_manifest"] = manifest

    if output_dir is not None:
        output_dir = output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory must be empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if equivalent:
            for key, content in payloads.items():
                (output_dir / ARTIFACT_FILENAMES[key]).write_bytes(content)
            evidence_document = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "status": "PASS",
                **report["evidence"],
                "artifacts": {
                    field: ARTIFACT_FILENAMES[
                        field.removesuffix("_sha256")
                        if field != "canonical_dataset_manifest_sha256"
                        else "canonical_dataset_manifest"
                    ]
                    for field in REQUIRED_EVIDENCE_FIELDS
                },
                "counts": {
                    split: {
                        "images": len(yolo["splits"][split]["images"]),
                        "annotations": len(yolo["splits"][split]["records"]),
                    }
                    for split in ("train", "val")
                },
            }
            (output_dir / ARTIFACT_FILENAMES["dataset_evidence"]).write_bytes(
                _json_bytes(evidence_document)
            )
        (output_dir / ARTIFACT_FILENAMES["equivalence_report"]).write_bytes(
            _json_bytes(report)
        )
    return report


def load_dataset_evidence(path: Path, *, verify_artifacts: bool = True) -> dict[str, str]:
    """Load and, by default, verify all six release-gate artifacts and hashes."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetEvidenceError(f"Invalid dataset evidence JSON: {path}") from exc
    if not isinstance(document, dict) or document.get("status") != "PASS":
        raise DatasetEvidenceError(f"Dataset evidence is not PASS: {path}")
    evidence: dict[str, str] = {}
    for field in REQUIRED_EVIDENCE_FIELDS:
        value = document.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise DatasetEvidenceError(f"Dataset evidence is missing a SHA-256 field: {field}")
        try:
            int(value, 16)
        except ValueError as exc:
            raise DatasetEvidenceError(f"Dataset evidence field is not hexadecimal: {field}") from exc
        evidence[field] = value
    if verify_artifacts:
        artifacts = document.get("artifacts")
        if not isinstance(artifacts, dict):
            raise DatasetEvidenceError("Dataset evidence does not list its canonical artifacts")
        evidence_root = path.resolve().parent
        for field, expected_sha256 in evidence.items():
            relative_value = artifacts.get(field)
            if not isinstance(relative_value, str) or not relative_value:
                raise DatasetEvidenceError(f"Dataset evidence is missing its artifact path: {field}")
            relative_path = Path(relative_value)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise DatasetEvidenceError(f"Dataset artifact path must be relative: {field}")
            artifact_path = (evidence_root / relative_path).resolve()
            if not artifact_path.is_file():
                raise DatasetEvidenceError(f"Dataset evidence artifact is missing: {artifact_path}")
            if sha256_file(artifact_path) != expected_sha256:
                raise DatasetEvidenceError(f"Dataset evidence artifact hash differs: {field}")
    return evidence


def verify_dataset_against_evidence(
    *,
    evidence_path: Path,
    yolo_dataset_yaml: Path,
    coco_train_annotations: Path,
    coco_val_annotations: Path,
    yolo_dataset_root: Path | None = None,
    coco_train_image_root: Path | None = None,
    coco_val_image_root: Path | None = None,
    bbox_decimals: int = 3,
) -> dict[str, str]:
    """Fail unless the current YOLO/COCO files reproduce the declared evidence.

    ``load_dataset_evidence`` protects the tracked canonical artifacts from
    tampering. This stronger preflight additionally canonicalizes the live
    training inputs, so changing an image, YOLO label, COCO record, or class map
    after evidence generation cannot silently reach a training run.
    """
    declared = load_dataset_evidence(evidence_path.resolve())
    report = build_dataset_equivalence_evidence(
        yolo_dataset_yaml=yolo_dataset_yaml,
        yolo_dataset_root=yolo_dataset_root,
        coco_train_annotations=coco_train_annotations,
        coco_val_annotations=coco_val_annotations,
        coco_train_image_root=coco_train_image_root,
        coco_val_image_root=coco_val_image_root,
        bbox_decimals=bbox_decimals,
    )
    if not report["equivalent"]:
        raise DatasetEvidenceError(
            "The live YOLO and COCO datasets are not equivalent; regenerate neither "
            "evidence nor weights until the label mismatch is resolved"
        )
    actual = report["evidence"]
    mismatches = {
        field: {"declared": declared[field], "actual": actual.get(field)}
        for field in REQUIRED_EVIDENCE_FIELDS
        if actual.get(field) != declared[field]
    }
    if mismatches:
        raise DatasetEvidenceError(
            "The live dataset differs from the declared canonical evidence: "
            + _canonical_json(mismatches)
        )
    return declared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify path/order-independent YOLO txt and COCO JSON dataset equivalence"
    )
    parser.add_argument("--yolo-data", type=Path, required=True, help="YOLO dataset.yaml")
    parser.add_argument("--yolo-root", type=Path, help="Override dataset.yaml path")
    parser.add_argument("--coco-train", type=Path, required=True)
    parser.add_argument("--coco-val", type=Path, required=True)
    parser.add_argument("--coco-train-images", type=Path)
    parser.add_argument("--coco-val-images", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bbox-decimals",
        type=int,
        default=3,
        help="Canonical pixel-coordinate decimal places (default: 3)",
    )
    parser.add_argument("--max-difference-examples", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = build_dataset_equivalence_evidence(
            yolo_dataset_yaml=args.yolo_data,
            yolo_dataset_root=args.yolo_root,
            coco_train_annotations=args.coco_train,
            coco_val_annotations=args.coco_val,
            coco_train_image_root=args.coco_train_images,
            coco_val_image_root=args.coco_val_images,
            output_dir=args.output_dir,
            bbox_decimals=args.bbox_decimals,
            max_difference_examples=args.max_difference_examples,
        )
    except (DatasetEvidenceError, FileNotFoundError, FileExistsError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "equivalent": report["equivalent"],
                "evidence": report["evidence"],
                "report": str(
                    (args.output_dir / ARTIFACT_FILENAMES["equivalence_report"]).resolve()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
