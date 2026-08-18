from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import yaml

from .common import sha256_file, write_json
from .contracts import (
    ContractError,
    Ontology,
    canonical_sha256,
    load_json_object,
    load_ontology,
    require_sha256,
    safe_relative_path,
)


ROUNDTRIP_SCHEMA = "mcu.cvat-roundtrip-report.v1"


class RoundTripError(ValueError):
    """Raised when a CVAT export cannot be compared without ambiguity."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decimal(value: Any, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RoundTripError(f"{field} must be numeric") from exc
    if not parsed.is_finite():
        raise RoundTripError(f"{field} must be finite")
    return parsed


def _quantized(value: Decimal, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    result = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    if result == 0:
        result = abs(result)
    return f"{result:.{decimals}f}"


def _bbox_from_coco(
    value: Any, *, width: int, height: int, decimals: int, field: str
) -> list[str]:
    if not isinstance(value, list) or len(value) != 4:
        raise RoundTripError(f"{field} must be a four-value bbox")
    x, y, box_width, box_height = [_decimal(item, field=field) for item in value]
    tolerance = Decimal(1).scaleb(-decimals)
    if box_width <= 0 or box_height <= 0:
        raise RoundTripError(f"{field} width and height must be positive")
    if x < -tolerance or y < -tolerance:
        raise RoundTripError(f"{field} begins outside the image")
    if x + box_width > Decimal(width) + tolerance:
        raise RoundTripError(f"{field} extends beyond image width")
    if y + box_height > Decimal(height) + tolerance:
        raise RoundTripError(f"{field} extends beyond image height")
    return [_quantized(item, decimals) for item in (x, y, box_width, box_height)]


def _bbox_from_yolo(
    values: list[str], *, width: int, height: int, decimals: int, field: str
) -> list[str]:
    if len(values) != 4:
        raise RoundTripError(f"{field} must contain four normalized coordinates")
    center_x, center_y, box_width, box_height = [
        _decimal(value, field=field) for value in values
    ]
    if box_width <= 0 or box_height <= 0:
        raise RoundTripError(f"{field} width and height must be positive")
    if not all(Decimal(0) <= value <= Decimal(1) for value in (center_x, center_y, box_width, box_height)):
        raise RoundTripError(f"{field} normalized coordinates must be in [0, 1]")
    x = (center_x - box_width / Decimal(2)) * Decimal(width)
    y = (center_y - box_height / Decimal(2)) * Decimal(height)
    pixel_width = box_width * Decimal(width)
    pixel_height = box_height * Decimal(height)
    tolerance = Decimal("0.0000001") * max(Decimal(width), Decimal(height))
    if x < -tolerance or y < -tolerance:
        raise RoundTripError(f"{field} begins outside the image")
    if x + pixel_width > Decimal(width) + tolerance:
        raise RoundTripError(f"{field} extends beyond image width")
    if y + pixel_height > Decimal(height) + tolerance:
        raise RoundTripError(f"{field} extends beyond image height")
    return [
        _quantized(item, decimals)
        for item in (x, y, pixel_width, pixel_height)
    ]


def _image_key(file_name: Any) -> str:
    if not isinstance(file_name, str) or not file_name:
        raise RoundTripError("Image file_name must be a non-empty string")
    try:
        return safe_relative_path(file_name, field="COCO image file_name")
    except ContractError as exc:
        raise RoundTripError(str(exc)) from exc


def _attributes(annotation: Mapping[str, Any]) -> dict[str, bool]:
    raw = annotation.get("attributes")
    raw_mapping = raw if isinstance(raw, Mapping) else {}
    result: dict[str, bool] = {}
    for name in ("occluded", "truncated"):
        value = annotation.get(name, raw_mapping.get(name, False))
        if value not in (True, False, 0, 1, "0", "1", "true", "false", "True", "False"):
            raise RoundTripError(f"Annotation attribute {name} must be boolean-like")
        result[name] = value in (True, 1, "1", "true", "True")
    return result


def canonicalize_coco_document(
    document: dict[str, Any],
    *,
    ontology: Ontology,
    decimals: int,
    include_attributes: bool,
    require_image_bindings: bool = False,
) -> dict[str, Any]:
    images = document.get("images")
    annotations = document.get("annotations")
    categories = document.get("categories")
    if not all(isinstance(value, list) for value in (images, annotations, categories)):
        raise RoundTripError("COCO images, annotations, and categories must be lists")
    category_lookup: dict[Any, tuple[int, str]] = {}
    seen_names: set[str] = set()
    for category in categories:
        if not isinstance(category, Mapping) or "id" not in category:
            raise RoundTripError("Every COCO category must contain id and name")
        category_id = category["id"]
        if isinstance(category_id, bool) or not isinstance(category_id, int):
            raise RoundTripError("Every COCO category id must be an integer")
        name = category.get("name")
        if not isinstance(name, str) or name not in ontology.classes_by_id.values():
            raise RoundTripError(f"COCO category is absent from ontology: {name!r}")
        if name in seen_names or category_id in category_lookup:
            raise RoundTripError(f"Duplicate COCO category: {name!r}")
        seen_names.add(name)
        class_id = ontology.class_id(name)
        if "ontology_class_id" in category:
            declared_class_id = category["ontology_class_id"]
            if (
                isinstance(declared_class_id, bool)
                or not isinstance(declared_class_id, int)
                or declared_class_id != class_id
            ):
                raise RoundTripError(
                    f"COCO category ontology_class_id differs from frozen class map: {name!r}"
                )
        category_lookup[category_id] = (class_id, name)
    expected_names = set(ontology.classes_by_id.values())
    if seen_names != expected_names:
        raise RoundTripError(
            "COCO category map must exactly equal the frozen ontology: "
            f"missing={sorted(expected_names - seen_names)}, "
            f"unexpected={sorted(seen_names - expected_names)}"
        )
    image_lookup: dict[Any, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    image_rows: list[dict[str, Any]] = []
    image_bindings: list[dict[str, Any]] = []
    for image in images:
        if not isinstance(image, Mapping) or "id" not in image:
            raise RoundTripError("Every COCO image must contain id and file_name")
        image_id = image["id"]
        if isinstance(image_id, bool) or not isinstance(image_id, (int, str)):
            raise RoundTripError("Every COCO image id must be an integer or string")
        if image_id in image_lookup:
            raise RoundTripError(f"Duplicate COCO image id: {image_id!r}")
        key = _image_key(image.get("file_name"))
        if key in seen_keys:
            raise RoundTripError(f"Duplicate COCO image path: {key}")
        seen_keys.add(key)
        try:
            width = int(image["width"])
            height = int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RoundTripError(f"Invalid COCO dimensions: {key}") from exc
        if width <= 0 or height <= 0:
            raise RoundTripError(f"Invalid COCO dimensions: {key}")
        row = {"image_key": key, "width": width, "height": height}
        image_lookup[image_id] = row
        image_rows.append(row)
        if require_image_bindings:
            stable_id = image.get("mcu_image_id", image.get("mcu_asset_id"))
            if not isinstance(stable_id, str) or not stable_id:
                raise RoundTripError(
                    f"Reference COCO image {key} is missing mcu_image_id/mcu_asset_id"
                )
            image_bindings.append(
                {
                    "image_id": stable_id,
                    "path": key,
                    "sha256": require_sha256(
                        image.get("sha256"), field=f"reference image {key} sha256"
                    ),
                    "width": width,
                    "height": height,
                }
            )
    records: list[dict[str, Any]] = []
    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, Mapping):
            raise RoundTripError(f"COCO annotation {index} must be an object")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id not in image_lookup:
            raise RoundTripError(f"COCO annotation references unknown image: {image_id!r}")
        if category_id not in category_lookup:
            raise RoundTripError(f"COCO annotation references unknown category: {category_id!r}")
        if annotation.get("iscrowd", 0) not in (0, False):
            raise RoundTripError("Crowd annotations are outside bbox round-trip scope")
        image_row = image_lookup[image_id]
        class_id, class_name = category_lookup[category_id]
        record: dict[str, Any] = {
            "image_key": image_row["image_key"],
            "class_id": class_id,
            "class_name": class_name,
            "bbox_xywh_pixels": _bbox_from_coco(
                annotation.get("bbox"),
                width=image_row["width"],
                height=image_row["height"],
                decimals=decimals,
                field=f"COCO annotation {index}",
            ),
        }
        if include_attributes:
            record["attributes"] = _attributes(annotation)
        records.append(record)
    return {
        "images": sorted(image_rows, key=_canonical_json),
        "records": sorted(records, key=_canonical_json),
        "image_bindings": sorted(
            image_bindings, key=lambda row: (str(row["path"]), str(row["image_id"]))
        ),
        "class_map": {
            str(key): value for key, value in sorted(ontology.classes_by_id.items())
        },
    }


def load_coco_export_document(path: Path) -> tuple[dict[str, Any], str]:
    if path.suffix.casefold() != ".zip":
        return load_json_object(path, label="CVAT COCO round-trip export"), path.name
    if not zipfile.is_zipfile(path):
        raise RoundTripError(f"CVAT COCO export must be a ZIP or JSON: {path}")
    with zipfile.ZipFile(path) as archive:
        candidates: list[tuple[str, dict[str, Any]]] = []
        for info in archive.infolist():
            if info.is_dir() or not info.filename.casefold().endswith(".json"):
                continue
            try:
                value = json.loads(archive.read(info).decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and all(
                isinstance(value.get(key), list)
                for key in ("images", "annotations", "categories")
            ):
                candidates.append((info.filename, value))
    if len(candidates) != 1:
        raise RoundTripError(
            f"CVAT COCO ZIP must contain exactly one detection annotation JSON; found {len(candidates)}"
        )
    return candidates[0][1], candidates[0][0]


def _yolo_names(archive: zipfile.ZipFile) -> list[str]:
    yaml_candidates = [
        info
        for info in archive.infolist()
        if not info.is_dir()
        and PurePosixPath(info.filename).name.casefold() in {"data.yaml", "dataset.yaml"}
    ]
    if yaml_candidates:
        if len(yaml_candidates) != 1:
            raise RoundTripError("YOLO export contains multiple dataset YAML files")
        try:
            document = yaml.safe_load(archive.read(yaml_candidates[0]).decode("utf-8-sig"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise RoundTripError("Invalid YOLO dataset YAML") from exc
        names = document.get("names") if isinstance(document, dict) else None
        if isinstance(names, list):
            if any(not isinstance(name, str) or not name for name in names):
                raise RoundTripError("YOLO class names must be non-empty strings")
            return list(names)
        if isinstance(names, dict):
            indexed: dict[int, str] = {}
            for raw_key, raw_value in names.items():
                if isinstance(raw_key, bool) or not isinstance(raw_value, str) or not raw_value:
                    raise RoundTripError("YOLO class ids and names are invalid")
                try:
                    class_id = int(raw_key)
                except (TypeError, ValueError) as exc:
                    raise RoundTripError("YOLO class ids must be integers") from exc
                if class_id in indexed:
                    raise RoundTripError("YOLO class ids collide after integer normalization")
                indexed[class_id] = raw_value
            if sorted(indexed) != list(range(len(indexed))):
                raise RoundTripError("YOLO class ids must be contiguous from zero")
            return [indexed[index] for index in range(len(indexed))]
        raise RoundTripError("YOLO dataset YAML is missing names")
    name_candidates = [
        info
        for info in archive.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).name.casefold() == "obj.names"
    ]
    if len(name_candidates) != 1:
        raise RoundTripError("YOLO export must contain one data.yaml/dataset.yaml or obj.names")
    names = [
        line.strip()
        for line in archive.read(name_candidates[0]).decode("utf-8-sig").splitlines()
        if line.strip()
    ]
    if not names:
        raise RoundTripError("YOLO class name list is empty")
    return names


def _canonicalize_yolo_zip(
    path: Path, *, reference: dict[str, Any], ontology: Ontology, decimals: int
) -> dict[str, Any]:
    if not zipfile.is_zipfile(path):
        raise RoundTripError(f"CVAT YOLO round-trip export must be a ZIP: {path}")
    dimensions = {
        Path(row["image_key"]).stem: (row["image_key"], row["width"], row["height"])
        for row in reference["images"]
    }
    if len(dimensions) != len(reference["images"]):
        raise RoundTripError("Reference image stems are ambiguous for YOLO")
    with zipfile.ZipFile(path) as archive:
        names = _yolo_names(archive)
        if names != ontology.names:
            raise RoundTripError(
                f"YOLO class map must exactly equal the frozen ontology: "
                f"yolo={names}, ontology={ontology.names}"
            )
        labels: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            if info.is_dir() or not info.filename.casefold().endswith(".txt"):
                continue
            basename = PurePosixPath(info.filename).name.casefold()
            if basename in {"train.txt", "valid.txt", "val.txt", "test.txt", "obj.names"}:
                continue
            stem = PurePosixPath(info.filename).stem
            if stem in labels:
                raise RoundTripError(f"YOLO export contains duplicate label stem: {stem}")
            labels[stem] = info
        if set(labels) != set(dimensions):
            missing = sorted(set(dimensions) - set(labels))[:10]
            unexpected = sorted(set(labels) - set(dimensions))[:10]
            raise RoundTripError(
                f"YOLO label/image set differs: missing={missing}, unexpected={unexpected}"
            )
        records: list[dict[str, Any]] = []
        for stem, info in sorted(labels.items()):
            image_key, width, height = dimensions[stem]
            text = archive.read(info).decode("utf-8-sig")
            for line_number, line in enumerate(text.splitlines(), start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise RoundTripError(
                        f"YOLO label must contain five fields: {info.filename}:{line_number}"
                    )
                try:
                    class_id = int(fields[0])
                except ValueError as exc:
                    raise RoundTripError("YOLO class id must be an integer") from exc
                if not 0 <= class_id < len(names):
                    raise RoundTripError(f"YOLO class id is outside class map: {class_id}")
                records.append(
                    {
                        "image_key": image_key,
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "bbox_xywh_pixels": _bbox_from_yolo(
                            fields[1:],
                            width=width,
                            height=height,
                            decimals=decimals,
                            field=f"{info.filename}:{line_number}",
                        ),
                    }
                )
    return {"images": reference["images"], "records": sorted(records, key=_canonical_json)}


def _difference(reference: list[dict[str, Any]], roundtrip: list[dict[str, Any]]) -> dict[str, Any]:
    left = Counter(_canonical_json(row) for row in reference)
    right = Counter(_canonical_json(row) for row in roundtrip)
    missing = left - right
    unexpected = right - left

    def examples(counter: Counter[str]) -> list[dict[str, Any]]:
        return [
            {"count": count, "value": json.loads(encoded)}
            for encoded, count in sorted(counter.items())[:10]
        ]

    return {
        "missing_count": sum(missing.values()),
        "unexpected_count": sum(unexpected.values()),
        "missing_examples": examples(missing),
        "unexpected_examples": examples(unexpected),
    }


def verify_cvat_roundtrip(
    *,
    reference_coco: Path,
    roundtrip_path: Path,
    export_format: str,
    ontology_path: Path,
    output_path: Path | None = None,
    bbox_decimals: int = 3,
) -> dict[str, Any]:
    if export_format not in {"coco", "yolo"}:
        raise RoundTripError("export_format must be coco or yolo")
    if not 0 <= bbox_decimals <= 8:
        raise RoundTripError("bbox_decimals must be between 0 and 8")
    reference_coco = reference_coco.resolve()
    roundtrip_path = roundtrip_path.resolve()
    ontology = load_ontology(ontology_path)
    reference_document = load_json_object(reference_coco, label="reference COCO")
    include_attributes = export_format == "coco"
    reference = canonicalize_coco_document(
        reference_document,
        ontology=ontology,
        decimals=bbox_decimals,
        include_attributes=include_attributes,
        require_image_bindings=True,
    )
    if export_format == "coco":
        roundtrip_document, member = load_coco_export_document(roundtrip_path)
        roundtrip = canonicalize_coco_document(
            roundtrip_document,
            ontology=ontology,
            decimals=bbox_decimals,
            include_attributes=True,
            require_image_bindings=False,
        )
    else:
        member = None
        roundtrip = _canonicalize_yolo_zip(
            roundtrip_path,
            reference=reference,
            ontology=ontology,
            decimals=bbox_decimals,
        )
    image_difference = _difference(reference["images"], roundtrip["images"])
    record_difference = _difference(reference["records"], roundtrip["records"])
    passed = not any(
        (
            image_difference["missing_count"],
            image_difference["unexpected_count"],
            record_difference["missing_count"],
            record_difference["unexpected_count"],
        )
    )
    report = {
        "schema_version": ROUNDTRIP_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "format": export_format,
        "bbox_decimal_places": bbox_decimals,
        "ontology": ontology.record(),
        "class_map": reference["class_map"],
        "class_map_sha256": canonical_sha256(reference["class_map"]),
        "reference": {
            "name": reference_coco.name,
            "sha256": sha256_file(reference_coco),
            "image_bindings": reference["image_bindings"],
            "image_bindings_sha256": canonical_sha256(reference["image_bindings"]),
        },
        "roundtrip_artifact": {
            "name": roundtrip_path.name,
            "sha256": sha256_file(roundtrip_path),
            "annotation_member": member,
        },
        "attribute_scope": (
            "occluded_and_truncated_compared"
            if export_format == "coco"
            else "not_representable_in_yolo_bbox_format"
        ),
        "counts": {
            "images": len(reference["images"]),
            "annotations": len(reference["records"]),
        },
        "differences": {"images": image_difference, "records": record_difference},
    }
    if output_path is not None:
        output_path = output_path.resolve()
        if output_path.exists():
            raise FileExistsError(f"Refusing to replace existing report: {output_path}")
        write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Verify a CVAT COCO or YOLO bbox export against canonical COCO"
    )
    parser.add_argument("--reference-coco", type=Path, required=True)
    parser.add_argument("--roundtrip", type=Path, required=True)
    parser.add_argument("--format", choices=("coco", "yolo"), required=True)
    parser.add_argument(
        "--ontology", type=Path, default=root / "configs" / "classes.smd_v1.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bbox-decimals", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_cvat_roundtrip(
            reference_coco=args.reference_coco,
            roundtrip_path=args.roundtrip,
            export_format=args.format,
            ontology_path=args.ontology,
            output_path=args.output,
            bbox_decimals=args.bbox_decimals,
        )
    except (ContractError, RoundTripError, FileNotFoundError, FileExistsError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
