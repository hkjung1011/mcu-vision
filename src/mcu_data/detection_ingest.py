from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps, UnidentifiedImageError

from .common import IMAGE_SUFFIXES, load_yaml, portable_path, sha256_file, utc_now, write_json
from .contracts import (
    ContractError,
    Ontology,
    canonical_sha256,
    load_ontology,
    safe_relative_path,
)


MANIFEST_SCHEMA = "mcu.detection-source-manifest.v1"
COCO_DESCRIPTION = "Canonical multi-object detection import; source data remains candidate-only"


class DetectionIngestError(ValueError):
    """Raised when a source archive cannot be ingested without ambiguity."""


@dataclass(frozen=True)
class ArchiveImage:
    member: str
    content: bytes
    width: int
    height: int
    sha256: str
    suffix: str


def _normalized_member(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or PureWindowsPath(normalized).drive
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DetectionIngestError(f"Unsafe archive member: {value!r}")
    return path.as_posix()


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = _normalized_member(info.filename)
        if name in members:
            raise DetectionIngestError(f"Duplicate normalized archive member: {name}")
        members[name] = info
    return members


def _load_registry_entry(path: Path, dataset_key: str) -> dict[str, Any]:
    document = load_yaml(path.resolve())
    datasets = document.get("datasets")
    if not isinstance(datasets, Mapping) or dataset_key not in datasets:
        raise DetectionIngestError(f"Dataset {dataset_key!r} is absent from registry {path}")
    entry = datasets[dataset_key]
    if not isinstance(entry, Mapping):
        raise DetectionIngestError(f"Dataset registry entry must be a mapping: {dataset_key}")
    record = dict(entry)
    required = (
        "source_id",
        "dataset_version",
        "author",
        "source_url",
        "rights_statement",
        "rights_url",
        "ingest_split_policy",
    )
    missing = [field for field in required if record.get(field) in (None, "")]
    if missing:
        raise DetectionIngestError(
            f"Dataset registry entry {dataset_key!r} is missing: {', '.join(missing)}"
        )
    if record["ingest_split_policy"] not in {"bootstrap_train_only", "preserve_source_split"}:
        raise DetectionIngestError(
            "ingest_split_policy must be bootstrap_train_only or preserve_source_split"
        )
    return record


def _source_split(annotation_member: str) -> str:
    tokens = [part.casefold() for part in PurePosixPath(annotation_member).parts]
    for token in reversed(tokens):
        if token in {"valid", "validation", "val"}:
            return "val"
        if token in {"test", "testing"}:
            return "test"
        if token in {"train", "training"}:
            return "train"
    return "train"


def _output_split(source_split: str, policy: str) -> str:
    if policy == "bootstrap_train_only":
        return "train"
    return source_split


def _coco_documents(
    archive: zipfile.ZipFile, members: Mapping[str, zipfile.ZipInfo]
) -> list[tuple[str, dict[str, Any]]]:
    documents: list[tuple[str, dict[str, Any]]] = []
    for member, info in sorted(members.items()):
        if not member.casefold().endswith(".json"):
            continue
        try:
            value = json.loads(archive.read(info).decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        if all(isinstance(value.get(key), list) for key in ("images", "annotations", "categories")):
            documents.append((member, value))
    if not documents:
        raise DetectionIngestError("No COCO annotation JSON was found in the archive")
    return documents


def _image_member(
    annotation_member: str,
    file_name: str,
    members: Mapping[str, zipfile.ZipInfo],
    basenames: Mapping[str, list[str]],
) -> str:
    normalized_file = safe_relative_path(file_name, field="COCO image file_name")
    annotation_parent = PurePosixPath(annotation_member).parent
    candidates = [
        (annotation_parent / PurePosixPath(normalized_file)).as_posix(),
        normalized_file,
    ]
    for candidate in candidates:
        if candidate in members:
            return candidate
    basename = PurePosixPath(normalized_file).name.casefold()
    matches = basenames.get(basename, [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DetectionIngestError(
            f"COCO image is missing from archive: {annotation_member}:{file_name}"
        )
    raise DetectionIngestError(
        f"COCO image basename is ambiguous in archive: {annotation_member}:{file_name}"
    )


def _decode_archive_image(member: str, content: bytes) -> ArchiveImage:
    try:
        with Image.open(BytesIO(content)) as probe:
            probe.verify()
        with Image.open(BytesIO(content)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise DetectionIngestError(f"Archive image cannot be decoded: {member}") from exc
    if width <= 0 or height <= 0:
        raise DetectionIngestError(f"Archive image has invalid dimensions: {member}")
    suffix = PurePosixPath(member).suffix.casefold()
    if suffix not in IMAGE_SUFFIXES:
        raise DetectionIngestError(f"Unsupported archive image extension: {member}")
    return ArchiveImage(
        member=member,
        content=content,
        width=width,
        height=height,
        sha256=sha256(content).hexdigest(),
        suffix=suffix,
    )


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DetectionIngestError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise DetectionIngestError(f"{field} must be finite")
    return result


def _validated_bbox(value: Any, *, width: int, height: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise DetectionIngestError(f"{field} must be a four-value COCO bbox")
    x, y, box_width, box_height = [
        _finite_float(item, field=field) for item in value
    ]
    tolerance = 1e-6
    if box_width <= 0 or box_height <= 0:
        raise DetectionIngestError(f"{field} width and height must be positive")
    if x < -tolerance or y < -tolerance:
        raise DetectionIngestError(f"{field} begins outside the image")
    if x + box_width > width + tolerance or y + box_height > height + tolerance:
        raise DetectionIngestError(f"{field} extends outside the image")
    return [max(0.0, x), max(0.0, y), box_width, box_height]


def _annotation_attributes(annotation: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    raw = annotation.get("attributes")
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, (str, int, float, bool, type(None))):
                result[key] = value
    for key in ("occluded", "truncated"):
        if key in annotation:
            value = annotation[key]
            if not isinstance(value, (bool, int)):
                raise DetectionIngestError(f"COCO annotation {key} must be boolean-like")
            result[key] = bool(value)
    return dict(sorted(result.items()))


def _write_bytes_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing output: {path}")
    path.write_bytes(content)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing output: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _yolo_line(annotation: Mapping[str, Any], *, width: int, height: int) -> str:
    x, y, box_width, box_height = [float(value) for value in annotation["bbox"]]
    center_x = (x + box_width / 2) / width
    center_y = (y + box_height / 2) / height
    return (
        f"{int(annotation['ontology_class_id'])} {center_x:.8f} {center_y:.8f} "
        f"{box_width / width:.8f} {box_height / height:.8f}"
    )


def _dataset_yaml(ontology: Ontology, splits: Iterable[str]) -> str:
    lines = ["# Derived from canonical COCO; class IDs are frozen by ontology_sha256."]
    for split in ("train", "val", "test"):
        if split in splits:
            lines.append(f"{split}: images/{split}")
    lines.append("names:")
    lines.extend(f"  {class_id}: {name}" for class_id, name in sorted(ontology.classes_by_id.items()))
    return "\n".join(lines) + "\n"


def ingest_coco_archive(
    *,
    archive_path: Path,
    registry_path: Path,
    dataset_key: str,
    ontology_path: Path,
    output_root: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    registry_path = registry_path.resolve()
    output_root = output_root.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_root}")
    if not zipfile.is_zipfile(archive_path):
        raise DetectionIngestError(f"Source archive must be a ZIP file: {archive_path}")
    resolved_manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else output_root / "source_manifest.json"
    )
    if resolved_manifest.exists():
        raise FileExistsError(f"Refusing to replace existing source manifest: {resolved_manifest}")
    registry = _load_registry_entry(registry_path, dataset_key)
    dataset_id = str(registry["source_id"])
    ontology = load_ontology(ontology_path)
    policy = str(registry["ingest_split_policy"])
    output_root.mkdir(parents=True, exist_ok=True)

    assets: list[dict[str, Any]] = []
    canonical_images: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canonical_annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    yolo_rows: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen_content: dict[str, str] = {}
    class_counts: Counter[str] = Counter()
    class_images: dict[str, set[str]] = defaultdict(set)
    source_split_counts: Counter[str] = Counter()
    annotation_id = 1
    image_id = 1

    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_members(archive)
        basename_index: dict[str, list[str]] = defaultdict(list)
        for member in members:
            basename_index[PurePosixPath(member).name.casefold()].append(member)
        documents = _coco_documents(archive, members)
        for annotation_member, document in documents:
            source_split = _source_split(annotation_member)
            split = _output_split(source_split, policy)
            source_split_counts[source_split] += len(document["images"])
            raw_categories = document["categories"]
            category_map: dict[Any, tuple[int, str, str]] = {}
            for category in raw_categories:
                if not isinstance(category, Mapping) or "id" not in category:
                    raise DetectionIngestError(
                        f"Invalid COCO category in {annotation_member}"
                    )
                source_name = category.get("name")
                if not isinstance(source_name, str) or not source_name:
                    raise DetectionIngestError("COCO category name must be non-empty")
                canonical_name = ontology.source_name(dataset_id, source_name)
                category_id = category["id"]
                if category_id in category_map:
                    raise DetectionIngestError(
                        f"Duplicate COCO category id in {annotation_member}: {category_id!r}"
                    )
                category_map[category_id] = (
                    ontology.class_id(canonical_name),
                    canonical_name,
                    source_name,
                )
            images = document["images"]
            annotations = document["annotations"]
            by_source_id: dict[Any, dict[str, Any]] = {}
            for source_image in images:
                if not isinstance(source_image, Mapping) or "id" not in source_image:
                    raise DetectionIngestError(f"Invalid COCO image in {annotation_member}")
                source_image_id = source_image["id"]
                if isinstance(source_image_id, bool) or not isinstance(source_image_id, int):
                    raise DetectionIngestError(
                        f"COCO image id must be an integer in {annotation_member}: "
                        f"{source_image_id!r}"
                    )
                if source_image_id in by_source_id:
                    raise DetectionIngestError(
                        f"Duplicate COCO image id in {annotation_member}: {source_image_id!r}"
                    )
                file_name = source_image.get("file_name")
                if not isinstance(file_name, str):
                    raise DetectionIngestError("COCO image file_name must be a string")
                member = _image_member(
                    annotation_member, file_name, members, basename_index
                )
                decoded = _decode_archive_image(member, archive.read(members[member]))
                try:
                    declared = (int(source_image["width"]), int(source_image["height"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise DetectionIngestError(
                        f"Invalid declared dimensions for {member}"
                    ) from exc
                if declared != (decoded.width, decoded.height):
                    raise DetectionIngestError(
                        f"COCO/decode dimension mismatch for {member}: "
                        f"declared={declared}, decoded={(decoded.width, decoded.height)}"
                    )
                if decoded.sha256 in seen_content:
                    raise DetectionIngestError(
                        "Exact duplicate source image content requires explicit review: "
                        f"{seen_content[decoded.sha256]} and {member}"
                    )
                seen_content[decoded.sha256] = member
                asset_id = f"{dataset_id}:{decoded.sha256}"
                canonical_name = f"{decoded.sha256[:20]}{decoded.suffix}"
                coco_image_path = output_root / "coco" / f"{split}2017" / canonical_name
                yolo_image_path = output_root / "yolo" / "images" / split / canonical_name
                _write_bytes_new(coco_image_path, decoded.content)
                _link_or_copy(coco_image_path, yolo_image_path)
                image_record = {
                    "id": image_id,
                    "file_name": canonical_name,
                    "width": decoded.width,
                    "height": decoded.height,
                    "mcu_asset_id": asset_id,
                    "sha256": decoded.sha256,
                    "source_dataset_id": dataset_id,
                    "source_image_id": source_image_id,
                    "source_annotation_member": annotation_member,
                    "source_member": member,
                    "source_split": source_split,
                    "output_role": "bootstrap_train_only" if policy == "bootstrap_train_only" else split,
                }
                canonical_images[split].append(image_record)
                asset_record = {
                    "asset_id": asset_id,
                    "path": (Path("yolo") / "images" / split / canonical_name).as_posix(),
                    "sha256": decoded.sha256,
                    "width": decoded.width,
                    "height": decoded.height,
                    "source_member": member,
                    "source_image_id": source_image_id,
                    "source_annotation_member": annotation_member,
                    "source_split": source_split,
                    "role": "bootstrap_train_only" if policy == "bootstrap_train_only" else split,
                }
                assets.append(asset_record)
                by_source_id[source_image_id] = {
                    "canonical": image_record,
                    "label_name": canonical_name,
                    "split": split,
                }
                image_id += 1
            for source_annotation_index, source_annotation in enumerate(annotations, start=1):
                if not isinstance(source_annotation, Mapping):
                    raise DetectionIngestError(
                        f"Invalid COCO annotation in {annotation_member}:{source_annotation_index}"
                    )
                source_image_id = source_annotation.get("image_id")
                if source_image_id not in by_source_id:
                    raise DetectionIngestError(
                        f"COCO annotation references an unknown image id: {source_image_id!r}"
                    )
                category_id = source_annotation.get("category_id")
                if category_id not in category_map:
                    raise DetectionIngestError(
                        f"COCO annotation references an unknown category id: {category_id!r}"
                    )
                if source_annotation.get("iscrowd", 0) not in (0, False):
                    raise DetectionIngestError("Crowd annotations cannot be derived to YOLO bbox labels")
                image_record = by_source_id[source_image_id]["canonical"]
                bbox = _validated_bbox(
                    source_annotation.get("bbox"),
                    width=int(image_record["width"]),
                    height=int(image_record["height"]),
                    field=f"{annotation_member}:annotation:{source_annotation_index}",
                )
                ontology_class_id, canonical_class, source_class = category_map[category_id]
                attributes = _annotation_attributes(source_annotation)
                annotation_record: dict[str, Any] = {
                    "id": annotation_id,
                    "image_id": image_record["id"],
                    "category_id": ontology_class_id + 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "ontology_class_id": ontology_class_id,
                    "source_category_name": source_class,
                    "source_annotation_id": source_annotation.get("id", source_annotation_index),
                }
                if attributes:
                    annotation_record["attributes"] = attributes
                split = by_source_id[source_image_id]["split"]
                canonical_annotations[split].append(annotation_record)
                label_name = by_source_id[source_image_id]["label_name"]
                yolo_rows[(split, label_name)].append(
                    _yolo_line(
                        annotation_record,
                        width=int(image_record["width"]),
                        height=int(image_record["height"]),
                    )
                )
                class_counts[canonical_class] += 1
                class_images[canonical_class].add(str(image_record["mcu_asset_id"]))
                annotation_id += 1

    categories = [
        {
            "id": class_id + 1,
            "name": name,
            "ontology_class_id": class_id,
            "supercategory": "mcu_smd",
        }
        for class_id, name in sorted(ontology.classes_by_id.items())
    ]
    rights = {
        "statement": str(registry["rights_statement"]),
        "url": str(registry["rights_url"]),
        "asserted_by": str(registry["author"]),
        "note": "Source assertion recorded verbatim; this manifest does not create a license grant.",
    }
    artifact_records: dict[str, dict[str, Any]] = {}
    present_splits = sorted(canonical_images)
    for split in present_splits:
        coco_document = {
            "info": {
                "description": COCO_DESCRIPTION,
                "source_dataset_id": dataset_id,
                "source_url": registry["source_url"],
                "ontology_id": ontology.ontology_id,
                "ontology_sha256": ontology.sha256,
                "rights": rights,
                "formal_evaluation_allowed": bool(
                    registry.get("formal_evaluation_allowed", False)
                ),
            },
            "licenses": [],
            "images": canonical_images[split],
            "annotations": canonical_annotations.get(split, []),
            "categories": categories,
        }
        annotation_path = output_root / "coco" / "annotations" / f"instances_{split}2017.json"
        write_json(annotation_path, coco_document)
        artifact_records[f"coco_{split}"] = {
            "path": annotation_path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(annotation_path),
        }
        for image_record in canonical_images[split]:
            label_path = (
                output_root
                / "yolo"
                / "labels"
                / split
                / Path(str(image_record["file_name"])).with_suffix(".txt")
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = yolo_rows.get((split, str(image_record["file_name"])), [])
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    dataset_yaml_path = output_root / "yolo" / "dataset.yaml"
    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml_path.write_text(_dataset_yaml(ontology, present_splits), encoding="utf-8")
    artifact_records["yolo_dataset"] = {
        "path": dataset_yaml_path.relative_to(output_root).as_posix(),
        "sha256": sha256_file(dataset_yaml_path),
    }

    image_binding = [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "width": row["width"],
            "height": row["height"],
            "role": row["role"],
            "asset_id": row["asset_id"],
            "source_member": row["source_member"],
            "source_image_id": row["source_image_id"],
            "source_annotation_member": row["source_annotation_member"],
            "source_split": row["source_split"],
        }
        for row in sorted(assets, key=lambda value: str(value["path"]))
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "CANDIDATE_ONLY_NOT_APPROVED",
        "annotation_state": "imported_source_annotations_pending_audit",
        "dataset_id": dataset_id,
        "dataset_key": dataset_key,
        "role": "bootstrap_train_only" if policy == "bootstrap_train_only" else "source_split_candidate",
        "autolabel_allowed": False,
        "created_at": utc_now(),
        "source": {
            "provider": registry.get("provider"),
            "author": registry["author"],
            "url": registry["source_url"],
            "dataset_version": registry["dataset_version"],
            "archive": {
                "name": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
            },
            "rights": rights,
        },
        "ontology": ontology.record(),
        "split_policy": policy,
        "source_split_image_counts": dict(sorted(source_split_counts.items())),
        "output_split_image_counts": {
            split: len(canonical_images[split]) for split in present_splits
        },
        "class_instance_counts": dict(sorted(class_counts.items())),
        "class_image_counts": {
            name: len(asset_ids) for name, asset_ids in sorted(class_images.items())
        },
        "images": image_binding,
        "image_list_sha256": canonical_sha256(image_binding),
        "artifacts": artifact_records,
        "claims": {
            "independent_image_count_verified": False,
            "formal_validation_or_test_ready": False,
            "raw_images_publishable_by_this_manifest": False,
        },
    }
    write_json(resolved_manifest, manifest)
    return manifest | {"manifest_path": portable_path(resolved_manifest)}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a licensed/provenance-recorded multi-object COCO ZIP as canonical COCO "
            "and a derived YOLO dataset without promoting it to approved training data"
        )
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=root / "configs" / "datasets.curated.yaml"
    )
    parser.add_argument("--dataset", required=True, help="Dataset key in the registry")
    parser.add_argument(
        "--ontology", type=Path, default=root / "configs" / "classes.smd_v1.yaml"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = ingest_coco_archive(
            archive_path=args.archive,
            registry_path=args.registry,
            dataset_key=args.dataset,
            ontology_path=args.ontology,
            output_root=args.output_root,
            manifest_path=args.manifest,
        )
    except (ContractError, DetectionIngestError, FileNotFoundError, FileExistsError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset_id": result["dataset_id"],
                "images": len(result["images"]),
                "class_instance_counts": result["class_instance_counts"],
                "manifest": result["manifest_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
