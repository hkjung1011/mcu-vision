from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from .common import IMAGE_SUFFIXES, load_yaml, portable_path, sha256_file, utc_now, write_json


def normalized_pixel_hash(image: Image.Image) -> str:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{normalized.width}x{normalized.height}:RGB".encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def class_from_path(
    path: Path, data_root: Path, known_classes: set[str], default_class: str | None = None
) -> str:
    relative_parts = path.relative_to(data_root).parts[:-1]
    for part in reversed(relative_parts):
        if part in known_classes:
            return part
    return default_class or "UNASSIGNED"


def source_from_path(path: Path, data_root: Path) -> str:
    parts = path.relative_to(data_root).parts
    return parts[0] if len(parts) > 1 else "local"


def inspect_image(
    path: Path,
    data_root: Path,
    known_classes: set[str],
    min_width: int,
    min_height: int,
    default_class: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "relative_path": path.relative_to(data_root).as_posix(),
        "class_name": class_from_path(path, data_root, known_classes, default_class),
        "source": source_from_path(path, data_root),
        "bytes": path.stat().st_size,
        "sha256_file": "",
        "sha256_pixels": "",
        "phash64": "",
        "width": 0,
        "height": 0,
        "format": "",
        "mode": "",
        "decode_ok": False,
        "quality_flags": "",
        "error": "",
    }
    try:
        record["sha256_file"] = sha256_file(path)
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as image:
            image.load()
            normalized = ImageOps.exif_transpose(image)
            record["width"], record["height"] = normalized.size
            record["format"] = image.format or ""
            record["mode"] = image.mode
            record["sha256_pixels"] = normalized_pixel_hash(image)
            record["phash64"] = str(imagehash.phash(normalized.convert("RGB"), hash_size=8))
        flags: list[str] = []
        if record["width"] < min_width:
            flags.append("LOW_WIDTH")
        if record["height"] < min_height:
            flags.append("LOW_HEIGHT")
        if record["width"] and record["height"]:
            aspect = max(record["width"] / record["height"], record["height"] / record["width"])
            if aspect > 4.0:
                flags.append("EXTREME_ASPECT_RATIO")
        if record["class_name"] == "UNASSIGNED":
            flags.append("UNASSIGNED_CLASS")
        record["quality_flags"] = "|".join(flags)
        record["decode_ok"] = True
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}:{exc}"
    return record


def duplicate_groups(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = str(record.get(key, ""))
        if value:
            grouped[value].append(record)
    result: list[dict[str, Any]] = []
    for digest, items in grouped.items():
        if len(items) < 2:
            continue
        result.append(
            {
                "key": key,
                "value": digest,
                "classes": sorted({str(item["class_name"]) for item in items}),
                "paths": sorted(str(item["relative_path"]) for item in items),
                "cross_class_conflict": len({str(item["class_name"]) for item in items}) > 1,
            }
        )
    return result


def near_duplicate_pairs(
    records: list[dict[str, Any]], threshold: int, maximum_pairs: int
) -> tuple[list[dict[str, Any]], bool]:
    candidates = [record for record in records if record.get("phash64")]
    integer_hashes = [(record, int(str(record["phash64"]), 16)) for record in candidates]
    pairs: list[dict[str, Any]] = []
    truncated = False
    for (left, left_hash), (right, right_hash) in itertools.combinations(integer_hashes, 2):
        distance = (left_hash ^ right_hash).bit_count()
        if distance > threshold:
            continue
        if left.get("sha256_pixels") == right.get("sha256_pixels"):
            continue
        pairs.append(
            {
                "distance": distance,
                "left_path": left["relative_path"],
                "right_path": right["relative_path"],
                "left_class": left["class_name"],
                "right_class": right["class_name"],
                "cross_class_conflict": left["class_name"] != right["class_name"],
                "review_status": "PENDING_HUMAN_REVIEW",
            }
        )
        if len(pairs) >= maximum_pairs:
            truncated = True
            break
    pairs.sort(key=lambda pair: (pair["distance"], pair["left_path"], pair["right_path"]))
    return pairs, truncated


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "relative_path",
        "class_name",
        "source",
        "bytes",
        "sha256_file",
        "sha256_pixels",
        "phash64",
        "width",
        "height",
        "format",
        "mode",
        "decode_ok",
        "quality_flags",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def audit_dataset(
    data_root: Path,
    classes_config: Path,
    report_root: Path,
    min_width: int,
    min_height: int,
    phash_threshold: int,
    maximum_near_pairs: int,
    default_class: str | None = None,
) -> dict[str, Any]:
    class_config = load_yaml(classes_config)
    known_classes = set(class_config.get("classes", {}))
    if default_class is not None and default_class not in known_classes:
        raise ValueError(f"Unknown --default-class: {default_class}")
    paths = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )
    records = [
        inspect_image(path, data_root, known_classes, min_width, min_height, default_class)
        for path in tqdm(paths, desc="audit", unit="image")
    ]
    file_duplicates = duplicate_groups(records, "sha256_file")
    pixel_duplicates = duplicate_groups(records, "sha256_pixels")
    near_pairs, near_pairs_truncated = near_duplicate_pairs(records, phash_threshold, maximum_near_pairs)

    report_root.mkdir(parents=True, exist_ok=True)
    write_csv(report_root / "image-audit.csv", records)
    write_json(
        report_root / "duplicates.json",
        {
            "exact_file_groups": file_duplicates,
            "exact_pixel_groups": pixel_duplicates,
            "near_duplicate_pairs": near_pairs,
            "near_duplicate_pairs_truncated": near_pairs_truncated,
            "phash_threshold": phash_threshold,
        },
    )
    counts_by_class: dict[str, int] = defaultdict(int)
    for record in records:
        if record["decode_ok"]:
            counts_by_class[str(record["class_name"])] += 1
    summary = {
        "created_at": utc_now(),
        "data_root": portable_path(data_root),
        "scanned_files": len(records),
        "decode_failures": sum(not bool(record["decode_ok"]) for record in records),
        "quality_flagged": sum(bool(record["quality_flags"]) for record in records),
        "counts_by_class": dict(sorted(counts_by_class.items())),
        "exact_file_duplicate_groups": len(file_duplicates),
        "exact_pixel_duplicate_groups": len(pixel_duplicates),
        "cross_class_exact_conflicts": sum(bool(group["cross_class_conflict"]) for group in pixel_duplicates),
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_pairs_truncated": near_pairs_truncated,
        "status": "REVIEW_REQUIRED" if file_duplicates or pixel_duplicates or near_pairs else "NO_DUPLICATES_DETECTED",
    }
    write_json(report_root / "audit-summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit image integrity and exact/near duplicates without deleting files.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--classes-config", type=Path, default=Path("configs/classes.provisional.yaml"))
    parser.add_argument("--report-root", type=Path, default=Path("data/reports/audit"))
    parser.add_argument("--min-width", type=int, default=320)
    parser.add_argument("--min-height", type=int, default=240)
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--maximum-near-pairs", type=int, default=100_000)
    parser.add_argument("--default-class", help="Assign this known class when no class directory is present")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_dataset(
        data_root=args.data_root.resolve(),
        classes_config=args.classes_config.resolve(),
        report_root=args.report_root.resolve(),
        min_width=args.min_width,
        min_height=args.min_height,
        phash_threshold=args.phash_threshold,
        maximum_near_pairs=args.maximum_near_pairs,
        default_class=args.default_class,
    )
    print(result)


if __name__ == "__main__":
    main()
