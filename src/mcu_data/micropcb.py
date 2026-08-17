from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from .common import sha256_file


RPI_MODEL_CODES = {
    "A": "raspberry_pi_a_plus",
    "H": "raspberry_pi_3_b_plus",
    "I": "raspberry_pi_1_b_plus",
}


def read_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {row["Image"]: row for row in rows}


def ensure_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.samefile(destination):
            return
        raise FileExistsError(f"Refusing to replace existing file: {destination}")
    os.link(source, destination)


def write_if_equal_or_absent(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(f"Refusing to replace existing generated file: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


def write_coco_dataset(records: list[dict[str, object]], output_root: Path) -> None:
    for split in ("train", "val"):
        split_records = [record for record in records if record["yolo_split"] == split]
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        for image_id, record in enumerate(split_records, start=1):
            image_name = Path(str(record["source_image"])).name
            source_image = Path(str(record["source_image"]))
            ensure_hardlink(source_image, output_root / f"{split}2017" / image_name)
            images.append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "width": record["width"],
                    "height": record["height"],
                }
            )
            annotations.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        record["bbox_left"],
                        record["bbox_top"],
                        record["bbox_width"],
                        record["bbox_height"],
                    ],
                    "area": int(record["bbox_width"]) * int(record["bbox_height"]),
                    "iscrowd": 0,
                }
            )
        coco = {
            "info": {"description": "micro-PCB Images Raspberry Pi SBC subset"},
            "licenses": [
                {
                    "id": 1,
                    "name": "CC BY 4.0",
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                }
            ],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "raspberry_pi_sbc", "supercategory": "board_sbc"}],
        }
        content = json.dumps(coco, ensure_ascii=False, indent=2) + "\n"
        write_if_equal_or_absent(output_root / "annotations" / f"instances_{split}2017.json", content)


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    manifest_path: Path,
    coco_output_root: Path | None = None,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    split_specs = (
        ("train", "train", source_root / "train_coded" / "train_coded"),
        ("test", "val", source_root / "test_coded" / "test_coded"),
    )
    for source_split, yolo_split, image_root in split_specs:
        sizes = read_index(source_root / f"{source_split}_sizes.csv")
        boxes = read_index(source_root / f"{source_split}_bboxes.csv")
        angles = read_index(source_root / f"{source_split}_angles.csv")
        ratios = read_index(source_root / f"{source_split}_ratio_top_to_bottom.csv")
        for image_name in sorted(sizes):
            model_code = image_name[0]
            if model_code not in RPI_MODEL_CODES:
                continue
            if image_name not in boxes:
                raise ValueError(f"Missing bounding box for {image_name}")

            source_image = image_root / image_name
            if not source_image.is_file():
                raise FileNotFoundError(source_image)

            width = int(sizes[image_name]["Width"])
            height = int(sizes[image_name]["Height"])
            left = int(boxes[image_name]["Left"])
            top = int(boxes[image_name]["Top"])
            box_width = int(boxes[image_name]["Width"])
            box_height = int(boxes[image_name]["Height"])
            x_center = (left + box_width / 2) / width
            y_center = (top + box_height / 2) / height
            norm_width = box_width / width
            norm_height = box_height / height
            normalized = (x_center, y_center, norm_width, norm_height)
            if not all(0.0 <= value <= 1.0 for value in normalized):
                raise ValueError(f"Out-of-range normalized bbox for {image_name}: {normalized}")

            destination_image = output_root / "images" / yolo_split / image_name
            destination_label = output_root / "labels" / yolo_split / f"{Path(image_name).stem}.txt"
            ensure_hardlink(source_image, destination_image)
            label = "0 " + " ".join(f"{value:.8f}" for value in normalized) + "\n"
            write_if_equal_or_absent(destination_label, label)

            model = RPI_MODEL_CODES[model_code]
            rotation_code, x_code, y_code, capture_serial = image_name[1], image_name[2], image_name[3], image_name[4]
            counts[f"{yolo_split}:{model}"] += 1
            records.append(
                {
                    "sample_id": f"micro_pcb_v2_{Path(image_name).stem}",
                    "dataset": "micro_pcb_images",
                    "source_url": "https://www.kaggle.com/datasets/frettapper/micropcb-images",
                    "license": "CC BY 4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "class_name": "raspberry_pi_sbc",
                    "model": model,
                    "model_code": model_code,
                    "rotation_code": rotation_code,
                    "x_code": x_code,
                    "y_code": y_code,
                    "capture_serial": capture_serial,
                    "condition_group_id": image_name[:4],
                    "source_split": source_split,
                    "yolo_split": yolo_split,
                    "physical_item_group": "NOT_VERIFIED",
                    "source_relative_path": source_image.relative_to(source_root).as_posix(),
                    "source_image": str(source_image.resolve()),
                    "processed_image": str(destination_image.resolve()),
                    "sha256": sha256_file(source_image),
                    "width": width,
                    "height": height,
                    "bbox_left": left,
                    "bbox_top": top,
                    "bbox_width": box_width,
                    "bbox_height": box_height,
                    "angle_deg": float(angles[image_name]["Angle"]),
                    "top_bottom_ratio": float(ratios[image_name]["Ratio of Top to Bottom Edge Length"]),
                }
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    private_runtime_fields = {"source_image", "processed_image"}
    fieldnames = [field for field in records[0] if field not in private_runtime_fields] if records else []
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    dataset_yaml = (
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: raspberry_pi_sbc\n"
    )
    write_if_equal_or_absent(output_root / "dataset.yaml", dataset_yaml)
    if coco_output_root is not None:
        write_coco_dataset(records, coco_output_root)
    train_condition_groups = {
        str(row["condition_group_id"]) for row in records if row["yolo_split"] == "train"
    }
    val_condition_groups = {
        str(row["condition_group_id"]) for row in records if row["yolo_split"] == "val"
    }
    summary: dict[str, object] = {
        "total_images": len(records),
        "split_counts": dict(Counter(str(row["yolo_split"]) for row in records)),
        "model_split_counts": dict(sorted(counts.items())),
        "storage": "NTFS hardlinks; images are not duplicated on disk",
        "coco_output_root": coco_output_root.as_posix() if coco_output_root is not None else None,
        "train_val_condition_group_overlap": len(train_condition_groups & val_condition_groups),
        "validation_images_with_condition_seen_in_train": sum(
            row["yolo_split"] == "val"
            and str(row["condition_group_id"]) in train_condition_groups
            for row in records
        ),
        "validation_policy": (
            "Source capture serials 1-4 are train and serial 5 is val. Physical specimen "
            "independence is not documented; formal AP requires an independent split."
        ),
    }
    (manifest_path.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the Raspberry Pi subset of micro-PCB Images")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coco-output-root", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_dataset(args.source_root, args.output_root, args.manifest, args.coco_output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
