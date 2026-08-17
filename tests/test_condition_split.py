from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from mcu_data.condition_split import assign_condition_groups, build_condition_split


FIELDS = [
    "sample_id",
    "class_name",
    "model",
    "condition_group_id",
    "source_relative_path",
    "yolo_split",
    "physical_item_group",
    "width",
    "height",
    "bbox_left",
    "bbox_top",
    "bbox_width",
    "bbox_height",
]


def _make_manifest(root: Path, *, groups_per_model: int = 10) -> tuple[Path, Path]:
    source_root = root / "source"
    manifest = root / "input.csv"
    rows: list[dict[str, str]] = []
    for model_index, model in enumerate(("model_a", "model_b")):
        for group_index in range(groups_per_model):
            group_id = f"{model}-condition-{group_index:02d}"
            for capture in range(2):
                name = f"m{model_index}-g{group_index:02d}-c{capture}.jpg"
                relative = Path(model) / name
                path = source_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (100, 80), color=(model_index * 50, group_index, capture)).save(path)
                rows.append(
                    {
                        "sample_id": Path(name).stem,
                        "class_name": "raspberry_pi_sbc",
                        "model": model,
                        "condition_group_id": group_id,
                        "source_relative_path": relative.as_posix(),
                        "yolo_split": "train" if capture == 0 else "val",
                        "physical_item_group": "NOT_VERIFIED",
                        "width": "100",
                        "height": "80",
                        "bbox_left": "10",
                        "bbox_top": "8",
                        "bbox_width": "50",
                        "bbox_height": "40",
                    }
                )
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest, source_root


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_duplicates(path: Path) -> None:
    pairs = [
        {
            "left_path": "train/m0-g00-c0.jpg",
            "right_path": "train/m0-g01-c0.jpg",
            "distance": 1,
        },
        {
            "left_path": "train/m0-g02-c0.jpg",
            "right_path": "train/m1-g02-c0.jpg",
            "distance": 4,
        },
    ]
    path.write_text(
        json.dumps({"near_duplicate_pairs": pairs}, indent=2) + "\n", encoding="utf-8"
    )


def test_assignment_has_zero_overlap_is_balanced_and_reproducible(tmp_path: Path) -> None:
    manifest, _ = _make_manifest(tmp_path)
    rows = _read_rows(manifest)

    first = assign_condition_groups(rows, seed=42, ratios={"train": 0.6, "val": 0.2, "test": 0.2})
    second = assign_condition_groups(rows, seed=42, ratios={"train": 0.6, "val": 0.2, "test": 0.2})
    different_seed = assign_condition_groups(
        rows, seed=43, ratios={"train": 0.6, "val": 0.2, "test": 0.2}
    )

    assert first == second
    assert first != different_seed
    for model in ("model_a", "model_b"):
        model_groups = {
            row["condition_group_id"] for row in rows if row["model"] == model
        }
        assert {split: sum(first[group] == split for group in model_groups) for split in ("train", "val", "test")} == {
            "train": 6,
            "val": 2,
            "test": 2,
        }


def test_builder_writes_manifest_yolo_coco_and_auditable_summary(tmp_path: Path) -> None:
    manifest, source_root = _make_manifest(tmp_path)
    output_manifest = tmp_path / "out" / "condition-split.csv"
    yolo_root = tmp_path / "yolo"
    coco_root = tmp_path / "coco"

    summary = build_condition_split(
        manifest,
        output_manifest,
        seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        source_root=source_root,
        yolo_output_root=yolo_root,
        coco_output_root=coco_root,
    )

    assert summary["condition_group_overlap_pass"] is True
    assert summary["condition_group_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert summary["split_image_counts"] == {"train": 24, "val": 8, "test": 8}
    assert summary["physical_item_independence_verified"] is False
    assert len(summary["assignment_sha256"]) == 64

    output_rows = _read_rows(output_manifest)
    groups_by_split = {
        split: {row["condition_group_id"] for row in output_rows if row["yolo_split"] == split}
        for split in ("train", "val", "test")
    }
    assert not groups_by_split["train"] & groups_by_split["val"]
    assert not groups_by_split["train"] & groups_by_split["test"]
    assert not groups_by_split["val"] & groups_by_split["test"]
    assert {row["previous_yolo_split"] for row in output_rows} == {"train", "val"}
    assert {row["split_seed"] for row in output_rows} == {"42"}

    assert (yolo_root / "dataset.yaml").is_file()
    assert sum(1 for path in (yolo_root / "images").rglob("*.jpg")) == 40
    assert sum(1 for path in (yolo_root / "labels").rglob("*.txt")) == 40
    for split, expected in (("train", 24), ("val", 8), ("test", 8)):
        coco = json.loads(
            (coco_root / "annotations" / f"instances_{split}2017.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(coco["images"]) == expected
        assert len(coco["annotations"]) == expected
        assert coco["categories"] == [
            {"id": 1, "name": "raspberry_pi_sbc", "supercategory": "mcu_vision"}
        ]


def test_phash_components_are_atomic_with_exact_model_balance(tmp_path: Path) -> None:
    manifest, source_root = _make_manifest(tmp_path)
    duplicates = tmp_path / "duplicates.json"
    _write_duplicates(duplicates)
    output_manifest = tmp_path / "out" / "phash-v2.csv"

    summary = build_condition_split(
        manifest,
        output_manifest,
        seed=42,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        source_root=source_root,
        yolo_output_root=tmp_path / "yolo-v2",
        coco_output_root=tmp_path / "coco-v2",
        duplicates_report=duplicates,
    )

    assert summary["policy"].endswith("_v2")
    assert summary["model_split_condition_group_counts"] == {
        "test:model_a": 2,
        "test:model_b": 2,
        "train:model_a": 6,
        "train:model_b": 6,
        "val:model_a": 2,
        "val:model_b": 2,
    }
    phash = summary["phash_audit"]
    assert phash["near_duplicate_pairs"] == 2
    assert phash["cross_split_near_duplicate_pairs"] == 0
    assert phash["cross_split_near_duplicate_pairs_pass"] is True
    assert phash["cross_split_phash_components"] == 0
    assert phash["cross_model_phash_components"] == 1
    assert phash["cross_model_phash_components_forced_to_train"] == 1

    rows = _read_rows(output_manifest)
    by_name = {Path(row["source_relative_path"]).name: row for row in rows}
    assert by_name["m0-g00-c0.jpg"]["yolo_split"] == by_name["m0-g01-c0.jpg"]["yolo_split"]
    assert by_name["m0-g02-c0.jpg"]["yolo_split"] == "train"
    assert by_name["m1-g02-c0.jpg"]["yolo_split"] == "train"
    assert {
        by_name["m0-g02-c0.jpg"]["leakage_component_id"],
        by_name["m1-g02-c0.jpg"]["leakage_component_id"],
    } == {by_name["m0-g02-c0.jpg"]["leakage_component_id"]}


def test_assignment_rejects_group_spanning_models(tmp_path: Path) -> None:
    manifest, _ = _make_manifest(tmp_path, groups_per_model=3)
    rows = _read_rows(manifest)
    rows[-1]["condition_group_id"] = rows[0]["condition_group_id"]
    try:
        assign_condition_groups(rows)
    except ValueError as error:
        assert "spans multiple models" in str(error)
    else:
        raise AssertionError("Expected a model-spanning condition group to be rejected")
