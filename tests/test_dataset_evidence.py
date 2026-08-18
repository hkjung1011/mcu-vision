from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from mcu_data.common import sha256_file
from mcu_data.dataset_evidence import (
    ARTIFACT_FILENAMES,
    REQUIRED_EVIDENCE_FIELDS,
    DatasetEvidenceError,
    build_dataset_equivalence_evidence,
    canonicalize_yolo_dataset,
    load_dataset_evidence,
    resolve_protocol_test_evidence,
    verify_dataset_against_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    yolo = root / "relocated" / "yolo"
    coco = root / "another-layout" / "coco"
    for split in ("train", "val"):
        (yolo / "images" / split).mkdir(parents=True)
        (yolo / "labels" / split).mkdir(parents=True)
    (coco / "annotations").mkdir(parents=True)
    (coco / "train2017").mkdir(parents=True)
    (coco / "val2017").mkdir(parents=True)

    Image.new("RGB", (100, 80), color=(20, 80, 120)).save(
        yolo / "images" / "train" / "train_chip.png"
    )
    Image.new("RGB", (200, 100), color=(120, 30, 40)).save(
        yolo / "images" / "val" / "val_chip.png"
    )
    shutil.copy2(
        yolo / "images" / "train" / "train_chip.png",
        coco / "train2017" / "train_chip.png",
    )
    shutil.copy2(
        yolo / "images" / "val" / "val_chip.png",
        coco / "val2017" / "val_chip.png",
    )
    # Deliberately reverse annotation order. The small decimal perturbation must
    # normalize to the same fixed six-decimal coordinate as the COCO box.
    (yolo / "labels" / "train" / "train_chip.txt").write_text(
        "1 0.750000004 0.50000000 0.10000000 0.25000000\n"
        "0 0.300000004 0.37500000 0.40000000 0.50000000\n",
        encoding="utf-8",
    )
    (yolo / "labels" / "val" / "val_chip.txt").write_text(
        "0 0.50000000 0.50000000 0.20000000 0.40000000\n",
        encoding="utf-8",
    )
    dataset_yaml = yolo / "dataset.yaml"
    dataset_yaml.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: chip\n"
        "  1: board\n",
        encoding="utf-8",
    )
    train_json = coco / "annotations" / "instances_train2017.json"
    train_json.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "id": 600,
                        "image_id": 55,
                        "category_id": 4,
                        "bbox": [70, 30, 10, 20],
                        "area": 200,
                        "iscrowd": 0,
                    },
                    {
                        "id": 500,
                        "image_id": 55,
                        "category_id": 99,
                        "bbox": [10, 10, 40, 40],
                        "area": 1600,
                        "iscrowd": 0,
                    },
                ],
                "categories": [
                    {"id": 4, "name": "board"},
                    {"id": 99, "name": "chip"},
                ],
                "images": [
                    {"id": 55, "height": 80, "width": 100, "file_name": "train_chip.png"}
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    val_json = coco / "annotations" / "instances_val2017.json"
    val_json.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 7, "file_name": "val_chip.png", "width": 200, "height": 100}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 7,
                        "category_id": 99,
                        "bbox": [80, 30, 40, 40],
                        "iscrowd": 0,
                    }
                ],
                "categories": [
                    {"id": 99, "name": "chip"},
                    {"id": 4, "name": "board"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return dataset_yaml, train_json, val_json, coco / "train2017", coco / "val2017"


def _build(fixture: tuple[Path, Path, Path, Path, Path], output: Path) -> dict[str, object]:
    dataset_yaml, train_json, val_json, train_images, val_images = fixture
    return build_dataset_equivalence_evidence(
        yolo_dataset_yaml=dataset_yaml,
        coco_train_annotations=train_json,
        coco_val_annotations=val_json,
        coco_train_image_root=train_images,
        coco_val_image_root=val_images,
        output_dir=output,
    )


def test_equivalent_yolo_and_coco_write_release_gate_evidence(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    report = _build(_write_fixture(tmp_path / "source"), output)

    assert report["status"] == "PASS"
    assert report["equivalent"] is True
    evidence = load_dataset_evidence(output / ARTIFACT_FILENAMES["dataset_evidence"])
    assert set(evidence) == set(REQUIRED_EVIDENCE_FIELDS)
    assert evidence["class_map_sha256"] == sha256_file(output / "class_map.json")
    assert evidence["train_image_list_sha256"] == sha256_file(
        output / "train_image_list.json"
    )
    assert evidence["val_image_list_sha256"] == sha256_file(output / "val_image_list.json")
    assert evidence["canonical_train_records_sha256"] == sha256_file(
        output / "canonical_train_records.jsonl"
    )
    assert evidence["canonical_val_records_sha256"] == sha256_file(
        output / "canonical_val_records.jsonl"
    )
    assert evidence["canonical_dataset_manifest_sha256"] == sha256_file(
        output / "canonical_dataset_manifest.json"
    )
    record_rows = [
        json.loads(line)
        for line in (output / "canonical_train_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(record_rows) == 2
    assert all(len(row["record_sha256"]) == 64 for row in record_rows)
    assert record_rows[0]["record"]["bbox_xywh_pixels"][0] in {
        "10.000",
        "70.000",
    }


def test_hashes_are_independent_of_paths_and_source_order(tmp_path: Path) -> None:
    first_fixture = _write_fixture(tmp_path / "first")
    first = _build(first_fixture, tmp_path / "first-output")

    relocated = tmp_path / "second" / "deeply" / "moved"
    shutil.copytree(tmp_path / "first" / "relocated", relocated / "relocated")
    shutil.copytree(tmp_path / "first" / "another-layout", relocated / "another-layout")
    second_fixture = (
        relocated / "relocated" / "yolo" / "dataset.yaml",
        relocated / "another-layout" / "coco" / "annotations" / "instances_train2017.json",
        relocated / "another-layout" / "coco" / "annotations" / "instances_val2017.json",
        relocated / "another-layout" / "coco" / "train2017",
        relocated / "another-layout" / "coco" / "val2017",
    )
    train_document = json.loads(second_fixture[1].read_text(encoding="utf-8"))
    train_document["annotations"].reverse()
    train_document["categories"].reverse()
    second_fixture[1].write_text(
        json.dumps(train_document, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    second = _build(second_fixture, tmp_path / "second-output")

    assert first["evidence"] == second["evidence"]
    assert (tmp_path / "first-output" / "canonical_dataset_manifest.json").read_bytes() == (
        tmp_path / "second-output" / "canonical_dataset_manifest.json"
    ).read_bytes()
    assert (tmp_path / "first-output" / "canonical_train_records.jsonl").read_bytes() == (
        tmp_path / "second-output" / "canonical_train_records.jsonl"
    ).read_bytes()


def test_bbox_mismatch_fails_without_issuing_evidence(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "source")
    train_document = json.loads(fixture[1].read_text(encoding="utf-8"))
    train_document["annotations"][0]["bbox"][0] = 71
    fixture[1].write_text(json.dumps(train_document), encoding="utf-8")
    output = tmp_path / "failed-evidence"

    report = _build(fixture, output)

    assert report["status"] == "FAIL"
    assert report["evidence"] == {}
    assert report["differences"]["train"]["records"]["yolo_only_count"] == 1
    assert not (output / "dataset_evidence.json").exists()
    assert (output / "dataset_equivalence_report.json").exists()


def test_load_dataset_evidence_rejects_non_pass_document(tmp_path: Path) -> None:
    path = tmp_path / "dataset_evidence.json"
    path.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")

    with pytest.raises(DatasetEvidenceError, match="not PASS"):
        load_dataset_evidence(path)


def test_live_dataset_preflight_detects_changes_after_evidence(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "source")
    evidence_dir = tmp_path / "evidence"
    _build(fixture, evidence_dir)
    evidence_path = evidence_dir / ARTIFACT_FILENAMES["dataset_evidence"]

    verified = verify_dataset_against_evidence(
        evidence_path=evidence_path,
        yolo_dataset_yaml=fixture[0],
        coco_train_annotations=fixture[1],
        coco_val_annotations=fixture[2],
        coco_train_image_root=fixture[3],
        coco_val_image_root=fixture[4],
    )
    assert set(verified) == set(REQUIRED_EVIDENCE_FIELDS)

    label = fixture[0].parent / "labels" / "train" / "train_chip.txt"
    label.write_text(
        label.read_text(encoding="utf-8").replace("0.750000004", "0.700000004"),
        encoding="utf-8",
    )
    with pytest.raises(DatasetEvidenceError, match="not equivalent"):
        verify_dataset_against_evidence(
            evidence_path=evidence_path,
            yolo_dataset_yaml=fixture[0],
            coco_train_annotations=fixture[1],
            coco_val_annotations=fixture[2],
            coco_train_image_root=fixture[3],
            coco_val_image_root=fixture[4],
        )


def test_load_dataset_evidence_detects_changed_artifact(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    _build(_write_fixture(tmp_path / "source"), output)
    (output / "class_map.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(DatasetEvidenceError, match="artifact hash differs: class_map_sha256"):
        load_dataset_evidence(output / "dataset_evidence.json")


def test_optional_locked_test_split_and_coco_attribute_hashes(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "source")
    dataset_yaml, train_json, val_json, train_images, val_images = fixture
    yolo_root = dataset_yaml.parent
    (yolo_root / "images" / "test").mkdir(parents=True)
    (yolo_root / "labels" / "test").mkdir(parents=True)
    Image.new("RGB", (120, 90), color=(10, 20, 30)).save(
        yolo_root / "images" / "test" / "test_chip.png"
    )
    (yolo_root / "labels" / "test" / "test_chip.txt").write_text(
        "0 0.50000000 0.50000000 0.50000000 0.40000000\n",
        encoding="utf-8",
    )
    dataset_yaml.write_text(
        dataset_yaml.read_text(encoding="utf-8").replace(
            "val: images/val\n", "val: images/val\ntest: images/test\n"
        ),
        encoding="utf-8",
    )
    coco_root = train_images.parent
    test_images = coco_root / "test2017"
    test_images.mkdir()
    shutil.copy2(
        yolo_root / "images" / "test" / "test_chip.png",
        test_images / "test_chip.png",
    )
    test_json = coco_root / "annotations" / "instances_test2017.json"
    test_json.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 91, "file_name": "test_chip.png", "width": 120, "height": 90}
                ],
                "annotations": [
                    {
                        "id": 92,
                        "image_id": 91,
                        "category_id": 99,
                        "bbox": [30, 27, 60, 36],
                        "iscrowd": 0,
                        "attributes": {"occluded": False, "truncated": False},
                    }
                ],
                "categories": [
                    {"id": 99, "name": "chip"},
                    {"id": 4, "name": "board"},
                ],
            }
        ),
        encoding="utf-8",
    )
    train_document = json.loads(train_json.read_text(encoding="utf-8"))
    train_document["annotations"][0]["attributes"] = {
        "occluded": True,
        "truncated": False,
    }
    train_json.write_text(json.dumps(train_document), encoding="utf-8")
    output = tmp_path / "evidence"

    report = build_dataset_equivalence_evidence(
        yolo_dataset_yaml=dataset_yaml,
        coco_train_annotations=train_json,
        coco_val_annotations=val_json,
        coco_train_image_root=train_images,
        coco_val_image_root=val_images,
        coco_test_annotations=test_json,
        coco_test_image_root=test_images,
        include_coco_attributes=True,
        output_dir=output,
    )

    assert report["status"] == "PASS"
    assert "test_image_list_sha256" in report["evidence"]
    assert "canonical_test_records_sha256" in report["evidence"]
    assert "canonical_annotation_attributes_sha256" in report["evidence"]
    assert (output / "test_image_list.json").is_file()
    assert (output / "canonical_test_records.jsonl").is_file()
    assert (output / "canonical_annotation_attributes.jsonl").is_file()
    loaded = load_dataset_evidence(output / "dataset_evidence.json")
    assert loaded["canonical_annotation_attributes_sha256"] == sha256_file(
        output / "canonical_annotation_attributes.jsonl"
    )


def test_cross_split_exact_sha_duplicate_is_rejected_before_evidence(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "source")
    dataset_yaml, train_json, val_json, train_images, val_images = fixture
    yolo_root = dataset_yaml.parent
    shutil.copy2(
        yolo_root / "images" / "train" / "train_chip.png",
        yolo_root / "images" / "val" / "val_chip.png",
    )
    shutil.copy2(
        train_images / "train_chip.png",
        val_images / "val_chip.png",
    )
    val_document = json.loads(val_json.read_text(encoding="utf-8"))
    val_document["images"][0].update({"width": 100, "height": 80})
    val_document["annotations"][0]["bbox"] = [40, 24, 20, 32]
    val_json.write_text(json.dumps(val_document), encoding="utf-8")
    output = tmp_path / "evidence"

    with pytest.raises(DatasetEvidenceError, match="cross-split exact image SHA-256"):
        _build(fixture, output)
    assert not (output / "dataset_evidence.json").exists()


def test_optional_test_split_participates_in_exact_sha_leakage_gate(
    tmp_path: Path,
) -> None:
    dataset_yaml, *_ = _write_fixture(tmp_path / "source")
    yolo_root = dataset_yaml.parent
    (yolo_root / "images" / "test").mkdir(parents=True)
    (yolo_root / "labels" / "test").mkdir(parents=True)
    shutil.copy2(
        yolo_root / "images" / "train" / "train_chip.png",
        yolo_root / "images" / "test" / "same_bytes_new_name.png",
    )
    (yolo_root / "labels" / "test" / "same_bytes_new_name.txt").write_text(
        "", encoding="utf-8"
    )
    dataset_yaml.write_text(
        dataset_yaml.read_text(encoding="utf-8").replace(
            "val: images/val\n", "val: images/val\ntest: images/test\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetEvidenceError, match="cross-split exact image SHA-256"):
        canonicalize_yolo_dataset(dataset_yaml, include_test=True)


def test_protocol_locked_test_resolution_is_explicit_and_fail_closed(
    tmp_path: Path,
) -> None:
    coco_root = tmp_path / "coco"
    (coco_root / "annotations").mkdir(parents=True)
    (coco_root / "test2017").mkdir()
    test_json = coco_root / "annotations" / "instances_test2017.json"
    test_json.write_text("{}", encoding="utf-8")

    assert resolve_protocol_test_evidence(
        dataset_config={"locked_test_evidence_enabled": False},
        coco_root=coco_root,
        coco_test_annotations=None,
        coco_test_image_root=None,
    ) == (None, None, False)
    with pytest.raises(DatasetEvidenceError, match="forbidden"):
        resolve_protocol_test_evidence(
            dataset_config={"locked_test_evidence_enabled": False},
            coco_root=coco_root,
            coco_test_annotations=test_json,
            coco_test_image_root=coco_root / "test2017",
        )
    resolved = resolve_protocol_test_evidence(
        dataset_config={
            "locked_test_evidence_enabled": True,
            "include_coco_attributes": True,
        },
        coco_root=coco_root,
        coco_test_annotations=None,
        coco_test_image_root=None,
    )
    assert resolved == (test_json.resolve(), (coco_root / "test2017").resolve(), True)
    with pytest.raises(DatasetEvidenceError, match="must be boolean"):
        resolve_protocol_test_evidence(
            dataset_config={"locked_test_evidence_enabled": "true"},
            coco_root=coco_root,
            coco_test_annotations=None,
            coco_test_image_root=None,
        )


def test_baseline_formal_protocol_binds_all_nine_tracked_evidence_hashes() -> None:
    protocol = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments" / "baseline_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    dataset = protocol["dataset"]
    assert dataset["locked_test_evidence_enabled"] is True
    assert dataset["include_coco_attributes"] is True
    expected_fields = [
        *REQUIRED_EVIDENCE_FIELDS,
        "test_image_list_sha256",
        "canonical_test_records_sha256",
        "canonical_annotation_attributes_sha256",
    ]
    assert protocol["comparison_rules"]["required_dataset_evidence"] == list(
        REQUIRED_EVIDENCE_FIELDS
    )
    assert protocol["comparison_rules"]["formal_required_dataset_evidence"] == expected_fields
    evidence_path = PROJECT_ROOT / dataset["equivalence_evidence"]
    evidence = load_dataset_evidence(evidence_path)
    assert list(evidence) == expected_fields
    assert dataset["evidence"] == evidence
