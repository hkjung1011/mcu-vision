from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml

from mcu_data.annotation_promotion import PromotionError, promote_reviewed_annotations
from mcu_data.autolabel import AUTOLABEL_SOURCE_SCHEMA
from mcu_data.common import sha256_file
from mcu_data.contracts import canonical_sha256, load_ontology
from mcu_data.cvat_roundtrip import verify_cvat_roundtrip


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "classes.smd_v1.yaml"


def _reference_coco(path: Path) -> Path:
    document = {
        "images": [
            {"id": 1, "file_name": "one.png", "width": 100, "height": 80},
            {"id": 2, "file_name": "two.png", "width": 200, "height": 100},
        ],
        "categories": [
            {"id": 8, "name": "smd_resistor"},
            {"id": 7, "name": "smd_capacitor"},
        ],
        "annotations": [
            {
                "id": 9,
                "image_id": 1,
                "category_id": 7,
                "bbox": [10, 8, 20, 16],
                "iscrowd": 0,
                "attributes": {"occluded": False, "truncated": False},
            },
            {
                "id": 10,
                "image_id": 1,
                "category_id": 8,
                "bbox": [50, 20, 30, 20],
                "iscrowd": 0,
                "attributes": {"occluded": True, "truncated": False},
            },
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _coco_export(path: Path, reference: Path, *, change_attribute: bool = False) -> Path:
    document = json.loads(reference.read_text(encoding="utf-8"))
    document["categories"].reverse()
    document["annotations"].reverse()
    if change_attribute:
        document["annotations"][0]["attributes"]["occluded"] = False
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("annotations/instances_default.json", json.dumps(document))
    return path


def _yolo_export(path: Path) -> Path:
    ontology = load_ontology(ONTOLOGY_PATH)
    dataset = {"names": {key: name for key, name in ontology.classes_by_id.items()}}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.yaml", yaml.safe_dump(dataset, sort_keys=False))
        archive.writestr(
            "labels/one.txt",
            "0 0.20000000 0.20000000 0.20000000 0.20000000\n"
            "1 0.65000000 0.37500000 0.30000000 0.25000000\n",
        )
        archive.writestr("labels/two.txt", "")
    return path


def test_cvat_coco_and_yolo_roundtrip_pass_with_multi_object_image(tmp_path: Path) -> None:
    reference = _reference_coco(tmp_path / "reference.json")
    coco = verify_cvat_roundtrip(
        reference_coco=reference,
        roundtrip_path=_coco_export(tmp_path / "cvat-coco.zip", reference),
        export_format="coco",
        ontology_path=ONTOLOGY_PATH,
    )
    assert coco["status"] == "PASS"
    assert coco["attribute_scope"] == "occluded_and_truncated_compared"
    assert coco["counts"] == {"images": 2, "annotations": 2}

    yolo = verify_cvat_roundtrip(
        reference_coco=reference,
        roundtrip_path=_yolo_export(tmp_path / "cvat-yolo.zip"),
        export_format="yolo",
        ontology_path=ONTOLOGY_PATH,
    )
    assert yolo["status"] == "PASS"
    assert yolo["attribute_scope"] == "not_representable_in_yolo_bbox_format"


def test_cvat_coco_roundtrip_detects_changed_attribute(tmp_path: Path) -> None:
    reference = _reference_coco(tmp_path / "reference.json")
    report = verify_cvat_roundtrip(
        reference_coco=reference,
        roundtrip_path=_coco_export(
            tmp_path / "changed.zip", reference, change_attribute=True
        ),
        export_format="coco",
        ontology_path=ONTOLOGY_PATH,
    )
    assert report["status"] == "FAIL"
    assert report["differences"]["records"]["missing_count"] == 1


def _promotion_files(tmp_path: Path) -> dict[str, Path]:
    ontology = load_ontology(ONTOLOGY_PATH)
    image_rows = [
        {"path": "one.png", "sha256": "1" * 64, "width": 100, "height": 80, "role": "unlabeled_train"},
        {"path": "two.png", "sha256": "2" * 64, "width": 200, "height": 100, "role": "unlabeled_train"},
    ]
    image_list_sha = canonical_sha256(image_rows)
    pending = tmp_path / "pending.json"
    pending.write_text(
        json.dumps(
            {
                "status": "complete",
                "annotation_state": "PENDING_HUMAN_REVIEW",
                "run_id": "pending_smd_001",
                "source_binding": {
                    "schema_version": AUTOLABEL_SOURCE_SCHEMA,
                    "role": "unlabeled_train",
                    "ontology_sha256": ontology.sha256,
                    "image_list_sha256": image_list_sha,
                    "images": image_rows,
                },
                "ontology": ontology.record(),
                "protocol": {"automatic_promotion_to_training": False},
            }
        ),
        encoding="utf-8",
    )
    export = tmp_path / "reviewed-coco.zip"
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr("annotations.json", "{}")
    roundtrip = tmp_path / "roundtrip.json"
    roundtrip.write_text(
        json.dumps(
            {
                "schema_version": "mcu.cvat-roundtrip-report.v1",
                "status": "PASS",
                "format": "coco",
                "ontology": ontology.record(),
                "roundtrip_artifact": {
                    "name": export.name,
                    "sha256": sha256_file(export),
                },
                "counts": {"images": 2, "annotations": 2},
            }
        ),
        encoding="utf-8",
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": "mcu.cvat-review.v1",
                "pending_run_manifest_sha256": sha256_file(pending),
                "source_image_list_sha256": image_list_sha,
                "ontology_sha256": ontology.sha256,
                "cvat_export_sha256": sha256_file(export),
                "roundtrip_report_sha256": sha256_file(roundtrip),
                "reviewer": {"id": "reviewer-01", "name": "Synthetic Test Reviewer"},
                "approved_at_utc": "2026-08-18T10:00:00+00:00",
                "cvat": {"task_id": 101, "job_ids": [201]},
                "images": [
                    {"path": "one.png", "disposition": "corrected"},
                    {"path": "two.png", "disposition": "confirmed_empty"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return {"pending": pending, "export": export, "roundtrip": roundtrip, "review": review}


def test_promotion_requires_hash_bound_complete_human_review(tmp_path: Path) -> None:
    files = _promotion_files(tmp_path)
    output = tmp_path / "promotion.json"
    result = promote_reviewed_annotations(
        pending_run_manifest=files["pending"],
        review_manifest=files["review"],
        cvat_export=files["export"],
        roundtrip_report=files["roundtrip"],
        ontology_path=ONTOLOGY_PATH,
        output_path=output,
    )
    assert result["status"] == "PASS"
    assert result["annotation_state"] == "reviewed_train"
    assert result["training_use"] == {
        "allowed": True,
        "split": "train",
        "validation_or_test_use": False,
        "included_image_count": 2,
        "excluded_rejected_image_count": 0,
    }
    assert output.is_file()


def test_promotion_rejects_incomplete_review(tmp_path: Path) -> None:
    files = _promotion_files(tmp_path)
    review = json.loads(files["review"].read_text(encoding="utf-8"))
    review["images"].pop()
    files["review"].write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(PromotionError, match="coverage is incomplete"):
        promote_reviewed_annotations(
            pending_run_manifest=files["pending"],
            review_manifest=files["review"],
            cvat_export=files["export"],
            roundtrip_report=files["roundtrip"],
            ontology_path=ONTOLOGY_PATH,
            output_path=tmp_path / "promotion.json",
        )


def test_promotion_rejects_tampered_pending_image_binding(tmp_path: Path) -> None:
    files = _promotion_files(tmp_path)
    pending = json.loads(files["pending"].read_text(encoding="utf-8"))
    pending["source_binding"]["images"][0]["width"] = 101
    files["pending"].write_text(json.dumps(pending), encoding="utf-8")
    review = json.loads(files["review"].read_text(encoding="utf-8"))
    review["pending_run_manifest_sha256"] = sha256_file(files["pending"])
    files["review"].write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(PromotionError, match="image_list_sha256"):
        promote_reviewed_annotations(
            pending_run_manifest=files["pending"],
            review_manifest=files["review"],
            cvat_export=files["export"],
            roundtrip_report=files["roundtrip"],
            ontology_path=ONTOLOGY_PATH,
            output_path=tmp_path / "promotion.json",
        )
