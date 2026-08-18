from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
import yaml
from PIL import Image

import mcu_data.detection_ingest as detection_ingest_module
from mcu_data.common import sha256_file
from mcu_data.contracts import load_ontology
from mcu_data.curated import download_curated
from mcu_data.detection_ingest import DetectionIngestError, ingest_coco_archive


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = PROJECT_ROOT / "configs" / "classes.smd_v1.yaml"
PDM_ASSERTION = (
    "PDM-1.0 asserted by the Roboflow Universe project maintained by Dainius; "
    "this records the source assertion and Public Domain Mark is not a license grant."
)


def _png(color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (100, 80), color=color).save(stream, format="PNG")
    return stream.getvalue()


def _registry(path: Path) -> Path:
    path.write_text(
        "datasets:\n"
        "  smd_components_raw:\n"
        "    provider: manual_roboflow\n"
        "    source_id: dainius_smdcomponents_v2\n"
        "    dataset_version: 2\n"
        "    author: Dainius Varna and Vytautas Abromavicius\n"
        "    source_url: https://universe.roboflow.com/dainius/smdcomponents/dataset/2\n"
        f"    rights_statement: >-\n      {PDM_ASSERTION}\n"
        "    rights_url: https://creativecommons.org/publicdomain/mark/1.0/\n"
        "    ingest_split_policy: bootstrap_train_only\n"
        "    formal_evaluation_allowed: false\n"
        "    allowed_source_labels:\n"
        "      Condensator: smd_capacitor\n"
        "      Resistor: smd_resistor\n"
        "      Diode: smd_diode\n"
        "      Transistor: smd_transistor\n",
        encoding="utf-8",
    )
    return path


def _archive(path: Path, *, category_name: str = "Condensator") -> Path:
    document = {
        "images": [
            {"id": 10, "file_name": "one.png", "width": 100, "height": 80},
            {"id": 11, "file_name": "two.png", "width": 100, "height": 80},
        ],
        "categories": [
            {"id": 7, "name": category_name},
            {"id": 9, "name": "Resistor"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 10,
                "category_id": 7,
                "bbox": [10, 8, 20, 16],
                "iscrowd": 0,
                "attributes": {"occluded": False},
            },
            {
                "id": 2,
                "image_id": 10,
                "category_id": 9,
                "bbox": [50, 20, 30, 20],
                "iscrowd": 0,
            },
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("train/_annotations.coco.json", json.dumps(document))
        archive.writestr("train/one.png", _png((20, 40, 60)))
        archive.writestr("train/two.png", _png((60, 40, 20)))
    return path


def test_ingest_preserves_multi_object_labels_rights_and_hashes(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "smd-v2.zip")
    project = tmp_path / "project"
    output = project / "data" / "quarantine" / "out"

    result = ingest_coco_archive(
        archive_path=archive,
        registry_path=_registry(tmp_path / "registry.yaml"),
        dataset_key="smd_components_raw",
        ontology_path=ONTOLOGY,
        output_root=output,
        project_root_path=project,
    )

    assert result["status"] == "CANDIDATE_ONLY_NOT_APPROVED"
    assert result["role"] == "bootstrap_train_only"
    assert result["autolabel_allowed"] is False
    assert result["source"]["rights"]["statement"] == PDM_ASSERTION
    assert result["source"]["archive"]["sha256"] == sha256_file(archive)
    assert result["images"][0]["asset_id"].startswith("dainius_smdcomponents_v2:")
    assert result["images"][0]["source_image_id"] == 10
    assert result["images"][0]["source_annotation_member"] == "train/_annotations.coco.json"
    assert result["ontology"]["sha256"] == load_ontology(ONTOLOGY).sha256
    assert result["class_instance_counts"] == {"smd_capacitor": 1, "smd_resistor": 1}
    assert result["class_image_counts"] == {"smd_capacitor": 1, "smd_resistor": 1}
    assert result["claims"]["formal_validation_or_test_ready"] is False
    assert result["claims"]["raw_images_publishable_by_this_manifest"] is False

    labels = sorted((output / "yolo" / "labels" / "train").glob("*.txt"))
    assert len(labels) == 2
    populated = [path for path in labels if path.read_text(encoding="utf-8")]
    assert len(populated) == 1
    lines = populated[0].read_text(encoding="utf-8").splitlines()
    assert [int(line.split()[0]) for line in lines] == [0, 1]
    coco = json.loads(
        (output / "coco" / "annotations" / "instances_train2017.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 2
    assert coco["licenses"] == []
    assert coco["info"]["rights"]["statement"] == PDM_ASSERTION
    assert "val" not in (output / "yolo" / "dataset.yaml").read_text(encoding="utf-8")


def test_ingest_rejects_source_label_without_explicit_alias(tmp_path: Path) -> None:
    project = tmp_path / "project"
    with pytest.raises(DetectionIngestError, match="not allowed"):
        ingest_coco_archive(
            archive_path=_archive(tmp_path / "bad.zip", category_name="MysteryChip"),
            registry_path=_registry(tmp_path / "registry.yaml"),
            dataset_key="smd_components_raw",
            ontology_path=ONTOLOGY,
            output_root=project / "data" / "quarantine" / "out",
            project_root_path=project,
        )


def test_dainius_cannot_bypass_alias_with_canonical_class_name(tmp_path: Path) -> None:
    project = tmp_path / "project"
    with pytest.raises(DetectionIngestError, match="not allowed"):
        ingest_coco_archive(
            archive_path=_archive(
                tmp_path / "canonical-bypass.zip", category_name="smd_capacitor"
            ),
            registry_path=_registry(tmp_path / "registry.yaml"),
            dataset_key="smd_components_raw",
            ontology_path=ONTOLOGY,
            output_root=project / "data" / "quarantine" / "out",
            project_root_path=project,
        )


def test_ingest_rejects_exact_duplicate_source_images(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.zip"
    document = {
        "images": [
            {"id": 1, "file_name": "one.png", "width": 100, "height": 80},
            {"id": 2, "file_name": "two.png", "width": 100, "height": 80},
        ],
        "categories": [{"id": 1, "name": "Condensator"}],
        "annotations": [],
    }
    content = _png((1, 2, 3))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("train/_annotations.coco.json", json.dumps(document))
        archive.writestr("train/one.png", content)
        archive.writestr("train/two.png", content)

    with pytest.raises(DetectionIngestError, match="Exact duplicate"):
        project = tmp_path / "project"
        ingest_coco_archive(
            archive_path=path,
            registry_path=_registry(tmp_path / "registry.yaml"),
            dataset_key="smd_components_raw",
            ontology_path=ONTOLOGY,
            output_root=project / "data" / "quarantine" / "out",
            project_root_path=project,
        )


def test_ingest_rejects_windows_drive_archive_member(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    document = {"images": [], "categories": [], "annotations": []}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("train/_annotations.coco.json", json.dumps(document))
        archive.writestr("C:/escape.png", _png((1, 2, 3)))

    with pytest.raises(DetectionIngestError, match="Unsafe archive member"):
        project = tmp_path / "project"
        ingest_coco_archive(
            archive_path=path,
            registry_path=_registry(tmp_path / "registry.yaml"),
            dataset_key="smd_components_raw",
            ontology_path=ONTOLOGY,
            output_root=project / "data" / "quarantine" / "out",
            project_root_path=project,
        )


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    [
        ("MAX_ARCHIVE_MEMBERS", 1, "member limit"),
        ("MAX_TOTAL_UNCOMPRESSED_BYTES", 1, "total uncompressed-size"),
        ("MAX_COMPRESSION_RATIO", 0.5, "compression-ratio"),
        ("MAX_IMAGE_PIXELS", 10, "pixel limit"),
    ],
)
def test_ingest_resource_limits_fail_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: float,
    message: str,
) -> None:
    monkeypatch.setattr(detection_ingest_module, limit_name, limit_value)
    project = tmp_path / "project"
    target = project / "data" / "quarantine" / limit_name.lower()
    with pytest.raises(DetectionIngestError, match=message):
        ingest_coco_archive(
            archive_path=_archive(tmp_path / f"{limit_name}.zip"),
            registry_path=_registry(tmp_path / f"{limit_name}.yaml"),
            dataset_key="smd_components_raw",
            ontology_path=ONTOLOGY,
            output_root=target,
            project_root_path=project,
        )
    assert not target.exists()


def test_ingest_rejects_nonignored_output_root(tmp_path: Path) -> None:
    with pytest.raises(DetectionIngestError, match="ignored data directories"):
        ingest_coco_archive(
            archive_path=_archive(tmp_path / "source.zip"),
            registry_path=_registry(tmp_path / "registry.yaml"),
            dataset_key="smd_components_raw",
            ontology_path=ONTOLOGY,
            output_root=tmp_path / "public-output",
            project_root_path=tmp_path / "project",
        )


def test_stm32_class_requires_verified_specimen_and_part_number(tmp_path: Path) -> None:
    ontology_document = yaml.safe_load(ONTOLOGY.read_text(encoding="utf-8"))
    ontology_document["source_aliases"]["stm32_self_capture"] = {
        "STM32 verified specimen": "stm32_bare_ic"
    }
    ontology_path = tmp_path / "ontology.yaml"
    ontology_path.write_text(
        yaml.safe_dump(ontology_document, sort_keys=False), encoding="utf-8"
    )
    registry_path = tmp_path / "stm32-registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "stm32": {
                        "provider": "self_capture",
                        "source_id": "stm32_self_capture",
                        "dataset_version": "session-001",
                        "author": "project operator",
                        "source_url": "local://stm32/session-001",
                        "rights_statement": "Project-owned self capture",
                        "rights_url": "local://rights/session-001",
                        "ingest_split_policy": "bootstrap_train_only",
                        "allowed_source_labels": {
                            "STM32 verified specimen": "stm32_bare_ic"
                        },
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def archive(path: Path, *, part_number: str | None) -> Path:
        image = {"id": 1, "file_name": "chip.png", "width": 100, "height": 80}
        if part_number is not None:
            image["specimen_evidence"] = {
                "specimen_id": "stm32-chip-001",
                "evidence_type": "legible_top_marking",
                "verified_part_number": part_number,
            }
        document = {
            "images": [image],
            "categories": [{"id": 1, "name": "STM32 verified specimen"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}
            ],
        }
        with zipfile.ZipFile(path, "w") as output:
            output.writestr("train/_annotations.coco.json", json.dumps(document))
            output.writestr("train/chip.png", _png((4, 5, 6)))
        return path

    project = tmp_path / "project"
    with pytest.raises(DetectionIngestError, match="requires specimen_evidence"):
        ingest_coco_archive(
            archive_path=archive(tmp_path / "unverified.zip", part_number=None),
            registry_path=registry_path,
            dataset_key="stm32",
            ontology_path=ontology_path,
            output_root=project / "data" / "quarantine" / "unverified",
            project_root_path=project,
        )
    assert not (project / "data" / "quarantine" / "unverified").exists()
    with pytest.raises(DetectionIngestError, match="must identify an STM32 part"):
        ingest_coco_archive(
            archive_path=archive(tmp_path / "wrong-part.zip", part_number="ATMEGA328P"),
            registry_path=registry_path,
            dataset_key="stm32",
            ontology_path=ontology_path,
            output_root=project / "data" / "quarantine" / "wrong-part",
            project_root_path=project,
        )
    result = ingest_coco_archive(
        archive_path=archive(tmp_path / "verified.zip", part_number="STM32F103C8T6"),
        registry_path=registry_path,
        dataset_key="stm32",
        ontology_path=ontology_path,
        output_root=project / "data" / "quarantine" / "verified",
        project_root_path=project,
    )
    assert result["images"][0]["specimen_evidence"]["verified_part_number"] == "STM32F103C8T6"


def test_manual_source_record_preserves_pdm_assertion_without_downloading(tmp_path: Path) -> None:
    output_root = tmp_path / "data" / "raw" / "curated"
    result = download_curated(
        PROJECT_ROOT / "configs" / "datasets.curated.yaml",
        "smd_components_raw",
        output_root,
        force=False,
    )
    assert result["status"] == "MANUAL_AUTH_REQUIRED"
    assert result["source_id"] == "dainius_smdcomponents_v2"
    assert result["rights_statement"] == PDM_ASSERTION
    assert result["license"] == ""
    assert not (output_root / "smd_components_raw").exists()
