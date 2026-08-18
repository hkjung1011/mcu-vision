from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from mcu_data.autolabel import (
    AUTOLABEL_SOURCE_SCHEMA,
    validate_calibration_binding,
    validate_source_binding,
    validate_teacher_binding,
)
from mcu_data.common import sha256_file
from mcu_data.contracts import ContractError, canonical_sha256, load_ontology


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "classes.smd_v1.yaml"


def _source_manifest(path: Path, source: Path, *, role: str = "unlabeled_train") -> Path:
    ontology = load_ontology(ONTOLOGY_PATH)
    image = source / "chip.png"
    row = {
        "path": "chip.png",
        "sha256": sha256_file(image),
        "width": 100,
        "height": 80,
        "role": role,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": AUTOLABEL_SOURCE_SCHEMA,
                "dataset_id": "self_capture_session_001",
                "role": role,
                "ontology_sha256": ontology.sha256,
                "images": [row],
                "image_list_sha256": canonical_sha256([row]),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_source_binding_rejects_validation_and_accepts_hash_bound_train(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (100, 80), color=(1, 2, 3)).save(source / "chip.png")
    ontology = load_ontology(ONTOLOGY_PATH)

    with pytest.raises(ContractError, match="unlabeled_train"):
        validate_source_binding(
            source_manifest_path=_source_manifest(
                tmp_path / "forbidden.json", source, role="gold_validation_locked"
            ),
            source=source,
            ontology=ontology,
        )

    binding = validate_source_binding(
        source_manifest_path=_source_manifest(tmp_path / "source.json", source),
        source=source,
        ontology=ontology,
    )
    assert binding["role"] == "unlabeled_train"
    assert binding["image_count"] == 1

    Image.new("RGB", (100, 80), color=(9, 9, 9)).save(source / "chip.png")
    with pytest.raises(ContractError, match="hash differs"):
        validate_source_binding(
            source_manifest_path=tmp_path / "source.json",
            source=source,
            ontology=ontology,
        )


def test_teacher_and_gold_calibration_are_bound_to_checkpoint_and_ontology(
    tmp_path: Path,
) -> None:
    ontology = load_ontology(ONTOLOGY_PATH)
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"synthetic-test-checkpoint")
    teacher_manifest = tmp_path / "teacher.json"
    teacher_manifest.write_text(
        json.dumps(
            {
                "training_annotation_state": "reviewed_train",
                "training_dataset_id": "reviewed_smd_v1",
                "checkpoint_sha256": sha256_file(checkpoint),
                "ontology_sha256": ontology.sha256,
                "class_map": {str(key): name for key, name in ontology.classes_by_id.items()},
            }
        ),
        encoding="utf-8",
    )
    teacher = validate_teacher_binding(
        teacher_manifest_path=teacher_manifest,
        model_path=checkpoint,
        ontology=ontology,
    )
    assert teacher["checkpoint_sha256"] == sha256_file(checkpoint)

    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "calibration_binding": {
                    "source_role": "gold_validation_locked",
                    "teacher_sha256": teacher["checkpoint_sha256"],
                    "ontology_sha256": ontology.sha256,
                    "image_list_sha256": "a" * 64,
                },
                "pseudo_label_calibration_by_class": [],
            }
        ),
        encoding="utf-8",
    )
    binding = validate_calibration_binding(
        calibration_path=calibration,
        teacher_sha256=teacher["checkpoint_sha256"],
        ontology=ontology,
    )
    assert binding["source_role"] == "gold_validation_locked"

    document = json.loads(calibration.read_text(encoding="utf-8"))
    document["calibration_binding"]["source_role"] = "val"
    calibration.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ContractError, match="gold_validation_locked"):
        validate_calibration_binding(
            calibration_path=calibration,
            teacher_sha256=teacher["checkpoint_sha256"],
            ontology=ontology,
        )
