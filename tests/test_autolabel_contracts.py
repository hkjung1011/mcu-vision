from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from mcu_data.autolabel import (
    AUTOLABEL_SOURCE_SCHEMA,
    _calibration_thresholds,
    validate_calibration_binding,
    validate_source_binding,
    validate_teacher_binding,
)
from mcu_data.common import load_yaml, sha256_file
from mcu_data.contracts import (
    ContractError,
    canonical_sha256,
    load_ontology,
    load_ontology_display_sidecar,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = PROJECT_ROOT / "configs" / "classes.smd_v1.yaml"
ONTOLOGY_DISPLAY_KO_PATH = PROJECT_ROOT / "configs" / "classes.smd_v1.display.ko.yaml"


def test_korean_display_sidecar_is_presentation_only_and_exactly_bound() -> None:
    ontology = load_ontology(ONTOLOGY_PATH)
    display = load_ontology_display_sidecar(ONTOLOGY_PATH, ONTOLOGY_DISPLAY_KO_PATH)
    assert list(display) == ontology.names
    assert all(
        set(entry).issubset({"display_name", "description", "terminology_note"})
        for entry in display.values()
    )
    assert display["stm32_bare_ic"]["display_name"] == "STM32 단품 IC 패키지"


def test_korean_display_sidecar_rejects_canonical_fields(tmp_path: Path) -> None:
    document = yaml.safe_load(ONTOLOGY_DISPLAY_KO_PATH.read_text(encoding="utf-8"))
    document["classes"]["smd_capacitor"]["id"] = 0
    tampered = tmp_path / "display.ko.yaml"
    tampered.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ContractError, match="may contain only"):
        load_ontology_display_sidecar(ONTOLOGY_PATH, tampered)


def _image_row(
    source: Path, *, name: str = "chip.png", image_id: str = "self-001-chip", role: str
) -> dict[str, object]:
    image = source / name
    return {
        "image_id": image_id,
        "path": name,
        "sha256": sha256_file(image),
        "width": 100,
        "height": 80,
        "role": role,
    }


def _trusted_registry(
    project: Path,
    *,
    unlabeled_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]] | None = None,
    test_rows: list[dict[str, object]] | None = None,
) -> Path:
    ontology = load_ontology(ONTOLOGY_PATH)
    evidence_path = project / "data" / "evidence" / "self_capture.locked.json"
    evidence_path.parent.mkdir(parents=True)
    splits = {}
    for role, rows in (
        ("unlabeled_train", unlabeled_rows),
        ("gold_validation_locked", validation_rows or []),
        ("test_locked", test_rows or []),
    ):
        normalized = sorted(rows, key=lambda row: (str(row["path"]), str(row["image_id"])))
        splits[role] = {
            "images": normalized,
            "image_list_sha256": canonical_sha256(normalized),
        }
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "mcu.locked-split-evidence.v1",
                "dataset_id": "self_capture_session_001",
                "ontology_sha256": ontology.sha256,
                "splits": splits,
            }
        ),
        encoding="utf-8",
    )
    entry = {
        "status": "APPROVED",
        "ontology_sha256": ontology.sha256,
        "locked_split_evidence": {
            "path": "data/evidence/self_capture.locked.json",
            "sha256": sha256_file(evidence_path),
        },
    }
    registry_path = project / "configs" / "data_trust_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "mcu.data-trust-registry.v1",
                "registry_id": "test-project-registry",
                "datasets": {"self_capture_session_001": entry},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return registry_path


def _source_manifest(
    path: Path,
    source: Path,
    registry_path: Path,
    *,
    role: str = "unlabeled_train",
    rows: list[dict[str, object]] | None = None,
) -> Path:
    ontology = load_ontology(ONTOLOGY_PATH)
    registry = load_yaml(registry_path)
    entry = registry["datasets"]["self_capture_session_001"]
    binding_rows = rows or [_image_row(source, role=role)]
    path.write_text(
        json.dumps(
            {
                "schema_version": AUTOLABEL_SOURCE_SCHEMA,
                "dataset_id": "self_capture_session_001",
                "role": role,
                "ontology_sha256": ontology.sha256,
                "trusted_registry_id": registry["registry_id"],
                "trusted_registry_sha256": sha256_file(registry_path),
                "trusted_dataset_record_sha256": canonical_sha256(entry),
                "locked_split_evidence_sha256": entry["locked_split_evidence"]["sha256"],
                "images": binding_rows,
                "image_list_sha256": canonical_sha256(binding_rows),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_default_project_registry_has_no_self_attested_approval(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (100, 80), color=(1, 2, 3)).save(source / "chip.png")
    ontology = load_ontology(ONTOLOGY_PATH)
    registry_path = PROJECT_ROOT / "configs" / "data_trust_registry.yaml"
    registry = load_yaml(registry_path)
    assert registry["datasets"] == {}
    row = _image_row(source, role="unlabeled_train")
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": AUTOLABEL_SOURCE_SCHEMA,
                "dataset_id": "caller_claimed_dataset",
                "role": "unlabeled_train",
                "ontology_sha256": ontology.sha256,
                "trusted_registry_id": registry["registry_id"],
                "trusted_registry_sha256": sha256_file(registry_path),
                "images": [row],
                "image_list_sha256": canonical_sha256([row]),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="is not approved in the project trusted registry"):
        validate_source_binding(
            source_manifest_path=manifest,
            source=source,
            ontology=ontology,
            trusted_registry_path=registry_path,
        )


def test_source_binding_rejects_validation_and_accepts_hash_bound_train(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (100, 80), color=(1, 2, 3)).save(source / "chip.png")
    ontology = load_ontology(ONTOLOGY_PATH)
    approved_row = _image_row(source, role="unlabeled_train")
    registry = _trusted_registry(
        tmp_path / "project", unlabeled_rows=[approved_row]
    )

    with pytest.raises(ContractError, match="unlabeled_train"):
        validate_source_binding(
            source_manifest_path=_source_manifest(
                tmp_path / "forbidden.json",
                source,
                registry,
                role="gold_validation_locked",
            ),
            source=source,
            ontology=ontology,
            trusted_registry_path=registry,
        )

    binding = validate_source_binding(
        source_manifest_path=_source_manifest(tmp_path / "source.json", source, registry),
        source=source,
        ontology=ontology,
        trusted_registry_path=registry,
    )
    assert binding["role"] == "unlabeled_train"
    assert binding["image_count"] == 1

    Image.new("RGB", (100, 80), color=(9, 9, 9)).save(source / "chip.png")
    with pytest.raises(ContractError, match="hash differs"):
        validate_source_binding(
            source_manifest_path=tmp_path / "source.json",
            source=source,
            ontology=ontology,
            trusted_registry_path=registry,
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


def test_source_binding_cannot_redeclare_locked_gold_sha_as_unlabeled(
    tmp_path: Path,
) -> None:
    source = tmp_path / "attack"
    source.mkdir()
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(source / "gold.png")
    approved = tmp_path / "approved"
    approved.mkdir()
    Image.new("RGB", (100, 80), color=(30, 20, 10)).save(approved / "train.png")
    approved_row = _image_row(
        approved, name="train.png", image_id="approved-train", role="unlabeled_train"
    )
    gold_row = _image_row(
        source,
        name="gold.png",
        image_id="locked-gold",
        role="gold_validation_locked",
    )
    registry = _trusted_registry(
        tmp_path / "project",
        unlabeled_rows=[approved_row],
        validation_rows=[gold_row],
    )
    attack_row = {**gold_row, "role": "unlabeled_train"}
    manifest = _source_manifest(
        tmp_path / "attack.json", source, registry, rows=[attack_row]
    )

    with pytest.raises(ContractError, match="redeclare locked validation/test"):
        validate_source_binding(
            source_manifest_path=manifest,
            source=source,
            ontology=load_ontology(ONTOLOGY_PATH),
            trusted_registry_path=registry,
        )


def test_locked_split_evidence_rejects_cross_role_image_id_reuse(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (100, 80), color=(1, 2, 3)).save(source / "train.png")
    Image.new("RGB", (100, 80), color=(4, 5, 6)).save(source / "gold.png")
    train_row = _image_row(
        source, name="train.png", image_id="shared-id", role="unlabeled_train"
    )
    gold_row = _image_row(
        source, name="gold.png", image_id="shared-id", role="gold_validation_locked"
    )
    registry = _trusted_registry(
        tmp_path, unlabeled_rows=[train_row], validation_rows=[gold_row]
    )
    manifest = _source_manifest(
        tmp_path / "source.json", source, registry, rows=[train_row]
    )
    with pytest.raises(ContractError, match="reuses image_id"):
        validate_source_binding(
            source_manifest_path=manifest,
            source=source / "train.png",
            ontology=load_ontology(ONTOLOGY_PATH),
            trusted_registry_path=registry,
        )


@pytest.mark.parametrize(
    "confidence", [-0.01, 1.01, float("inf"), float("nan"), "not-a-number"]
)
def test_calibration_confidence_must_be_finite_unit_interval(
    tmp_path: Path, confidence: object
) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "pseudo_label_calibration_by_class": [
                    {"category_name": "smd_capacitor", "confidence": confidence}
                ],
                "pseudo_label_calibration": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="finite and in"):
        _calibration_thresholds(path)
