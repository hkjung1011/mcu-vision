from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from PIL import Image

from mcu_data.common import sha256_file, write_json
from mcu_data.contracts import load_json_object, load_ontology
from mcu_data.manual_seed import (
    ManualSeedError,
    prepare_manual_seed_task,
    validate_manual_seed_task,
)


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "project"
    ontology_path = root / "configs" / "classes.smd_v1.yaml"
    _write_yaml(
        ontology_path,
        {
            "schema_version": 1,
            "ontology_id": "mcu_smd_detection_v1",
            "classes": {
                "smd_capacitor": {"id": 0},
                "smd_resistor": {"id": 1},
                "smd_diode": {"id": 2},
                "smd_transistor": {"id": 3},
                "stm32_dev_board": {"id": 4},
                "stm32_bare_ic": {"id": 5},
            },
            "source_aliases": {},
        },
    )
    config_path = root / "configs" / "sources.wikimedia.yaml"
    _write_yaml(config_path, {"schema_version": 2, "sources": {}})
    config_sha = sha256_file(config_path)
    image_root = root / "data" / "quarantine" / "wikimedia_commons_v2" / "images"
    route_root = image_root / "stm32_dev_board"
    route_root.mkdir(parents=True)
    page_ids = (101, 102)
    records: list[dict] = []
    collector_rows: list[dict] = []
    for index, page_id in enumerate(page_ids, start=1):
        image_path = route_root / f"commons_{page_id}_board-{index}.jpg"
        Image.new("RGB", (64 + index, 48 + index), (20, 130, 70)).save(
            image_path, format="JPEG"
        )
        revision = 9000 + page_id
        record = {
            "source_page_id": page_id,
            "source_page_title": f"File:Board {page_id}.jpg",
            "source_page_url": f"https://commons.wikimedia.org/wiki/File:Board_{page_id}.jpg",
            "source_page_revision_id": revision,
            "source_page_revision_url": (
                f"https://commons.wikimedia.org/wiki/File:Board_{page_id}.jpg?oldid={revision}"
            ),
            "source_image_timestamp": "2026-08-19T00:00:00Z",
            "source_image_sha1": hashlib.sha1(
                image_path.read_bytes(), usedforsecurity=False
            ).hexdigest(),
            "source_metadata_snapshot_sha256": f"{page_id:064x}",
            "sha256": sha256_file(image_path),
            "bytes": image_path.stat().st_size,
            "width": 64 + index,
            "height": 48 + index,
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "artist": f"Author {index}",
            "collection_route": "stm32_dev_board",
            "visual_decision": "CANDIDATE_STM32_DEV_BOARD_NOT_APPROVED",
            "reason": "Development board candidate; review pending.",
        }
        records.append(record)
        collector_rows.append(
            {
                **record,
                "collector_schema_version": 2,
                "collector_config_sha256": config_sha,
                "status": "ACCEPTED",
                "class_name": "stm32_dev_board",
                "qa_status": "PENDING_HUMAN_REVIEW",
                "training_eligibility": "PROHIBITED_PENDING_HUMAN_REVIEW",
                "source_page_latest_revision_id": revision,
            }
        )
    probe_path = root / "data" / "manifests" / "probe.json"
    write_json(
        probe_path,
        {
            "schema_version": 2,
            "status": "QUARANTINE_REVIEWED_INSUFFICIENT",
            "source": {
                "collector_schema_version": 2,
                "collection_config_sha256": config_sha,
                "tracked_images": False,
                "quarantine_only": True,
                "source_page_revision_ids_recorded": True,
                "source_metadata_snapshot_sha256_recorded": True,
                "post_download_source_revalidated": True,
            },
            "dev_board_grouping": {
                "groups": [
                    {"group_id": "family-a", "source_page_ids": [101]},
                    {"group_id": "family-b", "source_page_ids": [102]},
                ]
            },
            "review_records": records,
            "training_use": {"allowed": False},
        },
    )
    collector_path = (
        root
        / "data"
        / "quarantine"
        / "wikimedia_commons_v2"
        / "evidence"
        / "stm32_dev_board.sources.jsonl"
    )
    collector_path.parent.mkdir(parents=True)
    collector_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in collector_rows),
        encoding="utf-8",
        newline="\n",
    )
    proposal_path = root / "configs" / "annotation" / "proposals.yaml"
    from mcu_data.contracts import load_ontology

    ontology_sha = load_ontology(ontology_path).sha256
    _write_yaml(
        proposal_path,
        {
            "schema_version": "mcu.manual-seed-proposals.v1",
            "proposal_id": "fixture",
            "status": "DRAFT_REQUIRES_HUMAN_REVIEW",
            "source_probe": {
                "path": "data/manifests/probe.json",
                "sha256": sha256_file(probe_path),
                "schema_version": 2,
            },
            "collection_route": "stm32_dev_board",
            "class_name": "stm32_dev_board",
            "ontology_sha256": ontology_sha,
            "claim_boundary": {
                "human_review_completed": False,
                "ground_truth": False,
                "training_use_allowed": False,
            },
            "proposals": [
                {
                    "source_page_id": page_id,
                    "leakage_group_id": f"family-{'a' if page_id == 101 else 'b'}",
                    "bbox_xywh_pixels": [0, 0, 64 + index, 48 + index],
                    "attributes": {"occluded": False, "truncated": False},
                    "review_state": "PROPOSED_REQUIRES_HUMAN_REVIEW",
                }
                for index, page_id in enumerate(page_ids, start=1)
            ],
        },
    )
    return {
        "root": root,
        "probe": probe_path,
        "collector": collector_path,
        "config": config_path,
        "proposal": proposal_path,
        "ontology": ontology_path,
        "image_root": image_root,
        "output": root / "data" / "staging" / "manual-seed-task",
    }


def _prepare(files: dict[str, Path]) -> dict:
    return prepare_manual_seed_task(
        probe_path=files["probe"],
        collector_records_path=files["collector"],
        collection_config_path=files["config"],
        proposal_path=files["proposal"],
        image_root=files["image_root"],
        ontology_path=files["ontology"],
        output_dir=files["output"],
        run_id="fixture-manual-seed",
        project_root=files["root"],
    )


def _rebind_proposal_to_probe(files: dict[str, Path]) -> None:
    proposal = yaml.safe_load(files["proposal"].read_text(encoding="utf-8"))
    proposal["source_probe"]["sha256"] = sha256_file(files["probe"])
    _write_yaml(files["proposal"], proposal)


def _rebind_review_to_task(files: dict[str, Path]) -> None:
    review_path = files["output"] / "review_manifest.template.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["pending_run_manifest_sha256"] = sha256_file(
        files["output"] / "run_manifest.json"
    )
    write_json(review_path, review)


def _tamper_reference(
    files: dict[str, Path], mutate: Callable[[dict], None]
) -> None:
    reference_path = files["output"] / "reference.coco.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    mutate(reference)
    write_json(reference_path, reference)
    task_path = files["output"] / "run_manifest.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["reference"]["sha256"] = sha256_file(reference_path)
    write_json(task_path, task)
    _rebind_review_to_task(files)


def _tamper_attribution(
    files: dict[str, Path], mutate: Callable[[dict], None]
) -> None:
    attribution_path = files["output"] / "attribution.json"
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    mutate(attribution)
    write_json(attribution_path, attribution)
    task_path = files["output"] / "run_manifest.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["artifacts"]["attribution"]["sha256"] = sha256_file(attribution_path)
    write_json(task_path, task)
    _rebind_review_to_task(files)


def test_prepare_manual_seed_is_revision_bound_and_never_training_approved(
    tmp_path: Path,
) -> None:
    files = _fixture(tmp_path)
    result = _prepare(files)
    output = files["output"]

    assert result["schema_version"] == "mcu.manual-seed-task.v1"
    assert result["preparation_status"] == "PASS_REVIEW_BUNDLE_ONLY"
    assert result["annotation_state"] == "PENDING_HUMAN_REVIEW"
    assert result["training_use"]["allowed"] is False
    assert result["training_use"]["approved_images"] == 0
    assert result["protocol"]["proposed_annotations_are_ground_truth"] is False
    assert result["source_provenance"]["collector_record_count"] == 2
    assert result["reference"]["images"] == 2
    assert result["reference"]["annotations"] == 2

    reference = json.loads((output / "reference.coco.json").read_text(encoding="utf-8"))
    assert len(reference["categories"]) == 6
    assert len(reference["images"]) == 2
    assert len(reference["annotations"]) == 2
    assert all(row["review_state"] == "PROPOSED_REQUIRES_HUMAN_REVIEW" for row in reference["annotations"])
    review = json.loads(
        (output / "review_manifest.template.json").read_text(encoding="utf-8")
    )
    assert review["status"] == "DRAFT_UNRESOLVED_DO_NOT_PROMOTE"
    assert all(row["disposition"] == "UNRESOLVED" for row in review["images"])
    with zipfile.ZipFile(output / "cvat_images.zip") as archive:
        assert sorted(archive.namelist()) == sorted(row["file_name"] for row in reference["images"])
    assert len(list((output / "previews").glob("*.proposal.jpg"))) == 2
    validation = validate_manual_seed_task(
        task_manifest_path=output / "run_manifest.json",
        ontology_path=files["ontology"],
        project_root=files["root"],
    )
    assert validation["status"] == "PASS_REVIEW_BUNDLE_ONLY"
    assert validation["images"] == 2
    assert validation["proposed_annotations"] == 2
    assert validation["human_review_complete"] is False
    assert validation["training_use_allowed"] is False


def test_prepare_manual_seed_rejects_tampered_runtime_image(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    image = next((files["image_root"] / "stm32_dev_board").glob("commons_101_*.jpg"))
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(ManualSeedError, match="byte count differs"):
        _prepare(files)


def test_prepare_manual_seed_rejects_incomplete_proposal_coverage(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    proposal = yaml.safe_load(files["proposal"].read_text(encoding="utf-8"))
    proposal["proposals"] = proposal["proposals"][:1]
    _write_yaml(files["proposal"], proposal)
    with pytest.raises(ManualSeedError, match="coverage is incomplete"):
        _prepare(files)


def test_prepare_manual_seed_rejects_collector_probe_mismatch(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    rows = [json.loads(line) for line in files["collector"].read_text().splitlines()]
    rows[0]["artist"] = "Different author"
    files["collector"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n"
    )
    with pytest.raises(ManualSeedError, match="Collector/probe field mismatch"):
        _prepare(files)


def test_prepare_manual_seed_rejects_out_of_bounds_bbox(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    proposal = yaml.safe_load(files["proposal"].read_text(encoding="utf-8"))
    proposal["proposals"][0]["bbox_xywh_pixels"] = [0, 0, 999, 999]
    _write_yaml(files["proposal"], proposal)
    with pytest.raises(ManualSeedError, match="exceeds image bounds"):
        _prepare(files)


def test_prepare_manual_seed_refuses_existing_destination(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    files["output"].mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        _prepare(files)


def test_validate_manual_seed_rejects_unlisted_file(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    _prepare(files)
    (files["output"] / "private-note.txt").write_text("not allowlisted", encoding="utf-8")
    with pytest.raises(ManualSeedError, match="unlisted files"):
        validate_manual_seed_task(
            task_manifest_path=files["output"] / "run_manifest.json",
            ontology_path=files["ontology"],
            project_root=files["root"],
        )


def test_validate_manual_seed_rejects_tampered_reference(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    _prepare(files)
    reference = files["output"] / "reference.coco.json"
    value = json.loads(reference.read_text(encoding="utf-8"))
    value["annotations"][0]["bbox"][0] = 1
    write_json(reference, value)
    with pytest.raises(ManualSeedError, match="reference COCO is missing or differs"):
        validate_manual_seed_task(
            task_manifest_path=files["output"] / "run_manifest.json",
            ontology_path=files["ontology"],
            project_root=files["root"],
        )


def test_prepare_manual_seed_rejects_duplicate_json_probe_key(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    text = files["probe"].read_text(encoding="utf-8")
    needle = '"schema_version": 2,'
    assert needle in text
    files["probe"].write_text(
        text.replace(needle, '"schema_version": 2,\n  "schema_version": 2,', 1),
        encoding="utf-8",
        newline="\n",
    )
    _rebind_proposal_to_probe(files)
    with pytest.raises(ManualSeedError, match="duplicate JSON key"):
        _prepare(files)


def test_validate_manual_seed_rejects_nonfinite_json_number(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    _prepare(files)
    task_path = files["output"] / "run_manifest.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["training_use"]["approved_images"] = float("nan")
    write_json(task_path, task)
    with pytest.raises(ManualSeedError, match="non-finite number"):
        validate_manual_seed_task(
            task_manifest_path=task_path,
            ontology_path=files["ontology"],
            project_root=files["root"],
        )


@pytest.mark.parametrize("payload", ['"source_page_id": 101', '"sentinel": NaN'])
def test_prepare_manual_seed_rejects_unsafe_jsonl(
    tmp_path: Path, payload: str
) -> None:
    files = _fixture(tmp_path)
    lines = files["collector"].read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0][:-1] + f", {payload}}}"
    files["collector"].write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    match = "duplicate JSON key" if "source_page_id" in payload else "non-finite number"
    with pytest.raises(ManualSeedError, match=match):
        _prepare(files)


@pytest.mark.parametrize("yaml_target", ["proposal", "ontology"])
def test_prepare_manual_seed_rejects_duplicate_yaml_key(
    tmp_path: Path, yaml_target: str
) -> None:
    files = _fixture(tmp_path)
    path = files[yaml_target]
    duplicate = (
        "status: DRAFT_REQUIRES_HUMAN_REVIEW"
        if yaml_target == "proposal"
        else "schema_version: 1"
    )
    path.write_text(
        path.read_text(encoding="utf-8") + f"\n{duplicate}\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ManualSeedError, match="Duplicate YAML key"):
        _prepare(files)


@pytest.mark.parametrize(
    ("license_name", "license_url"),
    [
        ("CC BY 4.0", "https://creativecommons.org/licenses/by-nc/4.0/"),
        ("CC BY-ND 4.0", "https://creativecommons.org/licenses/by-nd/4.0/"),
    ],
)
def test_prepare_manual_seed_rejects_unapproved_license_pair(
    tmp_path: Path, license_name: str, license_url: str
) -> None:
    files = _fixture(tmp_path)
    probe = json.loads(files["probe"].read_text(encoding="utf-8"))
    probe["review_records"][0]["license"] = license_name
    probe["review_records"][0]["license_url"] = license_url
    write_json(files["probe"], probe)
    rows = [
        json.loads(line)
        for line in files["collector"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["license"] = license_name
    rows[0]["license_url"] = license_url
    files["collector"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    _rebind_proposal_to_probe(files)
    with pytest.raises(ManualSeedError, match="label/URL pair is not explicitly approved"):
        _prepare(files)


def test_prepare_manual_seed_rejects_source_sha1_mismatch(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    probe = json.loads(files["probe"].read_text(encoding="utf-8"))
    probe["review_records"][0]["source_image_sha1"] = "0" * 40
    write_json(files["probe"], probe)
    rows = [
        json.loads(line)
        for line in files["collector"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["source_image_sha1"] = "0" * 40
    files["collector"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    _rebind_proposal_to_probe(files)
    with pytest.raises(ManualSeedError, match="image SHA-1 differs"):
        _prepare(files)


def test_prepare_manual_seed_rejects_non_utc_source_timestamp(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    probe = json.loads(files["probe"].read_text(encoding="utf-8"))
    probe["review_records"][0]["source_image_timestamp"] = "2026-08-19T09:00:00+09:00"
    write_json(files["probe"], probe)
    rows = [
        json.loads(line)
        for line in files["collector"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["source_image_timestamp"] = "2026-08-19T09:00:00+09:00"
    files["collector"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    _rebind_proposal_to_probe(files)
    with pytest.raises(ManualSeedError, match="timezone-aware UTC"):
        _prepare(files)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["info"].__setitem__("ground_truth", True),
            "reference info differs",
        ),
        (
            lambda value: value["licenses"][0].__setitem__(
                "url", "https://creativecommons.org/licenses/by-nc/4.0/"
            ),
            "reference licenses differ",
        ),
        (
            lambda value: value["images"][0].__setitem__("license", 999),
            "reference image records differ",
        ),
        (
            lambda value: value["images"][0].__setitem__(
                "source_page_revision_url", "https://commons.wikimedia.org/wiki/File:fake"
            ),
            "reference image records differ",
        ),
        (
            lambda value: value["annotations"][0].__setitem__("id", 999),
            "reference proposals differ",
        ),
        (
            lambda value: value["annotations"][0].__setitem__(
                "area", value["annotations"][0]["area"] + 1
            ),
            "reference proposals differ",
        ),
    ],
    ids=("info", "license", "image-license", "image-source", "annotation-id", "area"),
)
def test_validate_manual_seed_rejects_resigned_reference_semantic_tamper(
    tmp_path: Path, mutate: Callable[[dict], None], match: str
) -> None:
    files = _fixture(tmp_path)
    _prepare(files)
    _tamper_reference(files, mutate)
    with pytest.raises(ManualSeedError, match=match):
        validate_manual_seed_task(
            task_manifest_path=files["output"] / "run_manifest.json",
            ontology_path=files["ontology"],
            project_root=files["root"],
        )


def test_validate_manual_seed_rejects_resigned_attribution_tamper(
    tmp_path: Path,
) -> None:
    files = _fixture(tmp_path)
    _prepare(files)
    _tamper_attribution(
        files,
        lambda value: value["records"][0].__setitem__("artist", "Impostor"),
    )
    with pytest.raises(ManualSeedError, match="attribution records differ"):
        validate_manual_seed_task(
            task_manifest_path=files["output"] / "run_manifest.json",
            ontology_path=files["ontology"],
            project_root=files["root"],
        )


def test_validate_manual_seed_rejects_non_utc_attribution_timestamp(
    tmp_path: Path,
) -> None:
    files = _fixture(tmp_path)
    _prepare(files)
    _tamper_attribution(
        files,
        lambda value: value.__setitem__(
            "generated_at_utc", "2026-08-19T09:00:00+09:00"
        ),
    )
    with pytest.raises(ManualSeedError, match="timezone-aware UTC"):
        validate_manual_seed_task(
            task_manifest_path=files["output"] / "run_manifest.json",
            ontology_path=files["ontology"],
            project_root=files["root"],
        )


def test_validate_manual_seed_rejects_duplicate_reference_json_key(
    tmp_path: Path,
) -> None:
    files = _fixture(tmp_path)
    _prepare(files)
    reference_path = files["output"] / "reference.coco.json"
    text = reference_path.read_text(encoding="utf-8")
    reference_path.write_text(
        '{"info": {},' + text.lstrip()[1:], encoding="utf-8", newline="\n"
    )
    task_path = files["output"] / "run_manifest.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["reference"]["sha256"] = sha256_file(reference_path)
    write_json(task_path, task)
    _rebind_review_to_task(files)
    with pytest.raises(ManualSeedError, match="duplicate JSON key"):
        validate_manual_seed_task(
            task_manifest_path=task_path,
            ontology_path=files["ontology"],
            project_root=files["root"],
        )


def test_validate_manual_seed_rejects_duplicate_review_json_key(tmp_path: Path) -> None:
    files = _fixture(tmp_path)
    _prepare(files)
    review_path = files["output"] / "review_manifest.template.json"
    text = review_path.read_text(encoding="utf-8")
    review_path.write_text(
        '{"status": "DRAFT_UNRESOLVED_DO_NOT_PROMOTE",' + text.lstrip()[1:],
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ManualSeedError, match="duplicate JSON key"):
        validate_manual_seed_task(
            task_manifest_path=files["output"] / "run_manifest.json",
            ontology_path=files["ontology"],
            project_root=files["root"],
        )


def test_public_manual_seed_preparation_record_is_review_only_and_source_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    record = load_json_object(
        root
        / "data"
        / "manifests"
        / "wikimedia_stm32_dev_board.manual-seed-review-preparation.json",
        label="public manual-seed preparation record",
    )
    source = record["source_evidence"]
    claim = record["claim_boundary"]
    assert record["status"] == "AWAITING_HUMAN_CVAT_REVIEW"
    assert record["candidate_images"] == 11
    assert record["proposed_annotations"] == 11
    assert record["conservative_leakage_groups"] == 6
    assert record["human_approved_images"] == 0
    assert record["human_approved_annotations"] == 0
    assert claim == {
        "human_review_complete": False,
        "proposals_are_ground_truth": False,
        "training_use_allowed": False,
        "formal_evaluation_allowed": False,
    }
    for path_field, hash_field in (
        ("probe_path", "probe_sha256"),
        ("proposal_path", "proposal_sha256"),
        ("ontology_path", "ontology_sha256"),
    ):
        path = root / source[path_field]
        assert path.is_file()
        assert sha256_file(path) == source[hash_field]
    ontology = load_ontology(root / source["ontology_path"])
    assert ontology.sha256 == source["ontology_sha256"]
    local = record["local_review_bundle"]
    assert local["tracked_by_git"] is False
    assert local["file_count"] == 27
    assert local["bytes"] == 16_573_098
    for field in (
        "task_manifest_sha256",
        "reference_coco_sha256",
        "cvat_images_zip_sha256",
        "attribution_sha256",
        "unresolved_review_template_sha256",
    ):
        assert len(local[field]) == 64
        int(local[field], 16)

    local_root = root / local["path"]
    if local_root.is_dir():
        validation = validate_manual_seed_task(
            task_manifest_path=local_root / "run_manifest.json",
            ontology_path=root / source["ontology_path"],
            project_root=root,
        )
        assert validation["task_manifest_sha256"] == local["task_manifest_sha256"]
        assert validation["source_probe_sha256"] == source["probe_sha256"]
        assert validation["proposal_sha256"] == source["proposal_sha256"]
        assert validation["ontology_sha256"] == source["ontology_sha256"]
        files = [path for path in local_root.rglob("*") if path.is_file()]
        assert len(files) == local["file_count"]
        assert sum(path.stat().st_size for path in files) == local["bytes"]
        assert sha256_file(local_root / "reference.coco.json") == local[
            "reference_coco_sha256"
        ]
        assert sha256_file(local_root / "cvat_images.zip") == local[
            "cvat_images_zip_sha256"
        ]
        assert sha256_file(local_root / "attribution.json") == local[
            "attribution_sha256"
        ]
        assert sha256_file(local_root / "review_manifest.template.json") == local[
            "unresolved_review_template_sha256"
        ]
