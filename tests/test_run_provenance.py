from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mcu_data.run_provenance import (
    COMPARATOR_IMPLEMENTATION_PATHS,
    PROVENANCE_SCHEMA_VERSION,
    REQUIRED_IMPLEMENTATION_PATHS,
    REQUIRED_SCOPE_VERSION,
    verify_run_provenance,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Fixture User")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    for relative in dict.fromkeys((*REQUIRED_IMPLEMENTATION_PATHS, *COMPARATOR_IMPLEMENTATION_PATHS)):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "first")
    first = _git(repo, "rev-parse", "HEAD")
    publishing = repo / "src/mcu_data/publishing.py"
    first_blob = _git(repo, "rev-parse", f"{first}:src/mcu_data/publishing.py")
    publishing.write_text("fixture:publication-only-v2\n", encoding="utf-8")
    _git(repo, "commit", "-am", "publication only")
    second = _git(repo, "rev-parse", "HEAD")
    second_blob = _git(repo, "rev-parse", f"{second}:src/mcu_data/publishing.py")
    return repo, first, second, first_blob, second_blob


def _run(run_id: str, commit: str, *, model: str = "yolo11m") -> dict[str, object]:
    metadata: dict[str, object] = {
        "git": {"commit": commit, "dirty": False, "changed_paths": []},
        "framework_version": "8.4.120" if model == "yolo11m" else "0.3.0",
        "pretrained_checkpoint": {"sha256": "a" * 64},
        "protocol_config": {"sha256": "b" * 64},
        "environment": {
            "torch": "2.12.1+cu130",
            "torch_cuda_runtime": "13.0",
            "cudnn": 92000,
        },
    }
    if model == "YOLOX-S":
        metadata["framework_commit"] = "6" * 40
        metadata["experiment_config"] = {"sha256": "c" * 64}
    return {"run_id": run_id, "model": model, "metadata": metadata}


def _runs(first: str, second: str, *, model: str = "yolo11m") -> list[dict[str, object]]:
    return [_run("seed42", first, model=model), _run("seed43", second, model=model)]


def _attestation(first: str, second: str, first_blob: str, second_blob: str) -> dict:
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "required_scope_version": REQUIRED_SCOPE_VERSION,
        "allowed_commits": sorted([first, second]),
        "implementation_paths": list(REQUIRED_IMPLEMENTATION_PATHS),
        "allowed_changes": {
            "src/mcu_data/publishing.py": {
                "blobs_by_commit": {first: first_blob, second: second_blob},
                "training_semantics_unchanged": True,
            }
        },
    }


def test_mixed_commits_fail_closed_without_attestation(tmp_path: Path) -> None:
    repo, first, second, _, _ = _repo(tmp_path)
    result = verify_run_provenance(_runs(first, second), repository=repo, attestation_path=None)
    assert result["status"] == "FAIL"
    assert any(item["field"] == "mixed_git_commits" for item in result["blockers"])


def test_exact_publication_only_attestation_passes(tmp_path: Path) -> None:
    repo, first, second, first_blob, second_blob = _repo(tmp_path)
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(_attestation(first, second, first_blob, second_blob)), encoding="utf-8"
    )
    result = verify_run_provenance(
        _runs(first, second), repository=repo, attestation_path=attestation
    )
    assert result["status"] == "PASS"
    assert result["comparator"]["clean"] is True
    assert set(result["comparator"]["implementation_blobs"]) == {
        "src/mcu_data/reporting.py",
        "src/mcu_data/run_provenance.py",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("framework_version", "changed"),
        ("pretrained_checkpoint.sha256", "c" * 64),
        ("protocol_config.sha256", "d" * 64),
        ("environment.torch", "changed"),
        ("environment.torch_cuda_runtime", "changed"),
        ("environment.cudnn", 1),
    ],
)
def test_same_commit_semantic_mismatch_is_blocked(
    tmp_path: Path, field: str, value: object
) -> None:
    repo, first, _, _, _ = _repo(tmp_path)
    runs = _runs(first, first)
    target = runs[1]["metadata"]  # type: ignore[index]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]  # type: ignore[index]
    target[parts[-1]] = value  # type: ignore[index]
    result = verify_run_provenance(runs, repository=repo, attestation_path=None)
    assert result["status"] == "FAIL"
    assert any(field in item["field"] for item in result["blockers"])


@pytest.mark.parametrize("field", ["framework_commit", "experiment_config.sha256"])
def test_yolox_framework_and_config_mismatch_are_blocked(tmp_path: Path, field: str) -> None:
    repo, first, _, _, _ = _repo(tmp_path)
    runs = _runs(first, first, model="YOLOX-S")
    target = runs[1]["metadata"]  # type: ignore[index]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]  # type: ignore[index]
    target[parts[-1]] = "f" * 40 if field == "framework_commit" else "f" * 64  # type: ignore[index]
    result = verify_run_provenance(runs, repository=repo, attestation_path=None)
    assert result["status"] == "FAIL"
    assert any(field in item["field"] for item in result["blockers"])


def test_attestation_cannot_remove_required_scope(tmp_path: Path) -> None:
    repo, first, second, first_blob, second_blob = _repo(tmp_path)
    document = _attestation(first, second, first_blob, second_blob)
    document["implementation_paths"] = ["src/mcu_data/publishing.py"]
    document["equal_manifest_fields_by_model"] = {"yolo11m": ["framework_version"]}
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps(document), encoding="utf-8")
    result = verify_run_provenance(
        _runs(first, second), repository=repo, attestation_path=attestation
    )
    fields = {item["field"] for item in result["blockers"]}
    assert "required_implementation_scope" in fields
    assert "required_manifest_scope:yolo11m" in fields


def test_dirty_comparator_and_recorded_run_are_blocked(tmp_path: Path) -> None:
    repo, first, _, _, _ = _repo(tmp_path)
    runs = _runs(first, first)
    runs[0]["metadata"]["git"]["dirty"] = True  # type: ignore[index]
    (repo / "untracked.txt").write_text("dirty", encoding="utf-8")
    result = verify_run_provenance(runs, repository=repo, attestation_path=None)
    fields = {item["field"] for item in result["blockers"]}
    assert "run_git_state" in fields
    assert "comparator_git_clean" in fields
