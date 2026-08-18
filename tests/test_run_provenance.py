from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mcu_data.run_provenance import verify_run_provenance


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    repo = tmp_path / "repository"
    path = repo / "train.py"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Fixture User")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    path.write_text("TRAIN = 1\n", encoding="utf-8")
    _git(repo, "add", "train.py")
    _git(repo, "commit", "-m", "first")
    first = _git(repo, "rev-parse", "HEAD")
    first_blob = _git(repo, "rev-parse", f"{first}:train.py")
    path.write_text("TRAIN = 1\nPUBLICATION = 2\n", encoding="utf-8")
    _git(repo, "commit", "-am", "publication only")
    second = _git(repo, "rev-parse", "HEAD")
    second_blob = _git(repo, "rev-parse", f"{second}:train.py")
    return repo, first, second, first_blob, second_blob


def _runs(first: str, second: str) -> list[dict[str, object]]:
    def run(run_id: str, commit: str) -> dict[str, object]:
        return {
            "run_id": run_id,
            "model": "detector",
            "metadata": {
                "git": {"commit": commit, "dirty": False, "changed_paths": []},
                "framework_version": "1.0",
                "pretrained_checkpoint": {"sha256": "a" * 64},
                "protocol_config": {"sha256": "b" * 64},
            },
        }
    return [run("seed42", first), run("seed43", second)]


def test_mixed_commits_fail_closed_without_attestation(tmp_path: Path) -> None:
    repo, first, second, _, _ = _repo(tmp_path)
    result = verify_run_provenance(
        _runs(first, second), repository=repo, attestation_path=None
    )
    assert result["status"] == "FAIL"
    assert any(item["field"] == "mixed_git_commits" for item in result["blockers"])


def test_exact_publication_only_attestation_passes_and_tamper_fails(tmp_path: Path) -> None:
    repo, first, second, first_blob, second_blob = _repo(tmp_path)
    attestation = tmp_path / "attestation.json"
    document = {
        "schema_version": 1,
        "allowed_commits": sorted([first, second]),
        "implementation_paths": ["train.py"],
        "allowed_changes": {
            "train.py": {
                "blobs_by_commit": {first: first_blob, second: second_blob},
                "training_semantics_unchanged": True,
            }
        },
        "equal_manifest_fields_by_model": [
            "framework_version",
            "pretrained_checkpoint.sha256",
            "protocol_config.sha256",
        ],
    }
    attestation.write_text(json.dumps(document), encoding="utf-8")
    runs = _runs(first, second)
    assert verify_run_provenance(
        runs, repository=repo, attestation_path=attestation
    )["status"] == "PASS"

    runs[1]["metadata"]["pretrained_checkpoint"]["sha256"] = "c" * 64  # type: ignore[index]
    mismatch = verify_run_provenance(runs, repository=repo, attestation_path=attestation)
    assert mismatch["status"] == "FAIL"
    assert any("pretrained_checkpoint" in item["field"] for item in mismatch["blockers"])

    document["allowed_changes"]["train.py"]["blobs_by_commit"][second] = "0" * 40
    attestation.write_text(json.dumps(document), encoding="utf-8")
    blob_tamper = verify_run_provenance(
        _runs(first, second), repository=repo, attestation_path=attestation
    )
    assert any(item["field"] == "implementation_blob" for item in blob_tamper["blockers"])


def test_dirty_recorded_run_is_blocked_even_on_one_commit(tmp_path: Path) -> None:
    repo, first, _, _, _ = _repo(tmp_path)
    runs = _runs(first, first)
    runs[0]["metadata"]["git"]["dirty"] = True  # type: ignore[index]
    result = verify_run_provenance(runs, repository=repo, attestation_path=None)
    assert result["status"] == "FAIL"
    assert result["blockers"][0]["field"] == "run_git_state"
