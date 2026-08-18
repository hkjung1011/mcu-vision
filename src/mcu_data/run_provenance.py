from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_IMPLEMENTATION_PATHS = (
    "scripts/run_compare_seeds.ps1",
    "scripts/train_yolo11_logged.py",
    "scripts/train_yolox_logged.py",
    "configs/yolox_s_micropcb.py",
    "src/mcu_data/common.py",
    "src/mcu_data/dataset_evidence.py",
    "src/mcu_data/methodology.py",
    "src/mcu_data/publishing.py",
    "src/mcu_data/reporting.py",
    "src/mcu_data/runlog.py",
    "src/mcu_data/yolox_metrics.py",
)


def _nested(document: dict[str, Any], dotted: str) -> Any:
    value: Any = document
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=30,
    )
    return completed.stdout.strip()


def verify_run_provenance(
    runs: list[dict[str, Any]],
    *,
    repository: Path,
    attestation_path: Path | None,
) -> dict[str, Any]:
    """Fail-closed verification of mixed training commits and semantic inputs."""
    commits_by_run = {
        str(run["run_id"]): str(run.get("metadata", {}).get("git", {}).get("commit") or "")
        for run in runs
    }
    commits = sorted(set(commits_by_run.values()))
    blockers: list[dict[str, Any]] = []
    invalid_git_state = {
        str(run["run_id"]): run.get("metadata", {}).get("git")
        for run in runs
        if not commits_by_run[str(run["run_id"])]
        or run.get("metadata", {}).get("git", {}).get("dirty") is not False
        or run.get("metadata", {}).get("git", {}).get("changed_paths") != []
    }
    if invalid_git_state:
        blockers.append({"field": "run_git_state", "actual": invalid_git_state})

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "mixed_commits": len(commits) > 1,
        "commits_by_run": commits_by_run,
        "attestation": None,
        "implementation_scope": list(DEFAULT_IMPLEMENTATION_PATHS),
        "implementation_blobs": {},
        "semantic_manifest_fields": {},
        "blockers": blockers,
    }
    try:
        for commit in commits:
            if commit:
                _git(repository, "cat-file", "-e", f"{commit}^{{commit}}")
    except (subprocess.SubprocessError, OSError) as exc:
        blockers.append({"field": "git_commit_object", "reason": str(exc)})
    if len(commits) <= 1:
        result["status"] = "PASS" if not blockers else "FAIL"
        return result
    if attestation_path is None:
        blockers.append({"field": "mixed_git_commits", "reason": "attestation is required"})
        result["status"] = "FAIL"
        return result

    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append({"field": "provenance_attestation", "reason": str(exc)})
        result["status"] = "FAIL"
        return result
    if not isinstance(attestation, dict) or attestation.get("schema_version") != 1:
        blockers.append({"field": "provenance_attestation", "reason": "schema_version must be 1"})
        result["status"] = "FAIL"
        return result
    allowed_commits = sorted(str(value) for value in attestation.get("allowed_commits", []))
    result["attestation"] = {
        "path": attestation_path.name,
        "allowed_commits": allowed_commits,
        "rationale": attestation.get("rationale"),
        "known_limitation": attestation.get("known_limitation"),
    }
    if commits != allowed_commits:
        blockers.append({"field": "attested_commit_set", "expected": allowed_commits, "actual": commits})

    implementation_paths = tuple(attestation.get("implementation_paths") or DEFAULT_IMPLEMENTATION_PATHS)
    result["implementation_scope"] = list(implementation_paths)
    allowed_changes = attestation.get("allowed_changes", {})
    try:
        for path in implementation_paths:
            blobs = {commit: _git(repository, "rev-parse", f"{commit}:{path}") for commit in commits}
            result["implementation_blobs"][path] = blobs
            if len(set(blobs.values())) == 1:
                continue
            expected = allowed_changes.get(path)
            expected_blobs = expected.get("blobs_by_commit", {}) if isinstance(expected, dict) else {}
            if blobs != expected_blobs or not expected.get("training_semantics_unchanged"):
                blockers.append(
                    {"field": "implementation_blob", "path": path, "expected": expected_blobs, "actual": blobs}
                )
        unscoped = sorted(set(allowed_changes) - set(implementation_paths))
        if unscoped:
            blockers.append({"field": "unscoped_allowed_change", "paths": unscoped})
    except (subprocess.SubprocessError, OSError) as exc:
        blockers.append({"field": "git_object_verification", "reason": str(exc)})

    configured_fields = attestation.get(
        "equal_manifest_fields_by_model",
        ["framework_version", "pretrained_checkpoint.sha256", "protocol_config.sha256"],
    )
    models = sorted({str(run.get("model")) for run in runs})
    for model in models:
        model_runs = [run for run in runs if str(run.get("model")) == model]
        fields = (
            configured_fields.get(model, [])
            if isinstance(configured_fields, dict)
            else configured_fields
        )
        if not fields:
            blockers.append({"field": f"semantic_manifest_fields:{model}", "reason": "no fields attested"})
        for field in fields:
            values = {
                str(run["run_id"]): _nested(run.get("metadata", {}), str(field))
                for run in model_runs
            }
            result["semantic_manifest_fields"][f"{model}:{field}"] = values
            serialized = {json.dumps(value, sort_keys=True) for value in values.values()}
            if None in values.values() or len(serialized) != 1:
                blockers.append({"field": f"semantic_manifest:{model}:{field}", "actual": values})

    result["status"] = "PASS" if not blockers else "FAIL"
    return result
