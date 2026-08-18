from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


PROVENANCE_SCHEMA_VERSION = 2
REQUIRED_SCOPE_VERSION = "formal-training-provenance-v2"
REQUIRED_IMPLEMENTATION_PATHS = (
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
COMPARATOR_IMPLEMENTATION_PATHS = (
    "src/mcu_data/reporting.py",
    "src/mcu_data/run_provenance.py",
)
COMMON_REQUIRED_MANIFEST_FIELDS = (
    "framework_version",
    "pretrained_checkpoint.sha256",
    "protocol_config.sha256",
    "environment.torch",
    "environment.torch_cuda_runtime",
    "environment.cudnn",
)
MODEL_REQUIRED_MANIFEST_FIELDS = {
    "yolo11m": COMMON_REQUIRED_MANIFEST_FIELDS,
    "yoloxs": COMMON_REQUIRED_MANIFEST_FIELDS
    + ("framework_commit", "experiment_config.sha256"),
}


def _normalized_model(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _nested(document: dict[str, Any], dotted: str) -> Any:
    value: Any = document
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=30,
    ).stdout.strip()


def _comparator_state(repository: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    try:
        head = _git(repository, "rev-parse", "HEAD")
        changed = _git(repository, "status", "--porcelain").splitlines()
        blobs = {
            path: _git(repository, "rev-parse", f"HEAD:{path}")
            for path in COMPARATOR_IMPLEMENTATION_PATHS
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "FAIL", "error": str(exc)}, [
            {"field": "comparator_git_state", "reason": str(exc)}
        ]
    if changed:
        blockers.append({"field": "comparator_git_clean", "changed_paths": changed})
    return {
        "status": "PASS" if not changed else "FAIL",
        "head": head,
        "clean": not changed,
        "implementation_blobs": blobs,
    }, blockers


def verify_run_provenance(
    runs: list[dict[str, Any]],
    *,
    repository: Path,
    attestation_path: Path | None,
) -> dict[str, Any]:
    """Fail closed on run inputs, framework/runtime identity, and comparator identity."""
    commits_by_run = {
        str(run["run_id"]): str(run.get("metadata", {}).get("git", {}).get("commit") or "")
        for run in runs
    }
    commits = sorted(set(commits_by_run.values()))
    blockers: list[dict[str, Any]] = []
    comparator, comparator_blockers = _comparator_state(repository)
    blockers.extend(comparator_blockers)
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
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "required_scope_version": REQUIRED_SCOPE_VERSION,
        "status": "PASS",
        "mixed_commits": len(commits) > 1,
        "commits_by_run": commits_by_run,
        "comparator": comparator,
        "attestation": None,
        "implementation_scope": list(REQUIRED_IMPLEMENTATION_PATHS),
        "implementation_blobs": {},
        "semantic_manifest_fields": {},
        "blockers": blockers,
    }
    try:
        for commit in commits:
            if commit:
                _git(repository, "cat-file", "-e", f"{commit}^{{commit}}")
    except (OSError, subprocess.SubprocessError) as exc:
        blockers.append({"field": "git_commit_object", "reason": str(exc)})

    attestation: dict[str, Any] = {}
    if len(commits) > 1:
        if attestation_path is None:
            blockers.append({"field": "mixed_git_commits", "reason": "attestation is required"})
        else:
            try:
                attestation = json.loads(attestation_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as exc:
                blockers.append({"field": "provenance_attestation", "reason": str(exc)})
            if attestation and (
                attestation.get("schema_version") != PROVENANCE_SCHEMA_VERSION
                or attestation.get("required_scope_version") != REQUIRED_SCOPE_VERSION
            ):
                blockers.append(
                    {"field": "provenance_attestation_schema", "reason": "required schema/scope mismatch"}
                )
            allowed_commits = sorted(str(value) for value in attestation.get("allowed_commits", []))
            result["attestation"] = {
                "path": attestation_path.name,
                "allowed_commits": allowed_commits,
                "rationale": attestation.get("rationale"),
            }
            if commits != allowed_commits:
                blockers.append(
                    {"field": "attested_commit_set", "expected": allowed_commits, "actual": commits}
                )

    declared_paths = tuple(attestation.get("implementation_paths", REQUIRED_IMPLEMENTATION_PATHS))
    missing_scope = sorted(set(REQUIRED_IMPLEMENTATION_PATHS) - set(declared_paths))
    if missing_scope:
        blockers.append({"field": "required_implementation_scope", "missing": missing_scope})
    implementation_paths = tuple(dict.fromkeys((*REQUIRED_IMPLEMENTATION_PATHS, *declared_paths)))
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
            if blobs != expected_blobs or expected.get("training_semantics_unchanged") is not True:
                blockers.append(
                    {"field": "implementation_blob", "path": path, "expected": expected_blobs, "actual": blobs}
                )
        unscoped = sorted(set(allowed_changes) - set(implementation_paths))
        if unscoped:
            blockers.append({"field": "unscoped_allowed_change", "paths": unscoped})
    except (OSError, subprocess.SubprocessError) as exc:
        blockers.append({"field": "git_object_verification", "reason": str(exc)})

    configured_fields = attestation.get("equal_manifest_fields_by_model", {})
    models = sorted({str(run.get("model")) for run in runs})
    for model in models:
        required_fields = MODEL_REQUIRED_MANIFEST_FIELDS.get(_normalized_model(model))
        if required_fields is None:
            blockers.append({"field": f"unsupported_provenance_model:{model}"})
            continue
        declared_fields = configured_fields.get(model, required_fields)
        missing_fields = sorted(set(required_fields) - set(declared_fields))
        if missing_fields:
            blockers.append(
                {"field": f"required_manifest_scope:{model}", "missing": missing_fields}
            )
        fields = tuple(dict.fromkeys((*required_fields, *declared_fields)))
        model_runs = [run for run in runs if str(run.get("model")) == model]
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
