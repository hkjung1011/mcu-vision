from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .common import sha256_file, write_json


TEXT_SUFFIXES = {".csv", ".log", ".md", ".txt", ".yaml", ".yml"}
OMITTED_JSON_KEYS = {"nvidia_smi"}


def _replacement_pairs(project_root: Path) -> list[tuple[str, str]]:
    candidates: list[tuple[Path, str]] = [(project_root.resolve(), "<PROJECT_ROOT>")]
    home = Path.home().resolve()
    if home != project_root.resolve():
        candidates.append((home, "<USER_HOME>"))
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        profile = Path(user_profile).resolve()
        if all(profile != path for path, _ in candidates):
            candidates.append((profile, "<USER_HOME>"))
    pairs: list[tuple[str, str]] = []
    for path, replacement in candidates:
        native = str(path)
        pairs.append((native, replacement))
        pairs.append((native.replace("\\", "/"), replacement))
        pairs.append((native.replace("\\", "\\\\"), replacement))
    return sorted(set(pairs), key=lambda item: len(item[0]), reverse=True)


def _scrub_text(value: str, replacements: list[tuple[str, str]]) -> tuple[str, bool]:
    changed = False
    for original, replacement in replacements:
        updated, count = re.subn(re.escape(original), replacement, value, flags=re.IGNORECASE)
        if count:
            value = updated
            changed = True
    return value, changed


def _scrub_json_value(
    value: Any,
    replacements: list[tuple[str, str]],
) -> tuple[Any, bool]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            if key in OMITTED_JSON_KEYS and item:
                result[key] = "<OMITTED_FROM_PUBLISHED_REPORT>"
                changed = True
                continue
            cleaned, item_changed = _scrub_json_value(item, replacements)
            result[key] = cleaned
            changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        result_list = []
        changed = False
        for item in value:
            cleaned, item_changed = _scrub_json_value(item, replacements)
            result_list.append(cleaned)
            changed = changed or item_changed
        return result_list, changed
    if isinstance(value, str):
        return _scrub_text(value, replacements)
    return value, False


def publish_evidence_file(
    source: Path,
    destination: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Copy evidence while removing local paths and volatile process listings.

    Numeric content is unchanged. Both the original local hash and the repository copy hash are
    returned so the publication transform is explicit and auditable.
    """
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_sha256 = sha256_file(source)
    replacements = _replacement_pairs(project_root)
    changed = False

    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        cleaned, changed = _scrub_json_value(value, replacements)
        if changed:
            write_json(destination, cleaned)
        else:
            shutil.copy2(source, destination)
    elif source.suffix.lower() == ".jsonl":
        output_lines: list[str] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            cleaned, line_changed = _scrub_json_value(value, replacements)
            changed = changed or line_changed
            output_lines.append(json.dumps(cleaned, ensure_ascii=False, sort_keys=True))
        if changed:
            destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8", newline="\n")
        else:
            shutil.copy2(source, destination)
    elif source.suffix.lower() in TEXT_SUFFIXES:
        with source.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            text = handle.read()
        cleaned_text, changed = _scrub_text(text, replacements)
        if changed:
            with destination.open("w", encoding="utf-8", newline="") as handle:
                handle.write(cleaned_text)
        else:
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)

    return {
        "source_original_sha256": original_sha256,
        "published_sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "sanitized_for_repository": changed,
    }


def validate_comparison_for_run(
    comparison_dir: Path,
    *,
    run_id: str,
    run_manifest_path: Path,
) -> dict[str, Any]:
    """Require a comparable comparison that contains this exact run manifest."""
    comparison_dir = comparison_dir.resolve()
    compatibility_path = comparison_dir / "protocol_compatibility.json"
    comparison_path = comparison_dir / "comparison.json"
    sources_manifest_path = comparison_dir / "sources_manifest.json"
    for path in (compatibility_path, comparison_path, sources_manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Comparison evidence is missing: {path}")

    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    if not compatibility.get("release_ready", False):
        blockers = ", ".join(
            str(item.get("field")) for item in compatibility.get("release_blockers", [])
        )
        raise ValueError(
            "Comparison is not formal-release ready"
            + (f" ({blockers})" if blockers else "")
        )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    rows = [row for row in comparison if str(row.get("run_id")) == run_id]
    if len(rows) != 1:
        raise ValueError(f"Comparison must contain run_id exactly once: {run_id}")
    if rows[0].get("status") != "complete":
        raise ValueError(f"Comparison row is not complete: {run_id}")

    sources_manifest = json.loads(sources_manifest_path.read_text(encoding="utf-8"))
    expected_suffix = f"sources/{run_id}/run_manifest.json"
    source_rows = [
        row
        for row in sources_manifest.get("files", [])
        if str(row.get("run_id")) == run_id
        and str(row.get("path", "")).replace("\\", "/").endswith(expected_suffix)
    ]
    if len(source_rows) != 1:
        raise ValueError(f"Comparison source bundle must contain this run manifest: {run_id}")
    current_manifest_sha256 = sha256_file(run_manifest_path)
    if source_rows[0].get("source_original_sha256") != current_manifest_sha256:
        raise ValueError("Run manifest changed after the comparison was generated")

    return {
        "comparison_id": comparison_dir.name,
        "protocol_compatibility_sha256": sha256_file(compatibility_path),
        "comparison_sha256": sha256_file(comparison_path),
        "sources_manifest_sha256": sha256_file(sources_manifest_path),
        "run_manifest_sha256": current_manifest_sha256,
        "run_id": run_id,
    }
