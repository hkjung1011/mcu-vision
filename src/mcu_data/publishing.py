from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .common import sha256_file, write_json


TEXT_SUFFIXES = {".csv", ".log", ".md", ".txt", ".yaml", ".yml"}
OMITTED_JSON_KEYS = {"nvidia_smi"}
NVIDIA_PROCESS_TABLE = re.compile(
    r"(?ms)^\+[-+]+\+\r?\n\| Processes:.*?^\+[-+]+\+\r?\n?"
)
WINDOWS_USER_HOME = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]Users[\\/][^\\/\s\"']+)"
)
FORMAL_MODELS = {"yolo11m", "yoloxs"}
FORMAL_SEEDS = {42, 43, 44}
FORMAL_COMMON = {
    "epochs": 100,
    "batch": 8,
    "imgsz": 640,
    "workers": 0,
    "amp": True,
    "fraction": 1.0,
    "multiscale_range": 0,
    "prediction_floor": 0.001,
    "nms_iou": 0.65,
    "class_agnostic_nms": False,
    "common_operating_confidence": 0.25,
    "common_match_iou": 0.5,
}
REQUIRED_DATASET_FIELDS = (
    "canonical_dataset_manifest_sha256",
    "class_map_sha256",
    "train_image_list_sha256",
    "val_image_list_sha256",
    "canonical_train_records_sha256",
    "canonical_val_records_sha256",
)
REQUIRED_SOURCE_FILES = (
    "run_manifest.json",
    "epoch_metrics.csv",
    "final_metrics.json",
    "latency.json",
    "gpu_summary.json",
)


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
    value, process_table_count = NVIDIA_PROCESS_TABLE.subn(
        "[NVIDIA-SMI PROCESS LIST OMITTED FROM PUBLISHED REPORT]\n",
        value,
    )
    changed = changed or bool(process_table_count)
    value, generic_user_count = WINDOWS_USER_HOME.subn("<USER_HOME>", value)
    changed = changed or bool(generic_user_count)
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
        value = json.loads(source.read_text(encoding="utf-8-sig"))
        cleaned, changed = _scrub_json_value(value, replacements)
        if changed:
            write_json(destination, cleaned)
        else:
            shutil.copy2(source, destination)
    elif source.suffix.lower() == ".jsonl":
        output_lines: list[str] = []
        for line in source.read_text(encoding="utf-8-sig").splitlines():
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


def _normalized_model(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _finite_fields(document: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    invalid = [
        field
        for field in fields
        if not isinstance(document.get(field), (int, float))
        or not math.isfinite(float(document[field]))
    ]
    if invalid:
        raise ValueError(f"Formal comparison {label} has invalid numeric fields: {invalid}")


def _numeric_digest(path: Path) -> str:
    values: list[list[Any]] = []
    if path.suffix.lower() in {".json", ".jsonl"}:
        documents = (
            [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if path.suffix.lower() == ".jsonl"
            else [json.loads(path.read_text(encoding="utf-8-sig"))]
        )

        def visit(value: Any, location: str) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, (int, float)):
                values.append([location, value])
            elif isinstance(value, dict):
                for key in sorted(value):
                    visit(value[key], f"{location}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, f"{location}[{index}]")

        for index, document in enumerate(documents):
            visit(document, f"document[{index}]")
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                for key, raw in row.items():
                    try:
                        number = float(raw) if raw not in (None, "") else None
                    except ValueError:
                        continue
                    if number is not None and math.isfinite(number):
                        values.append([f"row[{row_index}].{key}", number])
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_formal_comparison(
    comparison_dir: Path,
    *,
    require_local_originals: bool = False,
    _allow_missing_formal_record: bool = False,
) -> dict[str, Any]:
    """Independently rebuild the formal six-run gate from the published source bundle."""
    comparison_dir = comparison_dir.resolve()
    compatibility_path = comparison_dir / "protocol_compatibility.json"
    comparison_path = comparison_dir / "comparison.json"
    sources_path = comparison_dir / "sources_manifest.json"
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8-sig"))
    if not (
        compatibility.get("release_ready") is True
        and compatibility.get("comparable") is True
        and compatibility.get("critical_mismatches") == []
        and compatibility.get("release_blockers") == []
        and int(compatibility.get("run_count", -1)) == 6
    ):
        raise ValueError("Formal comparison compatibility claims are not an unblocked six-run PASS")
    provenance_path = comparison_dir / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    if provenance.get("status") != "PASS" or compatibility.get("run_provenance") != provenance:
        raise ValueError("Formal comparison provenance is missing, failed, or differs from compatibility")
    attestation_path = comparison_dir / "run_provenance_attestation.json"
    if provenance.get("mixed_commits") is True and (
        not attestation_path.is_file() or not isinstance(provenance.get("attestation"), dict)
    ):
        raise ValueError("Formal mixed-commit provenance attestation is missing or unbound")
    if provenance.get("mixed_commits") is True:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8-sig"))
        if sorted(attestation.get("allowed_commits", [])) != sorted(
            provenance["attestation"].get("allowed_commits", [])
        ):
            raise ValueError("Formal provenance attestation commit set differs from provenance")
    expectations = compatibility.get("release_expectations", {})
    if (
        {_normalized_model(value) for value in expectations.get("models", [])} != FORMAL_MODELS
        or {int(value) for value in expectations.get("seeds", [])} != FORMAL_SEEDS
        or int(expectations.get("runs", -1)) != 6
    ):
        raise ValueError("Formal comparison release expectations are not the exact model/seed matrix")

    comparison_rows = json.loads(comparison_path.read_text(encoding="utf-8-sig"))
    if not isinstance(comparison_rows, list) or len(comparison_rows) != 6:
        raise ValueError("Formal comparison.json must contain exactly six rows")
    row_ids = [str(row.get("run_id") or "") for row in comparison_rows if isinstance(row, dict)]
    if len(row_ids) != 6 or len(set(row_ids)) != 6:
        raise ValueError("Formal comparison run_id values must be six unique identifiers")

    sources = json.loads(sources_path.read_text(encoding="utf-8-sig"))
    records = sources.get("files", []) if isinstance(sources, dict) else []
    if not isinstance(records, list):
        raise ValueError("Formal sources_manifest.files must be a list")
    record_paths = [str(record.get("path") or "") for record in records if isinstance(record, dict)]
    if len(record_paths) != len(records) or len(set(record_paths)) != len(record_paths):
        raise ValueError("Formal source bundle paths must be unique")
    by_run_file: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Formal source bundle record must be an object")
        run_id = str(record.get("run_id") or "")
        relative = str(record.get("path") or "").replace("\\", "/")
        filename = Path(relative).name
        key = (run_id, filename)
        if key in by_run_file:
            raise ValueError(f"Duplicate formal source evidence: {run_id}/{filename}")
        path = (comparison_dir / relative).resolve()
        try:
            path.relative_to(comparison_dir)
        except ValueError as exc:
            raise ValueError("Formal source evidence escapes comparison directory") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        if str(record.get("published_sha256", "")).lower() != sha256_file(path):
            raise ValueError(f"Formal source published SHA-256 mismatch: {relative}")
        if int(record.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Formal source byte-size mismatch: {relative}")
        original_sha = str(record.get("source_original_sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", original_sha):
            raise ValueError(f"Formal source original SHA-256 is invalid: {relative}")
        by_run_file[key] = (record, path)

    manifests: dict[str, dict[str, Any]] = {}
    metrics_by_run: dict[str, dict[str, Any]] = {}
    critical_values: dict[str, set[str]] = {}
    actual_pairs: set[tuple[str, int]] = set()
    for run_id in row_ids:
        missing = [name for name in REQUIRED_SOURCE_FILES if (run_id, name) not in by_run_file]
        if missing:
            raise ValueError(f"Formal run evidence is incomplete for {run_id}: {missing}")
        manifest = json.loads(by_run_file[(run_id, "run_manifest.json")][1].read_text(encoding="utf-8-sig"))
        if (
            str(manifest.get("run_id")) != run_id
            or manifest.get("status") != "complete"
            or manifest.get("stage") == "smoke_not_comparable"
            or not str(manifest.get("best_checkpoint", {}).get("sha256") or "")
        ):
            raise ValueError(f"Formal run manifest is not a complete checkpointed run: {run_id}")
        protocol = manifest.get("protocol", {})
        model = _normalized_model(manifest.get("model"))
        seed = int(protocol.get("seed", -1))
        actual_pairs.add((model, seed))
        for field, expected in FORMAL_COMMON.items():
            defaults = {"fraction": 1.0, "multiscale_range": 0}
            actual = protocol.get(field, defaults.get(field))
            if actual != expected:
                raise ValueError(f"Formal protocol {field} mismatch for {run_id}")
        dataset = manifest.get("dataset", {})
        critical = {
            "train_annotation_sha256": dataset.get("train_annotation_sha256"),
            "val_annotation_sha256": dataset.get("val_annotation_sha256"),
            "protocol_config_sha256": manifest.get("protocol_config", {}).get("sha256"),
            **{field: dataset.get(field) for field in REQUIRED_DATASET_FIELDS},
        }
        for field, value in critical.items():
            if value in (None, ""):
                raise ValueError(f"Formal critical evidence {field} is missing for {run_id}")
            critical_values.setdefault(field, set()).add(json.dumps(value, sort_keys=True))

        with by_run_file[(run_id, "epoch_metrics.csv")][1].open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            epochs = list(csv.DictReader(handle))
        if len(epochs) != 100:
            raise ValueError(f"Formal epoch evidence must contain 100 rows: {run_id}")
        final = json.loads(by_run_file[(run_id, "final_metrics.json")][1].read_text(encoding="utf-8-sig"))
        metrics = final.get("metrics", final.get("metrics_common", {}))
        _finite_fields(
            metrics,
            ("ap50_95", "ap50", "ap75", "ar100", "precision", "recall", "f1", "tp", "fp", "fn"),
            f"metrics/{run_id}",
        )
        metrics_by_run[run_id] = metrics
        latency = json.loads(by_run_file[(run_id, "latency.json")][1].read_text(encoding="utf-8-sig"))
        _finite_fields(latency, ("e2e_p50_ms", "e2e_p95_ms", "sustained_fps"), f"latency/{run_id}")
        gpu = json.loads(by_run_file[(run_id, "gpu_summary.json")][1].read_text(encoding="utf-8-sig"))
        _finite_fields(gpu, ("peak_memory_used_mib",), f"gpu/{run_id}")
        manifests[run_id] = manifest

    expected_pairs = {(model, seed) for model in FORMAL_MODELS for seed in FORMAL_SEEDS}
    if actual_pairs != expected_pairs:
        raise ValueError("Formal source manifests do not form the exact model/seed matrix")
    mismatched = [field for field, values in critical_values.items() if len(values) != 1]
    if mismatched:
        raise ValueError(f"Formal critical fields differ across source manifests: {mismatched}")
    for row in comparison_rows:
        manifest = manifests[str(row["run_id"])]
        if (
            row.get("status") != "complete"
            or _normalized_model(row.get("model")) != _normalized_model(manifest.get("model"))
            or int(row.get("seed", -1)) != int(manifest.get("protocol", {}).get("seed", -2))
        ):
            raise ValueError("Formal comparison row differs from its source run manifest")
        for field in ("ap50_95", "ap50", "ap75", "ar100", "precision", "recall", "f1", "tp", "fp", "fn"):
            if row.get(field) != metrics_by_run[str(row["run_id"])].get(field):
                raise ValueError(f"Formal comparison row metric differs from source evidence: {field}")
    required_records = [
        (run_id, filename, *by_run_file[(run_id, filename)])
        for run_id in sorted(row_ids)
        for filename in REQUIRED_SOURCE_FILES
    ]
    bindings_path = comparison_dir / "local_source_bindings.json"
    bindings: dict[tuple[str, str], dict[str, Any]] = {}
    if bindings_path.is_file():
        binding_document = json.loads(bindings_path.read_text(encoding="utf-8-sig"))
        for binding in binding_document.get("files", []):
            key = (str(binding.get("run_id") or ""), str(binding.get("filename") or ""))
            if key in bindings:
                raise ValueError(f"Duplicate local source binding: {key}")
            bindings[key] = binding
    if require_local_originals and not bindings:
        raise ValueError("Formal promotion requires local original run-artifact bindings")

    source_chain = []
    for run_id, filename, record, published_path in required_records:
        published_numeric = _numeric_digest(published_path)
        if bindings:
            binding = bindings.get((run_id, filename))
            if not binding:
                raise ValueError(f"Local original binding is missing: {run_id}/{filename}")
            original_path = Path(str(binding.get("source_path") or "")).resolve()
            if not original_path.is_file():
                raise FileNotFoundError(f"Local original run artifact is missing: {original_path}")
            original_sha = sha256_file(original_path)
            if (
                original_sha != str(record.get("source_original_sha256", "")).lower()
                or original_sha != str(binding.get("source_original_sha256", "")).lower()
            ):
                raise ValueError(f"Local original SHA-256 differs for {run_id}/{filename}")
            if _numeric_digest(original_path) != published_numeric:
                raise ValueError(f"Published numeric evidence differs from local original: {run_id}/{filename}")
        source_chain.append(
            {
                "run_id": run_id,
                "filename": filename,
                "source_original_sha256": str(record["source_original_sha256"]).lower(),
                "published_sha256": str(record["published_sha256"]).lower(),
                "bytes": int(record["bytes"]),
                "numeric_digest": published_numeric,
            }
        )
    chain_payload = json.dumps(source_chain, separators=(",", ":"), sort_keys=True)
    formal_record = {
        "schema_version": 1,
        "status": "PASS",
        "run_count": 6,
        "model_seed_pairs": [list(pair) for pair in sorted(actual_pairs)],
        "protocol_compatibility_sha256": sha256_file(compatibility_path),
        "comparison_sha256": sha256_file(comparison_path),
        "sources_manifest_sha256": sha256_file(sources_path),
        "run_provenance_sha256": sha256_file(provenance_path),
        "run_provenance_attestation_sha256": (
            sha256_file(attestation_path) if attestation_path.is_file() else None
        ),
        "source_chain_sha256": hashlib.sha256(chain_payload.encode("utf-8")).hexdigest(),
        "source_chain": source_chain,
    }
    formal_path = comparison_dir / "formal_validation.json"
    if formal_path.is_file():
        recorded = json.loads(formal_path.read_text(encoding="utf-8-sig"))
        if recorded != formal_record:
            raise ValueError("Formal validation digest chain differs from comparison evidence")
    elif not _allow_missing_formal_record:
        raise FileNotFoundError("Formal comparison is missing formal_validation.json")
    return formal_record


def create_formal_validation(comparison_dir: Path) -> dict[str, Any]:
    record = validate_formal_comparison(
        comparison_dir,
        require_local_originals=True,
        _allow_missing_formal_record=True,
    )
    write_json(comparison_dir.resolve() / "formal_validation.json", record)
    return record


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
    provenance_path = comparison_dir / "run_provenance.json"
    for path in (compatibility_path, comparison_path, sources_manifest_path, provenance_path):
        if not path.exists():
            raise FileNotFoundError(f"Comparison evidence is missing: {path}")
    validate_formal_comparison(comparison_dir, require_local_originals=True)

    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    if not compatibility.get("release_ready", False):
        blockers = ", ".join(
            str(item.get("field")) for item in compatibility.get("release_blockers", [])
        )
        raise ValueError(
            "Comparison is not formal-release ready"
            + (f" ({blockers})" if blockers else "")
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "PASS" or compatibility.get("run_provenance") != provenance:
        raise ValueError("Comparison run provenance is missing, failed, or differs from compatibility")
    attestation_path = comparison_dir / "run_provenance_attestation.json"
    if provenance.get("mixed_commits") is True and not attestation_path.is_file():
        raise FileNotFoundError("Mixed-commit comparison is missing its provenance attestation")

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

    metrics_suffix = f"sources/{run_id}/final_metrics.json"
    metric_rows = [
        row
        for row in sources_manifest.get("files", [])
        if str(row.get("run_id")) == run_id
        and str(row.get("path", "")).replace("\\", "/").endswith(metrics_suffix)
    ]
    if len(metric_rows) != 1:
        raise ValueError(f"Comparison source bundle must contain this run's final metrics: {run_id}")
    native_final_metrics_sha256 = str(metric_rows[0].get("source_original_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", native_final_metrics_sha256):
        raise ValueError(f"Comparison final_metrics source SHA-256 is invalid: {run_id}")

    return {
        "comparison_id": comparison_dir.name,
        "protocol_compatibility_sha256": sha256_file(compatibility_path),
        "comparison_sha256": sha256_file(comparison_path),
        "sources_manifest_sha256": sha256_file(sources_manifest_path),
        "run_provenance_sha256": sha256_file(provenance_path),
        "run_provenance_attestation_sha256": (
            sha256_file(attestation_path) if attestation_path.is_file() else None
        ),
        "formal_validation_sha256": sha256_file(comparison_dir / "formal_validation.json"),
        "run_manifest_sha256": current_manifest_sha256,
        "native_final_metrics_sha256": native_final_metrics_sha256,
        "run_id": run_id,
    }
