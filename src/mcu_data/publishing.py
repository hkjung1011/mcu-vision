from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
from pathlib import Path
from typing import Any

from .common import safe_stem, sha256_file, write_json


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
BUNDLED_SOURCE_FILES = (
    "terminal.log",
    "run_manifest.json",
    "epoch_metrics.csv",
    "epoch_metrics.jsonl",
    "epoch_metrics_extra.jsonl",
    "final_metrics.json",
    "per_class_metrics.csv",
    "latency.json",
    "latency_samples.csv",
    "gpu_summary.json",
)
REQUIRED_PROTOCOL_ARTIFACTS = (
    "protocol_snapshot.yaml",
    "protocol_rationale.csv",
    "protocol_references.json",
    "experiment_methodology.md",
    "parameter_rationale.md",
    "protocol_rationale.png",
    "formal_execution_status.json",
)
REQUIRED_FORMAL_USER_ARTIFACTS = (
    "comparison.csv",
    "comparison.json",
    "aggregate_comparison.csv",
    "aggregate_comparison.json",
    "comparison_terminal.txt",
    "terminal_summary.png",
    "comparison_dashboard.png",
    "training_curves.png",
    "aggregate_comparison.png",
    "experiment_report.md",
    "experiment_methodology.md",
    "parameter_rationale.md",
    "protocol_snapshot.yaml",
    "protocol_rationale.csv",
    "protocol_rationale.png",
    "protocol_references.json",
    "protocol_artifacts.json",
    "formal_execution_status.json",
    "ubuntu_handoff.md",
    "onnx_split_evaluation.md",
    "protocol_compatibility.json",
    "run_provenance.json",
    "sources_manifest.json",
    "evidence_manifest.json",
)
FORMAL_DERIVED_IMAGE_ARTIFACTS = (
    "terminal_summary.png",
    "comparison_dashboard.png",
    "training_curves.png",
    "aggregate_comparison.png",
    "protocol_rationale.png",
)
AGGREGATE_METRICS = (
    "ap50_95",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar100",
    "precision",
    "recall",
    "f1",
    "latency_p50_ms",
    "latency_p95_ms",
    "fps",
    "peak_gpu_memory_mib",
    "train_elapsed_s",
)
FORMAL_PRIVATE_FILES = {"local_source_bindings.json"}
WEIGHT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".engine",
    ".safetensors",
    ".weights",
    ".tflite",
    ".bin",
    ".pb",
    ".torchscript",
}
IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".svg",
}


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


def _published_content_equivalent(
    original: Path,
    published: Path,
    *,
    publication_project_root: Path,
) -> bool:
    """Reapply the sole approved publication transform and require an exact result."""
    suffix = original.suffix.lower()
    replacements = _replacement_pairs(publication_project_root)
    if suffix == ".json":
        expected, _ = _scrub_json_value(
            json.loads(original.read_text(encoding="utf-8-sig")),
            replacements,
        )
        return expected == json.loads(published.read_text(encoding="utf-8-sig"))
    if suffix == ".jsonl":
        original_rows = [
            json.loads(line)
            for line in original.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        published_rows = [
            json.loads(line)
            for line in published.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        expected_rows = [_scrub_json_value(row, replacements)[0] for row in original_rows]
        return expected_rows == published_rows
    if suffix in TEXT_SUFFIXES:
        expected, _ = _scrub_text(
            original.read_text(encoding="utf-8", errors="replace"),
            replacements,
        )
        return expected == published.read_text(encoding="utf-8", errors="replace")
    return sha256_file(original) == sha256_file(published)


def _contained_manifest_path(root: Path, relative: Any, *, label: str) -> tuple[str, Path]:
    normalized = str(relative or "").replace("\\", "/")
    if not normalized or normalized.startswith("/"):
        raise ValueError(f"{label} path is empty or absolute: {normalized!r}")
    resolved = (root / normalized).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes comparison directory: {normalized}") from exc
    canonical = resolved.relative_to(root).as_posix()
    if canonical != normalized:
        raise ValueError(f"{label} path is not canonical: {normalized}")
    return canonical, resolved


def _validate_file_record(
    root: Path,
    record: dict[str, Any],
    *,
    label: str,
) -> tuple[str, Path]:
    relative, path = _contained_manifest_path(root, record.get("path"), label=label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} file is missing: {relative}")
    expected_sha = str(record.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or sha256_file(path) != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch: {relative}")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise ValueError(f"{label} byte-size mismatch: {relative}")
    return relative, path


def _validate_formal_artifact_inventory(
    comparison_dir: Path,
    source_bundle_paths: set[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evidence_path = comparison_dir / "evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
    if evidence.get("generative_ai_used_for_images") is not False:
        raise ValueError(
            "Formal evidence_manifest.json requires exact generative_ai_used_for_images=false"
        )
    if evidence.get("local_absolute_paths_included") is not False:
        raise ValueError("Formal evidence manifest must declare local_absolute_paths_included=false")
    records: dict[str, dict[str, Any]] = {}
    paths_by_group: dict[str, set[str]] = {"sources": set(), "derived_images": set()}
    for group in ("sources", "derived_images"):
        values = evidence.get(group)
        if not isinstance(values, list):
            raise ValueError(f"Formal evidence_manifest.{group} must be a list")
        for raw in values:
            if not isinstance(raw, dict):
                raise ValueError(f"Formal evidence_manifest.{group} record must be an object")
            relative, _ = _validate_file_record(
                comparison_dir,
                raw,
                label=f"formal evidence_manifest.{group}",
            )
            if relative in records:
                raise ValueError(f"Duplicate formal artifact inventory path: {relative}")
            records[relative] = {"path": relative, "bytes": int(raw["bytes"]), "sha256": raw["sha256"], "kind": group}
            paths_by_group[group].add(relative)
    forbidden_self_records = {"evidence_manifest.json", "formal_validation.json"} & records.keys()
    if forbidden_self_records:
        raise ValueError(
            "Formal evidence inventory must exclude self/cyclic records: "
            f"{sorted(forbidden_self_records)}"
        )
    expected_sources = (
        set(REQUIRED_FORMAL_USER_ARTIFACTS)
        - {"evidence_manifest.json"}
        - set(FORMAL_DERIVED_IMAGE_ARTIFACTS)
    ) | source_bundle_paths
    if (comparison_dir / "run_provenance_attestation.json").is_file():
        expected_sources.add("run_provenance_attestation.json")
    expected_images = set(FORMAL_DERIVED_IMAGE_ARTIFACTS)
    if paths_by_group["sources"] != expected_sources:
        missing = sorted(expected_sources - paths_by_group["sources"])
        extra = sorted(paths_by_group["sources"] - expected_sources)
        raise ValueError(
            "Formal evidence source inventory must exactly match generated artifacts and the "
            f"verified source manifest: missing={missing}, extra={extra}"
        )
    if paths_by_group["derived_images"] != expected_images:
        missing = sorted(expected_images - paths_by_group["derived_images"])
        extra = sorted(paths_by_group["derived_images"] - expected_images)
        raise ValueError(
            "Formal derived-image inventory must contain exactly the five approved renderer "
            f"outputs: missing={missing}, extra={extra}"
        )
    return evidence, records


def _validate_protocol_artifacts(
    comparison_dir: Path,
    execution_status: dict[str, Any],
) -> dict[str, Any]:
    path = comparison_dir / "protocol_artifacts.json"
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("generative_ai_used_for_images") is not False:
        raise ValueError(
            "Formal protocol_artifacts.json requires exact generative_ai_used_for_images=false"
        )
    if document.get("execution_status") != execution_status:
        raise ValueError("Formal protocol artifacts do not bind the execution-status overlay")
    records = document.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("Formal protocol_artifacts.artifacts must be a list")
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("Formal protocol artifact record must be an object")
        relative, artifact_path = _contained_manifest_path(
            comparison_dir,
            raw.get("path"),
            label="formal protocol artifact",
        )
        if relative in seen:
            raise ValueError(f"Duplicate formal protocol artifact path: {relative}")
        seen.add(relative)
        if relative not in REQUIRED_PROTOCOL_ARTIFACTS:
            raise ValueError(f"Unrecognized formal protocol artifact path: {relative}")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Formal protocol artifact is missing: {relative}")
        expected_sha = str(raw.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or sha256_file(artifact_path) != expected_sha:
            raise ValueError(f"Formal protocol artifact SHA-256 mismatch: {relative}")
        if int(raw.get("bytes", -1)) != artifact_path.stat().st_size:
            raise ValueError(f"Formal protocol artifact byte-size mismatch: {relative}")
    missing = sorted(set(REQUIRED_PROTOCOL_ARTIFACTS) - seen)
    if missing:
        raise FileNotFoundError(f"Formal protocol artifacts are incomplete: {missing}")
    return document


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
        relative, path = _contained_manifest_path(
            comparison_dir,
            record.get("path"),
            label="formal source bundle",
        )
        filename = Path(relative).name
        expected_relative = (Path("sources") / safe_stem(run_id) / filename).as_posix()
        if (
            run_id not in row_ids
            or filename not in BUNDLED_SOURCE_FILES
            or relative != expected_relative
        ):
            raise ValueError(
                "Formal source bundle contains an unrecognized run artifact path: "
                f"{run_id}/{relative}"
            )
        key = (run_id, filename)
        if key in by_run_file:
            raise ValueError(f"Duplicate formal source evidence: {run_id}/{filename}")
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
    latency_by_run: dict[str, dict[str, Any]] = {}
    gpu_by_run: dict[str, dict[str, Any]] = {}
    epochs_by_run: dict[str, list[dict[str, str]]] = {}
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
        try:
            epoch_numbers = [int(row.get("epoch", "")) for row in epochs]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Formal epoch evidence has invalid epoch numbers: {run_id}") from exc
        if epoch_numbers != list(range(1, 101)):
            raise ValueError(
                f"Formal epoch evidence must be the ordered, unique 1..100 sequence: {run_id}"
            )
        epochs_by_run[run_id] = epochs
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
        latency_by_run[run_id] = latency
        gpu = json.loads(by_run_file[(run_id, "gpu_summary.json")][1].read_text(encoding="utf-8-sig"))
        _finite_fields(gpu, ("peak_memory_used_mib",), f"gpu/{run_id}")
        gpu_by_run[run_id] = gpu
        manifests[run_id] = manifest

    expected_pairs = {(model, seed) for model in FORMAL_MODELS for seed in FORMAL_SEEDS}
    if actual_pairs != expected_pairs:
        raise ValueError("Formal source manifests do not form the exact model/seed matrix")
    mismatched = [field for field, values in critical_values.items() if len(values) != 1]
    if mismatched:
        raise ValueError(f"Formal critical fields differ across source manifests: {mismatched}")
    protocol_hashes = {
        str(manifest.get("protocol_config", {}).get("sha256") or "").lower()
        for manifest in manifests.values()
    }
    protocol_snapshot_path = comparison_dir / "protocol_snapshot.yaml"
    if len(protocol_hashes) != 1 or sha256_file(protocol_snapshot_path) not in protocol_hashes:
        raise ValueError(
            "Formal protocol_snapshot.yaml differs from the protocol_config SHA-256 bound by runs"
        )
    for row in comparison_rows:
        run_id = str(row["run_id"])
        manifest = manifests[run_id]
        if (
            row.get("status") != "complete"
            or _normalized_model(row.get("model")) != _normalized_model(manifest.get("model"))
            or int(row.get("seed", -1)) != int(manifest.get("protocol", {}).get("seed", -2))
        ):
            raise ValueError("Formal comparison row differs from its source run manifest")
        for field in (
            "ap50_95",
            "ap50",
            "ap75",
            "ap_small",
            "ap_medium",
            "ap_large",
            "ar100",
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn",
            "best_f1",
            "best_f1_confidence",
        ):
            if row.get(field) != metrics_by_run[run_id].get(field):
                raise ValueError(f"Formal comparison row metric differs from source evidence: {field}")
        last_epoch = epochs_by_run[run_id][-1]
        elapsed = last_epoch.get("elapsed_s")
        source_values = {
            "params": manifest.get("model_details", {}).get("parameters"),
            "checkpoint_mib": manifest.get("best_checkpoint", {}).get("mib"),
            "latency_p50_ms": latency_by_run[run_id]["e2e_p50_ms"],
            "latency_p95_ms": latency_by_run[run_id]["e2e_p95_ms"],
            "fps": latency_by_run[run_id]["sustained_fps"],
            "system_peak_gpu_memory_mib": gpu_by_run[run_id]["peak_memory_used_mib"],
            "train_elapsed_s": float(elapsed) if elapsed not in (None, "") else None,
            "dataset_sha256": manifest.get("dataset", {}).get("val_annotation_sha256"),
            "protocol_sha256": manifest.get("protocol_config", {}).get("sha256"),
        }
        for field, expected in source_values.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Formal comparison row measurement differs from source evidence: {field}"
                )
        epoch_peak = last_epoch.get("train_peak_allocated_mib") or last_epoch.get(
            "gpu_peak_allocated_mib"
        )
        if epoch_peak not in (None, "") and row.get("peak_gpu_memory_mib") != float(epoch_peak):
            raise ValueError(
                "Formal comparison row measurement differs from source evidence: "
                "peak_gpu_memory_mib"
            )

    comparison_csv_path = comparison_dir / "comparison.csv"
    with comparison_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        comparison_csv_rows = list(csv.DictReader(handle))
    comparison_fields = list(comparison_csv_rows[0]) if comparison_csv_rows else []
    expected_csv_rows = [
        {
            field: "" if row.get(field) is None else str(row.get(field))
            for field in comparison_fields
        }
        for row in comparison_rows
    ]
    if (
        not comparison_csv_rows
        or set(comparison_fields) != set(comparison_rows[0])
        or comparison_csv_rows != expected_csv_rows
    ):
        raise ValueError("Formal comparison.csv must exactly mirror comparison.json")

    aggregate_path = comparison_dir / "aggregate_comparison.json"
    aggregate_rows = json.loads(aggregate_path.read_text(encoding="utf-8-sig"))
    if not isinstance(aggregate_rows, list) or len(aggregate_rows) != 2:
        raise ValueError("Formal aggregate comparison must contain exactly two model rows")
    expected_aggregate_rows: list[dict[str, Any]] = []
    for model in dict.fromkeys(str(row["model"]) for row in comparison_rows):
        model_rows = [row for row in comparison_rows if str(row["model"]) == model]
        aggregate: dict[str, Any] = {"model": model, "runs": len(model_rows)}
        for metric in AGGREGATE_METRICS:
            samples = [float(row[metric]) for row in model_rows if row.get(metric) is not None]
            aggregate[f"{metric}_mean"] = statistics.fmean(samples) if samples else None
            aggregate[f"{metric}_std"] = (
                statistics.stdev(samples) if len(samples) > 1 else None
            )
        expected_aggregate_rows.append(aggregate)
    if aggregate_rows != expected_aggregate_rows:
        raise ValueError("Formal aggregate_comparison.json differs from the six comparison rows")
    aggregate_csv_path = comparison_dir / "aggregate_comparison.csv"
    with aggregate_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        aggregate_csv_rows = list(csv.DictReader(handle))
    aggregate_fields = list(expected_aggregate_rows[0])
    expected_aggregate_csv = [
        {
            field: "" if row.get(field) is None else str(row.get(field))
            for field in aggregate_fields
        }
        for row in expected_aggregate_rows
    ]
    if (
        not aggregate_csv_rows
        or list(aggregate_csv_rows[0]) != aggregate_fields
        or aggregate_csv_rows != expected_aggregate_csv
    ):
        raise ValueError(
            "Formal aggregate_comparison.csv must exactly mirror aggregate_comparison.json"
        )

    execution_path = comparison_dir / "formal_execution_status.json"
    execution_status = json.loads(execution_path.read_text(encoding="utf-8-sig"))
    expected_execution_runs = sorted(
        (
            {
                "model": str(row["model"]),
                "seed": int(row["seed"]),
                "run_id": str(row["run_id"]),
                "status": "complete",
                "observed_epoch_rows": 100,
            }
            for row in comparison_rows
        ),
        key=lambda item: (_normalized_model(item["model"]), item["seed"], item["run_id"]),
    )
    required_execution_values = {
        "status": "PASS",
        "scope": "formal_2_model_x_3_seed_100_epoch_comparison",
        "summary": "2 models × 3 seeds × 100 epochs",
        "run_count": 6,
        "seeds": [42, 43, 44],
        "epochs_per_run": 100,
        "runs": expected_execution_runs,
    }
    if any(execution_status.get(key) != value for key, value in required_execution_values.items()):
        raise ValueError("Formal execution-status overlay differs from the verified six-run evidence")
    expected_models = sorted(
        {str(row["model"]) for row in comparison_rows}, key=_normalized_model
    )
    if execution_status.get("models") != expected_models:
        raise ValueError("Formal execution-status model list differs from comparison rows")
    expected_validation = {
        "protocol_compatibility": "PASS",
        "exact_model_seed_matrix": "PASS",
        "complete_epoch_evidence": "PASS",
        "formal_artifact_binding": "PASS",
    }
    if execution_status.get("validation") != expected_validation:
        raise ValueError("Formal execution-status validation fields are not exact")
    if execution_status.get("protocol_snapshot_sha256") != sha256_file(protocol_snapshot_path):
        raise ValueError("Formal execution-status protocol snapshot SHA-256 mismatch")
    if execution_status.get("protocol_compatibility_sha256") != sha256_file(compatibility_path):
        raise ValueError("Formal execution-status compatibility SHA-256 mismatch")

    evidence_manifest, artifact_inventory = _validate_formal_artifact_inventory(
        comparison_dir,
        set(record_paths),
    )
    protocol_artifacts = _validate_protocol_artifacts(comparison_dir, execution_status)
    terminal_text = (comparison_dir / "comparison_terminal.txt").read_text(encoding="utf-8-sig")
    report_text = (comparison_dir / "experiment_report.md").read_text(encoding="utf-8-sig")
    run_labels = [
        f"{row['model']} seed={row['seed']} run_id={row['run_id']}" for row in comparison_rows
    ]
    if len(set(run_labels)) != 6 or any(label not in terminal_text for label in run_labels):
        raise ValueError("Formal terminal labels must uniquely include model, seed, and run_id")
    if any(
        f"| {row['model']} | {row['seed']} | {row['run_id']} |" not in report_text
        for row in comparison_rows
    ):
        raise ValueError("Formal Markdown rows must include model, seed, and run_id")
    formal_phrase = "FORMAL EXECUTION STATUS: PASS — 2 models × 3 seeds × 100 epochs"
    for filename in ("experiment_report.md", "experiment_methodology.md", "parameter_rationale.md"):
        text = (comparison_dir / filename).read_text(encoding="utf-8-sig")
        if formal_phrase not in text:
            raise ValueError(f"Formal status overlay is missing from {filename}")
        if "(ubuntu_handoff.md)" not in text or "(protocol_snapshot.yaml)" not in text:
            raise ValueError(f"Formal protocol/Ubuntu handoff links are missing from {filename}")

    source_records_for_chain = [
        (run_id, filename, *by_run_file[(run_id, filename)])
        for run_id, filename in sorted(by_run_file)
    ]
    bindings_path = comparison_dir / "local_source_bindings.json"
    bindings: dict[tuple[str, str], dict[str, Any]] = {}
    publication_project_root: Path | None = None
    if bindings_path.is_file():
        binding_document = json.loads(bindings_path.read_text(encoding="utf-8-sig"))
        root_text = str(binding_document.get("publication_project_root") or "")
        publication_project_root = Path(root_text).resolve() if root_text else None
        if (
            publication_project_root is None
            or not (publication_project_root / ".gitattributes").is_file()
        ):
            raise ValueError(
                "Local source bindings require the publication project root with .gitattributes"
            )
        for binding in binding_document.get("files", []):
            key = (str(binding.get("run_id") or ""), str(binding.get("filename") or ""))
            if key in bindings:
                raise ValueError(f"Duplicate local source binding: {key}")
            bindings[key] = binding
    if require_local_originals and not bindings:
        raise ValueError("Formal promotion requires local original run-artifact bindings")

    source_chain = []
    for run_id, filename, record, published_path in source_records_for_chain:
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
            if not _published_content_equivalent(
                original_path,
                published_path,
                publication_project_root=publication_project_root,
            ):
                raise ValueError(
                    "Published numeric evidence differs from local original, or an unapproved "
                    f"nonnumeric change was found: {run_id}/{filename}"
                )
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
    artifact_chain = sorted(
        [
            {
                "path": relative,
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]).lower(),
                "kind": str(record["kind"]),
            }
            for relative, record in artifact_inventory.items()
        ]
        + [
            {
                "path": "evidence_manifest.json",
                "bytes": (comparison_dir / "evidence_manifest.json").stat().st_size,
                "sha256": sha256_file(comparison_dir / "evidence_manifest.json"),
                "kind": "artifact_inventory",
            }
        ],
        key=lambda item: item["path"],
    )
    artifact_payload = json.dumps(artifact_chain, separators=(",", ":"), sort_keys=True)
    publication_allowlist = sorted(
        {item["path"] for item in artifact_chain} | {"formal_validation.json"}
    )
    formal_record = {
        "schema_version": 2,
        "status": "PASS",
        "formal_release": True,
        "run_count": 6,
        "model_seed_pairs": [list(pair) for pair in sorted(actual_pairs)],
        "protocol_compatibility_sha256": sha256_file(compatibility_path),
        "comparison_sha256": sha256_file(comparison_path),
        "sources_manifest_sha256": sha256_file(sources_path),
        "run_provenance_sha256": sha256_file(provenance_path),
        "run_provenance_attestation_sha256": (
            sha256_file(attestation_path) if attestation_path.is_file() else None
        ),
        "evidence_manifest_sha256": sha256_file(comparison_dir / "evidence_manifest.json"),
        "protocol_artifacts_sha256": sha256_file(comparison_dir / "protocol_artifacts.json"),
        "formal_execution_status_sha256": sha256_file(execution_path),
        "ubuntu_handoff_sha256": sha256_file(comparison_dir / "ubuntu_handoff.md"),
        "source_chain_sha256": hashlib.sha256(chain_payload.encode("utf-8")).hexdigest(),
        "source_chain": source_chain,
        "artifact_chain_sha256": hashlib.sha256(artifact_payload.encode("utf-8")).hexdigest(),
        "artifact_chain": artifact_chain,
        "required_user_artifacts": list(REQUIRED_FORMAL_USER_ARTIFACTS),
        "publication_allowlist": publication_allowlist,
        "image_provenance": {
            "evidence_manifest_generative_ai_used_for_images": evidence_manifest.get(
                "generative_ai_used_for_images"
            ),
            "protocol_artifacts_generative_ai_used_for_images": protocol_artifacts.get(
                "generative_ai_used_for_images"
            ),
        },
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


def validated_formal_publication_plan(
    comparison_dir: Path,
    *,
    require_local_originals: bool = True,
) -> dict[str, Any]:
    """Return the exact verified copy plan and reject every unlisted source payload."""
    comparison_dir = comparison_dir.resolve()
    formal_record = validate_formal_comparison(
        comparison_dir,
        require_local_originals=require_local_originals,
    )
    raw_allowlist = formal_record.get("publication_allowlist")
    if not isinstance(raw_allowlist, list) or not raw_allowlist:
        raise ValueError("Formal validation has no publication allowlist")
    allowlist: set[str] = set()
    for raw in raw_allowlist:
        relative, path = _contained_manifest_path(
            comparison_dir,
            raw,
            label="formal publication allowlist",
        )
        if relative in allowlist:
            raise ValueError(f"Duplicate formal publication allowlist path: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Formal publication file is missing: {relative}")
        allowlist.add(relative)

    actual = {
        path.relative_to(comparison_dir).as_posix()
        for path in comparison_dir.rglob("*")
        if path.is_file()
    }
    unlisted = sorted(actual - allowlist - FORMAL_PRIVATE_FILES)
    missing = sorted(allowlist - actual)
    unlisted_weights = sorted(
        path for path in unlisted if Path(path).suffix.lower() in WEIGHT_SUFFIXES
    )
    unlisted_images = sorted(
        path for path in unlisted if Path(path).suffix.lower() in IMAGE_SUFFIXES
    )
    if missing:
        raise FileNotFoundError(f"Formal publication allowlist files are missing: {missing}")
    if unlisted:
        raise ValueError(
            "Formal promotion source contains unlisted files; refuse stale/raw/weight payloads: "
            f"unlisted={unlisted}, images={unlisted_images}, weights={unlisted_weights}"
        )

    evidence = json.loads(
        (comparison_dir / "evidence_manifest.json").read_text(encoding="utf-8-sig")
    )
    derived_images = {
        str(record.get("path") or "").replace("\\", "/")
        for record in evidence.get("derived_images", [])
        if isinstance(record, dict)
    }
    allowed_weights = sorted(
        path for path in allowlist if Path(path).suffix.lower() in WEIGHT_SUFFIXES
    )
    allowed_images = {
        path for path in allowlist if Path(path).suffix.lower() in IMAGE_SUFFIXES
    }
    raw_images = sorted(allowed_images - derived_images)
    if allowed_weights:
        raise ValueError(f"Formal comparison publication must not include weights: {allowed_weights}")
    if raw_images:
        raise ValueError(f"Formal comparison publication contains undeclared raw images: {raw_images}")
    return {
        "formal_validation": formal_record,
        "relative_paths": sorted(allowlist),
        "scan": {
            "files_scanned": len(actual),
            "private_files_excluded": sorted(actual & FORMAL_PRIVATE_FILES),
            "unlisted_files": [],
            "raw_image_files": raw_images,
            "weight_files": allowed_weights,
            "derived_image_files": sorted(derived_images),
        },
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
    expected_manifest_path = f"sources/{safe_stem(run_id)}/run_manifest.json"
    source_rows = [
        row
        for row in sources_manifest.get("files", [])
        if str(row.get("run_id")) == run_id
        and str(row.get("path", "")).replace("\\", "/") == expected_manifest_path
    ]
    if len(source_rows) != 1:
        raise ValueError(f"Comparison source bundle must contain this run manifest: {run_id}")
    current_manifest_sha256 = sha256_file(run_manifest_path)
    if source_rows[0].get("source_original_sha256") != current_manifest_sha256:
        raise ValueError("Run manifest changed after the comparison was generated")

    expected_metrics_path = f"sources/{safe_stem(run_id)}/final_metrics.json"
    metric_rows = [
        row
        for row in sources_manifest.get("files", [])
        if str(row.get("run_id")) == run_id
        and str(row.get("path", "")).replace("\\", "/") == expected_metrics_path
    ]
    if len(metric_rows) != 1:
        raise ValueError(f"Comparison source bundle must contain this run's final metrics: {run_id}")
    native_final_metrics_sha256 = str(metric_rows[0].get("source_original_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", native_final_metrics_sha256):
        raise ValueError(f"Comparison final_metrics source SHA-256 is invalid: {run_id}")

    run_source_records = []
    for record in sources_manifest.get("files", []):
        if str(record.get("run_id")) != run_id:
            continue
        relative, published_path = _contained_manifest_path(
            comparison_dir,
            record.get("path"),
            label=f"comparison source/{run_id}",
        )
        run_source_records.append(
            {
                "path": relative,
                "filename": published_path.name,
                "bytes": published_path.stat().st_size,
                "published_sha256": sha256_file(published_path),
                "source_original_sha256": str(record.get("source_original_sha256", "")).lower(),
            }
        )

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
        "verified_source_files": sorted(run_source_records, key=lambda item: item["path"]),
    }
