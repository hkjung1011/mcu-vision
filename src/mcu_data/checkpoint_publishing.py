from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .common import sha256_file


def _replace_local_path(value: str, project_root: Path) -> str:
    candidates = {str(project_root.resolve()), str(Path.home().resolve())}
    for candidate in sorted(candidates, key=len, reverse=True):
        for spelling in (candidate, candidate.replace("\\", "/"), candidate.replace("\\", "\\\\")):
            value = re.sub(re.escape(spelling), "<PROJECT_ROOT>", value, flags=re.IGNORECASE)
    return value


def _scrub_checkpoint_value(value: Any, project_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _scrub_checkpoint_value(item, project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_checkpoint_value(item, project_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_checkpoint_value(item, project_root) for item in value)
    if isinstance(value, (Path, str)):
        return _replace_local_path(str(value), project_root)
    return value


def _scrub_model_metadata(model: Any, project_root: Path) -> None:
    if model is None:
        return
    for attribute in ("args", "pt_path", "save_dir", "yaml_file"):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            if value is not None:
                setattr(model, attribute, _scrub_checkpoint_value(value, project_root))


def _flatten_tensors(value: Any) -> list[Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - framework environment only
        raise RuntimeError("PyTorch is required to verify a YOLO11 checkpoint") from exc
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors: list[Any] = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def forbidden_local_path_bytes(project_root: Path) -> set[bytes]:
    candidates = {str(project_root.resolve()), str(Path.home().resolve())}
    spellings: set[bytes] = set()
    for candidate in candidates:
        for value in (candidate, candidate.replace("\\", "/"), candidate.replace("\\", "\\\\")):
            spellings.add(value.encode("utf-8").lower())
    return {value for value in spellings if value}


def assert_binary_has_no_local_paths(path: Path, project_root: Path) -> None:
    content = path.read_bytes().lower()
    if any(value in content for value in forbidden_local_path_bytes(project_root)):
        raise ValueError(f"Published binary still contains a machine-local path: {path.name}")


def verify_formal_checkpoint_bridge(
    *,
    project_root: Path,
    artifact_path: Path,
    checkpoint: Path,
    run_manifest: Path,
    framework: str,
) -> dict[str, Any]:
    """Bind a promoted public checkpoint to the original completed-run checkpoint."""
    project_root = project_root.resolve()
    artifact_path = artifact_path.resolve()
    expected_root = (project_root / "reports" / "runs").resolve()
    try:
        artifact_relative = artifact_path.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError("Native artifact must be under reports/runs/<release>") from exc
    if len(artifact_relative.parts) != 2 or artifact_relative.name != "artifact_manifest.json":
        raise ValueError("Native artifact must be reports/runs/<release>/artifact_manifest.json")
    with artifact_path.open("r", encoding="utf-8-sig") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError("Native artifact manifest must be a JSON object")
    release_name = str(artifact.get("release_name") or "")
    if release_name != artifact_relative.parts[0]:
        raise ValueError("Native artifact release_name differs from its directory")
    with run_manifest.open("r", encoding="utf-8-sig") as handle:
        run_document = json.load(handle)
    if str(artifact.get("source_run_id") or "") != str(run_document.get("run_id") or ""):
        raise ValueError("Native artifact source_run_id differs from the run manifest")
    if str(artifact.get("source_run_manifest_sha256", "")).lower() != sha256_file(run_manifest).lower():
        raise ValueError("Native artifact does not bind the selected run manifest")
    record = artifact.get("checkpoint")
    if not isinstance(record, dict):
        raise ValueError("Native artifact is missing checkpoint publication evidence")
    recorded_path = (project_root / str(record.get("path", ""))).resolve()
    expected_weight_root = (project_root / "weights" / "trained" / release_name).resolve()
    try:
        recorded_path.relative_to(expected_weight_root)
    except ValueError as exc:
        raise ValueError("Native artifact checkpoint is outside its trained release directory") from exc
    if recorded_path != checkpoint.resolve():
        raise ValueError("Native artifact checkpoint path differs from --checkpoint")
    if str(record.get("sha256", "")).lower() != sha256_file(checkpoint).lower():
        raise ValueError("Native artifact published checkpoint SHA-256 mismatch")
    source_sha = str(record.get("source_original_sha256", "")).lower()
    if len(source_sha) != 64 or any(character not in "0123456789abcdef" for character in source_sha):
        raise ValueError("Native artifact source_original_sha256 is invalid")
    best_checkpoint = run_document.get("best_checkpoint")
    if not isinstance(best_checkpoint, dict) or str(best_checkpoint.get("sha256", "")).lower() != source_sha:
        raise ValueError("Run manifest original checkpoint differs from publication bridge")
    if framework == "yolo11":
        passed = (
            record.get("metadata_sanitized") is True
            and record.get("state_dict_bitwise_equal") is True
            and float(record.get("forward_max_abs_difference", -1)) == 0.0
            and record.get("ultralytics_load") == "PASS"
        )
        if not passed:
            raise ValueError("YOLO11 native artifact did not pass the formal checkpoint sanitizer")
    return {
        "source_original_sha256": source_sha,
        "published_sha256": str(record["sha256"]).lower(),
        "metadata_sanitized": bool(record.get("metadata_sanitized")),
        "state_dict_bitwise_equal": record.get("state_dict_bitwise_equal") is True,
        "forward_max_abs_difference": record.get("forward_max_abs_difference"),
        "ultralytics_load": record.get("ultralytics_load"),
    }


def sanitize_yolo11_checkpoint(
    source: Path,
    destination: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Remove local metadata while proving tensor and forward-output identity."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - YOLO11 environment only
        raise RuntimeError("Run this publisher with the pinned YOLO11 environment") from exc

    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model") is None:
        raise ValueError(f"Unexpected YOLO11 checkpoint structure: {source}")
    for key in tuple(checkpoint):
        if key not in {"model", "ema"}:
            checkpoint[key] = _scrub_checkpoint_value(checkpoint[key], project_root)
    _scrub_model_metadata(checkpoint.get("model"), project_root)
    _scrub_model_metadata(checkpoint.get("ema"), project_root)
    torch.save(checkpoint, destination)

    published = torch.load(destination, map_location="cpu", weights_only=False)
    source_state = checkpoint["model"].state_dict()
    published_state = published["model"].state_dict()
    if source_state.keys() != published_state.keys():
        raise ValueError("Published YOLO11 state_dict keys changed")
    for name in source_state:
        if not torch.equal(source_state[name], published_state[name]):
            raise ValueError(f"Published YOLO11 tensor changed: {name}")

    source_model = checkpoint["model"].float().eval()
    published_model = published["model"].float().eval()
    sample = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    with torch.inference_mode():
        source_outputs = _flatten_tensors(source_model(sample))
        published_outputs = _flatten_tensors(published_model(sample))
    if len(source_outputs) != len(published_outputs) or not source_outputs:
        raise ValueError("Could not compare published YOLO11 forward outputs")
    maximum_absolute_difference = 0.0
    for original, public in zip(source_outputs, published_outputs, strict=True):
        if original.shape != public.shape:
            raise ValueError("Published YOLO11 output shape changed")
        difference = float((original - public).abs().max().item()) if original.numel() else 0.0
        maximum_absolute_difference = max(maximum_absolute_difference, difference)
    if maximum_absolute_difference != 0.0:
        raise ValueError(f"Published YOLO11 forward output changed: {maximum_absolute_difference}")

    YOLO(str(destination), task="detect")
    assert_binary_has_no_local_paths(destination, project_root)
    return {
        "source_original_sha256": sha256_file(source),
        "published_sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "metadata_sanitized": True,
        "state_dict_bitwise_equal": True,
        "forward_max_abs_difference": maximum_absolute_difference,
        "ultralytics_load": "PASS",
    }
