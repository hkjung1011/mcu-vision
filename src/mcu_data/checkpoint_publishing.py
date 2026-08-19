from __future__ import annotations

import copy
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import sha256_file
from .publishing import (
    assert_public_binary_privacy,
    assert_public_text_privacy,
    load_json_strict,
)


WINDOWS_USER_HOME_TEXT = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]Users[\\/][^\\/\s\"']+)"
)
WINDOWS_USER_HOME_BYTES = re.compile(
    rb"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]Users[\\/][^\\/\s\"']+)"
)
YOLOX_CHECKPOINT_KEYS = {
    "model",
    "optimizer",
    "start_epoch",
    "epoch",
    "best_ap",
    "curr_ap",
}


def _replace_local_path(value: str, project_root: Path) -> str:
    candidates = {str(project_root.resolve()), str(Path.home().resolve())}
    for candidate in sorted(candidates, key=len, reverse=True):
        for spelling in (candidate, candidate.replace("\\", "/"), candidate.replace("\\", "\\\\")):
            value = re.sub(re.escape(spelling), "<PROJECT_ROOT>", value, flags=re.IGNORECASE)
    return WINDOWS_USER_HOME_TEXT.sub("<USER_HOME>", value)


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
    raw_content = path.read_bytes()
    try:
        assert_public_binary_privacy(raw_content, label=path.name)
    except ValueError as exc:
        raise ValueError(
            f"Published binary still contains a machine-local path: {path.name}"
        ) from exc
    content = raw_content.lower()
    compact_utf16 = content.replace(b"\x00", b"")
    if (
        any(value in content or value in compact_utf16 for value in forbidden_local_path_bytes(project_root))
        or WINDOWS_USER_HOME_BYTES.search(content)
        or WINDOWS_USER_HOME_BYTES.search(compact_utf16)
    ):
        raise ValueError(f"Published binary still contains a machine-local path: {path.name}")


def _assert_checkpoint_object_privacy(value: Any, *, label: str) -> None:
    strings: list[str] = []
    seen: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, (str, Path)):
            strings.append(str(item))
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen:
                return
            seen.add(identity)
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple, set)):
            identity = id(item)
            if identity in seen:
                return
            seen.add(identity)
            for child in item:
                visit(child)

    visit(value)
    assert_public_text_privacy(strings, label=label)


def _load_yolox_state_dict_cpu(path: Path) -> tuple[dict[str, Any], Mapping[str, Any], Any]:
    if path.suffix.lower() != ".pth":
        raise ValueError("Formal YOLOX checkpoints must use the .pth suffix")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - YOLOX environment only
        raise RuntimeError("PyTorch is required to verify a YOLOX checkpoint") from exc
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"YOLOX checkpoint failed restricted CPU load: {path.name}") from exc
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("YOLOX checkpoint must be a non-empty mapping")
    if any(not isinstance(key, str) for key in checkpoint):
        raise ValueError("YOLOX checkpoint top-level keys must be strings")
    unknown_keys = sorted(set(checkpoint) - YOLOX_CHECKPOINT_KEYS)
    if unknown_keys:
        raise ValueError(f"YOLOX checkpoint contains unknown top-level keys: {unknown_keys}")
    if "start_epoch" in checkpoint and type(checkpoint["start_epoch"]) is not int:
        raise ValueError("YOLOX checkpoint start_epoch must be an exact integer")
    if "epoch" in checkpoint and type(checkpoint["epoch"]) is not int:
        raise ValueError("YOLOX checkpoint epoch must be an exact integer")
    if "optimizer" in checkpoint and not isinstance(checkpoint["optimizer"], Mapping):
        raise ValueError("YOLOX checkpoint optimizer must be a mapping when present")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("YOLOX checkpoint must contain a non-empty model state_dict")
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise ValueError("YOLOX model state_dict must map string names to tensors")
        if tensor.device.type != "cpu":
            raise ValueError("YOLOX checkpoint tensors must load onto CPU")
    return checkpoint, state_dict, torch


def _yolox_proof(
    checkpoint: dict[str, Any],
    state_dict: Mapping[str, Any],
    *,
    bitwise_equal_after_copy: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "framework": "yolox",
        "publication_transform": "EXACT_FILE_COPY",
        "cpu_load": "PASS",
        "checkpoint_structure": "YOLOX_MODEL_STATE_DICT_PASS",
        "top_level_keys": sorted(checkpoint),
        "state_dict": {
            "status": "PASS",
            "tensor_count": len(state_dict),
            "tensor_bytes": sum(
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in state_dict.values()
            ),
            "bitwise_equal_after_copy": bitwise_equal_after_copy,
        },
        "forward": {
            "status": "NOT_PERFORMED",
            "max_abs_difference": None,
            "reason": "Architecture construction is not part of checkpoint publication.",
        },
        "privacy_scan": "PASS",
    }


def publish_yolox_checkpoint(
    source: Path,
    destination: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Publish an exact YOLOX .pth after restricted CPU/state_dict/privacy verification."""
    source = source.resolve()
    destination = destination.resolve()
    source_checkpoint, source_state, torch = _load_yolox_state_dict_cpu(source)
    _assert_checkpoint_object_privacy(source_checkpoint, label=source.name)
    assert_binary_has_no_local_paths(source, project_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    published_checkpoint, published_state, _ = _load_yolox_state_dict_cpu(destination)
    _assert_checkpoint_object_privacy(published_checkpoint, label=destination.name)
    if source_state.keys() != published_state.keys():
        raise ValueError("Published YOLOX state_dict keys changed")
    for name in source_state:
        source_tensor = source_state[name]
        published_tensor = published_state[name]
        if (
            source_tensor.dtype != published_tensor.dtype
            or source_tensor.shape != published_tensor.shape
            or not torch.equal(source_tensor, published_tensor)
        ):
            raise ValueError(f"Published YOLOX tensor changed: {name}")
    source_sha = sha256_file(source)
    published_sha = sha256_file(destination)
    if source_sha != published_sha:
        raise ValueError("Formal YOLOX publication must be an exact checkpoint copy")
    assert_binary_has_no_local_paths(destination, project_root)
    proof = _yolox_proof(
        published_checkpoint,
        published_state,
        bitwise_equal_after_copy=True,
    )
    return {
        "source_original_sha256": source_sha,
        "published_sha256": published_sha,
        "bytes": destination.stat().st_size,
        "proof": proof,
    }


def validate_yolox_checkpoint_proof(
    checkpoint_path: Path,
    record: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Recompute the public-clone YOLOX checkpoint proof without claiming a forward test."""
    checkpoint, state_dict, _ = _load_yolox_state_dict_cpu(checkpoint_path)
    _assert_checkpoint_object_privacy(checkpoint, label=checkpoint_path.name)
    assert_binary_has_no_local_paths(checkpoint_path, project_root)
    if str(record.get("source_original_sha256", "")).lower() != sha256_file(checkpoint_path):
        raise ValueError("YOLOX publication must retain exact source checkpoint bytes")
    actual = _yolox_proof(checkpoint, state_dict, bitwise_equal_after_copy=True)
    if record.get("proof") != actual:
        raise ValueError("YOLOX checkpoint proof differs from recomputed CPU/privacy evidence")
    return actual


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
    artifact = load_json_strict(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("Native artifact manifest must be a JSON object")
    release_name = str(artifact.get("release_name") or "")
    if release_name != artifact_relative.parts[0]:
        raise ValueError("Native artifact release_name differs from its directory")
    run_document = load_json_strict(run_manifest)
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
            and record.get("source_forward_captured_before_scrub") is True
            and record.get("ultralytics_load") == "PASS"
        )
        if not passed:
            raise ValueError("YOLO11 native artifact did not pass the formal checkpoint sanitizer")
    elif framework == "yolox":
        validate_yolox_checkpoint_proof(
            checkpoint,
            record,
            project_root=project_root,
        )
    else:
        raise ValueError(f"Unsupported formal checkpoint framework: {framework!r}")
    result = {
        "source_original_sha256": source_sha,
        "published_sha256": str(record["sha256"]).lower(),
        "proof": record.get("proof"),
    }
    if framework == "yolo11":
        result.update(
            {
                "metadata_sanitized": bool(record.get("metadata_sanitized")),
                "state_dict_bitwise_equal": record.get("state_dict_bitwise_equal") is True,
                "forward_max_abs_difference": record.get("forward_max_abs_difference"),
                "source_forward_captured_before_scrub": record.get(
                    "source_forward_captured_before_scrub"
                ) is True,
                "ultralytics_load": record.get("ultralytics_load"),
            }
        )
    return result


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
    # .float() mutates a module in-place; isolate it so the checkpoint tensors retain their
    # original dtype/bytes before metadata-only publication.
    original_model = copy.deepcopy(checkpoint["model"]).float().eval()
    sample = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    with torch.inference_mode():
        original_outputs = [
            tensor.detach().clone() for tensor in _flatten_tensors(original_model(sample))
        ]
    if not original_outputs:
        raise ValueError("Could not capture original YOLO11 forward outputs before scrubbing")
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

    published_model = published["model"].float().eval()
    with torch.inference_mode():
        published_outputs = _flatten_tensors(published_model(sample))
    if len(original_outputs) != len(published_outputs):
        raise ValueError("Could not compare published YOLO11 forward outputs")
    maximum_absolute_difference = 0.0
    for original, public in zip(original_outputs, published_outputs, strict=True):
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
        "source_forward_captured_before_scrub": True,
        "ultralytics_load": "PASS",
    }
