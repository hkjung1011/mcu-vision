from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import zlib
from pathlib import Path
from typing import Any

import yaml

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
PUBLIC_TEXT_SUFFIXES = TEXT_SUFFIXES | {".json", ".jsonl", ".svg"}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
GIF_MAGICS = (b"GIF87a", b"GIF89a")
ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
PICKLE_MAGIC = b"\x80"
GENERIC_WINDOWS_ABSOLUTE_TEXT = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/][^\s`\"'<>|]+|"
    r"\\\\[^\\/\s]+[\\/][^\s`\"'<>|]+)"
)
GENERIC_POSIX_ABSOLUTE_TEXT = re.compile(
    r"(?m)(?<![A-Za-z0-9:/])/(?![/\s])[A-Za-z0-9._~+@=-]+"
    r"(?:/[A-Za-z0-9._~+@=-]+)*"
)
GENERIC_WINDOWS_ABSOLUTE_BYTES = re.compile(
    rb"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/][^\s`\"'<>|]+|"
    rb"\\\\[^\\/\s]+[\\/][^\s`\"'<>|]+)"
)
GENERIC_POSIX_ABSOLUTE_BYTES = re.compile(
    rb"(?m)(?<![A-Za-z0-9:/])/(?![/\s])[A-Za-z0-9._~+@=-]+"
    rb"(?:/[A-Za-z0-9._~+@=-]+)*"
)
MIN_BINARY_TEXT_RUN = 2
CHECKPOINT_BINARY_TEXT_RUN = 6
JSON_NUMBER_TEXT = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z"
)


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON number: {value}")
    return parsed


def loads_json_strict(text: str, *, label: str = "JSON") -> Any:
    """Decode RFC-style JSON while rejecting duplicate keys and NaN/Infinity."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {exc}") from exc


def load_json_strict(path: Path, *, label: str | None = None) -> Any:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSON file: {path}") from exc
    if "\x00" in text:
        raise ValueError(f"JSON file contains NUL bytes: {path}")
    return loads_json_strict(text, label=label or path.name)


def load_jsonl_strict(path: Path, *, label: str | None = None) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSONL file: {path}") from exc
    if "\x00" in text:
        raise ValueError(f"JSONL file contains NUL bytes: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = loads_json_strict(
            line,
            label=f"{label or path.name} line {line_number}",
        )
        if not isinstance(value, dict):
            raise ValueError(
                f"Invalid {label or path.name} line {line_number}: JSONL rows must be objects"
            )
        rows.append(value)
    return rows


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ": "),
        )
        for row in rows
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _newline_hash_variants(path: Path) -> set[str]:
    """Hash the exact, canonical-LF, and checkout-CRLF forms of UTF-8 text."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(f"Protocol snapshot must be valid UTF-8: {path}") from exc
    if "\x00" in text:
        raise ValueError(f"Protocol snapshot contains NUL bytes: {path}")
    lf_bytes = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
    return {
        sha256_file(path),
        hashlib.sha256(lf_bytes).hexdigest(),
        hashlib.sha256(crlf_bytes).hexdigest(),
    }


def _json_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for key, item in value.items():
            strings.append(str(key))
            strings.extend(_json_strings(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(_json_strings(item))
        return strings
    return []


def assert_public_text_privacy(strings: list[str], *, label: str) -> None:
    for value in strings:
        inspected = value
        for placeholder in ("<PROJECT_ROOT>", "<USER_HOME>"):
            inspected = inspected.replace(placeholder + "/", "")
            inspected = inspected.replace(placeholder + "\\", "")
        if (
            GENERIC_WINDOWS_ABSOLUTE_TEXT.search(inspected)
            or GENERIC_POSIX_ABSOLUTE_TEXT.search(inspected)
        ):
            raise ValueError(f"Public artifact contains an absolute local path: {label}")


def assert_public_binary_privacy(
    content: bytes,
    *,
    label: str,
    minimum_text_run: int = CHECKPOINT_BINARY_TEXT_RUN,
) -> None:
    """Reject paths in conservative printable runs from an opaque binary payload."""
    if type(minimum_text_run) is not int or minimum_text_run < MIN_BINARY_TEXT_RUN:
        raise ValueError("minimum_text_run must be an integer >= 2")
    printable = re.compile(rb"[\x20-\x7e]{" + str(minimum_text_run).encode("ascii") + rb",}")
    candidates = printable.findall(content)
    utf16_le = re.compile(
        rb"(?:[\x20-\x7e]\x00){" + str(minimum_text_run).encode("ascii") + rb",}"
    )
    utf16_be = re.compile(
        rb"(?:\x00[\x20-\x7e]){" + str(minimum_text_run).encode("ascii") + rb",}"
    )
    candidates.extend(candidate[::2] for candidate in utf16_le.findall(content))
    candidates.extend(candidate[1::2] for candidate in utf16_be.findall(content))
    inspected_candidates = []
    for candidate in candidates:
        for placeholder in (b"<PROJECT_ROOT>", b"<USER_HOME>"):
            candidate = candidate.replace(placeholder + b"/", b"")
            candidate = candidate.replace(placeholder + b"\\", b"")
        inspected_candidates.append(candidate)
    for candidate in inspected_candidates:
        for pattern in (GENERIC_WINDOWS_ABSOLUTE_BYTES, GENERIC_POSIX_ABSOLUTE_BYTES):
            if any(len(match.group(0)) >= minimum_text_run for match in pattern.finditer(candidate)):
                raise ValueError(
                    f"Public binary artifact contains an absolute local path: {label}"
                )


def _png_text_metadata(content: bytes, *, label: str) -> bytes:
    """Return decoded PNG text chunks while deliberately excluding compressed IDAT bytes."""
    metadata: list[bytes] = []
    offset = len(PNG_MAGIC)
    while offset + 12 <= len(content):
        length = int.from_bytes(content[offset : offset + 4], "big")
        chunk_type = content[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        next_offset = data_end + 4
        if next_offset > len(content):
            break
        payload = content[data_start:data_end]
        try:
            if chunk_type == b"tEXt":
                metadata.append(payload)
            elif chunk_type == b"zTXt":
                keyword, compressed = payload.split(b"\x00", 1)
                if not compressed or compressed[0] != 0:
                    raise ValueError("unsupported PNG zTXt compression method")
                metadata.extend((keyword, zlib.decompress(compressed[1:])))
            elif chunk_type == b"iTXt":
                keyword, remainder = payload.split(b"\x00", 1)
                if len(remainder) < 2:
                    raise ValueError("truncated PNG iTXt header")
                compression_flag, compression_method = remainder[:2]
                language, translated, text = remainder[2:].split(b"\x00", 2)
                if compression_flag not in {0, 1} or compression_method != 0:
                    raise ValueError("unsupported PNG iTXt compression settings")
                if compression_flag == 1:
                    text = zlib.decompress(text)
                metadata.extend((keyword, language, translated, text))
        except (ValueError, zlib.error) as exc:
            raise ValueError(f"Invalid PNG text metadata: {label}") from exc
        offset = next_offset
        if chunk_type == b"IEND":
            break
    return b"\n".join(metadata)


class _ProtobufDecodeError(ValueError):
    pass


def _protobuf_varint(content: memoryview, offset: int) -> tuple[int, int]:
    result = 0
    for shift in range(0, 70, 7):
        if offset >= len(content):
            raise _ProtobufDecodeError("truncated protobuf varint")
        byte = content[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
    raise _ProtobufDecodeError("oversized protobuf varint")


def _protobuf_fields(content: memoryview) -> list[tuple[int, int, memoryview | None]]:
    fields: list[tuple[int, int, memoryview | None]] = []
    offset = 0
    while offset < len(content):
        key, offset = _protobuf_varint(content, offset)
        field_number, wire_type = key >> 3, key & 7
        if field_number == 0:
            raise _ProtobufDecodeError("protobuf field number zero")
        if wire_type == 0:
            _, offset = _protobuf_varint(content, offset)
            fields.append((field_number, wire_type, None))
        elif wire_type == 1:
            offset += 8
            if offset > len(content):
                raise _ProtobufDecodeError("truncated protobuf fixed64")
            fields.append((field_number, wire_type, None))
        elif wire_type == 2:
            length, offset = _protobuf_varint(content, offset)
            end = offset + length
            if end > len(content):
                raise _ProtobufDecodeError("truncated protobuf bytes field")
            fields.append((field_number, wire_type, content[offset:end]))
            offset = end
        elif wire_type == 5:
            offset += 4
            if offset > len(content):
                raise _ProtobufDecodeError("truncated protobuf fixed32")
            fields.append((field_number, wire_type, None))
        else:
            raise _ProtobufDecodeError(f"unsupported protobuf wire type {wire_type}")
    return fields


# ONNX v1.22 onnx.proto3 field numbers. Tensor payload fields are deliberately absent.
_ONNX_PRIVACY_TEXT_FIELDS = {
    "model": {2, 3, 4, 6},
    "graph": {10},
    "node": {6},
    "attribute": {13},
    "tensor": {12},
    "value_info": {3},
    "function": {8},
    "entry": {1, 2},
}
_ONNX_PRIVACY_BYTES_FIELDS = {"attribute": {4, 9}}
_ONNX_MESSAGE_FIELDS = {
    "model": {7: "graph", 14: "entry", 20: "training", 25: "function"},
    "graph": {
        1: "node",
        5: "tensor",
        11: "value_info",
        12: "value_info",
        13: "value_info",
        14: "tensor_annotation",
        15: "sparse_tensor",
        16: "entry",
    },
    "node": {5: "attribute", 9: "entry"},
    "attribute": {
        5: "tensor",
        6: "graph",
        10: "tensor",
        11: "graph",
        22: "sparse_tensor",
        23: "sparse_tensor",
    },
    "tensor": {13: "entry", 16: "entry"},
    "sparse_tensor": {1: "tensor", 2: "tensor"},
    "value_info": {4: "entry"},
    "tensor_annotation": {2: "entry"},
    "training": {1: "graph", 2: "graph", 3: "entry", 4: "entry"},
    "function": {7: "node", 11: "attribute", 12: "value_info", 14: "entry"},
}


def _onnx_metadata_strings(content: bytes) -> list[str]:
    """Decode path-bearing ONNX protobuf metadata without reading tensor payload fields."""
    strings: list[str] = []
    saw_graph = False

    def append_text(payload: memoryview) -> None:
        try:
            strings.append(bytes(payload).decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("ONNX metadata contains invalid UTF-8") from exc

    def visit(payload: memoryview, message_type: str, depth: int) -> None:
        nonlocal saw_graph
        if depth > 64:
            raise _ProtobufDecodeError("ONNX protobuf nesting is too deep")
        for field_number, wire_type, value in _protobuf_fields(payload):
            if wire_type != 2 or value is None:
                continue
            if message_type == "model" and field_number == 7:
                saw_graph = True
            child_type = _ONNX_MESSAGE_FIELDS.get(message_type, {}).get(field_number)
            if child_type is not None:
                visit(value, child_type, depth + 1)
            elif field_number in _ONNX_PRIVACY_TEXT_FIELDS.get(message_type, set()):
                append_text(value)
            elif field_number in _ONNX_PRIVACY_BYTES_FIELDS.get(message_type, set()):
                append_text(value)

    visit(memoryview(content), "model", 0)
    if not saw_graph:
        raise _ProtobufDecodeError("ONNX ModelProto graph field is missing")
    return strings


def assert_public_onnx_privacy(content: bytes, *, label: str) -> None:
    """Scan ONNX metadata/external-data paths, excluding raw tensor payload bytes."""
    try:
        strings = _onnx_metadata_strings(content)
    except _ProtobufDecodeError:
        # Legacy/fake fixtures are not parseable ModelProto files; retain the conservative gate.
        assert_public_binary_privacy(
            content,
            label=label,
            minimum_text_run=CHECKPOINT_BINARY_TEXT_RUN,
        )
        return
    assert_public_text_privacy(strings, label=label)


def _validate_yaml_schema(text: str, *, label: str) -> None:
    try:
        node = yaml.compose(text)
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML schema: {label}") from exc

    def finite_values(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite YAML number in {label}")
        if isinstance(value, dict):
            for key, child in value.items():
                finite_values(key)
                finite_values(child)
        elif isinstance(value, list):
            for child in value:
                finite_values(child)

    finite_values(document)

    def visit(value: yaml.Node | None) -> None:
        if value is None:
            return
        if isinstance(value, yaml.MappingNode):
            seen: set[str] = set()
            for key_node, child in value.value:
                key = str(getattr(key_node, "value", ""))
                if key in seen:
                    raise ValueError(f"Duplicate YAML mapping key in {label}: {key!r}")
                seen.add(key)
                visit(child)
        elif isinstance(value, yaml.SequenceNode):
            for child in value.value:
                visit(child)
        elif isinstance(value, yaml.ScalarNode):
            scalar = str(value.value)
            if (
                value.style is None
                and JSON_NUMBER_TEXT.fullmatch(scalar)
                and ("e" in scalar.lower())
                and not math.isfinite(float(scalar))
            ):
                raise ValueError(f"Non-finite YAML exponent in {label}: {scalar}")

    visit(node)


def _detected_binary_magic(content: bytes) -> str | None:
    if content.startswith(PNG_MAGIC):
        return "png"
    if content.startswith(JPEG_MAGIC):
        return "jpeg"
    if content.startswith(GIF_MAGICS):
        return "gif"
    if content.startswith(ZIP_MAGICS):
        return "zip"
    if content.startswith(PICKLE_MAGIC):
        return "pickle_or_torch_legacy"
    return None


def scan_public_file(path: Path, *, relative_path: str) -> dict[str, Any]:
    """Validate public artifact path, media signature, schema, UTF-8, and path privacy."""
    raw_relative = str(relative_path)
    normalized = raw_relative.replace("\\", "/")
    if (
        not normalized
        or raw_relative != normalized
        or normalized.startswith("/")
        or Path(normalized).is_absolute()
        or Path(normalized).as_posix() != normalized
        or ":" in normalized
        or ".." in Path(normalized).parts
    ):
        raise ValueError(f"Public artifact path is not canonical: {relative_path!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Public artifact is missing: {normalized}")
    content = path.read_bytes()
    suffix = path.suffix.lower()
    magic = _detected_binary_magic(content)
    expected_magic = {
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".gif": "gif",
    }.get(suffix)
    if expected_magic is not None and magic != expected_magic:
        raise ValueError(
            f"Public artifact signature does not match {suffix}: {normalized}"
        )
    allowed_magic = {
        "png": {".png"},
        "jpeg": {".jpg", ".jpeg"},
        "gif": {".gif"},
        "zip": {".pt", ".pth", ".torchscript", ".zip"},
        "pickle_or_torch_legacy": {".pt", ".pth"},
    }
    if magic is not None and suffix not in allowed_magic[magic]:
        raise ValueError(
            f"Public artifact has disguised {magic} content: {normalized}"
        )
    if suffix in {".pt", ".pth", ".torchscript"} and magic not in {
        "zip",
        "pickle_or_torch_legacy",
    }:
        raise ValueError(f"Public torch checkpoint has an unknown signature: {normalized}")

    schema = "BINARY"
    if suffix in PUBLIC_TEXT_SUFFIXES:
        if magic is not None or b"\x00" in content:
            raise ValueError(f"Public text artifact contains binary/NUL content: {normalized}")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Public text artifact is not valid UTF-8: {normalized}") from exc
        strings = [text]
        if suffix == ".json":
            document = loads_json_strict(text, label=normalized)
            if not isinstance(document, (dict, list)):
                raise ValueError(f"Public JSON top level must be object/array: {normalized}")
            strings = _json_strings(document)
            schema = "JSON_PASS"
        elif suffix == ".jsonl":
            rows = load_jsonl_strict(path, label=normalized)
            strings = _json_strings(rows)
            schema = "JSONL_PASS"
        elif suffix == ".csv":
            try:
                rows = list(csv.reader(text.splitlines(), strict=True))
            except csv.Error as exc:
                raise ValueError(f"Public CSV is malformed: {normalized}") from exc
            if not rows or not rows[0] or len(set(rows[0])) != len(rows[0]):
                raise ValueError(f"Public CSV has an invalid/duplicate header: {normalized}")
            if any(len(row) != len(rows[0]) for row in rows[1:]):
                raise ValueError(f"Public CSV row width differs from header: {normalized}")
            schema = "CSV_PASS"
        elif suffix in {".yaml", ".yml"}:
            _validate_yaml_schema(text, label=normalized)
            schema = "YAML_PASS"
        elif suffix == ".svg":
            if "<svg" not in text[:4096].lower():
                raise ValueError(f"Public SVG signature is missing: {normalized}")
            schema = "SVG_PASS"
        else:
            schema = "UTF8_TEXT_PASS"
        assert_public_text_privacy(strings, label=normalized)
    else:
        if magic == "png":
            metadata = _png_text_metadata(content, label=normalized)
            assert_public_binary_privacy(
                metadata,
                label=normalized,
                minimum_text_run=CHECKPOINT_BINARY_TEXT_RUN,
            )
        elif suffix == ".onnx":
            assert_public_onnx_privacy(content, label=normalized)
        else:
            assert_public_binary_privacy(
                content,
                label=normalized,
                minimum_text_run=CHECKPOINT_BINARY_TEXT_RUN,
            )
    return {
        "path": normalized,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "extension": suffix,
        "detected_magic": magic,
        "schema": schema,
        "privacy": "PASS",
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
        value = load_json_strict(source)
        cleaned, changed = _scrub_json_value(value, replacements)
        destination.write_bytes(_canonical_json_bytes(cleaned))
    elif source.suffix.lower() == ".jsonl":
        output_rows: list[dict[str, Any]] = []
        for value in load_jsonl_strict(source):
            cleaned, line_changed = _scrub_json_value(value, replacements)
            changed = changed or line_changed
            if not isinstance(cleaned, dict):  # defensive: scrub preserves object shape
                raise ValueError(f"Published JSONL row changed shape: {source}")
            output_rows.append(cleaned)
        destination.write_bytes(_canonical_jsonl_bytes(output_rows))
    elif source.suffix.lower() in TEXT_SUFFIXES:
        try:
            text = source.read_text(encoding="utf-8-sig")
        except UnicodeError as exc:
            raise ValueError(f"Published text is not valid UTF-8: {source}") from exc
        if "\x00" in text:
            raise ValueError(f"Published text contains NUL bytes: {source}")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned_text, changed = _scrub_text(text, replacements)
        if source.suffix.lower() in {".yaml", ".yml"}:
            _validate_yaml_schema(cleaned_text, label=source.name)
        destination.write_text(cleaned_text, encoding="utf-8", newline="\n")
    else:
        shutil.copy2(source, destination)

    return {
        "source_original_sha256": original_sha256,
        "published_sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "sanitized_for_repository": changed,
        "canonical_serialization": source.suffix.lower() in TEXT_SUFFIXES
        or source.suffix.lower() in {".json", ".jsonl"},
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
            load_jsonl_strict(path)
            if path.suffix.lower() == ".jsonl"
            else [load_json_strict(path)]
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


def _expected_published_bytes(
    original: Path,
    *,
    publication_project_root: Path,
) -> bytes:
    """Reapply the sole approved publication transform into deterministic bytes."""
    suffix = original.suffix.lower()
    replacements = _replacement_pairs(publication_project_root)
    if suffix == ".json":
        expected, _ = _scrub_json_value(
            load_json_strict(original),
            replacements,
        )
        return _canonical_json_bytes(expected)
    if suffix == ".jsonl":
        original_rows = load_jsonl_strict(original)
        expected_rows = [_scrub_json_value(row, replacements)[0] for row in original_rows]
        return _canonical_jsonl_bytes(expected_rows)
    if suffix in TEXT_SUFFIXES:
        text = original.read_text(encoding="utf-8-sig")
        if "\x00" in text:
            raise ValueError(f"Published text contains NUL bytes: {original}")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        expected, _ = _scrub_text(
            text,
            replacements,
        )
        return expected.encode("utf-8")
    return original.read_bytes()


def _published_content_equivalent(
    original: Path,
    published: Path,
    *,
    publication_project_root: Path,
) -> bool:
    return published.read_bytes() == _expected_published_bytes(
        original,
        publication_project_root=publication_project_root,
    )


def _contained_manifest_path(root: Path, relative: Any, *, label: str) -> tuple[str, Path]:
    raw = str(relative or "")
    normalized = raw.replace("\\", "/")
    if (
        not normalized
        or raw != normalized
        or normalized.startswith("/")
        or ":" in normalized
        or ".." in Path(normalized).parts
    ):
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


def _require_exact_record_size(record: dict[str, Any], path: Path, *, label: str) -> None:
    value = record.get("bytes")
    if type(value) is not int or value < 0 or value != path.stat().st_size:
        raise ValueError(f"{label} byte-size mismatch: {path.name}")


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
    _require_exact_record_size(record, path, label=label)
    return relative, path


def _validate_formal_artifact_inventory(
    comparison_dir: Path,
    source_bundle_paths: set[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evidence_path = comparison_dir / "evidence_manifest.json"
    evidence = load_json_strict(evidence_path)
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
    document = load_json_strict(path)
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
        _require_exact_record_size(raw, artifact_path, label="formal protocol artifact")
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
    compatibility = load_json_strict(compatibility_path)
    if not (
        compatibility.get("release_ready") is True
        and compatibility.get("comparable") is True
        and compatibility.get("critical_mismatches") == []
        and compatibility.get("release_blockers") == []
        and int(compatibility.get("run_count", -1)) == 6
    ):
        raise ValueError("Formal comparison compatibility claims are not an unblocked six-run PASS")
    provenance_path = comparison_dir / "run_provenance.json"
    provenance = load_json_strict(provenance_path)
    if provenance.get("status") != "PASS" or compatibility.get("run_provenance") != provenance:
        raise ValueError("Formal comparison provenance is missing, failed, or differs from compatibility")
    attestation_path = comparison_dir / "run_provenance_attestation.json"
    if provenance.get("mixed_commits") is True and (
        not attestation_path.is_file() or not isinstance(provenance.get("attestation"), dict)
    ):
        raise ValueError("Formal mixed-commit provenance attestation is missing or unbound")
    if provenance.get("mixed_commits") is True:
        attestation = load_json_strict(attestation_path)
        if sorted(attestation.get("allowed_commits", [])) != sorted(
            provenance["attestation"].get("allowed_commits", [])
        ):
            raise ValueError("Formal provenance attestation commit set differs from provenance")
    expectations = compatibility.get("release_expectations", {})
    expected_seed_values = expectations.get("seeds", [])
    if (
        {_normalized_model(value) for value in expectations.get("models", [])} != FORMAL_MODELS
        or not isinstance(expected_seed_values, list)
        or any(type(value) is not int for value in expected_seed_values)
        or set(expected_seed_values) != FORMAL_SEEDS
        or int(expectations.get("runs", -1)) != 6
    ):
        raise ValueError("Formal comparison release expectations are not the exact model/seed matrix")

    comparison_rows = load_json_strict(comparison_path)
    if not isinstance(comparison_rows, list) or len(comparison_rows) != 6:
        raise ValueError("Formal comparison.json must contain exactly six rows")
    row_ids = [str(row.get("run_id") or "") for row in comparison_rows if isinstance(row, dict)]
    if len(row_ids) != 6 or len(set(row_ids)) != 6:
        raise ValueError("Formal comparison run_id values must be six unique identifiers")

    sources = load_json_strict(sources_path)
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
        _require_exact_record_size(record, path, label="formal source")
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
        manifest = load_json_strict(by_run_file[(run_id, "run_manifest.json")][1])
        if (
            str(manifest.get("run_id")) != run_id
            or manifest.get("status") != "complete"
            or manifest.get("stage") == "smoke_not_comparable"
            or not str(manifest.get("best_checkpoint", {}).get("sha256") or "")
        ):
            raise ValueError(f"Formal run manifest is not a complete checkpointed run: {run_id}")
        protocol = manifest.get("protocol", {})
        model = _normalized_model(manifest.get("model"))
        seed_value = protocol.get("seed")
        if type(seed_value) is not int:
            raise ValueError(
                f"Formal run seed must have exact JSON integer type: {run_id}"
            )
        seed = seed_value
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
        final = load_json_strict(by_run_file[(run_id, "final_metrics.json")][1])
        metrics = final.get("metrics", final.get("metrics_common", {}))
        _finite_fields(
            metrics,
            ("ap50_95", "ap50", "ap75", "ar100", "precision", "recall", "f1", "tp", "fp", "fn"),
            f"metrics/{run_id}",
        )
        metrics_by_run[run_id] = metrics
        latency = load_json_strict(by_run_file[(run_id, "latency.json")][1])
        _finite_fields(latency, ("e2e_p50_ms", "e2e_p95_ms", "sustained_fps"), f"latency/{run_id}")
        latency_by_run[run_id] = latency
        gpu = load_json_strict(by_run_file[(run_id, "gpu_summary.json")][1])
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
    if len(protocol_hashes) != 1 or not protocol_hashes <= _newline_hash_variants(
        protocol_snapshot_path
    ):
        raise ValueError(
            "Formal protocol_snapshot.yaml differs from the protocol_config SHA-256 bound by runs"
        )
    for row in comparison_rows:
        run_id = str(row["run_id"])
        manifest = manifests[run_id]
        if (
            row.get("status") != "complete"
            or _normalized_model(row.get("model")) != _normalized_model(manifest.get("model"))
            or type(row.get("seed")) is not int
            or row.get("seed") != manifest.get("protocol", {}).get("seed")
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
    aggregate_rows = load_json_strict(aggregate_path)
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
    execution_status = load_json_strict(execution_path)
    if (
        not isinstance(execution_status.get("seeds"), list)
        or any(type(value) is not int for value in execution_status.get("seeds", []))
        or not isinstance(execution_status.get("runs"), list)
        or any(
            not isinstance(item, dict) or type(item.get("seed")) is not int
            for item in execution_status.get("runs", [])
        )
    ):
        raise ValueError("Formal execution-status seeds must use exact JSON integer types")
    expected_execution_runs = sorted(
        (
            {
                "model": str(row["model"]),
                "seed": row["seed"],
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
        binding_document = load_json_strict(bindings_path)
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
        recorded = load_json_strict(formal_path)
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

    evidence = load_json_strict(comparison_dir / "evidence_manifest.json")
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
    public_file_records = [
        scan_public_file(comparison_dir / relative, relative_path=relative)
        for relative in sorted(allowlist)
    ]
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
            "public_file_records": public_file_records,
        },
    }


def copy_public_file_exact(source: Path, destination: Path) -> dict[str, Any]:
    """Copy an already validated formal artifact without invalidating its hash chain."""
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(source)
    shutil.copy2(source, destination)
    published_sha = sha256_file(destination)
    if source_sha != published_sha:
        raise ValueError(f"Verified public artifact changed during copy: {source.name}")
    return {
        "source_original_sha256": source_sha,
        "published_sha256": published_sha,
        "bytes": destination.stat().st_size,
        "sanitized_for_repository": False,
        "canonical_serialization": source.suffix.lower() in PUBLIC_TEXT_SUFFIXES,
        "copy_mode": "EXACT_VERIFIED_BYTES",
    }


def validate_published_comparison_release(release_dir: Path) -> dict[str, Any]:
    """Validate a public-clone comparison including its sole implicit manifest file."""
    release_dir = release_dir.resolve()
    manifest_path = release_dir / "artifact_manifest.json"
    document = load_json_strict(manifest_path)
    if not isinstance(document, dict):
        raise ValueError("Published comparison artifact_manifest.json must be an object")
    if (
        document.get("schema_version") != 3
        or document.get("status") != "PASS"
        or document.get("formal_release") is not True
    ):
        raise ValueError("Published comparison manifest is not a schema-3 formal PASS")
    raw_records = document.get("files")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Published comparison manifest files must be a non-empty list")
    records: dict[str, dict[str, Any]] = {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("Published comparison file record must be an object")
        relative, path = _contained_manifest_path(
            release_dir,
            raw.get("path"),
            label="published comparison file",
        )
        if relative == "artifact_manifest.json" or relative in records:
            raise ValueError(f"Duplicate/reserved published comparison path: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Published comparison file is missing: {relative}")
        _require_exact_record_size(raw, path, label="published comparison")
        if str(raw.get("published_sha256", "")).lower() != sha256_file(path):
            raise ValueError(f"Published comparison SHA-256 mismatch: {relative}")
        records[relative] = raw
    actual = {
        path.relative_to(release_dir).as_posix()
        for path in release_dir.rglob("*")
        if path.is_file()
    }
    expected = set(records) | {"artifact_manifest.json"}
    if actual != expected:
        raise ValueError(
            "Published comparison file set differs from manifest: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    recomputed = [
        scan_public_file(release_dir / relative, relative_path=relative)
        for relative in sorted(records)
    ]
    public_scan = document.get("public_scan")
    if not isinstance(public_scan, dict) or public_scan.get("files") != recomputed:
        raise ValueError("Published comparison public signature/privacy scan differs")
    scan_public_file(manifest_path, relative_path="artifact_manifest.json")
    weight_files = sorted(
        relative for relative in records if Path(relative).suffix.lower() in WEIGHT_SUFFIXES
    )
    evidence = load_json_strict(release_dir / "evidence_manifest.json")
    derived_images = {
        str(record.get("path") or "").replace("\\", "/")
        for record in evidence.get("derived_images", [])
        if isinstance(record, dict)
    }
    image_files = {
        relative for relative in records if Path(relative).suffix.lower() in IMAGE_SUFFIXES
    }
    raw_images = sorted(image_files - derived_images)
    if (
        weight_files
        or raw_images
        or document.get("weights_included") is not False
        or document.get("raw_images_included") is not False
    ):
        raise ValueError(
            f"Published comparison raw/weight scan failed: raw={raw_images}, weights={weight_files}"
        )
    validate_formal_comparison(release_dir)
    return {
        "status": "PASS",
        "files": sorted(actual),
        "public_file_records": recomputed,
        "raw_image_files": raw_images,
        "weight_files": weight_files,
    }


def validate_published_run_release(
    report_root: Path,
    weight_root: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Validate exact public report/weight file sets and recompute every public scan."""
    report_root = report_root.resolve()
    weight_root = weight_root.resolve()
    project_root = project_root.resolve()
    manifest_path = report_root / "artifact_manifest.json"
    document = load_json_strict(manifest_path)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 3
        or document.get("status") != "PASS"
        or document.get("formal_release") is not True
    ):
        raise ValueError("Published run manifest is not a schema-3 formal PASS")
    raw_records = document.get("published_evidence")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Published run evidence list is missing")
    records: dict[str, dict[str, Any]] = {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("Published run evidence record must be an object")
        relative, path = _contained_manifest_path(
            report_root,
            raw.get("path"),
            label="published run evidence",
        )
        if relative == "artifact_manifest.json" or relative in records:
            raise ValueError(f"Duplicate/reserved published run path: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Published run report is missing: {relative}")
        try:
            _require_exact_record_size(raw, path, label="published run report")
        except ValueError as exc:
            raise ValueError(f"Published run report hash/size mismatch: {relative}") from exc
        if str(raw.get("published_sha256", "")).lower() != sha256_file(path):
            raise ValueError(f"Published run report hash/size mismatch: {relative}")
        records[relative] = raw
    reports = document.get("reports")
    if not isinstance(reports, list) or reports != sorted(records):
        raise ValueError("Published run reports list differs from evidence records")
    actual_reports = {
        path.relative_to(report_root).as_posix()
        for path in report_root.rglob("*")
        if path.is_file()
    }
    expected_reports = set(records) | {"artifact_manifest.json"}
    if actual_reports != expected_reports:
        raise ValueError(
            "Published run report file set differs from manifest: "
            f"missing={sorted(expected_reports - actual_reports)}, "
            f"extra={sorted(actual_reports - expected_reports)}"
        )
    report_scans = [
        scan_public_file(report_root / relative, relative_path=relative)
        for relative in sorted(records)
    ]
    scan_public_file(manifest_path, relative_path="artifact_manifest.json")

    checkpoint_record = document.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise ValueError("Published run checkpoint record is missing")
    checkpoint_text = str(checkpoint_record.get("path") or "").replace("\\", "/")
    checkpoint_path = (project_root / checkpoint_text).resolve()
    try:
        checkpoint_relative = checkpoint_path.relative_to(weight_root).as_posix()
    except ValueError as exc:
        raise ValueError("Published run checkpoint path escapes its weight release") from exc
    if not checkpoint_path.is_file():
        raise ValueError("Published run checkpoint hash/size mismatch")
    try:
        _require_exact_record_size(
            checkpoint_record,
            checkpoint_path,
            label="published run checkpoint",
        )
    except ValueError as exc:
        raise ValueError("Published run checkpoint hash/size mismatch") from exc
    if str(checkpoint_record.get("sha256", "")).lower() != sha256_file(checkpoint_path):
        raise ValueError("Published run checkpoint hash/size mismatch")
    actual_weights = {
        path.relative_to(weight_root).as_posix()
        for path in weight_root.rglob("*")
        if path.is_file()
    }
    if actual_weights != {checkpoint_relative}:
        raise ValueError(
            "Published weight file set must contain exactly the declared checkpoint: "
            f"actual={sorted(actual_weights)}"
        )
    weight_scan = scan_public_file(
        checkpoint_path,
        relative_path=(Path("weights") / checkpoint_relative).as_posix(),
    )
    model = _normalized_model(document.get("model"))
    if model.startswith("yolox"):
        from .checkpoint_publishing import validate_yolox_checkpoint_proof

        validate_yolox_checkpoint_proof(
            checkpoint_path,
            checkpoint_record,
            project_root=project_root,
        )
    elif model.startswith("yolo11"):
        if not (
            checkpoint_record.get("metadata_sanitized") is True
            and checkpoint_record.get("state_dict_bitwise_equal") is True
            and checkpoint_record.get("source_forward_captured_before_scrub") is True
            and checkpoint_record.get("forward_max_abs_difference") == 0.0
            and checkpoint_record.get("ultralytics_load") == "PASS"
        ):
            raise ValueError("Published YOLO11 checkpoint proof is not an exact formal PASS")
    else:
        raise ValueError(f"Unsupported published checkpoint model: {document.get('model')!r}")
    public_scan = document.get("public_scan")
    if (
        not isinstance(public_scan, dict)
        or public_scan.get("report_files") != report_scans
        or public_scan.get("weight_files") != [weight_scan]
    ):
        raise ValueError("Published run signature/privacy scan differs from recomputed evidence")
    report_weight_files = sorted(
        relative for relative in records if Path(relative).suffix.lower() in WEIGHT_SUFFIXES
    )
    formal_images = {
        (Path("formal_comparison") / record["path"]).as_posix()
        for record in load_json_strict(
            report_root / "formal_comparison" / "evidence_manifest.json"
        ).get("derived_images", [])
        if isinstance(record, dict)
    }
    report_images = {
        relative for relative in records if Path(relative).suffix.lower() in IMAGE_SUFFIXES
    }
    raw_images = sorted(report_images - formal_images)
    if (
        report_weight_files
        or raw_images
        or document.get("weights_in_reports") is not False
        or document.get("raw_images_included") is not False
        or document.get("promoted_checkpoint_count") != 1
    ):
        raise ValueError(
            f"Published run raw/weight assertions differ: raw={raw_images}, "
            f"report_weights={report_weight_files}"
        )
    validate_formal_comparison(report_root / "formal_comparison")
    return {
        "status": "PASS",
        "report_files": sorted(actual_reports),
        "weight_files": sorted(actual_weights),
        "report_scans": report_scans,
        "weight_scans": [weight_scan],
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

    compatibility = load_json_strict(compatibility_path)
    if not compatibility.get("release_ready", False):
        blockers = ", ".join(
            str(item.get("field")) for item in compatibility.get("release_blockers", [])
        )
        raise ValueError(
            "Comparison is not formal-release ready"
            + (f" ({blockers})" if blockers else "")
        )
    provenance = load_json_strict(provenance_path)
    if provenance.get("status") != "PASS" or compatibility.get("run_provenance") != provenance:
        raise ValueError("Comparison run provenance is missing, failed, or differs from compatibility")
    attestation_path = comparison_dir / "run_provenance_attestation.json"
    if provenance.get("mixed_commits") is True and not attestation_path.is_file():
        raise FileNotFoundError("Mixed-commit comparison is missing its provenance attestation")

    comparison = load_json_strict(comparison_path)
    rows = [row for row in comparison if str(row.get("run_id")) == run_id]
    if len(rows) != 1:
        raise ValueError(f"Comparison must contain run_id exactly once: {run_id}")
    if rows[0].get("status") != "complete":
        raise ValueError(f"Comparison row is not complete: {run_id}")

    sources_manifest = load_json_strict(sources_manifest_path)
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
