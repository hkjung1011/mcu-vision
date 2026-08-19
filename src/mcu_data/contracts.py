from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from .common import load_yaml, sha256_file


class ContractError(ValueError):
    """Raised when provenance or ontology input is incomplete or ambiguous."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{field} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractError(f"{field} must be hexadecimal") from exc
    return value.lower()


def safe_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or PureWindowsPath(normalized).drive
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ContractError(f"{field} must not be absolute or traverse directories: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class Ontology:
    path: Path
    ontology_id: str
    sha256: str
    classes_by_id: dict[int, str]
    aliases_by_source: dict[str, dict[str, str]]

    @property
    def names(self) -> list[str]:
        return [self.classes_by_id[index] for index in sorted(self.classes_by_id)]

    def source_name(self, dataset_id: str, label: str) -> str:
        aliases = self.aliases_by_source.get(dataset_id, {})
        if label not in aliases:
            raise ContractError(
                f"Source label {label!r} has no ontology mapping for dataset {dataset_id!r}"
            )
        return aliases[label]

    def class_id(self, name: str) -> int:
        for class_id, class_name in self.classes_by_id.items():
            if class_name == name:
                return class_id
        raise ContractError(f"Unknown ontology class: {name!r}")

    def record(self) -> dict[str, Any]:
        return {
            "ontology_id": self.ontology_id,
            "sha256": self.sha256,
            "classes": {str(key): value for key, value in sorted(self.classes_by_id.items())},
        }


def load_ontology(path: Path) -> Ontology:
    resolved = path.resolve()
    document = load_yaml(resolved)
    ontology_id = document.get("ontology_id")
    if not isinstance(ontology_id, str) or not ontology_id:
        raise ContractError("ontology_id must be a non-empty string")
    raw_classes = document.get("classes")
    if not isinstance(raw_classes, Mapping) or not raw_classes:
        raise ContractError("ontology classes must be a non-empty mapping")
    classes_by_id: dict[int, str] = {}
    for name, value in raw_classes.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise ContractError("Every ontology class must be a name-to-mapping entry")
        try:
            class_id = int(value["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"Ontology class {name!r} has an invalid id") from exc
        if class_id < 0 or class_id in classes_by_id:
            raise ContractError(f"Ontology class id is negative or duplicated: {class_id}")
        classes_by_id[class_id] = name
    expected = list(range(len(classes_by_id)))
    if sorted(classes_by_id) != expected:
        raise ContractError(
            f"Ontology class ids must be contiguous from zero; found {sorted(classes_by_id)}"
        )
    aliases_by_source: dict[str, dict[str, str]] = {}
    raw_aliases = document.get("source_aliases", {})
    if not isinstance(raw_aliases, Mapping):
        raise ContractError("source_aliases must be a mapping")
    known_names = set(classes_by_id.values())
    for source_id, aliases in raw_aliases.items():
        if not isinstance(source_id, str) or not isinstance(aliases, Mapping):
            raise ContractError("Every source alias entry must be a mapping")
        normalized: dict[str, str] = {}
        for source_label, canonical_name in aliases.items():
            if not isinstance(source_label, str) or not isinstance(canonical_name, str):
                raise ContractError("Source aliases must map strings to strings")
            if canonical_name not in known_names:
                raise ContractError(
                    f"Alias {source_id}:{source_label} maps to unknown class {canonical_name!r}"
                )
            normalized[source_label] = canonical_name
        aliases_by_source[source_id] = normalized
    return Ontology(
        path=resolved,
        ontology_id=ontology_id,
        sha256=sha256_file(resolved),
        classes_by_id=classes_by_id,
        aliases_by_source=aliases_by_source,
    )


def load_ontology_display_sidecar(
    canonical_path: Path,
    sidecar_path: Path,
) -> dict[str, dict[str, str]]:
    """Bind presentation-only localized text to a frozen canonical ontology."""
    ontology = load_ontology(canonical_path)
    document = load_yaml(sidecar_path.resolve())
    expected_top_level = {
        "schema_version",
        "ontology_id",
        "locale",
        "presentation_only",
        "canonical_source",
        "classes",
    }
    if set(document) != expected_top_level:
        raise ContractError(
            "Display sidecar top-level fields differ: "
            f"missing={sorted(expected_top_level - set(document))}, "
            f"extra={sorted(set(document) - expected_top_level)}"
        )
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or document.get("ontology_id") != ontology.ontology_id
        or document.get("presentation_only") is not True
        or document.get("canonical_source") != canonical_path.name
    ):
        raise ContractError("Display sidecar identity/canonical binding differs")
    locale = document.get("locale")
    if not isinstance(locale, str) or not locale:
        raise ContractError("Display sidecar locale must be a non-empty string")
    raw_classes = document.get("classes")
    if not isinstance(raw_classes, Mapping):
        raise ContractError("Display sidecar classes must be a mapping")
    sidecar_names = list(raw_classes)
    if sidecar_names != ontology.names:
        raise ContractError(
            "Display sidecar class keys/order differ from canonical ontology: "
            f"sidecar={sidecar_names}, canonical={ontology.names}"
        )
    allowed_fields = {"display_name", "description", "terminology_note"}
    result: dict[str, dict[str, str]] = {}
    display_names: set[str] = set()
    for class_name, raw_entry in raw_classes.items():
        if not isinstance(raw_entry, Mapping):
            raise ContractError(f"Display sidecar class {class_name!r} must be a mapping")
        fields = set(raw_entry)
        if not {"display_name", "description"}.issubset(fields) or not fields.issubset(
            allowed_fields
        ):
            raise ContractError(
                f"Display sidecar class {class_name!r} may contain only "
                "display_name, description, and optional terminology_note"
            )
        entry: dict[str, str] = {}
        for field, value in raw_entry.items():
            if not isinstance(value, str) or not value.strip():
                raise ContractError(
                    f"Display sidecar {class_name}.{field} must be a non-empty string"
                )
            entry[field] = value.strip()
        if entry["display_name"] in display_names:
            raise ContractError("Display sidecar display_name values must be unique")
        display_names.add(entry["display_name"])
        result[class_name] = entry
    return result


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return value
