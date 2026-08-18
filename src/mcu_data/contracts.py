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
        if label in self.classes_by_id.values():
            return label
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


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return value
