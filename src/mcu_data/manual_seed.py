from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

import yaml
from PIL import Image, ImageDraw

from .common import sha256_file, utc_now, write_json
from .contracts import (
    ContractError,
    Ontology,
    canonical_sha256,
    load_ontology,
    require_sha256,
    safe_relative_path,
)


PROPOSAL_SCHEMA = "mcu.manual-seed-proposals.v1"
TASK_SCHEMA = "mcu.manual-seed-task.v1"
SOURCE_SCHEMA = "mcu.manual-seed-source.v1"
REVIEW_TEMPLATE_SCHEMA = "mcu.cvat-review.v1"
ELIGIBLE_DECISION = "CANDIDATE_STM32_DEV_BOARD_NOT_APPROVED"
REVIEW_STATE = "PROPOSED_REQUIRES_HUMAN_REVIEW"

# The accepted Commons evidence uses these exact label/URL pairs.  Keeping the
# allowlist pair-based prevents a permissive label prefix from accidentally
# admitting NonCommercial or NoDerivatives variants.
APPROVED_LICENSE_PAIRS = frozenset(
    {
        ("CC0", "http://creativecommons.org/publicdomain/zero/1.0/deed.en"),
        ("CC0", "https://creativecommons.org/publicdomain/zero/1.0/"),
        ("CC BY 2.0", "https://creativecommons.org/licenses/by/2.0"),
        ("CC BY 2.0", "https://creativecommons.org/licenses/by/2.0/"),
        ("CC BY 3.0", "https://creativecommons.org/licenses/by/3.0"),
        ("CC BY 3.0", "https://creativecommons.org/licenses/by/3.0/"),
        ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0"),
        ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
        ("CC BY-SA 2.0", "https://creativecommons.org/licenses/by-sa/2.0"),
        ("CC BY-SA 2.0", "https://creativecommons.org/licenses/by-sa/2.0/"),
        ("CC BY-SA 3.0", "https://creativecommons.org/licenses/by-sa/3.0"),
        ("CC BY-SA 3.0", "https://creativecommons.org/licenses/by-sa/3.0/"),
        ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0"),
        ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
        ("Public domain", "https://creativecommons.org/publicdomain/mark/1.0/"),
    }
)


class ManualSeedError(ContractError):
    """Raised when a manual-seed review bundle cannot be trusted."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ManualSeedError("YAML mapping keys must be scalar and hashable") from exc
        if duplicate:
            raise ManualSeedError(f"Duplicate YAML key is prohibited: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ManualSeedError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_nonfinite(nested, label=label)
    elif isinstance(value, list):
        for nested in value:
            _reject_nonfinite(nested, label=label)


def _strict_json_value(text: str, *, label: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, nested in pairs:
            if key in value:
                raise ManualSeedError(f"{label} contains a duplicate JSON key: {key!r}")
            value[key] = nested
        return value

    def reject_constant(token: str) -> Any:
        raise ManualSeedError(f"{label} contains a non-finite number: {token}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except ManualSeedError:
        raise
    except json.JSONDecodeError as exc:
        raise ManualSeedError(f"Invalid {label} JSON") from exc
    _reject_nonfinite(value, label=label)
    return value


def _load_strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ManualSeedError(f"Cannot read {label}: {path}") from exc
    value = _strict_json_value(text, label=label)
    if not isinstance(value, dict):
        raise ManualSeedError(f"{label} must be a JSON object: {path}")
    return value


def _read_strict_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ManualSeedError(f"Cannot read {label}: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row_label = f"{label} line {line_number}"
        value = _strict_json_value(line, label=row_label)
        if not isinstance(value, dict):
            raise ManualSeedError(f"{row_label} must be a JSON object")
        rows.append(value)
    return rows


def _load_strict_yaml_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = yaml.load(handle, Loader=_UniqueKeySafeLoader)
    except ManualSeedError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ManualSeedError(f"Invalid {label} YAML: {path}") from exc
    if not isinstance(value, dict):
        raise ManualSeedError(f"{label} must be a YAML mapping: {path}")
    _reject_nonfinite(value, label=label)
    return value


def _load_strict_ontology(path: Path) -> Ontology:
    _load_strict_yaml_object(path, label="ontology")
    return load_ontology(path)


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _required_text(mapping: Mapping[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManualSeedError(f"{field} must be a non-empty string")
    return value.strip()


def _strict_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManualSeedError(f"{field} must be a positive integer")
    return value


def _strict_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManualSeedError(f"{field} must be a non-negative integer")
    return value


def _digest(value: Any, *, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ManualSeedError(f"{field} must be a {length}-character hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ManualSeedError(f"{field} must be hexadecimal") from exc
    return value.lower()


def _utc_timestamp(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManualSeedError(f"{field} must be a non-empty UTC timestamp")
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ManualSeedError(f"{field} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ManualSeedError(f"{field} must be timezone-aware UTC")
    return text


def _repository_path(value: Any, *, field: str, root: Path) -> Path:
    relative = safe_relative_path(value, field=field)
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ManualSeedError(f"{field} escapes the repository root") from exc
    return resolved


def _relative_to_root(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ManualSeedError(f"Path is outside the repository root: {path}") from exc


def _validate_revision_record(record: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    prefix = f"review_records[{index}]"
    page_id = _strict_positive_int(record.get("source_page_id"), field=f"{prefix}.source_page_id")
    revision_id = _strict_positive_int(
        record.get("source_page_revision_id"), field=f"{prefix}.source_page_revision_id"
    )
    page_url = _required_text(record, "source_page_url")
    revision_url = _required_text(record, "source_page_revision_url")
    parsed_page = urlparse(page_url)
    parsed_revision = urlparse(revision_url)
    if parsed_page.scheme != "https" or parsed_page.netloc.casefold() != "commons.wikimedia.org":
        raise ManualSeedError(f"{prefix}.source_page_url must use HTTPS Wikimedia Commons")
    if (
        parsed_revision.scheme != "https"
        or parsed_revision.netloc.casefold() != "commons.wikimedia.org"
        or parse_qs(parsed_revision.query).get("oldid") != [str(revision_id)]
    ):
        raise ManualSeedError(f"{prefix}.source_page_revision_url is not fixed to its oldid")
    timestamp = _utc_timestamp(
        record.get("source_image_timestamp"), field=f"{prefix}.source_image_timestamp"
    )
    license_name = _required_text(record, "license")
    license_url = _required_text(record, "license_url")
    if (license_name, license_url) not in APPROVED_LICENSE_PAIRS:
        raise ManualSeedError(
            f"{prefix} license label/URL pair is not explicitly approved"
        )
    return {
        **dict(record),
        "source_page_id": page_id,
        "source_page_revision_id": revision_id,
        "source_image_sha1": _digest(
            record.get("source_image_sha1"), length=40, field=f"{prefix}.source_image_sha1"
        ),
        "source_metadata_snapshot_sha256": require_sha256(
            record.get("source_metadata_snapshot_sha256"),
            field=f"{prefix}.source_metadata_snapshot_sha256",
        ),
        "sha256": require_sha256(record.get("sha256"), field=f"{prefix}.sha256"),
        "bytes": _strict_positive_int(record.get("bytes"), field=f"{prefix}.bytes"),
        "width": _strict_positive_int(record.get("width"), field=f"{prefix}.width"),
        "height": _strict_positive_int(record.get("height"), field=f"{prefix}.height"),
        "artist": _required_text(record, "artist"),
        "license": license_name,
        "license_url": license_url,
        "source_page_url": page_url,
        "source_page_revision_url": revision_url,
        "source_image_timestamp": timestamp,
    }


def _eligible_probe_records(probe: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    if probe.get("schema_version") != 2:
        raise ManualSeedError("Commons acquisition probe schema_version must be 2")
    if probe.get("status") != "QUARANTINE_REVIEWED_INSUFFICIENT":
        raise ManualSeedError("Commons acquisition probe must remain quarantine-reviewed")
    source = probe.get("source")
    if not isinstance(source, Mapping):
        raise ManualSeedError("Commons acquisition probe source must be an object")
    required_true = (
        "source_page_revision_ids_recorded",
        "source_metadata_snapshot_sha256_recorded",
        "post_download_source_revalidated",
        "quarantine_only",
    )
    if any(source.get(field) is not True for field in required_true):
        raise ManualSeedError("Commons acquisition probe is missing revision-bound provenance")
    if source.get("tracked_images") is not False:
        raise ManualSeedError("Commons source images must remain untracked quarantine data")
    training = probe.get("training_use")
    if not isinstance(training, Mapping) or training.get("allowed") is not False:
        raise ManualSeedError("Commons probe must explicitly prohibit training use")
    records = probe.get("review_records")
    if not isinstance(records, list):
        raise ManualSeedError("Commons acquisition probe review_records must be a list")
    eligible: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, Mapping):
            raise ManualSeedError(f"review_records[{index}] must be an object")
        record = _validate_revision_record(raw, index=index)
        if (
            record.get("collection_route") == "stm32_dev_board"
            and record.get("visual_decision") == ELIGIBLE_DECISION
        ):
            page_id = int(record["source_page_id"])
            if page_id in eligible:
                raise ManualSeedError(f"Duplicate eligible Commons page id: {page_id}")
            eligible[page_id] = record
    if not eligible:
        raise ManualSeedError("Commons probe contains no eligible STM32 development-board candidates")
    grouping = probe.get("dev_board_grouping")
    if not isinstance(grouping, Mapping) or not isinstance(grouping.get("groups"), list):
        raise ManualSeedError("Commons probe is missing conservative development-board groups")
    group_by_page: dict[int, str] = {}
    for index, raw_group in enumerate(grouping["groups"], start=1):
        if not isinstance(raw_group, Mapping):
            raise ManualSeedError(f"dev_board_grouping.groups[{index}] must be an object")
        group_id = _required_text(raw_group, "group_id")
        raw_ids = raw_group.get("source_page_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ManualSeedError(f"Conservative group {group_id} has no page ids")
        for raw_id in raw_ids:
            page_id = _strict_positive_int(raw_id, field=f"group {group_id} source_page_ids")
            if page_id in group_by_page:
                raise ManualSeedError(f"Commons page {page_id} appears in multiple groups")
            group_by_page[page_id] = group_id
    if set(group_by_page) != set(eligible):
        raise ManualSeedError("Conservative grouping must cover each eligible page exactly once")
    return eligible, group_by_page


def _proposal_rows(
    proposal: dict[str, Any],
    *,
    eligible: Mapping[int, dict[str, Any]],
    group_by_page: Mapping[int, str],
    ontology: Ontology,
) -> dict[int, dict[str, Any]]:
    if proposal.get("schema_version") != PROPOSAL_SCHEMA:
        raise ManualSeedError(f"Proposal schema must be {PROPOSAL_SCHEMA}")
    if proposal.get("status") != "DRAFT_REQUIRES_HUMAN_REVIEW":
        raise ManualSeedError("Proposal status must remain DRAFT_REQUIRES_HUMAN_REVIEW")
    if proposal.get("collection_route") != "stm32_dev_board":
        raise ManualSeedError("Proposal collection_route must be stm32_dev_board")
    class_name = proposal.get("class_name")
    if class_name != "stm32_dev_board" or ontology.class_id(str(class_name)) != 4:
        raise ManualSeedError("Proposal class must be frozen ontology class 4 stm32_dev_board")
    if require_sha256(
        proposal.get("ontology_sha256"), field="proposal ontology_sha256"
    ) != ontology.sha256:
        raise ManualSeedError("Proposal ontology hash differs from selected ontology")
    boundary = proposal.get("claim_boundary")
    if not isinstance(boundary, Mapping) or any(
        boundary.get(field) is not False
        for field in ("human_review_completed", "ground_truth", "training_use_allowed")
    ):
        raise ManualSeedError("Proposal claim boundary must prohibit ground-truth and training claims")
    raw_rows = proposal.get("proposals")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ManualSeedError("Proposal list must be non-empty")
    rows: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, Mapping):
            raise ManualSeedError(f"proposals[{index}] must be an object")
        page_id = _strict_positive_int(raw.get("source_page_id"), field=f"proposals[{index}].source_page_id")
        if page_id not in eligible:
            raise ManualSeedError(f"Proposal contains an ineligible Commons page: {page_id}")
        if page_id in rows:
            raise ManualSeedError(f"Duplicate proposal page id: {page_id}")
        if raw.get("leakage_group_id") != group_by_page[page_id]:
            raise ManualSeedError(f"Proposal leakage group differs from probe for page {page_id}")
        bbox = raw.get("bbox_xywh_pixels")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ManualSeedError(f"Proposal bbox must contain four integers for page {page_id}")
        x, y, width, height = [
            _strict_nonnegative_int(value, field=f"proposal page {page_id} bbox")
            for value in bbox
        ]
        if width <= 0 or height <= 0:
            raise ManualSeedError(f"Proposal bbox has zero area for page {page_id}")
        record = eligible[page_id]
        if x + width > record["width"] or y + height > record["height"]:
            raise ManualSeedError(f"Proposal bbox exceeds image bounds for page {page_id}")
        attributes = raw.get("attributes")
        if not isinstance(attributes, Mapping) or set(attributes) != {"occluded", "truncated"}:
            raise ManualSeedError(f"Proposal attributes are incomplete for page {page_id}")
        if any(attributes[field] not in (True, False) for field in attributes):
            raise ManualSeedError(f"Proposal attributes must be booleans for page {page_id}")
        if raw.get("review_state") != REVIEW_STATE:
            raise ManualSeedError(f"Proposal review state is not pending for page {page_id}")
        rows[page_id] = {
            "source_page_id": page_id,
            "leakage_group_id": group_by_page[page_id],
            "bbox_xywh_pixels": [x, y, width, height],
            "attributes": dict(attributes),
            "review_state": REVIEW_STATE,
        }
    if set(rows) != set(eligible):
        missing = sorted(set(eligible) - set(rows))
        raise ManualSeedError(f"Proposal coverage is incomplete: missing={missing}")
    return rows


def _collector_records(
    path: Path,
    *,
    eligible: Mapping[int, dict[str, Any]],
    expected_config_sha256: str,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(
        _read_strict_jsonl(path, label="Commons collector records"), start=1
    ):
        if not isinstance(raw, Mapping):
            raise ManualSeedError(f"Collector record {index} must be an object")
        page_id = _strict_positive_int(
            raw.get("source_page_id"), field=f"collector record {index} source_page_id"
        )
        if page_id not in eligible:
            continue
        if page_id in rows:
            raise ManualSeedError(f"Duplicate collector record for Commons page {page_id}")
        if raw.get("collector_schema_version") != 2 or raw.get("status") != "ACCEPTED":
            raise ManualSeedError(f"Collector record is not an accepted schema-v2 row: {page_id}")
        if raw.get("class_name") != "stm32_dev_board":
            raise ManualSeedError(f"Collector record class differs for Commons page {page_id}")
        if raw.get("qa_status") != "PENDING_HUMAN_REVIEW" or raw.get(
            "training_eligibility"
        ) != "PROHIBITED_PENDING_HUMAN_REVIEW":
            raise ManualSeedError(f"Collector record lost its review/training prohibition: {page_id}")
        if require_sha256(
            raw.get("collector_config_sha256"),
            field=f"collector record {page_id} collector_config_sha256",
        ) != expected_config_sha256:
            raise ManualSeedError(f"Collector configuration hash differs for Commons page {page_id}")
        probe = eligible[page_id]
        exact_fields = (
            "source_page_title",
            "source_page_url",
            "source_page_revision_url",
            "source_image_timestamp",
            "source_image_sha1",
            "source_metadata_snapshot_sha256",
            "sha256",
            "bytes",
            "width",
            "height",
            "license",
            "license_url",
            "artist",
        )
        for field in exact_fields:
            if raw.get(field) != probe.get(field):
                raise ManualSeedError(
                    f"Collector/probe field mismatch for Commons page {page_id}: {field}"
                )
        if raw.get("source_page_latest_revision_id") != probe["source_page_revision_id"]:
            raise ManualSeedError(f"Collector/probe revision mismatch for Commons page {page_id}")
        rows[page_id] = dict(raw)
    if set(rows) != set(eligible):
        missing = sorted(set(eligible) - set(rows))
        raise ManualSeedError(f"Collector records do not cover all eligible pages: missing={missing}")
    return rows


def _source_image(image_root: Path, page_id: int) -> Path:
    route_root = (image_root / "stm32_dev_board").resolve()
    if not route_root.is_dir():
        raise FileNotFoundError(route_root)
    matches = sorted(route_root.glob(f"commons_{page_id}_*.jpg"))
    if len(matches) != 1:
        raise ManualSeedError(
            f"Expected exactly one quarantined image for Commons page {page_id}; found {len(matches)}"
        )
    path = matches[0]
    if path.is_symlink():
        raise ManualSeedError(f"Quarantined image must not be a symlink: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(route_root)
    except ValueError as exc:
        raise ManualSeedError(f"Quarantined image escapes its route root: {path}") from exc
    return resolved


def _validate_image(path: Path, record: Mapping[str, Any]) -> None:
    if path.stat().st_size != record["bytes"]:
        raise ManualSeedError(f"Quarantined image byte count differs from probe: {path.name}")
    if sha256_file(path) != record["sha256"]:
        raise ManualSeedError(f"Quarantined image SHA-256 differs from probe: {path.name}")
    if _sha1_file(path) != record["source_image_sha1"]:
        raise ManualSeedError(f"Quarantined image SHA-1 differs from probe: {path.name}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG" or image.size != (record["width"], record["height"]):
                raise ManualSeedError(f"Quarantined image format/dimensions differ: {path.name}")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ManualSeedError(f"Quarantined image cannot be decoded: {path.name}") from exc


def _write_preview(source: Path, output: Path, bbox: list[int], label: str) -> None:
    with Image.open(source) as opened:
        opened.load()
        image = opened.convert("RGB")
    scale = min(1.0, 1400.0 / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        )
    x, y, width, height = bbox
    scaled = [round(value * scale) for value in (x, y, width, height)]
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(max(image.size) / 500))
    draw.rectangle(
        (scaled[0], scaled[1], scaled[0] + scaled[2], scaled[1] + scaled[3]),
        outline=(220, 20, 60),
        width=line_width,
    )
    draw.text((scaled[0] + line_width, scaled[1] + line_width), label, fill=(220, 20, 60))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="JPEG", quality=90, optimize=True)


def _write_images_zip(path: Path, files: list[tuple[Path, str]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for source, relative in files:
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())


def prepare_manual_seed_task(
    *,
    probe_path: Path,
    collector_records_path: Path,
    collection_config_path: Path,
    proposal_path: Path,
    image_root: Path,
    ontology_path: Path,
    output_dir: Path,
    run_id: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root if project_root is not None else _project_root()).resolve()
    probe_path = probe_path.resolve()
    collector_records_path = collector_records_path.resolve()
    collection_config_path = collection_config_path.resolve()
    proposal_path = proposal_path.resolve()
    image_root = image_root.resolve()
    ontology_path = ontology_path.resolve()
    output_dir = output_dir.resolve()
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id):
        raise ManualSeedError("run_id must contain only letters, numbers, dot, underscore, or hyphen")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing manual-seed task: {output_dir}")
    for path in (
        probe_path,
        collector_records_path,
        collection_config_path,
        proposal_path,
        ontology_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    proposal = _load_strict_yaml_object(proposal_path, label="manual-seed proposals")
    declared_probe = proposal.get("source_probe")
    if not isinstance(declared_probe, Mapping):
        raise ManualSeedError("Proposal source_probe must be an object")
    resolved_declared_probe = _repository_path(
        declared_probe.get("path"), field="proposal source_probe.path", root=root
    )
    if resolved_declared_probe != probe_path:
        raise ManualSeedError("Selected probe path differs from proposal source_probe.path")
    probe_sha = sha256_file(probe_path)
    if require_sha256(
        declared_probe.get("sha256"), field="proposal source_probe.sha256"
    ) != probe_sha:
        raise ManualSeedError("Commons probe SHA-256 differs from proposal binding")
    if declared_probe.get("schema_version") != 2:
        raise ManualSeedError("Proposal must bind Commons probe schema version 2")
    probe = _load_strict_json_object(probe_path, label="Commons acquisition probe")
    ontology = _load_strict_ontology(ontology_path)
    eligible, group_by_page = _eligible_probe_records(probe)
    expected_config_sha = require_sha256(
        probe["source"].get("collection_config_sha256"),
        field="probe source.collection_config_sha256",
    )
    if sha256_file(collection_config_path) != expected_config_sha:
        raise ManualSeedError("Live Commons collection configuration differs from the probe")
    _load_strict_yaml_object(
        collection_config_path, label="Commons collection configuration"
    )
    collector_rows = _collector_records(
        collector_records_path,
        eligible=eligible,
        expected_config_sha256=expected_config_sha,
    )
    proposals = _proposal_rows(
        proposal,
        eligible=eligible,
        group_by_page=group_by_page,
        ontology=ontology,
    )

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir()
    try:
        image_rows: list[dict[str, Any]] = []
        coco_images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        attribution_rows: list[dict[str, Any]] = []
        copied_files: list[tuple[Path, str]] = []
        license_ids: dict[tuple[str, str], int] = {}
        licenses: list[dict[str, Any]] = []
        for annotation_id, page_id in enumerate(sorted(eligible), start=1):
            record = eligible[page_id]
            proposal_row = proposals[page_id]
            source = _source_image(image_root, page_id)
            _validate_image(source, record)
            relative = f"images/{source.name}"
            copied = temporary / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, copied)
            if sha256_file(copied) != record["sha256"]:
                raise ManualSeedError(f"Copied image hash differs for Commons page {page_id}")
            stable_id = f"commons:{page_id}:rev:{record['source_page_revision_id']}"
            binding = {
                "image_id": stable_id,
                "path": relative,
                "sha256": record["sha256"],
                "width": record["width"],
                "height": record["height"],
                "role": "unlabeled_train",
            }
            image_rows.append(binding)
            license_key = (record["license"], record["license_url"])
            if license_key not in license_ids:
                license_ids[license_key] = len(licenses) + 1
                licenses.append(
                    {"id": license_ids[license_key], "name": license_key[0], "url": license_key[1]}
                )
            coco_images.append(
                {
                    "id": annotation_id,
                    "file_name": relative,
                    "width": record["width"],
                    "height": record["height"],
                    "mcu_image_id": stable_id,
                    "sha256": record["sha256"],
                    "license": license_ids[license_key],
                    "source_page_id": page_id,
                    "source_page_revision_id": record["source_page_revision_id"],
                    "source_page_revision_url": record["source_page_revision_url"],
                    "leakage_group_id": proposal_row["leakage_group_id"],
                }
            )
            bbox = proposal_row["bbox_xywh_pixels"]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": annotation_id,
                    "category_id": ontology.class_id("stm32_dev_board"),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "attributes": proposal_row["attributes"],
                    "review_state": REVIEW_STATE,
                }
            )
            attribution_rows.append(
                {
                    "image_id": stable_id,
                    "source_page_title": record["source_page_title"],
                    "source_page_revision_url": record["source_page_revision_url"],
                    "artist": record["artist"],
                    "license": record["license"],
                    "license_url": record["license_url"],
                    "source_image_sha1": record["source_image_sha1"],
                    "download_sha256": record["sha256"],
                }
            )
            _write_preview(
                source,
                temporary / "previews" / f"{source.stem}.proposal.jpg",
                bbox,
                "stm32_dev_board | PROPOSAL",
            )
            copied_files.append((copied, relative))

        image_rows.sort(key=lambda row: (str(row["path"]), str(row["image_id"])))
        image_list_sha = canonical_sha256(image_rows)
        roundtrip_bindings = [
            {key: row[key] for key in ("image_id", "path", "sha256", "width", "height")}
            for row in image_rows
        ]
        class_map = {str(key): value for key, value in sorted(ontology.classes_by_id.items())}
        reference = {
            "info": {
                "description": "Revision-bound Commons STM32 development-board proposals; human review required",
                "annotation_state": "PENDING_HUMAN_REVIEW",
                "ground_truth": False,
                "training_use_allowed": False,
                "source_probe_sha256": probe_sha,
                "proposal_sha256": sha256_file(proposal_path),
                "ontology_sha256": ontology.sha256,
            },
            "licenses": licenses,
            "images": coco_images,
            "annotations": annotations,
            "categories": [
                {"id": key, "name": value, "ontology_class_id": key}
                for key, value in sorted(ontology.classes_by_id.items())
            ],
        }
        reference_path = temporary / "reference.coco.json"
        write_json(reference_path, reference)
        attribution_path = temporary / "attribution.json"
        write_json(
            attribution_path,
            {
                "schema_version": "mcu.source-attribution.v1",
                "generated_at_utc": utc_now(),
                "source_probe_sha256": probe_sha,
                "records": attribution_rows,
            },
        )
        images_zip = temporary / "cvat_images.zip"
        _write_images_zip(images_zip, copied_files)
        previews = sorted((temporary / "previews").glob("*.jpg"))
        task = {
            "schema_version": TASK_SCHEMA,
            "run_id": run_id,
            "status": "complete",
            "preparation_status": "PASS_REVIEW_BUNDLE_ONLY",
            "annotation_state": "PENDING_HUMAN_REVIEW",
            "preparation_kind": "manual_seed",
            "created_at_utc": utc_now(),
            "source_binding": {
                "schema_version": SOURCE_SCHEMA,
                "dataset_id": "wikimedia_stm32_dev_board_manual_seed_v1",
                "role": "unlabeled_train",
                "ontology_sha256": ontology.sha256,
                "image_list_sha256": image_list_sha,
                "image_count": len(image_rows),
                "images": image_rows,
            },
            "source_provenance": {
                "probe_path": _relative_to_root(probe_path, root=root),
                "probe_sha256": probe_sha,
                "probe_schema_version": 2,
                "collector_records_path": _relative_to_root(
                    collector_records_path, root=root
                ),
                "collector_records_sha256": sha256_file(collector_records_path),
                "collector_record_count": len(collector_rows),
                "collection_config_path": _relative_to_root(
                    collection_config_path, root=root
                ),
                "proposal_path": _relative_to_root(proposal_path, root=root),
                "proposal_sha256": sha256_file(proposal_path),
                "collection_config_sha256": probe["source"]["collection_config_sha256"],
                "revision_bound": True,
                "license_evidence_complete": True,
                "conservative_grouping_complete": True,
                "human_review_complete": False,
            },
            "ontology": ontology.record(),
            "class_map": class_map,
            "class_map_sha256": canonical_sha256(class_map),
            "protocol": {
                "proposal_source": "model_assisted_visual_draft",
                "human_review_required": True,
                "cvat_roundtrip_required": True,
                "automatic_promotion_to_training": False,
                "proposed_annotations_are_ground_truth": False,
                "generative_ai_used_for_images": False,
                "model_assisted_bbox_proposals": True,
            },
            "reference": {
                "path": "reference.coco.json",
                "sha256": sha256_file(reference_path),
                "image_bindings_sha256": canonical_sha256(roundtrip_bindings),
                "images": len(coco_images),
                "annotations": len(annotations),
            },
            "artifacts": {
                "cvat_images_zip": {
                    "path": "cvat_images.zip",
                    "sha256": sha256_file(images_zip),
                    "bytes": images_zip.stat().st_size,
                },
                "attribution": {
                    "path": "attribution.json",
                    "sha256": sha256_file(attribution_path),
                },
                "preview_count": len(previews),
                "preview_sha256": {
                    path.name: sha256_file(path) for path in previews
                },
            },
            "training_use": {
                "allowed": False,
                "reason": "Human review and CVAT COCO round-trip have not been completed",
                "formal_evaluation_allowed": False,
                "approved_images": 0,
                "approved_annotations": 0,
            },
        }
        task_path = temporary / "run_manifest.json"
        write_json(task_path, task)
        review_template = {
            "schema_version": REVIEW_TEMPLATE_SCHEMA,
            "status": "DRAFT_UNRESOLVED_DO_NOT_PROMOTE",
            "pending_run_manifest_sha256": sha256_file(task_path),
            "source_image_list_sha256": image_list_sha,
            "ontology_sha256": ontology.sha256,
            "class_map_sha256": task["class_map_sha256"],
            "cvat_export_sha256": None,
            "roundtrip_report_sha256": None,
            "roundtrip_reference_sha256": task["reference"]["sha256"],
            "roundtrip_image_bindings_sha256": task["reference"]["image_bindings_sha256"],
            "reviewer": {"id": None, "name": None},
            "approved_at_utc": None,
            "cvat": {"task_id": None, "job_ids": []},
            "images": [
                {
                    **{key: row[key] for key in ("image_id", "path", "sha256", "width", "height")},
                    "disposition": "UNRESOLVED",
                }
                for row in image_rows
            ],
        }
        write_json(temporary / "review_manifest.template.json", review_template)
        temporary.replace(output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return task


def validate_manual_seed_task(
    *,
    task_manifest_path: Path,
    ontology_path: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = (project_root if project_root is not None else _project_root()).resolve()
    task_manifest_path = task_manifest_path.resolve()
    task_dir = task_manifest_path.parent
    task = _load_strict_json_object(
        task_manifest_path, label="manual-seed task manifest"
    )
    if task.get("schema_version") != TASK_SCHEMA:
        raise ManualSeedError(f"Manual-seed task schema must be {TASK_SCHEMA}")
    if (
        task.get("status") != "complete"
        or task.get("preparation_status") != "PASS_REVIEW_BUNDLE_ONLY"
        or task.get("annotation_state") != "PENDING_HUMAN_REVIEW"
        or task.get("preparation_kind") != "manual_seed"
    ):
        raise ManualSeedError("Manual-seed task is not a complete review-only preparation")
    training = task.get("training_use")
    if not isinstance(training, Mapping) or any(
        training.get(field) is not False for field in ("allowed", "formal_evaluation_allowed")
    ):
        raise ManualSeedError("Manual-seed task must prohibit training and formal evaluation")
    if training.get("approved_images") != 0 or training.get("approved_annotations") != 0:
        raise ManualSeedError("Manual-seed task must contain zero approved items")
    protocol = task.get("protocol")
    if not isinstance(protocol, Mapping) or (
        protocol.get("human_review_required") is not True
        or protocol.get("cvat_roundtrip_required") is not True
        or protocol.get("automatic_promotion_to_training") is not False
        or protocol.get("proposed_annotations_are_ground_truth") is not False
    ):
        raise ManualSeedError("Manual-seed task review protocol is not fail-closed")
    ontology = _load_strict_ontology(ontology_path)
    if task.get("ontology") != ontology.record():
        raise ManualSeedError("Manual-seed task ontology differs from the selected ontology")
    class_map = {str(key): value for key, value in sorted(ontology.classes_by_id.items())}
    if task.get("class_map") != class_map or require_sha256(
        task.get("class_map_sha256"), field="task class_map_sha256"
    ) != canonical_sha256(class_map):
        raise ManualSeedError("Manual-seed task class map differs from the frozen ontology")

    provenance = task.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise ManualSeedError("Manual-seed task source_provenance must be an object")
    required_provenance_true = (
        "revision_bound",
        "license_evidence_complete",
        "conservative_grouping_complete",
    )
    if any(provenance.get(field) is not True for field in required_provenance_true):
        raise ManualSeedError("Manual-seed task provenance is incomplete")
    if provenance.get("human_review_complete") is not False:
        raise ManualSeedError("Manual-seed task must not claim completed human review")
    probe_path = _repository_path(
        provenance.get("probe_path"), field="task source_provenance.probe_path", root=root
    )
    proposal_path = _repository_path(
        provenance.get("proposal_path"), field="task source_provenance.proposal_path", root=root
    )
    collector_path = _repository_path(
        provenance.get("collector_records_path"),
        field="task source_provenance.collector_records_path",
        root=root,
    )
    config_path = _repository_path(
        provenance.get("collection_config_path"),
        field="task source_provenance.collection_config_path",
        root=root,
    )
    expected_paths = (probe_path, proposal_path, collector_path, config_path, ontology_path.resolve())
    if any(not path.is_file() for path in expected_paths):
        raise ManualSeedError("Manual-seed task source evidence is missing")
    probe_sha = require_sha256(provenance.get("probe_sha256"), field="task probe_sha256")
    proposal_sha = require_sha256(
        provenance.get("proposal_sha256"), field="task proposal_sha256"
    )
    collector_sha = require_sha256(
        provenance.get("collector_records_sha256"), field="task collector_records_sha256"
    )
    config_sha = require_sha256(
        provenance.get("collection_config_sha256"), field="task collection_config_sha256"
    )
    for path, expected, label in (
        (probe_path, probe_sha, "probe"),
        (proposal_path, proposal_sha, "proposal"),
        (collector_path, collector_sha, "collector records"),
        (config_path, config_sha, "collection config"),
    ):
        if sha256_file(path) != expected:
            raise ManualSeedError(f"Manual-seed task {label} hash differs")
    probe = _load_strict_json_object(probe_path, label="Commons acquisition probe")
    eligible, group_by_page = _eligible_probe_records(probe)
    if require_sha256(
        probe["source"].get("collection_config_sha256"),
        field="probe collection_config_sha256",
    ) != config_sha:
        raise ManualSeedError("Manual-seed task collection config differs from probe")
    _load_strict_yaml_object(
        config_path, label="Commons collection configuration"
    )
    collector_rows = _collector_records(
        collector_path, eligible=eligible, expected_config_sha256=config_sha
    )
    if provenance.get("collector_record_count") != len(collector_rows):
        raise ManualSeedError("Manual-seed task collector record count differs")
    proposal = _load_strict_yaml_object(proposal_path, label="manual-seed proposals")
    proposals = _proposal_rows(
        proposal, eligible=eligible, group_by_page=group_by_page, ontology=ontology
    )

    source_binding = task.get("source_binding")
    if not isinstance(source_binding, Mapping) or source_binding.get("schema_version") != SOURCE_SCHEMA:
        raise ManualSeedError(f"Manual-seed source schema must be {SOURCE_SCHEMA}")
    if source_binding.get("role") != "unlabeled_train":
        raise ManualSeedError("Manual-seed source role must be unlabeled_train")
    if source_binding.get("ontology_sha256") != ontology.sha256:
        raise ManualSeedError("Manual-seed source ontology differs")
    raw_images = source_binding.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ManualSeedError("Manual-seed source binding images must be non-empty")
    normalized_images: list[dict[str, Any]] = []
    image_by_path: dict[str, dict[str, Any]] = {}
    page_by_path: dict[str, int] = {}
    for index, raw in enumerate(raw_images, start=1):
        if not isinstance(raw, Mapping):
            raise ManualSeedError(f"Manual-seed source image {index} must be an object")
        relative = safe_relative_path(raw.get("path"), field=f"source images[{index}].path")
        if relative in image_by_path:
            raise ManualSeedError(f"Duplicate manual-seed source image path: {relative}")
        image_id = _required_text(raw, "image_id")
        if raw.get("role") != "unlabeled_train":
            raise ManualSeedError(f"Manual-seed source image role differs: {relative}")
        width = _strict_positive_int(raw.get("width"), field=f"source image {relative} width")
        height = _strict_positive_int(raw.get("height"), field=f"source image {relative} height")
        digest = require_sha256(raw.get("sha256"), field=f"source image {relative} sha256")
        normalized = {
            "image_id": image_id,
            "path": relative,
            "sha256": digest,
            "width": width,
            "height": height,
            "role": "unlabeled_train",
        }
        copied = (task_dir / relative).resolve()
        try:
            copied.relative_to(task_dir)
        except ValueError as exc:
            raise ManualSeedError(f"Manual-seed source image escapes task directory: {relative}") from exc
        if not copied.is_file() or copied.is_symlink() or sha256_file(copied) != digest:
            raise ManualSeedError(f"Manual-seed copied image is missing or differs: {relative}")
        with Image.open(copied) as opened:
            opened.load()
            if opened.size != (width, height):
                raise ManualSeedError(f"Manual-seed copied image dimensions differ: {relative}")
        prefix = "commons:"
        if not image_id.startswith(prefix) or ":rev:" not in image_id:
            raise ManualSeedError(f"Manual-seed stable image id is invalid: {image_id}")
        try:
            page_id = int(image_id.split(":", 2)[1])
        except (IndexError, ValueError) as exc:
            raise ManualSeedError(f"Manual-seed stable image id is invalid: {image_id}") from exc
        if page_id not in eligible:
            raise ManualSeedError(f"Manual-seed source image is absent from probe: {image_id}")
        record = eligible[page_id]
        expected_id = f"commons:{page_id}:rev:{record['source_page_revision_id']}"
        if image_id != expected_id or (digest, width, height) != (
            record["sha256"],
            record["width"],
            record["height"],
        ):
            raise ManualSeedError(f"Manual-seed source image differs from probe: {image_id}")
        if _sha1_file(copied) != record["source_image_sha1"]:
            raise ManualSeedError(
                f"Manual-seed copied image SHA-1 differs from probe: {image_id}"
            )
        normalized_images.append(normalized)
        image_by_path[relative] = normalized
        page_by_path[relative] = page_id
    normalized_images.sort(key=lambda row: (str(row["path"]), str(row["image_id"])))
    if len(normalized_images) != len(eligible) or set(page_by_path.values()) != set(eligible):
        raise ManualSeedError("Manual-seed source image coverage differs from eligible probe records")
    if source_binding.get("image_count") != len(normalized_images) or require_sha256(
        source_binding.get("image_list_sha256"), field="task source image_list_sha256"
    ) != canonical_sha256(normalized_images):
        raise ManualSeedError("Manual-seed source image-list binding differs")

    reference_record = task.get("reference")
    if not isinstance(reference_record, Mapping):
        raise ManualSeedError("Manual-seed task reference must be an object")
    reference_relative = safe_relative_path(
        reference_record.get("path"), field="task reference.path"
    )
    reference_path = (task_dir / reference_relative).resolve()
    try:
        reference_path.relative_to(task_dir)
    except ValueError as exc:
        raise ManualSeedError("Manual-seed reference escapes task directory") from exc
    reference_sha = require_sha256(
        reference_record.get("sha256"), field="task reference.sha256"
    )
    if not reference_path.is_file() or sha256_file(reference_path) != reference_sha:
        raise ManualSeedError("Manual-seed reference COCO is missing or differs")
    reference = _load_strict_json_object(
        reference_path, label="manual-seed reference COCO"
    )
    if set(reference) != {"info", "licenses", "images", "annotations", "categories"}:
        raise ManualSeedError("Manual-seed reference COCO fields differ")
    expected_info = {
        "description": (
            "Revision-bound Commons STM32 development-board proposals; "
            "human review required"
        ),
        "annotation_state": "PENDING_HUMAN_REVIEW",
        "ground_truth": False,
        "training_use_allowed": False,
        "source_probe_sha256": probe_sha,
        "proposal_sha256": proposal_sha,
        "ontology_sha256": ontology.sha256,
    }
    if reference.get("info") != expected_info:
        raise ManualSeedError("Manual-seed reference info differs from bound evidence")
    categories = reference.get("categories")
    expected_categories = [
        {"id": key, "name": value, "ontology_class_id": key}
        for key, value in sorted(ontology.classes_by_id.items())
    ]
    if categories != expected_categories:
        raise ManualSeedError("Manual-seed reference category map differs")
    reference_images = reference.get("images")
    reference_annotations = reference.get("annotations")
    if not isinstance(reference_images, list) or not isinstance(reference_annotations, list):
        raise ManualSeedError("Manual-seed reference COCO is missing images or annotations")
    path_for_page = {page_id: relative for relative, page_id in page_by_path.items()}
    expected_license_ids: dict[tuple[str, str], int] = {}
    expected_licenses: list[dict[str, Any]] = []
    expected_reference_images: list[dict[str, Any]] = []
    expected_reference_annotations: list[dict[str, Any]] = []
    for annotation_id, page_id in enumerate(sorted(eligible), start=1):
        record = eligible[page_id]
        proposal_row = proposals[page_id]
        license_key = (record["license"], record["license_url"])
        if license_key not in expected_license_ids:
            expected_license_ids[license_key] = len(expected_licenses) + 1
            expected_licenses.append(
                {
                    "id": expected_license_ids[license_key],
                    "name": license_key[0],
                    "url": license_key[1],
                }
            )
        relative = path_for_page[page_id]
        binding = image_by_path[relative]
        expected_reference_images.append(
            {
                "id": annotation_id,
                "file_name": relative,
                "width": record["width"],
                "height": record["height"],
                "mcu_image_id": binding["image_id"],
                "sha256": record["sha256"],
                "license": expected_license_ids[license_key],
                "source_page_id": page_id,
                "source_page_revision_id": record["source_page_revision_id"],
                "source_page_revision_url": record["source_page_revision_url"],
                "leakage_group_id": group_by_page[page_id],
            }
        )
        bbox = proposal_row["bbox_xywh_pixels"]
        expected_reference_annotations.append(
            {
                "id": annotation_id,
                "image_id": annotation_id,
                "category_id": ontology.class_id("stm32_dev_board"),
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "attributes": proposal_row["attributes"],
                "review_state": REVIEW_STATE,
            }
        )
    if reference.get("licenses") != expected_licenses:
        raise ManualSeedError("Manual-seed reference licenses differ from approved evidence")
    if reference_images != expected_reference_images:
        raise ManualSeedError("Manual-seed reference image records differ from bound evidence")
    if reference_annotations != expected_reference_annotations:
        raise ManualSeedError("Manual-seed reference proposals differ from bound proposals")
    expected_bindings = [
        {key: row[key] for key in ("image_id", "path", "sha256", "width", "height")}
        for row in normalized_images
    ]
    if require_sha256(
        reference_record.get("image_bindings_sha256"),
        field="task reference.image_bindings_sha256",
    ) != canonical_sha256(expected_bindings):
        raise ManualSeedError("Manual-seed reference image-binding hash differs")
    if reference_record.get("images") != len(reference_images) or reference_record.get(
        "annotations"
    ) != len(reference_annotations):
        raise ManualSeedError("Manual-seed reference counts differ")

    artifacts = task.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ManualSeedError("Manual-seed task artifacts must be an object")
    zip_record = artifacts.get("cvat_images_zip")
    attribution_record = artifacts.get("attribution")
    if not isinstance(zip_record, Mapping) or not isinstance(attribution_record, Mapping):
        raise ManualSeedError("Manual-seed task artifact records are incomplete")
    zip_path = task_dir / safe_relative_path(zip_record.get("path"), field="cvat_images_zip.path")
    if (
        not zip_path.is_file()
        or zip_path.stat().st_size != zip_record.get("bytes")
        or sha256_file(zip_path) != require_sha256(
            zip_record.get("sha256"), field="cvat_images_zip.sha256"
        )
    ):
        raise ManualSeedError("Manual-seed CVAT image ZIP is missing or differs")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(image_by_path):
            raise ManualSeedError("Manual-seed CVAT image ZIP inventory differs")
        for relative, binding in image_by_path.items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != binding["sha256"]:
                raise ManualSeedError(f"Manual-seed CVAT ZIP image differs: {relative}")
    attribution_path = task_dir / safe_relative_path(
        attribution_record.get("path"), field="attribution.path"
    )
    if not attribution_path.is_file() or sha256_file(attribution_path) != require_sha256(
        attribution_record.get("sha256"), field="attribution.sha256"
    ):
        raise ManualSeedError("Manual-seed attribution artifact is missing or differs")
    attribution = _load_strict_json_object(
        attribution_path, label="manual-seed attribution"
    )
    expected_attribution_records = [
        {
            "image_id": f"commons:{page_id}:rev:{eligible[page_id]['source_page_revision_id']}",
            "source_page_title": eligible[page_id]["source_page_title"],
            "source_page_revision_url": eligible[page_id]["source_page_revision_url"],
            "artist": eligible[page_id]["artist"],
            "license": eligible[page_id]["license"],
            "license_url": eligible[page_id]["license_url"],
            "source_image_sha1": eligible[page_id]["source_image_sha1"],
            "download_sha256": eligible[page_id]["sha256"],
        }
        for page_id in sorted(eligible)
    ]
    if (
        set(attribution)
        != {"schema_version", "generated_at_utc", "source_probe_sha256", "records"}
        or attribution.get("schema_version") != "mcu.source-attribution.v1"
        or attribution.get("source_probe_sha256") != probe_sha
        or attribution.get("records") != expected_attribution_records
    ):
        raise ManualSeedError(
            "Manual-seed attribution records differ from bound source evidence"
        )
    _utc_timestamp(
        attribution.get("generated_at_utc"), field="attribution generated_at_utc"
    )
    preview_hashes = artifacts.get("preview_sha256")
    if not isinstance(preview_hashes, Mapping) or artifacts.get("preview_count") != len(
        normalized_images
    ):
        raise ManualSeedError("Manual-seed preview inventory differs")
    preview_root = task_dir / "previews"
    actual_previews = {path.name: path for path in preview_root.glob("*.jpg") if path.is_file()}
    if set(actual_previews) != set(preview_hashes):
        raise ManualSeedError("Manual-seed preview file set differs")
    for name, expected in preview_hashes.items():
        if sha256_file(actual_previews[name]) != require_sha256(
            expected, field=f"preview {name} sha256"
        ):
            raise ManualSeedError(f"Manual-seed preview differs: {name}")

    review_template_path = task_dir / "review_manifest.template.json"
    review = _load_strict_json_object(
        review_template_path, label="manual-seed review template"
    )
    expected_review = {
        "schema_version": REVIEW_TEMPLATE_SCHEMA,
        "status": "DRAFT_UNRESOLVED_DO_NOT_PROMOTE",
        "pending_run_manifest_sha256": sha256_file(task_manifest_path),
        "source_image_list_sha256": source_binding["image_list_sha256"],
        "ontology_sha256": ontology.sha256,
        "class_map_sha256": canonical_sha256(class_map),
        "cvat_export_sha256": None,
        "roundtrip_report_sha256": None,
        "roundtrip_reference_sha256": reference_sha,
        "roundtrip_image_bindings_sha256": reference_record["image_bindings_sha256"],
        "reviewer": {"id": None, "name": None},
        "approved_at_utc": None,
        "cvat": {"task_id": None, "job_ids": []},
        "images": [
            {
                **{
                    key: row[key]
                    for key in ("image_id", "path", "sha256", "width", "height")
                },
                "disposition": "UNRESOLVED",
            }
            for row in normalized_images
        ],
    }
    if review != expected_review:
        raise ManualSeedError("Manual-seed review template is not unresolved and hash-bound")
    expected_files = {
        "run_manifest.json",
        "reference.coco.json",
        "attribution.json",
        "cvat_images.zip",
        "review_manifest.template.json",
        *image_by_path,
        *(f"previews/{name}" for name in preview_hashes),
    }
    actual_files = {
        path.relative_to(task_dir).as_posix()
        for path in task_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ManualSeedError(
            "Manual-seed task contains missing or unlisted files: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    return {
        "status": "PASS_REVIEW_BUNDLE_ONLY",
        "run_id": task.get("run_id"),
        "task_manifest_sha256": sha256_file(task_manifest_path),
        "source_probe_sha256": probe_sha,
        "proposal_sha256": proposal_sha,
        "collector_records_sha256": collector_sha,
        "ontology_sha256": ontology.sha256,
        "class_map_sha256": canonical_sha256(class_map),
        "images": len(normalized_images),
        "proposed_annotations": len(reference_annotations),
        "human_review_complete": False,
        "training_use_allowed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    root = _project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a revision-bound Commons STM32 manual-seed CVAT review bundle. "
            "The output remains PENDING_HUMAN_REVIEW and is never training-approved."
        )
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=root / "data" / "manifests" / "wikimedia_stm32_v2.acquisition-probe.json",
    )
    parser.add_argument(
        "--collector-records",
        type=Path,
        default=(
            root
            / "data"
            / "quarantine"
            / "wikimedia_commons_v2"
            / "evidence"
            / "stm32_dev_board.sources.jsonl"
        ),
    )
    parser.add_argument(
        "--collection-config",
        type=Path,
        default=root / "configs" / "sources.wikimedia.yaml",
    )
    parser.add_argument(
        "--proposals",
        type=Path,
        default=root / "configs" / "annotation" / "wikimedia_stm32_dev_board_manual_seed_v1.yaml",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=root / "data" / "quarantine" / "wikimedia_commons_v2" / "images",
    )
    parser.add_argument(
        "--ontology", type=Path, default=root / "configs" / "classes.smd_v1.yaml"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = prepare_manual_seed_task(
        probe_path=args.probe,
        collector_records_path=args.collector_records,
        collection_config_path=args.collection_config,
        proposal_path=args.proposals,
        image_root=args.image_root,
        ontology_path=args.ontology,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps(task, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
