from __future__ import annotations

import math
import re
import statistics
from pathlib import Path
from typing import Any

import yaml

from .methodology import canonical_text_sha256, load_protocol


PAIRED_TWO_SEED_TIER = "paired_2seed_descriptive"
APPROVED_POLICY_SHA256 = {
    "rpi_bootstrap_paired_2seed_release_v1": (
        "1865539e9b3569dd4942d9d17495a3644e059df70259b555bd7985e7bdf76f27"
    ),
}
REQUIRED_ALLOWED_CLAIMS = (
    "per_run_metrics",
    "model_mean",
    "sample_standard_deviation",
    "paired_seed_deltas",
)
REQUIRED_PROHIBITED_CLAIMS = (
    "statistical_significance",
    "population_superiority",
    "production_ready",
    "independent_test",
)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key in formal release policy: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def normalize_model_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _exact_keys(document: dict[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(document)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a non-empty list of strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return list(value)


def _integer_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise ValueError(f"{label} values must use exact YAML integer types")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return list(value)


def _load_unique_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(f"Formal release policy must be UTF-8: {path}") from exc
    if "\x00" in text:
        raise ValueError(f"Formal release policy must not contain NUL bytes: {path}")
    try:
        document = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Formal release policy is invalid YAML: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Formal release policy must be a YAML mapping: {path}")
    return document


def load_formal_release_policy(
    path: Path,
    *,
    base_protocol_path: Path,
) -> dict[str, Any]:
    """Load the frozen release matrix without changing the immutable training protocol."""
    path = path.resolve()
    base_protocol_path = base_protocol_path.resolve()
    document = _load_unique_yaml(path)
    _exact_keys(
        document,
        {
            "schema_version",
            "policy_id",
            "status",
            "evidence_tier",
            "base_protocol",
            "release_matrix",
            "decision_record",
            "statistics",
            "claims",
        },
        label="formal release policy",
    )
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("status") != "FROZEN"
    ):
        raise ValueError("Formal release policy must be schema_version=1 and status=FROZEN")
    policy_id = document.get("policy_id")
    if not isinstance(policy_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", policy_id):
        raise ValueError("Formal release policy_id is invalid")
    actual_policy_sha = canonical_text_sha256(path)
    if APPROVED_POLICY_SHA256.get(policy_id) != actual_policy_sha:
        raise ValueError("Formal release policy SHA-256 is not an approved frozen policy")
    if document.get("evidence_tier") != PAIRED_TWO_SEED_TIER:
        raise ValueError(
            f"Formal release evidence_tier must be {PAIRED_TWO_SEED_TIER!r}"
        )

    base = document.get("base_protocol")
    if not isinstance(base, dict):
        raise ValueError("Formal release base_protocol must be a mapping")
    _exact_keys(base, {"protocol_id", "sha256"}, label="base_protocol")
    expected_base_sha = str(base.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_base_sha):
        raise ValueError("Formal release base_protocol.sha256 is invalid")
    actual_base_sha = canonical_text_sha256(base_protocol_path)
    base_document = load_protocol(base_protocol_path)
    if base.get("protocol_id") != base_document.get("protocol_id"):
        raise ValueError("Formal release policy base protocol ID mismatch")
    if expected_base_sha != actual_base_sha:
        raise ValueError("Formal release policy base protocol SHA-256 mismatch")

    matrix = document.get("release_matrix")
    if not isinstance(matrix, dict):
        raise ValueError("Formal release release_matrix must be a mapping")
    _exact_keys(
        matrix,
        {"models", "seeds", "epochs_per_run", "expected_runs", "exact_pairs"},
        label="release_matrix",
    )
    models = _string_list(matrix.get("models"), label="release_matrix.models")
    if len({normalize_model_name(value) for value in models}) != len(models):
        raise ValueError("release_matrix.models collapse to duplicate normalized model IDs")
    seeds = _integer_list(matrix.get("seeds"), label="release_matrix.seeds")
    epochs = matrix.get("epochs_per_run")
    expected_runs = matrix.get("expected_runs")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("release_matrix.epochs_per_run must be a positive exact integer")
    if type(expected_runs) is not int or expected_runs != len(models) * len(seeds):
        raise ValueError("release_matrix.expected_runs must equal models × seeds")
    if (
        [normalize_model_name(value) for value in models] != ["yolo11m", "yoloxs"]
        or seeds != [42, 43]
        or epochs != 100
        or expected_runs != 4
    ):
        raise ValueError("Paired descriptive release matrix must be exact 2 models × seeds 42/43")
    expected_pairs = [
        [model, seed]
        for model in models
        for seed in seeds
    ]
    if matrix.get("exact_pairs") != expected_pairs:
        raise ValueError("release_matrix.exact_pairs must be the ordered models × seeds matrix")

    decision = document.get("decision_record")
    if not isinstance(decision, dict):
        raise ValueError("Formal release decision_record must be a mapping")
    _exact_keys(
        decision,
        {
            "selection_decided_after_training",
            "reason",
            "included_seed_rule",
            "excluded_seed",
            "excluded_models",
            "exclusion_rule",
        },
        label="decision_record",
    )
    if decision.get("selection_decided_after_training") is not True:
        raise ValueError("Formal release policy must disclose the post-hoc selection decision")
    if type(decision.get("excluded_seed")) is not int:
        raise ValueError("decision_record.excluded_seed must use exact integer type")
    if decision.get("excluded_seed") != 44:
        raise ValueError("Paired descriptive release policy must symmetrically exclude seed 44")
    excluded_models = _string_list(
        decision.get("excluded_models"), label="decision_record.excluded_models"
    )
    if {normalize_model_name(value) for value in excluded_models} != {
        normalize_model_name(value) for value in models
    }:
        raise ValueError("The excluded seed must be excluded symmetrically for both models")
    for field in ("reason", "included_seed_rule", "exclusion_rule"):
        if not isinstance(decision.get(field), str) or not decision[field].strip():
            raise ValueError(f"decision_record.{field} must be a non-empty string")

    statistics_block = document.get("statistics")
    if not isinstance(statistics_block, dict):
        raise ValueError("Formal release statistics must be a mapping")
    _exact_keys(
        statistics_block,
        {"n_per_model", "paired_n", "degrees_of_freedom", "interpretation"},
        label="statistics",
    )
    if (
        statistics_block.get("n_per_model") != len(seeds)
        or statistics_block.get("paired_n") != len(seeds)
        or statistics_block.get("degrees_of_freedom") != len(seeds) - 1
        or statistics_block.get("interpretation") != "descriptive_only"
        or any(
            type(statistics_block.get(field)) is not int
            for field in ("n_per_model", "paired_n", "degrees_of_freedom")
        )
    ):
        raise ValueError("Formal release statistics do not match the selected paired seeds")

    claims = document.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("Formal release claims must be a mapping")
    _exact_keys(claims, {"allowed", "prohibited"}, label="claims")
    allowed = _string_list(claims.get("allowed"), label="claims.allowed")
    prohibited = _string_list(claims.get("prohibited"), label="claims.prohibited")
    if tuple(allowed) != REQUIRED_ALLOWED_CLAIMS:
        raise ValueError("Formal release allowed claims differ from the frozen descriptive tier")
    if tuple(prohibited) != REQUIRED_PROHIBITED_CLAIMS:
        raise ValueError("Formal release prohibited claims differ from the frozen descriptive tier")

    return {
        "schema_version": 1,
        "policy_id": policy_id,
        "policy_sha256": actual_policy_sha,
        "evidence_tier": PAIRED_TWO_SEED_TIER,
        "base_protocol_id": str(base["protocol_id"]),
        "base_protocol_sha256": expected_base_sha,
        "models": models,
        "normalized_models": [normalize_model_name(value) for value in models],
        "seeds": seeds,
        "epochs_per_run": epochs,
        "expected_runs": expected_runs,
        "exact_pairs": expected_pairs,
        "n_per_model": len(seeds),
        "paired_n": len(seeds),
        "degrees_of_freedom": len(seeds) - 1,
        "interpretation": "descriptive_only",
        "claims": {"allowed": allowed, "prohibited": prohibited},
        "decision_record": decision,
    }


def public_policy_binding(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        key: policy[key]
        for key in (
            "policy_id",
            "policy_sha256",
            "evidence_tier",
            "base_protocol_id",
            "base_protocol_sha256",
            "models",
            "seeds",
            "epochs_per_run",
            "expected_runs",
            "exact_pairs",
            "n_per_model",
            "paired_n",
            "degrees_of_freedom",
            "interpretation",
            "claims",
            "decision_record",
        )
    }


def compute_paired_seed_deltas(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Compute paired descriptive deltas in the policy-declared model direction."""
    models = list(policy["models"])
    seeds = list(policy["seeds"])
    first_normalized, second_normalized = [normalize_model_name(value) for value in models]
    by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        seed = row.get("seed")
        if type(seed) is not int:
            raise ValueError("Paired-delta input seeds must use exact integer types")
        pair = (normalize_model_name(row.get("model")), seed)
        if pair in by_pair:
            raise ValueError(f"Duplicate paired-delta row: {pair}")
        by_pair[pair] = row
    expected = {
        (normalize_model_name(model), seed)
        for model in models
        for seed in seeds
    }
    if set(by_pair) != expected:
        raise ValueError("Paired-delta rows do not form the exact release-policy matrix")

    metrics = (
        "ap50_95",
        "ap50",
        "ap75",
        "ar100",
        "precision",
        "recall",
        "f1",
        "latency_p50_ms",
        "latency_p95_ms",
        "fps",
    )
    by_metric: dict[str, Any] = {}
    for metric in metrics:
        pairs = []
        deltas = []
        for seed in seeds:
            first_raw = by_pair[(first_normalized, seed)].get(metric)
            second_raw = by_pair[(second_normalized, seed)].get(metric)
            if not isinstance(first_raw, (int, float)) or not isinstance(
                second_raw, (int, float)
            ):
                raise ValueError(f"Paired-delta metric is missing or nonnumeric: {metric}/seed{seed}")
            first = float(first_raw)
            second = float(second_raw)
            if not math.isfinite(first) or not math.isfinite(second):
                raise ValueError(f"Paired-delta metric is nonfinite: {metric}/seed{seed}")
            delta = first - second
            pairs.append(
                {
                    "seed": seed,
                    "model_a_value": first,
                    "model_b_value": second,
                    "delta_model_a_minus_model_b": delta,
                }
            )
            deltas.append(delta)
        by_metric[metric] = {
            "pairs": pairs,
            "mean_delta": statistics.fmean(deltas),
            "sample_sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
        }
    return {
        "model_a": models[0],
        "model_b": models[1],
        "direction": "model_a_minus_model_b",
        "seeds": seeds,
        "paired_n": len(seeds),
        "degrees_of_freedom": len(seeds) - 1,
        "interpretation": "descriptive_only",
        "by_metric": by_metric,
    }


def write_policy_snapshot(source: Path, destination: Path) -> str:
    text = source.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    destination.write_text(normalized, encoding="utf-8", newline="\n")
    return canonical_text_sha256(destination)
