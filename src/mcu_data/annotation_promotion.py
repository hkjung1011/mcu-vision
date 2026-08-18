from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import sha256_file, utc_now, write_json
from .contracts import (
    ContractError,
    canonical_sha256,
    load_json_object,
    load_ontology,
    require_sha256,
    safe_relative_path,
)
from .cvat_roundtrip import ROUNDTRIP_SCHEMA


REVIEW_SCHEMA = "mcu.cvat-review.v1"
PROMOTION_SCHEMA = "mcu.annotation-promotion.v1"
APPROVED_DISPOSITIONS = {"approved", "corrected", "confirmed_empty", "rejected"}
AUTOLABEL_SOURCE_SCHEMA = "mcu.autolabel-source.v1"


class PromotionError(ValueError):
    """Raised when pending annotations do not meet the human-approval gate."""


def _required_nonempty(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PromotionError(f"{field} must be a non-empty string")
    return value.strip()


def _approved_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise PromotionError("approved_at_utc must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PromotionError("approved_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PromotionError("approved_at_utc must include a timezone")
    return value


def _pending_images(
    pending: dict[str, Any], *, ontology_sha256: str
) -> tuple[dict[str, dict[str, Any]], str]:
    if pending.get("annotation_state") != "PENDING_HUMAN_REVIEW":
        raise PromotionError("Pending run annotation_state must be PENDING_HUMAN_REVIEW")
    if pending.get("status") != "complete":
        raise PromotionError("Pending autolabel run must be complete")
    source_binding = pending.get("source_binding")
    if not isinstance(source_binding, dict) or source_binding.get("role") != "unlabeled_train":
        raise PromotionError("Pending run source role must be unlabeled_train")
    if source_binding.get("schema_version") != AUTOLABEL_SOURCE_SCHEMA:
        raise PromotionError(
            f"Pending source binding schema must be {AUTOLABEL_SOURCE_SCHEMA}"
        )
    if require_sha256(
        source_binding.get("ontology_sha256"),
        field="pending source ontology_sha256",
    ) != ontology_sha256:
        raise PromotionError("Pending source binding ontology differs from selected ontology")
    pending_ontology = pending.get("ontology")
    if not isinstance(pending_ontology, dict) or require_sha256(
        pending_ontology.get("sha256"), field="pending ontology.sha256"
    ) != ontology_sha256:
        raise PromotionError("Pending run ontology differs from selected ontology")
    protocol = pending.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("automatic_promotion_to_training") is not False:
        raise PromotionError("Pending run must explicitly disable automatic promotion")
    raw_images = source_binding.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise PromotionError("Pending run source binding must contain images")
    images: dict[str, dict[str, Any]] = {}
    binding_rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_images, start=1):
        if not isinstance(row, dict):
            raise PromotionError(f"Pending source image {index} must be an object")
        path = safe_relative_path(row.get("path"), field=f"pending images[{index}].path")
        if path in images:
            raise PromotionError(f"Duplicate pending source image: {path}")
        if row.get("role") != "unlabeled_train":
            raise PromotionError(f"Pending source image is not unlabeled_train: {path}")
        try:
            width = int(row["width"])
            height = int(row["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PromotionError(f"Pending source image dimensions are invalid: {path}") from exc
        if width <= 0 or height <= 0:
            raise PromotionError(f"Pending source image dimensions are invalid: {path}")
        normalized = {
            "path": path,
            "sha256": require_sha256(
                row.get("sha256"), field=f"pending images[{index}].sha256"
            ),
            "width": width,
            "height": height,
            "role": "unlabeled_train",
        }
        images[path] = normalized
        binding_rows.append(normalized)
    binding_rows.sort(key=lambda row: str(row["path"]))
    image_list_sha = require_sha256(
        source_binding.get("image_list_sha256"), field="pending source image_list_sha256"
    )
    if canonical_sha256(binding_rows) != image_list_sha:
        raise PromotionError("Pending source image_list_sha256 does not match its image records")
    return images, image_list_sha


def promote_reviewed_annotations(
    *,
    pending_run_manifest: Path,
    review_manifest: Path,
    cvat_export: Path,
    roundtrip_report: Path,
    ontology_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    pending_run_manifest = pending_run_manifest.resolve()
    review_manifest = review_manifest.resolve()
    cvat_export = cvat_export.resolve()
    roundtrip_report = roundtrip_report.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing promotion manifest: {output_path}")
    for path in (pending_run_manifest, review_manifest, cvat_export, roundtrip_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    ontology = load_ontology(ontology_path)
    pending = load_json_object(pending_run_manifest, label="pending run manifest")
    review = load_json_object(review_manifest, label="CVAT review manifest")
    roundtrip = load_json_object(roundtrip_report, label="CVAT round-trip report")
    pending_images, source_image_list_sha = _pending_images(
        pending, ontology_sha256=ontology.sha256
    )

    if review.get("schema_version") != REVIEW_SCHEMA:
        raise PromotionError(f"Review manifest schema must be {REVIEW_SCHEMA}")
    expected_pending_sha = require_sha256(
        review.get("pending_run_manifest_sha256"),
        field="review pending_run_manifest_sha256",
    )
    actual_pending_sha = sha256_file(pending_run_manifest)
    if expected_pending_sha != actual_pending_sha:
        raise PromotionError("Review manifest is bound to a different pending run")
    if require_sha256(
        review.get("source_image_list_sha256"),
        field="review source_image_list_sha256",
    ) != source_image_list_sha:
        raise PromotionError("Review manifest source image list differs from pending run")
    if require_sha256(
        review.get("ontology_sha256"), field="review ontology_sha256"
    ) != ontology.sha256:
        raise PromotionError("Review manifest ontology differs from selected ontology")
    export_sha = sha256_file(cvat_export)
    if require_sha256(
        review.get("cvat_export_sha256"), field="review cvat_export_sha256"
    ) != export_sha:
        raise PromotionError("CVAT export hash differs from review manifest")
    roundtrip_sha = sha256_file(roundtrip_report)
    if require_sha256(
        review.get("roundtrip_report_sha256"), field="review roundtrip_report_sha256"
    ) != roundtrip_sha:
        raise PromotionError("Round-trip report hash differs from review manifest")

    reviewer = review.get("reviewer")
    if not isinstance(reviewer, dict):
        raise PromotionError("Review manifest reviewer must be an object")
    reviewer_id = _required_nonempty(reviewer, "id")
    reviewer_name = _required_nonempty(reviewer, "name")
    approved_at = _approved_timestamp(review.get("approved_at_utc"))
    cvat = review.get("cvat")
    if not isinstance(cvat, dict) or cvat.get("task_id") in (None, ""):
        raise PromotionError("Review manifest must contain cvat.task_id")
    job_ids = cvat.get("job_ids")
    if not isinstance(job_ids, list) or not job_ids or any(job in (None, "") for job in job_ids):
        raise PromotionError("Review manifest must contain at least one cvat.job_ids entry")
    if len({str(job) for job in job_ids}) != len(job_ids):
        raise PromotionError("Review manifest cvat.job_ids must be unique")

    raw_reviews = review.get("images")
    if not isinstance(raw_reviews, list) or not raw_reviews:
        raise PromotionError("Review manifest images must be a non-empty list")
    reviewed: dict[str, str] = {}
    for index, row in enumerate(raw_reviews, start=1):
        if not isinstance(row, dict):
            raise PromotionError(f"Review image {index} must be an object")
        path = safe_relative_path(row.get("path"), field=f"review images[{index}].path")
        disposition = row.get("disposition")
        if disposition not in APPROVED_DISPOSITIONS:
            raise PromotionError(
                f"Review image {path} has unresolved disposition {disposition!r}"
            )
        if path in reviewed:
            raise PromotionError(f"Duplicate reviewed image: {path}")
        reviewed[path] = str(disposition)
    if set(reviewed) != set(pending_images):
        missing = sorted(set(pending_images) - set(reviewed))[:10]
        unexpected = sorted(set(reviewed) - set(pending_images))[:10]
        raise PromotionError(
            f"Review coverage is incomplete or contains unknown images: "
            f"missing={missing}, unexpected={unexpected}"
        )

    if roundtrip.get("schema_version") != ROUNDTRIP_SCHEMA or roundtrip.get("status") != "PASS":
        raise PromotionError("CVAT round-trip report must be PASS")
    if roundtrip.get("format") != "coco":
        raise PromotionError("Promotion requires a PASS COCO round-trip report; YOLO loses attributes")
    roundtrip_ontology = roundtrip.get("ontology")
    if not isinstance(roundtrip_ontology, dict) or roundtrip_ontology.get("sha256") != ontology.sha256:
        raise PromotionError("Round-trip report ontology differs from selected ontology")
    artifact = roundtrip.get("roundtrip_artifact")
    if not isinstance(artifact, dict) or artifact.get("sha256") != export_sha:
        raise PromotionError("Round-trip report is bound to a different CVAT export")
    roundtrip_counts = roundtrip.get("counts")
    if not isinstance(roundtrip_counts, dict):
        raise PromotionError("Round-trip report counts must be an object")
    try:
        roundtrip_image_count = int(roundtrip_counts.get("images", -1))
    except (TypeError, ValueError) as exc:
        raise PromotionError("Round-trip image count must be an integer") from exc
    if roundtrip_image_count != len(pending_images):
        raise PromotionError("Round-trip image count differs from pending source image count")

    dispositions = Counter(reviewed.values())
    promotion = {
        "schema_version": PROMOTION_SCHEMA,
        "status": "PASS",
        "annotation_state": "reviewed_train",
        "promoted_at_utc": utc_now(),
        "source_role": "unlabeled_train",
        "source_image_list_sha256": source_image_list_sha,
        "ontology": ontology.record(),
        "pending_run": {
            "sha256": actual_pending_sha,
            "run_id": pending.get("run_id"),
        },
        "review": {
            "manifest_sha256": sha256_file(review_manifest),
            "reviewer": {"id": reviewer_id, "name": reviewer_name},
            "approved_at_utc": approved_at,
            "cvat_task_id": cvat["task_id"],
            "cvat_job_ids": job_ids,
            "image_count": len(reviewed),
            "disposition_counts": dict(sorted(dispositions.items())),
        },
        "approved_export": {
            "sha256": export_sha,
            "roundtrip_report_sha256": roundtrip_sha,
            "format": "coco",
        },
        "training_use": {
            "allowed": True,
            "split": "train",
            "validation_or_test_use": False,
            "included_image_count": len(reviewed) - dispositions.get("rejected", 0),
            "excluded_rejected_image_count": dispositions.get("rejected", 0),
        },
    }
    write_json(output_path, promotion)
    return promotion


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Promote a complete human-reviewed CVAT COCO export to reviewed_train by hashes; "
            "never copies pending labels directly"
        )
    )
    parser.add_argument("--pending-run-manifest", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--cvat-export", type=Path, required=True)
    parser.add_argument("--roundtrip-report", type=Path, required=True)
    parser.add_argument(
        "--ontology", type=Path, default=root / "configs" / "classes.smd_v1.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = promote_reviewed_annotations(
            pending_run_manifest=args.pending_run_manifest,
            review_manifest=args.review_manifest,
            cvat_export=args.cvat_export,
            roundtrip_report=args.roundtrip_report,
            ontology_path=args.ontology,
            output_path=args.output,
        )
    except (ContractError, PromotionError, FileNotFoundError, FileExistsError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
