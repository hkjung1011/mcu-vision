from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcu_data.common import sha256_file, write_json
from mcu_data.checkpoint_publishing import (
    publish_yolox_checkpoint,
    sanitize_yolo11_checkpoint,
)
from mcu_data.publishing import (
    IMAGE_SUFFIXES,
    WEIGHT_SUFFIXES,
    copy_public_file_exact,
    load_json_strict,
    scan_public_file,
    validate_comparison_for_run,
    validate_formal_comparison,
    validate_published_run_release,
    validated_formal_publication_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy a verified run into Git/LFS tracked release folders")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        required=True,
        help="Comparable multi-run report containing this exact run manifest",
    )
    parser.add_argument("--release-name")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_json_strict(manifest_path)
    if manifest.get("status") != "complete":
        parser.error(f"Only complete runs can be promoted: status={manifest.get('status')}")
    if manifest.get("stage") == "smoke_not_comparable":
        parser.error("Smoke runs cannot be promoted to weights/trained.")
    comparison_root = args.comparison_dir.resolve()
    try:
        comparison_publication = validated_formal_publication_plan(
            comparison_root,
            require_local_originals=True,
        )
        comparison_evidence = validate_comparison_for_run(
            comparison_root,
            run_id=str(manifest["run_id"]),
            run_manifest_path=manifest_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    release_name = str(args.release_name or manifest["run_id"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", release_name):
        parser.error(
            "--release-name must be a plain 1-128 character identifier using only letters, "
            "numbers, dot, underscore, or hyphen"
        )
    weight_root = PROJECT_ROOT / "weights" / "trained" / release_name
    report_root = PROJECT_ROOT / "reports" / "runs" / release_name
    if weight_root.exists() or report_root.exists():
        raise FileExistsError(f"Release already exists: {release_name}")
    checkpoint_value = manifest.get("best_checkpoint", {})
    checkpoint_text = checkpoint_value.get("path")
    if not checkpoint_text:
        parser.error("run_manifest.json does not contain best_checkpoint.path")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    checkpoint = checkpoint.resolve()
    try:
        checkpoint.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("Best checkpoint must be contained in --run-dir") from exc
    if checkpoint.suffix.lower() not in WEIGHT_SUFFIXES:
        raise ValueError(f"Best checkpoint has an unsupported weight suffix: {checkpoint.suffix}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
    expected_sha256 = checkpoint_value.get("sha256")
    actual_sha256 = sha256_file(checkpoint)
    if expected_sha256 and expected_sha256 != actual_sha256:
        raise ValueError("Checkpoint SHA-256 differs from run_manifest.json")

    weight_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    promoted_checkpoint = weight_root / checkpoint.name
    framework = str(manifest.get("framework", "")).lower()
    is_yolo11 = "ultralytics" in framework or str(manifest.get("model", "")).lower().startswith("yolo11")
    if is_yolo11:
        checkpoint_publication = sanitize_yolo11_checkpoint(
            checkpoint,
            promoted_checkpoint,
            project_root=PROJECT_ROOT,
        )
    else:
        checkpoint_publication = publish_yolox_checkpoint(
            checkpoint,
            promoted_checkpoint,
            project_root=PROJECT_ROOT,
        )
    promoted_weight_files = sorted(
        path.relative_to(weight_root).as_posix()
        for path in weight_root.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    )
    all_weight_payload_files = sorted(
        path.relative_to(weight_root).as_posix()
        for path in weight_root.rglob("*")
        if path.is_file()
    )
    if (
        promoted_weight_files != [promoted_checkpoint.name]
        or all_weight_payload_files != promoted_weight_files
    ):
        raise ValueError(
            "Promoted weight payload must contain exactly the declared checkpoint: "
            f"files={all_weight_payload_files}, weights={promoted_weight_files}"
        )
    copied_reports = []
    publication_records = []
    for source_record in comparison_evidence["verified_source_files"]:
        source = comparison_root / str(source_record["path"])
        relative_path = (Path("run_evidence") / str(source_record["filename"])).as_posix()
        destination = report_root / relative_path
        record = copy_public_file_exact(source, destination)
        publication_records.append(
            {
                "path": relative_path,
                "comparison_bundle_path": source_record["path"],
                "comparison_bundle_sha256": source_record["published_sha256"],
                **record,
            }
        )
        copied_reports.append(relative_path)

    for relative_text in comparison_publication["relative_paths"]:
        source = comparison_root / relative_text
        relative_path = (Path("formal_comparison") / relative_text).as_posix()
        destination = report_root / relative_path
        record = copy_public_file_exact(source, destination)
        publication_records.append({"path": relative_path, **record})
        copied_reports.append(relative_path)

    validate_formal_comparison(report_root / "formal_comparison")

    report_files = {
        path.relative_to(report_root).as_posix()
        for path in report_root.rglob("*")
        if path.is_file()
    }
    report_weight_files = sorted(
        path for path in report_files if Path(path).suffix.lower() in WEIGHT_SUFFIXES
    )
    formal_derived_images = {
        (Path("formal_comparison") / path).as_posix()
        for path in comparison_publication["scan"]["derived_image_files"]
        if "/" not in path
    }
    report_image_files = {
        path for path in report_files if Path(path).suffix.lower() in IMAGE_SUFFIXES
    }
    raw_image_files = sorted(report_image_files - formal_derived_images)
    prediction_files = sorted(
        path for path in report_files if "prediction" in Path(path).name.lower()
    )
    if report_weight_files:
        raise ValueError(f"Run report payload unexpectedly contains weights: {report_weight_files}")
    if raw_image_files:
        raise ValueError(f"Run report payload contains undeclared raw images: {raw_image_files}")
    checkpoint_record = {
        "path": promoted_checkpoint.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": checkpoint_publication["bytes"],
        "sha256": checkpoint_publication["published_sha256"],
        "source_original_sha256": checkpoint_publication["source_original_sha256"],
    }
    if is_yolo11:
        checkpoint_record.update(
            {
                "metadata_sanitized": checkpoint_publication["metadata_sanitized"],
                "state_dict_bitwise_equal": checkpoint_publication["state_dict_bitwise_equal"],
                "forward_max_abs_difference": checkpoint_publication[
                    "forward_max_abs_difference"
                ],
                "source_forward_captured_before_scrub": checkpoint_publication.get(
                    "source_forward_captured_before_scrub", False
                ),
                "ultralytics_load": checkpoint_publication["ultralytics_load"],
            }
        )
    else:
        checkpoint_record["proof"] = checkpoint_publication["proof"]
    report_scans = [
        scan_public_file(report_root / relative, relative_path=relative)
        for relative in sorted(report_files)
    ]
    weight_scan = scan_public_file(
        promoted_checkpoint,
        relative_path=(Path("weights") / promoted_checkpoint.name).as_posix(),
    )
    artifact = {
        "schema_version": 3,
        "status": "PASS",
        "formal_release": True,
        "release_name": release_name,
        "source_run_id": manifest.get("run_id"),
        "source_run_manifest_sha256": sha256_file(manifest_path),
        "validated_by_comparison": comparison_evidence,
        "local_source_path_included": False,
        "model": manifest.get("model"),
        "stage": manifest.get("stage"),
        "checkpoint": checkpoint_record,
        "reports": sorted(set(copied_reports)),
        "published_evidence": sorted(publication_records, key=lambda item: str(item["path"])),
        "source_scan": {
            "comparison": comparison_publication["scan"],
            "published_report_files_scanned": len(report_files),
            "raw_image_files": raw_image_files,
            "report_weight_files": report_weight_files,
            "prediction_files": prediction_files,
            "promoted_checkpoint_files": [
                promoted_checkpoint.relative_to(PROJECT_ROOT).as_posix()
            ],
            "weight_payload_files_scanned": all_weight_payload_files,
        },
        "public_scan": {
            "status": "PASS",
            "report_files": report_scans,
            "weight_files": [weight_scan],
        },
        "publication_note": (
            "Repository copies redact local user/project paths and raw nvidia-smi process listings. "
            "Original and published SHA-256 values are recorded per file; numeric metrics are unchanged. "
            "YOLO11 checkpoints require bitwise-equal tensors, zero forward difference, and a "
            "successful Ultralytics reload after metadata sanitization. YOLOX checkpoints are exact "
            ".pth copies restricted-loaded on CPU with a known model state_dict; no forward test is claimed."
        ),
        "raw_images_included": bool(raw_image_files),
        "predictions_included": bool(prediction_files),
        "weights_in_reports": bool(report_weight_files),
        "promoted_checkpoint_count": len(promoted_weight_files),
    }
    write_json(report_root / "artifact_manifest.json", artifact)
    validate_published_run_release(
        report_root,
        weight_root,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    print("\nReady for review, then git add/commit/push. Checkpoint is covered by Git LFS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
