from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcu_data.common import sha256_file, write_json
from mcu_data.publishing import publish_evidence_file, validate_comparison_for_run


REPORT_FILES = [
    "run_manifest.json",
    "terminal.log",
    "pip-freeze.txt",
    "epoch_metrics.csv",
    "epoch_metrics.jsonl",
    "epoch_metrics_extra.jsonl",
    "final_metrics.json",
    "per_class_metrics.csv",
    "confidence_curve.csv",
    "autolabel_thresholds.csv",
    "confusion_counts.csv",
    "confusion_normalized.csv",
    "latency.json",
    "latency_samples.csv",
    "gpu_summary.json",
    "pretrained_weights_summary.csv",
    "best_weights_summary.csv",
    "protocol_snapshot.yaml",
    "protocol_rationale.csv",
    "protocol_references.json",
    "protocol_artifacts.json",
    "experiment_methodology.md",
    "parameter_rationale.md",
]

SUMMARY_FILES = [
    "aggregate_comparison.csv",
    "aggregate_comparison.json",
    "comparison.csv",
    "comparison.json",
    "comparison_terminal.txt",
    "evidence_manifest.json",
    "experiment_methodology.md",
    "parameter_rationale.md",
    "experiment_report.md",
    "protocol_artifacts.json",
    "protocol_compatibility.json",
    "protocol_rationale.csv",
    "protocol_references.json",
    "protocol_snapshot.yaml",
]


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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        parser.error(f"Only complete runs can be promoted: status={manifest.get('status')}")
    if manifest.get("stage") == "smoke_not_comparable":
        parser.error("Smoke runs cannot be promoted to weights/trained.")
    try:
        comparison_evidence = validate_comparison_for_run(
            args.comparison_dir,
            run_id=str(manifest["run_id"]),
            run_manifest_path=manifest_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    release_name = args.release_name or manifest["run_id"]
    weight_root = PROJECT_ROOT / "weights" / "trained" / release_name
    report_root = PROJECT_ROOT / "reports" / "runs" / release_name
    if weight_root.exists() or report_root.exists():
        raise FileExistsError(f"Release already exists: {release_name}")
    checkpoint_value = manifest.get("best_checkpoint", {})
    checkpoint_text = checkpoint_value.get("path")
    if not checkpoint_text:
        parser.error("run_manifest.json does not contain best_checkpoint.path")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
    expected_sha256 = checkpoint_value.get("sha256")
    actual_sha256 = sha256_file(checkpoint)
    if expected_sha256 and expected_sha256 != actual_sha256:
        raise ValueError("Checkpoint SHA-256 differs from run_manifest.json")

    weight_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    promoted_checkpoint = weight_root / checkpoint.name
    shutil.copy2(checkpoint, promoted_checkpoint)
    copied_reports = []
    publication_records = []
    for relative in REPORT_FILES:
        source = run_dir / relative
        if source.exists():
            destination = report_root / source.name
            record = publish_evidence_file(source, destination, project_root=PROJECT_ROOT)
            publication_records.append({"path": source.name, **record})
            copied_reports.append(source.name)
    native_args = run_dir / "native" / "args.yaml"
    if native_args.exists():
        destination = report_root / "native_args.yaml"
        record = publish_evidence_file(native_args, destination, project_root=PROJECT_ROOT)
        publication_records.append({"path": "native_args.yaml", **record})
        copied_reports.append("native_args.yaml")
    for relative in SUMMARY_FILES:
        source = run_dir / "plots" / "summary" / relative
        if source.exists():
            destination = report_root / "summary" / source.name
            relative_path = f"summary/{source.name}"
            record = publish_evidence_file(source, destination, project_root=PROJECT_ROOT)
            publication_records.append({"path": relative_path, **record})
            copied_reports.append(relative_path)
    visual_groups = (
        ("*.png", Path("plots")),
        ("plots/latest_overview.png", Path("plots")),
        ("plots/epochs/*.png", Path("plots/epochs")),
        ("plots/summary/*.png", Path("summary")),
    )
    for pattern, destination_dir in visual_groups:
        for source in run_dir.glob(pattern):
            destination = report_root / destination_dir / source.name
            relative_path = (destination_dir / source.name).as_posix()
            record = publish_evidence_file(source, destination, project_root=PROJECT_ROOT)
            publication_records.append({"path": relative_path, **record})
            copied_reports.append(relative_path)
    artifact = {
        "release_name": release_name,
        "source_run_id": manifest.get("run_id"),
        "source_run_manifest_sha256": sha256_file(manifest_path),
        "validated_by_comparison": comparison_evidence,
        "local_source_path_included": False,
        "model": manifest.get("model"),
        "stage": manifest.get("stage"),
        "checkpoint": {
            "path": promoted_checkpoint.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": promoted_checkpoint.stat().st_size,
            "sha256": sha256_file(promoted_checkpoint),
        },
        "reports": sorted(set(copied_reports)),
        "published_evidence": sorted(publication_records, key=lambda item: str(item["path"])),
        "publication_note": (
            "Repository copies redact local user/project paths and raw nvidia-smi process listings. "
            "Original and published SHA-256 values are recorded per file; numeric metrics are unchanged."
        ),
        "raw_images_included": False,
        "predictions_included": False,
    }
    write_json(report_root / "artifact_manifest.json", artifact)
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    print("\nReady for review, then git add/commit/push. Checkpoint is covered by Git LFS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
