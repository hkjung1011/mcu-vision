from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .common import sha256_file, utc_now, write_json
from .checkpoint_publishing import sanitize_yolo11_checkpoint
from .publishing import publish_evidence_file


SNAPSHOT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
COMPLETE_EVIDENCE = (
    "final_metrics.json",
    "per_class_metrics.csv",
    "confidence_curve.csv",
    "confidence_curve.png",
    "confusion_counts.csv",
    "confusion_counts.png",
    "confusion_normalized.csv",
    "confusion_normalized.png",
    "latency.json",
    "latency_samples.csv",
    "plots/summary/comparison_terminal.txt",
    "plots/summary/terminal_summary.png",
    "plots/summary/training_curves.png",
    "plots/summary/comparison_dashboard.png",
    "plots/summary/evidence_manifest.json",
    "plots/summary/protocol_compatibility.json",
)
COMMON_EVIDENCE = (
    "run_manifest.json",
    "epoch_metrics.csv",
    "gpu_summary.json",
    "pip-freeze.txt",
    "terminal.log",
    "plots/latest_overview.png",
)
INTERRUPTED_EVIDENCE = (
    "epoch_metrics.jsonl",
    "native_coco_metrics.json",
)


def _portable_source(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    try:
        if value in (None, "", "null"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration_minutes(manifest: dict[str, Any]) -> float | None:
    try:
        start = datetime.fromisoformat(str(manifest["start_utc"]))
        end = datetime.fromisoformat(str(manifest["end_utc"]))
    except (KeyError, TypeError, ValueError):
        return None
    return (end - start).total_seconds() / 60.0


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to inspect checkpoints") from exc
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in ("start_epoch", "best_ap", "curr_ap", "epoch")
        if isinstance(value.get(key), (int, float))
    }


def _run_summary(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    epoch_rows = _read_rows(run_dir / "epoch_metrics.csv")
    if not epoch_rows:
        raise ValueError(f"No epoch metrics: {run_dir}")
    last = epoch_rows[-1]
    protocol = manifest.get("protocol", {})
    status = str(manifest.get("status"))
    completed_epochs = int(last["epoch"])
    planned_epochs = int(protocol.get("epochs", completed_epochs))
    final_metrics_path = run_dir / "final_metrics.json"
    metrics: dict[str, Any] = {}
    metric_source = "native_epoch_metrics_interrupted"
    if final_metrics_path.exists() and status == "complete":
        metrics = json.loads(final_metrics_path.read_text(encoding="utf-8")).get("metrics", {})
        metric_source = "common_coco_validation"
    native_rows = [row for row in epoch_rows if _float(row.get("map50_95")) is not None]
    best_native = max(native_rows, key=lambda row: float(row["map50_95"])) if native_rows else {}
    latency_path = run_dir / "latency.json"
    latency = (
        json.loads(latency_path.read_text(encoding="utf-8")) if latency_path.exists() else {}
    )
    gpu_path = run_dir / "gpu_summary.json"
    gpu = json.loads(gpu_path.read_text(encoding="utf-8")) if gpu_path.exists() else {}
    model_details = manifest.get("model_details", {})
    gflops = _float(model_details.get("gflops_ultralytics"))
    if gflops is None:
        match = re.search(r"Gflops:\s*([0-9.]+)", str(model_details.get("summary", "")))
        gflops = float(match.group(1)) if match else None
    summary = {
        "run_id": str(manifest["run_id"]),
        "model": str(manifest.get("model")),
        "framework": str(manifest.get("framework")),
        "seed": int(protocol.get("seed")),
        "status": status,
        "release_eligible": False,
        "completed_epochs": completed_epochs,
        "planned_epochs": planned_epochs,
        "metric_source": metric_source,
        "common_ap50_95": _float(metrics.get("ap50_95")),
        "common_ap50": _float(metrics.get("ap50")),
        "common_ap75": _float(metrics.get("ap75")),
        "precision_at_025": _float(metrics.get("precision")),
        "recall_at_025": _float(metrics.get("recall")),
        "f1_at_025": _float(metrics.get("f1")),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "native_best_epoch": int(best_native["epoch"]) if best_native else None,
        "native_best_ap50_95": _float(best_native.get("map50_95")),
        "native_last_ap50_95": _float(last.get("map50_95")),
        "native_last_ap50": _float(last.get("map50")),
        "native_last_ap75": _float(last.get("ap75")),
        "native_last_ar100": _float(last.get("ar100")),
        "latency_e2e_p50_ms": _float(latency.get("e2e_p50_ms")),
        "latency_e2e_p95_ms": _float(latency.get("e2e_p95_ms")),
        "sustained_fps": _float(latency.get("sustained_fps")),
        "parameters": model_details.get("parameters"),
        "gflops": gflops,
        "model_summary": model_details.get("summary"),
        "train_peak_allocated_mib": _float(
            gpu.get("train_peak_allocated_mib", last.get("gpu_peak_allocated_mib"))
        ),
        "wall_minutes": _duration_minutes(manifest),
    }
    return summary


def _copy_evidence(
    run_dir: Path,
    report_run_dir: Path,
    relative_paths: Iterable[str],
    *,
    campaign_dir: Path,
    project_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in relative_paths:
        source = run_dir / relative
        if not source.exists():
            raise FileNotFoundError(f"Required progress evidence is missing: {source}")
        destination = report_run_dir / relative
        record = publish_evidence_file(source, destination, project_root=project_root)
        records.append(
            {
                "path": destination.relative_to(project_root).as_posix(),
                "source_relative_to_campaign": _portable_source(source, campaign_dir),
                **record,
            }
        )
    return records


def publish_progress_snapshot(
    campaign_dir: Path,
    snapshot_name: str,
    *,
    project_root: Path,
) -> dict[str, Any]:
    if not SNAPSHOT_NAME_PATTERN.fullmatch(snapshot_name):
        raise ValueError("snapshot_name must be one safe path component")
    project_root = project_root.resolve()
    campaign_dir = campaign_dir.resolve()
    report_root = project_root / "reports" / "progress" / snapshot_name
    weight_root = project_root / "weights" / "progress" / snapshot_name
    for output_root in (report_root, weight_root):
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError("Progress snapshot is immutable; choose a new snapshot name")

    run_dirs = sorted(
        path
        for path in campaign_dir.iterdir()
        if path.is_dir() and (path / "run_manifest.json").exists()
    )
    if not run_dirs:
        raise ValueError(f"No publishable runs found: {campaign_dir}")

    report_root.mkdir(parents=True, exist_ok=True)
    weight_root.mkdir(parents=True, exist_ok=True)
    file_records: list[dict[str, Any]] = []
    weight_records: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    campaign_plan = campaign_dir / "campaign_plan.json"
    if campaign_plan.exists():
        record = publish_evidence_file(
            campaign_plan,
            report_root / "campaign_plan.json",
            project_root=project_root,
        )
        file_records.append(
            {
                "path": (report_root / "campaign_plan.json").relative_to(project_root).as_posix(),
                "source_relative_to_campaign": "campaign_plan.json",
                **record,
            }
        )

    for run_dir in run_dirs:
        summary = _run_summary(run_dir)
        if summary["status"] not in {"complete", "interrupted"}:
            raise ValueError(f"Unsupported progress status: {summary['run_id']}={summary['status']}")
        run_summaries.append(summary)
        report_run_dir = report_root / summary["run_id"]
        evidence = list(COMMON_EVIDENCE)
        evidence.extend(COMPLETE_EVIDENCE if summary["status"] == "complete" else INTERRUPTED_EVIDENCE)
        file_records.extend(
            _copy_evidence(
                run_dir,
                report_run_dir,
                evidence,
                campaign_dir=campaign_dir,
                project_root=project_root,
            )
        )

        if summary["framework"].lower().startswith("ultralytics"):
            source_weights = [(run_dir / "native" / "weights" / "best.pt", "best")]
        elif summary["status"] == "complete":
            source_weights = [(run_dir / "best_ckpt.pth", "best")]
        else:
            source_weights = [
                (run_dir / "best_ckpt.pth", "best"),
                (run_dir / "latest_ckpt.pth", "resume_epoch_70"),
            ]

        for source, role in source_weights:
            if not source.exists():
                raise FileNotFoundError(f"Required checkpoint is missing: {source}")
            suffix = source.suffix
            destination = weight_root / f"{summary['run_id']}_{role}{suffix}"
            if suffix == ".pt":
                record = sanitize_yolo11_checkpoint(
                    source,
                    destination,
                    project_root=project_root,
                )
            else:
                record = publish_evidence_file(source, destination, project_root=project_root)
                record["metadata_sanitized"] = False
                record["torch_load_metadata"] = _checkpoint_metadata(destination)
            weight_records.append(
                {
                    "path": destination.relative_to(project_root).as_posix(),
                    "source_relative_to_campaign": _portable_source(source, campaign_dir),
                    "run_id": summary["run_id"],
                    "role": role,
                    "status": summary["status"],
                    "release_eligible": False,
                    **record,
                }
            )

    results_path = report_root / "results.csv"
    fieldnames = list(run_summaries[0])
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_summaries)
    file_records.append(
        {
            "path": results_path.relative_to(project_root).as_posix(),
            "source_relative_to_campaign": "DERIVED_FROM_PUBLISHED_RUN_EVIDENCE",
            "source_original_sha256": None,
            "published_sha256": sha256_file(results_path),
            "bytes": results_path.stat().st_size,
            "sanitized_for_repository": False,
        }
    )

    protocol = json.loads((run_dirs[0] / "run_manifest.json").read_text(encoding="utf-8"))
    dataset = protocol.get("dataset", {})
    protocol_config = protocol.get("protocol_config", {})
    manifest = {
        "schema_version": 1,
        "snapshot_name": snapshot_name,
        "created_utc": utc_now(),
        "publication_stage": "INTERIM_PROGRESS",
        "formal_release": False,
        "release_ready": False,
        "independent_camera_tested": False,
        "task": "one_class_raspberry_pi_sbc_detection",
        "scope_warning": "Not an STM32, small-SMD, OCR, or multi-class model.",
        "campaign_id": campaign_dir.name,
        "source_git_commit": protocol.get("git", {}).get("commit"),
        "protocol_sha256": protocol_config.get("sha256"),
        "dataset_evidence_sha256": dataset.get("equivalence_evidence_sha256"),
        "dataset": {
            "class_names": dataset.get("classes") or ["raspberry_pi_sbc"],
            "train_images": 1500,
            "validation_images": 195,
            "test_images": 180,
            "physical_specimen_independence": "NOT_VERIFIED",
        },
        "campaign_progress": {
            "complete_runs": sum(row["status"] == "complete" for row in run_summaries),
            "planned_runs": 6,
            "completed_formal_epochs": sum(
                row["completed_epochs"] for row in run_summaries if row["status"] == "complete"
            ),
            "recorded_epochs_including_interrupted": sum(
                row["completed_epochs"] for row in run_summaries
            ),
            "planned_epochs": 600,
        },
        "evidence_policy": {
            "raw_or_processed_dataset_images_included": False,
            "prediction_boxes_included": False,
            "generative_ai_used_for_images": False,
            "graphs": "Deterministic matplotlib renderings from the published CSV/JSON/log evidence.",
            "terminal_summary_png": "Rendered text evidence, not a literal monitor screenshot.",
            "terminal_logs": "Local paths and NVIDIA process tables are redacted; numeric training content is unchanged.",
        },
        "runs": run_summaries,
        "files": sorted(file_records, key=lambda item: item["path"]),
        "checkpoints": sorted(weight_records, key=lambda item: item["path"]),
    }
    manifest_path = report_root / "progress_manifest.json"
    write_json(manifest_path, manifest)
    digest_path = report_root / "progress_manifest.sha256"
    digest_path.write_text(
        f"{sha256_file(manifest_path)}  progress_manifest.json\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def verify_progress_snapshot(report_root: Path, *, project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    report_root = report_root.resolve()
    manifest_path = report_root / "progress_manifest.json"
    digest_path = report_root / "progress_manifest.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_digest = digest_path.read_text(encoding="utf-8").split()[0].lower()
    if sha256_file(manifest_path) != expected_digest:
        raise ValueError("progress_manifest.sha256 does not match progress_manifest.json")
    if manifest.get("formal_release") or manifest.get("release_ready"):
        raise ValueError("A progress snapshot must not claim formal release readiness")

    records = [*manifest.get("files", []), *manifest.get("checkpoints", [])]
    for record in records:
        path = (project_root / str(record["path"])).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise ValueError(f"Snapshot path escapes the project root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot file is missing: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Snapshot byte size changed: {path}")
        if sha256_file(path) != str(record["published_sha256"]).lower():
            raise ValueError(f"Snapshot SHA-256 changed: {path}")

    forbidden_dataset_images = [
        path
        for path in report_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    ]
    if forbidden_dataset_images:
        raise ValueError(f"Dataset-like raster files are not allowed: {forbidden_dataset_images[0]}")
    return {
        "status": "PASS",
        "snapshot_name": manifest.get("snapshot_name"),
        "evidence_files": len(manifest.get("files", [])),
        "checkpoints": len(manifest.get("checkpoints", [])),
        "manifest_sha256": expected_digest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a privacy-scrubbed, non-release training progress snapshot."
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--snapshot-name", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = publish_progress_snapshot(
        args.campaign_dir,
        args.snapshot_name,
        project_root=args.project_root,
    )
    print(json.dumps(manifest["campaign_progress"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
