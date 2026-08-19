from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import kagglehub

from .common import load_yaml, portable_path, utc_now, write_json


def download_curated(config_path: Path, dataset_name: str, output_root: Path, force: bool) -> dict[str, Any]:
    config = load_yaml(config_path)
    datasets = config.get("datasets", {})
    if dataset_name not in datasets:
        raise KeyError(f"Unknown dataset {dataset_name!r}; configured: {', '.join(sorted(datasets))}")
    dataset = dict(datasets[dataset_name])
    provider = dataset.get("provider")
    destination = output_root / dataset_name
    record: dict[str, Any] = {
        "dataset_name": dataset_name,
        "provider": provider,
        "source_url": dataset.get("source_url", ""),
        "source_id": dataset.get("source_id", dataset_name),
        "dataset_version": dataset.get("dataset_version"),
        "author": dataset.get("author", ""),
        "license": dataset.get("license", ""),
        "license_url": dataset.get("license_url", ""),
        "rights_statement": dataset.get("rights_statement", ""),
        "rights_url": dataset.get("rights_url", ""),
        "ingest_split_policy": dataset.get("ingest_split_policy"),
        "formal_evaluation_allowed": dataset.get("formal_evaluation_allowed"),
        "purpose": dataset.get("purpose", ""),
        "notes": dataset.get("notes", []),
        "requested_at": utc_now(),
        "destination": portable_path(destination),
    }

    if provider == "kaggle":
        destination.mkdir(parents=True, exist_ok=True)
        resolved = kagglehub.dataset_download(
            str(dataset["handle"]), output_dir=str(destination), force_download=force
        )
        record.update(
            {"status": "DOWNLOADED", "resolved_path": portable_path(Path(resolved)), "completed_at": utc_now()}
        )
    elif provider == "manual_roboflow":
        record.update(
            {
                "status": "MANUAL_AUTH_REQUIRED",
                "action": (
                    "Open source_url, choose the configured raw dataset version, and export COCO. "
                    "Do not count or publish it until archive/hash/rights ingestion passes."
                ),
            }
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    write_json(output_root.parent.parent / "manifests" / f"{dataset_name}.dataset-source.json", record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a reviewed third-party dataset and record its provenance.")
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.curated.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/curated"))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = download_curated(
        config_path=args.config.resolve(),
        dataset_name=args.dataset,
        output_root=args.output_root.resolve(),
        force=args.force,
    )
    print(result)


if __name__ == "__main__":
    main()
