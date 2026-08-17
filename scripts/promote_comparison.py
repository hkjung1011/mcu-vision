from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcu_data.common import sha256_file, write_json
from mcu_data.publishing import publish_evidence_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a verified multi-run comparison into a Git-tracked report folder"
    )
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--allow-not-comparable", action="store_true")
    args = parser.parse_args()

    source_root = args.comparison_dir.resolve()
    compatibility_path = source_root / "protocol_compatibility.json"
    if not compatibility_path.exists():
        parser.error(f"Missing protocol_compatibility.json: {source_root}")
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    if not compatibility.get("comparable", False) and not args.allow_not_comparable:
        fields = ", ".join(
            str(item.get("field")) for item in compatibility.get("critical_mismatches", [])
        )
        parser.error(
            "Comparison failed the protocol gate"
            + (f" ({fields})" if fields else "")
            + ". Use a matched full run or pass --allow-not-comparable explicitly."
        )

    destination_root = PROJECT_ROOT / "reports" / "comparisons" / args.release_name
    if destination_root.exists():
        raise FileExistsError(f"Release already exists: {destination_root}")
    destination_root.mkdir(parents=True)

    copied: list[dict[str, object]] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        record = publish_evidence_file(source, destination, project_root=PROJECT_ROOT)
        copied.append(
            {
                "path": relative.as_posix(),
                **record,
            }
        )

    artifact = {
        "release_name": args.release_name,
        "source_comparison_id": source_root.name,
        "source_protocol_compatibility_sha256": sha256_file(compatibility_path),
        "local_source_path_included": False,
        "protocol_comparable": bool(compatibility.get("comparable", False)),
        "files": copied,
        "publication_note": (
            "Repository copies redact local user/project paths and raw nvidia-smi process listings. "
            "Original and published SHA-256 values are recorded per file; numeric metrics are unchanged."
        ),
        "raw_images_included": False,
        "weights_included": False,
    }
    write_json(destination_root / "artifact_manifest.json", artifact)
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    print("\nReady for review, then git add/commit/push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
