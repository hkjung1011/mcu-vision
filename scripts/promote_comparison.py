from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcu_data.common import sha256_file, write_json
from mcu_data.publishing import (
    copy_public_file_exact,
    load_json_strict,
    scan_public_file,
    validate_formal_comparison,
    validate_published_comparison_release,
    validated_formal_publication_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a verified multi-run comparison into a Git-tracked report folder"
    )
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--release-name", required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.release_name):
        parser.error(
            "--release-name must be a plain 1-128 character identifier using only letters, "
            "numbers, dot, underscore, or hyphen"
        )

    source_root = args.comparison_dir.resolve()
    try:
        publication_plan = validated_formal_publication_plan(
            source_root,
            require_local_originals=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    compatibility_path = source_root / "protocol_compatibility.json"
    if not compatibility_path.exists():
        parser.error(f"Missing protocol_compatibility.json: {source_root}")
    compatibility = load_json_strict(compatibility_path)
    if not compatibility.get("release_ready", False):
        fields = ", ".join(
            str(item.get("field")) for item in compatibility.get("release_blockers", [])
        )
        parser.error(
            "Comparison is not formal-release ready"
            + (f" ({fields})" if fields else "")
            + ". Complete the configured model/seed/data gates."
        )

    destination_root = PROJECT_ROOT / "reports" / "comparisons" / args.release_name
    if destination_root.exists():
        raise FileExistsError(f"Release already exists: {destination_root}")
    destination_root.mkdir(parents=True)

    copied: list[dict[str, object]] = []
    for relative_text in publication_plan["relative_paths"]:
        relative = Path(relative_text)
        source = source_root / relative
        destination = destination_root / relative
        record = copy_public_file_exact(source, destination)
        copied.append(
            {
                "path": relative.as_posix(),
                **record,
            }
        )
    copied_paths = {
        path.relative_to(destination_root).as_posix()
        for path in destination_root.rglob("*")
        if path.is_file()
    }
    if copied_paths != set(publication_plan["relative_paths"]):
        raise RuntimeError("Published comparison file set differs from the verified allowlist")
    validate_formal_comparison(destination_root)
    public_file_records = [
        scan_public_file(destination_root / relative, relative_path=relative)
        for relative in sorted(publication_plan["relative_paths"])
    ]

    artifact = {
        "schema_version": 3,
        "status": "PASS",
        "formal_release": True,
        "release_name": args.release_name,
        "source_comparison_id": source_root.name,
        "source_protocol_compatibility_sha256": sha256_file(compatibility_path),
        "source_formal_validation_sha256": sha256_file(
            source_root / "formal_validation.json"
        ),
        "source_evidence_manifest_sha256": sha256_file(
            source_root / "evidence_manifest.json"
        ),
        "local_source_path_included": False,
        "protocol_comparable": bool(compatibility.get("comparable", False)),
        "protocol_release_ready": bool(compatibility.get("release_ready", False)),
        "files": copied,
        "source_scan": publication_plan["scan"],
        "public_scan": {
            "status": "PASS",
            "files": public_file_records,
        },
        "publication_note": (
            "Repository copies redact local user/project paths and raw nvidia-smi process listings. "
            "Original and published SHA-256 values are recorded per file; numeric metrics are unchanged."
        ),
        "raw_images_included": bool(publication_plan["scan"]["raw_image_files"]),
        "weights_included": bool(publication_plan["scan"]["weight_files"]),
    }
    write_json(destination_root / "artifact_manifest.json", artifact)
    validate_published_comparison_release(destination_root)
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    print("\nReady for review, then git add/commit/push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
