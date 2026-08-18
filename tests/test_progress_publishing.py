import json
from pathlib import Path

import pytest

from mcu_data.common import sha256_file, write_json
from mcu_data.progress_publishing import verify_progress_snapshot


def test_verify_progress_snapshot_checks_hashes_and_release_boundary(tmp_path: Path) -> None:
    report = tmp_path / "reports" / "progress" / "snapshot"
    evidence = report / "run" / "metrics.csv"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("epoch,ap\n1,0.5\n", encoding="utf-8")
    manifest = {
        "snapshot_name": "snapshot",
        "formal_release": False,
        "release_ready": False,
        "files": [
            {
                "path": evidence.relative_to(tmp_path).as_posix(),
                "bytes": evidence.stat().st_size,
                "published_sha256": sha256_file(evidence),
            }
        ],
        "checkpoints": [],
    }
    manifest_path = report / "progress_manifest.json"
    write_json(manifest_path, manifest)
    (report / "progress_manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  progress_manifest.json\n", encoding="utf-8"
    )

    result = verify_progress_snapshot(report, project_root=tmp_path)
    assert result["status"] == "PASS"

    evidence.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="byte size changed|SHA-256 changed"):
        verify_progress_snapshot(report, project_root=tmp_path)


def test_verify_progress_snapshot_rejects_formal_release_claim(tmp_path: Path) -> None:
    report = tmp_path / "reports" / "progress" / "snapshot"
    report.mkdir(parents=True)
    manifest_path = report / "progress_manifest.json"
    write_json(
        manifest_path,
        {
            "snapshot_name": "snapshot",
            "formal_release": True,
            "release_ready": False,
            "files": [],
            "checkpoints": [],
        },
    )
    (report / "progress_manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  progress_manifest.json\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="must not claim formal release"):
        verify_progress_snapshot(report, project_root=tmp_path)
