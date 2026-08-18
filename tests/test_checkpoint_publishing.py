from __future__ import annotations

from pathlib import Path
import json

import pytest

from mcu_data.common import sha256_file
from mcu_data.checkpoint_publishing import (
    assert_binary_has_no_local_paths,
    verify_formal_checkpoint_bridge,
)
from mcu_data import checkpoint_publishing, progress_publishing


def test_progress_and_formal_publishers_share_exact_sanitizer() -> None:
    assert progress_publishing.sanitize_yolo11_checkpoint is checkpoint_publishing.sanitize_yolo11_checkpoint


def test_binary_privacy_gate_rejects_project_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    safe = tmp_path / "safe.onnx"
    safe.write_bytes(b"onnx-data")
    assert_binary_has_no_local_paths(safe, project)

    unsafe = tmp_path / "unsafe.pt"
    unsafe.write_bytes(b"prefix:" + str(project.resolve()).encode("utf-8") + b":suffix")
    with pytest.raises(ValueError, match="machine-local path"):
        assert_binary_has_no_local_paths(unsafe, project)


def test_formal_checkpoint_bridge_binds_original_and_published_hashes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    checkpoint = project / "weights" / "trained" / "release" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"published checkpoint")
    original_sha = "1" * 64
    run_manifest = project / "runs" / "run_manifest.json"
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text(
        json.dumps({"run_id": "run42", "best_checkpoint": {"sha256": original_sha}}),
        encoding="utf-8",
    )
    artifact = project / "reports" / "runs" / "release" / "artifact_manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "release_name": "release",
                "source_run_id": "run42",
                "source_run_manifest_sha256": sha256_file(run_manifest),
                "checkpoint": {
                    "path": checkpoint.relative_to(project).as_posix(),
                    "sha256": sha256_file(checkpoint),
                    "source_original_sha256": original_sha,
                    "metadata_sanitized": True,
                    "state_dict_bitwise_equal": True,
                    "forward_max_abs_difference": 0.0,
                    "ultralytics_load": "PASS",
                },
            }
        ),
        encoding="utf-8",
    )

    result = verify_formal_checkpoint_bridge(
        project_root=project,
        artifact_path=artifact,
        checkpoint=checkpoint,
        run_manifest=run_manifest,
        framework="yolo11",
    )
    assert result["source_original_sha256"] == original_sha
    assert result["published_sha256"] == sha256_file(checkpoint)

    document = json.loads(artifact.read_text(encoding="utf-8"))
    document["checkpoint"]["source_original_sha256"] = "2" * 64
    artifact.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="original checkpoint"):
        verify_formal_checkpoint_bridge(
            project_root=project,
            artifact_path=artifact,
            checkpoint=checkpoint,
            run_manifest=run_manifest,
            framework="yolo11",
        )
