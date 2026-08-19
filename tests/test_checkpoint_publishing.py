from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mcu_data.common import sha256_file
from mcu_data.checkpoint_publishing import (
    assert_binary_has_no_local_paths,
    publish_yolox_checkpoint,
    validate_yolox_checkpoint_proof,
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

    arbitrary_user = tmp_path / "arbitrary.pt"
    arbitrary_user.write_bytes(b"D:\\Users\\AnotherPerson\\private\\weights.pt")
    with pytest.raises(ValueError, match="machine-local path"):
        assert_binary_has_no_local_paths(arbitrary_user, project)


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
                    "source_forward_captured_before_scrub": True,
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


class _FakeTensor:
    device = SimpleNamespace(type="cpu")
    dtype = "float32"
    shape = (2,)

    def __init__(self, values: tuple[float, ...] = (1.0, 2.0)) -> None:
        self.values = values

    def numel(self) -> int:
        return len(self.values)

    def element_size(self) -> int:
        return 4


def _fake_torch(load_calls: list[tuple[Path, str, bool]], checkpoint: object) -> object:
    def load(path: Path, *, map_location: str, weights_only: bool) -> object:
        load_calls.append((Path(path), map_location, weights_only))
        return checkpoint

    return SimpleNamespace(
        load=load,
        is_tensor=lambda value: isinstance(value, _FakeTensor),
        equal=lambda left, right: left.values == right.values,
    )


def test_yolox_publisher_restricted_loads_cpu_state_dict_and_is_truthful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str, bool]] = []
    checkpoint = {
        "start_epoch": 100,
        "model": {"backbone.weight": _FakeTensor()},
        "optimizer": {},
    }
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls, checkpoint))
    source = tmp_path / "source.pth"
    source.write_bytes(b"PK\x03\x04safe-yolox-checkpoint")
    destination = tmp_path / "public" / "best.pth"

    record = publish_yolox_checkpoint(source, destination, project_root=tmp_path)

    assert destination.read_bytes() == source.read_bytes()
    assert [(location, restricted) for _, location, restricted in calls] == [
        ("cpu", True),
        ("cpu", True),
    ]
    proof = record["proof"]
    assert proof["cpu_load"] == "PASS"
    assert proof["checkpoint_structure"] == "YOLOX_MODEL_STATE_DICT_PASS"
    assert proof["state_dict"]["bitwise_equal_after_copy"] is True
    assert proof["forward"] == {
        "status": "NOT_PERFORMED",
        "max_abs_difference": None,
        "reason": "Architecture construction is not part of checkpoint publication.",
    }
    assert "forward_max_abs_difference" not in record
    assert validate_yolox_checkpoint_proof(
        destination,
        record,
        project_root=tmp_path,
    ) == proof


@pytest.mark.parametrize(
    ("filename", "checkpoint", "message"),
    [
        ("best.pt", {"model": {"x": _FakeTensor()}}, "must use the .pth suffix"),
        ("best.pth", {"weights": {"x": _FakeTensor()}}, "unknown top-level keys"),
        ("best.pth", {"model": {}}, "non-empty model state_dict"),
    ],
)
def test_yolox_publisher_rejects_unknown_suffix_or_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    checkpoint: object,
    message: str,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], checkpoint))
    source = tmp_path / filename
    source.write_bytes(b"PK\x03\x04fixture")
    with pytest.raises(ValueError, match=message):
        publish_yolox_checkpoint(
            source,
            tmp_path / "public" / filename,
            project_root=tmp_path,
        )


def test_yolox_publisher_runs_generic_binary_privacy_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = {
        "model": {"x": _FakeTensor()},
        "optimizer": {"state": {"resume_path": "/opt/private/run/checkpoint"}},
    }
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], checkpoint))
    source = tmp_path / "best.pth"
    source.write_bytes(b"PK\x03\x04safe-archive-bytes")
    with pytest.raises(ValueError, match="absolute local path"):
        publish_yolox_checkpoint(
            source,
            tmp_path / "public" / "best.pth",
            project_root=tmp_path,
        )
    assert not (tmp_path / "public" / "best.pth").exists()


def test_actual_yolo11_sanitizer_in_framework_environment(tmp_path: Path) -> None:
    python = os.environ.get("MCU_YOLO11_PYTHON")
    if not python:
        pytest.skip("Set MCU_YOLO11_PYTHON to run the actual Ultralytics checkpoint integration")
    project = Path(__file__).resolve().parents[1]
    source = project / "weights" / "pretrained" / "yolo11m.pt"
    if not source.is_file():
        pytest.skip("YOLO11 integration checkpoint is unavailable")
    destination = tmp_path / "published.pt"
    program = (
        "import json,sys; from pathlib import Path; "
        "from mcu_data.checkpoint_publishing import sanitize_yolo11_checkpoint; "
        "print(json.dumps(sanitize_yolo11_checkpoint(Path(sys.argv[1]),Path(sys.argv[2]),"
        "project_root=Path(sys.argv[3]))))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project / "src")
    completed = subprocess.run(
        [python, "-c", program, str(source), str(destination), str(project)],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
        env=environment,
    )
    result = json.loads(completed.stdout)
    assert result["source_original_sha256"] == sha256_file(source)
    assert result["state_dict_bitwise_equal"] is True
    assert result["source_forward_captured_before_scrub"] is True
    assert result["forward_max_abs_difference"] == 0.0
    assert result["ultralytics_load"] == "PASS"
    assert destination.stat().st_size < source.stat().st_size * 1.2
