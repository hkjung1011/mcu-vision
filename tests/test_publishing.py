import json
import subprocess
import sys
from pathlib import Path

import pytest

from mcu_data.common import sha256_file
from mcu_data.publishing import (
    assert_public_binary_privacy,
    load_json_strict,
    load_jsonl_strict,
    publish_evidence_file,
    scan_public_file,
    validate_comparison_for_run,
)


def test_publish_evidence_file_redacts_local_paths_and_process_table(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "runs" / "run_manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "checkpoint": str(project / "runs" / "best.pt"),
                "environment": {"nvidia_smi": "process list"},
                "metric": 0.123456789,
            }
        ),
        encoding="utf-8",
    )
    destination = project / "reports" / "run_manifest.json"

    record = publish_evidence_file(source, destination, project_root=project)
    published = json.loads(destination.read_text(encoding="utf-8"))

    assert published["checkpoint"].startswith("<PROJECT_ROOT>")
    assert published["environment"]["nvidia_smi"] == "<OMITTED_FROM_PUBLISHED_REPORT>"
    assert published["metric"] == 0.123456789
    assert record["sanitized_for_repository"] is True
    assert record["source_original_sha256"] != record["published_sha256"]


def test_publish_evidence_file_redacts_python_repr_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "terminal.log"
    source.parent.mkdir(parents=True)
    escaped_project = str(project).replace("\\", "\\\\")
    source.write_text(f"Namespace(config='{escaped_project}\\\\config.yaml')\n", encoding="utf-8")
    destination = project / "published" / "terminal.log"

    publish_evidence_file(source, destination, project_root=project)

    assert escaped_project.lower() not in destination.read_text(encoding="utf-8").lower()
    assert "<PROJECT_ROOT>" in destination.read_text(encoding="utf-8")


def test_publish_evidence_file_redacts_text_nvidia_process_table(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "terminal.log"
    source.parent.mkdir(parents=True)
    source.write_text(
        "GPU summary\n"
        "+----------------+\n"
        "| Processes:     |\n"
        "| PID chrome.exe |\n"
        "+----------------+\n"
        "metric=0.987654321\n",
        encoding="utf-8",
    )
    destination = project / "published" / "terminal.log"

    record = publish_evidence_file(source, destination, project_root=project)
    published = destination.read_text(encoding="utf-8")

    assert "chrome.exe" not in published
    assert "PROCESS LIST OMITTED" in published
    assert "metric=0.987654321" in published
    assert record["sanitized_for_repository"] is True


def test_publish_evidence_file_accepts_utf8_bom_json(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "campaign_plan.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"metric": 0.5}', encoding="utf-8-sig")
    destination = project / "published" / "campaign_plan.json"

    publish_evidence_file(source, destination, project_root=project)

    assert json.loads(destination.read_text(encoding="utf-8-sig"))["metric"] == 0.5


@pytest.mark.parametrize(
    ("suffix", "payload", "loader"),
    [
        (".json", '{"seed": 42, "seed": 43}\n', load_json_strict),
        (".jsonl", '{"seed": 42, "seed": 43}\n', load_jsonl_strict),
    ],
)
def test_strict_publication_rejects_duplicate_json_keys(
    tmp_path: Path,
    suffix: str,
    payload: str,
    loader: object,
) -> None:
    source = tmp_path / f"evidence{suffix}"
    source.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        loader(source)  # type: ignore[operator]
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        publish_evidence_file(
            source,
            tmp_path / "published" / source.name,
            project_root=tmp_path,
        )


def test_publication_serialization_is_deterministic_not_parse_only(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b'{"z":1,"a":{"path":"C:\\\\fixture\\\\x"}}\r\n')
    second.write_bytes(b'{\n  "a": {"path": "C:\\\\fixture\\\\x"},\n  "z": 1\n}\n')
    first_public = tmp_path / "public" / "first.json"
    second_public = tmp_path / "public" / "second.json"
    publish_evidence_file(first, first_public, project_root=tmp_path)
    publish_evidence_file(second, second_public, project_root=tmp_path)
    assert first_public.read_bytes() == second_public.read_bytes()
    assert first_public.read_bytes().endswith(b"\n")
    assert b"\r" not in first_public.read_bytes()


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        ("image.png", b"\xff\xd8\xffjpeg", "signature does not match"),
        ("report.txt", b"\x89PNG\r\n\x1a\ndisguised", "disguised png"),
        ("report.txt", b"nul\x00text", "binary/NUL"),
        ("report.txt", b"private=/opt/build/secret/file", "absolute local path"),
        ("report.txt", b"private=/secret.txt", "absolute local path"),
        ("report.txt", b"private=D:\\build\\secret.txt", "absolute local path"),
        ("checkpoint.pth", b"not-a-torch-file", "unknown signature"),
    ],
)
def test_public_file_scan_rejects_signature_text_and_generic_path_spoofs(
    tmp_path: Path,
    filename: str,
    payload: bytes,
    message: str,
) -> None:
    path = tmp_path / filename
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        scan_public_file(path, relative_path=filename)


def test_public_file_scan_accepts_matching_png_and_torch_magic(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"PK\x03\x04fixture")
    assert scan_public_file(image, relative_path="image.png")["detected_magic"] == "png"
    assert scan_public_file(checkpoint, relative_path="weights/best.pth")["detected_magic"] == "zip"


@pytest.mark.parametrize("payload", [b"C:\\x", b"\\\\s\\x", b"/a"])
def test_binary_privacy_scan_rejects_short_absolute_paths(payload: bytes) -> None:
    with pytest.raises(ValueError, match="absolute local path"):
        assert_public_binary_privacy(payload, label="short-path.bin")


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("evidence.json", '{"metric": 1e999}\n'),
        ("evidence.jsonl", '{"metric": 1e999}\n'),
        ("evidence.yaml", "metric: 1e999\n"),
        ("evidence.yml", "metric: .inf\n"),
    ],
)
def test_publication_rejects_non_finite_exponents_in_structured_text(
    tmp_path: Path,
    filename: str,
    payload: str,
) -> None:
    source = tmp_path / filename
    source.write_text(payload, encoding="utf-8")
    destination = tmp_path / "public" / filename
    with pytest.raises(ValueError, match="Non-finite"):
        publish_evidence_file(source, destination, project_root=tmp_path)
    with pytest.raises(ValueError, match="Non-finite"):
        scan_public_file(source, relative_path=filename)


def test_strict_json_accepts_largest_finite_exponent(tmp_path: Path) -> None:
    source = tmp_path / "finite.json"
    source.write_text('{"metric": 1e308}\n', encoding="utf-8")
    assert load_json_strict(source)["metric"] == 1e308


def test_hash_bound_public_trees_preserve_lf_in_autocrlf_clone(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    source_repo = tmp_path / "source"
    clone_repo = tmp_path / "clone"
    source_repo.mkdir()
    (source_repo / ".gitattributes").write_bytes((project / ".gitattributes").read_bytes())
    relative_paths = (
        "reports/comparisons/release/evidence.json",
        "reports/runs/release/artifact_manifest.json",
        "reports/deployments/release/deployment_release_manifest.json",
        "weights/trained/release/deployment.json",
    )
    payload = b'{\n  "status": "PASS"\n}\n'
    for relative in relative_paths:
        path = source_repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    subprocess.run(["git", "init", "-q"], cwd=source_repo, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=source_repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=source_repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=source_repo, check=True)
    subprocess.run(["git", "add", "."], cwd=source_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source_repo, check=True)
    subprocess.run(
        ["git", "-c", "core.autocrlf=true", "clone", "-q", str(source_repo), str(clone_repo)],
        check=True,
    )

    for relative in relative_paths:
        assert (clone_repo / relative).read_bytes() == payload
        attribute = subprocess.run(
            ["git", "check-attr", "text", "--", relative],
            cwd=clone_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert attribute.endswith("text: unset")


@pytest.mark.parametrize(
    "local_path",
    [r"D:\Users\AnotherPerson\private\weights.pt", "E:/Users/BuildAgent/secret/file.json"],
)
def test_publish_evidence_file_redacts_any_windows_user_profile(
    tmp_path: Path, local_path: str
) -> None:
    project = tmp_path / "project"
    source = project / "source.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"path": local_path}), encoding="utf-8")
    destination = project / "published.json"
    publish_evidence_file(source, destination, project_root=project)
    published = destination.read_text(encoding="utf-8")
    assert "AnotherPerson" not in published
    assert "BuildAgent" not in published
    assert "<USER_HOME>" in published


def test_validate_comparison_rejects_forged_one_run_release_ready_claim(tmp_path: Path) -> None:
    run_id = "full_seed42"
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps({"run_id": run_id, "status": "complete"}), encoding="utf-8")
    final_metrics = tmp_path / "final_metrics.json"
    final_metrics.write_text(json.dumps({"metrics": {"ap50_95": 0.5}}), encoding="utf-8")
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    provenance = {"status": "PASS", "schema_version": 2, "mixed_commits": False}
    (comparison / "run_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    (comparison / "protocol_compatibility.json").write_text(
        json.dumps({
            "comparable": True, "release_ready": True, "critical_mismatches": [],
            "release_blockers": [], "run_count": 6,
            "release_expectations": {"models": ["yolo11m", "YOLOX-S"], "seeds": [42, 43, 44], "runs": 6},
            "run_provenance": provenance,
        }), encoding="utf-8"
    )
    (comparison / "comparison.json").write_text(
        json.dumps([{"run_id": run_id, "status": "complete"}]), encoding="utf-8"
    )
    (comparison / "sources_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "run_id": run_id,
                        "path": f"sources/{run_id}/run_manifest.json",
                        "source_original_sha256": sha256_file(run_manifest),
                    },
                    {
                        "run_id": run_id,
                        "path": f"sources/{run_id}/final_metrics.json",
                        "source_original_sha256": sha256_file(final_metrics),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly six rows"):
        validate_comparison_for_run(comparison, run_id=run_id, run_manifest_path=run_manifest)
    project = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project / "scripts" / "promote_comparison.py"),
            "--comparison-dir",
            str(comparison),
            "--release-name",
            "forged_one_run_fixture",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "exactly six rows" in completed.stderr
