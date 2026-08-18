import json
from pathlib import Path

import pytest

from mcu_data.common import sha256_file
from mcu_data.publishing import publish_evidence_file, validate_comparison_for_run


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


def test_validate_comparison_requires_exact_run_manifest(tmp_path: Path) -> None:
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
        json.dumps({"comparable": True, "release_ready": False}), encoding="utf-8"
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

    with pytest.raises(ValueError, match="not formal-release ready"):
        validate_comparison_for_run(comparison, run_id=run_id, run_manifest_path=run_manifest)
    (comparison / "protocol_compatibility.json").write_text(
        json.dumps({"comparable": True, "release_ready": True, "run_provenance": provenance}),
        encoding="utf-8",
    )
    evidence = validate_comparison_for_run(
        comparison, run_id=run_id, run_manifest_path=run_manifest
    )
    assert evidence["run_id"] == run_id
    assert evidence["native_final_metrics_sha256"] == sha256_file(final_metrics)
    run_manifest.write_text(json.dumps({"run_id": run_id, "status": "changed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after"):
        validate_comparison_for_run(comparison, run_id=run_id, run_manifest_path=run_manifest)
