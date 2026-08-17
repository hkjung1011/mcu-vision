import json
from pathlib import Path

from mcu_data.publishing import publish_evidence_file


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
