import json
from pathlib import Path

import yaml

from mcu_data.autolabel import class_aware_nms, tile_starts
from mcu_data.common import sha256_file
from mcu_data.methodology import write_protocol_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tile_starts_covers_last_edge_without_duplicate() -> None:
    assert tile_starts(500, 640, 0.2) == [0]
    assert tile_starts(1280, 640, 0.2) == [0, 512, 640]


def test_class_aware_nms_keeps_overlapping_different_classes() -> None:
    predictions = [
        {"class_id": 0, "confidence": 0.9, "xyxy": [0.0, 0.0, 10.0, 10.0]},
        {"class_id": 0, "confidence": 0.8, "xyxy": [1.0, 1.0, 11.0, 11.0]},
        {"class_id": 1, "confidence": 0.7, "xyxy": [1.0, 1.0, 11.0, 11.0]},
    ]
    kept = class_aware_nms(predictions, 0.5)
    assert [(row["class_id"], row["confidence"]) for row in kept] == [(0, 0.9), (1, 0.7)]


def test_protocol_artifacts_are_generated(tmp_path: Path) -> None:
    source = tmp_path / "protocol.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "protocol_id": "test",
                "status": "PROVISIONAL",
                "experiment_type": "system_benchmark",
                "common": {"image_size": 640},
                "yolo11m": {},
                "yolox_s": {},
                "comparison_rules": {},
                "rationale": [
                    {
                        "id": "R01",
                        "item": "image_size",
                        "label": "Input size",
                        "selected_value": 640,
                        "status": "TO_TUNE",
                        "reason": "test reason",
                        "adjustment_rule": "test adjustment",
                        "optimality": "not optimal",
                        "verification_status": "smoke only",
                        "references": ["REF"],
                    }
                ],
                "references": [
                    {"id": "REF", "title": "Reference", "url": "https://example.com", "type": "paper"}
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    result = write_protocol_artifacts(source, output, print_terminal=False)
    assert result["rationale_items"] == 1
    assert (output / "protocol_rationale.csv").exists()
    assert (output / "protocol_rationale.png").exists()
    assert (output / "parameter_rationale.md").exists()
    assert "test reason" in (output / "experiment_methodology.md").read_text(encoding="utf-8")
    assert "smoke only" in (output / "parameter_rationale.md").read_text(encoding="utf-8")


def test_protocol_artifacts_are_autocrlf_independent(tmp_path: Path) -> None:
    document = {
        "protocol_id": "newline-stable",
        "status": "PROVISIONAL",
        "experiment_type": "system_benchmark",
        "task": "fixture",
        "common": {"image_size": 640},
        "yolo11m": {},
        "yolox_s": {},
        "comparison_rules": {},
        "rationale": [
            {
                "id": "R01",
                "item": "image_size",
                "label": "Input size",
                "selected_value": 640,
                "status": "TO_TUNE",
                "reason": "newline fixture",
                "adjustment_rule": "fixture",
                "optimality": "not optimal",
                "verification_status": "fixture",
                "references": [],
            }
        ],
        "references": [],
    }
    yaml_text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    lf_source = tmp_path / "lf" / "protocol.yaml"
    crlf_source = tmp_path / "crlf" / "protocol.yaml"
    lf_source.parent.mkdir()
    crlf_source.parent.mkdir()
    lf_source.write_bytes(yaml_text.encode("utf-8"))
    crlf_source.write_bytes(yaml_text.replace("\n", "\r\n").encode("utf-8"))

    lf_output = tmp_path / "lf-output"
    crlf_output = tmp_path / "crlf-output"
    lf_result = write_protocol_artifacts(lf_source, lf_output, print_terminal=False)
    crlf_result = write_protocol_artifacts(crlf_source, crlf_output, print_terminal=False)

    assert lf_result["source_sha256"] == crlf_result["source_sha256"]
    assert lf_result["artifacts"] == crlf_result["artifacts"]
    for record in lf_result["artifacts"]:
        relative = record["path"]
        assert (lf_output / relative).read_bytes() == (crlf_output / relative).read_bytes()
        if Path(relative).suffix in {".yaml", ".csv", ".json", ".md"}:
            assert b"\r" not in (lf_output / relative).read_bytes()
    assert (lf_output / "protocol_artifacts.json").read_bytes() == (
        crlf_output / "protocol_artifacts.json"
    ).read_bytes()


def test_baseline_rationale_has_decision_and_verification_fields() -> None:
    protocol = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments" / "baseline_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    rationale = protocol["rationale"]
    assert len(rationale) == 14
    assert len({item["id"] for item in rationale}) == len(rationale)
    required = {
        "label",
        "selected_value",
        "status",
        "reason",
        "adjustment_rule",
        "optimality",
        "verification_status",
        "references",
    }
    assert all(required <= set(item) for item in rationale)
    reference_ids = {item["id"] for item in protocol["references"]}
    assert all(set(item["references"]) <= reference_ids for item in rationale)


def test_committed_methodology_records_match_lf_stable_files() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "reports" / "methodology" / "protocol_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    for record in manifest["artifacts"]:
        path = PROJECT_ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
        if path.suffix in {".yaml", ".csv", ".json", ".md"}:
            assert b"\r" not in path.read_bytes()
