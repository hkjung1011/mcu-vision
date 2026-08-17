from pathlib import Path

import yaml

from mcu_data.autolabel import class_aware_nms, tile_starts
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


def test_baseline_rationale_has_decision_and_verification_fields() -> None:
    protocol = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments" / "baseline_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    rationale = protocol["rationale"]
    assert len(rationale) == 12
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
