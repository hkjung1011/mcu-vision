from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_formal_model_cards_bind_current_release_evidence() -> None:
    cases = {
        "yolo11m": {
            "release": "rpi_phash_v2_paired2_yolo11m",
            "card": ROOT / "docs/model_cards/rpi_phash_v2_paired2_yolo11m.ko.md",
            "checkpoint": "best.pt",
            "selected_run": "yolo11m_seed43",
            "framework_version": "8.4.120",
            "upstream_license_literal": "AGPL-3.0",
            "notice_component": "[Ultralytics]",
        },
        "yolox_s": {
            "release": "rpi_phash_v2_paired2_yolox_s",
            "card": ROOT / "docs/model_cards/rpi_phash_v2_paired2_yolox_s.ko.md",
            "checkpoint": "best_ckpt.pth",
            "selected_run": "yolox_s_seed43",
            "framework_version": "0.3.0",
            "upstream_license_literal": "Apache-2.0",
            "notice_component": "[YOLOX]",
        },
    }

    split_summary = _read_json(
        ROOT / "data/manifests/micropcb_raspberry_pi_sbc.phash_v2.summary.json"
    )
    dataset_source = _read_json(
        ROOT / "data/manifests/micro_pcb_images.dataset-source.json"
    )
    third_party_notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert split_summary["split_image_counts"] == {
        "train": 1500,
        "test": 180,
        "val": 195,
    }

    for case in cases.values():
        release = case["release"]
        card = case["card"].read_text(encoding="utf-8")
        run_manifest = _read_json(
            ROOT / f"reports/runs/{release}/run_evidence/run_manifest.json"
        )
        artifact = _read_json(ROOT / f"reports/runs/{release}/artifact_manifest.json")
        native_metrics = _read_json(
            ROOT / f"reports/runs/{release}/run_evidence/final_metrics.json"
        )
        deployment = _read_json(
            ROOT / f"reports/deployments/{release}/deployment_metadata.json"
        )
        val_eval = _read_json(
            ROOT / f"reports/deployments/{release}/val/onnx_split_evaluation.json"
        )
        test_eval = _read_json(
            ROOT / f"reports/deployments/{release}/test/onnx_split_evaluation.json"
        )

        assert case["selected_run"] == run_manifest["run_id"]
        assert case["framework_version"] == str(run_manifest["framework_version"])
        assert artifact["status"] == "PASS"
        assert deployment["status"] == "PASS"
        assert val_eval["status"] == "PASS"
        assert test_eval["status"] == "PASS"

        checkpoint = artifact["checkpoint"]
        onnx = deployment["artifacts"]["onnx"]
        source_commit = run_manifest["git"]["commit"]
        parameters = run_manifest["model_details"]["parameters"]
        dataset_evidence_sha256 = run_manifest["dataset"][
            "equivalence_evidence_sha256"
        ]
        val_inference = val_eval["inference"]
        test_inference = test_eval["inference"]
        required_literals = (
            release,
            case["selected_run"],
            case["framework_version"],
            case["checkpoint"],
            checkpoint["sha256"],
            onnx["sha256"],
            source_commit,
            f"{parameters:,}",
            dataset_evidence_sha256,
            f"{native_metrics['metrics']['ap50_95']:.10f}",
            f"{native_metrics['metrics']['ar100']:.10f}",
            f"{val_eval['metrics']['ap50_95']:.10f}",
            f"{test_eval['metrics']['ap50_95']:.10f}",
            f"{val_inference['inference_ms_p50']:.3f}",
            f"{val_inference['inference_ms_p95']:.3f}",
            f"{test_inference['inference_ms_p50']:.3f}",
            f"{test_inference['inference_ms_p95']:.3f}",
            "train 1,500 / validation 195 / internal test 180",
            "paired_2seed_descriptive",
            "locked internal pHash split",
            "NOT_FOR_THRESHOLD_SELECTION",
            "NOT VERIFIED",
            dataset_source["license"],
            case["upstream_license_literal"],
        )
        for literal in required_literals:
            assert literal in card

        assert "<MODEL_" not in card
        assert "production-ready" in card
        assert "100 epochs" in card
        assert "0.25" in card
        assert "0.65" in card
        assert "physical specimen" in card
        for key in ("color", "placement", "normalization"):
            assert deployment["preprocessing"][key] in card
        notice_lines = [
            line
            for line in third_party_notices.splitlines()
            if case["notice_component"] in line
        ]
        assert len(notice_lines) == 1
        assert case["upstream_license_literal"] in notice_lines[0]
