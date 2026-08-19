from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = PROJECT_ROOT / "data" / "manifests" / "wikimedia_stm32_v1.acquisition-probe.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "sources.wikimedia.yaml"


def test_wikimedia_stm32_probe_is_quarantine_only_and_count_consistent() -> None:
    probe = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    records = probe["review_records"]

    assert probe["status"] == "QUARANTINE_REVIEWED_INSUFFICIENT"
    assert probe["source"]["tracked_images"] is False
    assert probe["source"]["quarantine_only"] is True
    assert probe["source"]["source_page_revision_ids_recorded"] is False
    assert probe["training_use"]["allowed"] is False
    assert probe["training_use"]["formal_evaluation_allowed"] is False
    assert probe["training_use"]["approved_images"] == 0
    assert probe["training_use"]["approved_annotations"] == 0
    assert probe["quarantine_integrity"]["records_rehashed"] == 21
    assert probe["quarantine_integrity"]["missing_files"] == 0
    assert probe["quarantine_integrity"]["sha256_mismatches"] == 0

    assert len(records) == 21
    assert len({record["source_page_id"] for record in records}) == 21
    assert len({record["sha256"] for record in records}) == 21
    decisions = Counter(record["visual_decision"] for record in records)
    assert decisions == {
        "CANDIDATE_STM32_DEV_BOARD_NOT_APPROVED": 11,
        "REROUTE_CANDIDATE_STM32_DEV_BOARD_NOT_APPROVED": 2,
        "REJECT_STM32_BARE_IC_MOUNTED_PACKAGE": 5,
        "REJECT_STM32_BARE_IC_DIE_MICROGRAPH": 3,
    }

    for record in records:
        assert record["source_page_url"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert len(record["sha256"]) == 64
        assert record["license"]
        assert record["license_url"].startswith(("http://", "https://"))
        assert record["artist"]
        assert record["reason"]


def test_wikimedia_dev_board_groups_cover_every_visual_candidate_once() -> None:
    probe = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    candidate_ids = {
        record["source_page_id"]
        for record in probe["review_records"]
        if "CANDIDATE_STM32_DEV_BOARD" in record["visual_decision"]
    }
    grouped_ids = [
        page_id
        for group in probe["dev_board_grouping"]["groups"]
        for page_id in group["source_page_ids"]
    ]

    assert len(probe["dev_board_grouping"]["groups"]) == 8
    assert len(grouped_ids) == 13
    assert len(set(grouped_ids)) == 13
    assert set(grouped_ids) == candidate_ids


def test_wikimedia_bare_ic_search_filters_known_false_positive_titles() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    exclusions = {value.casefold() for value in config["classes"]["stm32_bare_ic"]["exclude_any"]}

    assert {"board", "embedded world", "microstm32", "-hd"} <= exclusions
    assert config["download"]["max_decoded_pixels"] == 25_000_000
    assert config["download"]["max_decoded_dimension_px"] == 8192
    assert config["download"]["max_frames"] == 1
