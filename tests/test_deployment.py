from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mcu_data.deployment import (
    PreprocessInfo,
    compare_arrays,
    decode_detections,
    load_coco_sample,
    preprocessing_spec,
    restore_boxes,
)


def _write_coco(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 2, "file_name": "two.jpg", "width": 20, "height": 10},
                    {"id": 1, "file_name": "one.jpg", "width": 30, "height": 15},
                ],
                "annotations": [{"id": 1, "image_id": 2, "category_id": 1, "bbox": [1, 1, 2, 2]}],
                "categories": [{"id": 1, "name": "chip"}],
            }
        ),
        encoding="utf-8",
    )


def test_load_coco_sample_prefers_lowest_annotated_id(tmp_path: Path) -> None:
    annotation = tmp_path / "instances.json"
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"one")
    (images / "two.jpg").write_bytes(b"two")
    _write_coco(annotation)

    sample = load_coco_sample(annotation, images)

    assert sample.image_id == 2
    assert sample.annotation_count == 1
    assert sample.image_path == (images / "two.jpg").resolve()
    assert sample.evidence()["synthetic_or_generated"] is False
    assert len(sample.evidence()["image_sha256"]) == 64


def test_load_coco_sample_rejects_path_escape(tmp_path: Path) -> None:
    annotation = tmp_path / "instances.json"
    images = tmp_path / "images"
    images.mkdir()
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "../outside.jpg", "width": 1, "height": 1}],
                "annotations": [],
                "categories": [{"id": 1, "name": "chip"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes image root"):
        load_coco_sample(annotation, images)


def test_compare_arrays_reports_exact_gate() -> None:
    native = np.array([1.0, 10.0, 0.0], dtype=np.float32)
    close = np.array([1.00001, 10.0005, 0.00005], dtype=np.float32)
    far = np.array([1.0, 10.0, 0.1], dtype=np.float32)

    passing = compare_arrays(native, close, absolute_tolerance=1e-4, relative_tolerance=1e-4)
    failing = compare_arrays(native, far, absolute_tolerance=1e-4, relative_tolerance=1e-4)

    assert passing["status"] == "PASS"
    assert passing["within_tolerance_fraction"] == 1.0
    assert failing["status"] == "FAIL"
    assert failing["within_tolerance_fraction"] == pytest.approx(2 / 3)


def test_decode_yolo11_and_yolox_use_class_aware_nms() -> None:
    yolo11 = np.array(
        [[[50.0, 51.0], [50.0, 51.0], [20.0, 20.0], [20.0, 20.0], [0.9, 0.8]]],
        dtype=np.float32,
    )
    yolox = np.array(
        [[[50.0, 50.0, 20.0, 20.0, 0.9, 1.0], [51.0, 51.0, 20.0, 20.0, 0.8, 1.0]]],
        dtype=np.float32,
    )

    decoded_yolo11 = decode_detections(
        yolo11,
        framework="yolo11",
        class_count=1,
        confidence=0.25,
        nms_iou=0.65,
        max_detections=100,
    )
    decoded_yolox = decode_detections(
        yolox,
        framework="yolox",
        class_count=1,
        confidence=0.25,
        nms_iou=0.65,
        max_detections=100,
    )

    assert decoded_yolo11.shape == (1, 6)
    assert decoded_yolox.shape == (1, 6)
    assert decoded_yolo11[0, 4] == pytest.approx(0.9)
    assert decoded_yolox[0, 4] == pytest.approx(0.9)


def test_restore_boxes_reverses_letterbox() -> None:
    info = PreprocessInfo(
        framework="yolo11",
        input_height=640,
        input_width=640,
        original_height=300,
        original_width=500,
        scale=1.0,
        pad_left=70.0,
        pad_top=170.0,
        color_order="test",
        normalization="test",
    )
    detections = np.array([[70.0, 170.0, 570.0, 470.0, 0.9, 0]], dtype=np.float32)

    restored = restore_boxes(detections, info)

    np.testing.assert_allclose(restored[0, :4], [0.0, 0.0, 500.0, 300.0])


def test_preprocessing_specs_make_framework_difference_explicit() -> None:
    yolo11 = preprocessing_spec("yolo11", 640)
    yolox = preprocessing_spec("yolox", 640)

    assert yolo11["placement"] == "centered letterbox"
    assert yolo11["normalization"] == "divide by 255.0"
    assert yolox["placement"] == "top-left letterbox"
    assert yolox["normalization"] == "none; values remain 0..255"
