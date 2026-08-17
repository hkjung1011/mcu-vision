from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mcu_data.common import portable_path, sha256_file


Framework = Literal["yolo11", "yolox"]


@dataclass(frozen=True)
class CocoSample:
    annotation_path: Path
    image_root: Path
    image_path: Path
    image_id: int
    file_name: str
    width: int
    height: int
    annotation_count: int
    categories: tuple[dict[str, Any], ...]

    def evidence(self) -> dict[str, Any]:
        return {
            "selection": "COCO image record; lowest requested/annotated image ID",
            "annotation_path": portable_path(self.annotation_path),
            "annotation_sha256": sha256_file(self.annotation_path),
            "image_root": portable_path(self.image_root),
            "image_path": portable_path(self.image_path),
            "image_sha256": sha256_file(self.image_path),
            "image_id": self.image_id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "annotation_count": self.annotation_count,
            "synthetic_or_generated": False,
        }


@dataclass(frozen=True)
class PreprocessInfo:
    framework: Framework
    input_height: int
    input_width: int
    original_height: int
    original_width: int
    scale: float
    pad_left: float
    pad_top: float
    color_order: str
    normalization: str
    padding_value: int = 114

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def load_coco_sample(
    annotation_path: Path,
    image_root: Path,
    image_id: int | None = None,
) -> CocoSample:
    """Resolve one real image referenced by a COCO annotation document.

    When ``image_id`` is omitted, the lowest image ID having at least one
    annotation is selected. This keeps repeated export verification stable.
    """

    annotation_path = annotation_path.resolve()
    image_root = image_root.resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)

    document = _read_json(annotation_path)
    images = document.get("images")
    annotations = document.get("annotations", [])
    categories = document.get("categories")
    if not isinstance(images, list) or not images:
        raise ValueError(f"COCO document has no image records: {annotation_path}")
    if not isinstance(annotations, list):
        raise ValueError(f"COCO annotations must be a list: {annotation_path}")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"COCO document has no categories: {annotation_path}")

    normalized_categories = tuple(sorted(categories, key=lambda item: int(item["id"])))
    category_ids = [int(item["id"]) for item in normalized_categories]
    if category_ids != list(range(1, len(category_ids) + 1)):
        raise ValueError(
            "Deployment export requires consecutive COCO category IDs starting at 1; "
            f"found {category_ids}"
        )

    annotation_counts: dict[int, int] = {}
    for annotation in annotations:
        current_id = int(annotation["image_id"])
        annotation_counts[current_id] = annotation_counts.get(current_id, 0) + 1

    indexed_images = {int(item["id"]): item for item in images}
    if image_id is None:
        annotated_ids = sorted(set(indexed_images).intersection(annotation_counts))
        selected_id = annotated_ids[0] if annotated_ids else min(indexed_images)
    else:
        selected_id = int(image_id)
    if selected_id not in indexed_images:
        raise ValueError(f"COCO image_id {selected_id} is not present in {annotation_path}")

    record = indexed_images[selected_id]
    file_name = str(record["file_name"])
    candidate = (image_root / file_name).resolve()
    try:
        candidate.relative_to(image_root)
    except ValueError as error:
        raise ValueError(f"COCO file_name escapes image root: {file_name}") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    return CocoSample(
        annotation_path=annotation_path,
        image_root=image_root,
        image_path=candidate,
        image_id=selected_id,
        file_name=file_name,
        width=int(record["width"]),
        height=int(record["height"]),
        annotation_count=annotation_counts.get(selected_id, 0),
        categories=normalized_categories,
    )


def category_names(sample: CocoSample) -> list[str]:
    return [str(category["name"]) for category in sample.categories]


def preprocessing_spec(framework: Framework, image_size: int) -> dict[str, Any]:
    if framework == "yolo11":
        return {
            "resize": "preserve aspect ratio; scale=min(input_h/original_h,input_w/original_w)",
            "resized_dimensions": "round(original_dimension * scale)",
            "placement": "centered letterbox",
            "padding_value": 114,
            "color": "OpenCV BGR -> RGB",
            "layout": "HWC -> NCHW",
            "dtype": "float32",
            "normalization": "divide by 255.0",
            "input_shape": [1, 3, image_size, image_size],
        }
    if framework == "yolox":
        return {
            "resize": "preserve aspect ratio; scale=min(input_h/original_h,input_w/original_w)",
            "resized_dimensions": "int(original_dimension * scale)",
            "placement": "top-left letterbox",
            "padding_value": 114,
            "color": "OpenCV BGR retained (YOLOX ValTransform convention)",
            "layout": "HWC -> NCHW",
            "dtype": "float32",
            "normalization": "none; values remain 0..255",
            "input_shape": [1, 3, image_size, image_size],
        }
    raise ValueError(f"Unsupported framework: {framework}")


def preprocess_image(
    image_bgr: np.ndarray,
    framework: Framework,
    image_size: int,
) -> tuple[np.ndarray, PreprocessInfo]:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected HWC BGR image with 3 channels, got {image_bgr.shape}")
    if framework == "yolo11":
        return _preprocess_yolo11(image_bgr, image_size)
    if framework == "yolox":
        return _preprocess_yolox(image_bgr, image_size)
    raise ValueError(f"Unsupported framework: {framework}")


def _preprocess_yolo11(image_bgr: np.ndarray, image_size: int) -> tuple[np.ndarray, PreprocessInfo]:
    import cv2

    original_height, original_width = image_bgr.shape[:2]
    scale = min(image_size / original_height, image_size / original_width)
    resized_width = max(1, min(image_size, round(original_width * scale)))
    resized_height = max(1, min(image_size, round(original_height * scale)))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    horizontal = image_size - resized_width
    vertical = image_size - resized_height
    left = round(horizontal / 2 - 0.1)
    right = round(horizontal / 2 + 0.1)
    top = round(vertical / 2 - 0.1)
    bottom = round(vertical / 2 + 0.1)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = padded[:, :, ::-1]
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor[None, ...], PreprocessInfo(
        framework="yolo11",
        input_height=image_size,
        input_width=image_size,
        original_height=original_height,
        original_width=original_width,
        scale=scale,
        pad_left=float(left),
        pad_top=float(top),
        color_order="BGR input -> RGB tensor",
        normalization="float32 / 255.0",
    )


def _preprocess_yolox(image_bgr: np.ndarray, image_size: int) -> tuple[np.ndarray, PreprocessInfo]:
    import cv2

    original_height, original_width = image_bgr.shape[:2]
    scale = min(image_size / original_height, image_size / original_width)
    resized_width = max(1, min(image_size, int(original_width * scale)))
    resized_height = max(1, min(image_size, int(original_height * scale)))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return tensor[None, ...], PreprocessInfo(
        framework="yolox",
        input_height=image_size,
        input_width=image_size,
        original_height=original_height,
        original_width=original_width,
        scale=scale,
        pad_left=0.0,
        pad_top=0.0,
        color_order="BGR tensor (YOLOX ValTransform convention)",
        normalization="float32 0..255 (no division)",
    )


def compare_arrays(
    native: np.ndarray,
    exported: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("Numerical tolerances must be non-negative")
    native = np.asarray(native)
    exported = np.asarray(exported)
    result: dict[str, Any] = {
        "native_shape": list(native.shape),
        "onnx_shape": list(exported.shape),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "element_count": int(native.size),
    }
    if native.shape != exported.shape:
        return result | {
            "status": "FAIL",
            "reason": "shape_mismatch",
            "finite": False,
            "within_tolerance_fraction": 0.0,
        }
    finite = bool(np.isfinite(native).all() and np.isfinite(exported).all())
    if not finite:
        return result | {
            "status": "FAIL",
            "reason": "non_finite_output",
            "finite": False,
            "within_tolerance_fraction": 0.0,
        }
    difference = np.abs(native.astype(np.float64) - exported.astype(np.float64))
    limit = absolute_tolerance + relative_tolerance * np.abs(native.astype(np.float64))
    within = difference <= limit
    return result | {
        "status": "PASS" if bool(within.all()) else "FAIL",
        "reason": "all_elements_within_tolerance" if bool(within.all()) else "tolerance_exceeded",
        "finite": True,
        "max_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()) if difference.size else 0.0,
        "p99_absolute_error": float(np.percentile(difference, 99)) if difference.size else 0.0,
        "within_tolerance_fraction": float(within.mean()) if within.size else 1.0,
    }


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(np.float32, copy=True)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(right_bottom - left_top, 0.0)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    box_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    union = box_area + boxes_area - intersection_area
    return np.divide(intersection_area, union, out=np.zeros_like(union), where=union > 0)


def class_aware_nms(detections: np.ndarray, iou_threshold: float, max_detections: int) -> np.ndarray:
    """Apply deterministic per-class NMS to ``[x1,y1,x2,y2,score,class]`` rows."""

    detections = np.asarray(detections, dtype=np.float32)
    if detections.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError(f"Expected Nx6 detections, got {detections.shape}")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")

    kept: list[int] = []
    for class_id in np.unique(detections[:, 5].astype(np.int64)):
        indices = np.flatnonzero(detections[:, 5].astype(np.int64) == class_id)
        order = indices[np.argsort(-detections[indices, 4], kind="stable")]
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            ious = _box_iou_one_to_many(detections[current, :4], detections[remaining, :4])
            order = remaining[ious <= iou_threshold]
    kept_array = np.asarray(kept, dtype=np.int64)
    kept_array = kept_array[np.argsort(-detections[kept_array, 4], kind="stable")]
    return detections[kept_array[:max_detections]]


def decode_detections(
    output: np.ndarray,
    *,
    framework: Framework,
    class_count: int,
    confidence: float,
    nms_iou: float,
    max_detections: int,
) -> np.ndarray:
    """Decode a batch-one native/ONNX output in the repository export format."""

    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    output = np.asarray(output)
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Expected a batch-one rank-3 output, got {output.shape}")
    if framework == "yolo11":
        expected_channels = 4 + class_count
        if output.shape[1] != expected_channels:
            raise ValueError(
                f"YOLO11 output channel mismatch: expected {expected_channels}, got {output.shape[1]}"
            )
        rows = output[0].transpose(1, 0)
        class_scores = rows[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(rows.shape[0]), class_ids]
    elif framework == "yolox":
        expected_channels = 5 + class_count
        if output.shape[2] != expected_channels:
            raise ValueError(
                f"YOLOX output channel mismatch: expected {expected_channels}, got {output.shape[2]}"
            )
        rows = output[0]
        class_scores = rows[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        scores = rows[:, 4] * class_scores[np.arange(rows.shape[0]), class_ids]
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    selected = scores >= confidence
    if not selected.any():
        return np.empty((0, 6), dtype=np.float32)
    boxes = _xywh_to_xyxy(rows[selected, :4])
    candidates = np.column_stack((boxes, scores[selected], class_ids[selected])).astype(np.float32)
    return class_aware_nms(candidates, nms_iou, max_detections)


def restore_boxes(detections: np.ndarray, info: PreprocessInfo) -> np.ndarray:
    restored = np.asarray(detections, dtype=np.float32).copy()
    if restored.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    restored[:, [0, 2]] = (restored[:, [0, 2]] - info.pad_left) / info.scale
    restored[:, [1, 3]] = (restored[:, [1, 3]] - info.pad_top) / info.scale
    restored[:, [0, 2]] = restored[:, [0, 2]].clip(0, info.original_width)
    restored[:, [1, 3]] = restored[:, [1, 3]].clip(0, info.original_height)
    return restored


def artifact_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
