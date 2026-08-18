from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import portable_path, safe_stem, sha256_file, write_json
from .methodology import load_protocol, write_protocol_artifacts
from .publishing import publish_evidence_file
from .run_provenance import verify_run_provenance
from .runlog import (
    checkpoint_file_record,
    collect_system_environment,
    configure_utf8_output,
    print_section,
)


EPOCH_COLUMNS = [
    "epoch",
    "elapsed_s",
    "train_total_loss",
    "train_box_loss",
    "train_iou_loss",
    "train_conf_loss",
    "train_cls_loss",
    "train_dfl_loss",
    "train_l1_loss",
    "val_box_loss",
    "val_cls_loss",
    "val_dfl_loss",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "lr",
    "train_peak_allocated_mib",
    "train_peak_reserved_mib",
    "gpu_peak_allocated_mib",
    "gpu_peak_reserved_mib",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _run_json(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    try:
        value = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        return {
            "available": False,
            "exit_code": completed.returncode,
            "error": (completed.stderr or completed.stdout).strip()[-1000:],
        }
    value["available"] = completed.returncode == 0
    return value


def build_status(root: Path) -> dict[str, Any]:
    system = collect_system_environment()
    collection_code = (
        "import json,importlib.metadata as m; print(json.dumps({"
        "'purpose':'dataset collection, audit, common evaluation and plots',"
        "'torch_required':False,'mcu_data_tools':m.version('mcu-data-tools'),"
        "'matplotlib':m.version('matplotlib'),'pycocotools':m.version('pycocotools')}))"
    )
    y11_code = (
        "import json,torch,ultralytics; print(json.dumps({"
        "'torch':torch.__version__,'cuda_runtime':torch.version.cuda,"
        "'cudnn':torch.backends.cudnn.version(),'cuda_available':torch.cuda.is_available(),"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'ultralytics':ultralytics.__version__}))"
    )
    yolox_code = (
        "import json,torch,yolox; print(json.dumps({"
        "'torch':torch.__version__,'cuda_runtime':torch.version.cuda,"
        "'cudnn':torch.backends.cudnn.version(),'cuda_available':torch.cuda.is_available(),"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'yolox':getattr(yolox,'__version__','source')}))"
    )
    environments = {
        "collection": _run_json(
            [str(root / ".venv-collect" / "Scripts" / "python.exe"), "-c", collection_code], root
        ),
        "yolo11": _run_json(
            [str(root / ".venv-yolo11" / "Scripts" / "python.exe"), "-c", y11_code], root
        ),
        "yolox": _run_json(
            [str(root / ".venv-yolox" / "Scripts" / "python.exe"), "-c", yolox_code], root
        ),
    }
    annotation_root = (
        root / "data" / "processed" / "micropcb_rpi_phash_v2_coco" / "annotations"
    )
    dataset: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        annotation_path = annotation_root / f"instances_{split}2017.json"
        value = _json(annotation_path, {})
        if value:
            dataset[split] = {
                "images": len(value.get("images", [])),
                "annotations": len(value.get("annotations", [])),
                "categories": len(value.get("categories", [])),
                "sha256": sha256_file(annotation_path),
                "path": str(annotation_path.resolve()),
            }
    checkpoints: dict[str, Any] = {}
    weight_root = root / "weights" / "pretrained"
    candidates = list(weight_root.glob("*")) if weight_root.exists() else []
    candidates.extend(root.glob("yolo*.pt"))
    for path in sorted(candidates):
        if path.is_file():
            checkpoints[path.name] = checkpoint_file_record(path)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except subprocess.SubprocessError:
        git_commit = None
        git_dirty = None
    return {
        "schema_version": 1,
        "project_root": str(root.resolve()),
        "cuda_toolkit_required_now": False,
        "cuda_toolkit_reason": (
            "PyTorch cu130 wheel contains the CUDA runtime; nvcc is only needed for custom CUDA "
            "extensions or a later target-side TensorRT build."
        ),
        "system": system,
        "environments": environments,
        "dataset": dataset,
        "pretrained_checkpoints": checkpoints,
        "git": {"commit": git_commit, "dirty": git_dirty},
    }


def print_status(status: dict[str, Any]) -> None:
    system = status["system"]
    print("\nMCU VISION - VERIFIED WINDOWS STATUS")
    print("=" * 60)
    print_section(
        "CUDA DECISION",
        {
            "CUDA Toolkit / nvcc": "NOT INSTALLED" if not system["cuda_toolkit_nvcc"] else "INSTALLED",
            "Required for current training": "NO",
            "Reason": "PyTorch cu130 wheel already includes CUDA runtime",
            "Install later when": "custom CUDA extension or Ubuntu TensorRT build is required",
        },
    )
    for name, value in status["environments"].items():
        print_section(f"ENVIRONMENT: {name}", value)
    dataset_rows: dict[str, Any] = {}
    for split, value in status["dataset"].items():
        dataset_rows[f"{split} images/boxes"] = f"{value['images']} / {value['annotations']}"
        dataset_rows[f"{split} sha256"] = value["sha256"]
    print_section("DATASET (PHASH/CONDITION BOOTSTRAP; NOT EXTERNAL CAMERA TEST)", dataset_rows)
    for name, value in status["pretrained_checkpoints"].items():
        print_section(
            f"PRETRAINED: {name}",
            {"size MiB": round(value["mib"], 3), "sha256": value["sha256"]},
        )


def status_main(argv: list[str] | None = None) -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description="Print verified CUDA, framework, data, and weight status")
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    status = build_status(args.project_root.resolve())
    print_status(status)
    if args.output:
        write_json(args.output, status)
        print(f"\nJSON saved: {args.output.resolve()}")


def normalize_yolo11_results(source: Path, destination: Path) -> list[dict[str, Any]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        native = list(csv.DictReader(handle))
    epoch_offset = 1 if native and int(float(native[0].get("epoch", 0))) == 0 else 0
    mapping = {
        "time": "elapsed_s",
        "train/box_loss": "train_box_loss",
        "train/cls_loss": "train_cls_loss",
        "train/dfl_loss": "train_dfl_loss",
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_cls_loss",
        "val/dfl_loss": "val_dfl_loss",
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map50_95",
        "lr/pg0": "lr",
    }
    rows: list[dict[str, Any]] = []
    for native_row in native:
        row = {column: "" for column in EPOCH_COLUMNS}
        row["epoch"] = int(float(native_row.get("epoch", 0))) + epoch_offset
        for old, new in mapping.items():
            value = native_row.get(old, "")
            row[new] = float(value) if value not in (None, "") else ""
        rows.append(row)
    _write_csv(destination, EPOCH_COLUMNS, rows)
    return rows


def normalize_yolox_tensorboard(
    tensorboard_dir: Path, destination: Path, steps_per_epoch: int
) -> list[dict[str, Any]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    train_mapping = {
        "train/total_loss": "train_total_loss",
        "train/iou_loss": "train_iou_loss",
        "train/conf_loss": "train_conf_loss",
        "train/cls_loss": "train_cls_loss",
        "train/l1_loss": "train_l1_loss",
        "train/lr": "lr",
    }
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for tag, column in train_mapping.items():
        if tag not in available:
            continue
        for event in accumulator.Scalars(tag):
            epoch = event.step // steps_per_epoch + 1
            grouped[epoch][column].append(float(event.value))
    rows_by_epoch: dict[int, dict[str, Any]] = {}
    for epoch, values in grouped.items():
        row = {column: "" for column in EPOCH_COLUMNS}
        row["epoch"] = epoch
        for column, samples in values.items():
            row[column] = statistics.fmean(samples)
        rows_by_epoch[epoch] = row
    for tag, column in [("val/COCOAP50", "map50"), ("val/COCOAP50_95", "map50_95")]:
        if tag not in available:
            continue
        for event in accumulator.Scalars(tag):
            epoch = int(event.step)
            row = rows_by_epoch.setdefault(
                epoch, {name: "" for name in EPOCH_COLUMNS} | {"epoch": epoch}
            )
            row[column] = float(event.value)
    rows = [rows_by_epoch[key] for key in sorted(rows_by_epoch)]
    _write_csv(destination, EPOCH_COLUMNS, rows)
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _iou_xywh(first: list[float], second: list[float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _limit_predictions_per_image(
    predictions: list[dict[str, Any]], max_detections: int
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[int(prediction["image_id"])].append(prediction)
    limited: list[dict[str, Any]] = []
    for image_id in sorted(grouped):
        limited.extend(
            sorted(grouped[image_id], key=lambda item: float(item.get("score", 0.0)), reverse=True)[
                :max_detections
            ]
        )
    return limited


def _match_counts(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    confidence: float,
    match_iou: float,
) -> tuple[int, int, int, dict[int, tuple[int, int, int]]]:
    gt_grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    pred_grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth:
        if not annotation.get("iscrowd", 0):
            gt_grouped[(annotation["image_id"], annotation["category_id"])].append(annotation)
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= confidence:
            pred_grouped[(prediction["image_id"], prediction["category_id"])].append(prediction)
    category_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    for key in gt_grouped.keys() | pred_grouped.keys():
        gt_items = gt_grouped.get(key, [])
        pred_items = sorted(pred_grouped.get(key, []), key=lambda item: item["score"], reverse=True)
        used: set[int] = set()
        for prediction in pred_items:
            candidates = [
                (_iou_xywh(prediction["bbox"], target["bbox"]), index)
                for index, target in enumerate(gt_items)
                if index not in used
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= match_iou:
                used.add(best_index)
                category_counts[key[1]][0] += 1
            else:
                category_counts[key[1]][1] += 1
        category_counts[key[1]][2] += len(gt_items) - len(used)
    tp = sum(value[0] for value in category_counts.values())
    fp = sum(value[1] for value in category_counts.values())
    fn = sum(value[2] for value in category_counts.values())
    return tp, fp, fn, {key: tuple(value) for key, value in category_counts.items()}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _coco_stat(value: float) -> float | None:
    value = float(value)
    return value if value >= 0 else None


def _wilson_lower_bound(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = proportion + z**2 / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    )
    return max(0.0, (center - margin) / denominator)


def _select_precision_threshold(
    rows: list[dict[str, Any]],
    target_precision: float,
    minimum_precision_lower_bound: float,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if int(row.get("tp", 0)) > 0
        and float(row.get("precision", 0.0)) >= target_precision
        and float(row.get("precision_wilson_lower_95", 0.0)) >= minimum_precision_lower_bound
    ]
    if not candidates:
        return {
            "status": "NOT_AVAILABLE",
            "target_precision": target_precision,
            "minimum_precision_wilson_lower_95": minimum_precision_lower_bound,
            "confidence": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "precision_wilson_lower_95": None,
            "note": (
                "No non-empty threshold met both the point precision target and the 95% Wilson "
                "lower-bound requirement on validation data."
            ),
        }
    selected = max(
        candidates,
        key=lambda row: (
            float(row["recall"]),
            float(row["precision"]),
            -float(row["confidence"]),
        ),
    )
    return {
        "status": "VALIDATION_DERIVED_PENDING_HUMAN_REVIEW",
        "target_precision": target_precision,
        "minimum_precision_wilson_lower_95": minimum_precision_lower_bound,
        **{
            key: selected[key]
            for key in (
                "confidence",
                "precision",
                "precision_wilson_lower_95",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
            )
        },
        "note": (
            "Use only as a high-confidence pseudo-label candidate threshold. Every proposal and "
            "every empty image still requires human review; freeze before independent test."
        ),
    }


def _confusion_matrix(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    category_ids: list[int],
    confidence: float,
    match_iou: float,
) -> np.ndarray:
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    background = len(category_ids)
    matrix = np.zeros((background + 1, background + 1), dtype=np.int64)
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth:
        if not annotation.get("iscrowd", 0):
            gt_by_image[annotation["image_id"]].append(annotation)
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= confidence:
            pred_by_image[prediction["image_id"]].append(prediction)
    for image_id in gt_by_image.keys() | pred_by_image.keys():
        targets = gt_by_image.get(image_id, [])
        detections = sorted(
            pred_by_image.get(image_id, []), key=lambda item: item["score"], reverse=True
        )
        used: set[int] = set()
        for detection in detections:
            candidates = [
                (_iou_xywh(detection["bbox"], target["bbox"]), index)
                for index, target in enumerate(targets)
                if index not in used
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            predicted_index = category_index.get(detection["category_id"], background)
            if best_iou >= match_iou:
                used.add(best_index)
                true_index = category_index.get(targets[best_index]["category_id"], background)
                matrix[true_index, predicted_index] += 1
            else:
                matrix[background, predicted_index] += 1
        for index, target in enumerate(targets):
            if index not in used:
                true_index = category_index.get(target["category_id"], background)
                matrix[true_index, background] += 1
    return matrix


def evaluate_predictions(
    ground_truth_path: Path,
    predictions_path: Path,
    output_dir: Path,
    confidence: float = 0.25,
    match_iou: float = 0.50,
    pseudo_label_target_precision: float = 0.98,
    pseudo_label_minimum_precision_lcb: float = 0.95,
) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    output_dir.mkdir(parents=True, exist_ok=True)
    gt_document = _json(ground_truth_path, {})
    prediction_document = _json(predictions_path, [])
    predictions = (
        prediction_document.get("predictions", [])
        if isinstance(prediction_document, dict)
        else prediction_document
    )
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a COCO result list or {'predictions': [...]}.")
    operating_max_detections = 100
    operating_predictions = _limit_predictions_per_image(predictions, operating_max_detections)
    coco_gt = COCO(str(ground_truth_path))
    categories = {item["id"]: item["name"] for item in gt_document.get("categories", [])}
    area_counts = {"small": 0, "medium": 0, "large": 0}
    for annotation in gt_document.get("annotations", []):
        area = float(annotation.get("area") or annotation["bbox"][2] * annotation["bbox"][3])
        if area < 32**2:
            area_counts["small"] += 1
        elif area < 96**2:
            area_counts["medium"] += 1
        else:
            area_counts["large"] += 1
    ap_values: dict[str, float | None] = {
        "ap50_95": 0.0,
        "ap50": 0.0,
        "ap75": 0.0,
        "ap_small": 0.0 if area_counts["small"] else None,
        "ap_medium": 0.0 if area_counts["medium"] else None,
        "ap_large": 0.0 if area_counts["large"] else None,
        "ar1": 0.0,
        "ar10": 0.0,
        "ar100": 0.0,
        "ar_small": 0.0 if area_counts["small"] else None,
        "ar_medium": 0.0 if area_counts["medium"] else None,
        "ar_large": 0.0 if area_counts["large"] else None,
    }
    per_category_ap = {category_id: 0.0 for category_id in categories}
    if predictions:
        coco_predictions = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_predictions, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        ap_values = {
            "ap50_95": _coco_stat(evaluator.stats[0]),
            "ap50": _coco_stat(evaluator.stats[1]),
            "ap75": _coco_stat(evaluator.stats[2]),
            "ap_small": _coco_stat(evaluator.stats[3]),
            "ap_medium": _coco_stat(evaluator.stats[4]),
            "ap_large": _coco_stat(evaluator.stats[5]),
            "ar1": _coco_stat(evaluator.stats[6]),
            "ar10": _coco_stat(evaluator.stats[7]),
            "ar100": _coco_stat(evaluator.stats[8]),
            "ar_small": _coco_stat(evaluator.stats[9]),
            "ar_medium": _coco_stat(evaluator.stats[10]),
            "ar_large": _coco_stat(evaluator.stats[11]),
        }
        precision = evaluator.eval["precision"]
        for category_index, category_id in enumerate(evaluator.params.catIds):
            values = precision[:, :, category_index, 0, -1]
            valid = values[values > -1]
            per_category_ap[category_id] = float(valid.mean()) if valid.size else 0.0
    tp, fp, fn, category_counts = _match_counts(
        gt_document.get("annotations", []), operating_predictions, confidence, match_iou
    )
    precision_value = _ratio(tp, tp + fp)
    recall_value = _ratio(tp, tp + fn)
    f1_value = _ratio(2 * tp, 2 * tp + fp + fn)
    threshold_rows: list[dict[str, Any]] = []
    per_class_threshold_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for threshold in np.linspace(0, 1, 101):
        threshold_tp, threshold_fp, threshold_fn, threshold_category_counts = _match_counts(
            gt_document.get("annotations", []), operating_predictions, float(threshold), match_iou
        )
        threshold_row = {
            "confidence": float(threshold),
            "precision": _ratio(threshold_tp, threshold_tp + threshold_fp),
            "recall": _ratio(threshold_tp, threshold_tp + threshold_fn),
            "f1": _ratio(2 * threshold_tp, 2 * threshold_tp + threshold_fp + threshold_fn),
            "tp": threshold_tp,
            "fp": threshold_fp,
            "fn": threshold_fn,
            "precision_wilson_lower_95": _wilson_lower_bound(
                threshold_tp, threshold_tp + threshold_fp
            ),
        }
        threshold_rows.append(threshold_row)
        for category_id in categories:
            class_tp, class_fp, class_fn = threshold_category_counts.get(category_id, (0, 0, 0))
            per_class_threshold_rows[category_id].append(
                {
                    "confidence": float(threshold),
                    "precision": _ratio(class_tp, class_tp + class_fp),
                    "recall": _ratio(class_tp, class_tp + class_fn),
                    "f1": _ratio(2 * class_tp, 2 * class_tp + class_fp + class_fn),
                    "tp": class_tp,
                    "fp": class_fp,
                    "fn": class_fn,
                    "precision_wilson_lower_95": _wilson_lower_bound(
                        class_tp, class_tp + class_fp
                    ),
                }
            )
    best_threshold = max(threshold_rows, key=lambda row: row["f1"])
    pseudo_label_calibration = _select_precision_threshold(
        threshold_rows,
        pseudo_label_target_precision,
        pseudo_label_minimum_precision_lcb,
    )
    pseudo_label_class_rows = []
    for category_id, category_name in categories.items():
        calibration = _select_precision_threshold(
            per_class_threshold_rows[category_id],
            pseudo_label_target_precision,
            pseudo_label_minimum_precision_lcb,
        )
        pseudo_label_class_rows.append(
            {"category_id": category_id, "category_name": category_name, **calibration}
        )
    per_class_rows = []
    for category_id, category_name in categories.items():
        class_tp, class_fp, class_fn = category_counts.get(category_id, (0, 0, 0))
        per_class_rows.append(
            {
                "category_id": category_id,
                "category_name": category_name,
                "ap50_95": per_category_ap.get(category_id, 0.0),
                "tp": class_tp,
                "fp": class_fp,
                "fn": class_fn,
                "precision": _ratio(class_tp, class_tp + class_fp),
                "recall": _ratio(class_tp, class_tp + class_fn),
                "f1": _ratio(2 * class_tp, 2 * class_tp + class_fp + class_fn),
            }
        )
    ordered_category_ids = sorted(categories)
    confusion = _confusion_matrix(
        gt_document.get("annotations", []),
        operating_predictions,
        ordered_category_ids,
        confidence,
        match_iou,
    )
    confusion_labels = [categories[category_id] for category_id in ordered_category_ids] + [
        "background"
    ]
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized_confusion = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(confusion, dtype=float),
        where=row_sums != 0,
    )
    result = {
        "schema_version": 2,
        "evaluation_set": "validation",
        "warning": "This is validation data, not an independent conveyor-camera test set.",
        "ground_truth": {
            "path": portable_path(ground_truth_path),
            "sha256": sha256_file(ground_truth_path),
            "images": len(gt_document.get("images", [])),
            "annotations": len(gt_document.get("annotations", [])),
        },
        "predictions": {
            "path": portable_path(predictions_path),
            "sha256": sha256_file(predictions_path),
            "count_raw": len(predictions),
            "count_after_operating_max_dets": len(operating_predictions),
        },
        "protocol": {
            "coco_ap_max_dets": 100,
            "operating_matcher": "score_sorted_class_aware_greedy_one_to_one",
            "operating_max_dets_per_image": operating_max_detections,
            "operating_confidence": confidence,
            "operating_match_iou": match_iou,
            "pseudo_label_target_precision": pseudo_label_target_precision,
            "pseudo_label_minimum_precision_wilson_lower_95": pseudo_label_minimum_precision_lcb,
        },
        "metrics": {
            **ap_values,
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1_value,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "best_f1": best_threshold["f1"],
            "best_f1_confidence": best_threshold["confidence"],
        },
        "per_class": per_class_rows,
        "area_ground_truth_counts": area_counts,
        "pseudo_label_calibration": pseudo_label_calibration,
        "pseudo_label_calibration_by_class": pseudo_label_class_rows,
    }
    write_json(output_dir / "final_metrics.json", result)
    _write_csv(output_dir / "per_class_metrics.csv", list(per_class_rows[0]) if per_class_rows else [], per_class_rows)
    _write_csv(output_dir / "confidence_curve.csv", list(threshold_rows[0]), threshold_rows)
    _write_csv(
        output_dir / "autolabel_thresholds.csv",
        list(pseudo_label_class_rows[0]) if pseudo_label_class_rows else [],
        pseudo_label_class_rows,
    )
    _write_confusion_csv(output_dir / "confusion_counts.csv", confusion_labels, confusion)
    _write_confusion_csv(
        output_dir / "confusion_normalized.csv", confusion_labels, normalized_confusion
    )
    _plot_confidence_curve(threshold_rows, output_dir / "confidence_curve.png")
    _plot_per_class(per_class_rows, output_dir / "per_class_ap.png")
    _plot_confusion(
        confusion_labels, confusion, output_dir / "confusion_counts.png", "Confusion counts"
    )
    _plot_confusion(
        confusion_labels,
        normalized_confusion,
        output_dir / "confusion_normalized.png",
        "Confusion matrix (row-normalized)",
    )
    print_evaluation(result)
    return result


def print_evaluation(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    calibration = result.get("pseudo_label_calibration", {})
    print("\nCOMMON COCO EVALUATION (VALIDATION SET)")
    print("=" * 92)
    print(f"AP50-95 : {metrics['ap50_95']:.4f} ({metrics['ap50_95'] * 100:.2f}%)")
    print(f"AP50    : {metrics['ap50']:.4f} ({metrics['ap50'] * 100:.2f}%)")
    print(
        "AP75/S/M/L: "
        f"{_format(metrics.get('ap75'))} / {_format(metrics.get('ap_small'))} / "
        f"{_format(metrics.get('ap_medium'))} / {_format(metrics.get('ap_large'))}"
    )
    print(
        "AR1/10/100/S: "
        f"{_format(metrics.get('ar1'))} / {_format(metrics.get('ar10'))} / "
        f"{_format(metrics.get('ar100'))} / {_format(metrics.get('ar_small'))}"
    )
    print(f"P/R/F1  : {metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1']:.4f}")
    print(f"TP/FP/FN: {metrics['tp']} / {metrics['fp']} / {metrics['fn']}")
    print(
        f"best F1 : {metrics['best_f1']:.4f} at confidence={metrics['best_f1_confidence']:.2f}"
    )
    if calibration.get("confidence") is None:
        print(
            f"pseudo-label threshold: NOT AVAILABLE at target precision "
            f"{calibration.get('target_precision', 0.98):.2f} and 95% lower bound "
            f"{calibration.get('minimum_precision_wilson_lower_95', 0.95):.2f}"
        )
    else:
        print(
            f"pseudo-label candidate threshold: confidence={calibration['confidence']:.2f}, "
            f"P={calibration['precision']:.4f}, P_LCB95={calibration['precision_wilson_lower_95']:.4f}, "
            f"R={calibration['recall']:.4f}; HUMAN REVIEW REQUIRED"
        )
    print("NOTICE  : validation result; independent conveyor-camera test is still required")


def _plot_confidence_curve(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    confidence = [row["confidence"] for row in rows]
    for metric in ("precision", "recall", "f1"):
        axis.plot(confidence, [row[metric] for row in rows], label=metric.upper(), linewidth=2)
    axis.set(xlabel="Confidence threshold", ylabel="Score (0..1)", ylim=(0, 1.02))
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_per_class(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(max(8, len(rows) * 0.55), 5.5), constrained_layout=True)
    axis.bar([row["category_name"] for row in rows], [row["ap50_95"] for row in rows])
    axis.set(ylabel="AP50-95 (0..1)", ylim=(0, 1.02))
    axis.tick_params(axis="x", rotation=45)
    axis.grid(True, axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_confusion_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *labels])
        for label, row in zip(labels, matrix.tolist(), strict=True):
            writer.writerow([label, *row])


def _plot_confusion(labels: list[str], matrix: np.ndarray, output: Path, title: str) -> None:
    size = max(6.0, len(labels) * 0.6)
    fig, axis = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    axis.tick_params(axis="x", rotation=45)
    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = f"{value:.2f}" if np.issubdtype(matrix.dtype, np.floating) else str(value)
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def evaluate_main(argv: list[str] | None = None) -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description="Evaluate any detector with one common COCO evaluator")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--pseudo-label-target-precision", type=float, default=0.98)
    parser.add_argument("--pseudo-label-minimum-precision-lcb", type=float, default=0.95)
    args = parser.parse_args(argv)
    evaluate_predictions(
        args.ground_truth.resolve(),
        args.predictions.resolve(),
        args.output_dir.resolve(),
        args.confidence,
        args.match_iou,
        args.pseudo_label_target_precision,
        args.pseudo_label_minimum_precision_lcb,
    )


def _read_epoch_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row in rows:
        converted.append(
            {
                key: (float(value) if value not in (None, "") else None)
                for key, value in row.items()
            }
        )
    return converted


def _load_run(run_dir: Path) -> dict[str, Any]:
    metadata = _json(run_dir / "run_manifest.json", _json(run_dir / "metadata.json", {})) or {}
    final_document = _json(run_dir / "final_metrics.json", {}) or {}
    metrics = final_document.get("metrics", final_document.get("metrics_common", {})) or {}
    epochs = _read_epoch_rows(run_dir / "epoch_metrics.csv")
    if epochs:
        latest = epochs[-1]
        metrics.setdefault("ap50", latest.get("map50"))
        metrics.setdefault("ap50_95", latest.get("map50_95"))
        metrics.setdefault("precision", latest.get("precision"))
        metrics.setdefault("recall", latest.get("recall"))
    latency = _json(run_dir / "latency.json", {}) or {}
    gpu = _json(run_dir / "gpu_summary.json", {}) or {}
    extra_rows = _read_jsonl(run_dir / "epoch_metrics_extra.jsonl")
    epoch_peak = epochs[-1].get("gpu_peak_allocated_mib") if epochs else None
    train_peak = epochs[-1].get("train_peak_allocated_mib") if epochs else None
    if extra_rows:
        epoch_peak = epoch_peak or extra_rows[-1].get("epoch_peak_allocated_mib")
        train_peak = train_peak or extra_rows[-1].get("train_peak_allocated_mib")
    checkpoint_record = metadata.get("best_checkpoint", {}) or {}
    checkpoint_text = checkpoint_record.get("path")
    checkpoint_path = Path(checkpoint_text) if checkpoint_text else None
    if checkpoint_path is not None and not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint_exists = bool(checkpoint_path and checkpoint_path.exists())
    checkpoint_hash_matches = bool(
        checkpoint_exists
        and checkpoint_record.get("sha256")
        and sha256_file(checkpoint_path) == checkpoint_record.get("sha256")
    )
    return {
        "run_dir": run_dir,
        "run_id": metadata.get("run_id", run_dir.name),
        "model": metadata.get("model", metadata.get("framework", run_dir.name)),
        "metadata": metadata,
        "metrics": metrics,
        "latency": latency,
        "gpu": gpu,
        "train_peak_allocated_mib": train_peak or epoch_peak,
        "epochs": epochs,
        "evidence_status": {
            "epoch_metrics_exists": (run_dir / "epoch_metrics.csv").exists(),
            "final_metrics_exists": (run_dir / "final_metrics.json").exists(),
            "latency_exists": (run_dir / "latency.json").exists(),
            "gpu_summary_exists": (run_dir / "gpu_summary.json").exists(),
            "checkpoint_exists": checkpoint_exists,
            "checkpoint_hash_matches": checkpoint_hash_matches,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compare_runs(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    provenance_attestation: Path | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [_load_run(path.resolve()) for path in run_dirs]
    run_ids = [str(run["run_id"]) for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate run_id values are not allowed in a comparison")
    expected_protocol = _comparison_protocol_document(runs)
    compatibility = _protocol_compatibility(runs, expected_protocol)
    provenance = verify_run_provenance(
        runs,
        repository=project_root(),
        attestation_path=provenance_attestation.resolve() if provenance_attestation else None,
    )
    compatibility["run_provenance"] = provenance
    if provenance["status"] != "PASS":
        compatibility["release_ready"] = False
        compatibility["release_blockers"].append(
            {"field": "run_provenance", "blockers": provenance["blockers"]}
        )
    rows = []
    for run in runs:
        metadata = run["metadata"]
        metrics = run["metrics"]
        latency = run["latency"]
        gpu = run["gpu"]
        rows.append(
            {
                "run_id": run["run_id"],
                "model": run["model"],
                "status": metadata.get("status", "unknown"),
                "ap50_95": metrics.get("ap50_95"),
                "ap50": metrics.get("ap50"),
                "ap75": metrics.get("ap75"),
                "ap_small": metrics.get("ap_small"),
                "ap_medium": metrics.get("ap_medium"),
                "ap_large": metrics.get("ap_large"),
                "ar100": metrics.get("ar100"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "tp": metrics.get("tp"),
                "fp": metrics.get("fp"),
                "fn": metrics.get("fn"),
                "best_f1": metrics.get("best_f1"),
                "best_f1_confidence": metrics.get("best_f1_confidence"),
                "params": metadata.get("model_details", {}).get("parameters"),
                "checkpoint_mib": metadata.get("best_checkpoint", {}).get("mib"),
                "latency_p50_ms": latency.get("e2e_p50_ms"),
                "latency_p95_ms": latency.get("e2e_p95_ms"),
                "fps": latency.get("sustained_fps"),
                "peak_gpu_memory_mib": run.get("train_peak_allocated_mib"),
                "system_peak_gpu_memory_mib": gpu.get("peak_memory_used_mib"),
                "train_elapsed_s": run["epochs"][-1].get("elapsed_s") if run["epochs"] else None,
                "seed": metadata.get("protocol", {}).get("seed"),
                "dataset_sha256": metadata.get("dataset", {}).get("val_annotation_sha256"),
                "protocol_sha256": metadata.get("protocol_config", {}).get("sha256"),
            }
        )
    aggregate_rows = _aggregate_rows(rows)
    fieldnames = list(rows[0]) if rows else []
    comparison_csv = output_dir / "comparison.csv"
    if fieldnames:
        _write_csv(comparison_csv, fieldnames, rows)
    write_json(output_dir / "comparison.json", rows)
    if aggregate_rows:
        _write_csv(output_dir / "aggregate_comparison.csv", list(aggregate_rows[0]), aggregate_rows)
    write_json(output_dir / "aggregate_comparison.json", aggregate_rows)
    write_json(output_dir / "protocol_compatibility.json", compatibility)
    write_json(output_dir / "run_provenance.json", provenance)
    if provenance_attestation is not None:
        publish_evidence_file(
            provenance_attestation.resolve(),
            output_dir / "run_provenance_attestation.json",
            project_root=project_root(),
        )
    terminal_text = _comparison_terminal_text(rows, compatibility, aggregate_rows)
    terminal_path = output_dir / "comparison_terminal.txt"
    terminal_path.write_text(terminal_text, encoding="utf-8", newline="\n")
    print(terminal_text, end="")
    comparison_source_note = (
        f"SOURCE: comparison.csv | SHA256: {sha256_file(comparison_csv)}"
        if comparison_csv.exists()
        else "SOURCE: comparison.json"
    )
    epoch_sources = [run["run_dir"] / "epoch_metrics.csv" for run in runs]
    epoch_source_note = "SOURCE: " + ", ".join(
        f"{path.parent.name}/epoch_metrics.csv:{sha256_file(path)[:12]}"
        for path in epoch_sources
        if path.exists()
    )
    _plot_training_curves(
        runs,
        output_dir / "training_curves.png",
        compatibility["comparable"],
        epoch_source_note,
    )
    _plot_dashboard(
        rows,
        output_dir / "comparison_dashboard.png",
        compatibility["comparable"],
        comparison_source_note,
    )
    _plot_terminal_snapshot(
        terminal_text,
        output_dir / "terminal_summary.png",
        f"SOURCE: comparison_terminal.txt | SHA256: {sha256_file(terminal_path)}",
    )
    snapshot_protocol = runs[0]["run_dir"] / "protocol_snapshot.yaml" if runs else None
    recorded_protocol = runs[0]["metadata"].get("protocol_config", {}).get("path") if runs else None
    if snapshot_protocol is not None and snapshot_protocol.exists():
        protocol_path = snapshot_protocol
    elif recorded_protocol:
        protocol_path = Path(recorded_protocol)
    else:
        protocol_path = project_root() / "configs" / "experiments" / "baseline_v1.yaml"
    if protocol_path.exists():
        write_protocol_artifacts(protocol_path, output_dir, print_terminal=False)
    _write_comparison_markdown(output_dir / "experiment_report.md", rows, compatibility, aggregate_rows)
    _bundle_run_evidence(output_dir, runs)
    _write_evidence_manifest(output_dir)
    print(f"\nComparison artifacts: {output_dir.resolve()}")
    return rows


def _write_comparison_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    compatibility: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    verdict = "PASS" if compatibility.get("comparable") else "FAIL / NOT COMPARABLE"
    if len(rows) < 2:
        verdict = "single run"
    lines = [
        "# MCU detector 실험 결과",
        "",
        f"- protocol 판정: **{verdict}**",
        f"- 정식 release 판정: **{'READY' if compatibility.get('release_ready') else 'BLOCKED'}**",
        "- AP/AR 출처: 두 framework prediction을 동일 `pycocotools==2.0.11` COCOeval로 재계산",
        "- 운영점 P/R/F1 출처: 공통 score-sorted class-aware greedy 1:1 matcher",
        "- 범위: validation 결과이며 독립적인 실제 컨베이어-camera test 결과가 아님",
        "- 비교 성격: framework-native recipe의 실사용 system benchmark; 순수 architecture ablation이 아님",
        "",
        "## 공통 평가표",
        "",
        "| 모델 | AP50-95 | AP50 | AP75 | APsmall | AR100 | P | R | F1 | TP/FP/FN | p50/p95 ms | FPS | VRAM MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    _format(row.get("ap50_95")),
                    _format(row.get("ap50")),
                    _format(row.get("ap75")),
                    _format(row.get("ap_small")),
                    _format(row.get("ar100")),
                    _format(row.get("precision")),
                    _format(row.get("recall")),
                    _format(row.get("f1")),
                    f"{_format(row.get('tp'), 0)}/{_format(row.get('fp'), 0)}/{_format(row.get('fn'), 0)}",
                    f"{_format(row.get('latency_p50_ms'), 2)}/{_format(row.get('latency_p95_ms'), 2)}",
                    _format(row.get("fps"), 2),
                    _format(row.get("peak_gpu_memory_mib"), 1),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 해석 규칙",
            "",
            "- 주 정확도 지표는 AP50-95이며, 소형 칩에서는 APsmall과 resize 후 box pixel 분포를 함께 봅니다.",
            "- confidence=0.25의 P/R/F1은 고정 보고점이지 최종 배포 threshold가 아닙니다.",
            "- YOLO11 `box/cls/dfl`과 YOLOX `iou/conf/cls/l1` loss는 정의가 달라 절대값을 직접 비교하지 않습니다.",
            "- YOLO11의 gradient accumulation과 YOLOX의 batch별 optimizer step이 달라 optimizer dynamics는 동일하지 않습니다.",
            "- 모델 차이가 seed 표준편차와 비슷하면 n=3으로 우열을 확정하지 않고 반복 수를 늘립니다.",
            "",
        ]
    )
    if compatibility.get("critical_mismatches"):
        lines.extend(["## 비교 불가 항목", ""])
        for mismatch in compatibility["critical_mismatches"]:
            lines.append(f"- `{mismatch['field']}`: `{json.dumps(mismatch['values'], ensure_ascii=False)}`")
        lines.append("")
    if compatibility.get("release_blockers"):
        lines.extend(["## 정식 release 차단 항목", ""])
        for blocker in compatibility["release_blockers"]:
            lines.append(f"- `{blocker['field']}`: `{json.dumps(blocker, ensure_ascii=False)}`")
        lines.append("")
    if aggregate_rows:
        lines.extend(
            [
                "## Seed 집계",
                "",
                "| 모델 | n | AP50-95 mean ± sample SD | p50 latency mean ± sample SD (ms) |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in aggregate_rows:
            lines.append(
                f"| {row['model']} | {row['runs']} | {_format(row.get('ap50_95_mean'))} ± "
                f"{_format(row.get('ap50_95_std'))} | {_format(row.get('latency_p50_ms_mean'), 2)} ± "
                f"{_format(row.get('latency_p50_ms_std'), 2)} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 자동 생성 증빙",
            "",
            "- `comparison.csv/json`: 표의 원본 수치",
            "- `aggregate_comparison.csv/json`: seed 평균·sample SD",
            "- `comparison_terminal.txt`: 실제 CLI가 출력한 것과 동일한 터미널 표 원문",
            "- `comparison_dashboard.png`, `training_curves.png`: 로그/CSV를 matplotlib로 그린 비생성형 그래프",
            "- `terminal_summary.png`: `comparison_terminal.txt`를 그대로 코드 렌더링한 이미지이며 화면 캡처가 아님",
            "- `evidence_manifest.json`: 원본 로그·CSV와 각 이미지의 SHA-256 및 `generative_ai_used=false` 기록",
            "- `protocol_rationale.csv/png`, `experiment_methodology.md`: 수치 선정 이유와 출처",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_protocol_document(runs: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[Path] = []
    if runs:
        candidates.append(runs[0]["run_dir"] / "protocol_snapshot.yaml")
        recorded = runs[0]["metadata"].get("protocol_config", {}).get("path")
        if recorded:
            candidates.append(Path(recorded))
    candidates.append(project_root() / "configs" / "experiments" / "baseline_v1.yaml")
    for candidate in candidates:
        if candidate.exists():
            document = load_protocol(candidate)
            document["_loaded_source_sha256"] = sha256_file(candidate)
            document["_loaded_source_path"] = str(candidate.resolve())
            return document
    return {}


def _normalize_model_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _protocol_compatibility(
    runs: list[dict[str, Any]],
    expected_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_protocol = expected_protocol or {}
    common = expected_protocol.get("common", {})
    rules = expected_protocol.get("comparison_rules", {})
    required_dataset_evidence = [
        str(field) for field in rules.get("required_dataset_evidence", [])
    ]
    expected_dataset_evidence = expected_protocol.get("dataset", {}).get("evidence", {})

    def value(run: dict[str, Any], field: str) -> Any:
        protocol = run["metadata"].get("protocol", {})
        dataset = run["metadata"].get("dataset", {})
        if field in {"train_annotation_sha256", "val_annotation_sha256", *required_dataset_evidence}:
            return run["metadata"].get("dataset", {}).get(field)
        if field == "protocol_config_sha256":
            return run["metadata"].get("protocol_config", {}).get("sha256")
        if field == "fraction":
            return protocol.get(field, 1.0)
        if field == "multiscale_range":
            return protocol.get(field, 0)
        return protocol.get(field)

    critical_fields = [
        "train_annotation_sha256",
        "val_annotation_sha256",
        "epochs",
        "batch",
        "imgsz",
        "workers",
        "amp",
        "fraction",
        "multiscale_range",
        "prediction_floor",
        "nms_iou",
        "class_agnostic_nms",
        "common_operating_confidence",
        "common_match_iou",
        "protocol_config_sha256",
        *required_dataset_evidence,
    ]
    mismatches = []
    for field in critical_fields:
        values = {run["run_id"]: value(run, field) for run in runs}
        present = [json.dumps(item, sort_keys=True) for item in values.values()]
        if len(set(present)) > 1:
            mismatches.append({"field": field, "values": values})
    seeds_by_model: dict[str, set[Any]] = defaultdict(set)
    for run in runs:
        seeds_by_model[str(run["model"])].add(value(run, "seed"))
    seed_sets = {model: sorted(seeds, key=str) for model, seeds in seeds_by_model.items()}
    if len({json.dumps(seeds, sort_keys=True) for seeds in seed_sets.values()}) > 1:
        mismatches.append({"field": "seed_set", "values": seed_sets})
    expected_models = [str(item) for item in rules.get("required_models", [])]
    expected_seeds = list(common.get("seeds", []))
    release_blockers: list[dict[str, Any]] = []
    if mismatches:
        release_blockers.append(
            {"field": "protocol_comparability", "reason": "critical settings differ between runs"}
        )
    loaded_protocol_sha256 = expected_protocol.get("_loaded_source_sha256")
    if loaded_protocol_sha256:
        wrong_protocol_sources = {
            run["run_id"]: run["metadata"].get("protocol_config", {}).get("sha256")
            for run in runs
            if run["metadata"].get("protocol_config", {}).get("sha256")
            != loaded_protocol_sha256
        }
        if wrong_protocol_sources:
            release_blockers.append(
                {
                    "field": "protocol_snapshot_integrity",
                    "expected": loaded_protocol_sha256,
                    "actual": wrong_protocol_sources,
                }
            )
    actual_model_map = {_normalize_model_name(model): model for model in seed_sets}
    expected_model_set = {_normalize_model_name(model) for model in expected_models}
    if set(actual_model_map) != expected_model_set:
        release_blockers.append(
            {
                "field": "model_set",
                "expected": expected_models,
                "actual": sorted(seed_sets),
            }
        )
    expected_seed_set = set(expected_seeds)
    for normalized_model in sorted(expected_model_set):
        display_model = actual_model_map.get(normalized_model)
        actual_seeds = set(seed_sets.get(display_model, [])) if display_model else set()
        if actual_seeds != expected_seed_set:
            release_blockers.append(
                {
                    "field": f"seed_set:{display_model or normalized_model}",
                    "expected": sorted(expected_seed_set),
                    "actual": sorted(actual_seeds),
                }
            )
    expected_pairs = {(model, seed) for model in expected_model_set for seed in expected_seed_set}
    actual_pairs = [(_normalize_model_name(run["model"]), value(run, "seed")) for run in runs]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        release_blockers.append(
            {
                "field": "complete_model_seed_matrix",
                "expected_runs": len(expected_pairs),
                "actual_runs": len(actual_pairs),
            }
        )
    if len(set(actual_pairs)) != len(actual_pairs):
        release_blockers.append(
            {"field": "duplicate_model_seed_pair", "reason": "each model/seed pair must occur once"}
        )
    incomplete = [
        run["run_id"]
        for run in runs
        if run["metadata"].get("status") != "complete"
        or run["metadata"].get("stage") == "smoke_not_comparable"
    ]
    if incomplete:
        release_blockers.append(
            {"field": "complete_non_smoke_runs", "actual": sorted(incomplete)}
        )
    expected_values = {
        "epochs": common.get("epochs"),
        "batch": common.get("batch_size"),
        "imgsz": common.get("image_size"),
        "workers": common.get("workers"),
        "amp": common.get("amp"),
        "fraction": 1.0,
        "prediction_floor": common.get("prediction_floor"),
        "nms_iou": common.get("nms_iou"),
        "class_agnostic_nms": common.get("class_agnostic_nms"),
        "common_operating_confidence": common.get("operating_confidence"),
        "common_match_iou": common.get("operating_match_iou"),
    }
    for field, expected in expected_values.items():
        wrong = {
            run["run_id"]: value(run, field)
            for run in runs
            if value(run, field) != expected
        }
        if expected is not None and wrong:
            release_blockers.append(
                {"field": f"expected:{field}", "expected": expected, "actual": wrong}
            )
    for field in required_dataset_evidence:
        missing = [
            run["run_id"]
            for run in runs
            if not run["metadata"].get("dataset", {}).get(field)
        ]
        if missing:
            release_blockers.append(
                {"field": f"dataset_evidence:{field}", "missing_runs": sorted(missing)}
            )
        expected = expected_dataset_evidence.get(field)
        wrong = {
            run["run_id"]: run["metadata"].get("dataset", {}).get(field)
            for run in runs
            if expected is not None
            and run["metadata"].get("dataset", {}).get(field) != expected
        }
        if wrong:
            release_blockers.append(
                {
                    "field": f"expected_dataset_evidence:{field}",
                    "expected": expected,
                    "actual": wrong,
                }
            )
    expected_epoch_count = common.get("epochs")
    wrong_epoch_counts = {
        run["run_id"]: len(run.get("epochs", []))
        for run in runs
        if expected_epoch_count is not None and len(run.get("epochs", [])) != expected_epoch_count
    }
    if wrong_epoch_counts:
        release_blockers.append(
            {
                "field": "completed_epoch_rows",
                "expected": expected_epoch_count,
                "actual": wrong_epoch_counts,
            }
        )
    required_evidence_flags = [
        "epoch_metrics_exists",
        "final_metrics_exists",
        "latency_exists",
        "gpu_summary_exists",
        "checkpoint_exists",
        "checkpoint_hash_matches",
    ]
    for flag in required_evidence_flags:
        missing = [
            run["run_id"] for run in runs if not run.get("evidence_status", {}).get(flag)
        ]
        if missing:
            release_blockers.append(
                {"field": f"run_evidence:{flag}", "missing_runs": sorted(missing)}
            )
    required_metrics = [
        "ap50_95",
        "ap50",
        "ap75",
        "ar100",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
    ]
    for metric in required_metrics:
        invalid = [
            run["run_id"]
            for run in runs
            if not isinstance(run.get("metrics", {}).get(metric), (int, float))
            or not math.isfinite(float(run["metrics"][metric]))
        ]
        if invalid:
            release_blockers.append(
                {"field": f"final_metric:{metric}", "invalid_runs": sorted(invalid)}
            )
    required_latency = ["e2e_p50_ms", "e2e_p95_ms", "sustained_fps"]
    for metric in required_latency:
        invalid = [
            run["run_id"]
            for run in runs
            if not isinstance(run.get("latency", {}).get(metric), (int, float))
            or not math.isfinite(float(run["latency"][metric]))
        ]
        if invalid:
            release_blockers.append(
                {"field": f"latency_metric:{metric}", "invalid_runs": sorted(invalid)}
            )
    invalid_gpu = [
        run["run_id"]
        for run in runs
        if not isinstance(run.get("gpu", {}).get("peak_memory_used_mib"), (int, float))
        or not math.isfinite(float(run["gpu"]["peak_memory_used_mib"]))
    ]
    if invalid_gpu:
        release_blockers.append(
            {"field": "gpu_metric:peak_memory_used_mib", "invalid_runs": sorted(invalid_gpu)}
        )
    return {
        "comparable": len(runs) >= 2 and not mismatches,
        "release_ready": len(runs) >= 2 and not mismatches and not release_blockers,
        "run_count": len(runs),
        "critical_mismatches": mismatches,
        "release_blockers": release_blockers,
        "seed_sets_by_model": seed_sets,
        "release_expectations": {
            "models": expected_models,
            "seeds": expected_seeds,
            "runs": len(expected_models) * len(expected_seeds),
            "dataset_evidence": required_dataset_evidence,
        },
        "note": (
            "Comparable PASS only means the supplied runs share critical settings. Release-ready also "
            "requires the configured model/seed matrix, full epochs, complete non-smoke runs, and dataset "
            "equivalence evidence. "
            "Native optimizer, effective batch, augmentation, and loss definitions intentionally remain "
            "framework-specific; final accuracy must come from common COCO evaluation."
        ),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    metric_names = [
        "ap50_95",
        "ap50",
        "ap75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar100",
        "precision",
        "recall",
        "f1",
        "latency_p50_ms",
        "latency_p95_ms",
        "fps",
        "peak_gpu_memory_mib",
        "train_elapsed_s",
    ]
    aggregate = []
    for model, model_rows in grouped.items():
        value: dict[str, Any] = {"model": model, "runs": len(model_rows)}
        for metric in metric_names:
            samples = [float(row[metric]) for row in model_rows if row.get(metric) is not None]
            value[f"{metric}_mean"] = statistics.fmean(samples) if samples else None
            value[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else None
        aggregate.append(value)
    return aggregate


def _comparison_terminal_text(
    rows: list[dict[str, Any]],
    compatibility: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
) -> str:
    lines = ["", "MODEL COMPARISON — COMMON COCO VALIDATION", "=" * 147]
    header = (
        f"{'MODEL':<22} {'AP50-95':>9} {'AP50':>8} {'AP75':>8} {'APsmall':>8} {'AR100':>8} "
        f"{'P':>8} {'R':>8} {'F1':>8} {'TP':>7} {'FP':>7} {'FN':>7} {'BEST@CONF':>11}"
    )
    lines.extend([header, "-" * len(header)])
    for row in rows:
        lines.append(
            f"{str(row['model']):<22} {_format(row['ap50_95']):>9} {_format(row['ap50']):>8} "
            f"{_format(row['ap75']):>8} {_format(row['ap_small']):>8} {_format(row['ar100']):>8} "
            f"{_format(row['precision']):>8} {_format(row['recall']):>8} {_format(row['f1']):>8} "
            f"{_format(row['tp'], 0):>7} {_format(row['fp'], 0):>7} {_format(row['fn'], 0):>7} "
            f"{_format(row['best_f1_confidence'], 2):>11}"
        )
    lines.extend(["", "DEPLOYMENT / RESOURCE MEASUREMENTS", "-" * 118])
    deployment_header = (
        f"{'MODEL':<22} {'PARAMS':>12} {'CKPT MiB':>10} {'P50 ms':>10} {'P95 ms':>10} "
        f"{'FPS':>9} {'TRAIN VRAM':>12} {'TRAIN min':>11}"
    )
    lines.extend([deployment_header, "-" * len(deployment_header)])
    for row in rows:
        elapsed_minutes = (
            float(row["train_elapsed_s"]) / 60 if row.get("train_elapsed_s") is not None else None
        )
        lines.append(
            f"{str(row['model']):<22} {_format(row['params'], 0):>12} "
            f"{_format(row['checkpoint_mib'], 2):>10} {_format(row['latency_p50_ms'], 2):>10} "
            f"{_format(row['latency_p95_ms'], 2):>10} {_format(row['fps'], 2):>9} "
            f"{_format(row['peak_gpu_memory_mib'], 1):>12} {_format(elapsed_minutes, 2):>11}"
        )
    lines.append(
        "NOTE: YOLO11 and YOLOX native loss values have different definitions; compare trends only."
    )
    lines.append(
        "NOTE: batch=8 is a shared micro-batch. Native gradient accumulation/optimizer dynamics differ."
    )
    if len(rows) < 2:
        lines.append("PROTOCOL: single-run report; comparison validity is not applicable.")
    elif compatibility["comparable"]:
        lines.append("PROTOCOL: PASS - critical data and training settings match.")
    else:
        fields = ", ".join(item["field"] for item in compatibility["critical_mismatches"])
        lines.append(f"PROTOCOL: FAIL - NOT COMPARABLE. Mismatched fields: {fields}")
    if compatibility.get("release_ready"):
        lines.append("RELEASE: READY - complete model/seed matrix and dataset evidence passed.")
    else:
        blockers = ", ".join(item["field"] for item in compatibility.get("release_blockers", []))
        lines.append(f"RELEASE: BLOCKED - {blockers or 'formal release requirements are incomplete'}")
    if any(row["runs"] > 1 for row in aggregate_rows):
        lines.extend(["", "SEED AGGREGATE (mean +/- sample standard deviation)", "-" * 76])
        for row in aggregate_rows:
            lines.append(
                f"{row['model']:<24} n={row['runs']:<3} "
                f"AP50-95={_format(row['ap50_95_mean'])} +/- {_format(row['ap50_95_std'])} "
                f"P50ms={_format(row['latency_p50_ms_mean'], 2)} +/- "
                f"{_format(row['latency_p50_ms_std'], 2)}"
            )
    return "\n".join(lines) + "\n"


def _plot_training_curves(
    runs: list[dict[str, Any]],
    output: Path,
    comparable: bool = True,
    source_note: str = "",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for run in runs:
        epochs = run["epochs"]
        if not epochs:
            continue
        x = [row["epoch"] for row in epochs]
        for key, axis, title in [
            ("map50_95", axes[0, 0], "Validation AP50-95 (native)"),
            ("map50", axes[0, 1], "Validation AP50 (native)"),
            ("train_total_loss", axes[1, 0], "YOLOX sampled train total loss"),
            ("train_box_loss", axes[1, 1], "YOLO11 train box loss"),
        ]:
            points = [(epoch, row.get(key)) for epoch, row in zip(x, epochs, strict=True) if row.get(key) is not None]
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o" if len(points) < 10 else None,
                    label=run["model"],
                )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(True, alpha=0.25)
    for axis in axes.flat:
        if axis.lines:
            axis.legend()
    prefix = "" if comparable or len(runs) < 2 else "NOT COMPARABLE — "
    fig.suptitle(prefix + "Training curves — native losses are framework-specific")
    if source_note:
        fig.text(0.01, 0.005, source_note + " | RENDERER: matplotlib | GENERATIVE_AI: false", fontsize=7)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.95))
    fig.savefig(
        output,
        dpi=180,
        metadata={"Software": "matplotlib", "Description": source_note + " | generative_ai=false"},
    )
    plt.close(fig)


def _plot_dashboard(
    rows: list[dict[str, Any]],
    output: Path,
    comparable: bool = True,
    source_note: str = "",
) -> None:
    if not rows:
        return
    labels = [str(row["model"]) for row in rows]
    panels = [
        ("ap50_95", "AP50-95", (0, 1)),
        ("ap50", "AP50", (0, 1)),
        ("ap75", "AP75", (0, 1)),
        ("ap_small", "AP small (<32² px)", (0, 1)),
        ("f1", "F1 at fixed operating point", (0, 1)),
        ("latency_p50_ms", "E2E latency p50 (ms)", None),
        ("latency_p95_ms", "E2E latency p95 (ms)", None),
        ("fps", "Sustained FPS", None),
        ("peak_gpu_memory_mib", "Training CUDA peak allocated (MiB)", None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for axis, (key, title, limits) in zip(axes.flat, panels, strict=True):
        raw_values = [row.get(key) for row in rows]
        values = [float(value) if value is not None else 0.0 for value in raw_values]
        axis.bar(labels, values)
        axis.set_title(title)
        if limits:
            axis.set_ylim(*limits)
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
        for index, (value, raw_value) in enumerate(zip(values, raw_values, strict=True)):
            label = f"{value:.3g}" if raw_value is not None else "N/A"
            axis.text(index, value, label, ha="center", va="bottom", fontsize=8)
    prefix = "" if comparable or len(rows) < 2 else "NOT COMPARABLE — "
    fig.suptitle(prefix + "MCU detector common-evaluation dashboard")
    if source_note:
        fig.text(0.01, 0.005, source_note + " | RENDERER: matplotlib | GENERATIVE_AI: false", fontsize=7)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.96))
    fig.savefig(
        output,
        dpi=180,
        metadata={"Software": "matplotlib", "Description": source_note + " | generative_ai=false"},
    )
    plt.close(fig)


def _plot_terminal_snapshot(
    terminal_text: str,
    output: Path,
    source_note: str,
) -> None:
    lines = terminal_text.rstrip().splitlines()
    fig = plt.figure(figsize=(17, max(3.5, len(lines) * 0.32)), facecolor="#111318")
    fig.text(
        0.025,
        0.95,
        "\n".join(lines),
        family="monospace",
        fontsize=11,
        color="#e6edf3",
        va="top",
    )
    fig.text(
        0.025,
        0.02,
        source_note + " | RENDERER: matplotlib (non-generative) | GENERATIVE_AI: false",
        family="monospace",
        fontsize=7,
        color="#8b949e",
        va="bottom",
    )
    plt.axis("off")
    fig.savefig(
        output,
        dpi=180,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        metadata={"Software": "matplotlib", "Description": source_note + " | generative_ai=false"},
    )
    plt.close(fig)


def _bundle_run_evidence(output_dir: Path, runs: list[dict[str, Any]]) -> None:
    filenames = (
        "terminal.log",
        "run_manifest.json",
        "epoch_metrics.csv",
        "epoch_metrics.jsonl",
        "epoch_metrics_extra.jsonl",
        "final_metrics.json",
        "per_class_metrics.csv",
        "latency.json",
        "latency_samples.csv",
        "gpu_summary.json",
    )
    records: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for run in runs:
        run_id = str(run["run_id"])
        directory_name = safe_stem(run_id)
        if directory_name in used_names:
            raise ValueError(f"Duplicate published run evidence key: {directory_name}")
        used_names.add(directory_name)
        for filename in filenames:
            source = run["run_dir"] / filename
            if not source.exists():
                continue
            relative = Path("sources") / directory_name / filename
            destination = output_dir / relative
            publication = publish_evidence_file(
                source, destination, project_root=project_root()
            )
            records.append(
                {
                    "run_id": run_id,
                    "path": relative.as_posix(),
                    **publication,
                }
            )
    write_json(
        output_dir / "sources_manifest.json",
        {
            "schema_version": 1,
            "local_source_paths_included": False,
            "publication_note": (
                "Local project/user paths and raw nvidia-smi process listings are redacted. "
                "Original and published SHA-256 values are retained; numeric metrics are unchanged."
            ),
            "files": records,
        },
    )


def _write_evidence_manifest(output_dir: Path) -> None:
    source_paths = [
        output_dir / "comparison.csv",
        output_dir / "comparison.json",
        output_dir / "comparison_terminal.txt",
        output_dir / "protocol_compatibility.json",
        output_dir / "run_provenance.json",
        output_dir / "run_provenance_attestation.json",
        output_dir / "sources_manifest.json",
    ]
    sources_root = output_dir / "sources"
    if sources_root.exists():
        source_paths.extend(sorted(path for path in sources_root.rglob("*") if path.is_file()))
    artifact_paths = [
        output_dir / "terminal_summary.png",
        output_dir / "comparison_dashboard.png",
        output_dir / "training_curves.png",
        output_dir / "protocol_rationale.png",
    ]

    def record(path: Path) -> dict[str, Any]:
        return {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    write_json(
        output_dir / "evidence_manifest.json",
        {
            "schema_version": 2,
            "decision_policy": "Judgments use terminal logs and numeric CSV/JSON only; PNG files are visualization derivatives.",
            "generative_ai_used_for_images": False,
            "image_renderer": f"matplotlib {matplotlib.__version__}",
            "local_absolute_paths_included": False,
            "source_bundle": "sources_manifest.json",
            "sources": [record(path) for path in source_paths if path.exists()],
            "derived_images": [record(path) for path in artifact_paths if path.exists()],
        },
    )


def compare_main(argv: list[str] | None = None) -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description="Print and plot multiple logged detector runs")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--provenance-attestation",
        type=Path,
        help="Required fail-closed attestation when selected run manifests use multiple Git commits",
    )
    args = parser.parse_args(argv)
    compare_runs(
        args.runs,
        args.output_dir.resolve(),
        provenance_attestation=args.provenance_attestation,
    )


def weights_main(argv: list[str] | None = None) -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description="Print per-tensor checkpoint statistics from a logged run")
    parser.add_argument("--csv", type=Path, required=True, dest="csv_path")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--filter", default="")
    parser.add_argument("--sort", choices=("l2", "numel", "name"), default="l2")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    with args.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.filter:
        rows = [row for row in rows if args.filter.lower() in row["name"].lower()]
    if args.sort == "name":
        rows.sort(key=lambda row: row["name"])
    elif args.sort == "numel":
        rows.sort(key=lambda row: int(row.get("numel") or 0), reverse=True)
    else:
        rows.sort(key=lambda row: float(row.get("l2_norm") or "-inf"), reverse=True)
    selected = rows if args.all else rows[: args.top]
    print(f"\nWEIGHT TENSOR STATISTICS: {args.csv_path.resolve()}")
    print(f"rows={len(rows)}, shown={len(selected)}, sort={args.sort}, filter={args.filter or '-'}")
    print("=" * 146)
    print(
        f"{'NAME':<64} {'SHAPE':<20} {'DTYPE':<9} {'NUMEL':>10} "
        f"{'MEAN':>10} {'STD':>10} {'MIN':>10} {'MAX':>10} {'L2':>10}"
    )
    print("-" * 146)
    for row in selected:
        name = row["name"] if len(row["name"]) <= 64 else "..." + row["name"][-61:]
        print(
            f"{name:<64} {row.get('shape', ''):<20} {row.get('dtype', ''):<9} "
            f"{int(row.get('numel') or 0):>10d} "
            f"{_short_number(row.get('mean')):>10} {_short_number(row.get('std')):>10} "
            f"{_short_number(row.get('min')):>10} {_short_number(row.get('max')):>10} "
            f"{_short_number(row.get('l2_norm')):>10}"
        )


def _short_number(value: Any) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value):.4g}"


if __name__ == "__main__":
    status_main(sys.argv[1:])
