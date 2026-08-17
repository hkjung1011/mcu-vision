from __future__ import annotations

import contextlib
import csv
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from loguru import logger
from pycocotools.cocoeval import COCOeval
from yolox.core import Trainer
from yolox.evaluators import COCOEvaluator
from yolox.evaluators.coco_evaluator import per_class_AP_table, per_class_AR_table
from yolox.utils import is_main_process

from .common import append_jsonl, write_json


RAW_COLUMNS = [
    "epoch",
    "elapsed_s",
    "epoch_seconds",
    "iterations",
    "train_total_loss",
    "train_iou_loss",
    "train_conf_loss",
    "train_cls_loss",
    "train_l1_loss",
    "num_fg_mean",
    "map50_95",
    "map50",
    "ap75",
    "ar100",
    "lr",
    "lr_mean",
    "gpu_peak_allocated_mib",
    "gpu_peak_reserved_mib",
    "best_map50_95",
]


class AuditedCOCOEvaluator(COCOEvaluator):
    """YOLOX evaluator that always uses standard pycocotools on Windows."""

    last_stats: list[float]
    audit_output_dir: Path | None = None

    def evaluate_prediction(self, data_dict, statistics):  # type: ignore[override]
        if not is_main_process():
            return 0, 0, None

        inference_time = float(statistics[0].item())
        nms_time = float(statistics[1].item())
        n_samples = float(statistics[2].item())
        denominator = max(n_samples * self.dataloader.batch_size, 1.0)
        forward_ms = 1000 * inference_time / denominator
        nms_ms = 1000 * nms_time / denominator
        info = (
            f"Average forward time: {forward_ms:.2f} ms, "
            f"Average NMS time: {nms_ms:.2f} ms, "
            f"Average inference time: {forward_ms + nms_ms:.2f} ms\n"
        )
        if self.audit_output_dir is not None:
            write_json(self.audit_output_dir / "predictions.coco.json", data_dict)

        self.last_stats = [0.0] * 12
        if not data_dict:
            return 0, 0, info

        logger.info("Evaluate with standard pycocotools COCOeval (no C++ JIT build).")
        coco_ground_truth = self.dataloader.dataset.coco
        coco_detections = coco_ground_truth.loadRes(data_dict)
        evaluator = COCOeval(coco_ground_truth, coco_detections, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        summary = io.StringIO()
        with contextlib.redirect_stdout(summary):
            evaluator.summarize()
        info += summary.getvalue()
        self.last_stats = [float(value) for value in evaluator.stats.tolist()]
        category_ids = sorted(coco_ground_truth.cats)
        category_names = [coco_ground_truth.cats[category_id]["name"] for category_id in category_ids]
        if self.per_class_AP:
            info += "per class AP:\n" + per_class_AP_table(evaluator, class_names=category_names) + "\n"
        if self.per_class_AR:
            info += "per class AR:\n" + per_class_AR_table(evaluator, class_names=category_names) + "\n"
        if self.audit_output_dir is not None:
            write_json(
                self.audit_output_dir / "native_coco_metrics.json",
                {
                    "ap50_95": self.last_stats[0],
                    "ap50": self.last_stats[1],
                    "ap75": self.last_stats[2],
                    "ap_small": self.last_stats[3],
                    "ap_medium": self.last_stats[4],
                    "ap_large": self.last_stats[5],
                    "ar1": self.last_stats[6],
                    "ar10": self.last_stats[7],
                    "ar100": self.last_stats[8],
                    "ar_small": self.last_stats[9],
                    "ar_medium": self.last_stats[10],
                    "ar_large": self.last_stats[11],
                    "forward_ms": forward_ms,
                    "nms_ms": nms_ms,
                },
            )
        return self.last_stats[0], self.last_stats[1], info


class MetricsTrainer(Trainer):
    """YOLOX trainer with full-precision epoch metrics and terminal JSON lines."""

    def before_train(self) -> None:
        super().before_train()
        self._audit_started = time.perf_counter()
        self._epoch_started = self._audit_started
        self._metric_sums: dict[str, float] = defaultdict(float)
        self._metric_counts: dict[str, int] = defaultdict(int)
        self._epoch_rows: list[dict[str, Any]] = []
        if isinstance(self.evaluator, AuditedCOCOEvaluator):
            self.evaluator.audit_output_dir = Path(self.file_name)
        optimizer_groups = []
        for index, group in enumerate(self.optimizer.param_groups):
            optimizer_groups.append(
                {
                    "index": index,
                    "parameters": sum(parameter.numel() for parameter in group["params"]),
                    "lr": group.get("lr"),
                    "weight_decay": group.get("weight_decay"),
                    "momentum": group.get("momentum"),
                    "nesterov": group.get("nesterov"),
                }
            )
        model = self.model.module if hasattr(self.model, "module") else self.model
        details = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "optimizer": type(self.optimizer).__name__,
            "optimizer_groups": optimizer_groups,
            "resolved_batch_size": self.args.batch_size,
            "steps_per_epoch": self.max_iter,
            "amp": self.amp_training,
            "base_lr_for_batch": self.exp.basic_lr_per_img * self.args.batch_size,
            "minimum_lr": self.exp.basic_lr_per_img
            * self.args.batch_size
            * self.exp.min_lr_ratio,
        }
        print("\nYOLOX RESOLVED TRAINING DETAILS")
        print("=" * 72)
        for key, value in details.items():
            print(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value}")
        manifest_path = Path(self.file_name) / "run_manifest.json"
        manifest = {}
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        manifest["resolved_training"] = details
        write_json(manifest_path, manifest)

    def before_epoch(self) -> None:
        self._metric_sums = defaultdict(float)
        self._metric_counts = defaultdict(int)
        self._epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        super().before_epoch()

    def after_iter(self) -> None:
        for name in ("total_loss", "iou_loss", "conf_loss", "cls_loss", "l1_loss", "num_fg", "lr"):
            if name not in self.meter:
                continue
            value = self.meter[name].latest
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.item()
            self._metric_sums[name] += float(value)
            self._metric_counts[name] += 1
        super().after_iter()

    def after_epoch(self) -> None:
        super().after_epoch()
        epoch_seconds = time.perf_counter() - self._epoch_started
        stats = getattr(self.evaluator, "last_stats", [0.0] * 12)

        def mean(name: str) -> float | None:
            count = self._metric_counts.get(name, 0)
            return self._metric_sums[name] / count if count else None

        latest_lr = self.meter["lr"].latest if "lr" in self.meter else None
        row = {
            "epoch": self.epoch + 1,
            "elapsed_s": time.perf_counter() - self._audit_started,
            "epoch_seconds": epoch_seconds,
            "iterations": self.max_iter,
            "train_total_loss": mean("total_loss"),
            "train_iou_loss": mean("iou_loss"),
            "train_conf_loss": mean("conf_loss"),
            "train_cls_loss": mean("cls_loss"),
            "train_l1_loss": mean("l1_loss"),
            "num_fg_mean": mean("num_fg"),
            "map50_95": stats[0] if len(stats) > 0 else None,
            "map50": stats[1] if len(stats) > 1 else None,
            "ap75": stats[2] if len(stats) > 2 else None,
            "ar100": stats[8] if len(stats) > 8 else None,
            "lr": latest_lr if latest_lr is not None else mean("lr"),
            "lr_mean": mean("lr"),
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "best_map50_95": self.best_ap,
        }
        if torch.is_tensor(row["lr"]):
            row["lr"] = float(row["lr"].item())
        self._epoch_rows.append(row)
        output_dir = Path(self.file_name)
        _write_epoch_csv(output_dir / "epoch_metrics.csv", self._epoch_rows)
        append_jsonl(output_dir / "epoch_metrics.jsonl", row)
        _plot_epochs(self._epoch_rows, output_dir / "plots" / "latest_overview.png")
        _plot_epochs(
            self._epoch_rows,
            output_dir / "plots" / "epochs" / f"epoch_{self.epoch + 1:03d}_overview.png",
        )
        if self.rank == 0 and self.args.logger == "tensorboard":
            self.tblogger.add_scalar("epoch/AP75", row["ap75"], row["epoch"])
            self.tblogger.add_scalar("epoch/AR100", row["ar100"], row["epoch"])
            self.tblogger.add_scalar(
                "epoch/gpu_peak_allocated_mib", row["gpu_peak_allocated_mib"], row["epoch"]
            )
            self.tblogger.flush()
        print(
            f"[EPOCH {row['epoch']:03d}/{self.max_epoch:03d}] "
            f"loss={_f(row['train_total_loss'])} iou={_f(row['train_iou_loss'])} "
            f"conf={_f(row['train_conf_loss'])} cls={_f(row['train_cls_loss'])} | "
            f"AP50={_f(row['map50'])} AP50-95={_f(row['map50_95'])} "
            f"AP75={_f(row['ap75'])} AR100={_f(row['ar100'])} | "
            f"LR={_f(row['lr'], 8)} VRAM_peak={_f(row['gpu_peak_allocated_mib'], 1)} MiB"
        )
        print("EPOCH_METRICS_JSON " + json.dumps(row, ensure_ascii=False, allow_nan=False))


def _f(value: Any, digits: int = 5) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _write_epoch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_epochs(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for key, label in [
        ("train_total_loss", "total"),
        ("train_iou_loss", "IoU"),
        ("train_conf_loss", "confidence"),
        ("train_cls_loss", "class"),
    ]:
        axes[0, 0].plot(epochs, [row[key] for row in rows], marker="o", label=label)
    axes[0, 0].set_title("YOLOX native training losses")
    axes[0, 0].legend()
    for key, label in [("map50", "AP50"), ("map50_95", "AP50-95"), ("ap75", "AP75")]:
        axes[0, 1].plot(epochs, [row[key] for row in rows], marker="o", label=label)
    axes[0, 1].set_title("Validation AP")
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, [row["lr"] for row in rows], marker="o")
    axes[1, 0].set_title("Learning rate")
    axes[1, 1].plot(
        epochs, [row["gpu_peak_allocated_mib"] for row in rows], marker="o", color="tab:red"
    )
    axes[1, 1].set_title("CUDA peak allocated memory (MiB)")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
    fig.suptitle("YOLOX-S fine-tuning — exact values are stored in epoch_metrics.csv")
    fig.savefig(path, dpi=180)
    plt.close(fig)
