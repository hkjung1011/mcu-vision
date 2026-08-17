from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO

from mcu_data.common import append_jsonl, sha256_file, write_json
from mcu_data.methodology import write_protocol_artifacts
from mcu_data.reporting import (
    EPOCH_COLUMNS,
    compare_runs,
    evaluate_predictions,
    normalize_yolo11_results,
)
from mcu_data.runlog import (
    GpuSampler,
    checkpoint_file_record,
    collect_system_environment,
    collect_torch_environment,
    collect_git_state,
    configure_utf8_output,
    print_section,
    state_dict_statistics,
    tee_console,
    utc_now_precise,
    write_pip_freeze,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "experiments" / "baseline_v1.yaml"


def _load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Protocol must be a YAML mapping: {path}")
    return document


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    pre_args, _ = pre_parser.parse_known_args()
    protocol = _load_protocol(pre_args.protocol_config.resolve())
    common = protocol["common"]
    recipe = protocol["yolo11m"]
    augmentation = recipe["augmentations"]
    parser = argparse.ArgumentParser(
        description="Logged and reproducible YOLO11 fine-tuning", parents=[pre_parser]
    )
    parser.add_argument("--run-id")
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data" / "processed" / "micropcb_rpi" / "dataset.yaml"
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "benchmarks")
    parser.add_argument("--epochs", type=int, default=common["epochs"])
    parser.add_argument("--batch", type=int, default=common["batch_size"])
    parser.add_argument("--imgsz", type=int, default=common["image_size"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=common["seeds"][0])
    parser.add_argument("--optimizer", default=recipe["optimizer"])
    parser.add_argument("--nominal-batch-size", type=int, default=recipe["nominal_batch_size"])
    parser.add_argument("--lr0", type=float, default=recipe["lr0"])
    parser.add_argument("--lrf", type=float, default=recipe["final_lr_ratio"])
    parser.add_argument("--momentum", type=float, default=recipe["momentum"])
    parser.add_argument("--weight-decay", type=float, default=recipe["weight_decay"])
    parser.add_argument("--warmup-epochs", type=float, default=recipe["warmup_epochs"])
    parser.add_argument("--close-mosaic", type=int, default=recipe["close_mosaic_epochs"])
    parser.add_argument("--freeze", type=int, default=recipe["freeze_layers"])
    parser.add_argument("--hsv-h", type=float, default=augmentation["hsv_h"])
    parser.add_argument("--hsv-s", type=float, default=augmentation["hsv_s"])
    parser.add_argument("--hsv-v", type=float, default=augmentation["hsv_v"])
    parser.add_argument("--degrees", type=float, default=augmentation["degrees"])
    parser.add_argument("--translate", type=float, default=augmentation["translate"])
    parser.add_argument("--scale", type=float, default=augmentation["scale"])
    parser.add_argument("--shear", type=float, default=augmentation["shear"])
    parser.add_argument("--perspective", type=float, default=augmentation["perspective"])
    parser.add_argument("--flipud", type=float, default=augmentation["flipud"])
    parser.add_argument("--fliplr", type=float, default=augmentation["fliplr"])
    parser.add_argument("--mosaic", type=float, default=augmentation["mosaic"])
    parser.add_argument("--mixup", type=float, default=augmentation["mixup"])
    parser.add_argument("--cutmix", type=float, default=augmentation["cutmix"])
    parser.add_argument("--copy-paste", type=float, default=augmentation["copy_paste"])
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=recipe["cos_lr"])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--gpu-sample-seconds", type=float, default=1.0)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--predict-batch", type=int, default=8)
    parser.add_argument("--fp32", action="store_true", help="Disable mixed precision")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Force a one-epoch infrastructure check")
    return parser.parse_args()


def _new_run_id(model: str) -> str:
    return Path(model).stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _model_source_path(model: YOLO) -> Path | None:
    candidates = [getattr(model, "ckpt_path", None), getattr(model, "model_name", None)]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            return path.resolve()
        local = PROJECT_ROOT / path.name
        if local.exists():
            return local.resolve()
    return None


def _optimizer_details(trainer: Any) -> dict[str, Any]:
    groups = []
    for index, group in enumerate(trainer.optimizer.param_groups):
        groups.append(
            {
                "index": index,
                "parameters": sum(parameter.numel() for parameter in group["params"]),
                "lr": group.get("lr"),
                "initial_lr": group.get("initial_lr"),
                "weight_decay": group.get("weight_decay"),
                "momentum": group.get("momentum"),
                "betas": group.get("betas"),
            }
        )
    return {
        "optimizer": type(trainer.optimizer).__name__,
        "groups": groups,
        "resolved_batch_size": trainer.batch_size,
        "nominal_batch_size": int(trainer.args.nbs),
        "gradient_accumulation_steps": int(trainer.accumulate),
        "approximate_effective_batch_size": int(trainer.batch_size * trainer.accumulate),
        "amp": bool(trainer.amp),
    }


def _plot_yolo11_epochs(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for key, label in [
        ("train_box_loss", "box"),
        ("train_cls_loss", "class"),
        ("train_dfl_loss", "DFL"),
    ]:
        axes[0, 0].plot(epochs, [row[key] for row in rows], marker="o", label=label)
    axes[0, 0].set_title("YOLO11 native training losses")
    axes[0, 0].legend()
    for key, label in [("precision", "P"), ("recall", "R")]:
        axes[0, 1].plot(epochs, [row[key] for row in rows], marker="o", label=label)
    axes[0, 1].set_title("Native validation P/R")
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].legend()
    for key, label in [("map50", "AP50"), ("map50_95", "AP50-95")]:
        axes[1, 0].plot(epochs, [row[key] for row in rows], marker="o", label=label)
    axes[1, 0].set_title("Native validation AP")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, [row["lr"] for row in rows], marker="o")
    axes[1, 1].set_title("Learning rate (parameter group 0)")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
    fig.suptitle("YOLO11 fine-tuning — exact values are stored in epoch_metrics.csv")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _export_predictions(model: YOLO, annotation_path: Path, image_root: Path, args: argparse.Namespace, output: Path) -> None:
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = sorted(document["images"], key=lambda item: item["id"])
    category_ids = [item["id"] for item in sorted(document["categories"], key=lambda item: item["id"])]
    predictions: list[dict[str, Any]] = []
    torch.cuda.empty_cache()
    for start in range(0, len(images), args.predict_batch):
        image_batch = images[start : start + args.predict_batch]
        source_batch = [str((image_root / item["file_name"]).resolve()) for item in image_batch]
        prediction_args = {
            "source": source_batch,
            "imgsz": args.imgsz,
            "device": 0,
            "conf": 0.001,
            "iou": 0.65,
            "max_det": 300,
            "stream": False,
            "verbose": False,
            "save": False,
        }
        prediction_args["quantize"] = None if args.fp32 else 16
        results = model.predict(**prediction_args)
        for image, result in zip(image_batch, results, strict=True):
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confidence = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_index in zip(xyxy, confidence, classes, strict=True):
                x1, y1, x2, y2 = [float(value) for value in box]
                predictions.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": int(category_ids[class_index]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
        print(f"[PREDICT {min(start + len(image_batch), len(images)):04d}/{len(images):04d}] boxes={len(predictions)}")
    write_json(output, predictions)
    print(f"COCO predictions: {len(predictions)} boxes -> {output}")


def _benchmark(model: YOLO, image_path: Path, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read benchmark image: {image_path}")
    predict_args = {
        "source": image,
        "imgsz": args.imgsz,
        "device": 0,
        "quantize": None if args.fp32 else 16,
        "conf": 0.25,
        "iou": 0.65,
        "max_det": 300,
        "verbose": False,
        "save": False,
    }
    torch.cuda.empty_cache()
    if args.benchmark_warmup:
        model.predict(**predict_args)
    for _ in range(max(args.benchmark_warmup - 1, 0)):
        model.predict(**predict_args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    total_started = time.perf_counter()
    for iteration in range(args.benchmark_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start_event.record()
        model.predict(**predict_args)
        end_event.record()
        torch.cuda.synchronize()
        e2e_ms = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "iteration": iteration + 1,
                "gpu_ms": float(start_event.elapsed_time(end_event)),
                "e2e_ms": e2e_ms,
            }
        )
    total_seconds = time.perf_counter() - total_started
    _write_rows(output_dir / "latency_samples.csv", rows)
    gpu_values = np.array([row["gpu_ms"] for row in rows])
    e2e_values = np.array([row["e2e_ms"] for row in rows])
    summary = {
        "batch": 1,
        "precision": "FP32" if args.fp32 else "FP16",
        "warmup_iterations": args.benchmark_warmup,
        "measured_iterations": args.benchmark_iterations,
        "gpu_p50_ms": float(np.percentile(gpu_values, 50)),
        "gpu_p95_ms": float(np.percentile(gpu_values, 95)),
        "e2e_p50_ms": float(np.percentile(e2e_values, 50)),
        "e2e_p95_ms": float(np.percentile(e2e_values, 95)),
        "sustained_fps": args.benchmark_iterations / total_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "note": "preloaded BGR camera-like frame; preprocess + inference + NMS, display excluded",
    }
    write_json(output_dir / "latency.json", summary)
    print_section("BATCH-1 LATENCY", summary)
    return summary


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def main() -> int:
    configure_utf8_output()
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.workers = 0
        args.fraction = min(args.fraction, 0.02)
        args.benchmark_warmup = min(args.benchmark_warmup, 2)
        args.benchmark_iterations = min(args.benchmark_iterations, 10)
    run_id = args.run_id or _new_run_id(args.model)
    output_root = args.output_root.resolve()
    run_dir = output_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"Run directory is not empty: {run_dir}. Use --exist-ok to continue.")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    pretrained_path = _model_source_path(model)
    model_parameters = sum(parameter.numel() for parameter in model.model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad
    )
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = float(get_flops(model.model, args.imgsz))
    except Exception:
        gflops = None
    model_details = {
        "parameters": model_parameters,
        "checkpoint_graph_trainable_parameters_before_trainer": trainable_parameters,
        "gflops_ultralytics": gflops,
        "input": f"1x3x{args.imgsz}x{args.imgsz}",
    }
    initial_stats = state_dict_statistics(model.model.state_dict(), run_dir / "pretrained_weights_summary.csv")
    pretrained_record = checkpoint_file_record(pretrained_path) if pretrained_path else None
    train_annotation = PROJECT_ROOT / "data" / "processed" / "micropcb_rpi_coco" / "annotations" / "instances_train2017.json"
    val_annotation = PROJECT_ROOT / "data" / "processed" / "micropcb_rpi_coco" / "annotations" / "instances_val2017.json"
    protocol = {
        "method": "pretrained checkpoint -> full detector fine-tuning",
        "comparison_kind": "framework_native_recipe_system_benchmark",
        "epochs": args.epochs,
        "batch": args.batch,
        "nominal_batch_size": args.nominal_batch_size,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "workers": args.workers,
        "optimizer_requested": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "close_mosaic": args.close_mosaic,
        "freeze_layers": args.freeze,
        "cos_lr": args.cos_lr,
        "augmentations": {
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
            "degrees": args.degrees,
            "translate": args.translate,
            "scale": args.scale,
            "shear": args.shear,
            "perspective": args.perspective,
            "flipud": args.flipud,
            "fliplr": args.fliplr,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "cutmix": args.cutmix,
            "copy_paste": args.copy_paste,
        },
        "fraction": args.fraction,
        "amp": not args.fp32,
        "deterministic": True,
        "validation_each_epoch": True,
        "prediction_floor": 0.001,
        "nms_iou": 0.65,
        "common_operating_confidence": 0.25,
        "common_match_iou": 0.50,
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "framework": "Ultralytics",
        "framework_version": ultralytics.__version__,
        "model": Path(args.model).stem,
        "stage": "smoke_not_comparable" if args.smoke else "fine_tune_candidate",
        "status": "running",
        "start_utc": utc_now_precise(),
        "command": [sys.executable, *sys.argv],
        "protocol": protocol,
        "model_details": model_details,
        "pretrained_checkpoint": pretrained_record,
        "pretrained_weight_statistics": initial_stats,
        "dataset": {
            "train_annotation_sha256": sha256_file(train_annotation),
            "val_annotation_sha256": sha256_file(val_annotation),
            "independent_test_available": False,
        },
        "protocol_config": {
            "path": str(args.protocol_config.resolve()),
            "sha256": sha256_file(args.protocol_config.resolve()),
        },
        "git": collect_git_state(PROJECT_ROOT),
        "environment": collect_system_environment() | collect_torch_environment(),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_pip_freeze(run_dir / "pip-freeze.txt")
    callback_state: dict[str, Any] = {
        "train_peak_mib": None,
        "train_peak_reserved_mib": None,
        "resolved": None,
        "gpu_by_epoch": {},
        "epoch_rows": [],
    }

    def on_pretrain_routine_end(trainer: Any) -> None:
        resolved = _optimizer_details(trainer)
        callback_state["resolved"] = resolved
        current = _read_json(run_dir / "run_manifest.json")
        current["resolved_training"] = resolved
        current["model_details"]["trainable_parameters_after_freeze"] = sum(
            parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
        )
        write_json(run_dir / "run_manifest.json", current)
        print_section("RESOLVED OPTIMIZER / AMP / BATCH", resolved)

    def on_train_epoch_start(_: Any) -> None:
        torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(_: Any) -> None:
        callback_state["train_peak_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        callback_state["train_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2

    def on_fit_epoch_end(trainer: Any) -> None:
        if trainer.epoch >= trainer.epochs:
            print("FINAL_BEST_VALIDATION callback recorded separately; no duplicate epoch row added.")
            return
        row = {column: "" for column in EPOCH_COLUMNS}
        row["epoch"] = int(trainer.epoch) + 1
        row["elapsed_s"] = float(time.time() - trainer.train_time_start)
        train_mapping = {
            "box_loss": "train_box_loss",
            "cls_loss": "train_cls_loss",
            "dfl_loss": "train_dfl_loss",
        }
        for name, value in (trainer.tloss or {}).items():
            destination = train_mapping.get(str(name))
            if destination:
                row[destination] = _to_float(value)
        metric_mapping = {
            "val/box_loss": "val_box_loss",
            "val/cls_loss": "val_cls_loss",
            "val/dfl_loss": "val_dfl_loss",
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
            "metrics/mAP50(B)": "map50",
            "metrics/mAP50-95(B)": "map50_95",
        }
        for name, value in (trainer.metrics or {}).items():
            destination = metric_mapping.get(str(name))
            if destination:
                row[destination] = _to_float(value)
        row["lr"] = _to_float((trainer.lr or {}).get("lr/pg0")) or 0.0
        row["train_peak_allocated_mib"] = callback_state["train_peak_mib"]
        row["train_peak_reserved_mib"] = callback_state["train_peak_reserved_mib"]
        row["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        row["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        callback_state["gpu_by_epoch"][int(row["epoch"])] = {
            key: row[key]
            for key in (
                "train_peak_allocated_mib",
                "train_peak_reserved_mib",
                "gpu_peak_allocated_mib",
                "gpu_peak_reserved_mib",
            )
        }
        callback_state["epoch_rows"].append(row)
        rows = callback_state["epoch_rows"]
        _write_rows(run_dir / "epoch_metrics.csv", rows)
        append_jsonl(run_dir / "epoch_metrics_extra.jsonl", row)
        _plot_yolo11_epochs(rows, run_dir / "plots" / "latest_overview.png")
        _plot_yolo11_epochs(
            rows, run_dir / "plots" / "epochs" / f"epoch_{int(row['epoch']):03d}_overview.png"
        )
        print(
            f"[EPOCH {int(row['epoch']):03d}/{trainer.epochs:03d}] "
            f"box={row['train_box_loss']:.5f} cls={row['train_cls_loss']:.5f} "
            f"dfl={row['train_dfl_loss']:.5f} | P={row['precision']:.5f} R={row['recall']:.5f} "
            f"AP50={row['map50']:.5f} AP50-95={row['map50_95']:.5f} | "
            f"LR={row['lr']:.8f} train_VRAM_peak={callback_state['train_peak_mib']:.1f} MiB"
        )
        print("EPOCH_METRICS_JSON " + json.dumps(row, ensure_ascii=False, allow_nan=False))

    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    exit_code = 1
    gpu_sampler = GpuSampler(run_dir / "gpu_samples.csv", args.gpu_sample_seconds)
    with tee_console(run_dir / "terminal.log"):
        print("\nYOLO11 LOGGED FINE-TUNING")
        print("=" * 72)
        print_section("RUN", {"run_id": run_id, "directory": run_dir, "stage": manifest["stage"]})
        print_section("CUDA", manifest["environment"])
        print_section("MODEL", model_details)
        print_section("PRETRAINED CHECKPOINT", pretrained_record or {"path": args.model})
        print_section("PRETRAINED WEIGHT STATISTICS", initial_stats)
        print_section("FINE-TUNING PROTOCOL", protocol)
        write_protocol_artifacts(args.protocol_config.resolve(), run_dir)
        gpu_sampler.start()
        try:
            model.train(
                data=str(args.data.resolve()),
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=0,
                workers=args.workers,
                seed=args.seed,
                deterministic=True,
                optimizer=args.optimizer,
                nbs=args.nominal_batch_size,
                lr0=args.lr0,
                lrf=args.lrf,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                close_mosaic=args.close_mosaic,
                cos_lr=args.cos_lr,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
                degrees=args.degrees,
                translate=args.translate,
                scale=args.scale,
                shear=args.shear,
                perspective=args.perspective,
                flipud=args.flipud,
                fliplr=args.fliplr,
                mosaic=args.mosaic,
                mixup=args.mixup,
                cutmix=args.cutmix,
                copy_paste=args.copy_paste,
                multi_scale=0.0,
                freeze=args.freeze,
                fraction=args.fraction,
                amp=not args.fp32,
                patience=0,
                val=True,
                plots=True,
                save=True,
                save_period=-1,
                project=str(run_dir),
                name="native",
                exist_ok=True,
                verbose=True,
            )
            exit_code = 0
        except Exception as exc:
            print(f"TRAINING_EXCEPTION {type(exc).__name__}: {exc}")
            manifest["training_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            gpu_summary = gpu_sampler.stop()
            write_json(run_dir / "gpu_summary.json", gpu_summary)
        print_section("GPU SAMPLING SUMMARY", gpu_summary)

        native_dir = run_dir / "native"
        if (native_dir / "results.csv").exists() and not (run_dir / "epoch_metrics.csv").exists():
            rows = normalize_yolo11_results(native_dir / "results.csv", run_dir / "epoch_metrics.csv")
            _plot_yolo11_epochs(rows, run_dir / "plots" / "latest_overview.png")
        best_path = native_dir / "weights" / "best.pt"
        if not best_path.exists():
            best_path = native_dir / "weights" / "last.pt"
        current_manifest = _read_json(run_dir / "run_manifest.json")
        current_manifest["status"] = "complete" if exit_code == 0 else "failed"
        current_manifest["end_utc"] = utc_now_precise()
        current_manifest["gpu_summary"] = gpu_summary
        if manifest.get("training_error"):
            current_manifest["training_error"] = manifest["training_error"]
        if exit_code == 0 and best_path.exists():
            best_record = checkpoint_file_record(best_path)
            current_manifest["best_checkpoint"] = best_record
            best_model = YOLO(str(best_path))
            current_manifest["best_weight_statistics"] = state_dict_statistics(
                best_model.model.state_dict(), run_dir / "best_weights_summary.csv"
            )
            print_section("FINAL CHECKPOINT", best_record)
            print_section("FINAL WEIGHT STATISTICS", current_manifest["best_weight_statistics"])
            val_root = PROJECT_ROOT / "data" / "processed" / "micropcb_rpi_coco" / "val2017"
            _export_predictions(
                best_model, val_annotation, val_root, args, run_dir / "predictions.coco.json"
            )
            evaluate_predictions(val_annotation, run_dir / "predictions.coco.json", run_dir)
            val_document = json.loads(val_annotation.read_text(encoding="utf-8"))
            benchmark_image = val_root / val_document["images"][0]["file_name"]
            _benchmark(best_model, benchmark_image, args, run_dir)
        write_json(run_dir / "run_manifest.json", current_manifest)
        if (run_dir / "epoch_metrics.csv").exists():
            compare_runs([run_dir], run_dir / "plots" / "summary")
        print(f"\nRUN STATUS: {'PASS' if exit_code == 0 else 'FAIL'}")
        print(f"Artifacts: {run_dir}")
    return exit_code


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
