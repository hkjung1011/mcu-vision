from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
import torch
import yaml
import yolox
from yolox.data import ValTransform
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess

from mcu_data.common import portable_path, sha256_file, write_json
from mcu_data.dataset_evidence import (
    resolve_protocol_test_evidence,
    verify_dataset_against_evidence,
)
from mcu_data.methodology import write_protocol_artifacts
from mcu_data.reporting import compare_runs, evaluate_predictions
from mcu_data.runlog import (
    GpuSampler,
    checkpoint_file_record,
    collect_system_environment,
    collect_torch_environment,
    collect_git_state,
    configure_utf8_output,
    print_section,
    run_streamed,
    state_dict_statistics,
    tee_console,
    utc_now_precise,
    write_pip_freeze,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "experiments" / "baseline_v1.yaml"


def _load_protocol(path: Path) -> dict:
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
    recipe = protocol["yolox_s"]
    parser = argparse.ArgumentParser(
        description="Logged and reproducible YOLOX-S fine-tuning", parents=[pre_parser]
    )
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "yolox_s_micropcb.py")
    parser.add_argument(
        "--pretrained", type=Path, default=PROJECT_ROOT / "weights" / "pretrained" / "yolox_s.pth"
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "micropcb_rpi_phash_v2_coco",
        help="COCO dataset root containing annotations/, train2017/, and val2017/",
    )
    parser.add_argument(
        "--yolo-data",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "micropcb_rpi_phash_v2"
        / "dataset.yaml",
        help="Matching YOLO dataset used only for the live equivalence preflight",
    )
    parser.add_argument(
        "--dataset-evidence",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "evidence"
        / "micropcb_rpi_phash_v2"
        / "dataset_evidence.json",
        help="PASS evidence whose hashes must be reproduced by the live YOLO/COCO inputs",
    )
    parser.add_argument("--coco-test", type=Path, help="Locked test COCO evidence input only")
    parser.add_argument(
        "--coco-test-images", type=Path, help="Locked test image root used only for evidence"
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "benchmarks")
    parser.add_argument("--epochs", type=int, default=common["epochs"])
    parser.add_argument("--batch", type=int, default=common["batch_size"])
    parser.add_argument("--imgsz", type=int, default=common["image_size"])
    parser.add_argument("--workers", type=int, default=common.get("workers", 0))
    parser.add_argument("--seed", type=int, default=common["seeds"][0])
    parser.add_argument("--no-aug-epochs", type=int, default=recipe["no_augmentation_epochs"])
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--gpu-sample-seconds", type=float, default=1.0)
    parser.add_argument("--fp32", action="store_true", help="Disable mixed precision")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Force a one-epoch infrastructure check")
    return parser.parse_args()


def _new_run_id() -> str:
    return "yolox_s_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_trusted_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    configure_utf8_output()
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.workers = 0
        args.no_aug_epochs = 1
    run_id = args.run_id or _new_run_id()
    output_root = args.output_root.resolve()
    run_dir = output_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"Run directory is not empty: {run_dir}. Use --exist-ok to continue.")
    run_dir.mkdir(parents=True, exist_ok=True)

    coco_root = args.coco_root.resolve()
    protocol_document = _load_protocol(args.protocol_config.resolve())
    train_annotation = coco_root / "annotations" / "instances_train2017.json"
    val_annotation = coco_root / "annotations" / "instances_val2017.json"
    test_annotation, test_image_root, include_coco_attributes = resolve_protocol_test_evidence(
        dataset_config=protocol_document["dataset"],
        coco_root=coco_root,
        coco_test_annotations=args.coco_test,
        coco_test_image_root=args.coco_test_images,
    )
    for required_path in (train_annotation, val_annotation):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    dataset_evidence = verify_dataset_against_evidence(
        evidence_path=args.dataset_evidence.resolve(),
        yolo_dataset_yaml=args.yolo_data.resolve(),
        coco_train_annotations=train_annotation,
        coco_val_annotations=val_annotation,
        coco_test_annotations=test_annotation,
        coco_test_image_root=test_image_root,
        include_coco_attributes=include_coco_attributes,
    )
    train_document = _read_json(train_annotation)
    categories = train_document.get("categories", [])
    category_ids = sorted(int(category["id"]) for category in categories)
    if not category_ids or category_ids != list(range(1, len(category_ids) + 1)):
        raise ValueError(
            "YOLOX requires consecutive COCO category IDs beginning at 1; "
            f"found {category_ids}"
        )
    num_classes = len(category_ids)

    environment = os.environ.copy()
    environment.update(
        {
            "MCU_OUTPUT_ROOT": str(output_root),
            "MCU_EPOCHS": str(args.epochs),
            "MCU_IMAGE_SIZE": str(args.imgsz),
            "MCU_WORKERS": str(args.workers),
            "MCU_SEED": str(args.seed),
            "MCU_NO_AUG_EPOCHS": str(args.no_aug_epochs),
            "MCU_PRINT_INTERVAL": str(args.print_interval),
            "MCU_EVAL_INTERVAL": "1",
            "MCU_PREDICTION_FLOOR": "0.001",
            "MCU_NMS_IOU": "0.65",
            "MCU_COCO_DATA_DIR": str(coco_root),
            "MCU_NUM_CLASSES": str(num_classes),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(PROJECT_ROOT / "src")
            + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
        }
    )
    os.environ.update({key: value for key, value in environment.items() if key.startswith("MCU_")})
    exp = get_exp(str(args.config.resolve()), None)
    model = exp.get_model()
    checkpoint = _load_trusted_checkpoint(args.pretrained.resolve())
    pretrained_state = checkpoint["model"]
    target_state = model.state_dict()
    matched = [
        name
        for name, value in pretrained_state.items()
        if name in target_state and tuple(value.shape) == tuple(target_state[name].shape)
    ]
    unexpected = [name for name in pretrained_state if name not in target_state]
    shape_mismatch = [
        name
        for name, value in pretrained_state.items()
        if name in target_state and tuple(value.shape) != tuple(target_state[name].shape)
    ]
    missing = [name for name in target_state if name not in pretrained_state]
    model_details = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "summary": get_model_info(model, (args.imgsz, args.imgsz)),
        "pretrained_tensors": len(pretrained_state),
        "matched_tensors": len(matched),
        "missing_names": len(missing),
        "unexpected_names": len(unexpected),
        "shape_mismatch_names": len(shape_mismatch),
        "shape_mismatch_examples": shape_mismatch[:12],
    }
    pretrained_record = checkpoint_file_record(args.pretrained.resolve())
    initial_stats = state_dict_statistics(pretrained_state, run_dir / "pretrained_weights_summary.csv")
    command = [
        sys.executable,
        "-m",
        "yolox.tools.train",
        "-f",
        str(args.config.resolve()),
        "-d",
        "1",
        "-b",
        str(args.batch),
        "-c",
        str(args.pretrained.resolve()),
        "-expn",
        run_id,
    ]
    if not args.fp32:
        command.append("--fp16")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "framework": "YOLOX",
        "framework_version": getattr(yolox, "__version__", "source"),
        "framework_commit": "6ddff4824372906469a7fae2dc3206c7aa4bbaee",
        "model": "YOLOX-S",
        "stage": "smoke_not_comparable" if args.smoke else "fine_tune_candidate",
        "status": "running",
        "start_utc": utc_now_precise(),
        "command": command,
        "protocol": {
            "comparison_kind": "framework_native_recipe_system_benchmark",
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "multiscale_range": exp.multiscale_range,
            "seed": args.seed,
            "workers": args.workers,
            "amp": not args.fp32,
            "validation_each_epoch": True,
            "prediction_floor": exp.test_conf,
            "nms_iou": exp.nmsthre,
            "class_agnostic_nms": False,
            "max_detections_for_coco_ap": 100,
            "common_operating_confidence": 0.25,
            "common_match_iou": 0.50,
            "optimizer": "SGD Nesterov",
            "momentum": exp.momentum,
            "weight_decay": exp.weight_decay,
            "warmup_epochs": exp.warmup_epochs,
            "no_aug_epochs": exp.no_aug_epochs,
            "base_lr_per_image": exp.basic_lr_per_img,
            "base_lr_for_batch": exp.basic_lr_per_img * args.batch,
            "minimum_lr": exp.basic_lr_per_img * args.batch * exp.min_lr_ratio,
            "augmentations": {
                "degrees": exp.degrees,
                "translate": exp.translate,
                "mosaic_scale": list(exp.mosaic_scale),
                "mixup_scale": list(exp.mixup_scale),
                "enable_mixup": bool(exp.enable_mixup),
                "shear": exp.shear,
                "flip_probability": exp.flip_prob,
                "hsv_probability": exp.hsv_prob,
            },
        },
        "model_details": model_details,
        "pretrained_checkpoint": pretrained_record,
        "pretrained_weight_statistics": initial_stats,
        "dataset": {
            "yolo_dataset_yaml": str(args.yolo_data.resolve()),
            "coco_root": str(coco_root),
            "classes": [category["name"] for category in categories],
            "num_classes": num_classes,
            "equivalence_evidence_path": portable_path(args.dataset_evidence.resolve()),
            "equivalence_evidence_sha256": sha256_file(args.dataset_evidence.resolve()),
            "train_annotation_sha256": sha256_file(train_annotation),
            "val_annotation_sha256": sha256_file(val_annotation),
            "locked_test_evidence_enabled": test_annotation is not None,
            "test_annotation_sha256": (
                sha256_file(test_annotation) if test_annotation is not None else None
            ),
            "independent_test_available": bool(
                protocol_document.get("common", {}).get(
                    "independent_test_available", False
                )
            ),
            **dataset_evidence,
        },
        "protocol_config": {
            "path": str(args.protocol_config.resolve()),
            "sha256": sha256_file(args.protocol_config.resolve()),
        },
        "experiment_config": {
            "path": str(args.config.resolve()),
            "sha256": sha256_file(args.config.resolve()),
        },
        "git": collect_git_state(PROJECT_ROOT),
        "environment": collect_system_environment() | collect_torch_environment(),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_pip_freeze(run_dir / "pip-freeze.txt")

    exit_code = 1
    gpu_sampler = GpuSampler(run_dir / "gpu_samples.csv", args.gpu_sample_seconds)
    with tee_console(run_dir / "terminal.log"):
        print("\nYOLOX-S LOGGED FINE-TUNING")
        print("=" * 72)
        print_section("RUN", {"run_id": run_id, "directory": run_dir, "stage": manifest["stage"]})
        print_section("CUDA", manifest["environment"])
        print_section("MODEL", model_details)
        print_section("PRETRAINED CHECKPOINT", pretrained_record)
        print_section("LIVE YOLO/COCO DATASET EQUIVALENCE", {"status": "PASS", **dataset_evidence})
        print_section("FINE-TUNING PROTOCOL", manifest["protocol"])
        write_protocol_artifacts(args.protocol_config.resolve(), run_dir)
        print("\nCOMMAND (notice: -o/--occupy is intentionally absent)")
        print(" ".join(command))
        gpu_sampler.start()
        try:
            exit_code = run_streamed(command, cwd=PROJECT_ROOT, env=environment)
        finally:
            gpu_summary = gpu_sampler.stop()
            write_json(run_dir / "gpu_summary.json", gpu_summary)
        print_section("GPU SAMPLING SUMMARY", gpu_summary)

        current_manifest = _read_json(run_dir / "run_manifest.json")
        current_manifest.update(manifest)
        current_manifest["status"] = "complete" if exit_code == 0 else "failed"
        current_manifest["end_utc"] = utc_now_precise()
        checkpoint_path = run_dir / "best_ckpt.pth"
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "last_epoch_ckpt.pth"
        if checkpoint_path.exists():
            record = checkpoint_file_record(checkpoint_path)
            current_manifest["best_checkpoint"] = record
            best_state = _load_trusted_checkpoint(checkpoint_path)["model"]
            current_manifest["best_weight_statistics"] = state_dict_statistics(
                best_state, run_dir / "best_weights_summary.csv"
            )
            print_section("FINAL CHECKPOINT", record)
        write_json(run_dir / "run_manifest.json", current_manifest)

        predictions = run_dir / "predictions.coco.json"
        if exit_code == 0 and checkpoint_path.exists():
            best_model = exp.get_model()
            best_model.load_state_dict(_load_trusted_checkpoint(checkpoint_path)["model"])
            best_model.cuda().eval()
            if not args.fp32:
                best_model.half()
            evaluator = exp.get_evaluator(batch_size=args.batch, is_distributed=False)
            evaluator.audit_output_dir = run_dir
            exp.eval(
                best_model,
                evaluator,
                is_distributed=False,
                half=not args.fp32,
                return_outputs=True,
            )
            if predictions.exists():
                evaluate_predictions(val_annotation, predictions, run_dir)
            val_document = _read_json(val_annotation)
            benchmark_image = coco_root / "val2017" / val_document["images"][0]["file_name"]
            _benchmark_yolox(best_model, benchmark_image, exp, args, run_dir)
        if (run_dir / "epoch_metrics.csv").exists():
            compare_runs([run_dir], run_dir / "plots" / "summary")
        print(f"\nRUN STATUS: {'PASS' if exit_code == 0 else 'FAIL'} (exit_code={exit_code})")
        print(f"Artifacts: {run_dir}")
    return exit_code


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _benchmark_yolox(model, image_path: Path, exp, args: argparse.Namespace, output_dir: Path) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read benchmark image: {image_path}")
    transform = ValTransform(legacy=False)

    def infer_once() -> None:
        transformed, _ = transform(image, None, (args.imgsz, args.imgsz))
        tensor = torch.from_numpy(transformed).unsqueeze(0).cuda(non_blocking=False)
        tensor = tensor.float() if args.fp32 else tensor.half()
        outputs = model(tensor)
        postprocess(outputs, exp.num_classes, 0.25, 0.65, class_agnostic=False)

    warmup = 2 if args.smoke else 20
    iterations = 10 if args.smoke else 100
    with torch.inference_mode():
        for _ in range(warmup):
            infer_once()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        total_started = time.perf_counter()
        for iteration in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            start_event.record()
            infer_once()
            end_event.record()
            torch.cuda.synchronize()
            rows.append(
                {
                    "iteration": iteration + 1,
                    "gpu_ms": float(start_event.elapsed_time(end_event)),
                    "e2e_ms": (time.perf_counter() - started) * 1000,
                }
            )
        total_seconds = time.perf_counter() - total_started
    with (output_dir / "latency_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gpu_values = np.array([row["gpu_ms"] for row in rows])
    e2e_values = np.array([row["e2e_ms"] for row in rows])
    summary = {
        "batch": 1,
        "precision": "FP32" if args.fp32 else "FP16",
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "gpu_p50_ms": float(np.percentile(gpu_values, 50)),
        "gpu_p95_ms": float(np.percentile(gpu_values, 95)),
        "e2e_p50_ms": float(np.percentile(e2e_values, 50)),
        "e2e_p95_ms": float(np.percentile(e2e_values, 95)),
        "sustained_fps": iterations / total_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "note": "preloaded BGR frame; preprocess + transfer + inference + NMS, display excluded",
    }
    write_json(output_dir / "latency.json", summary)
    print_section("BATCH-1 LATENCY", summary)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
