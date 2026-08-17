from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
import torch

from mcu_data.common import portable_path, sha256_file, write_json
from mcu_data.deployment import (
    artifact_record,
    category_names,
    compare_arrays,
    decode_detections,
    load_coco_sample,
    preprocess_image,
    preprocessing_spec,
    restore_boxes,
)
from mcu_data.publishing import validate_comparison_for_run


DEFAULT_YOLOX_CONFIG = PROJECT_ROOT / "configs" / "yolox_s_micropcb.py"
FORMAL_ABSOLUTE_TOLERANCE = 1e-3
FORMAL_RELATIVE_TOLERANCE = 1e-4
FORMAL_CONFIDENCE = 0.25
FORMAL_NMS_IOU = 0.65
FORMAL_MAX_DETECTIONS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained YOLO11/YOLOX checkpoint to fixed batch-1 ONNX and verify "
            "the raw native and ONNX Runtime outputs on one image referenced by COCO."
        )
    )
    parser.add_argument("--framework", choices=("yolo11", "yolox"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--run-manifest",
        type=Path,
        required=True,
        help="Completed run_manifest.json; checkpoint SHA-256, stage, dataset, and image size are verified",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        help="Formal release-ready comparison containing this exact run manifest",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Allow export without a release-ready comparison; output under weights/trained remains forbidden",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", help="Artifact stem; default is checkpoint stem")
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--coco-images", type=Path, required=True)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--absolute-tolerance", type=float, default=FORMAL_ABSOLUTE_TOLERANCE)
    parser.add_argument("--relative-tolerance", type=float, default=FORMAL_RELATIVE_TOLERANCE)
    parser.add_argument("--confidence", type=float, default=FORMAL_CONFIDENCE)
    parser.add_argument("--nms-iou", type=float, default=FORMAL_NMS_IOU)
    parser.add_argument("--max-detections", type=int, default=FORMAL_MAX_DETECTIONS)
    parser.add_argument("--yolox-config", type=Path, default=DEFAULT_YOLOX_CONFIG)
    parser.add_argument(
        "--allow-class-name-mismatch",
        action="store_true",
        help="Diagnostic only: permit YOLO11 checkpoint names to differ from COCO names (count must match)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _normalize_model_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        keys = sorted(int(key) for key in value)
        if keys != list(range(len(keys))):
            raise ValueError(f"Model class keys must be 0..N-1, found {keys}")
        return [str(value[key] if key in value else value[str(key)]) for key in keys]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError(f"Unsupported model names container: {type(value).__name__}")


def _as_single_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"Expected one tensor model output, got {type(value).__name__}")


def _export_yolo11(
    checkpoint: Path,
    onnx_path: Path,
    input_array: np.ndarray,
    opset: int,
) -> tuple[np.ndarray, list[str], dict[str, str]]:
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f, Detect

    wrapper = YOLO(str(checkpoint), task="detect")
    names = _normalize_model_names(wrapper.names)
    model = copy.deepcopy(wrapper.model).cpu().float().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, Detect):
            module.dynamic = False
            module.export = True
            module.format = "onnx"
            module.max_det = 100
            module.agnostic_nms = False
            module.xyxy = False
            module.shape = None
        elif isinstance(module, C2f):
            module.forward = module.forward_split

    tensor = torch.from_numpy(input_array)
    with torch.inference_mode():
        native = _as_single_tensor(model(tensor)).detach().cpu().numpy()
    torch.onnx.export(
        model,
        tensor,
        str(onnx_path),
        export_params=True,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=None,
        dynamo=False,
    )
    return native, names, {
        "framework": "Ultralytics",
        "framework_version": ultralytics.__version__,
        "output_semantics": "[batch, 4+classes, anchors]; decoded xywh pixels + class probabilities; NMS external",
    }


def _export_yolox(
    checkpoint: Path,
    config: Path,
    onnx_path: Path,
    input_array: np.ndarray,
    opset: int,
    image_size: int,
    class_count: int,
) -> tuple[np.ndarray, dict[str, str]]:
    import yolox
    from torch import nn
    from yolox.exp import get_exp
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    os.environ["MCU_IMAGE_SIZE"] = str(image_size)
    os.environ["MCU_NUM_CLASSES"] = str(class_count)
    experiment = get_exp(str(config.resolve()), None)
    model = experiment.get_model().cpu().float().eval()
    checkpoint_document = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint_document.get("model", checkpoint_document)
    if not isinstance(state_dict, dict):
        raise ValueError("YOLOX checkpoint must be a state dict or contain a 'model' state dict")
    model.load_state_dict(state_dict, strict=True)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = True
    for parameter in model.parameters():
        parameter.requires_grad = False

    tensor = torch.from_numpy(input_array)
    with torch.inference_mode():
        native = _as_single_tensor(model(tensor)).detach().cpu().numpy()
    torch.onnx.export(
        model,
        tensor,
        str(onnx_path),
        export_params=True,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=None,
        dynamo=False,
    )
    return native, {
        "framework": "YOLOX",
        "framework_version": getattr(yolox, "__version__", "source"),
        "framework_commit": "6ddff4824372906469a7fae2dc3206c7aa4bbaee",
        "output_semantics": (
            "[batch, anchors, 5+classes]; decoded xywh pixels + objectness + class probabilities; "
            "score=objectness*class probability; NMS external"
        ),
    }


def _embed_onnx_metadata(
    onnx_path: Path,
    *,
    framework: str,
    image_size: int,
    class_names: list[str],
    checkpoint_sha256: str,
    preprocessing: dict[str, Any],
    output_semantics: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str]:
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    del model.metadata_props[:]
    values = {
        "mcu_vision.framework": framework,
        "mcu_vision.input_size": str(image_size),
        "mcu_vision.batch": "1",
        "mcu_vision.class_names": json.dumps(class_names, ensure_ascii=False),
        "mcu_vision.checkpoint_sha256": checkpoint_sha256,
        "mcu_vision.preprocessing": json.dumps(preprocessing, ensure_ascii=False, sort_keys=True),
        "mcu_vision.output_semantics": output_semantics,
    }
    for key, value in values.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(onnx_path))
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    def dimensions(value_info: Any) -> list[int | str | None]:
        result: list[int | str | None] = []
        for dimension in value_info.type.tensor_type.shape.dim:
            if dimension.dim_value:
                result.append(int(dimension.dim_value))
            elif dimension.dim_param:
                result.append(str(dimension.dim_param))
            else:
                result.append(None)
        return result

    inputs = [{"name": item.name, "shape": dimensions(item)} for item in model.graph.input]
    outputs = [{"name": item.name, "shape": dimensions(item)} for item in model.graph.output]
    return onnx.__version__, inputs, outputs, str(model.opset_import[0].version)


def _run_onnx(onnx_path: Path, input_array: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CPUExecutionProvider" not in available:
        raise RuntimeError(f"ONNX Runtime CPU provider is unavailable; providers={available}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError(
            f"Deployment model must have exactly one input/output; got {len(session.get_inputs())}/"
            f"{len(session.get_outputs())}"
        )
    input_meta = session.get_inputs()[0]
    outputs = session.run(None, {input_meta.name: input_array})
    return np.asarray(outputs[0]), {
        "onnxruntime_version": ort.__version__,
        "provider": session.get_providers()[0],
        "input_name": input_meta.name,
        "output_name": session.get_outputs()[0].name,
    }


def _detection_summary(
    output: np.ndarray,
    *,
    framework: str,
    class_count: int,
    confidence: float,
    nms_iou: float,
    max_detections: int,
    preprocess_info: Any,
) -> dict[str, Any]:
    detections = decode_detections(
        output,
        framework=framework,  # type: ignore[arg-type]
        class_count=class_count,
        confidence=confidence,
        nms_iou=nms_iou,
        max_detections=max_detections,
    )
    restored = restore_boxes(detections, preprocess_info)
    return {
        "count": int(restored.shape[0]),
        "class_ids": sorted(set(int(item) for item in restored[:, 5])) if restored.size else [],
        "maximum_score": float(restored[:, 4].max()) if restored.size else None,
        "first_five": restored[:5].round(6).tolist(),
    }


def _clean_previous(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing deployment artifacts:\n{formatted}\nUse --force.")
    for path in existing:
        if not path.is_file():
            raise ValueError(f"Expected an artifact file, got: {path}")
        path.unlink()


def _verify_run_manifest(
    manifest_path: Path,
    *,
    checkpoint_sha256: str,
    framework: str,
    image_size: int,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Run manifest must be a JSON object: {manifest_path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Run manifest status must be complete, got {manifest.get('status')!r}")
    best_checkpoint = manifest.get("best_checkpoint")
    if not isinstance(best_checkpoint, dict) or not best_checkpoint.get("sha256"):
        raise ValueError("Run manifest is missing best_checkpoint.sha256")
    if str(best_checkpoint["sha256"]).lower() != checkpoint_sha256.lower():
        raise ValueError(
            "Run manifest/checkpoint SHA-256 mismatch: "
            f"manifest={best_checkpoint['sha256']}, checkpoint={checkpoint_sha256}"
        )
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or int(protocol.get("imgsz", -1)) != image_size:
        raise ValueError(
            f"Run manifest protocol.imgsz must equal export size {image_size}, "
            f"got {protocol.get('imgsz') if isinstance(protocol, dict) else None}"
        )
    manifest_framework = str(manifest.get("framework", "")).lower()
    if framework == "yolo11" and "ultralytics" not in manifest_framework:
        raise ValueError(f"Expected an Ultralytics run manifest, got {manifest.get('framework')!r}")
    if framework == "yolox" and "yolox" not in manifest_framework:
        raise ValueError(f"Expected a YOLOX run manifest, got {manifest.get('framework')!r}")
    protocol_config = manifest.get("protocol_config")
    if isinstance(protocol_config, dict):
        protocol_config = dict(protocol_config)
        configured_path = protocol_config.get("path")
        if configured_path:
            protocol_config["path"] = portable_path(Path(str(configured_path)))
    experiment_config = manifest.get("experiment_config")
    if isinstance(experiment_config, dict):
        experiment_config = dict(experiment_config)
        configured_path = experiment_config.get("path")
        if configured_path:
            experiment_config["path"] = portable_path(Path(str(configured_path)))
    return {
        "artifact": artifact_record(manifest_path) | {"file_name": manifest_path.name},
        "run_id": manifest.get("run_id"),
        "model": manifest.get("model"),
        "stage": manifest.get("stage"),
        "status": manifest.get("status"),
        "protocol_config": protocol_config,
        "experiment_config": experiment_config,
        "dataset": {
            key: manifest.get("dataset", {}).get(key)
            for key in (
                "canonical_dataset_manifest_sha256",
                "class_map_sha256",
                "val_annotation_sha256",
                "val_image_list_sha256",
            )
        },
        "protocol": {
            key: protocol.get(key)
            for key in (
                "epochs",
                "batch",
                "imgsz",
                "seed",
                "amp",
                "nms_iou",
                "common_operating_confidence",
                "max_detections_for_coco_ap",
            )
        },
    }


def _require_inside_project(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Formal {label} must be inside the repository: {resolved}") from error
    return resolved


def main() -> int:
    args = parse_args()
    if args.batch != 1:
        raise ValueError("This audited deployment profile is fixed to batch=1")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.opset < 11:
        raise ValueError("--opset must be at least 11")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_suffix = ".pt" if args.framework == "yolo11" else ".pth"
    if checkpoint.suffix.lower() != expected_suffix:
        raise ValueError(f"{args.framework} checkpoint must use {expected_suffix}: {checkpoint}")

    output_dir = args.output_dir.resolve()
    trained_root = (PROJECT_ROOT / "weights" / "trained").resolve()
    if args.diagnostic:
        try:
            output_dir.relative_to(trained_root)
        except ValueError:
            pass
        else:
            raise ValueError("Diagnostic exports cannot be written under weights/trained")
    elif args.comparison_dir is None:
        raise ValueError("Formal deployment export requires --comparison-dir; use --diagnostic only for smoke checks")
    else:
        if args.force:
            raise ValueError("--force is diagnostic-only; formal deployment artifacts are immutable")
        for value, label in (
            (checkpoint, "checkpoint"),
            (args.run_manifest, "run manifest"),
            (args.comparison_dir, "comparison"),
            (args.coco_annotations, "COCO annotation"),
            (args.coco_images, "COCO image root"),
            (output_dir, "output directory"),
        ):
            _require_inside_project(Path(value), label)
        try:
            output_dir.relative_to(trained_root)
        except ValueError as error:
            raise ValueError("Formal deployment artifacts must be written under weights/trained") from error
        formal_values = {
            "absolute_tolerance": (float(args.absolute_tolerance), FORMAL_ABSOLUTE_TOLERANCE),
            "relative_tolerance": (float(args.relative_tolerance), FORMAL_RELATIVE_TOLERANCE),
            "confidence": (float(args.confidence), FORMAL_CONFIDENCE),
            "nms_iou": (float(args.nms_iou), FORMAL_NMS_IOU),
            "max_detections": (int(args.max_detections), FORMAL_MAX_DETECTIONS),
        }
        mismatches = [
            f"{name}={actual} (required {expected})"
            for name, (actual, expected) in formal_values.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError("Formal export profile cannot be relaxed: " + ", ".join(mismatches))
        if args.allow_class_name_mismatch:
            raise ValueError("--allow-class-name-mismatch is diagnostic-only")
        if args.framework == "yolox":
            _require_inside_project(args.yolox_config, "YOLOX experiment config")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_name or checkpoint.stem
    if Path(stem).name != stem or stem in {"", ".", ".."}:
        raise ValueError("--output-name must be a plain file stem")
    final_onnx = output_dir / f"{stem}.onnx"
    partial_onnx = output_dir / f".{stem}.partial.onnx"
    failed_onnx = output_dir / f"{stem}.verification-failed.onnx"
    metadata_path = output_dir / f"{stem}.deployment.json"
    _clean_previous([final_onnx, partial_onnx, failed_onnx, metadata_path], args.force)

    sample = load_coco_sample(args.coco_annotations, args.coco_images, args.image_id)
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode COCO image: {sample.image_path}")
    if (image.shape[1], image.shape[0]) != (sample.width, sample.height):
        raise ValueError(
            "COCO image dimensions do not match decoded file: "
            f"record={sample.width}x{sample.height}, decoded={image.shape[1]}x{image.shape[0]}"
        )
    input_array, preprocessing_observed = preprocess_image(image, args.framework, args.imgsz)
    preprocessing = preprocessing_spec(args.framework, args.imgsz)
    coco_names = category_names(sample)
    checkpoint_record = artifact_record(checkpoint)
    run_manifest = _verify_run_manifest(
        args.run_manifest,
        checkpoint_sha256=checkpoint_record["sha256"],
        framework=args.framework,
        image_size=args.imgsz,
    )
    if not args.diagnostic and run_manifest.get("stage") == "smoke_not_comparable":
        raise ValueError("Smoke runs cannot be exported as formal deployment artifacts")
    expected_val_sha256 = str(run_manifest.get("dataset", {}).get("val_annotation_sha256") or "")
    actual_annotation_sha256 = sample.evidence()["annotation_sha256"]
    if not expected_val_sha256 or expected_val_sha256.lower() != actual_annotation_sha256.lower():
        raise ValueError(
            "Export verification must use the run's exact validation annotation: "
            f"manifest={expected_val_sha256 or 'MISSING'}, actual={actual_annotation_sha256}"
        )
    comparison_evidence = None
    if not args.diagnostic:
        comparison_evidence = validate_comparison_for_run(
            args.comparison_dir,
            run_id=str(run_manifest["run_id"]),
            run_manifest_path=args.run_manifest.resolve(),
        )
        protocol_values = run_manifest.get("protocol", {})
        if float(protocol_values.get("nms_iou", -1)) != FORMAL_NMS_IOU:
            raise ValueError("Run manifest NMS IoU does not match the formal deployment profile")
        if float(protocol_values.get("common_operating_confidence", -1)) != FORMAL_CONFIDENCE:
            raise ValueError("Run manifest operating confidence does not match the formal deployment profile")
        if args.framework == "yolox":
            experiment_config = run_manifest.get("experiment_config")
            if not isinstance(experiment_config, dict):
                raise ValueError("YOLOX run manifest is missing experiment_config evidence")
            expected_config_sha256 = str(experiment_config.get("sha256") or "").lower()
            actual_config_sha256 = sha256_file(args.yolox_config.resolve()).lower()
            if expected_config_sha256 != actual_config_sha256:
                raise ValueError(
                    "YOLOX export config differs from the experiment config frozen by the run: "
                    f"manifest={expected_config_sha256 or 'MISSING'}, actual={actual_config_sha256}"
                )

    status = "FAIL"
    failure: str | None = None
    deployment: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "profile": "fixed_batch1_fp32_onnx",
        "framework": args.framework,
        "model_input": {
            "batch": 1,
            "channels": 3,
            "height": args.imgsz,
            "width": args.imgsz,
            "dtype": "float32",
            "name": "images",
        },
        "classes": {"count": len(coco_names), "names": coco_names, "source": "COCO categories by ID"},
        "preprocessing": preprocessing,
        "postprocessing": {
            "confidence": args.confidence,
            "nms_iou": args.nms_iou,
            "class_agnostic_nms": False,
            "max_detections": args.max_detections,
        },
        "export": {"opset": args.opset, "dynamic_axes": False, "nms_embedded": False},
        "verification_image": sample.evidence() | {"preprocessing_observed": preprocessing_observed.to_dict()},
        "artifacts": {"checkpoint": checkpoint_record | {"file_name": checkpoint.name}},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "export_device": "cpu",
        },
    }
    deployment["training_run"] = {key: value for key, value in run_manifest.items() if key != "artifact"}
    deployment["artifacts"]["run_manifest"] = run_manifest["artifact"]
    deployment["release_validation"] = (
        {"status": "PASS", "formal_release": True, **comparison_evidence}
        if comparison_evidence is not None
        else {"status": "DIAGNOSTIC_ONLY", "formal_release": False}
    )
    try:
        if args.framework == "yolo11":
            native, checkpoint_names, framework_metadata = _export_yolo11(
                checkpoint, partial_onnx, input_array, args.opset
            )
            if len(checkpoint_names) != len(coco_names):
                raise ValueError(
                    "Checkpoint/COCO class count mismatch: "
                    f"checkpoint={len(checkpoint_names)}, COCO={len(coco_names)}"
                )
            if checkpoint_names != coco_names and not args.allow_class_name_mismatch:
                raise ValueError(
                    "Checkpoint/COCO class name mismatch: "
                    f"checkpoint={checkpoint_names}, COCO={coco_names}. "
                    "Use --allow-class-name-mismatch only for a diagnostic export."
                )
            deployment["classes"]["checkpoint_names"] = checkpoint_names
            deployment["classes"]["name_match"] = checkpoint_names == coco_names
        else:
            native, framework_metadata = _export_yolox(
                checkpoint,
                args.yolox_config,
                partial_onnx,
                input_array,
                args.opset,
                args.imgsz,
                len(coco_names),
            )
        deployment["framework_metadata"] = framework_metadata

        onnx_version, graph_inputs, graph_outputs, graph_opset = _embed_onnx_metadata(
            partial_onnx,
            framework=args.framework,
            image_size=args.imgsz,
            class_names=coco_names,
            checkpoint_sha256=checkpoint_record["sha256"],
            preprocessing=preprocessing,
            output_semantics=framework_metadata["output_semantics"],
        )
        exported, runtime = _run_onnx(partial_onnx, input_array)
        numeric = compare_arrays(
            native,
            exported,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        native_summary = _detection_summary(
            native,
            framework=args.framework,
            class_count=len(coco_names),
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            max_detections=args.max_detections,
            preprocess_info=preprocessing_observed,
        )
        onnx_summary = _detection_summary(
            exported,
            framework=args.framework,
            class_count=len(coco_names),
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            max_detections=args.max_detections,
            preprocess_info=preprocessing_observed,
        )
        deployment["export"].update(
            {
                "onnx_version": onnx_version,
                "graph_opset": int(graph_opset),
                "inputs": graph_inputs,
                "outputs": graph_outputs,
            }
        )
        deployment["verification"] = {
            "status": numeric["status"],
            "gate": "all raw output elements finite and within atol + rtol*abs(native)",
            "numeric": numeric,
            "runtime": runtime,
            "native_post_nms": native_summary,
            "onnx_post_nms": onnx_summary,
            "post_nms_count_match": native_summary["count"] == onnx_summary["count"],
        }
        if numeric["status"] != "PASS":
            partial_onnx.replace(failed_onnx)
            deployment["artifacts"]["onnx_failed_verification"] = artifact_record(failed_onnx) | {
                "file_name": failed_onnx.name
            }
            failure = "Native/ONNX numerical verification failed"
        else:
            partial_onnx.replace(final_onnx)
            deployment["artifacts"]["onnx"] = artifact_record(final_onnx) | {
                "file_name": final_onnx.name
            }
            status = "PASS"
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        if partial_onnx.exists():
            partial_onnx.replace(failed_onnx)
            deployment["artifacts"]["onnx_failed_verification"] = artifact_record(failed_onnx) | {
                "file_name": failed_onnx.name
            }
    deployment["status"] = status
    if failure:
        deployment["failure"] = failure
    write_json(metadata_path, deployment)

    print(f"DEPLOYMENT EXPORT: {status}")
    print(f"framework       {args.framework}")
    print(f"checkpoint      {checkpoint_record['bytes']:>12,} bytes  {checkpoint_record['sha256']}")
    artifact = deployment["artifacts"].get("onnx") or deployment["artifacts"].get(
        "onnx_failed_verification"
    )
    if artifact:
        print(f"onnx            {artifact['bytes']:>12,} bytes  {artifact['sha256']}")
    verification = deployment.get("verification", {}).get("numeric")
    if verification:
        print(f"output shape    {verification['native_shape']}")
        print(f"max abs error   {verification.get('max_absolute_error')}")
        print(f"within tol      {verification.get('within_tolerance_fraction')}")
    print(f"COCO image      id={sample.image_id} sha256={sample.evidence()['image_sha256']}")
    print(f"metadata        {metadata_path}")
    if failure:
        print(f"failure         {failure}", file=sys.stderr)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
