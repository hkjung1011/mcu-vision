from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
import onnxruntime as ort

from mcu_data.common import sha256_file, write_json
from mcu_data.deployment import decode_detections, preprocess_image, restore_boxes
from mcu_data.deployment_publishing import validate_promoted_deployment_for_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an audited mcu-vision ONNX deployment artifact on an Ubuntu/Windows camera."
    )
    parser.add_argument("--metadata", type=Path, required=True, help="*.deployment.json from export_deployment.py")
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help=(
            "Required for formal camera runs: "
            "reports/deployments/<release>/deployment_release_manifest.json"
        ),
    )
    parser.add_argument("--camera", default="0", help="OpenCV camera index or video/stream path")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--confidence", type=float, help="Override the release confidence threshold")
    parser.add_argument("--nms-iou", type=float, help="Override the release NMS IoU")
    parser.add_argument("--width", type=int, help="Requested camera capture width")
    parser.add_argument("--height", type=int, help="Requested camera capture height")
    parser.add_argument("--warmup", type=int, default=10, help="Frames excluded from latency statistics")
    parser.add_argument("--max-frames", type=int, help="Stop after this many successfully captured frames")
    parser.add_argument(
        "--min-measured-frames",
        type=int,
        default=30,
        help="Minimum post-warmup frames required for a PASS latency report",
    )
    parser.add_argument("--no-display", action="store_true", help="Benchmark without opening a GUI window")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Explicitly allow DIAGNOSTIC_ONLY export metadata; result cannot be a release PASS",
    )
    parser.add_argument("--output-json", type=Path, help="Write the final machine-readable summary to this file")
    return parser.parse_args()


def _load_metadata(path: Path, *, diagnostic: bool) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise ValueError(f"Deployment metadata is missing or not PASS: {path}")
    if value.get("schema_version") != 1:
        raise ValueError(f"Unsupported deployment schema: {value.get('schema_version')}")
    release_validation = value.get("release_validation", {})
    formal = (
        isinstance(release_validation, dict)
        and release_validation.get("status") == "PASS"
        and release_validation.get("formal_release") is True
    )
    diagnostic_only = (
        isinstance(release_validation, dict)
        and release_validation.get("status") == "DIAGNOSTIC_ONLY"
        and release_validation.get("formal_release") is False
    )
    if not formal and not (diagnostic and diagnostic_only):
        raise ValueError(
            "Camera runner requires formal release metadata. Use --diagnostic only for an explicitly "
            "DIAGNOSTIC_ONLY export."
        )
    return value


def _resolve_onnx(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    record = metadata["artifacts"]["onnx"]
    sibling = metadata_path.parent / record["file_name"]
    candidate = sibling if sibling.is_file() else Path(record["path"])
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    actual_sha = sha256_file(candidate)
    if actual_sha.lower() != str(record["sha256"]).lower():
        raise ValueError(
            f"ONNX SHA-256 mismatch: expected={record['sha256']}, actual={actual_sha}, path={candidate}"
        )
    return candidate


def _camera_source(value: str) -> int | str:
    stripped = value.strip()
    return int(stripped) if stripped.isdecimal() else stripped


def _draw(frame: np.ndarray, detections: np.ndarray, names: list[str]) -> None:
    for x1, y1, x2, y2, score, class_id_value in detections:
        class_id = int(class_id_value)
        label_name = names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
        label = f"{label_name} {score:.3f}"
        point1 = (int(round(x1)), int(round(y1)))
        point2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(frame, point1, point2, (0, 220, 0), 2)
        cv2.putText(
            frame,
            label,
            (point1[0], max(18, point1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _repository_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.min_measured_frames <= 0:
        raise ValueError("--min-measured-frames must be positive")
    if args.max_frames is not None and args.max_frames - args.warmup < args.min_measured_frames:
        raise ValueError(
            "--max-frames must allow at least --min-measured-frames after warmup: "
            f"max_frames={args.max_frames}, warmup={args.warmup}, minimum={args.min_measured_frames}"
        )
    metadata_path = args.metadata.resolve()
    metadata = _load_metadata(metadata_path, diagnostic=args.diagnostic)
    formal_release = metadata["release_validation"].get("formal_release") is True
    if formal_release and args.release_manifest is None:
        raise ValueError(
            "Formal camera runs require --release-manifest from promote_deployment.py"
        )
    onnx_path = _resolve_onnx(metadata_path, metadata)
    release_evidence: dict[str, Any]
    if formal_release:
        assert args.release_manifest is not None
        release_evidence = validate_promoted_deployment_for_runtime(
            project_root=PROJECT_ROOT,
            release_manifest_path=args.release_manifest,
            deployment_metadata_path=metadata_path,
            onnx_path=onnx_path,
        )
    else:
        release_evidence = {
            "status": "DIAGNOSTIC_ONLY",
            "reason": "promoted release manifest is not required for explicit diagnostic metadata",
        }

    available = ort.get_available_providers()
    if args.provider not in available:
        raise ValueError(f"Requested provider {args.provider!r} is unavailable; available={available}")
    session = ort.InferenceSession(str(onnx_path), providers=[args.provider])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Expected an ONNX model with exactly one input and one output")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    framework = str(metadata["framework"])
    if framework not in {"yolo11", "yolox"}:
        raise ValueError(f"Unsupported framework in deployment metadata: {framework}")
    model_input = metadata["model_input"]
    if int(model_input["batch"]) != 1 or int(model_input["height"]) != int(model_input["width"]):
        raise ValueError("Camera runner requires a fixed square batch-1 deployment model")
    image_size = int(model_input["height"])
    names = [str(item) for item in metadata["classes"]["names"]]
    postprocessing = metadata["postprocessing"]
    confidence = float(args.confidence if args.confidence is not None else postprocessing["confidence"])
    nms_iou = float(args.nms_iou if args.nms_iou is not None else postprocessing["nms_iou"])
    max_detections = int(postprocessing["max_detections"])

    capture = cv2.VideoCapture(_camera_source(args.camera))
    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {args.camera}")

    latencies_ms: list[float] = []
    inference_ms: list[float] = []
    frame_number = 0
    started = time.perf_counter()
    try:
        while True:
            frame_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                print("capture ended or returned no frame", file=sys.stderr)
                break
            frame_number += 1
            input_array, transform = preprocess_image(frame, framework, image_size)  # type: ignore[arg-type]
            infer_start = time.perf_counter()
            output = session.run([output_name], {input_name: input_array})[0]
            infer_end = time.perf_counter()
            detections = decode_detections(
                output,
                framework=framework,  # type: ignore[arg-type]
                class_count=len(names),
                confidence=confidence,
                nms_iou=nms_iou,
                max_detections=max_detections,
            )
            detections = restore_boxes(detections, transform)
            frame_end = time.perf_counter()
            if frame_number > args.warmup:
                latencies_ms.append((frame_end - frame_start) * 1000)
                inference_ms.append((infer_end - infer_start) * 1000)

            if not args.no_display:
                _draw(frame, detections, names)
                cv2.putText(
                    frame,
                    f"E2E {(frame_end - frame_start) * 1000:.1f} ms | det {len(detections)}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("mcu-vision ONNX", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
            if frame_number % 30 == 0:
                print(
                    f"frame={frame_number} detections={len(detections)} "
                    f"e2e_ms={(frame_end - frame_start) * 1000:.2f} "
                    f"inference_ms={(infer_end - infer_start) * 1000:.2f}",
                    flush=True,
                    file=sys.stderr,
                )
            if args.max_frames is not None and frame_number >= args.max_frames:
                break
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    enough_measurements = len(latencies_ms) >= args.min_measured_frames
    result_status = "FAIL"
    if enough_measurements:
        result_status = "PASS" if formal_release else "DIAGNOSTIC_ONLY"
    summary = {
        "status": result_status,
        "scope": "runtime measurement only; accuracy/safety acceptance is not evaluated here",
        "acceptance_status": "NOT_EVALUATED",
        "formal_release_metadata": formal_release,
        "release_evidence": release_evidence,
        "onnx": _repository_path(onnx_path),
        "onnx_sha256": metadata["artifacts"]["onnx"]["sha256"],
        "framework": framework,
        "provider": session.get_providers()[0],
        "frames_total": frame_number,
        "warmup_frames_excluded": min(frame_number, args.warmup),
        "measured_frames": len(latencies_ms),
        "minimum_measured_frames_required": args.min_measured_frames,
        "wall_seconds": elapsed,
        "capture_loop_fps": frame_number / elapsed if elapsed > 0 else None,
        "e2e_ms_p50": _percentile(latencies_ms, 50),
        "e2e_ms_p95": _percentile(latencies_ms, 95),
        "inference_ms_p50": _percentile(inference_ms, 50),
        "inference_ms_p95": _percentile(inference_ms, 95),
        "confidence": confidence,
        "nms_iou": nms_iou,
    }
    if args.output_json:
        write_json(args.output_json.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result_status in {"PASS", "DIAGNOSTIC_ONLY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
