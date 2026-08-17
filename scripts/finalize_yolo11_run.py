from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ultralytics import YOLO

from mcu_data.common import write_json
from mcu_data.reporting import compare_runs, evaluate_predictions
from mcu_data.runlog import checkpoint_file_record, print_section, tee_console, utc_now_precise
from train_yolo11_logged import _benchmark, _export_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume evaluation/reporting from an existing YOLO11 checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--predict-batch", type=int, default=8)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint or run_dir / "native" / "weights" / "best.pt"
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    adapter_args = SimpleNamespace(
        imgsz=args.imgsz,
        predict_batch=args.predict_batch,
        fp32=args.fp32,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_iterations=args.benchmark_iterations,
    )
    annotation = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "micropcb_rpi_coco"
        / "annotations"
        / "instances_val2017.json"
    )
    image_root = PROJECT_ROOT / "data" / "processed" / "micropcb_rpi_coco" / "val2017"
    document = json.loads(annotation.read_text(encoding="utf-8"))
    benchmark_image = image_root / document["images"][0]["file_name"]
    with tee_console(run_dir / "terminal.log"):
        print("\nRESUME YOLO11 FINAL EVALUATION (NO RETRAINING)")
        print("=" * 72)
        print_section("CHECKPOINT", checkpoint_file_record(checkpoint))
        model = YOLO(str(checkpoint))
        _export_predictions(model, annotation, image_root, adapter_args, run_dir / "predictions.coco.json")
        evaluate_predictions(annotation, run_dir / "predictions.coco.json", run_dir)
        _benchmark(model, benchmark_image, adapter_args, run_dir)
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "complete"
        manifest["finalized_utc"] = utc_now_precise()
        manifest["best_checkpoint"] = checkpoint_file_record(checkpoint)
        write_json(manifest_path, manifest)
        compare_runs([run_dir], run_dir / "plots" / "summary")
        print(f"\nFINALIZATION PASS: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
