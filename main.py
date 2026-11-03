"""Single-image local runner for the NOX five-stage inspection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from pipeline.router import PipelineRouter
from utils.config_loader import load_config
from utils.overlay import draw_aoi_overlay


def main() -> None:
    """Inspect one image, print stage latencies, and save an annotated image."""
    parser = argparse.ArgumentParser(description="Run local NOX image inspection")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--product", default="prismatic_cell")
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Inspection image not found: {args.image}")
    config = load_config(args.product)
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not decode image: {args.image}")

    router = PipelineRouter(config)
    processed, decision, timings = router.run(frame, "station_01", "cam_01")
    output_dir = Path(config["runtime"]["data_dir"]) / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.image.stem}_annotated{args.image.suffix or '.jpg'}"
    annotated = draw_aoi_overlay(processed.copy(), decision["detections"])
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(f"Could not write annotated image: {output_path}")

    print(f"Product: {config['product_id']}")
    print(f"Result: {decision['result']} ({decision['severity']})")
    print(f"Defect: {decision['defect']} | Confidence: {decision['confidence']:.3f}")
    print(f"Anomaly score: {decision['anomaly_score']:.3f}")
    print(f"{'Stage':<26} {'Latency (ms)':>14}")
    for stage, latency in timings.items():
        print(f"{stage:<26} {latency:>14.3f}")
    print(f"Annotated image: {output_path}")


if __name__ == "__main__":
    main()
