"""Flask API for synchronous NOX five-stage image inspection."""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request

from pipeline.router import PipelineRouter
from utils.config_loader import load_config
from utils.overlay import draw_aoi_overlay


def create_app() -> Flask:
    """Create the Flask app and its upload route without loading model weights."""
    app = Flask(__name__)
    app.config["PIPELINE_ROUTER"] = None
    app.config["PRODUCT_CONFIG"] = None

    @app.get("/")
    def index() -> str:
        """Return a short health response."""
        return "NOX Inspection Server is Running"

    @app.post("/upload")
    def upload() -> tuple[Any, int]:
        """Decode an image, inspect it, save artifacts, log timings, and return JSON."""
        router: PipelineRouter | None = app.config["PIPELINE_ROUTER"]
        config: dict[str, Any] | None = app.config["PRODUCT_CONFIG"]
        if router is None or config is None:
            return jsonify({"error": "inspection pipeline is not initialized"}), 503

        image_bytes = request.get_data()
        if not image_bytes:
            return jsonify({"error": "request body contains no image"}), 400
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "request body is not a supported image"}), 400

        station_id = request.args.get(
            "station_id", request.headers.get("X-Station-ID", "station_01")
        )
        camera_id = request.args.get(
            "camera_id", request.headers.get("X-Camera-ID", "cam_01")
        )
        data_dir = Path(config["runtime"]["data_dir"])
        directories = {
            "original": data_dir / "original",
            "annotated": data_dir / "annotated",
            "pass": data_dir / "pass",
            "fail": data_dir / "fail",
            "logs": data_dir / "logs",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        image_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        original_path = directories["original"] / f"{image_id}.jpg"
        if not cv2.imwrite(str(original_path), frame):
            return jsonify({"error": "could not save original image"}), 500

        processed, decision, timings = router.run(frame, station_id, camera_id)
        annotated_path = directories["annotated"] / f"{image_id}.jpg"
        annotated = draw_aoi_overlay(processed.copy(), decision["detections"])
        if not cv2.imwrite(str(annotated_path), annotated):
            return jsonify({"error": "could not save annotated image"}), 500
        shutil.copy2(
            original_path,
            directories[decision["result"].lower()] / original_path.name,
        )

        csv_path = directories["logs"] / "inspection.csv"
        columns = [
            "ImageID",
            "Timestamp",
            "ProductID",
            "StationID",
            "CameraID",
            "Result",
            "Defect",
            "Confidence",
            "AnomalyScore",
            "Stage1(ms)",
            "Stage2(ms)",
            "Stage3(ms)",
            "Stage4(ms)",
            "Stage5(ms)",
            "Total(ms)",
        ]
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "ImageID": image_id,
                    "Timestamp": timestamp,
                    "ProductID": config["product_id"],
                    "StationID": station_id,
                    "CameraID": camera_id,
                    "Result": decision["result"],
                    "Defect": decision["defect"],
                    "Confidence": decision["confidence"],
                    "AnomalyScore": decision["anomaly_score"],
                    "Stage1(ms)": round(timings["stage_1_preprocessing_ms"], 3),
                    "Stage2(ms)": round(timings["stage_2_detection_ms"], 3),
                    "Stage3(ms)": round(timings["stage_3_classification_ms"], 3),
                    "Stage4(ms)": round(timings["stage_4_anomaly_ms"], 3),
                    "Stage5(ms)": round(timings["stage_5_decision_ms"], 3),
                    "Total(ms)": round(timings["total_ms"], 3),
                }
            )

        response = {
            "image_id": image_id,
            "product_id": config["product_id"],
            "result": decision["result"],
            "defect": decision["defect"],
            "confidence": decision["confidence"],
            "severity": decision["severity"],
            "station_id": station_id,
            "camera_id": camera_id,
            "anomaly": decision["anomaly"],
            "detections": [
                {
                    key: detection[key]
                    for key in (
                        "label",
                        "class_id",
                        "confidence",
                        "bbox",
                        "defect_type",
                        "cnn_confidence",
                        "severity",
                    )
                    if key in detection
                }
                for detection in decision["detections"]
            ],
            "timings_ms": timings,
            "annotated_image": str(annotated_path),
        }
        stage_timings = " ".join(
            f"S{stage}={timings[f'stage_{stage}_{name}_ms']:.2f}ms"
            for stage, name in (
                (1, "preprocessing"),
                (2, "detection"),
                (3, "classification"),
                (4, "anomaly"),
                (5, "decision"),
            )
        )
        print(
            f"[{timestamp}] {decision['result']} | {decision['defect']} | "
            f"{stage_timings} total={timings['total_ms']:.2f}ms"
        )
        return jsonify(response), 200

    return app


def main() -> None:
    """Load product config and all strict models, then serve the upload API."""
    parser = argparse.ArgumentParser(description="NOX battery-cell inspection API")
    parser.add_argument("--product", default="prismatic_cell")
    args = parser.parse_args()
    config = load_config(args.product)
    router = PipelineRouter(config)
    app = create_app()
    app.config["PRODUCT_CONFIG"] = config
    app.config["PIPELINE_ROUTER"] = router
    app.run(host="0.0.0.0", port=5000, debug=False)


app = create_app()


if __name__ == "__main__":
    main()
