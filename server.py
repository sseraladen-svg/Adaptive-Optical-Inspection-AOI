import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request

from pipeline.router import PipelineRouter
from utils.config_loader import load_config
from utils.overlay import draw_aoi_overlay

parser = argparse.ArgumentParser()
parser.add_argument(
    "--product",
    default="prismatic_cell",
    help="Product config id under config/",
)
args, _ = parser.parse_known_args()
CONFIG = load_config(args.product)
router: PipelineRouter | None = None

app = Flask(__name__)
DATA_DIR = Path("data")
ORIGINAL_DIR = DATA_DIR / "original"
ANNOTATED_DIR = DATA_DIR / "annotated"
PASS_DIR = DATA_DIR / "pass"
FAIL_DIR = DATA_DIR / "fail"
LOG_DIR = DATA_DIR / "logs"
for directory in (ORIGINAL_DIR, ANNOTATED_DIR, PASS_DIR, FAIL_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CSV_FILE = LOG_DIR / "inspection.csv"
if not CSV_FILE.exists():
    with CSV_FILE.open("w", newline="", encoding="utf-8") as log_file:
        csv.writer(log_file).writerow(
            [
                "ImageID",
                "Timestamp",
                "ProductID",
                "StationID",
                "CameraID",
                "Result",
                "Severity",
                "DefectType",
                "Confidence",
                "Preprocessing(ms)",
                "Detection(ms)",
                "Classification(ms)",
                "Stage 4 Anomaly(ms)",
                "Stage 5 Decision(ms)",
                "Total(ms)",
                "OriginalImage",
                "AnnotatedImage",
            ]
        )

print(f"[NOX Inspection] Loaded product config: {CONFIG['product_id']}")
DEFAULT_STATION_ID = "station_01"
DEFAULT_CAMERA_ID = "cam_01"


@app.route("/")
def index():
    return "NOX Inspection Server is Running! Use POST /upload to send images."


@app.route("/upload", methods=["POST"])
def upload():
    if router is None:
        return "Inspection pipeline is not initialized", 503

    image_bytes = request.data
    if not image_bytes:
        return "No Image", 400

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return "Invalid Image", 400

    station_id = request.args.get(
        "station_id", request.headers.get("X-Station-ID", DEFAULT_STATION_ID)
    )
    camera_id = request.args.get(
        "camera_id", request.headers.get("X-Camera-ID", DEFAULT_CAMERA_ID)
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    original_path = ORIGINAL_DIR / f"{image_id}.jpg"
    if not cv2.imwrite(str(original_path), frame):
        return "Could not save input image", 500

    decision, timings = router.run(frame, station_id, camera_id)
    processed = decision.pop("_processed_frame")
    annotated_frame = draw_aoi_overlay(processed, decision["detections"])
    annotated_path = ANNOTATED_DIR / f"{image_id}.jpg"
    if not cv2.imwrite(str(annotated_path), annotated_frame):
        return "Could not save annotated image", 500

    result = decision["result"]
    defect = decision["defect"]
    confidence = decision["confidence"]
    shutil.copy2(
        original_path,
        (FAIL_DIR if result == "FAIL" else PASS_DIR) / f"{image_id}.jpg",
    )
    with CSV_FILE.open("a", newline="", encoding="utf-8") as log_file:
        csv.writer(log_file).writerow(
            [
                image_id,
                timestamp,
                CONFIG["product_id"],
                station_id,
                camera_id,
                result,
                decision["severity"],
                defect,
                confidence,
                round(timings["preprocessing_ms"], 2),
                round(timings["detection_ms"], 2),
                round(timings["classification_ms"], 2),
                round(timings["stage_4_anomaly_ms"], 2),
                round(timings["stage_5_decision_ms"], 2),
                round(timings["total_ms"], 2),
                str(original_path),
                str(annotated_path),
            ]
        )

    print(
        f"[{timestamp}] {result} ({decision['severity']}) | "
        f"Station/Camera: {station_id}/{camera_id} | Defect: {defect} | "
        f"Confidence: {confidence:.2f} | "
        f"S1: {timings['preprocessing_ms']:.2f}ms | "
        f"S2: {timings['detection_ms']:.2f}ms | "
        f"S3: {timings['classification_ms']:.2f}ms | "
        f"S4: {timings['stage_4_anomaly_ms']:.2f}ms | "
        f"S5: {timings['stage_5_decision_ms']:.2f}ms | "
        f"Total: {timings['total_ms']:.2f}ms"
    )
    return result, 200


if __name__ == "__main__":
    router = PipelineRouter(CONFIG)
    print(f"[NOX Inspection] Starting server for {CONFIG['product_id']}")
    app.run(host="0.0.0.0", port=5000, debug=False)
