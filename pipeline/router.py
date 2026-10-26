from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from stage_1_preprocessing.preprocessor import Preprocessor
from stage_2_detection.detector import Detector
from stage_3_classification.classifier import Classifier
from stage_4_anomaly.anomaly_detector import AnomalyDetector
from stage_5_decision.decision_engine import DecisionEngine


class PipelineRouter:
    """Initialize the inspection stages once and route each frame through them."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        models = config["models"]
        class_config = config.get("classification", {})
        anomaly_config = config.get("anomaly_detection", {})

        self.preprocessor = Preprocessor(config)
        self.detector = Detector(config)

        classifier_model = Path(
            class_config.get(
                "onnx_model",
                models.get("cnn_classifier_onnx", models["cnn_classifier"]),
            )
        )
        if classifier_model.suffix.lower() in {".pt", ".pth"}:
            classifier_model = classifier_model.with_suffix(".onnx")
        classes = config["defect_classes"]
        class_names = (
            [classes[key] for key in sorted(classes)]
            if isinstance(classes, dict)
            else classes
        )
        self.classifier = Classifier(
            model_path=classifier_model,
            class_names=class_names,
            input_size=int(class_config.get("input_size", 224)),
        )

        anomaly_model = Path(
            anomaly_config.get("memory_bank_path", models["anomaly"])
        )
        if anomaly_model.suffix.lower() not in {".npy", ".npz"} and not anomaly_model.is_dir():
            anomaly_model = anomaly_model.parent / "patchcore_model"
        self.anomaly_detector = AnomalyDetector(anomaly_model)
        self.anomaly_threshold = float(
            anomaly_config.get(
                "threshold",
                config["decision_rules"].get("anomaly_threshold", 0.5),
            )
        )
        self.decision_engine = DecisionEngine(config)

    def run(
        self, frame: Any, station_id: str, camera_id: str
    ) -> tuple[dict[str, Any], dict[str, float]]:
        total_start = time.perf_counter()

        start = time.perf_counter()
        processed = self.preprocessor.process(frame)
        preprocessing_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        _, detections = self.detector.detect(processed)
        detection_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        if detections:
            classifications = self.classifier.classify_all(processed, detections)
            if len(classifications) != len(detections):
                raise ValueError("Classifier result count does not match detection count")
            for detection, classification in zip(detections, classifications):
                detection["detection_confidence"] = detection["confidence"]
                detection["defect_type"] = classification["defect_type"]
                detection["classification_confidence"] = classification["confidence"]
                detection["confidence"] = classification["confidence"]
                detection["label"] = classification["defect_type"]
        classification_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        anomaly_score = self.anomaly_detector.predict(processed)
        anomaly_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        decision = self.decision_engine.decide(
            detections,
            station_id=station_id,
            camera_id=camera_id,
            anomaly_score=anomaly_score,
            anomaly_threshold=self.anomaly_threshold,
        )
        decision["_processed_frame"] = processed
        decision_ms = (time.perf_counter() - start) * 1000

        timings = {
            "preprocessing_ms": preprocessing_ms,
            "detection_ms": detection_ms,
            "classification_ms": classification_ms,
            "stage_4_anomaly_ms": anomaly_ms,
            "stage_5_decision_ms": decision_ms,
            "total_ms": (time.perf_counter() - total_start) * 1000,
        }
        return decision, timings
