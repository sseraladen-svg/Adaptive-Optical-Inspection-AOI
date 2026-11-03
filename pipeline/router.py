"""Synchronous five-stage inspection pipeline router."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from stage_1_preprocessing.preprocessor import Preprocessor
from stage_2_detection.detector import Detector
from stage_3_classification.classifier import Classifier
from stage_4_anomaly.anomaly_detector import AnomalyDetector
from stage_5_decision.decision_engine import DecisionEngine


class PipelineRouter:
    """Initialize S1-S5 once and execute every frame sequentially."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Create every stage; configured model errors fail during initialization."""
        self.preprocessor = Preprocessor(config)
        self.detector = Detector(config)
        self.classifier = Classifier(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.decision_engine = DecisionEngine(config)

    def run(
        self,
        frame: np.ndarray,
        station_id: str,
        camera_id: str,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
        """Run S1→S2→optional S3→S4→S5 and return processed image and timings.

        Args:
            frame: Input uint8 BGR numpy image.
            station_id: Identifier included in the decision result.
            camera_id: Identifier included in the decision result.

        Returns:
            A tuple of the preprocessed frame, final decision mapping (including
            a copy of the timing mapping), and per-stage/total latency in ms.
        """
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        stage_start = time.perf_counter()
        processed = self.preprocessor.process(frame)
        timings["stage_1_preprocessing_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000

        stage_start = time.perf_counter()
        detections = self.detector.detect(processed)
        timings["stage_2_detection_ms"] = (time.perf_counter() - stage_start) * 1000
        if not isinstance(detections, list):
            raise TypeError("Stage 2 detector must return a list of detections")

        stage_start = time.perf_counter()
        classified_detections = (
            self.classifier.classify_all(processed, detections) if detections else []
        )
        timings["stage_3_classification_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000
        if len(classified_detections) != len(detections):
            raise ValueError("Stage 3 result count does not match Stage 2 detections")

        stage_start = time.perf_counter()
        anomaly = self.anomaly_detector.predict(processed)
        timings["stage_4_anomaly_ms"] = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        decision = self.decision_engine.decide(
            classified_detections,
            anomaly,
            station_id=station_id,
            camera_id=camera_id,
        )
        timings["stage_5_decision_ms"] = (time.perf_counter() - stage_start) * 1000
        timings["total_ms"] = (time.perf_counter() - total_start) * 1000
        decision["timings"] = dict(timings)
        return processed, decision, timings
