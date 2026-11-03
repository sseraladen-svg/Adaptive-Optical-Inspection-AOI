"""Config-driven aggregation of detector, classifier, and anomaly results."""

from __future__ import annotations

from typing import Any


class DecisionEngine:
    """Apply configured confidence thresholds and severity labels."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Read all decision thresholds and labels from the product config."""
        self.fail_confidence_threshold = float(
            config["decision_rules"]["fail_confidence_threshold"]
        )
        self.severity_map = config["decision_rules"]["severity_map"]
        self.anomaly_threshold = float(config["anomaly_detection"]["threshold"])

    def evaluate_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        """Copy a detection and attach its configured severity and CNN score."""
        if "label" not in detection or "confidence" not in detection:
            raise ValueError("each detection must contain label and confidence")
        label = detection.get("defect_type", detection["label"])
        severity = self.severity_map.get(label)
        if severity not in {"WARN", "FAIL"}:
            raise ValueError(f"No valid configured severity for defect class {label!r}")
        return {
            **detection,
            "defect_type": label,
            "cnn_confidence": float(detection.get("cnn_confidence", detection["confidence"])),
            "severity": severity,
        }

    def decide(
        self,
        detections: list[dict[str, Any]],
        anomaly: dict[str, Any],
        station_id: str,
        camera_id: str,
    ) -> dict[str, Any]:
        """Return the final inspection result after aggregating stages 2-4.

        A result fails when any detection confidence or the normalized anomaly
        score reaches its configured threshold. Returned detections include
        configured severity labels for overlay rendering.
        """
        if not isinstance(detections, list):
            raise TypeError("detections must be a list")
        if not isinstance(anomaly, dict):
            raise TypeError("anomaly result must be a mapping")
        for key in ("score", "is_anomaly", "available"):
            if key not in anomaly:
                raise ValueError(f"anomaly result is missing {key!r}")
        if anomaly["available"] is not True:
            raise RuntimeError("anomaly detector reported unavailable")
        if not isinstance(anomaly["is_anomaly"], bool):
            raise TypeError("anomaly is_anomaly must be a boolean")
        anomaly_score = float(anomaly["score"])
        if not 0 <= anomaly_score <= 1:
            raise ValueError("normalized anomaly score must be in [0, 1]")
        if anomaly["is_anomaly"] != (anomaly_score > self.anomaly_threshold):
            raise ValueError("anomaly result threshold status does not match config")

        annotated = [self.evaluate_detection(detection) for detection in detections]
        failing_detections = [
            detection
            for detection in annotated
            if detection["cnn_confidence"] > self.fail_confidence_threshold
        ]
        if failing_detections:
            primary = max(
                failing_detections,
                key=lambda detection: detection["cnn_confidence"],
            )
            result = "FAIL"
            defect = primary["defect_type"]
            confidence = primary["cnn_confidence"]
        elif anomaly["is_anomaly"]:
            result = "FAIL"
            defect = "anomaly"
            confidence = anomaly_score
        else:
            result = "PASS"
            defect = "None"
            confidence = 1.0

        return {
            "result": result,
            "defect": defect,
            "confidence": float(confidence),
            "severity": "FAIL" if result == "FAIL" else (
                "WARN" if annotated else "NONE"
            ),
            "station_id": station_id,
            "camera_id": camera_id,
            "detections": annotated,
            "anomaly_score": anomaly_score,
            "anomaly": dict(anomaly),
        }
