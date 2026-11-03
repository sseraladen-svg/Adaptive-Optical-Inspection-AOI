"""ONNX Runtime inference for defect crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class Classifier:
    """Classify YOLO crops with a CPU-executed ResNet18 ONNX model."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Validate and load the configured ONNX model once at startup."""
        self.config = config
        self.model_path = Path(config["models"]["cnn_classifier"])
        if not self.model_path.is_file():
            raise FileNotFoundError(f"CNN ONNX model not found: {self.model_path}")

        classification = config["classification"]
        self.input_size = int(classification["input_size"])
        self.mean = np.asarray(classification["mean"], dtype=np.float32)
        self.std = np.asarray(classification["std"], dtype=np.float32)
        class_mapping = config["defect_classes"]
        self.class_names = [
            class_mapping[key]
            for key in sorted(class_mapping, key=lambda class_id: int(class_id))
        ]
        if self.input_size <= 0 or self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("Invalid classifier input_size, mean, or std in config")
        if np.any(self.std <= 0):
            raise ValueError("Classifier normalization std values must be positive")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required for CNN classification") from exc

        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("CNN ONNX model must have exactly one input and one output")
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape
        self.output_name = outputs[0].name
        if len(self.input_shape) != 4:
            raise ValueError("CNN ONNX input must be a four-dimensional image batch")
        self.channels_last = self.input_shape[-1] == 3
        image_shape = self.input_shape[1:] if not self.channels_last else self.input_shape[1:3]
        expected_shape = (
            [self.input_size, self.input_size, 3]
            if self.channels_last
            else [3, self.input_size, self.input_size]
        )
        if any(
            isinstance(actual, int) and actual != expected
            for actual, expected in zip(image_shape, expected_shape)
        ):
            raise ValueError(
                f"CNN ONNX input shape {self.input_shape} does not match configured "
                f"image size {self.input_size}"
            )

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize and normalize a non-empty BGR crop for the ONNX graph."""
        if not isinstance(crop, np.ndarray) or crop.size == 0:
            raise ValueError("classification crop must be a non-empty numpy image")
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("classification crop must have three BGR channels")
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        if self.channels_last:
            return normalized[np.newaxis, ...]
        return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

    @staticmethod
    def _crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Extract a clipped [x1, y1, x2, y2] crop or raise for invalid boxes."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy image")
        if len(bbox) != 4:
            raise ValueError("detection bbox must contain [x1, y1, x2, y2]")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"detection bbox has no area within frame: {bbox}")
        return frame[y1:y2, x1:x2]

    def _predict_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Run one batched ONNX inference and return label/confidence pairs."""
        if not crops:
            return []
        batch = np.concatenate([self._prepare_crop(crop) for crop in crops], axis=0)
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        scores = np.asarray(outputs[0], dtype=np.float32)
        if scores.ndim == 1:
            scores = scores[np.newaxis, :]
        if scores.ndim != 2 or scores.shape[0] != len(crops):
            raise ValueError(f"Unexpected classifier output shape: {scores.shape}")
        if scores.shape[1] != len(self.class_names):
            raise ValueError(
                f"Classifier returned {scores.shape[1]} classes; "
                f"config defines {len(self.class_names)}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("Classifier returned non-finite scores")

        sums = scores.sum(axis=1)
        if np.all(scores >= 0) and np.allclose(sums, 1.0, atol=1e-4):
            probabilities = scores
        else:
            shifted = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        predictions = []
        for row in probabilities:
            class_index = int(np.argmax(row))
            confidence = float(row[class_index])
            if not 0 <= confidence <= 1:
                raise ValueError("Classifier confidence is outside [0, 1]")
            predictions.append((self.class_names[class_index], confidence))
        return predictions

    def classify(self, frame: np.ndarray, detection: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of one detection updated with CNN class and confidence."""
        crop = self._crop(frame, detection["bbox"])
        defect_type, confidence = self._predict_crops([crop])[0]
        return {
            **detection,
            "defect_type": defect_type,
            "cnn_confidence": confidence,
        }

    def classify_all(
        self, frame: np.ndarray, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Classify detections in a single batch, preserving their input order."""
        crops = [self._crop(frame, detection["bbox"]) for detection in detections]
        predictions = self._predict_crops(crops)
        return [
            {
                **detection,
                "defect_type": defect_type,
                "cnn_confidence": confidence,
            }
            for detection, (defect_type, confidence) in zip(detections, predictions)
        ]
