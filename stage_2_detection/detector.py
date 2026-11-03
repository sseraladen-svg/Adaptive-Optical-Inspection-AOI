"""Strict, config-driven YOLO11 segmentation inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class Detector:
    """Run YOLO segmentation and return one standardized record per detection."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Validate and load the configured YOLO model once at startup."""
        self.config = config
        model_path = Path(config["models"]["yolo_seg"])
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.confidence_threshold = float(config["detection"]["confidence_threshold"])
        self.class_names = {int(key): value for key, value in config["defect_classes"].items()}
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        print(
            f"[Detector] Loaded {model_path} "
            f"(confidence threshold={self.confidence_threshold})"
        )

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return detections with label, class id, confidence, bbox, and mask.

        Args:
            frame: Preprocessed uint8 BGR image with shape (height, width, 3).

        Returns:
            Detection dictionaries; each bbox is [x1, y1, x2, y2] in pixels.
            A detection mask is a two-dimensional numpy array, or None when the
            loaded YOLO model returns boxes without segmentation masks.
        """
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy image")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a 3-channel BGR image")

        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None

            for index, (bbox, class_id, confidence) in enumerate(
                zip(boxes_xyxy, class_ids, confidences)
            ):
                if class_id not in self.class_names:
                    raise ValueError(f"YOLO returned unmapped class id: {class_id}")
                mask = masks[index] if masks is not None else None
                detections.append(
                    {
                        "label": self.class_names[class_id],
                        "class_id": int(class_id),
                        "confidence": float(confidence),
                        "bbox": [int(round(value)) for value in bbox],
                        "mask": mask,
                    }
                )
        return detections
