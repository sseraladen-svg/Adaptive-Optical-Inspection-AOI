"""Draw severity-colored segmentation masks, boxes, and labels on BGR frames."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


_SEVERITY_COLORS = {
    "FAIL": (0, 0, 255),
    "WARN": (0, 255, 255),
    "NONE": (0, 255, 0),
}


def draw_aoi_overlay(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    mask_alpha: float = 0.25,
) -> np.ndarray:
    """Return a copy of frame with masks, bounding boxes, and class labels.

    Detections use [x1, y1, x2, y2] pixel boxes and optional 2D masks. The
    caller owns the input frame; it is not modified.
    """
    if (
        not isinstance(frame, np.ndarray)
        or frame.size == 0
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("overlay frame must be a three-channel BGR numpy image")
    if not isinstance(detections, list):
        raise TypeError("detections must be a list")
    if not math.isfinite(mask_alpha) or not 0 <= mask_alpha <= 1:
        raise ValueError("mask_alpha must be in [0, 1]")

    output = frame.copy()
    height, width = output.shape[:2]
    for detection in detections:
        if "bbox" not in detection or len(detection["bbox"]) != 4:
            raise ValueError("each detection must contain a four-value bbox")
        bbox = tuple(float(value) for value in detection["bbox"])
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError(f"detection bbox must contain finite values: {bbox}")
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"detection bbox has no area: {detection['bbox']}")
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        severity = detection.get("severity", "FAIL")
        color = _SEVERITY_COLORS.get(severity)
        if color is None:
            raise ValueError(f"unknown detection severity: {severity!r}")

        mask = detection.get("mask")
        if mask is not None:
            mask_array = np.asarray(mask)
            if mask_array.ndim != 2:
                raise ValueError("detection mask must be a two-dimensional array")
            resized_mask = cv2.resize(
                mask_array.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5
            if mask_alpha > 0 and np.any(resized_mask):
                blended = cv2.addWeighted(
                    output,
                    1.0 - mask_alpha,
                    np.full_like(output, color),
                    mask_alpha,
                    0,
                )
                output[resized_mask] = blended[resized_mask]

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = detection.get("defect_type", detection.get("label"))
        if not isinstance(label, str) or not label:
            raise ValueError("each detection must contain a non-empty label")
        confidence = float(
            detection.get("cnn_confidence", detection.get("confidence", 0.0))
        )
        cv2.putText(
            output,
            f"{label}:{confidence:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return output
