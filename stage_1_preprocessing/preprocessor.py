"""Config-driven image preprocessing for inspection frames."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class Preprocessor:
    """Crop, resize, denoise, and optionally normalize a BGR image."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Create preprocessing operators from the full product configuration."""
        preprocessing = config["preprocessing"]
        self.roi = config.get("roi")
        self.resize_dims = tuple(int(value) for value in preprocessing["resize"])
        self.blur_kernel = tuple(int(value) for value in preprocessing["blur_kernel"])
        self.use_clahe = bool(preprocessing["clahe"])
        self.clahe_clip_limit = float(preprocessing["clahe_clip_limit"])
        self.clahe_tile_grid_size = tuple(
            int(value) for value in preprocessing["clahe_tile_grid_size"]
        )

        if len(self.resize_dims) != 2 or any(value <= 0 for value in self.resize_dims):
            raise ValueError("preprocessing.resize must be [width, height] > 0")
        if len(self.blur_kernel) != 2 or any(
            value <= 0 or value % 2 == 0 for value in self.blur_kernel
        ):
            raise ValueError("preprocessing.blur_kernel must contain positive odd values")
        if len(self.clahe_tile_grid_size) != 2 or any(
            value <= 0 for value in self.clahe_tile_grid_size
        ):
            raise ValueError("preprocessing.clahe_tile_grid_size must be positive [x, y]")
        if self.clahe_clip_limit <= 0:
            raise ValueError("preprocessing.clahe_clip_limit must be positive")

        self._clahe = (
            cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size,
            )
            if self.use_clahe
            else None
        )

    def crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """Return the ROI crop using configured [x1, y1, x2, y2] coordinates."""
        if self.roi is None:
            return frame
        if not isinstance(self.roi, (list, tuple)) or len(self.roi) != 4:
            raise ValueError("roi must be null or [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (int(value) for value in self.roi)
        height, width = frame.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
            raise ValueError(f"configured ROI {self.roi} is outside frame shape {frame.shape}")
        return frame[y1:y2, x1:x2]

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize a BGR image to configured [width, height]."""
        return cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_LINEAR)

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        """Apply the configured Gaussian blur kernel to a BGR image."""
        return cv2.GaussianBlur(frame, self.blur_kernel, 0)

    def normalize_lighting(self, frame: np.ndarray) -> np.ndarray:
        """Apply configured CLAHE to the LAB lightness channel when enabled."""
        if self._clahe is None:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        adjusted = self._clahe.apply(lightness)
        return cv2.cvtColor(
            cv2.merge((adjusted, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Return the preprocessed BGR frame, raising on invalid or empty inputs."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy image")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a 3-channel BGR image")
        processed = self.crop_roi(frame)
        processed = self.resize(processed)
        processed = self.denoise(processed)
        return self.normalize_lighting(processed)
