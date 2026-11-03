"""Patch-descriptor nearest-neighbor anomaly detection with a pickle memory bank."""

from __future__ import annotations

import pickle
import argparse
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np


class AnomalyDetector:
    """Score full frames against a configured normal patch-feature memory bank.

    The pickle file must be a trusted local artifact containing either an
    ``(n_samples, feature_size)`` numeric array or a mapping with a
    ``memory_bank`` array. Each vector is an OpenCV HOG descriptor generated
    from an overlapping image patch using the configured HOG geometry.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Load and validate the configured HOG feature bank and score limits."""
        self.config = config
        self.model_path = Path(config["models"]["anomaly"])
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Anomaly memory bank not found: {self.model_path}")

        anomaly_config = config["anomaly_detection"]
        hog_config = anomaly_config["hog"]
        self.threshold = float(anomaly_config["threshold"])
        self.distance_scale = float(anomaly_config["distance_scale"])
        self.max_memory_bank_samples = int(anomaly_config["max_memory_bank_samples"])
        self.win_size = tuple(int(value) for value in hog_config["win_size"])
        self.patch_stride = tuple(int(value) for value in hog_config["patch_stride"])
        self.block_size = tuple(int(value) for value in hog_config["block_size"])
        self.block_stride = tuple(int(value) for value in hog_config["block_stride"])
        self.cell_size = tuple(int(value) for value in hog_config["cell_size"])
        self.bins = int(hog_config["bins"])
        self._validate_hog_config()
        self.hog = self._new_hog(anomaly_config["hog"])

        # Pickle must only be loaded from a trusted, locally provisioned model.
        with self.model_path.open("rb") as model_file:
            loaded = pickle.load(model_file)
        if isinstance(loaded, dict):
            if "memory_bank" not in loaded:
                raise ValueError(f"{self.model_path} is missing the 'memory_bank' field")
            loaded = loaded["memory_bank"]
        memory_bank = np.asarray(loaded, dtype=np.float32)
        if memory_bank.ndim != 2 or memory_bank.shape[0] == 0:
            raise ValueError("Anomaly memory_bank must be a non-empty 2D array")
        if not np.all(np.isfinite(memory_bank)):
            raise ValueError("Anomaly memory_bank contains non-finite values")
        expected_size = self.hog.getDescriptorSize()
        if memory_bank.shape[1] != expected_size:
            raise ValueError(
                f"Anomaly memory bank feature size {memory_bank.shape[1]} does not "
                f"match configured HOG descriptor size {expected_size}"
            )
        if self.max_memory_bank_samples <= 0:
            raise ValueError("Anomaly max_memory_bank_samples must be positive")
        if not 0 <= self.threshold <= 1 or self.distance_scale <= 0:
            raise ValueError("Anomaly threshold must be [0, 1] and distance_scale positive")
        self.memory_bank = self._limit_memory_bank(
            memory_bank,
            self.max_memory_bank_samples,
        )

    def _validate_hog_config(self) -> None:
        """Reject invalid or incompatible configured HOG geometry."""
        values = (
            self.win_size,
            self.patch_stride,
            self.block_size,
            self.block_stride,
            self.cell_size,
        )
        if any(len(value) != 2 or any(dimension <= 0 for dimension in value) for value in values):
            raise ValueError("HOG dimensions must be pairs of positive integers")
        if self.bins <= 0:
            raise ValueError("HOG bins must be positive")
        if any(
            window_dimension < block_dimension
            for window_dimension, block_dimension in zip(self.win_size, self.block_size)
        ):
            raise ValueError("HOG window dimensions must be at least the block dimensions")
        for window, block, stride, cell in zip(
            self.win_size, self.block_size, self.block_stride, self.cell_size
        ):
            if (window - block) % stride or block % cell:
                raise ValueError("HOG geometry must align by stride and cell dimensions")

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Return one configured HOG feature vector for every full-frame patch."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("anomaly input must be a non-empty numpy image")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("anomaly input must be a full 3-channel BGR frame")
        return self._extract_with_hog(
            frame,
            self.hog,
            self.win_size,
            self.patch_stride,
        )

    @staticmethod
    def _new_hog(hog_config: dict[str, Any]) -> cv2.HOGDescriptor:
        """Create an OpenCV HOG extractor using configured geometry."""
        return cv2.HOGDescriptor(
            tuple(int(value) for value in hog_config["win_size"]),
            tuple(int(value) for value in hog_config["block_size"]),
            tuple(int(value) for value in hog_config["block_stride"]),
            tuple(int(value) for value in hog_config["cell_size"]),
            int(hog_config["bins"]),
        )

    @staticmethod
    def _extract_with_hog(
        frame: np.ndarray,
        hog: cv2.HOGDescriptor,
        win_size: tuple[int, int],
        patch_stride: tuple[int, int],
    ) -> np.ndarray:
        """Extract finite HOG vectors from overlapping patches across a BGR image."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("anomaly input must be a non-empty numpy image")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("anomaly input must be a full 3-channel BGR frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if width < win_size[0] or height < win_size[1]:
            scale = max(win_size[0] / width, win_size[1] / height)
            resized_size = (
                max(win_size[0], int(round(width * scale))),
                max(win_size[1], int(round(height * scale))),
            )
            gray = cv2.resize(gray, resized_size, interpolation=cv2.INTER_CUBIC)
            height, width = gray.shape

        x_positions = list(range(0, width - win_size[0] + 1, patch_stride[0]))
        y_positions = list(range(0, height - win_size[1] + 1, patch_stride[1]))
        x_positions.append(width - win_size[0])
        y_positions.append(height - win_size[1])
        vectors = []
        for y in sorted(set(y_positions)):
            for x in sorted(set(x_positions)):
                patch = gray[y : y + win_size[1], x : x + win_size[0]]
                feature = hog.compute(patch)
                if feature is None or feature.size == 0:
                    raise ValueError("HOG feature extraction returned no features")
                vectors.append(np.asarray(feature, dtype=np.float32).reshape(-1))
        features = np.stack(vectors)
        if not np.all(np.isfinite(features)):
            raise ValueError("HOG feature extraction returned non-finite values")
        return features

    def predict(self, frame: np.ndarray) -> dict[str, Any]:
        """Return normalized nearest-neighbor anomaly score and threshold status."""
        patch_features = self.extract_features(frame)
        if patch_features.size == 0 or self.memory_bank.size == 0:
            raise ValueError("Nearest-neighbor search returned no valid distances")
        patch_norms = np.einsum("ij,ij->i", patch_features, patch_features)[:, None]
        bank_norms = np.einsum("ij,ij->i", self.memory_bank, self.memory_bank)[None, :]
        squared_distances = (
            patch_norms
            + bank_norms
            - 2.0 * (patch_features @ self.memory_bank.T)
        )
        if not np.all(np.isfinite(squared_distances)):
            raise ValueError("Nearest-neighbor search returned invalid distances")
        np.maximum(squared_distances, 0, out=squared_distances)
        nearest_distances = np.sqrt(np.min(squared_distances, axis=1))
        maximum_patch_distance = float(np.max(nearest_distances))
        score = maximum_patch_distance / (
            maximum_patch_distance + self.distance_scale
        )
        return {
            "score": float(score),
            "is_anomaly": bool(score > self.threshold),
            "available": True,
        }

    @staticmethod
    def _limit_memory_bank(memory_bank: np.ndarray, maximum_samples: int) -> np.ndarray:
        """Uniformly sample templates so bank size stays within its configured cap."""
        if len(memory_bank) <= maximum_samples:
            return memory_bank
        indices = np.linspace(
            0,
            len(memory_bank) - 1,
            num=maximum_samples,
            dtype=np.intp,
        )
        return memory_bank[indices]

    @classmethod
    def save_memory_bank(
        cls,
        frames: Iterable[np.ndarray],
        output_path: str | Path,
        config: dict[str, Any],
    ) -> None:
        """Build a trusted pickle memory bank from normal full-frame BGR images."""
        anomaly_config = config["anomaly_detection"]
        hog_config = anomaly_config["hog"]
        hog = cls._new_hog(hog_config)
        win_size = tuple(int(value) for value in hog_config["win_size"])
        patch_stride = tuple(int(value) for value in hog_config["patch_stride"])
        features = [
            cls._extract_with_hog(frame, hog, win_size, patch_stride)
            for frame in frames
        ]
        if not features:
            raise ValueError("At least one normal image is required for a memory bank")
        bank = np.concatenate(features, axis=0).astype(np.float32)
        bank = cls._limit_memory_bank(
            bank,
            int(anomaly_config["max_memory_bank_samples"]),
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as model_file:
            pickle.dump({"memory_bank": bank}, model_file, protocol=pickle.HIGHEST_PROTOCOL)


def build_memory_bank(config: dict[str, Any], normal_dir: Path) -> Path:
    """Build the configured anomaly bank from normal full-frame images."""
    from stage_1_preprocessing.preprocessor import Preprocessor

    if not normal_dir.is_dir():
        raise FileNotFoundError(f"Normal-image directory not found: {normal_dir}")

    preprocessor = Preprocessor(config)
    image_paths = sorted(
        path
        for path in normal_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported normal images found in {normal_dir}")

    def frames() -> Iterator[np.ndarray]:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not decode normal image: {image_path}")
            yield preprocessor.process(frame)

    output_path = Path(config["models"]["anomaly"])
    AnomalyDetector.save_memory_bank(frames(), output_path, config)
    print(f"Saved normal patch memory bank from {len(image_paths)} images to {output_path}")
    return output_path


def main() -> None:
    """Build the configured pickle bank from a directory of normal images."""
    from utils.config_loader import load_config

    parser = argparse.ArgumentParser(description="Build a NOX normal patch memory bank")
    parser.add_argument("--product", default="prismatic_cell")
    parser.add_argument("--normal-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.product)
    normal_dir = args.normal_dir or Path(config["training"]["normal_image_dir"])
    build_memory_bank(config, normal_dir)


if __name__ == "__main__":
    main()
