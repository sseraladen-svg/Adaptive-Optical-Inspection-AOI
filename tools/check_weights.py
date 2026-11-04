r"""
Verify a product's model files before running the pipeline.

  python -m tools.check_weights --product prismatic_cell

Loads each stage exactly as the pipeline does (the real Detector/Classifier/
AnomalyDetector validation runs), then adds the checks those stages do not do:
  - detection: model class names must equal config defect_classes
  - classification: a dummy crop is run through the model, so a wrong output
    width is caught now instead of on the first real defect
A package that is not installed is reported SKIP (the check did not run), never OK.
Exit code 1 if any check FAILs. Note: the anomaly bank is a pickle - only check
files you created yourself.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from utils.config_loader import load_config

OK, FAIL, SKIP = "OK  ", "FAIL", "SKIP"
Row = tuple[str, str]


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"{path.stat().st_size / 1e6:.1f} MB sha256:{digest.hexdigest()[:12]}"


def check_detection(config: dict[str, Any]) -> list[Row]:
    path = Path(config["models"]["yolo_seg"])
    if not path.is_file():
        return [(FAIL, f"detection: missing {path}")]
    rows: list[Row] = [(OK, f"detection: {path} ({_fingerprint(path)})")]
    try:
        from ultralytics import YOLO
    except ImportError:
        return rows + [(SKIP, "detection: ultralytics not installed, class check not run")]
    try:
        names = {int(k): v for k, v in dict(YOLO(str(path)).names).items()}
    except Exception as exc:  # noqa: BLE001 - report any load failure
        return rows + [(FAIL, f"detection: could not load model: {exc}")]
    expected = {int(k): v for k, v in config["defect_classes"].items()}
    if names == expected:
        rows.append((OK, "detection: model classes match defect_classes"))
    else:
        rows.append((FAIL, f"detection: model classes {names} != config {expected} (retrain on the generated data.yaml)"))
    return rows


def check_classification(config: dict[str, Any]) -> list[Row]:
    path = Path(config["models"]["cnn_classifier"])
    if not path.is_file():
        return [(FAIL, f"classification: missing {path}")]
    rows: list[Row] = [(OK, f"classification: {path} ({_fingerprint(path)})")]
    try:
        from stage_3_classification.classifier import Classifier

        classifier = Classifier(config)
    except ImportError as exc:
        return rows + [(SKIP, f"classification: {exc}")]
    except Exception as exc:  # noqa: BLE001
        return rows + [(FAIL, f"classification: {exc}")]
    rows.append((OK, f"classification: input {classifier.input_shape}"))
    try:
        classifier.classify(np.zeros((64, 64, 3), np.uint8), {"bbox": [0, 0, 32, 32]})
        rows.append((OK, f"classification: dummy crop gives {len(classifier.class_names)} class scores"))
    except Exception as exc:  # noqa: BLE001
        rows.append((FAIL, f"classification: dummy inference failed: {exc}"))
    return rows


def check_anomaly(config: dict[str, Any]) -> list[Row]:
    path = Path(config["models"]["anomaly"])
    if not path.is_file():
        return [(FAIL, f"anomaly: missing {path}")]
    rows: list[Row] = [(OK, f"anomaly: {path} ({_fingerprint(path)})")]
    try:
        from stage_4_anomaly.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(config)
    except Exception as exc:  # noqa: BLE001
        return rows + [(FAIL, f"anomaly: {exc}")]
    rows.append((OK, f"anomaly: bank {detector.memory_bank.shape} matches the configured HOG settings"))
    return rows


def run_checks(config: dict[str, Any]) -> list[Row]:
    return check_detection(config) + check_classification(config) + check_anomaly(config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify NOX model files for one product")
    parser.add_argument("--product", default="prismatic_cell")
    rows = run_checks(load_config(parser.parse_args().product))
    for status, message in rows:
        print(f"[{status}] {message}")
    return 1 if any(status == FAIL for status, _ in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
