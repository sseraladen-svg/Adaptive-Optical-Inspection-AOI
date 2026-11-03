"""Load and validate product inspection configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_REQUIRED_KEYS = {
    "product_id",
    "strict_models",
    "roi",
    "preprocessing",
    "models",
    "defect_classes",
    "decision_rules",
}


def load_config(product_id: str, config_dir: str | Path = "config") -> dict[str, Any]:
    """Load a product YAML config and reject missing or malformed critical fields.

    Args:
        product_id: Product config filename without the `.yaml` suffix.
        config_dir: Directory containing product YAML files.

    Returns:
        The validated product configuration mapping.

    Raises:
        FileNotFoundError: If the selected product config does not exist.
        ValueError: If required fields or values are invalid.
    """
    if not product_id or Path(product_id).name != product_id:
        raise ValueError("product_id must be a non-empty filename stem")

    path = Path(config_dir) / f"{product_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No config found for product {product_id!r}: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")

    missing = sorted(_REQUIRED_KEYS.difference(config))
    if missing:
        raise ValueError(f"{path}: missing required key(s): {', '.join(missing)}")

    if config["product_id"] != product_id:
        raise ValueError(
            f"{path}: product_id must match filename {product_id!r}, "
            f"got {config['product_id']!r}"
        )
    if config["strict_models"] is not True:
        raise ValueError(f"{path}: strict_models must be true for fail-fast inference")

    config.setdefault("stations", 1)
    config.setdefault("cameras_per_station", 1)
    config.setdefault("reorientation_steps", [])

    preprocessing = config["preprocessing"]
    models = config["models"]
    rules = config["decision_rules"]
    if not isinstance(preprocessing, dict):
        raise ValueError(f"{path}: preprocessing must be a mapping")
    if not isinstance(models, dict):
        raise ValueError(f"{path}: models must be a mapping")
    if not isinstance(rules, dict):
        raise ValueError(f"{path}: decision_rules must be a mapping")

    for key in ("resize", "blur_kernel", "clahe"):
        if key not in preprocessing:
            raise ValueError(f"{path}: preprocessing.{key} is required")
    preprocessing.setdefault("clahe_clip_limit", 2.0)
    preprocessing.setdefault("clahe_tile_grid_size", [8, 8])
    for key in ("yolo_seg", "cnn_classifier", "anomaly"):
        if not isinstance(models.get(key), str) or not models[key].strip():
            raise ValueError(f"{path}: models.{key} must be a non-empty path")
    expected_extensions = {
        "yolo_seg": ".pt",
        "cnn_classifier": ".onnx",
        "anomaly": ".pkl",
    }
    for key, extension in expected_extensions.items():
        if Path(models[key]).suffix.lower() != extension:
            raise ValueError(f"{path}: models.{key} must use the {extension} format")
    if not isinstance(config["defect_classes"], dict) or not config["defect_classes"]:
        raise ValueError(f"{path}: defect_classes must be a non-empty class-id mapping")
    try:
        class_ids = [int(class_id) for class_id in config["defect_classes"]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: defect_classes keys must be integer class ids") from exc
    if len(set(class_ids)) != len(class_ids):
        raise ValueError(f"{path}: defect_classes contains duplicate class ids")
    if any(not isinstance(name, str) or not name for name in config["defect_classes"].values()):
        raise ValueError(f"{path}: defect class names must be non-empty strings")
    config["defect_classes"] = {
        int(class_id): name for class_id, name in config["defect_classes"].items()
    }

    for key in ("fail_confidence_threshold", "severity_map"):
        if key not in rules:
            raise ValueError(f"{path}: decision_rules.{key} is required")
    threshold = rules["fail_confidence_threshold"]
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError(f"{path}: fail_confidence_threshold must be in [0, 1]")
    severity_map = rules["severity_map"]
    if not isinstance(severity_map, dict) or any(
        severity not in {"WARN", "FAIL"} for severity in severity_map.values()
    ):
        raise ValueError(f"{path}: severity_map values must be WARN or FAIL")
    missing_severity = sorted(set(config["defect_classes"].values()) - set(severity_map))
    if missing_severity:
        raise ValueError(
            f"{path}: severity_map is missing defect class(es): {', '.join(missing_severity)}"
        )

    detection = config.setdefault("detection", {})
    classification = config.setdefault("classification", {})
    anomaly = config.setdefault("anomaly_detection", {})
    runtime = config.setdefault("runtime", {})
    training = config.setdefault("training", {})
    if not isinstance(detection, dict) or not isinstance(classification, dict):
        raise ValueError(f"{path}: detection and classification must be mappings")
    if not isinstance(anomaly, dict):
        raise ValueError(f"{path}: anomaly_detection must be a mapping")
    if not isinstance(runtime, dict):
        raise ValueError(f"{path}: runtime must be a mapping")
    if not isinstance(training, dict):
        raise ValueError(f"{path}: training must be a mapping")

    detection.setdefault("confidence_threshold", threshold)
    classification.setdefault("input_size", 224)
    classification.setdefault("mean", [0.485, 0.456, 0.406])
    classification.setdefault("std", [0.229, 0.224, 0.225])
    classification.setdefault("onnx_opset", 17)
    anomaly.setdefault("threshold", 0.5)
    anomaly.setdefault("distance_scale", 1.0)
    anomaly.setdefault("hog", {})
    runtime.setdefault("data_dir", "data")

    if not 0 <= float(detection["confidence_threshold"]) <= 1:
        raise ValueError(f"{path}: detection.confidence_threshold must be in [0, 1]")
    if not 0 <= float(anomaly["threshold"]) <= 1:
        raise ValueError(f"{path}: anomaly_detection.threshold must be in [0, 1]")
    if float(anomaly["distance_scale"]) <= 0:
        raise ValueError(f"{path}: anomaly_detection.distance_scale must be positive")
    if not isinstance(runtime["data_dir"], str) or not runtime["data_dir"].strip():
        raise ValueError(f"{path}: runtime.data_dir must be a non-empty path")

    hog = anomaly["hog"]
    if not isinstance(hog, dict):
        raise ValueError(f"{path}: anomaly_detection.hog must be a mapping")
    for key in (
        "win_size",
        "patch_stride",
        "block_size",
        "block_stride",
        "cell_size",
        "bins",
    ):
        if key not in hog:
            raise ValueError(f"{path}: anomaly_detection.hog.{key} is required")

    return config
