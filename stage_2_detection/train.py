"""Train the configured YOLO segmentation model and save its best weights."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from utils.config_loader import load_config


def train(config: dict[str, Any]) -> Path:
    """Train and validate YOLO using product-configured paths and parameters."""
    training = config["training"]
    base_model_path = Path(training["yolo_base_model"])
    dataset_config = Path(training["yolo_dataset_config"])
    output_model = Path(config["models"]["yolo_seg"])
    if not base_model_path.is_file():
        raise FileNotFoundError(f"YOLO base model not found: {base_model_path}")
    if not dataset_config.is_file():
        raise FileNotFoundError(f"YOLO dataset config not found: {dataset_config}")

    from ultralytics import YOLO

    model = YOLO(str(base_model_path))
    model.train(
        data=str(dataset_config),
        epochs=int(training["yolo_epochs"]),
        imgsz=int(training["yolo_image_size"]),
        batch=int(training["yolo_batch_size"]),
        workers=int(training["yolo_workers"]),
        device=training["yolo_device"],
        project=str(training["yolo_run_dir"]),
        name=config["product_id"],
        pretrained=True,
        verbose=True,
    )
    best_model = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(f"YOLO training did not produce best weights: {best_model}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, output_model)
    trained_model = YOLO(str(output_model))
    metrics = trained_model.val(data=str(dataset_config), imgsz=int(training["yolo_image_size"]))
    print(f"Saved YOLO weights to {output_model}")
    print(f"Validation metrics: {metrics}")
    return output_model


def main() -> None:
    """Train the model for the product selected on the command line."""
    parser = argparse.ArgumentParser(description="Train configured NOX YOLO segmentation")
    parser.add_argument("--product", default="prismatic_cell")
    args = parser.parse_args()
    train(load_config(args.product))


if __name__ == "__main__":
    main()
