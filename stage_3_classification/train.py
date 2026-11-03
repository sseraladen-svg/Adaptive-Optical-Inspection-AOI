"""Train a ResNet18 crop classifier and export the configured ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from utils.config_loader import load_config


def train(
    config: dict[str, Any],
    dataset_dir: Path | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Path:
    """Train using class-named crop folders and export ONNX to the config path.

    The dataset layout must be ``dataset_dir/<defect class>/*``. The folder
    names must match the configured defect classes exactly.
    """
    if dataset_dir is None:
        dataset_dir = Path(config["training"]["classifier_dataset_dir"])
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Defect crop dataset not found: {dataset_dir}")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    classification = config["classification"]
    input_size = int(classification["input_size"])
    mean = tuple(float(value) for value in classification["mean"])
    std = tuple(float(value) for value in classification["std"])
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = datasets.ImageFolder(str(dataset_dir), transform=transform)
    class_mapping = config["defect_classes"]
    class_names = [
        class_mapping[key]
        for key in sorted(class_mapping, key=lambda class_id: int(class_id))
    ]
    if set(dataset.classes) != set(class_names):
        raise ValueError(
            "Dataset class folders must match configured defect classes; "
            f"dataset={dataset.classes}, config={class_names}"
        )
    class_indices = {name: index for index, name in enumerate(class_names)}
    dataset.target_transform = lambda target: class_indices[dataset.classes[target]]
    if len(dataset) < 2 or len(class_names) < 2:
        raise ValueError("Training requires at least two images across two classes")

    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    if train_size == 0 or validation_size == 0:
        raise ValueError("80/20 split requires at least two images")
    generator = torch.Generator().manual_seed(seed)
    train_set, validation_set = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                correct += int((model(images).argmax(dim=1) == labels).sum().item())
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss / train_size:.4f}, "
            f"val_accuracy={correct / validation_size:.4f}"
        )

    output_path = Path(config["models"]["cnn_classifier"])
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("models.cnn_classifier must point to an .onnx model")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    checkpoint_path = output_path.with_suffix(".pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": class_names,
            "architecture": "resnet18",
        },
        checkpoint_path,
    )
    dummy_input = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=int(classification["onnx_opset"]),
        dynamo=False,
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Exported ONNX classifier: {output_path}")
    return output_path


def main() -> None:
    """Parse training options and train for one configured product."""
    parser = argparse.ArgumentParser(description="Train and export NOX ResNet18 classifier")
    parser.add_argument("--product", default="prismatic_cell")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.product)
    train(
        config,
        args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
