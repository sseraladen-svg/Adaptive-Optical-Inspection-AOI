from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from script.generate_synthetic_dataset import CLASSES, generate


def test_synthetic_dataset_writes_usable_yolo_config_and_outputs(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_image = np.full((120, 120, 3), (40, 80, 120), dtype=np.uint8)
    assert cv2.imwrite(str(base_dir / "cell.png"), base_image)

    output_root = tmp_path / "generated"
    config_path = generate(20, val_split=0.2, base_dir=base_dir, output_root=output_root)

    data_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data_config["names"] == CLASSES
    dataset_root = Path(data_config["path"])
    for split, expected_count in (("train", 16), ("val", 4)):
        images = dataset_root / "images" / split
        labels = dataset_root / "labels" / split
        assert len(list(images.glob("*.jpg"))) == expected_count
        assert len(list(labels.glob("*.txt"))) == expected_count
        for class_name in CLASSES.values():
            assert (output_root / "cnn_classifier" / split / class_name).is_dir()

    assert list((output_root / "anomaly" / "train" / "good").glob("*.jpg"))


def test_synthetic_dataset_requires_readable_base_images(tmp_path: Path) -> None:
    base_dir = tmp_path / "empty"
    base_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No readable base images"):
        generate(10, base_dir=base_dir, output_root=tmp_path / "out")


def test_synthetic_dataset_rejects_empty_validation_split(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    assert cv2.imwrite(
        str(base_dir / "cell.png"),
        np.zeros((120, 120, 3), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="at least one validation image"):
        generate(
            2,
            val_split=0.1,
            base_dir=base_dir,
            output_root=tmp_path / "out",
        )
