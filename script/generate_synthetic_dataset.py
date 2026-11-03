"""
script/generate_synthetic_dataset.py

Generates a synthetic training set from a clean base cell-surface image
(dataset/synthetic_base/*.png) by procedurally drawing defects with OpenCV,
then auto-deriving:
  - YOLO segmentation polygon labels (exact pixels drawn = exact label, no
    manual annotation needed)
  - CNN classifier crops, sorted into class folders
  - "good" (no-defect) images for anomaly detection training

This unblocks YOLO/CNN/anomaly pipeline development before real prismatic-
cell photos exist. Swap in real images later by pointing this script's
BASE_DIR at real captures — output format is identical either way.

Usage:
    python script/generate_synthetic_dataset.py --count 300
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

BASE_DIR = Path("dataset/synthetic_base")
OUT_ROOT = Path("dataset/synthetic_generated")

# class_id -> name, must match config/prismatic_cell.yaml defect_classes
CLASSES = {0: "scratch", 1: "dust", 2: "stain", 3: "unclassified"}


# ==========================================================
# Defect drawing functions — each returns a binary mask (0/255)
# of exactly the pixels it modified, so the YOLO label is exact.
# ==========================================================

def draw_scratch(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
    length = random.randint(int(w * 0.3), int(w * 0.7))
    angle = random.uniform(0, 2 * np.pi)
    x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
    y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))

    thickness = random.randint(1, 2)
    color = (200, 200, 200)  # light gray scratch on dark blue surface

    # slight jitter along the line for a non-perfectly-straight scratch
    num_pts = 6
    pts = []
    for i in range(num_pts):
        t = i / (num_pts - 1)
        px = int(x1 + (x2 - x1) * t + random.randint(-3, 3))
        py = int(y1 + (y2 - y1) * t + random.randint(-3, 3))
        pts.append((px, py))

    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)
        cv2.line(mask, pts[i], pts[i + 1], 255, thickness + 2, cv2.LINE_AA)

    return mask


def draw_dust(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
    n_specks = random.randint(4, 10)

    for _ in range(n_specks):
        ox = cx + random.randint(-15, 15)
        oy = cy + random.randint(-15, 15)
        r = random.randint(1, 3)
        shade = random.randint(20, 60)
        color = (shade, shade, shade)
        cv2.circle(img, (ox, oy), r, color, -1, cv2.LINE_AA)
        cv2.circle(mask, (ox, oy), r + 1, 255, -1, cv2.LINE_AA)

    return mask


def draw_stain(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = random.randint(int(w * 0.2), int(w * 0.8)), random.randint(int(h * 0.2), int(h * 0.8))
    axes = (random.randint(15, 35), random.randint(10, 25))
    angle = random.randint(0, 180)

    overlay = img.copy()
    stain_color = (30, 60, 90)  # muddy brownish-dark patch
    cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, stain_color, -1, cv2.LINE_AA)
    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    return mask


def draw_unclassified(img: np.ndarray) -> np.ndarray:
    """Irregular anomaly — random polygon blotch, doesn't fit the other 3 patterns."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = random.randint(int(w * 0.2), int(w * 0.8)), random.randint(int(h * 0.2), int(h * 0.8))
    n_pts = random.randint(5, 8)
    radius = random.randint(10, 25)

    pts = []
    for i in range(n_pts):
        ang = 2 * np.pi * i / n_pts
        r = radius + random.randint(-5, 5)
        px = int(cx + r * np.cos(ang))
        py = int(cy + r * np.sin(ang))
        pts.append([px, py])

    pts = np.array([pts], dtype=np.int32)
    color = (random.randint(10, 50),) * 3
    cv2.fillPoly(img, pts, color, cv2.LINE_AA)
    cv2.fillPoly(mask, pts, 255, cv2.LINE_AA)

    return mask


DEFECT_FUNCS = {
    0: draw_scratch,
    1: draw_dust,
    2: draw_stain,
    3: draw_unclassified,
}


# ==========================================================
# Mask -> YOLO polygon
# ==========================================================

def mask_to_yolo_line(mask: np.ndarray, class_id: int) -> str | None:
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 5:
        return None

    epsilon = 0.01 * cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, epsilon, True)

    pts = []
    for p in poly:
        x, y = p[0]
        pts.append(f"{x / w:.6f}")
        pts.append(f"{y / h:.6f}")

    if len(pts) < 6:
        return None

    return f"{class_id} " + " ".join(pts)


# ==========================================================
# Augmentation (lighting/rotation/noise) — applied after defect drawing
# ==========================================================

def augment(img: np.ndarray) -> np.ndarray:
    # brightness/contrast jitter
    alpha = random.uniform(0.85, 1.15)  # contrast
    beta = random.randint(-15, 15)      # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # slight gaussian noise
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


# ==========================================================
# Main generation loop
# ==========================================================

def load_base_images(base_dir: Path = BASE_DIR) -> list[np.ndarray]:
    bases = []
    for image_path in sorted(base_dir.iterdir()):
        if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            img = cv2.imread(str(image_path))
            if img is not None:
                bases.append(img)
    if not bases:
        raise FileNotFoundError(f"No readable base images found in {base_dir}")
    return bases


def generate(
    count: int,
    val_split: float = 0.15,
    base_dir: Path = BASE_DIR,
    output_root: Path = OUT_ROOT,
    seed: int = 42,
) -> Path:
    if count < 2:
        raise ValueError("count must be at least 2 for train/validation data")
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base-image directory not found: {base_dir}")

    random.seed(seed)
    np.random.seed(seed)
    bases = load_base_images(base_dir)
    print(f"[synthetic] Loaded {len(bases)} base image(s) from {base_dir}")

    n_val = int(count * val_split)
    if n_val == 0:
        raise ValueError("count and val_split must produce at least one validation image")

    directories = {
        "yolo_images_train": output_root / "yolo_seg" / "images" / "train",
        "yolo_images_val": output_root / "yolo_seg" / "images" / "val",
        "yolo_labels_train": output_root / "yolo_seg" / "labels" / "train",
        "yolo_labels_val": output_root / "yolo_seg" / "labels" / "val",
        "cnn_train": output_root / "cnn_classifier" / "train",
        "cnn_val": output_root / "cnn_classifier" / "val",
        "anomaly_train": output_root / "anomaly" / "train" / "good",
        "anomaly_val": output_root / "anomaly" / "val" / "good",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    for split in ("cnn_train", "cnn_val"):
        for class_name in CLASSES.values():
            (directories[split] / class_name).mkdir(parents=True, exist_ok=True)

    for i in range(count):
        is_val = i < n_val
        base = random.choice(bases).copy()
        h, w = base.shape[:2]

        # resize base to a consistent working size
        base = cv2.resize(base, (400, 400))
        h, w = 400, 400

        img = base.copy()

        # 15% of samples are "good" (no defect) — for anomaly detection + PASS-class balance
        is_good = random.random() < 0.15

        yolo_lines = []
        crops = []  # (class_id, crop_img)

        if not is_good:
            n_defects = random.choices([1, 2], weights=[0.8, 0.2])[0]
            for _ in range(n_defects):
                class_id = random.choice(list(CLASSES.keys()))
                mask = DEFECT_FUNCS[class_id](img)
                line = mask_to_yolo_line(mask, class_id)
                if line:
                    yolo_lines.append(line)

                    ys, xs = np.where(mask > 0)
                    if len(xs) > 0:
                        x1, x2 = max(0, xs.min() - 5), min(w, xs.max() + 5)
                        y1, y2 = max(0, ys.min() - 5), min(h, ys.max() + 5)
                        crop = img[y1:y2, x1:x2]
                        if crop.size > 0:
                            crops.append((class_id, crop))

        img = augment(img)

        image_dir = directories["yolo_images_val" if is_val else "yolo_images_train"]
        label_dir = directories["yolo_labels_val" if is_val else "yolo_labels_train"]
        cnn_dir = directories["cnn_val" if is_val else "cnn_train"]
        anomaly_dir = directories["anomaly_val" if is_val else "anomaly_train"]

        fname = f"synth_{i:05d}"
        if not cv2.imwrite(str(image_dir / f"{fname}.jpg"), img):
            raise OSError(f"Could not write synthetic image {fname}")

        with (label_dir / f"{fname}.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        if is_good:
            if not cv2.imwrite(str(anomaly_dir / f"{fname}.jpg"), img):
                raise OSError(f"Could not write normal synthetic image {fname}")

        for j, (class_id, crop) in enumerate(crops):
            cls_name = CLASSES[class_id]
            crop_resized = cv2.resize(crop, (224, 224))
            if not cv2.imwrite(str(cnn_dir / cls_name / f"{fname}_{j}.jpg"), crop_resized):
                raise OSError(f"Could not write synthetic crop {fname}_{j}")

    yolo_root = (output_root / "yolo_seg").resolve()
    data_config_path = yolo_root / "data.yaml"
    with data_config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(
            {
                "path": str(yolo_root),
                "train": "images/train",
                "val": "images/val",
                "names": CLASSES,
            },
            config_file,
            sort_keys=False,
        )
    print(f"[synthetic] Generated {count} images ({n_val} val / {count - n_val} train)")
    print(f"[synthetic] YOLO seg config:  {data_config_path}")
    print(f"[synthetic] CNN classifier:   {output_root / 'cnn_classifier'}")
    print(f"[synthetic] Anomaly (good):   {output_root / 'anomaly'}")
    return data_config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300, help="Total number of synthetic images to generate")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(args.count, args.val_split, args.base_dir, args.output_dir, args.seed)
