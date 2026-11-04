r"""
Run the full pipeline over a folder of images (no camera, no server).

  python -m tools.run_folder --product prismatic_cell --images C:\data\cells --out data\folder_run

Writes an annotated copy of every image to --out and prints one line per image.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from utils.overlay import draw_aoi_overlay

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def run_folder(router: Any, images_dir: Path, out_dir: Path, station_id: str, camera_id: str) -> list[dict[str, Any]]:
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"no images in {images_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"SKIP unreadable: {path.name}")
            continue
        processed, decision, timings = router.run(frame, station_id, camera_id)
        annotated = draw_aoi_overlay(processed.copy(), decision["detections"])
        cv2.imwrite(str(out_dir / f"{path.stem}.jpg"), annotated)
        rows.append({"image": path.name, "result": decision["result"], "defect": decision["defect"],
                     "confidence": decision["confidence"], "total_ms": timings["total_ms"]})
        print(f"{path.name:32s} {decision['result']:4s} {decision['defect']:14s} {timings['total_ms']:8.1f} ms")
    counts = Counter(r["result"] for r in rows)
    print(f"\n{dict(counts)}  annotated images: {out_dir}")
    return rows


def main() -> None:
    from pipeline.router import PipelineRouter
    from utils.config_loader import load_config

    parser = argparse.ArgumentParser(description="Inspect a folder of images")
    parser.add_argument("--product", default="prismatic_cell")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/folder_run"))
    parser.add_argument("--station", default="station_01")
    parser.add_argument("--camera", default="cam_01")
    args = parser.parse_args()
    run_folder(PipelineRouter(load_config(args.product)), args.images, args.out, args.station, args.camera)


if __name__ == "__main__":
    main()
