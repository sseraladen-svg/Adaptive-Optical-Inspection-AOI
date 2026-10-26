import argparse
import os
import time

import cv2

from pipeline.router import PipelineRouter
from utils.config_loader import load_config
from utils.overlay import draw_aoi_overlay

IMAGE_PATH = "received_images/latest.jpg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product",
        default="prismatic_cell",
        help="Product config id under config/",
    )
    args = parser.parse_args()

    config = load_config(args.product)
    router = PipelineRouter(config)
    print(f"[NOX Inspection] Starting... (product={config['product_id']})")

    last_modified = 0.0
    while True:
        if not os.path.exists(IMAGE_PATH):
            time.sleep(0.5)
            continue

        modified = os.path.getmtime(IMAGE_PATH)
        if modified != last_modified:
            last_modified = modified
            frame = cv2.imread(IMAGE_PATH)
            if frame is None:
                continue

            decision, timings = router.run(frame, "station_01", "cam_01")
            processed = decision.pop("_processed_frame")
            annotated_frame = draw_aoi_overlay(processed, decision["detections"])
            cv2.imshow("NOX Inspection", annotated_frame)

            print(
                f"\nResult: {decision['result']} ({decision['severity']}) | "
                f"Defect: {decision['defect']} | "
                f"Total: {timings['total_ms']:.2f}ms"
            )
            for detection in decision["detections"]:
                print(
                    f"  {detection['label']} | {detection['confidence']:.2f} | "
                    f"{detection['severity']}"
                )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
