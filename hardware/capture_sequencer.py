"""Hardware-agnostic capture/actuation interfaces; no drivers are loaded here."""

from __future__ import annotations

from typing import Any

import numpy as np


class CaptureSequencer:
    """Document camera capture boundaries without implementing hardware access."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Store product station layout for a future hardware adapter."""
        self.stations = int(config["stations"])
        self.cameras_per_station = int(config["cameras_per_station"])
        self.reorientation_steps = tuple(config["reorientation_steps"])

    def capture_station(self, station_id: str, camera_id: str) -> np.ndarray:
        """Return a BGR frame from a future adapter or raise until one is wired."""
        raise NotImplementedError(
            "No camera driver is configured "
            f"(station={station_id}, camera={camera_id})"
        )

    def trigger_reorientation(self, step: str) -> None:
        """Trigger a future actuator adapter or raise until one is wired."""
        raise NotImplementedError(f"No actuator driver is configured for {step!r}")
