import numpy as np
import pytest

from utils.overlay import draw_aoi_overlay


@pytest.mark.parametrize(
    ("severity", "expected_color"),
    [
        ("FAIL", (0, 0, 64)),
        ("WARN", (0, 64, 64)),
    ],
)
def test_mask_uses_alpha_blend_only_inside_mask(
    severity: str, expected_color: tuple[int, int, int]
) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    detection = {
        "bbox": [1, 1, 6, 6],
        "label": "scratch",
        "severity": severity,
        "mask": mask,
    }

    annotated = draw_aoi_overlay(frame, [detection], mask_alpha=0.25)

    np.testing.assert_array_equal(annotated[3, 3], expected_color)
    np.testing.assert_array_equal(annotated[0, 0], (0, 0, 0))
    np.testing.assert_array_equal(frame, np.zeros_like(frame))


def test_overlay_rejects_non_finite_mask_alpha() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="mask_alpha"):
        draw_aoi_overlay(frame, [], mask_alpha=float("nan"))


def test_overlay_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="three-channel BGR"):
        draw_aoi_overlay(np.zeros((0, 0, 3), dtype=np.uint8), [])


def test_overlay_requires_detection_list() -> None:
    with pytest.raises(TypeError, match="detections must be a list"):
        draw_aoi_overlay(np.zeros((8, 8, 3), dtype=np.uint8), None)  # type: ignore[arg-type]


def test_overlay_rejects_non_finite_bbox() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    detection = {"bbox": [1, 1, np.inf, 6], "label": "scratch"}
    with pytest.raises(ValueError, match="finite"):
        draw_aoi_overlay(frame, [detection])


def test_overlay_clips_bbox_to_frame() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    annotated = draw_aoi_overlay(
        frame,
        [{"bbox": [-4, -4, 4, 4], "label": "scratch"}],
    )
    assert np.any(annotated[0, :5])
    assert np.any(annotated[:5, 0])


def test_overlay_ignores_bbox_outside_frame() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    annotated = draw_aoi_overlay(
        frame,
        [{"bbox": [20, 20, 24, 24], "label": "scratch"}],
    )
    np.testing.assert_array_equal(annotated, frame)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_overlay_rejects_invalid_confidence(confidence: float) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    detection = {
        "bbox": [1, 1, 6, 6],
        "label": "scratch",
        "confidence": confidence,
    }
    with pytest.raises(ValueError, match="confidence"):
        draw_aoi_overlay(frame, [detection])


def test_overlay_rejects_non_finite_mask() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    detection = {
        "bbox": [1, 1, 6, 6],
        "label": "scratch",
        "mask": np.full((4, 4), np.nan, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="finite"):
        draw_aoi_overlay(frame, [detection])


def test_empty_mask_does_not_color_pixels_outside_bbox() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    detection = {
        "bbox": [2, 2, 5, 5],
        "label": "scratch",
        "mask": np.zeros((4, 4), dtype=np.uint8),
    }

    annotated = draw_aoi_overlay(frame, [detection])

    np.testing.assert_array_equal(annotated[0, 0], (0, 0, 0))


def test_boolean_mask_is_rendered() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[3, 3] = True
    detection = {
        "bbox": [1, 1, 6, 6],
        "label": "scratch",
        "mask": mask,
    }

    annotated = draw_aoi_overlay(frame, [detection], mask_alpha=1.0)

    np.testing.assert_array_equal(annotated[3, 3], (0, 0, 255))


def test_mask_is_resized_to_frame_dimensions() -> None:
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[0, 0] = 255
    detection = {
        "bbox": [1, 1, 10, 6],
        "label": "scratch",
        "mask": mask,
    }

    annotated = draw_aoi_overlay(frame, [detection], mask_alpha=1.0)

    assert tuple(annotated[1, 1]) == (0, 0, 255)
