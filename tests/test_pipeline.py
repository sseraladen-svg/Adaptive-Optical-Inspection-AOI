from __future__ import annotations

import csv
from pathlib import Path
import sys
import types
from typing import Any

import cv2
import numpy as np
import pytest

from pipeline import router as router_module
from stage_1_preprocessing.preprocessor import Preprocessor
from stage_3_classification.classifier import Classifier
from stage_3_classification.train import train as train_classifier
from stage_4_anomaly.anomaly_detector import AnomalyDetector
from stage_4_anomaly.anomaly_detector import build_memory_bank
from stage_5_decision.decision_engine import DecisionEngine
from utils.config_loader import load_config


@pytest.fixture
def config() -> dict[str, Any]:
    """Return a freshly loaded real product config."""
    return load_config("prismatic_cell")


def install_fake_stages(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    """Replace inference models while retaining the real router and decisions."""

    class FakePreprocessor:
        def __init__(self, _config: dict[str, Any]) -> None:
            events.append("init_s1")

        def process(self, frame: np.ndarray) -> np.ndarray:
            events.append("s1")
            return cv2.resize(frame, (32, 32))

    class FakeDetector:
        detections: list[dict[str, Any]] = [
            {
                "label": "scratch",
                "class_id": 0,
                "confidence": 0.9,
                "bbox": [1, 1, 12, 12],
                "mask": np.ones((8, 8), dtype=np.uint8),
            }
        ]

        def __init__(self, _config: dict[str, Any]) -> None:
            events.append("init_s2")

        def detect(self, _frame: np.ndarray) -> list[dict[str, Any]]:
            events.append("s2")
            return [dict(detection) for detection in self.detections]

    class FakeClassifier:
        def __init__(self, _config: dict[str, Any]) -> None:
            events.append("init_s3")

        def classify_all(
            self, _frame: np.ndarray, detections: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            events.append("s3")
            return [
                {**detection, "defect_type": "scratch", "cnn_confidence": 0.91}
                for detection in detections
            ]

    class FakeAnomalyDetector:
        input_shapes: list[tuple[int, ...]] = []

        def __init__(self, _config: dict[str, Any]) -> None:
            events.append("init_s4")

        def predict(self, frame: np.ndarray) -> dict[str, Any]:
            events.append("s4")
            self.input_shapes.append(frame.shape)
            return {"score": 0.1, "is_anomaly": False, "available": True}

    monkeypatch.setattr(router_module, "Preprocessor", FakePreprocessor)
    monkeypatch.setattr(router_module, "Detector", FakeDetector)
    monkeypatch.setattr(router_module, "Classifier", FakeClassifier)
    monkeypatch.setattr(router_module, "AnomalyDetector", FakeAnomalyDetector)


def test_config_loading(config: dict[str, Any]) -> None:
    """The real product YAML parses, validates, and enables strict model checks."""
    assert config["product_id"] == "prismatic_cell"
    assert config["strict_models"] is True
    assert config["defect_classes"][0] == "scratch"
    assert config["models"]["cnn_classifier"].endswith(".onnx")
    assert (
        config["training"]["yolo_dataset_config"]
        == "dataset/synthetic_generated/yolo_seg/data.yaml"
    )
    assert (
        config["training"]["classifier_dataset_dir"]
        == "dataset/synthetic_generated/cnn_classifier/train"
    )
    assert (
        config["training"]["normal_image_dir"]
        == "dataset/synthetic_generated/anomaly/train/good"
    )


def test_router_structure(config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The router runs stages sequentially and returns processed frame, decision, timings."""
    events: list[str] = []
    install_fake_stages(monkeypatch, events)
    router = router_module.PipelineRouter(config)
    events.clear()

    processed, decision, timings = router.run(
        np.zeros((64, 64, 3), dtype=np.uint8),
        "station-test",
        "camera-test",
    )

    assert events == ["s1", "s2", "s3", "s4"]
    assert processed.shape == (32, 32, 3)
    assert decision["result"] == "FAIL"
    assert decision["defect"] == "scratch"
    assert decision["station_id"] == "station-test"
    assert decision["camera_id"] == "camera-test"
    assert decision["timings"] == timings
    assert set(timings) == {
        "stage_1_preprocessing_ms",
        "stage_2_detection_ms",
        "stage_3_classification_ms",
        "stage_4_anomaly_ms",
        "stage_5_decision_ms",
        "total_ms",
    }
    assert all(value >= 0 for value in timings.values())


def test_conditional_s3(config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 3 is skipped for an empty stage 2 result."""
    events: list[str] = []
    install_fake_stages(monkeypatch, events)
    monkeypatch.setattr(router_module.Detector, "detections", [])
    router = router_module.PipelineRouter(config)
    events.clear()

    _, decision, _ = router.run(np.zeros((64, 64, 3), dtype=np.uint8), "s", "c")

    assert events == ["s1", "s2", "s4"]
    assert decision["result"] == "PASS"
    assert decision["detections"] == []


def test_anomaly_always_runs(config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 4 receives the entire preprocessed frame even when S2 has no detections."""
    events: list[str] = []
    install_fake_stages(monkeypatch, events)
    monkeypatch.setattr(router_module.Detector, "detections", [])
    router = router_module.PipelineRouter(config)
    events.clear()

    processed, _, _ = router.run(
        np.zeros((64, 64, 3), dtype=np.uint8),
        "station-test",
        "camera-test",
    )

    assert events == ["s1", "s2", "s4"]
    assert router.anomaly_detector.input_shapes == [processed.shape]


def test_fail_fast_missing_model(config: dict[str, Any], tmp_path: Path) -> None:
    """Router construction fails immediately when its configured YOLO model is absent."""
    config["models"]["yolo_seg"] = str(tmp_path / "missing_yolo.pt")
    with pytest.raises(FileNotFoundError, match="YOLO model not found"):
        router_module.PipelineRouter(config)


def test_classifier_fails_on_missing_onnx(config: dict[str, Any], tmp_path: Path) -> None:
    """Classifier never falls back to dummy labels when ONNX weights are absent."""
    config["models"]["cnn_classifier"] = str(tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError, match="CNN ONNX model not found"):
        Classifier(config)


def test_classifier_training_uses_configured_crop_directory(
    config: dict[str, Any], tmp_path: Path
) -> None:
    """Classifier training defaults to its configured generated crop directory."""
    dataset_dir = tmp_path / "configured-crops"
    config["training"]["classifier_dataset_dir"] = str(dataset_dir)

    with pytest.raises(FileNotFoundError) as error:
        train_classifier(config)
    assert str(dataset_dir) in str(error.value)


def test_classifier_uses_cpu_onnx_and_preserves_class_mapping(
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ONNX inference uses only CPU and maps output indices through product config."""
    model_path = tmp_path / "classifier.onnx"
    model_path.touch()
    config["models"]["cnn_classifier"] = str(model_path)
    provider_calls: list[list[str]] = []

    class FakeInput:
        name = "images"
        shape = [None, 3, 224, 224]

    class FakeOutput:
        name = "logits"

    class FakeSession:
        def __init__(self, _path: str, providers: list[str]) -> None:
            provider_calls.append(providers)

        def get_inputs(self) -> list[FakeInput]:
            return [FakeInput()]

        def get_outputs(self) -> list[FakeOutput]:
            return [FakeOutput()]

        def run(
            self, _outputs: list[str], inputs: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            return [
                np.tile(
                    np.asarray([[0.0, 0.0, 4.0, 0.0]], dtype=np.float32),
                    (inputs["images"].shape[0], 1),
                )
            ]

    fake_onnxruntime = types.SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    classifier = Classifier(config)
    detection = {
        "label": "scratch",
        "confidence": 0.8,
        "bbox": [0, 0, 16, 16],
    }
    result = classifier.classify(
        np.zeros((32, 32, 3), dtype=np.uint8),
        detection,
    )

    assert classifier.session is not None
    assert result["defect_type"] == "stain"
    assert result["cnn_confidence"] == pytest.approx(0.9479, abs=1e-3)
    assert result["label"] == "scratch"
    assert classifier.input_name == "images"
    assert provider_calls == [["CPUExecutionProvider"]]


def test_detector_returns_standardized_detections(
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """YOLO boxes and segmentation masks are converted to the public detection shape."""
    model_path = tmp_path / "detector.pt"
    model_path.touch()
    config["models"]["yolo_seg"] = str(model_path)

    class Tensor:
        def __init__(self, values: Any) -> None:
            self.values = np.asarray(values)

        def cpu(self) -> "Tensor":
            return self

        def numpy(self) -> np.ndarray:
            return self.values

    class Boxes:
        xyxy = Tensor([[1.2, 2.4, 8.6, 9.1]])
        cls = Tensor([0])
        conf = Tensor([0.87])

        def __len__(self) -> int:
            return 1

    class Masks:
        data = Tensor([np.ones((4, 4), dtype=np.uint8)])

    fake_model = lambda _frame, **_kwargs: [
        types.SimpleNamespace(boxes=Boxes(), masks=Masks())
    ]
    fake_ultralytics = types.SimpleNamespace(
        YOLO=lambda _path: fake_model,
    )
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    from stage_2_detection.detector import Detector

    detector = Detector(config)
    detections = detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))
    assert detections[0]["label"] == "scratch"
    assert detections[0]["class_id"] == 0
    assert detections[0]["confidence"] == pytest.approx(0.87)
    assert detections[0]["bbox"] == [1, 2, 9, 9]
    assert detections[0]["mask"].shape == (4, 4)


def test_anomaly_memory_bank_round_trip(config: dict[str, Any], tmp_path: Path) -> None:
    """A generated trusted memory bank can be loaded and scores its source frame."""
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    frame[:, 48:52] = 255
    memory_bank_path = tmp_path / "patchcore.pkl"
    AnomalyDetector.save_memory_bank([frame], memory_bank_path, config)
    config["models"]["anomaly"] = str(memory_bank_path)

    detector = AnomalyDetector(config)
    result = detector.predict(frame)

    assert result["available"] is True
    assert result["score"] == pytest.approx(0.0)
    assert result["is_anomaly"] is False


def test_anomaly_memory_bank_respects_configured_sample_cap(
    config: dict[str, Any], tmp_path: Path
) -> None:
    config["anomaly_detection"]["max_memory_bank_samples"] = 3
    frames = [
        np.full((256, 256, 3), value, dtype=np.uint8)
        for value in (0, 64, 128, 192)
    ]
    memory_bank_path = tmp_path / "capped_memory_bank.pkl"
    AnomalyDetector.save_memory_bank(frames, memory_bank_path, config)
    config["models"]["anomaly"] = str(memory_bank_path)

    detector = AnomalyDetector(config)

    assert detector.memory_bank.shape == (3, detector.hog.getDescriptorSize())


def test_anomaly_vectorized_nearest_neighbor_matches_direct_distances(
    config: dict[str, Any], tmp_path: Path
) -> None:
    frame = np.full((256, 256, 3), 80, dtype=np.uint8)
    memory_bank_path = tmp_path / "reference_memory_bank.pkl"
    AnomalyDetector.save_memory_bank([frame], memory_bank_path, config)
    config["models"]["anomaly"] = str(memory_bank_path)
    detector = AnomalyDetector(config)

    patch_features = detector.extract_features(frame)
    direct_distances = [
        np.min(np.linalg.norm(detector.memory_bank - feature, axis=1))
        for feature in patch_features
    ]
    expected_score = max(direct_distances) / (
        max(direct_distances) + detector.distance_scale
    )

    result = detector.predict(frame)

    assert result["score"] == pytest.approx(expected_score, abs=1e-6)


def test_anomaly_fails_on_missing_pickle(config: dict[str, Any], tmp_path: Path) -> None:
    """Anomaly stage fails at initialization when its configured bank is absent."""
    config["models"]["anomaly"] = str(tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError, match="Anomaly memory bank not found"):
        AnomalyDetector(config)


def test_anomaly_bank_builder_uses_explicit_normal_image_directory(
    config: dict[str, Any], tmp_path: Path
) -> None:
    """Bank generation fails with its selected source directory, not a fallback."""
    normal_dir = tmp_path / "configured-normal-images"
    with pytest.raises(FileNotFoundError) as error:
        build_memory_bank(config, normal_dir)
    assert str(normal_dir) in str(error.value)


def test_anomaly_rejects_empty_memory_bank(config: dict[str, Any], tmp_path: Path) -> None:
    """Memory-bank creation rejects an empty normal-image set."""
    with pytest.raises(ValueError, match="At least one normal image"):
        AnomalyDetector.save_memory_bank(
            [],
            tmp_path / "empty.pkl",
            config,
        )


def test_decision_fails_on_anomaly(config: dict[str, Any]) -> None:
    """An anomaly above the configured threshold fails even without detections."""
    result = DecisionEngine(config).decide(
        [],
        {"score": 0.8, "is_anomaly": True, "available": True},
        "station",
        "camera",
    )
    assert result["result"] == "FAIL"
    assert result["defect"] == "anomaly"


def test_decision_detection_threshold_is_strict(config: dict[str, Any]) -> None:
    """A detection exactly at threshold does not exceed the configured limit."""
    decision = DecisionEngine(config).decide(
        [
            {
                "label": "scratch",
                "confidence": 0.8,
                "cnn_confidence": config["decision_rules"]["fail_confidence_threshold"],
            }
        ],
        {"score": 0.0, "is_anomaly": False, "available": True},
        "station",
        "camera",
    )
    assert decision["result"] == "PASS"


def test_flask_upload_returns_json_and_logs_timings(
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Upload route returns JSON-safe results, saves the image, and logs all stages."""
    import server

    class FakeRouter:
        def run(
            self, frame: np.ndarray, station_id: str, camera_id: str
        ) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
            timings = {
                "stage_1_preprocessing_ms": 1.0,
                "stage_2_detection_ms": 2.0,
                "stage_3_classification_ms": 3.0,
                "stage_4_anomaly_ms": 4.0,
                "stage_5_decision_ms": 5.0,
                "total_ms": 15.0,
            }
            decision = {
                "result": "FAIL",
                "defect": "scratch",
                "confidence": 0.9,
                "severity": "FAIL",
                "station_id": station_id,
                "camera_id": camera_id,
                "anomaly_score": 0.1,
                "anomaly": {"score": 0.1, "is_anomaly": False, "available": True},
                "detections": [
                    {
                        "label": "scratch",
                        "class_id": 0,
                        "confidence": 0.9,
                        "bbox": [1, 1, 6, 6],
                        "defect_type": "scratch",
                        "cnn_confidence": 0.9,
                        "severity": "FAIL",
                        "mask": np.ones((4, 4), dtype=np.uint8),
                    }
                ],
            }
            return frame, decision, timings

    config["runtime"]["data_dir"] = str(tmp_path / "inspection_data")
    app = server.create_app()
    app.config["PRODUCT_CONFIG"] = config
    app.config["PIPELINE_ROUTER"] = FakeRouter()
    client = app.test_client()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", frame)
    assert encoded
    response = client.post(
        "/upload",
        data=buffer.tobytes(),
        headers={"X-Station-ID": "station-test", "X-Camera-ID": "camera-test"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["result"] == "FAIL"
    assert payload["timings_ms"]["stage_4_anomaly_ms"] == 4.0
    assert "mask" not in payload["detections"][0]
    csv_path = tmp_path / "inspection_data" / "logs" / "inspection.csv"
    with csv_path.open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    assert rows[0]["AnomalyScore"] == "0.1"
    assert rows[0]["Stage5(ms)"] == "5.0"
