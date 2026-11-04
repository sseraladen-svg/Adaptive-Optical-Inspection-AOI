"""Tests for tools/check_weights.py and tools/run_folder.py (no torch / ultralytics needed)."""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import check_weights
from tools.run_folder import run_folder
from utils.config_loader import load_config


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    config = copy.deepcopy(load_config("prismatic_cell"))
    config["models"] = {
        "yolo_seg": str(tmp_path / "yolo.pt"),
        "cnn_classifier": str(tmp_path / "cnn.onnx"),
        "anomaly": str(tmp_path / "bank.pkl"),
    }
    return config


def statuses(rows):
    return [s for s, _ in rows]


def test_missing_files_all_fail(cfg):
    rows = check_weights.run_checks(cfg)
    assert statuses(rows) == [check_weights.FAIL] * 3


def _fake_ultralytics(monkeypatch, names):
    class FakeYolo:
        def __init__(self, path):
            self.names = names

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo))


def test_detection_class_mismatch_is_caught(cfg, monkeypatch):
    Path(cfg["models"]["yolo_seg"]).write_bytes(b"x")
    _fake_ultralytics(monkeypatch, {0: "defect"})
    rows = check_weights.check_detection(cfg)
    assert rows[-1][0] == check_weights.FAIL and "retrain" in rows[-1][1]


def test_detection_matching_classes_ok(cfg, monkeypatch):
    Path(cfg["models"]["yolo_seg"]).write_bytes(b"x")
    _fake_ultralytics(monkeypatch, {int(k): v for k, v in cfg["defect_classes"].items()})
    assert statuses(check_weights.check_detection(cfg)) == [check_weights.OK, check_weights.OK]


def test_detection_without_ultralytics_is_skip_not_ok(cfg, monkeypatch):
    Path(cfg["models"]["yolo_seg"]).write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "ultralytics", None)          # makes `import ultralytics` raise ImportError
    assert check_weights.check_detection(cfg)[-1][0] == check_weights.SKIP


def test_anomaly_bank_roundtrip_ok(cfg, tmp_path):
    from stage_4_anomaly.anomaly_detector import build_memory_bank

    images = tmp_path / "normal"
    images.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        cv2.imwrite(str(images / f"n{i}.png"), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    build_memory_bank(cfg, images)
    rows = check_weights.check_anomaly(cfg)
    assert statuses(rows) == [check_weights.OK, check_weights.OK]


def test_anomaly_garbage_file_fails(cfg):
    Path(cfg["models"]["anomaly"]).write_bytes(b"not a pickle")
    assert check_weights.check_anomaly(cfg)[-1][0] == check_weights.FAIL


def _tiny_onnx(path: Path, n_classes: int, size: int) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.eye(3, n_classes, dtype=np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["pooled"], axes=[2, 3], keepdims=0),
         helper.make_node("MatMul", ["pooled", "w"], ["logits"])],
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", n_classes])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_classifier_ok_and_wrong_width_caught(cfg):
    size = cfg["classification"]["input_size"]
    path = Path(cfg["models"]["cnn_classifier"])
    _tiny_onnx(path, len(cfg["defect_classes"]), size)
    assert check_weights.check_classification(cfg)[-1][0] == check_weights.OK
    _tiny_onnx(path, len(cfg["defect_classes"]) - 1, size)            # model with the wrong number of classes
    rows = check_weights.check_classification(cfg)
    assert rows[-1][0] == check_weights.FAIL and "dummy inference failed" in rows[-1][1]


def test_main_exit_code(cfg, monkeypatch):
    monkeypatch.setattr(check_weights, "load_config", lambda product: cfg)
    monkeypatch.setattr(sys, "argv", ["check_weights", "--product", "prismatic_cell"])
    assert check_weights.main() == 1


class FakeRouter:
    def __init__(self, fail: bool):
        self.fail = fail

    def run(self, frame, station_id, camera_id):
        decision = {"result": "FAIL" if self.fail else "PASS", "defect": "stain" if self.fail else "None",
                    "confidence": 0.9, "detections": []}
        return frame, decision, {"total_ms": 1.5}


def test_run_folder_writes_annotated_and_summarises(tmp_path):
    images = tmp_path / "in"
    images.mkdir()
    for name in ("a.jpg", "b.png"):
        cv2.imwrite(str(images / name), np.zeros((32, 32, 3), np.uint8))
    (images / "notes.txt").write_text("ignore me")
    rows = run_folder(FakeRouter(fail=True), images, tmp_path / "out", "s1", "c1")
    assert [r["image"] for r in rows] == ["a.jpg", "b.png"] and all(r["result"] == "FAIL" for r in rows)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a.jpg", "b.jpg"]


def test_run_folder_empty_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        run_folder(FakeRouter(False), tmp_path / "empty", tmp_path / "out", "s", "c")
