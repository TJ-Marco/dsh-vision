"""识别算法模块的纯函数与可测部分（不含真实权重/框架依赖）。"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import ModelUnavailableError
from app.vision.object_detection import _postprocess_onnx
from app.vision.ocr import _polygon_to_bbox
from app.vision.scene import SCENE_LABELS

# -- YOLOv8 ONNX 后处理 ---------------------------------------------------


def _synthetic_output() -> np.ndarray:
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    # 高分目标：class0(person)，640 空间 cx=320 cy=320 w=100 h=100
    output[0, 0, 0] = 320.0
    output[0, 1, 0] = 320.0
    output[0, 2, 0] = 100.0
    output[0, 3, 0] = 100.0
    output[0, 4, 0] = 0.9
    # 低分目标：应被阈值过滤
    output[0, 0, 1] = 100.0
    output[0, 1, 1] = 100.0
    output[0, 2, 1] = 50.0
    output[0, 3, 1] = 50.0
    output[0, 4, 1] = 0.1
    return output


def test_onnx_postprocess_threshold_and_decode():
    boxes, scores, ids = _postprocess_onnx(_synthetic_output(), (640, 640), 0.35, 1.0, (0, 0))
    assert len(boxes) == 1
    assert int(ids[0]) == 0
    assert abs(float(scores[0]) - 0.9) < 1e-6
    # cx,cy,w,h → x1,y1,x2,y2 = [270, 270, 370, 370]
    assert boxes[0].tolist() == [270.0, 270.0, 370.0, 370.0]


def test_onnx_postprocess_nms_keeps_high_conf():
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    for idx, (cx, cy, conf) in enumerate([(320, 320, 0.9), (322, 322, 0.5)]):
        output[0, 0, idx] = cx
        output[0, 1, idx] = cy
        output[0, 2, idx] = 100.0
        output[0, 3, idx] = 100.0
        output[0, 4, idx] = conf
    boxes, scores, ids = _postprocess_onnx(output, (640, 640), 0.35, 1.0, (0, 0))
    assert len(boxes) == 1
    assert abs(float(scores[0]) - 0.9) < 1e-6


def test_onnx_postprocess_scale_and_pad_mapping():
    """letterbox 缩放/填充映射回原图坐标（目标框在原图范围内）。"""
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    output[0, 0, 0] = 100.0
    output[0, 1, 0] = 100.0
    output[0, 2, 0] = 40.0
    output[0, 3, 0] = 40.0
    output[0, 4, 0] = 0.8
    # 原图 320x320 放大到 640 空间（scale=0.5）：640 空间框映射回原图坐标
    boxes, _, _ = _postprocess_onnx(output, (320, 320), 0.35, 0.5, (0, 0))
    assert boxes[0].tolist() == [160.0, 160.0, 240.0, 240.0]


# -- OCR 工具 ---------------------------------------------------------------


def test_polygon_to_bbox_normalizes_any_order():
    polygon = [[30, 10], [10, 10], [10, 40], [30, 40]]
    assert _polygon_to_bbox(polygon) == [10.0, 10.0, 30.0, 40.0]


# -- 场景标签 ---------------------------------------------------------------


def test_scene_labels_are_nonempty_strings():
    assert len(SCENE_LABELS) > 5
    assert all(isinstance(label, str) and label for label in SCENE_LABELS)


# -- 依赖缺失时的明确报错 ---------------------------------------------------


def test_captioning_without_transformers_raises(workdir, monkeypatch):
    import builtins

    from app.config import AppSettings
    from app.core.cache import ModelCache
    from app.vision.captioning import BlipCaptioner

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("transformers"):
            raise ImportError("transformers not installed (test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    model = BlipCaptioner(ModelCache(workdir), AppSettings())
    with pytest.raises(ModelUnavailableError):
        model.load()
