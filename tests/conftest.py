"""pytest 共享夹具：假模型 + 测试应用 + 样例输入。

测试使用假模型替换真实模型（不联网、不加载权重），验证端点、编排、
schema 的完整链路；纯函数与下载器逻辑在各自测试文件中单独覆盖。
"""

from __future__ import annotations

import io
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# 保证从任意 cwd 都能导入 app 包；缓存/自动下载指向工作区，测试确定性
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault(
    "DSH_VISION_CACHE",
    str(Path(__file__).resolve().parent.parent / ".venv" / "pytest-cache"),
)
os.environ.setdefault("DSH_VISION_AUTO_DOWNLOAD", "false")

from app.main import create_app  # noqa: E402
from app.vision.base import BaseVisionModel  # noqa: E402

# 测试工作目录根（os.makedirs 默认 0o777；不用 pytest 的 tmp_path——
# 其 basetemp 的 0o700 目录在本机沙箱下无法再建子目录）
_WORK_ROOT = Path(__file__).resolve().parent.parent / ".venv" / "pytest-work"


class FakeDetector(BaseVisionModel):
    name = "object_detection"
    license = "test"

    def load(self) -> None:
        pass

    def predict(self, image_bgr, options):
        return {"items": [{"label": "person", "confidence": 0.93, "bbox": [10, 10, 50, 80]}]}


class FakeCaptioner(BaseVisionModel):
    name = "captioning"
    license = "test"

    def load(self) -> None:
        pass

    def predict(self, image_bgr, options):
        return {"text": "a test image"}


class FakeScene(BaseVisionModel):
    name = "scene"
    license = "test"

    def load(self) -> None:
        pass

    def predict(self, image_bgr, options):
        return {"label": "outdoor", "confidence": 0.9}


class FakeOcr(BaseVisionModel):
    name = "ocr"
    license = "test"

    def load(self) -> None:
        pass

    def predict(self, image_bgr, options):
        return {"items": [{"text": "HELLO", "confidence": 0.99, "bbox": [0, 0, 30, 10], "lang": "en"}]}


class FakeFaces(BaseVisionModel):
    name = "faces"
    license = "test"

    def load(self) -> None:
        pass

    def predict(self, image_bgr, options):
        return {
            "items": [
                {
                    "bbox": [1, 2, 3, 4],
                    "confidence": 0.98,
                    "landmarks": [[1, 1]],
                    "embedding_md5": "abc",
                }
            ],
        }


FAKES = {
    "object_detection": FakeDetector,
    "captioning": FakeCaptioner,
    "scene": FakeScene,
    "ocr": FakeOcr,
    "faces": FakeFaces,
}


def fake_registrar(registry, settings, cache, downloader) -> None:
    """把全部假模型注册进测试应用的注册表。"""
    for name, factory in FAKES.items():
        registry.register(name, factory, license="test")


@pytest.fixture()
def app():
    """装配了假模型的测试应用。"""
    return create_app(registrar=fake_registrar)


@pytest.fixture()
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture()
def sample_image() -> bytes:
    """64x64 纯色 PNG 字节。"""
    img = Image.fromarray(np.full((64, 64, 3), 100, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def workdir() -> Path:
    """每次测试独立的可写工作目录。"""
    _WORK_ROOT.mkdir(parents=True, exist_ok=True)
    path = _WORK_ROOT / uuid.uuid4().hex[:8]
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def sample_video(workdir) -> str:
    """20 帧移动色块测试视频（MJPG/AVI），返回路径。"""
    import cv2

    path = str(workdir / "sample.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    for i in range(20):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        x = (i * 4) % 48
        frame[16:48, x : x + 16] = (0, 0, 255)
        writer.write(frame)
    writer.release()
    return path
