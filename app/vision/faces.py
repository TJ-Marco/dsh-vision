"""人脸检测与特征：InsightFace（buffalo_l）。

buffalo_l 模型包由 InsightFace 首次运行时自动下载到缓存目录
``<cache>/faces/models/buffalo_l``。响应只返回特征摘要（embedding 的
MD5），不返回原始 512 维向量，避免高敏生物特征进入对话上下文。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from app.core.errors import ModelUnavailableError
from app.vision.base import BaseVisionModel


class FaceAnalyzer(BaseVisionModel):
    """InsightFace 人脸检测 + 特征提取。"""

    name = "faces"
    license = "MIT"
    #: InsightFace 模型包名（与 models.json 一致）
    model_pack = "buffalo_l"

    def __init__(self, cache, settings=None, downloader=None) -> None:
        # InsightFace 按 root/models/<pack> 布局下载，root 指向缓存目录
        self._root: Path = cache.model_dir("faces")
        self._auto_download = settings.resolve_auto_download() if settings is not None else True
        self._downloader = downloader
        self._app = None

    def load(self) -> None:
        if self._auto_download and self._downloader is not None:
            try:
                self._downloader.ensure("faces")
            except Exception as exc:  # noqa: BLE001
                raise ModelUnavailableError(f"人脸模型下载失败: {exc}") from exc
        try:
            from insightface.app import FaceAnalysis  # 懒加载
        except ImportError as exc:  # pragma: no cover
            raise ModelUnavailableError(
                "人脸识别需要 insightface：pip install insightface",
            ) from exc
        self._app = FaceAnalysis(
            name=self.model_pack,
            root=str(self._root),
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 = CPU

    def predict(self, image_bgr: np.ndarray, options: dict) -> dict:
        if self._app is None:
            self.load()
        faces = self._app.get(image_bgr)
        items = []
        for face in faces:
            # buffalo_l 输出 106 点二维关键点；部分模型为 68 点三维
            landmarks = getattr(face, "landmark_2d_106", None) or getattr(face, "landmark_3d_68", None)
            embedding = np.asarray(face.embedding, dtype=np.float32)
            items.append(
                {
                    "bbox": [round(float(v), 1) for v in face.bbox],
                    "confidence": round(float(face.det_score), 4),
                    "landmarks": [
                        [round(float(point[0]), 1), round(float(point[1]), 1)] for point in (landmarks or [])
                    ],
                    "embedding_md5": hashlib.md5(embedding.tobytes()).hexdigest(),
                }
            )
        return {"items": items}


def register(registry, settings, cache, downloader=None) -> None:
    """把人脸分析器工厂注册进模型注册表。"""
    registry.register(
        "faces",
        lambda: FaceAnalyzer(cache, settings, downloader),
        license=FaceAnalyzer.license,
        description="InsightFace 人脸检测（buffalo_l；MIT）",
    )
