"""文字识别（OCR）：PaddleOCR 主选 / EasyOCR 可切换。

引擎由 config.yaml 的 ``ocr.engine`` 决定；两种引擎均懒加载。
PaddleOCR 权重由其内置下载器在首次使用时自动下载到模型缓存；
EasyOCR 同样首次运行时下载到 ~/.EasyOCR（可通过环境变量 EASYOCR_MODULE_PATH 覆盖）。
"""

from __future__ import annotations

import numpy as np

from app.config import AppSettings
from app.core.errors import ModelUnavailableError
from app.vision.base import BaseVisionModel


class OcrEngine(BaseVisionModel):
    """多引擎 OCR 封装（paddle | easy）。"""

    name = "ocr"
    license = "Apache-2.0"

    def __init__(self, settings: AppSettings) -> None:
        self._engine = settings.ocr.engine
        self._langs = settings.ocr.langs
        self._use_angle_cls = settings.ocr.use_angle_cls
        self._model = None

    def load(self) -> None:
        if self._engine == "paddle":
            self._load_paddle()
        elif self._engine == "easy":
            self._load_easy()
        else:
            raise ModelUnavailableError(
                f"未知 OCR 引擎: {self._engine}；config.yaml 可选 paddle|easy",
            )

    def _load_paddle(self) -> None:
        try:
            from paddleocr import PaddleOCR  # 懒加载
        except ImportError as exc:  # pragma: no cover
            raise ModelUnavailableError(
                "PaddleOCR 未安装：pip install -r requirements-ocr-paddle.txt " "（或 requirements.txt 默认已含）",
            ) from exc
        self._model = PaddleOCR(
            use_angle_cls=self._use_angle_cls,
            lang="ch",  # PP-OCRv4 中文模型同时支持中英混合
            show_log=False,
        )

    def _load_easy(self) -> None:
        try:
            import easyocr  # 懒加载
        except ImportError as exc:  # pragma: no cover
            raise ModelUnavailableError(
                "EasyOCR 未安装：pip install -r requirements-ocr-easy.txt",
            ) from exc
        self._model = easyocr.Reader(self._langs)

    def predict(self, image_bgr: np.ndarray, options: dict) -> dict:
        if self._model is None:
            self.load()
        items = self._predict_paddle(image_bgr) if self._engine == "paddle" else self._predict_easy(image_bgr)
        return {"items": items}

    def _predict_paddle(self, image_bgr: np.ndarray) -> list[dict]:
        # PaddleOCR.ocr 返回 list[line]，line = [4点框, (text, score)]；无文字返回 None
        raw = self._model.ocr(image_bgr, cls=self._use_angle_cls)
        items: list[dict] = []
        for line in raw or []:
            if not line:
                continue
            box, (text, score) = line
            items.append(
                {
                    "text": str(text),
                    "confidence": round(float(score), 4),
                    "bbox": _polygon_to_bbox(box),
                    "lang": self._langs[0] if self._langs else "",
                }
            )
        return items

    def _predict_easy(self, image_bgr: np.ndarray) -> list[dict]:
        # easyocr.readtext 返回 [(box_4点, text, score), ...]
        raw = self._model.readtext(image_bgr)
        items: list[dict] = []
        for box, text, score in raw:
            items.append(
                {
                    "text": str(text),
                    "confidence": round(float(score), 4),
                    "bbox": _polygon_to_bbox(box),
                    "lang": self._langs[0] if self._langs else "",
                }
            )
        return items


def _polygon_to_bbox(polygon) -> list[float]:
    """把 4 点多边形（任意顺序）转为 [x1, y1, x2, y2] 轴对齐框。"""
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def register(registry, settings, cache, downloader=None) -> None:
    """把 OCR 引擎工厂注册进模型注册表。"""
    registry.register(
        "ocr",
        lambda: OcrEngine(settings),
        license=OcrEngine.license,
        description=f"OCR（{settings.ocr.engine}；Apache-2.0）",
    )
