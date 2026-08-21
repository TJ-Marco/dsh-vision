"""图像描述：BLIP（Salesforce/blip-image-captioning-base）。

依赖 transformers 与 torch，均懒加载；权重由 transformers 的
``from_pretrained`` 自动下载到模型缓存目录（首次调用）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.errors import ModelUnavailableError
from app.vision.base import BaseVisionModel, resolve_device


class BlipCaptioner(BaseVisionModel):
    """BLIP 图像描述模型。"""

    name = "captioning"
    license = "BSD-3-Clause"
    #: Hugging Face 仓库 id（与 models.json 的 source 一致）
    repo_id = "Salesforce/blip-image-captioning-base"

    def __init__(self, cache, settings) -> None:
        self._cache_dir: Path = cache.model_dir("captioning")
        self._device = resolve_device(settings)
        self._processor = None
        self._model = None

    def load(self) -> None:
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的明确指引
            raise ModelUnavailableError(
                "图像描述需要 transformers+torch：pip install transformers",
            ) from exc
        self._processor = BlipProcessor.from_pretrained(self.repo_id, cache_dir=str(self._cache_dir))
        self._model = BlipForConditionalGeneration.from_pretrained(
            self.repo_id,
            cache_dir=str(self._cache_dir),
        ).to(self._device)

    def predict(self, image_bgr: np.ndarray, options: dict) -> dict:
        if self._model is None:
            self.load()
        pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self._processor(pil_image, return_tensors="pt").to(self._device)
        generated = self._model.generate(**inputs, max_new_tokens=50)
        text = self._processor.decode(generated[0], skip_special_tokens=True)
        return {"text": text.strip()}


def register(registry, settings, cache, downloader=None) -> None:
    """把描述器工厂注册进模型注册表。"""
    registry.register(
        "captioning",
        lambda: BlipCaptioner(cache, settings),
        license=BlipCaptioner.license,
        description="BLIP 图像描述（英文；BSD-3-Clause）",
    )
