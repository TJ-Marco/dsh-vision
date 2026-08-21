"""场景分类：CLIP 零样本分类（openai/clip-vit-base-patch32）。

用一组预置场景描述 prompt 与图像做相似度对比，取最高分标签；
免训练、可随需求扩充标签列表（无需换模型）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.errors import ModelUnavailableError
from app.vision.base import BaseVisionModel, resolve_device

#: 预置场景类别（可按需扩充）
SCENE_LABELS = [
    "indoor",
    "outdoor",
    "city street",
    "natural landscape",
    "office",
    "home interior",
    "classroom",
    "restaurant",
    "vehicle interior",
    "beach",
    "mountain",
    "forest",
    "night scene",
    "text document",
    "computer screen",
    "food on a table",
    "crowd of people",
    "sports field",
]


class ClipSceneClassifier(BaseVisionModel):
    """CLIP 零样本场景分类器。"""

    name = "scene"
    license = "MIT"
    repo_id = "openai/clip-vit-base-patch32"

    def __init__(self, cache, settings) -> None:
        self._cache_dir: Path = cache.model_dir("scene")
        self._device = resolve_device(settings)
        self._processor = None
        self._model = None
        self._prompts = [f"a photo of {label}" for label in SCENE_LABELS]

    def load(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover
            raise ModelUnavailableError(
                "场景分类需要 transformers+torch：pip install transformers",
            ) from exc
        self._processor = CLIPProcessor.from_pretrained(self.repo_id, cache_dir=str(self._cache_dir))
        self._model = CLIPModel.from_pretrained(self.repo_id, cache_dir=str(self._cache_dir)).to(self._device)

    def predict(self, image_bgr: np.ndarray, options: dict) -> dict:
        if self._model is None:
            self.load()
        pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self._processor(
            text=self._prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        ).to(self._device)
        with np.errstate(all="ignore"):
            probs = self._model(**inputs).logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
        best = int(np.argmax(probs))
        return {"label": SCENE_LABELS[best], "confidence": round(float(probs[best]), 4)}


def register(registry, settings, cache, downloader=None) -> None:
    """把场景分类器工厂注册进模型注册表。"""
    registry.register(
        "scene",
        lambda: ClipSceneClassifier(cache, settings),
        license=ClipSceneClassifier.license,
        description="CLIP 零样本场景分类（MIT）",
    )
