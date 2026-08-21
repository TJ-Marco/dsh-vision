"""视觉模型统一抽象接口。

所有识别模型实现本接口：输入单张 BGR ndarray（OpenCV 约定），输出
JSON 安全（可被 json.dumps 序列化）的字典。统一接口保证模型可替换
（换实现 = 换工厂），也便于单测时注入 mock。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseVisionModel(ABC):
    """所有识别模型的统一接口。

    结果字典约定（与 app/vision/pipeline.py 的规范化逻辑对应）：
    - 列表型任务（目标/文字/人脸）返回 ``{"items": [...]}``；
    - 图像描述返回 ``{"text": "..."}``；
    - 场景分类返回 ``{"label": "...", "confidence": 0.8}``。
    """

    #: 模型在注册表中的名称（与 config.models.enabled 键一致）
    name: str = ""
    #: 模型许可（用于 /models 接口与 README 展示）
    license: str = ""

    @abstractmethod
    def load(self) -> None:
        """加载权重与初始化（懒加载，注册表首次 get 时调用）。"""

    @abstractmethod
    def predict(self, image_bgr: np.ndarray, options: dict) -> dict:
        """对单张 BGR 图像推理，返回 JSON 安全的结果字典。

        :param image_bgr: BGR 三通道图像（OpenCV 约定）
        :param options: 每次请求的选项（min_confidence、ocr_langs 等）
        """


def resolve_device(settings) -> str:
    """按配置解析实际推理设备（cuda|cpu）。

    - ``auto``：有可用 CUDA 用 GPU，否则 CPU；
    - ``cuda`` / ``cpu``：强制指定。
    """
    requested = settings.resolve_device()
    if requested == "cuda":
        return "cuda"
    if requested == "auto":
        try:
            import torch  # 懒探测，避免无 torch 部署直接报错

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return "cpu"


def default_threshold(settings) -> float:
    """读取 config.yaml 的全局置信度阈值。"""
    return float(settings.models.confidence.get("default", 0.35))
