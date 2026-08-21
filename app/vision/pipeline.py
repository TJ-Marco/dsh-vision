"""单图多任务编排：任务名 → 模型 → 并行推理 → 规范化合并。

API 层只认识任务名（objects/caption/scene/ocr/faces），本模块负责：
1. 把任务名映射到注册表模型名（解耦命名）；
2. 并行执行选中的模型；
3. 把模型结果规范化为 ImageResponse 各字段的形状。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.core.errors import InvalidInputError
from app.core.registry import ModelRegistry

#: 任务名 → 注册表模型名
TASK_TO_MODEL = {
    "objects": "object_detection",
    "caption": "captioning",
    "scene": "scene",
    "ocr": "ocr",
    "faces": "faces",
}

#: 任务名 → ImageResponse 字段名
TASK_FIELD = {
    "objects": "objects",
    "caption": "caption",
    "scene": "scene",
    "ocr": "text",
    "faces": "faces",
}

#: 全部可选任务（用于校验与文档）
ALL_TASKS = tuple(TASK_TO_MODEL)


class ImagePipeline:
    """把图像任务分发给注册的模型并合并结果。"""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def resolve_tasks(self, requested: list[str] | None) -> list[str]:
        """解析请求的任务列表；缺省返回全部已启用任务。"""
        if requested is None:
            return [task for task in ALL_TASKS if self._registry.is_enabled(TASK_TO_MODEL[task])]
        unknown = [task for task in requested if task not in TASK_TO_MODEL]
        if unknown:
            raise InvalidInputError(
                f"未知任务: {', '.join(unknown)}；可选: {', '.join(ALL_TASKS)}",
            )
        # 去重并保持请求顺序
        return list(dict.fromkeys(requested))

    def run(self, image_bgr: np.ndarray, tasks: list[str] | None, options: dict) -> dict:
        """并行执行选中任务并返回合并结果（键为 ImageResponse 字段名）。

        :param image_bgr: BGR 图像
        :param tasks: 任务名列表；None = 全部启用项
        :param options: 透传给模型的请求选项
        :return: {"objects": [...], "caption": str, "scene": {...}|None, ...}
        """
        selected = self.resolve_tasks(tasks)
        if not selected:
            raise InvalidInputError("没有可执行的任务（相关模型均未启用）")

        def run_one(task: str) -> tuple[str, dict]:
            model = self._registry.get(TASK_TO_MODEL[task])
            return task, model.predict(image_bgr, options)

        results: dict[str, dict] = {}
        models_used: dict[str, str] = {}
        # 注意：模型实例的线程安全由各模型实现保证（模块 4）；如遇冲突，
        # 可在各模型内加锁或改为串行执行。
        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            for task, payload in pool.map(run_one, selected):
                results[TASK_FIELD[task]] = self._normalize(task, payload)
                models_used[TASK_FIELD[task]] = self._registry.describe_name(TASK_TO_MODEL[task])
        results["models_used"] = models_used
        return results

    @staticmethod
    def _normalize(task: str, payload: dict) -> dict | list | str | None:
        """把模型结果规范化为响应字段形状。"""
        if task == "caption":
            return payload.get("text", "")
        if task == "scene":
            # 无场景判定时返回 None（响应中为 null）
            return payload if payload else None
        return payload.get("items", [])
