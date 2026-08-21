"""并行抽帧识别：线程池逐帧运行图像 pipeline。

每个 worker 线程执行一帧的完整识别（内部按任务再并行）；模型实例的
线程安全由各模型实现保证（Phase 2 模块 4），如遇冲突可在模型内加锁。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.vision.pipeline import ImagePipeline


def process_frames(
    pipeline: ImagePipeline,
    frames: list[np.ndarray],
    tasks: list[str] | None,
    options: dict,
    workers: int,
) -> list[dict]:
    """对每帧运行图像 pipeline，返回与 frames 等长的结果列表。"""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda frame: pipeline.run(frame, tasks, options), frames))
