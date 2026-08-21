"""抽帧：uniform（ffmpeg 优先 / OpenCV 退化）与 scene（OpenCV 直方图突变）。

返回 ``(frames, timestamps)`` 两个等长列表：
- uniform-ffmpeg：时间戳按目标采样间隔 ``i / fps``；
- uniform/scene-OpenCV：时间戳按源帧率 ``source_index / source_fps``。
抽帧数量始终受 ``max_frames`` 上限约束（超出时均匀保留首尾帧）。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from app.core.errors import InvalidInputError

#: 场景切换检测：H 通道直方图分箱数
_HIST_BINS = 32


def extract_frames(
    video_path: str,
    *,
    fps: float,
    max_frames: int,
    strategy: str,
    scene_threshold: float,
) -> tuple[list[np.ndarray], list[float]]:
    """按策略抽帧并返回 (帧列表, 时间戳列表)。"""
    if strategy == "uniform":
        if shutil.which("ffmpeg"):
            try:
                frames, timestamps = _uniform_ffmpeg(video_path, fps, max_frames)
                if frames:
                    return frames, timestamps
            except Exception:
                # ffmpeg 抽帧失败（编解码器缺失等），退化 OpenCV
                pass
        return _uniform_opencv(video_path, fps, max_frames)
    if strategy == "scene":
        return _scene_opencv(video_path, scene_threshold, max_frames)
    raise InvalidInputError(f"未知抽帧策略: {strategy}；可选 uniform|scene")


def _even_sample(items: list, limit: int) -> list:
    """把列表均匀压缩到 limit 个，保留首尾元素。"""
    count = len(items)
    if count <= limit:
        return items
    indices = [round(index * (count - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indices]


def _uniform_ffmpeg(
    video_path: str,
    fps: float,
    max_frames: int,
) -> tuple[list[np.ndarray], list[float]]:
    """用 ffmpeg 的 fps filter 一次性抽取目标帧率帧序列。"""
    with tempfile.TemporaryDirectory(prefix="dsh-vision-ffmpeg-") as tmp_dir:
        pattern = str(Path(tmp_dir) / "frame_%06d.jpg")
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        files = sorted(Path(tmp_dir).glob("frame_*.jpg"))
        frames = [cv2.imread(str(f)) for f in files]
        frames = [f for f in frames if f is not None]
    frames = _even_sample(frames, max_frames)
    timestamps = [round(index / fps, 3) for index in range(len(frames))]
    if not frames:
        raise InvalidInputError("未能从视频中抽到任何帧（uniform-ffmpeg）")
    return frames, timestamps


def _uniform_opencv(
    video_path: str,
    fps: float,
    max_frames: int,
) -> tuple[list[np.ndarray], list[float]]:
    """OpenCV 均匀采样：按源帧率步进取帧。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise InvalidInputError(f"无法打开视频文件: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, round(source_fps / fps))
    frames: list[np.ndarray] = []
    indices: list[int] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            frames.append(frame)
            indices.append(index)
        index += 1
    cap.release()

    pairs = _even_sample(list(zip(frames, indices, strict=False)), max_frames)
    frames = [f for f, _ in pairs]
    indices = [i for _, i in pairs]
    if not frames:
        raise InvalidInputError("未能从视频中抽到任何帧（uniform-opencv）")
    timestamps = [round(i / source_fps, 3) for i in indices]
    return frames, timestamps


def _scene_opencv(
    video_path: str,
    scene_threshold: float,
    max_frames: int,
) -> tuple[list[np.ndarray], list[float]]:
    """场景切换抽帧：帧间 H 通道直方图卡方距离超过阈值即保留。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise InvalidInputError(f"无法打开视频文件: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0

    frames: list[np.ndarray] = []
    indices: list[int] = []
    previous_hist: np.ndarray | None = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [_HIST_BINS], [0, 180])
        cv2.normalize(hist, hist)
        if previous_hist is None:
            distance = float("inf")
        else:
            distance = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        if distance > scene_threshold:
            frames.append(frame)
            indices.append(index)
        previous_hist = hist
        index += 1
    cap.release()

    if not frames:
        # 全片无场景突变：退化到均匀取首尾与中点，保证有输出
        frames, indices = _fallback_middle_frames(video_path, source_fps, max_frames)

    pairs = _even_sample(list(zip(frames, indices, strict=False)), max_frames)
    frames = [f for f, _ in pairs]
    indices = [i for _, i in pairs]
    if not frames:
        raise InvalidInputError("未能从视频中抽到任何帧（scene）")
    timestamps = [round(i / source_fps, 3) for i in indices]
    return frames, timestamps


def _fallback_middle_frames(
    video_path: str,
    source_fps: float,
    max_frames: int,
) -> tuple[list[np.ndarray], list[int]]:
    """场景策略无突变时：取首、中、尾共至多 max_frames 帧。"""
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = sorted({0, count // 2, count - 1}) if count > 0 else [0]
    frames: list[np.ndarray] = []
    indices: list[int] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in picks:
            frames.append(frame)
            indices.append(index)
        index += 1
    cap.release()
    return frames, indices
