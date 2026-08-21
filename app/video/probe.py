"""视频元数据探测：优先 ffprobe，退化用 OpenCV。

ffprobe 提供精确的容器时长与真实帧率；未安装 ffmpeg 时用 OpenCV
VideoCapture 的属性读取（可满足抽帧与时间戳需求）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

import cv2

from app.core.errors import InvalidInputError


@dataclass(frozen=True)
class VideoMeta:
    """视频元数据。"""

    path: str
    duration_sec: float
    width: int
    height: int
    fps: float
    source: str  # "ffprobe" | "opencv"


def probe_video(path: str) -> VideoMeta:
    """探测视频元数据；ffprobe 失败或缺失时退化到 OpenCV。"""
    if shutil.which("ffprobe"):
        try:
            return _probe_ffprobe(path)
        except Exception:
            # ffprobe 存在但解析失败（个别容器），退化到 OpenCV
            pass
    return _probe_opencv(path)


def _probe_ffprobe(path: str) -> VideoMeta:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    result.check_returncode()
    data = json.loads(result.stdout)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise InvalidInputError("视频中没有视频流")
    stream = streams[0]
    fps = _parse_rational(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    duration = float(
        data.get("format", {}).get("duration") or stream.get("duration") or 0.0,
    )
    if duration <= 0 and fps > 0:
        frame_count = int(stream.get("nb_frames") or 0)
        duration = frame_count / fps
    return VideoMeta(
        path=path,
        duration_sec=duration,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        source="ffprobe",
    )


def _parse_rational(value: str) -> float:
    """解析 ffprobe 的 "30000/1001" 形式帧率。"""
    try:
        num, _, den = value.partition("/")
        den = den or "1"
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _probe_opencv(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise InvalidInputError(f"无法打开视频文件: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        raise InvalidInputError("无法读取视频帧率（文件可能不是有效视频）")
    duration = (count / fps) if count > 0 else 0.0
    return VideoMeta(path=path, duration_sec=duration, width=width, height=height, fps=fps, source="opencv")
