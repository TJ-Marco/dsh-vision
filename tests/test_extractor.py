"""抽帧策略单元测试。"""

from __future__ import annotations

import pytest

from app.core.errors import InvalidInputError
from app.video.extractor import extract_frames


def test_uniform_opencv_sampling(sample_video):
    """20 帧 @10fps 按 5fps 采样 → 最多 10 帧，时间戳单调。"""
    frames, timestamps = extract_frames(
        sample_video,
        fps=5.0,
        max_frames=10,
        strategy="uniform",
        scene_threshold=30.0,
    )
    assert 0 < len(frames) <= 10
    assert len(frames) == len(timestamps)
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0.0


def test_uniform_respects_max_frames(sample_video):
    """max_frames 上限生效。"""
    frames, _ = extract_frames(
        sample_video,
        fps=10.0,
        max_frames=4,
        strategy="uniform",
        scene_threshold=30.0,
    )
    assert len(frames) <= 4


def test_scene_strategy_produces_frames(sample_video):
    """移动色块应触发直方图突变，产出至少一帧。"""
    frames, timestamps = extract_frames(
        sample_video,
        fps=1.0,
        max_frames=10,
        strategy="scene",
        scene_threshold=0.1,
    )
    assert len(frames) > 0
    assert len(frames) == len(timestamps)


def test_unknown_strategy_raises(sample_video):
    with pytest.raises(InvalidInputError):
        extract_frames(
            sample_video,
            fps=1.0,
            max_frames=10,
            strategy="bogus",
            scene_threshold=30.0,
        )
