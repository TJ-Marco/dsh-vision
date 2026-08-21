"""时间线聚合单元测试。"""

from __future__ import annotations

from app.video.timeline import build_timeline


def test_single_continuous_run():
    results = [
        {"objects": [{"label": "car", "confidence": 0.9}]},
        {"objects": [{"label": "car", "confidence": 0.95}]},
        {"objects": [{"label": "car", "confidence": 0.8}]},
    ]
    timeline = build_timeline(results, [0.0, 0.5, 1.0])
    assert timeline == [
        {
            "event": "car",
            "start_sec": 0.0,
            "end_sec": 1.0,
            "max_confidence": 0.95,
        }
    ]


def test_gap_splits_runs():
    """中间断帧 → 拆成两个事件。"""
    results = [
        {"objects": [{"label": "car", "confidence": 0.9}]},
        {"objects": []},
        {"objects": [{"label": "car", "confidence": 0.6}]},
    ]
    timeline = build_timeline(results, [0.0, 1.0, 2.0])
    assert [e["event"] for e in timeline] == ["car", "car"]
    assert timeline[0]["end_sec"] == 0.0
    assert timeline[1]["start_sec"] == 2.0


def test_multiple_labels_and_empty():
    results = [
        {"objects": [{"label": "car", "confidence": 0.9}, {"label": "person", "confidence": 0.7}]},
        {"objects": [{"label": "car", "confidence": 0.8}]},
        {"objects": []},
    ]
    timeline = build_timeline(results, [0.0, 0.5, 1.0])
    events = {e["event"]: e for e in timeline}
    assert events["car"]["end_sec"] == 0.5
    assert events["person"]["start_sec"] == 0.0


def test_empty_input():
    assert build_timeline([], []) == []
