"""/api/video 端点测试（假模型全链路）。"""

from __future__ import annotations


def test_video_full_response(client, sample_video):
    with open(sample_video, "rb") as fh:
        resp = client.post("/api/video", files={"file": ("clip.avi", fh, "video/x-msvideo")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sampled_frames"] > 0
    assert body["duration_sec"] > 0
    first = body["results"][0]
    assert first["timestamp_sec"] == 0.0
    assert first["objects"][0]["label"] == "person"
    # 假模型每帧都检出 person → 聚合为单一事件
    assert len(body["timeline"]) == 1
    assert body["timeline"][0]["event"] == "person"
    assert body["processing_time_ms"] >= 0


def test_video_unknown_strategy_returns_400(client, sample_video):
    with open(sample_video, "rb") as fh:
        resp = client.post(
            "/api/video",
            files={"file": ("clip.avi", fh, "video/x-msvideo")},
            data={"strategy": "bogus"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"


def test_video_json_without_url_returns_400(client):
    resp = client.post("/api/video", json={"tasks": ["objects"]})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"


def test_video_not_a_video_returns_400(client):
    resp = client.post("/api/video", files={"file": ("a.txt", b"not a video", "text/plain")})
    assert resp.status_code == 400


def test_video_with_keyframes(client, sample_video):
    with open(sample_video, "rb") as fh:
        resp = client.post(
            "/api/video",
            files={"file": ("clip.avi", fh, "video/x-msvideo")},
            data={"include_keyframes": "true", "max_frames": "4"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) <= 4
    assert body["results"][0]["keyframe"]  # base64 JPEG 非空
