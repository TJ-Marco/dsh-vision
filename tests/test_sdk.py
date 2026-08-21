"""Python SDK 测试：用 httpx.MockTransport 模拟服务端，无真实网络。"""

from __future__ import annotations

import httpx
import pytest

from app.sdk import VisionClient, VisionClientError

BASE = "http://vision.test"


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _ok_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/image":
        return httpx.Response(200, json={"request_id": "x", "width": 64, "height": 64, "objects": []})
    if request.url.path == "/api/video":
        return httpx.Response(
            200, json={"request_id": "y", "duration_sec": 3.0, "sampled_frames": 2, "results": [], "timeline": []}
        )
    if request.url.path == "/models":
        return httpx.Response(200, json={"models": []})
    return httpx.Response(200, json={"status": "ok"})


def test_health_and_models():
    with VisionClient(BASE, transport=_mock_transport(_ok_handler)) as client:
        assert client.health()["status"] == "ok"
        assert isinstance(client.models(), list)


def test_recognize_image_bytes_sends_multipart():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["content_type"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"request_id": "x", "width": 1, "height": 1, "objects": []})

    with VisionClient(BASE, transport=_mock_transport(handler)) as client:
        client.recognize_image_bytes(b"fake-image", filename="a.png", tasks=["objects"])
    assert seen["path"] == "/api/image"
    assert "multipart/form-data" in seen["content_type"]


def test_recognize_image_url_sends_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["json"] = request.read().decode()
        return httpx.Response(200, json={"request_id": "x", "width": 1, "height": 1, "objects": []})

    with VisionClient(BASE, transport=_mock_transport(handler)) as client:
        client.recognize_image_url("https://example.com/a.jpg", tasks=["ocr"])
    assert seen["path"] == "/api/image"
    assert "image_url" in seen["json"]
    assert "ocr" in seen["json"]


def test_recognize_video_sends_multipart():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(
            200, json={"request_id": "y", "duration_sec": 1.0, "sampled_frames": 1, "results": [], "timeline": []}
        )

    with VisionClient(BASE, transport=_mock_transport(handler)) as client:
        client.recognize_video(b"fake-video", fps=1.0)
    assert seen["path"] == "/api/video"


def test_error_response_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_input", "message": "bad"})

    with (
        VisionClient(BASE, transport=_mock_transport(handler)) as client,
        pytest.raises(VisionClientError) as exc_info,
    ):
        client.recognize_image_bytes(b"x")
    assert exc_info.value.status_code == 400
