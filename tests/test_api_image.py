"""/api/image 端点测试（假模型全链路）。"""

from __future__ import annotations

import app.api.routes_image as routes_image  # noqa: E402


def test_health_and_models(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    models = client.get("/models")
    assert models.status_code == 200
    names = {m["name"] for m in models.json()["models"]}
    assert names == {"object_detection", "captioning", "scene", "ocr", "faces"}
    assert all(m["registered"] for m in models.json()["models"])


def test_image_multipart_full_response(client, sample_image):
    resp = client.post("/api/image", files={"file": ("a.png", sample_image, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["width"] == 64 and body["height"] == 64
    assert body["objects"] == [{"label": "person", "confidence": 0.93, "bbox": [10.0, 10.0, 50.0, 80.0]}]
    assert body["caption"] == "a test image"
    assert body["scene"] == {"label": "outdoor", "confidence": 0.9}
    assert body["text"] == [{"text": "HELLO", "confidence": 0.99, "bbox": [0.0, 0.0, 30.0, 10.0], "lang": "en"}]
    assert body["faces"][0]["embedding_md5"] == "abc"
    assert set(body["models_used"]) == {"objects", "caption", "scene", "text", "faces"}
    assert body["processing_time_ms"] >= 0


def test_image_partial_tasks(client, sample_image):
    resp = client.post(
        "/api/image",
        files={"file": ("a.png", sample_image, "image/png")},
        data={"tasks": "objects"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["objects"] and not body["caption"] and body["text"] == []
    assert set(body["models_used"]) == {"objects"}


def test_unknown_task_returns_400(client, sample_image):
    resp = client.post(
        "/api/image",
        files={"file": ("a.png", sample_image, "image/png")},
        data={"tasks": "bogus"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"


def test_json_without_url_returns_400(client):
    resp = client.post("/api/image", json={"tasks": ["objects"]})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"


def test_json_image_url_uses_downloaded_bytes(client, sample_image, monkeypatch):
    """JSON 通道：image_url 经下载器取字节后走同一识别链路。"""
    monkeypatch.setattr(routes_image, "download_bytes", lambda url: sample_image)
    resp = client.post("/api/image", json={"image_url": "https://example.com/a.png"})
    assert resp.status_code == 200
    assert resp.json()["caption"] == "a test image"


def test_non_image_returns_400(client):
    resp = client.post("/api/image", files={"file": ("a.txt", b"not an image", "text/plain")})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"
