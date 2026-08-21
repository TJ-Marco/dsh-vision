"""模型下载器测试（网络调用用 monkeypatch 替代，保持确定性）。"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from app.core.cache import ModelCache
from app.core.downloader import ModelDownloader


def _make_cache(workdir: Path) -> ModelCache:
    return ModelCache(workdir / "cache")


def test_manifest_loaded_from_disk(workdir):
    downloader = ModelDownloader(_make_cache(workdir))
    ids = downloader.manifest_ids()
    assert "object_detection" in ids
    assert "faces" in ids


def test_valid_against_recorded_sha(workdir):
    cache = _make_cache(workdir)
    downloader = ModelDownloader(cache)
    dest = cache.model_dir("object_detection") / "yolov8n.pt"
    dest.write_bytes(b"fake-weights")
    digest = cache.compute_sha256(dest)
    cache.record_sha256("object_detection", "yolov8n.pt", digest)
    entry = downloader._manifest["models"]["object_detection"]
    assert downloader._valid("object_detection", "yolov8n.pt", dest, entry)

    # 损坏 → 校验失败
    dest.write_bytes(b"tampered")
    assert not downloader._valid("object_detection", "yolov8n.pt", dest, entry)


def test_download_http_extracts_zip(workdir, monkeypatch):
    """zip 模型包下载后按 extract 路径解压（跳过网络）。"""
    cache = _make_cache(workdir)
    downloader = ModelDownloader(cache)
    model_dir = cache.model_dir("faces")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("det_10g.onnx", b"fake-det")
    (model_dir / "pack.zip").write_bytes(buf.getvalue())

    monkeypatch.setattr(downloader, "_stream_download", lambda url, dest: None)
    entry = {"source": "https://example.com/pack.zip", "filename": "pack.zip", "extract": "models/buffalo_l"}
    downloader._download_http("faces", entry, model_dir)

    assert (model_dir / "models" / "buffalo_l" / "det_10g.onnx").exists()
    assert (model_dir / "models" / "buffalo_l" / "_complete").exists()
    # 幂等：再次调用不报错
    downloader._download_http("faces", entry, model_dir)


def test_ensure_returns_model_dir(workdir):
    cache = _make_cache(workdir)
    downloader = ModelDownloader(cache)
    assert downloader.ensure("object_detection") == cache.model_dir("object_detection")
