"""dsh-vision Python SDK：对外部调用方提供的 HTTP 客户端封装。

用法：:

    from app.sdk import VisionClient

    with VisionClient("http://127.0.0.1:8000") as client:
        result = client.recognize_image("photo.jpg")
        result = client.recognize_image_url("https://example.com/a.jpg", tasks=["objects"])
        result = client.recognize_video("clip.mp4", fps=1.0)

底层使用 httpx；传入自定义 transport 可做无服务单测（httpx.MockTransport）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from app.sdk.errors import VisionClientError


class VisionClient:
    """dsh-vision 服务的同步客户端。"""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """创建客户端。

        :param base_url: 服务地址，如 http://127.0.0.1:8000
        :param timeout: 请求超时（秒）
        :param transport: 自定义 httpx 传输层（测试注入用）
        """
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    # -- 生命周期 ---------------------------------------------------------

    def close(self) -> None:
        """关闭底层连接池。"""
        self._client.close()

    def __enter__(self) -> VisionClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- 系统接口 ---------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """GET /health：版本、设备与已加载模型。"""
        return self._request("GET", "/health")

    def models(self) -> list[dict[str, Any]]:
        """GET /models：模型清单与许可。"""
        return self._request("GET", "/models")["models"]

    # -- 图像识别 ---------------------------------------------------------

    def recognize_image(
        self,
        image: bytes | str | Path,
        *,
        tasks: list[str] | None = None,
        min_confidence: float | None = None,
        ocr_langs: list[str] | None = None,
    ) -> dict[str, Any]:
        """识别一张图片。

        :param image: 图片字节、本地路径，或 http(s) URL（自动走 URL 通道）
        """
        if isinstance(image, str) and image.startswith(("http://", "https://")):
            return self.recognize_image_url(image, tasks=tasks, min_confidence=min_confidence, ocr_langs=ocr_langs)
        return self.recognize_image_bytes(
            _read_bytes(image),
            filename="image.jpg",
            tasks=tasks,
            min_confidence=min_confidence,
            ocr_langs=ocr_langs,
        )

    def recognize_image_url(
        self,
        image_url: str,
        *,
        tasks: list[str] | None = None,
        min_confidence: float | None = None,
        ocr_langs: list[str] | None = None,
    ) -> dict[str, Any]:
        """通过 URL 识别图片（JSON 通道）。"""
        payload: dict[str, Any] = {"image_url": image_url}
        if tasks is not None:
            payload["tasks"] = tasks
        if min_confidence is not None:
            payload["min_confidence"] = min_confidence
        if ocr_langs is not None:
            payload["ocr_langs"] = ocr_langs
        return self._request("POST", "/api/image", json=payload)

    def recognize_image_bytes(
        self,
        data: bytes,
        *,
        filename: str = "image.jpg",
        tasks: list[str] | None = None,
        min_confidence: float | None = None,
        ocr_langs: list[str] | None = None,
    ) -> dict[str, Any]:
        """上传图片字节识别（multipart 通道）。"""
        files: dict[str, Any] = {"file": (filename, data)}
        data_fields: dict[str, str] = {}
        if tasks is not None:
            data_fields["tasks"] = ",".join(tasks)
        if min_confidence is not None:
            data_fields["min_confidence"] = str(min_confidence)
        if ocr_langs is not None:
            data_fields["ocr_langs"] = ",".join(ocr_langs)
        return self._request("POST", "/api/image", files=files, data=data_fields)

    # -- 视频识别 ---------------------------------------------------------

    def recognize_video(
        self,
        video: bytes | str | Path,
        *,
        filename: str = "video.mp4",
        fps: float | None = None,
        max_frames: int | None = None,
        strategy: str | None = None,
        tasks: list[str] | None = None,
        include_keyframes: bool = False,
    ) -> dict[str, Any]:
        """识别一段视频（multipart 上传）。

        :param video: 视频字节或本地路径
        """
        files = {"file": (filename, _read_bytes(video))}
        data_fields: dict[str, str] = {}
        if fps is not None:
            data_fields["fps"] = str(fps)
        if max_frames is not None:
            data_fields["max_frames"] = str(max_frames)
        if strategy is not None:
            data_fields["strategy"] = strategy
        if tasks is not None:
            data_fields["tasks"] = ",".join(tasks)
        if include_keyframes:
            data_fields["include_keyframes"] = "true"
        return self._request("POST", "/api/video", files=files, data=data_fields)

    # -- 内部 -------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        response = self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise VisionClientError(response.status_code, response.text)
        return response.json()


def _read_bytes(value: bytes | str | Path) -> bytes:
    """把字节或路径统一为字节。"""
    if isinstance(value, bytes):
        return value
    return Path(value).read_bytes()
