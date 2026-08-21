"""图像工具：字节解码、EXIF 方向校正、限幅缩放、URL 下载、关键帧编码。"""

from __future__ import annotations

import base64
import io

import cv2
import httpx
import numpy as np
from PIL import Image, ImageOps

from app.core.errors import DownloadError, InvalidInputError

#: 最大边长：超过则等比缩小（控制推理显存/内存）
MAX_IMAGE_SIDE = 2048
#: URL 下载体积上限（字节）
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


def decode_image(data: bytes) -> np.ndarray:
    """把图片字节解码为 BGR ndarray。

    处理步骤：PIL 解码 → EXIF 方向校正（手机照片旋转修正）→ 转 RGB →
    超限等比缩放 → 转 OpenCV BGR。解码失败抛 :class:`InvalidInputError`。
    """
    try:
        pil_image = Image.open(io.BytesIO(data))
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
    except Exception as exc:  # 非图片或损坏文件
        raise InvalidInputError(f"图片解码失败: {exc}") from exc

    # 最大边限幅，保持宽高比
    width, height = pil_image.size
    longest = max(width, height)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        pil_image = pil_image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    # PIL RGB → OpenCV BGR
    rgb = np.asarray(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def download_bytes(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = 30.0,
) -> bytes:
    """下载 URL 内容并限制体积上限。

    流式读取，超过 ``max_bytes`` 立即中止；网络/HTTP 错误统一转为
    :class:`DownloadError`（HTTP 400）。
    """
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes(chunk_size=1 << 16):
                total += len(chunk)
                if total > max_bytes:
                    raise DownloadError(f"下载内容超过 {max_bytes} 字节上限")
                chunks.append(chunk)
            return b"".join(chunks)
    except httpx.HTTPError as exc:
        raise DownloadError(f"URL 下载失败: {exc}") from exc


def encode_jpeg_base64(image_bgr: np.ndarray, quality: int = 85) -> str:
    """把 BGR 图像编码为 JPEG base64（视频关键帧用）。"""
    ok, buf = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise InvalidInputError("关键帧 JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")
