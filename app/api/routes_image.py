"""POST /api/image：图像识别端点。

支持两种输入形式（按 Content-Type 自动区分）：
- ``multipart/form-data``：file 字段上传图片 + tasks/min_confidence/ocr_langs 表单字段；
- ``application/json``：``{"image_url": "...", "tasks": [...]}``。

识别算法模块尚未接入（Phase 2 模块 4）时，请求会得到 404 model_not_found
的明确提示，接口链路本身可完整运行与测试。
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Request

from app.api.form import form_text, parse_float, parse_tasks
from app.api.schemas import ImageRequest, ImageResponse
from app.core.errors import InvalidInputError
from app.utils.image import decode_image, download_bytes
from app.vision.pipeline import ImagePipeline

router = APIRouter(prefix="/api", tags=["image"])


def _build_options(payload: ImageRequest) -> dict:
    """从请求构造透传给模型的选项字典。"""
    options: dict = {"min_confidence": payload.min_confidence}
    if payload.ocr_langs is not None:
        options["ocr_langs"] = payload.ocr_langs
    return options


async def _parse_request(request: Request) -> tuple[ImageRequest, bytes]:
    """按 Content-Type 解析请求，返回 (请求模型, 图片字节)。"""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("file")
        if file is None:
            raise InvalidInputError("multipart 请求必须包含 file 字段")
        payload = ImageRequest(
            image_url=None,
            tasks=parse_tasks(await form_text(form, "tasks")),
            min_confidence=parse_float(await form_text(form, "min_confidence")),
            ocr_langs=parse_tasks(await form_text(form, "ocr_langs")),
        )
        raw = await file.read()
        return payload, raw
    payload = ImageRequest.model_validate_json(await request.body())
    if payload.image_url is None:
        raise InvalidInputError("JSON 请求必须包含 image_url")
    return payload, download_bytes(payload.image_url)


@router.post("/image", response_model=ImageResponse)
async def recognize_image(request: Request) -> ImageResponse:
    """图像识别：返回目标、场景、文字、人脸、描述（按请求任务）。"""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()

    payload, raw = await _parse_request(request)
    image_bgr = decode_image(raw)
    height, width = image_bgr.shape[:2]

    pipeline: ImagePipeline = request.app.state.pipeline
    results = pipeline.run(image_bgr, payload.tasks, _build_options(payload))

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return ImageResponse(
        request_id=request_id,
        width=width,
        height=height,
        **results,  # objects/caption/scene/text/faces（pipeline 按响应字段名返回）
        processing_time_ms=elapsed_ms,
    )
