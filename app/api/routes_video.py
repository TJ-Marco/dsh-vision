"""POST /api/video：视频识别端点（抽帧 + 时间戳，同步 MVP）。

支持 multipart 文件上传或 JSON ``video_url``；按配置（fps / max_frames /
strategy）抽帧，并行识别，输出逐帧结果与聚合时间线。异步任务模式
（jobs）按设计留作后续增强。
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Request

from app.api.form import form_text, parse_bool, parse_float, parse_int, parse_tasks
from app.api.schemas import FrameResult, VideoRequest, VideoResponse
from app.core.errors import InvalidInputError
from app.utils.image import download_bytes, encode_jpeg_base64
from app.video.extractor import extract_frames
from app.video.probe import probe_video
from app.video.timeline import build_timeline
from app.video.worker import process_frames
from app.vision.pipeline import ImagePipeline

router = APIRouter(prefix="/api", tags=["video"])


async def _parse_request(request: Request) -> tuple[VideoRequest, bytes]:
    """按 Content-Type 解析请求，返回 (请求模型, 视频字节)。"""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("file")
        if file is None:
            raise InvalidInputError("multipart 请求必须包含 file 字段")
        payload = VideoRequest(
            video_url=None,
            fps=parse_float(await form_text(form, "fps")),
            max_frames=parse_int(await form_text(form, "max_frames")),
            strategy=await form_text(form, "strategy"),
            tasks=parse_tasks(await form_text(form, "tasks")),
            include_keyframes=parse_bool(await form_text(form, "include_keyframes")),
        )
        raw = await file.read()
        return payload, raw
    payload = VideoRequest.model_validate_json(await request.body())
    if payload.video_url is None:
        raise InvalidInputError("JSON 请求必须包含 video_url")
    return payload, download_bytes(payload.video_url)


@router.post("/video", response_model=VideoResponse)
async def recognize_video(request: Request) -> VideoResponse:
    """视频识别：抽帧并行识别，返回逐帧结果与事件时间线。"""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()

    payload, raw = await _parse_request(request)
    settings = request.app.state.settings

    # 写入临时文件供 OpenCV/ffmpeg 读取（用完即删）
    with tempfile.NamedTemporaryFile(prefix="dsh-vision-", suffix=".mp4", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        meta = probe_video(tmp_path)
        fps = payload.fps or settings.video.default_fps
        max_frames = payload.max_frames or settings.video.max_frames
        strategy = payload.strategy or settings.video.strategy

        frames, timestamps = extract_frames(
            tmp_path,
            fps=fps,
            max_frames=max_frames,
            strategy=strategy,
            scene_threshold=settings.video.scene_threshold,
        )

        pipeline: ImagePipeline = request.app.state.pipeline
        frame_results = process_frames(
            pipeline,
            frames,
            payload.tasks,
            {"min_confidence": None},
            settings.video.workers,
        )
        timeline = build_timeline(frame_results, timestamps)

        results = []
        for index, (frame, result) in enumerate(zip(frames, frame_results, strict=False)):
            keyframe = encode_jpeg_base64(frame) if payload.include_keyframes else None
            results.append(
                FrameResult(
                    frame_index=index,
                    timestamp_sec=timestamps[index],
                    objects=result.get("objects", []),
                    text=result.get("text", []),
                    faces=result.get("faces", []),
                    keyframe=keyframe,
                )
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return VideoResponse(
            request_id=request_id,
            duration_sec=round(meta.duration_sec, 3),
            sampled_frames=len(frames),
            results=results,
            timeline=timeline,
            processing_time_ms=elapsed_ms,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
