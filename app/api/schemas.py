"""请求/响应 Pydantic 模型。

这些模型是 FastAPI 生成 OpenAPI（/openapi.json）的基础，也是 DeepSeek
function calling 注册工具参数时的 JSON Schema 来源（见 DESIGN.md 第 9 节）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: 可选任务名（用于文档；合法性由 app/vision/pipeline.py 统一校验）
TASK_NAMES = ("objects", "caption", "scene", "ocr", "faces")
TASK_NAMES_DESC = "可选: " + ", ".join(TASK_NAMES) + "；缺省为全部启用项"


class ImageRequest(BaseModel):
    """POST /api/image 的 JSON 请求体；multipart 表单字段与其一一对应。"""

    image_url: str | None = Field(default=None, description="图片 URL；与文件上传二选一")
    tasks: list[str] | None = Field(default=None, description=TASK_NAMES_DESC)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="置信度阈值")
    ocr_langs: list[str] | None = Field(default=None, description="OCR 语言覆盖")


class VideoRequest(BaseModel):
    """POST /api/video 的 JSON 请求体；multipart 表单字段与其一一对应。"""

    video_url: str | None = Field(default=None, description="视频 URL；与文件上传二选一")
    fps: float | None = Field(default=None, gt=0.0, le=30.0, description="目标采样帧率")
    max_frames: int | None = Field(default=None, ge=1, le=1000, description="抽帧上限")
    strategy: str | None = Field(default=None, description="抽帧策略: uniform | scene（由抽取器校验）")
    tasks: list[str] | None = Field(default=None, description=TASK_NAMES_DESC)
    include_keyframes: bool = Field(default=False, description="是否返回关键帧 base64")


class ObjectItem(BaseModel):
    """单个检测目标。"""

    label: str
    confidence: float
    bbox: list[float] = Field(description="[x1, y1, x2, y2]，像素坐标")


class SceneItem(BaseModel):
    """场景分类结果。"""

    label: str
    confidence: float


class TextItem(BaseModel):
    """单条 OCR 识别文本。"""

    text: str
    confidence: float
    bbox: list[float]
    lang: str = ""


class FaceItem(BaseModel):
    """单张人脸（仅返回特征摘要，不返回原始向量）。"""

    bbox: list[float]
    confidence: float
    landmarks: list[list[float]] = Field(default_factory=list)
    embedding_md5: str = ""


class ImageResponse(BaseModel):
    """POST /api/image 响应。"""

    request_id: str
    width: int
    height: int
    objects: list[ObjectItem] = Field(default_factory=list)
    caption: str = ""
    scene: SceneItem | None = None
    text: list[TextItem] = Field(default_factory=list)
    faces: list[FaceItem] = Field(default_factory=list)
    models_used: dict[str, str] = Field(default_factory=dict)
    processing_time_ms: int = 0


class FrameResult(BaseModel):
    """视频单帧识别结果。"""

    frame_index: int
    timestamp_sec: float
    objects: list[ObjectItem] = Field(default_factory=list)
    text: list[TextItem] = Field(default_factory=list)
    faces: list[FaceItem] = Field(default_factory=list)
    keyframe: str | None = Field(default=None, description="关键帧 JPEG base64（可选）")


class TimelineEvent(BaseModel):
    """时间线聚合事件：同一目标连续出现的时间段。"""

    event: str
    start_sec: float
    end_sec: float
    max_confidence: float


class VideoResponse(BaseModel):
    """POST /api/video 响应。"""

    request_id: str
    duration_sec: float
    sampled_frames: int
    results: list[FrameResult] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    processing_time_ms: int = 0


class HealthResponse(BaseModel):
    """GET /health 响应。"""

    status: str = "ok"
    version: str
    device: str
    loaded_models: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """GET /models 中单个模型的信息。"""

    name: str
    enabled: bool
    registered: bool
    loaded: bool
    license: str = ""


class ModelsResponse(BaseModel):
    """GET /models 响应。"""

    models: list[ModelInfo]
