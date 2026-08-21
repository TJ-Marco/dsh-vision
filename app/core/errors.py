"""统一业务异常与 HTTP 错误映射。

所有识别链路中可预期的失败都抛 ``VisionError`` 子类，由
:func:`install_error_handlers` 注册的处理器统一转为结构化 JSON 响应，
保证客户端拿到的错误格式一致。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class VisionError(Exception):
    """业务异常基类：携带 HTTP 状态码与机器可读错误码。"""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id


class InvalidInputError(VisionError):
    """输入不合法：解码失败、任务名未知、参数非法。"""

    status_code = 400
    code = "invalid_input"


class DownloadError(VisionError):
    """URL 下载失败（网络错误 / 超过体积上限）。"""

    status_code = 400
    code = "download_failed"


class ModelNotFoundError(VisionError):
    """请求的模型未注册。"""

    status_code = 404
    code = "model_not_found"


class ModelUnavailableError(VisionError):
    """模型未启用或仍在加载中。"""

    status_code = 503
    code = "model_unavailable"


class InferenceError(VisionError):
    """模型推理内部错误。"""

    status_code = 500
    code = "inference_failed"


def install_error_handlers(app: FastAPI) -> None:
    """为 FastAPI 应用注册统一的 VisionError 处理器。"""

    @app.exception_handler(VisionError)
    async def handle_vision_error(request: Request, exc: VisionError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.code,
                "message": exc.message,
                "request_id": exc.request_id,
            },
        )
