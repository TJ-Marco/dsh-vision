"""健康检查与模型清单接口。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app import __version__
from app.api.schemas import HealthResponse, ModelInfo, ModelsResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """存活检查：返回版本、设备与已加载模型。"""
    registry = request.app.state.registry
    loaded = [item["name"] for item in registry.describe() if item["loaded"]]
    return HealthResponse(
        version=__version__,
        device=request.app.state.settings.resolve_device(),
        loaded_models=loaded,
    )


@router.get("/models", response_model=ModelsResponse)
async def models(request: Request) -> ModelsResponse:
    """列出可用模型及其启用/注册/加载状态与许可。"""
    registry = request.app.state.registry
    return ModelsResponse(models=[ModelInfo(**item) for item in registry.describe()])
