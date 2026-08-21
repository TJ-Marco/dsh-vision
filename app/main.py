"""FastAPI 应用入口：装配配置、缓存、模型注册表、编排器与路由。

应用以模块级 ``app = create_app()`` 方式导出，支持 ``uvicorn app.main:app``
直接启动。模型加载为懒加载：应用启动秒级，首个识别请求才触发下载/载入。
"""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.api import routes_health, routes_image, routes_video
from app.config import get_settings
from app.core.cache import create_cache
from app.core.downloader import ModelDownloader
from app.core.errors import install_error_handlers
from app.core.registry import ModelRegistry
from app.vision.pipeline import ImagePipeline

#: 识别算法模块清单：每个模块导出 ``register(registry, settings, cache, downloader)``。
MODEL_MODULES = [
    "app.vision.object_detection",
    "app.vision.captioning",
    "app.vision.scene",
    "app.vision.ocr",
    "app.vision.faces",
]


def _register_models(registry: ModelRegistry, settings, cache, downloader) -> None:
    """导入并触发各识别模块的注册（模型本身懒加载，启动不下载）。"""
    for module_name in MODEL_MODULES:
        module = __import__(module_name, fromlist=["register"])
        module.register(registry, settings, cache, downloader)


def create_app(registrar=None) -> FastAPI:
    """创建并装配 FastAPI 应用。

    :param registrar: 模型注册回调 ``(registry, settings, cache, downloader)``；
        缺省使用内置的五个识别模块。测试可传入假模型注册器。
    """
    settings = get_settings()
    cache = create_cache(settings)
    downloader = ModelDownloader(cache)
    registry = ModelRegistry(settings, cache)
    pipeline = ImagePipeline(registry)

    app = FastAPI(
        title="dsh-vision",
        description="视觉识别服务：图像/视频识别，供 DeepSeek function calling 调用",
        version=__version__,
    )
    app.state.settings = settings
    app.state.cache = cache
    app.state.downloader = downloader
    app.state.registry = registry
    app.state.pipeline = pipeline

    if registrar is None:
        registrar = _register_models
    registrar(registry, settings, cache, downloader)
    install_error_handlers(app)
    app.include_router(routes_health.router)
    app.include_router(routes_image.router)
    app.include_router(routes_video.router)
    return app


#: WSGI/ASGI 入口（uvicorn app.main:app）
app = create_app()
