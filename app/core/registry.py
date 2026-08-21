"""模型注册表：名称 → 模型工厂的懒加载单例中心。

识别算法模块（Phase 2 模块 4）通过 :meth:`ModelRegistry.register` 把各自的
工厂函数接入；API 层只依赖注册表，因此"替换模型 = 换一个工厂"，与调用方
完全解耦。
"""

from __future__ import annotations

from collections.abc import Callable

from app.config import AppSettings
from app.core.cache import ModelCache
from app.core.errors import ModelNotFoundError, ModelUnavailableError
from app.vision.base import BaseVisionModel

#: 模型工厂：返回一个 BaseVisionModel 实例
ModelFactory = Callable[[], BaseVisionModel]


class ModelRegistry:
    """注册并懒加载各识别模型；未启用/未注册的任务给出明确错误。"""

    def __init__(self, settings: AppSettings, cache: ModelCache) -> None:
        self._settings = settings
        self._cache = cache
        self._factories: dict[str, ModelFactory] = {}
        self._meta: dict[str, dict] = {}
        self._instances: dict[str, BaseVisionModel] = {}

    def register(
        self,
        name: str,
        factory: ModelFactory,
        *,
        license: str = "",
        description: str = "",
    ) -> None:
        """注册一个模型工厂及其元数据（许可用于 /models 与 README 展示）。"""
        self._factories[name] = factory
        self._meta[name] = {"license": license, "description": description}

    def is_enabled(self, name: str) -> bool:
        """该模型是否在 config.yaml 中启用。"""
        return self._settings.models.enabled.get(name, True)

    def get(self, name: str) -> BaseVisionModel:
        """获取（必要时加载）模型实例；失败抛出带清晰信息的业务异常。"""
        if not self.is_enabled(name):
            raise ModelUnavailableError(
                f"模型 {name} 未启用（config.yaml -> models.enabled）",
            )
        factory = self._factories.get(name)
        if factory is None:
            raise ModelNotFoundError(
                f"模型 {name} 未注册；识别算法模块将在 Phase 2 模块 4 接入",
            )
        instance = self._instances.get(name)
        if instance is None:
            instance = factory()
            instance.load()
            self._instances[name] = instance
        return instance

    def describe(self) -> list[dict]:
        """供 GET /models 接口展示的模型元数据（不触发加载）。"""
        names = set(self._factories) | set(self._settings.models.enabled)
        return [
            {
                "name": name,
                "enabled": self.is_enabled(name),
                "registered": name in self._factories,
                "loaded": name in self._instances,
                "license": self._meta.get(name, {}).get("license", ""),
            }
            for name in sorted(names)
        ]

    def describe_name(self, name: str) -> str:
        """返回模型实例的展示名（未加载时为注册名）。"""
        instance = self._instances.get(name)
        return instance.name if instance is not None else name
