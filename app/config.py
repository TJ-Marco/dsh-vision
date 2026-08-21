"""应用配置：加载 config.yaml，支持环境变量覆盖（环境变量 > 文件 > 默认值）。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

#: 环境变量名
ENV_CACHE_DIR = "DSH_VISION_CACHE"  # 模型缓存目录
ENV_DEVICE = "DSH_VISION_DEVICE"  # auto | cuda | cpu
ENV_AUTO_DOWNLOAD = "DSH_VISION_AUTO_DOWNLOAD"  # true | false
ENV_CONFIG = "DSH_VISION_CONFIG"  # config.yaml 路径

#: 默认配置文件路径（与 app/ 同级）
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
#: 默认模型缓存目录
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "dsh-vision"


class ServerSettings(BaseModel):
    """HTTP 服务配置。"""

    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 64


class ModelsSettings(BaseModel):
    """模型相关配置。"""

    cache_dir: Path | None = None
    device: str = "auto"
    manifest: str = "models.json"
    auto_download: bool = True
    enabled: dict[str, bool] = Field(
        default_factory=lambda: {
            "object_detection": True,
            "captioning": True,
            "scene": True,
            "ocr": True,
            "faces": True,
        }
    )
    confidence: dict[str, float] = Field(default_factory=lambda: {"default": 0.35})


class OcrSettings(BaseModel):
    """OCR 引擎配置。"""

    engine: str = "paddle"  # paddle | easy
    langs: list[str] = Field(default_factory=lambda: ["ch", "en"])
    use_angle_cls: bool = True


class VideoSettings(BaseModel):
    """视频抽帧配置。"""

    default_fps: float = 1.0
    max_frames: int = 120
    strategy: str = "uniform"  # uniform | scene
    scene_threshold: float = 30.0
    workers: int = 4


class LoggingSettings(BaseModel):
    """日志配置。"""

    level: str = "INFO"


class AppSettings(BaseModel):
    """应用整体配置。"""

    server: ServerSettings = Field(default_factory=ServerSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    def resolve_cache_dir(self) -> Path:
        """按 环境变量 > 配置 > 默认值 的顺序解析模型缓存目录。"""
        env = os.environ.get(ENV_CACHE_DIR)
        if env:
            return Path(env).expanduser()
        if self.models.cache_dir is not None:
            return self.models.cache_dir.expanduser()
        return DEFAULT_CACHE_DIR

    def resolve_device(self) -> str:
        """按 环境变量 > 配置 的顺序解析设备；'auto' 由调用方判定。"""
        env = os.environ.get(ENV_DEVICE)
        if env:
            return env
        return self.models.device

    def resolve_auto_download(self) -> bool:
        """按 环境变量 > 配置 的顺序解析是否自动下载权重。"""
        env = os.environ.get(ENV_AUTO_DOWNLOAD)
        if env is not None:
            return env.strip().lower() in ("1", "true", "yes", "on")
        return self.models.auto_download


def load_settings(path: Path | str | None = None) -> AppSettings:
    """从 YAML 文件加载配置；文件缺失或字段缺省时使用内置默认值。"""
    config_path = (
        Path(path)
        if path is not None
        else Path(
            os.environ.get(ENV_CONFIG, str(DEFAULT_CONFIG_PATH)),
        )
    )
    if not config_path.exists():
        return AppSettings()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppSettings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """进程级配置单例（首次调用后缓存）。"""
    return load_settings()
