"""模型缓存目录管理：路径解析、目录初始化、SHA256 校验记录。

模型文件不入库（见 .gitignore），首次运行时由下载器（Phase 2 模块 5）
下载到缓存目录；本模块负责目录的创建与校验值的持久化，供启动时快速
核验已下载文件是否完整。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.config import AppSettings


class ModelCache:
    """管理模型文件与校验记录的缓存目录。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._sha_file = root / ".sha256.json"

    def model_dir(self, model_id: str) -> Path:
        """返回某模型专用子目录（不存在则创建）。"""
        path = self.root / model_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record_sha256(self, model_id: str, file_name: str, digest: str) -> None:
        """记录一次成功下载的校验值，供后续启动快速核验。"""
        records = self._load_shas()
        records.setdefault(model_id, {})[file_name] = digest
        self._sha_file.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def sha256_of(self, model_id: str, file_name: str) -> str | None:
        """读取此前记录的校验值；从未记录返回 None。"""
        return self._load_shas().get(model_id, {}).get(file_name)

    def _load_shas(self) -> dict:
        if not self._sha_file.exists():
            return {}
        try:
            data = json.loads(self._sha_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            # 记录文件损坏时降级为空记录，不阻塞服务
            return {}

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """流式计算文件 SHA256（对大文件友好，不整体载入内存）。"""
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


def create_cache(settings: AppSettings) -> ModelCache:
    """按配置创建缓存实例。"""
    return ModelCache(settings.resolve_cache_dir())
