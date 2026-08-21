"""模型下载器：按 models.json 清单把权重下载到缓存目录。

三种来源形式：
- ``http(s)://`` 直链（YOLOv8 .pt、InsightFace 模型包）：分段断点续传 + SHA256 校验；
- ``huggingface://<repo>``（BLIP/CLIP）：有 huggingface_hub 时预下载整个仓库，
  否则由 transformers 首次 ``from_pretrained`` 时自行下载；
- ``paddle://``（OCR）：由 PaddleOCR 内置下载器首次使用时自行下载。

下载完成的文件会记录实际 SHA256 到缓存 ``.sha256.json``，后续启动快速校验，
避免重复下载与静默损坏。
"""

from __future__ import annotations

import json
import socket
import time
import urllib.request
import zipfile
from pathlib import Path

from app.core.cache import ModelCache
from app.core.errors import DownloadError

socket.setdefaulttimeout(30)

#: 分块与单连接上限（超过则断开换新连接续传，绕代理单连接限制）
_CHUNK = 1 << 20
_SEGMENT_CAP = 12 << 20
#: 单个文件下载总时长预算（秒）
_DOWNLOAD_TIMEOUT_SEC = 600

#: 项目根目录下的模型清单文件
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "models.json"


class ModelDownloader:
    """按清单下载/校验模型权重的下载器。"""

    def __init__(self, cache: ModelCache, manifest_path: Path | None = None) -> None:
        self._cache = cache
        self._manifest_path = manifest_path or DEFAULT_MANIFEST_PATH
        self._manifest = self._load_manifest()

    # -- 对外接口 ---------------------------------------------------------

    def ensure(self, model_id: str) -> Path:
        """确保模型权重就绪（缺失则下载）；返回模型缓存目录。"""
        entry = self._manifest.get("models", {}).get(model_id)
        model_dir = self._cache.model_dir(model_id)
        if entry is None:
            return model_dir  # 清单外的模型：交给其自身加载器
        source = entry.get("source", "")
        if source.startswith(("http://", "https://")):
            self._download_http(model_id, entry, model_dir)
        elif source.startswith("huggingface://"):
            self._download_huggingface(entry, model_dir)
        # paddle:// 及其他：由对应框架首次使用时自行下载
        return model_dir

    def manifest_ids(self) -> list[str]:
        """返回清单中的模型 id 列表。"""
        return list(self._manifest.get("models", {}))

    # -- 来源处理 ---------------------------------------------------------

    def _download_http(self, model_id: str, entry: dict, model_dir: Path) -> Path:
        filename = entry.get("filename") or entry["source"].rsplit("/", 1)[-1]
        dest = model_dir / filename
        if not (dest.exists() and self._valid(model_id, filename, dest, entry)):
            self._stream_download(entry["source"], dest)
            if not self._valid(model_id, filename, dest, entry):
                raise DownloadError(f"下载文件校验失败: {dest}")
            # 记录实际校验值，供后续启动快速核验
            self._cache.record_sha256(model_id, filename, self._cache.compute_sha256(dest))

        # zip 模型包解压（幂等：目标目录缺完成标记才解，如 InsightFace buffalo_l）
        extract = entry.get("extract")
        if extract and dest.suffix.lower() == ".zip":
            target = model_dir / extract
            marker = target / "_complete"
            if not marker.exists():
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(target)
                marker.write_text("ok", encoding="utf-8")
        return dest

    def _download_huggingface(self, entry: dict, model_dir: Path) -> None:
        repo = entry["source"].split("huggingface://", 1)[1].rstrip("/")
        try:
            from huggingface_hub import snapshot_download  # 可选依赖
        except ImportError:
            return  # transformers 首次 from_pretrained 时自行下载
        snapshot_download(repo_id=repo, cache_dir=str(model_dir))

    # -- 校验与下载 -------------------------------------------------------

    def _valid(self, model_id: str, filename: str, path: Path, entry: dict) -> bool:
        """校验文件：优先清单 sha256，其次缓存记录，都没有则信任（首次）。"""
        manifest_sha = entry.get("sha256")
        if manifest_sha:
            return self._cache.compute_sha256(path) == manifest_sha
        recorded = self._cache.sha256_of(model_id, filename)
        if recorded:
            return self._cache.compute_sha256(path) == recorded
        return True

    def _stream_download(self, url: str, dest: Path) -> None:
        """分段断点续传下载，直到 Content-Range 声明的总大小。

        带总时长预算：网络持续停滞超时即报错，避免无限挂起。
        """
        size = dest.stat().st_size if dest.exists() else 0
        total: int | None = None
        deadline = time.monotonic() + _DOWNLOAD_TIMEOUT_SEC
        for _ in range(300):  # 足够多的续传段
            if time.monotonic() > deadline:
                raise DownloadError(f"下载超时（>{_DOWNLOAD_TIMEOUT_SEC}s）: {url}")
            if total is not None and size >= total:
                break
            try:
                headers = {"Range": f"bytes={size}-"} if size else {}
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    content_range = resp.headers.get("Content-Range")
                    if total is None and content_range:
                        total = int(content_range.rsplit("/", 1)[1])
                    elif total is None:
                        total = int(resp.headers.get("Content-Length") or 0)
                    if size and resp.status != 206:
                        raise DownloadError(f"服务器不支持断点续传: {url}")
                    segment = 0
                    with dest.open("ab" if size else "wb") as fh:
                        while segment < _SEGMENT_CAP:
                            chunk = resp.read(_CHUNK)
                            if not chunk:
                                break
                            fh.write(chunk)
                            size += len(chunk)
                            segment += len(chunk)
            except DownloadError:
                raise
            except Exception:
                # 网络抖动：短暂等待后从当前位置继续
                time.sleep(1)
                continue
        if total is not None and size < total:
            raise DownloadError(f"下载未完成: {size}/{total} 字节 ({url})")
        if total is None:
            raise DownloadError(f"无法获取文件大小: {url}")

    def _load_manifest(self) -> dict:
        if not self._manifest_path.exists():
            return {"models": {}}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"models": {}}
        except json.JSONDecodeError:
            return {"models": {}}
