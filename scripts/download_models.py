"""一键下载模型权重（models.json 清单驱动）。

用法：
    python scripts/download_models.py                     # 下载全部
    python scripts/download_models.py --models object_detection,faces
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从任意 cwd 都能导入 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402
from app.core.cache import create_cache  # noqa: E402
from app.core.downloader import DEFAULT_MANIFEST_PATH, ModelDownloader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 dsh-vision 模型权重")
    parser.add_argument(
        "--models",
        default="all",
        help="逗号分隔的模型名（object_detection,captioning,scene,ocr,faces）或 all",
    )
    parser.add_argument("--config", default=None, help="config.yaml 路径（可选）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    cache = create_cache(settings)
    downloader = ModelDownloader(cache, DEFAULT_MANIFEST_PATH)

    if args.models.strip().lower() == "all":
        model_ids = downloader.manifest_ids()
    else:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]

    if not model_ids:
        print("没有可下载的模型（请检查 models.json）")
        return 1

    failed: list[str] = []
    for model_id in model_ids:
        print(f"[download] {model_id} ...", flush=True)
        try:
            downloader.ensure(model_id)
            print(f"[download] {model_id} 就绪（缓存: {settings.resolve_cache_dir() / model_id}）", flush=True)
        except Exception as exc:  # noqa: BLE001 - 逐个报告，不中断整体
            print(f"[download] {model_id} 失败: {exc}", flush=True)
            failed.append(model_id)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
