"""Python SDK 视频识别示例。

用法：
    python examples/video_example.py <服务地址> <视频路径>

示例：
    python examples/video_example.py http://127.0.0.1:8000 ./demo.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sdk import VisionClient  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    base_url, video = sys.argv[1], sys.argv[2]

    with VisionClient(base_url) as client:
        result = client.recognize_video(
            video,
            fps=1.0,
            max_frames=60,
            tasks=["objects", "ocr"],
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
