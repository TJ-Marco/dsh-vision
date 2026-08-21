"""Python SDK 图像识别示例。

用法：
    python examples/image_example.py <服务地址> <图片路径或URL>

示例：
    python examples/image_example.py http://127.0.0.1:8000 ./demo.jpg
    python examples/image_example.py http://127.0.0.1:8000 https://example.com/a.jpg
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
    base_url, image = sys.argv[1], sys.argv[2]

    with VisionClient(base_url) as client:
        result = client.recognize_image(image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
