"""SDK 客户端异常。"""

from __future__ import annotations


class VisionClientError(Exception):
    """服务返回错误状态时的客户端异常。"""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"dsh-vision 服务返回 {status_code}: {body}")
        self.status_code = status_code
        self.body = body
