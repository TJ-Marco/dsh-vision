"""multipart 表单解析辅助（图像/视频路由共用）。"""

from __future__ import annotations

import json

from fastapi import UploadFile

from app.core.errors import InvalidInputError


async def form_text(form, key: str) -> str | None:
    """读取表单文本字段；兼容客户端把字段当文件上传的情况。"""
    value = form.get(key)
    if value is None:
        return None
    if isinstance(value, UploadFile):
        data = await value.read()
        return data.decode("utf-8", errors="replace") or None
    return str(value)


def parse_tasks(value: str | None) -> list[str] | None:
    """解析表单 tasks 字段：支持逗号分隔（"objects,ocr"）或 JSON 数组。"""
    if value is None or value == "":
        return None
    stripped = value.strip()
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise InvalidInputError("tasks 字段必须是列表")
        return [str(task) for task in parsed]
    return [task.strip() for task in stripped.split(",") if task.strip()]


def parse_float(value: str | None) -> float | None:
    """解析表单浮点字段；空值返回 None，非法值报错。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise InvalidInputError(f"数值字段非法: {value}") from exc


def parse_int(value: str | None) -> int | None:
    """解析表单整数字段；空值返回 None，非法值报错。"""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise InvalidInputError(f"整数字段非法: {value}") from exc


def parse_bool(value: str | None) -> bool:
    """解析表单布尔字段；空值视为 False。"""
    if value is None or value == "":
        return False
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise InvalidInputError(f"布尔字段非法: {value}；可选 true|false")
