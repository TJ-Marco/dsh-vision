"""目标检测：YOLOv8（ultralytics 主路径 / ONNX 可选路径）。

权重加载约定（模型不入库）：
- 缓存目录存在 ``yolov8n.pt`` → ultralytics 路径（默认）；
- 存在 ``yolov8n.onnx`` → onnxruntime 路径（无 torch 部署用）。
权重文件由模块 5（下载器/导出脚本）负责准备；缺失时给出明确指引。

ONNX 后处理（``_postprocess_onnx``）为纯函数，便于用合成张量单测。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.errors import ModelUnavailableError
from app.vision.base import BaseVisionModel, resolve_device

#: 默认输入边长（YOLOv8 标准）
_INPUT_SIDE = 640
#: ONNX 输出通道布局：cx, cy, w, h + 类别得分
_BOX_COORDS = 4


class YoloDetector(BaseVisionModel):
    """YOLOv8 目标检测器。"""

    name = "object_detection"
    license = "AGPL-3.0"

    def __init__(self, cache, settings, downloader=None) -> None:
        self._model_dir: Path = cache.model_dir("object_detection")
        self._device = resolve_device(settings)
        self._auto_download = settings.resolve_auto_download()
        self._downloader = downloader
        self._model = None  # 懒加载：ultralytics YOLO 或 onnxruntime 会话
        self._names: list[str] = []

    # -- 生命周期 ---------------------------------------------------------

    def load(self) -> None:
        if self._auto_download and self._downloader is not None:
            try:
                self._downloader.ensure("object_detection")
            except Exception as exc:  # noqa: BLE001 - 下载失败转为可用提示
                raise ModelUnavailableError(f"模型下载失败: {exc}") from exc
        pt_path = self._model_dir / "yolov8n.pt"
        onnx_path = self._model_dir / "yolov8n.onnx"
        if pt_path.exists():
            self._load_ultralytics(pt_path)
        elif onnx_path.exists():
            self._load_onnx(onnx_path)
        else:
            raise ModelUnavailableError(
                "目标检测权重缺失（yolov8n.pt / yolov8n.onnx）；"
                "请先运行 python scripts/download_models.py --models object_detection"
                "（或开启 config 的 models.auto_download）",
            )

    # -- 推理 -------------------------------------------------------------

    def predict(self, image_bgr: np.ndarray, options: dict) -> dict:
        min_conf = options.get("min_confidence") or 0.35
        if self._model is None:
            self.load()
        if self._use_onnx:
            boxes, scores, class_ids = self._predict_onnx(image_bgr, min_conf)
        else:
            boxes, scores, class_ids = self._predict_ultralytics(image_bgr, min_conf)
        items = [
            {
                "label": self._names[int(class_id)],
                "confidence": round(float(score), 4),
                "bbox": [round(float(v), 1) for v in box],
            }
            for box, score, class_id in zip(boxes, scores, class_ids, strict=False)
        ]
        return {"items": items}

    # -- ultralytics 路径 -------------------------------------------------

    def _load_ultralytics(self, weight: Path) -> None:
        try:
            from ultralytics import YOLO  # 懒加载：避免应用启动依赖 torch
        except ImportError as exc:  # pragma: no cover
            raise ModelUnavailableError(
                "目标检测需要 ultralytics+torch：pip install ultralytics",
            ) from exc

        self._model = YOLO(str(weight))
        # 强制 fp32：部分发布权重为 fp16，本机 CPU 上 fp16 推理会输出垃圾结果
        self._model.model = self._model.model.float()
        self._names = list(self._model.names.values())
        self._use_onnx = False

    def _predict_ultralytics(self, image_bgr: np.ndarray, min_conf: float):
        """手动推理管线（绕开 ultralytics predict 的输入处理）。

        实测在部分 torch CPU 环境下，``model.predict`` 的预处理会篡改输入张量
        （全零或数值放大），导致全部类别置信度饱和为 1.0 的垃圾输出；改为
        LetterBox → 归一化 → 直接前向 → NMS，并把结果映射回原图坐标。
        """
        import torch
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.ops import non_max_suppression

        height, width = image_bgr.shape[:2]
        letterboxed = LetterBox(new_shape=640, stride=32, auto=True)(image=image_bgr)
        im = letterboxed[..., ::-1].transpose((2, 0, 1))[None].astype(np.float32)
        im = np.ascontiguousarray(im) / 255.0

        with torch.no_grad():
            out = self._model.model(torch.from_numpy(im))
        raw = out[0] if isinstance(out, list | tuple) else out
        pred = non_max_suppression(raw, conf_thres=min_conf, iou_thres=0.45)[0]
        if pred is None or len(pred) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        # letterbox 坐标 → 原图坐标（与 ultralytics LetterBox 的缩放/居中填充对应）
        scale = min(640 / height, 640 / width)
        new_w, new_h = round(width * scale), round(height * scale)
        pad_x = (640 - new_w) % 32 / 2
        pad_y = (640 - new_h) % 32 / 2
        boxes = pred[:, :4].numpy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        # 裁剪到原图范围
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        return boxes, pred[:, 4].numpy(), pred[:, 5].numpy().astype(int)

    # -- ONNX 路径 --------------------------------------------------------

    def _load_onnx(self, weight: Path) -> None:
        try:
            import onnxruntime as ort  # 懒加载
        except ImportError as exc:  # pragma: no cover
            raise ModelUnavailableError(
                "ONNX 路径需要 onnxruntime：pip install onnxruntime",
            ) from exc

        self._session = ort.InferenceSession(str(weight), providers=["CPUExecutionProvider"])
        self._names = self._default_names()
        self._use_onnx = True
        self._model = self._session

    def _predict_onnx(self, image_bgr: np.ndarray, min_conf: float):
        input_tensor, scale, pad = _letterbox(image_bgr, _INPUT_SIDE)
        outputs = self._session.run(None, {self._session.get_inputs()[0].name: input_tensor})
        return _postprocess_onnx(outputs[0], image_bgr.shape[:2], min_conf, scale, pad)

    @staticmethod
    def _default_names() -> list[str]:
        """COCO-80 类别名（与 YOLOv8 预训练权重一致；.pt 路径从模型读取）。"""
        return [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]


def _letterbox(image: np.ndarray, side: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比缩放并填充到 side×side，返回 (归一化张量, 缩放比, 填充像素)。"""
    height, width = image.shape[:2]
    scale = min(side / height, side / width)
    new_h, new_w = round(height * scale), round(width * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = side - new_w, side - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # BGR → RGB、HWC → CHW、归一化、加 batch 维
    rgb = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb), scale, (left, top)


def _postprocess_onnx(
    output: np.ndarray,
    orig_shape: tuple[int, int],
    min_conf: float,
    scale: float,
    pad: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YOLOv8 ONNX 输出后处理（纯函数，可单测）。

    :param output: 形状 (1, 4+num_classes, num_anchors)
    :return: (boxes[N,4], scores[N], class_ids[N])，坐标为原图像素 [x1,y1,x2,y2]
    """
    pred = output[0]  # (4+C, A)
    pred = pred.transpose(1, 0)  # (A, 4+C)
    scores_all = pred[:, _BOX_COORDS:]
    class_ids = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(pred)), class_ids]

    keep = scores >= min_conf
    pred, scores, class_ids = pred[keep], scores[keep], class_ids[keep]
    if len(pred) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # YOLOv8 输出为 cx,cy,w,h → 转 x1,y1,x2,y2 并映射回原图
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack(
        [
            (cx - w / 2 - pad[0]) / scale,
            (cy - h / 2 - pad[1]) / scale,
            (cx + w / 2 - pad[0]) / scale,
            (cy + h / 2 - pad[1]) / scale,
        ],
        axis=1,
    )

    # NMS（按置信度降序，IoU 阈值 0.45）
    order = np.argsort(-scores)
    keep_idx: list[int] = []
    while order.size > 0:
        first = order[0]
        keep_idx.append(int(first))
        rest = order[1:]
        if rest.size == 0:
            break
        ious = _iou(boxes[first], boxes[rest])
        order = rest[ious < 0.45]
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    class_ids = class_ids[keep_idx]

    # 裁剪到原图范围
    height, width = orig_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes, scores, class_ids


def _iou(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """单个框与一组框的 IoU。"""
    x1 = np.maximum(box_a[0], boxes_b[:, 0])
    y1 = np.maximum(box_a[1], boxes_b[:, 1])
    x2 = np.minimum(box_a[2], boxes_b[:, 2])
    y2 = np.minimum(box_a[3], boxes_b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def register(registry, settings, cache, downloader=None) -> None:
    """把检测器工厂注册进模型注册表（模块 4 接入点）。"""
    registry.register(
        "object_detection",
        lambda: YoloDetector(cache, settings, downloader),
        license=YoloDetector.license,
        description="YOLOv8 目标检测（COCO-80；AGPL-3.0）",
    )
