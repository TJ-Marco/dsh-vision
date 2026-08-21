# dsh-vision

> 供 DeepSeek（及任意 LLM agent）通过 function calling 调用的开源视觉识别服务：
> **图像识别**（目标检测 / 图像描述 / 场景分类 / OCR / 人脸）+ **视频识别**（抽帧 + 时间戳）。
> FastAPI HTTP API + Python SDK，独立运行，**不依赖任何 DeepSeek 内部服务**。

## 特性

- 🖼️ **图像识别**：目标检测（YOLOv8）、图像描述（BLIP）、场景分类（CLIP 零样本）、
  文字识别（PaddleOCR / EasyOCR 可切换）、人脸检测与特征摘要（InsightFace）
- 🎬 **视频识别**：ffmpeg/OpenCV 抽帧（均匀采样 / 场景切换）、并行推理、
  逐帧时间戳 + 事件时间线聚合
- 🔌 **双输入**：文件上传与图片/视频 URL 均支持
- 📦 **模型不入库**：首次运行自动下载到缓存目录（可预下载、SHA256 校验、断点续传）
- 🤖 **原生适配 function calling**：OpenAPI 自动生成，注册工具参数即用
- 🐳 **Docker 一键部署**（CPU / GPU 双版本）
- 🧩 **模型可替换**：统一抽象接口，换模型只换一个工厂函数

## 快速开始

### 1. 安装

```bash
git clone <your-fork-url> && cd dsh-vision
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt                  # CPU 部署；GPU 用 requirements-gpu.txt
```

### 2. 启动服务

```bash
# 可选：预下载模型权重（否则首次调用自动下载）
python scripts/download_models.py

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. 图像识别

```bash
# 文件上传
curl -X POST http://127.0.0.1:8000/api/image \
     -F "file=@photo.jpg" -F "tasks=objects,caption,scene,ocr,faces"

# 或图片 URL
curl -X POST http://127.0.0.1:8000/api/image \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/photo.jpg", "tasks": ["objects", "ocr"]}'
```

### 4. 视频识别

```bash
curl -X POST http://127.0.0.1:8000/api/video \
     -F "file=@clip.mp4" -F "fps=1.0" -F "tasks=objects,ocr" -F "include_keyframes=false"
```

### 5. Python SDK

```python
from app.sdk import VisionClient

with VisionClient("http://127.0.0.1:8000") as client:
    # 图像：本地路径 / 字节 / URL 均可
    result = client.recognize_image("photo.jpg", tasks=["objects", "scene"])
    result = client.recognize_image_url("https://example.com/a.jpg", min_confidence=0.5)

    # 视频
    result = client.recognize_video("clip.mp4", fps=1.0, max_frames=60, tasks=["objects", "ocr"])
```

完整示例见 [`examples/image_example.py`](examples/image_example.py) 与 [`examples/video_example.py`](examples/video_example.py)。

## 与 DeepSeek function calling 集成

1. 启动服务后，`GET http://<host>:8000/openapi.json` 即为完整 OpenAPI 规范；
2. 在 DeepSeek 侧注册工具（参数与 `/api/image` 一一对应）：

```jsonc
{
  "type": "function",
  "function": {
    "name": "vision_recognize_image",
    "description": "识别图片内容：目标、场景、文字、人脸、图像描述。输入图片 URL。",
    "parameters": {
      "type": "object",
      "properties": {
        "image_url": { "type": "string", "description": "图片的公开 HTTP(S) URL" },
        "tasks": {
          "type": "array",
          "items": { "type": "string", "enum": ["objects", "caption", "scene", "ocr", "faces"] }
        },
        "min_confidence": { "type": "number", "minimum": 0, "maximum": 1 }
      },
      "required": ["image_url"]
    }
  }
}
```

调用链：`DeepSeek → function calling(HTTP POST /api/image) → JSON 回填`，服务端零耦合、可独立部署。

## API 文档

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活检查（版本 / 设备 / 已加载模型） |
| GET | `/models` | 模型清单与许可 |
| POST | `/api/image` | 图像识别 |
| POST | `/api/video` | 视频识别 |
| GET | `/openapi.json` | OpenAPI 规范（function calling 注册用） |

### 图像识别响应示例

```jsonc
{
  "request_id": "img_8f3a...",
  "width": 1920, "height": 1080,
  "objects": [{ "label": "person", "confidence": 0.93, "bbox": [120, 80, 640, 1080] }],
  "caption": "a group of people walking on a city street",
  "scene": { "label": "city street", "confidence": 0.87 },
  "text": [{ "text": "COFFEE", "confidence": 0.96, "bbox": [10, 20, 300, 80], "lang": "en" }],
  "faces": [{ "bbox": [100, 100, 200, 280], "confidence": 0.99, "embedding_md5": "ab12..." }],
  "models_used": { "objects": "object_detection", "caption": "captioning" },
  "processing_time_ms": 842
}
```

> 人脸仅返回特征摘要（`embedding_md5`），不返回原始 512 维向量，避免高敏生物特征进入对话上下文。

### 视频识别响应示例

```jsonc
{
  "request_id": "vid_2c91...",
  "duration_sec": 182.4, "sampled_frames": 96,
  "results": [
    { "frame_index": 0, "timestamp_sec": 0.0, "objects": [...], "text": [...], "faces": [...] }
  ],
  "timeline": [   // 聚合事件：同一目标连续出现的时间段
    { "event": "car", "start_sec": 0.0, "end_sec": 12.5, "max_confidence": 0.95 }
  ],
  "processing_time_ms": 15023
}
```

### 错误约定

| HTTP | 场景 |
|---|---|
| 400 | 图片/视频无法解码、URL 下载失败、任务名/策略未知 |
| 404 | 请求的模型未注册 |
| 422 | 参数校验失败 |
| 503 | 模型未启用、权重缺失或正在下载 |

错误响应体：`{"error": "<机器可读错误码>", "message": "<说明>", "request_id": "..."}`。

## Docker 部署

**CPU 版**：

```bash
docker build -f docker/Dockerfile -t dsh-vision:0.1.0 .
docker run -p 8000:8000 -v dsh-vision-models:/models dsh-vision:0.1.0
# 或
docker compose -f docker/compose.yaml up -d
```

**GPU 版**（宿主机需 NVIDIA Container Toolkit）：

```bash
docker build -f docker/Dockerfile.gpu -t dsh-vision:gpu .
docker run --gpus all -p 8000:8000 -v dsh-vision-models:/models dsh-vision:gpu
```

模型缓存挂载在 `/models` 卷，首次请求自动下载，重启不重复下载。

## 模型许可列表

> 本项目代码采用 **MIT License**；依赖模型的许可以下逐项列出，请在使用/再分发前自行核验最新条款。

| 功能 | 模型 | 框架 | 许可 ⚠️ |
|---|---|---|---|
| 目标检测 | YOLOv8n（COCO-80） | ultralytics / onnxruntime | **AGPL-3.0** |
| 图像描述 | BLIP-base（Salesforce） | transformers | BSD-3-Clause（权重需核验） |
| 场景分类 | CLIP ViT-B/32（OpenAI） | transformers | MIT |
| OCR | PaddleOCR PP-OCRv4 | paddleocr | Apache-2.0 |
| OCR（备选） | EasyOCR | easyocr | Apache-2.0 |
| 人脸 | InsightFace buffalo_l | insightface | MIT（模型包需核验） |

**关于 AGPL-3.0**：YOLOv8/ultralytics 为 AGPL-3.0 依赖。作为依赖使用不影响本项目 MIT 分发；
若需要完全规避 AGPL，可将检测器替换为 torchvision Faster R-CNN（BSD-3-Clause）等——本项目通过
`BaseVisionModel` 抽象支持无痛替换（实现一个工厂函数即可）。

## 配置

`config.yaml` 支持全部运行配置（端口、模型缓存目录、设备、置信度阈值、OCR 引擎/语言、抽帧参数等），
关键项可用环境变量覆盖：

| 环境变量 | 说明 |
|---|---|
| `DSH_VISION_CACHE` | 模型缓存目录 |
| `DSH_VISION_DEVICE` | `auto` / `cuda` / `cpu` |
| `DSH_VISION_AUTO_DOWNLOAD` | `true` / `false`（首次调用是否自动下载权重） |
| `DSH_VISION_CONFIG` | 配置文件路径 |

## 开发与测试

```bash
pip install -r requirements-dev.txt
ruff check app scripts tests examples     # PEP8 自查
python -m pytest tests                    # 单元测试（假模型，无需权重）
```

## 如何贡献

提交信息规范、版本标签约定与发布检查清单见 [CONTRIBUTING.md](CONTRIBUTING.md)。基本流程：

1. Fork 本仓库并创建特性分支；
2. 遵守 PEP8（ruff 检查通过）、补充测试、保持中文注释与英文代码注释的既有风格；
3. 提交前运行 `ruff check` 与 `python -m pytest tests`；
4. 发起 Pull Request，说明改动与验证方式。

## License

代码：MIT License（见 [LICENSE](LICENSE)）。
依赖模型的许可见上方「模型许可列表」。
