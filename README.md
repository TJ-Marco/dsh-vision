# dsh-vision

> An open-source vision recognition service designed to be called by DeepSeek (or any LLM agent) via function calling:
> **image recognition** (object detection / image captioning / scene classification / OCR / face) + **video recognition** (frame sampling + timestamps).
> FastAPI HTTP API + Python SDK, fully standalone — **no dependency on any DeepSeek internal service**.

English | [中文](README.zh.md)

## Features

- 🖼️ **Image recognition**: object detection (YOLOv8), image captioning (BLIP), zero-shot scene classification (CLIP), OCR (PaddleOCR / EasyOCR switchable), face detection with feature digest (InsightFace)
- 🎬 **Video recognition**: frame extraction via ffmpeg/OpenCV (uniform sampling / scene-change detection), parallel inference, per-frame timestamps + event timeline aggregation
- 🔌 **Dual input**: file upload and image/video URLs both supported
- 📦 **Models are not bundled**: downloaded automatically to a cache directory on first use (pre-download supported, SHA-256 verified, resumable)
- 🤖 **Function-calling ready**: OpenAPI generated automatically — register tool parameters and use
- 🐳 **One-command Docker deployment** (CPU / GPU variants)
- 🧩 **Swappable models**: unified abstraction interface — swap a model by replacing one factory function

## Quick Start

### 1. Install

```bash
git clone <your-fork-url> && cd dsh-vision
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt                  # CPU; use requirements-gpu.txt for GPU
```

### 2. Start the server

```bash
# Optional: pre-download model weights (otherwise they auto-download on first use)
python scripts/download_models.py

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Image recognition

```bash
# File upload
curl -X POST http://127.0.0.1:8000/api/image \
     -F "file=@photo.jpg" -F "tasks=objects,caption,scene,ocr,faces"

# Or image URL
curl -X POST http://127.0.0.1:8000/api/image \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/photo.jpg", "tasks": ["objects", "ocr"]}'
```

### 4. Video recognition

```bash
curl -X POST http://127.0.0.1:8000/api/video \
     -F "file=@clip.mp4" -F "fps=1.0" -F "tasks=objects,ocr" -F "include_keyframes=false"
```

### 5. Python SDK

```python
from app.sdk import VisionClient

with VisionClient("http://127.0.0.1:8000") as client:
    # Image: local path / bytes / URL all accepted
    result = client.recognize_image("photo.jpg", tasks=["objects", "scene"])
    result = client.recognize_image_url("https://example.com/a.jpg", min_confidence=0.5)

    # Video
    result = client.recognize_video("clip.mp4", fps=1.0, max_frames=60, tasks=["objects", "ocr"])
```

Full examples: [`examples/image_example.py`](examples/image_example.py) and [`examples/video_example.py`](examples/video_example.py).

## Integration with DeepSeek function calling

1. Start the server, then `GET http://<host>:8000/openapi.json` is the full OpenAPI spec;
2. Register a tool on the DeepSeek side (parameters mirror `POST /api/image`):

```jsonc
{
  "type": "function",
  "function": {
    "name": "vision_recognize_image",
    "description": "Recognize image content: objects, scene, text, faces, caption. Takes an image URL.",
    "parameters": {
      "type": "object",
      "properties": {
        "image_url": { "type": "string", "description": "A public HTTP(S) URL of the image" },
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

Call chain: `DeepSeek → function calling(HTTP POST /api/image) → JSON returned` — zero coupling, fully standalone deployment.

## API Documentation

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness (version / device / loaded models) |
| GET | `/models` | Model inventory and licenses |
| POST | `/api/image` | Image recognition |
| POST | `/api/video` | Video recognition |
| GET | `/openapi.json` | OpenAPI spec (for function-calling registration) |

### Image response example

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

> Faces return only a feature digest (`embedding_md5`), never the raw 512-d embedding, so sensitive biometric data never enters the conversation context.

### Video response example

```jsonc
{
  "request_id": "vid_2c91...",
  "duration_sec": 182.4, "sampled_frames": 96,
  "results": [
    { "frame_index": 0, "timestamp_sec": 0.0, "objects": [...], "text": [...], "faces": [...] }
  ],
  "timeline": [   // aggregated events: continuous time ranges per target
    { "event": "car", "start_sec": 0.0, "end_sec": 12.5, "max_confidence": 0.95 }
  ],
  "processing_time_ms": 15023
}
```

### Error conventions

| HTTP | Scenario |
|---|---|
| 400 | Undecodable image/video, URL download failure, unknown task/strategy |
| 404 | Requested model not registered |
| 422 | Parameter validation failure |
| 503 | Model disabled, weights missing, or still downloading |

Error body: `{"error": "<machine-readable code>", "message": "<description>", "request_id": "..."}`.

## Docker Deployment

**CPU**:

```bash
docker build -f docker/Dockerfile -t dsh-vision:0.1.0 .
docker run -p 8000:8000 -v dsh-vision-models:/models dsh-vision:0.1.0
# or
docker compose -f docker/compose.yaml up -d
```

**GPU** (host needs NVIDIA Container Toolkit):

```bash
docker build -f docker/Dockerfile.gpu -t dsh-vision:gpu .
docker run --gpus all -p 8000:8000 -v dsh-vision-models:/models dsh-vision:gpu
```

Model cache is mounted on the `/models` volume — downloaded on first request, not re-downloaded on restart.

## Model License List

> Project code is **MIT licensed**; the licenses of dependency models are listed below — verify the latest terms before use/re-distribution.

| Feature | Model | Framework | License ⚠️ |
|---|---|---|---|
| Object detection | YOLOv8n (COCO-80) | ultralytics / onnxruntime | **AGPL-3.0** |
| Image captioning | BLIP-base (Salesforce) | transformers | BSD-3-Clause (verify weights) |
| Scene classification | CLIP ViT-B/32 (OpenAI) | transformers | MIT |
| OCR | PaddleOCR PP-OCRv4 | paddleocr | Apache-2.0 |
| OCR (alternative) | EasyOCR | easyocr | Apache-2.0 |
| Face | InsightFace buffalo_l | insightface | MIT (verify model pack) |

**About AGPL-3.0**: YOLOv8/ultralytics is an AGPL-3.0 dependency. Using it as a dependency does not affect the MIT distribution of this project; if you need to avoid AGPL entirely, swap the detector for e.g. torchvision Faster R-CNN (BSD-3-Clause) — the `BaseVisionModel` abstraction makes this a drop-in replacement (implement one factory function).

## Configuration

`config.yaml` covers all runtime settings (port, model cache dir, device, confidence thresholds, OCR engine/languages, sampling parameters, etc.); key items can be overridden with environment variables:

| Env var | Description |
|---|---|
| `DSH_VISION_CACHE` | Model cache directory |
| `DSH_VISION_DEVICE` | `auto` / `cuda` / `cpu` |
| `DSH_VISION_AUTO_DOWNLOAD` | `true` / `false` (auto-download weights on first use) |
| `DSH_VISION_CONFIG` | Config file path |

## Development & Testing

```bash
pip install -r requirements-dev.txt
ruff check app scripts tests examples     # PEP8 self-check
python -m pytest tests                    # unit tests (fake models, no weights needed)
```

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit conventions, versioning, and the release checklist. In short:

1. Fork this repository and create a feature branch;
2. Follow PEP8 (ruff clean), add tests, keep existing comment style (Chinese prose / English code comments);
3. Run `ruff check` and `python -m pytest tests` before committing;
4. Open a Pull Request describing the change and how it was verified.

## License

Code: MIT License (see [LICENSE](LICENSE)).
Dependency models: see the Model License List above.
