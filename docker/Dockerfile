# syntax=docker/dockerfile:1
# dsh-vision 多阶段构建（CPU 版）
# 构建：docker build -f docker/Dockerfile -t dsh-vision:0.1.0 .
# GPU 版：见 docker/Dockerfile.gpu

# ---- Stage 1: 安装依赖（独立层，便于缓存）----
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
# --prefix 安装到独立目录，运行时阶段整体拷贝，避免分层膨胀
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: 运行镜像 ----
FROM python:3.11-slim AS runtime

# 系统依赖：ffmpeg（视频抽帧必需）
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 模型缓存目录（挂载卷），关闭字节码缓存
ENV DSH_VISION_CACHE=/models \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 拷贝 Python 依赖
COPY --from=builder /install /usr/local

WORKDIR /app
COPY app ./app
COPY config.yaml models.json ./

# 非 root 运行（安全最佳实践）
RUN useradd --create-home --uid 10001 dsh \
    && mkdir -p /models \
    && chown -R dsh:dsh /models /app
USER dsh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
