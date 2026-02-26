# ============================================================
# AumOS Video Engine — Multi-stage Dockerfile
# ============================================================
# Stage 1: builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install hatch for building
RUN pip install --no-cache-dir hatch

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# ============================================================
# Stage 2: runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system deps (OpenCV, FFmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r aumos && useradd -r -g aumos aumos

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY migrations/ ./migrations/ 2>/dev/null || true

RUN chown -R aumos:aumos /app
USER aumos

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "aumos_video_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================
# Stage 3: GPU runtime (optional override)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS gpu-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libpq5 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r aumos && useradd -r -g aumos aumos

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/

RUN chown -R aumos:aumos /app
USER aumos

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GPU_ENABLED=true

EXPOSE 8000

CMD ["uvicorn", "aumos_video_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
