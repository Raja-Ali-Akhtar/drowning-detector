# ============================================================
# Drowning Detection System — Production Docker Image
# Multi-stage build for minimal image size
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# System deps for building wheels (opencv, torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/base.txt
COPY requirements/prod.txt requirements/prod.txt

RUN pip install --no-cache-dir --prefix=/install -r requirements/prod.txt


# ── Stage 2: Runtime ────────────────────────────────────────
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.10 + runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy application code
COPY drowning_detector/ drowning_detector/
COPY pyproject.toml .

# Install the package itself
RUN pip install --no-cache-dir --no-deps -e .

# Create required directories
RUN mkdir -p logs && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["gunicorn", "drowning_detector.api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-"]
