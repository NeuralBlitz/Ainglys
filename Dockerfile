FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package sources before installation; editable installs can mask broken images.
COPY pyproject.toml ./
COPY src/ ./src/
COPY projects/acca/src/ ./projects/acca/src/
# Docker is Linux-only: install the pinned CPU build, not multi-GB CUDA bundles.
# Keep this pinned version synchronized with uv.lock.
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir "torch==2.10.0+cpu" \
         --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir .

FROM base AS production

COPY . .

# Run services as an unprivileged user; model artifacts have an explicit write root.
RUN useradd --system --uid 10001 --create-home appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1

# Exactly one foreground process; the optional Streamlit UI is a separate Compose service.
CMD ["uvicorn", "ai_hub.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS development

RUN python -m pip install --no-cache-dir '.[dev]' jupyter jupyterlab
COPY . .

EXPOSE 8000 8501 8888

CMD ["/bin/bash"]
