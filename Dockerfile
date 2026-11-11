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
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir .

FROM base AS production

COPY . .

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["sh", "-c", "ai-hub api start --host 0.0.0.0 --port 8000 & streamlit run src/ai_hub/web.py --server.port 8501 --server.address 0.0.0.0"]

FROM base AS development

RUN python -m pip install --no-cache-dir '.[dev]' jupyter jupyterlab
COPY . .

EXPOSE 8000 8501 8888

CMD ["/bin/bash"]
