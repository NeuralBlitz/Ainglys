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

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e "."

FROM base as production

COPY src/ ./src/
COPY . .

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["sh", "-c", "ai-hub api start --host 0.0.0.0 --port 8000 & streamlit run src/ai_hub/web.py --server.port 8501 --server.address 0.0.0.0"]

FROM base as development

RUN pip install -e ".[dev]" jupyter jupyterlab

EXPOSE 8000 8501 8888

CMD ["/bin/bash"]
