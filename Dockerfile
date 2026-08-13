ARG PYTHON_IMAGE=python:3.11-alpine@sha256:25976e9d34a0fab1f278cae931f34c8303d97bf0c0d7f85b6b4dcf641d7702a4

FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY site ./site
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MEMORYFLOW_SITE_DIR=/app/site \
    PORT=8000

LABEL org.opencontainers.image.title="MemoryFlow Lab" \
      org.opencontainers.image.description="Reproducible LLM KV-cache memory co-design simulator" \
      org.opencontainers.image.source="https://github.com/KIM3310/memoryflow-lab" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app
RUN addgroup --gid 10001 --system memoryflow \
    && adduser --uid 10001 --ingroup memoryflow --system --disabled-password \
        --no-create-home --shell /sbin/nologin memoryflow
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl \
    && python -m pip uninstall --yes pip setuptools wheel \
    && rm -rf /wheels /root/.cache
COPY --chown=10001:10001 site ./site

USER 10001:10001

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

CMD ["python", "-m", "uvicorn", "memoryflow.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
