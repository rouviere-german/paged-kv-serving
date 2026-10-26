# syntax=docker/dockerfile:1
#
# Multi-stage, non-root image. Base images are pinned by digest (resolved 2026-09-13); Dependabot's
# docker ecosystem proposes digest bumps. python:3.12-slim is multi-arch, so the index digest works
# on amd64 and arm64.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
# A deterministic random-init checkpoint (seeded weights, fixed timestamp), built once at image
# build time so the container needs no download. The manifest digest it prints is the value
# docker-compose.yml pins through APP_CHECKPOINT_MANIFEST_SHA256. The atomic writer creates files
# 0600 (mkstemp), and the runtime user is not root: the files become world-readable but stay
# root-owned and read-only, so the service can load its checkpoint and cannot rewrite it.
RUN /opt/venv/bin/python -m paged_kv_serving build-checkpoint --out /opt/checkpoint --seed 0 \
    && chmod 0555 /opt/checkpoint \
    && chmod 0444 /opt/checkpoint/*

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="paged-kv-serving" \
      org.opencontainers.image.description="Paged KV cache, continuous batching and admission control on CPU" \
      org.opencontainers.image.licenses="MIT"
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/checkpoint /opt/checkpoint
# The root filesystem is read-only at runtime: the admin audit log lives under APP_DATA_DIR on a
# tmpfs (mount a volume there to keep it). OPENBLAS_NUM_THREADS matches APP_ENGINE_BLAS_THREADS,
# so BLAS never starts a thread pool it is about to be told not to use (docs/adr/0006).
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_CHECKPOINT_DIR=/opt/checkpoint \
    APP_WEIGHTS_ROOT=/opt/weights \
    APP_DATA_DIR=/tmp/data \
    APP_ENGINE_BLAS_THREADS=1
WORKDIR /app
USER 10001:10001
EXPOSE 8000
# Start-up verifies the checkpoint, profiles the latency model and warms the KV pool first.
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-m", "paged_kv_serving.core.healthcheck"]
ENTRYPOINT ["python", "-m", "paged_kv_serving"]
CMD ["serve"]
