FROM ghcr.io/astral-sh/uv:0.9.30-python3.13-trixie-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

# Copy the lock files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install required dependencies.
# The `--no-install-project` flag is used to avoid installing the project itself,
# which is handled in a later step to avoid re-downloading dependencies
# when the source code changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY src src
COPY README.md .
COPY dataset.h5 .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Reuse the installed dependencies in a smaller base image
FROM python:3.13-slim-trixie

# Install g++ for torch compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY --from=builder /app/src src
COPY --from=builder /app/dataset.h5 .

# Make the venv available without uv run
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root after install
USER appuser

ENTRYPOINT ["python", "src/csi_vae/jobs/job.py"]