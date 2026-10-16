FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first to maximise layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.12-slim AS prod

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY static ./static
COPY templates ./templates

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --system --no-create-home appuser
USER appuser

EXPOSE 8000

# Run uvicorn directly: main.py enables hot-reload, which is not wanted in production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
