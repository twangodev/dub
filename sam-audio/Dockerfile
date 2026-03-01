FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install project dependencies (torch, sam-audio-infer, fastapi, etc.)
COPY pyproject.toml uv.lock ./
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev --no-install-project

# Install sam-audio from Facebook's repo (model architecture code).
# This also pulls dacvae, imagebind, laion-clap, perception-models from git.
RUN uv pip install git+https://github.com/facebookresearch/sam-audio.git

COPY server.py .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Pass HF_TOKEN and GEMINI_API_KEY as env vars at runtime.
# Model weights download on first start; mount /root/.cache to persist them.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
