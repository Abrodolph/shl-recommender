# SHL Assessment Recommender — production image.
#
# Ships the catalog + precomputed embeddings so nothing large downloads at boot
# (CLAUDE.md §4). Binds to $PORT (Render/Fly/Railway/HF Spaces all set it).
# /health answers immediately; the embedding model + indexes warm in a background
# thread on startup (see app/main.py lifespan).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    PORT=8000

WORKDIR /app

# CPU-only torch first (avoids the multi-GB CUDA build that sentence-transformers
# would otherwise pull), then the rest of the pinned requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# App code + shipped data artifacts (catalog.json, embeddings.npy, ids).
COPY app/ ./app/
COPY data/ ./data/

# Pre-download the embedding model into the image so the first query isn't
# blocked on a network fetch at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

# Shell form so $PORT is expanded at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
