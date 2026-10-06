# ============================================================
# AudioTag AI -- Production Container (Render / Cloud Run)
# Multi-Label Polyphonic Music Tagging Engine
# ============================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUDIOTAG_ENGINE=onnx \
    AUDIOTAG_ALLOWED_ORIGINS=*

WORKDIR /app

# Install native audio decoders (ffmpeg and libsndfile for MP3/OGG/FLAC/M4A)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install optimized production dependencies (ONNX Runtime, sub-30s build)
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (respecting .dockerignore)
COPY . .

# Render injects $PORT (default 10000) automatically into every Docker container.
# We read it at container start time and bind uvicorn to whatever Render assigns.
EXPOSE 10000

# Single-process startup: bind to Render-injected $PORT (default 10000).
# Render auto-detects whichever port the process binds to and routes traffic there.
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
