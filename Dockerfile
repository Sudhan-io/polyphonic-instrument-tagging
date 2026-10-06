# ============================================================
# AudioTag AI — Production Container
# Multi-Label Polyphonic Music Tagging Engine
# ============================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system audio libraries (ffmpeg & libsndfile for decoding MP3/OGG/FLAC/M4A)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with CPU wheels to prevent 2.5 GB CUDA bloat on cloud containers
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application files
COPY . .

# Expose the service port
EXPOSE 8000

# Health check with extended grace period for Render cold starts
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Launch FastAPI via uvicorn with single worker to respect 512 MB memory limit
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
