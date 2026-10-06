# ============================================================
# AudioTag AI — Production Container (Render / Cloud Run)
# Multi-Label Polyphonic Music Tagging Engine
# ============================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUDIOTAG_ENGINE=onnx \
    AUDIOTAG_ALLOWED_ORIGINS=*

WORKDIR /app

# Install native audio decoders and networking utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    socat \
    && rm -rf /var/lib/apt/lists/*

# Install optimized production dependencies (ONNX Runtime, sub-30s build)
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (respecting .dockerignore)
COPY . .

# Expose both Render default ports (8000 and 10000)
EXPOSE 8000 10000

# Dual-port listener: serves on $PORT (default 8000) while mirroring port 10000 via socat
CMD ["sh", "-c", "PORT=${PORT:-8000}; if [ \"$PORT\" = \"8000\" ]; then socat TCP-LISTEN:10000,fork,reuseaddr TCP:127.0.0.1:8000 & else socat TCP-LISTEN:8000,fork,reuseaddr TCP:127.0.0.1:$PORT & fi; exec uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1"]
