import os
import io
import logging
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, Query, HTTPException

logger = logging.getLogger("audiotag.analyze")

# 50 MB hard cap — prevents OOM attacks via enormous audio uploads
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Only accept genuine audio MIME types
ALLOWED_MIME_PREFIXES = ("audio/", "video/")

try:
    from backend.app.core.inference import predict_instruments, DEFAULT_THRESHOLD
except ImportError:
    from app.core.inference import predict_instruments, DEFAULT_THRESHOLD

router = APIRouter()


@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0.0, le=1.0)
):
    """
    Analyze uploaded audio (.wav, .mp3, .ogg, .flac) and predict the presence
    of up to 18 instruments simultaneously using AudioResNet-SE (PyTorch GPU).

    Limits:
      - Max file size: 50 MB
      - Accepted MIME types: audio/* or video/*
    """
    # --- 1. MIME Type Guard ---
    content_type = file.content_type or ""
    if not any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: '{content_type}'. Expected audio/*, received something else."
        )

    # --- 2. File Size Guard (read-once into memory, then write to temp) ---
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // (1024*1024)} MB). Maximum supported upload is 50 MB."
        )

    # --- 3. Extension Determination ---
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}:
        ext = ".wav"

    # --- 4. Write to Secure Temp File ---
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = temp_file.name
    temp_file.close()

    try:
        with open(temp_path, "wb") as buf:
            buf.write(content)

        logger.info("Running inference on '%s' (%d bytes, threshold=%.2f)",
                    file.filename, len(content), threshold)

        results = predict_instruments(temp_path, threshold=threshold)
        results["filename"] = file.filename

        return results

    except RuntimeError as e:
        logger.error("Model runtime error for '%s': %s", file.filename, e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected inference error for '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.warning("Could not remove temp file %s: %s", temp_path, e)