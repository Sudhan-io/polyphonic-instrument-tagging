import os
import io
import logging
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, Query, HTTPException

logger = logging.getLogger("audiotag.analyze")

# 50 MB hard cap — prevents OOM attacks via enormous audio uploads
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Accept genuine audio MIME types and common container representations
ALLOWED_MIME_PREFIXES = ("audio/", "video/")
ALLOWED_EXACT_MIMES = {
    "application/ogg",
    "application/x-ogg",
    "application/flac",
    "application/x-flac",
    "application/octet-stream"  # Common browser fallback for audio files
}
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".aiff"}

from starlette.concurrency import run_in_threadpool

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
    Analyze uploaded audio (.wav, .mp3, .ogg, .flac, .m4a) and predict the presence
    of up to 18 instruments simultaneously using AudioResNet-SE.

    Limits:
      - Max file size: 50 MB
      - Supported formats: WAV, MP3, OGG, FLAC, M4A
    """
    filename = file.filename or "audio.wav"
    ext = os.path.splitext(filename)[1].lower()
    content_type = (file.content_type or "").lower()

    # --- 1. MIME & Extension Guard ---
    is_valid_mime = any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES) or (content_type in ALLOWED_EXACT_MIMES)
    is_valid_ext = ext in SUPPORTED_EXTENSIONS

    if not is_valid_mime and not is_valid_ext:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media format. Content-Type: '{content_type}', File: '{filename}'. Expected audio (.wav, .mp3, .ogg, .flac, .m4a)."
        )

    if ext not in SUPPORTED_EXTENSIONS:
        ext = ".wav"

    # --- 2. Stream Upload Directly to Temp File (Zero RAM Bloat) ---
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = temp_file.name
    total_bytes = 0

    try:
        CHUNK_SIZE = 64 * 1024
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                temp_file.close()
                raise HTTPException(
                    status_code=413,
                    detail="File too large. Maximum supported upload is 50 MB."
                )
            temp_file.write(chunk)
        temp_file.close()

        if total_bytes < 100:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty or corrupted (under 100 bytes)."
            )

        logger.info("Running non-blocking inference on '%s' (%d bytes, threshold=%.2f)",
                    filename, total_bytes, threshold)

        # Offload synchronous CPU inference to threadpool to prevent blocking the async event loop
        results = await run_in_threadpool(predict_instruments, temp_path, threshold=threshold)
        results["filename"] = filename

        return results

    except RuntimeError as e:
        logger.error("Model runtime error for '%s': %s", filename, e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected inference error for '%s'", filename)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.warning("Could not remove temp file %s: %s", temp_path, e)