import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

for path in [PROJECT_ROOT, BACKEND_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("audiotag")

try:
    from backend.app.routes.analyze import router as analyze_router
    from backend.app.core.model_loader import load_models, get_model
except ImportError:
    from app.routes.analyze import router as analyze_router
    from app.core.model_loader import load_models, get_model

app = FastAPI(
    title="AudioTag AI",
    description="Multi-label instrument detection engine powered by CNN Log-Mel Spectrogram analysis",
    version="1.0.0"
)

# CORS middleware — enable local development and production cloud domains
_raw_origins = os.environ.get(
    "AUDIOTAG_ALLOWED_ORIGINS",
    "*"
)
if _raw_origins.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    # Always guarantee localhost and render production domains are permitted
    for standard_origin in (
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://polyphonic-instrument-tagging.onrender.com"
    ):
        if standard_origin not in ALLOWED_ORIGINS:
            ALLOWED_ORIGINS.append(standard_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates setup
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR) if os.path.exists(TEMPLATES_DIR) else None

# Preload models
load_models()

# Mount API routes
app.include_router(analyze_router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the editorial acoustic web application."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h1>AudioTag AI Web Application</h1><p>Templates not found.</p>")


@app.get("/api/health")
def health_check():
    model, engine = get_model()
    return {
        "status": "online",
        "service": "AudioTag AI",
        "version": "1.0.1",
        "build": "d66f759",
        "engine": engine or "onnx",
        "model_loaded": model is not None
    }