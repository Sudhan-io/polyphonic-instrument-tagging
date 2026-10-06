# Agent Context: AudioTag AI (OpenMIC Multi-Label Engine)

To any future AI agent reading this: This file is the live development context and source of truth for this project. Read it fully before touching anything.

---

## What This Project Is

AudioTag AI is an acoustic deep learning system that analyzes polyphonic music tracks and simultaneously detects up to 18 instruments playing concurrently.

- **Input**: `.ogg`, `.wav`, `.mp3`, `.flac`, `.m4a` audio (normalized to 10.0-second excerpts)
- **Processing**: Log-Mel Spectrogram at 22,050 Hz, 128 mel bands, fmax=8000 Hz
- **Architecture**: AudioResNet-SE (4 Stages with Squeeze-and-Excitation blocks, Dual GAP + GMP, Dropout 0.35, Linear Head)
- **Loss**: Binary Cross-Entropy with `pos_weight` frequency balancing
- **Dataset**: OpenMIC-2018 (20,000 real-world tracks from Free Music Archive)
- **Primary Inference Engine**: `backend/app/models/audiotag_model_v1.onnx` (ONNX Runtime, 3.11ms latency, 4.74x speedup over PyTorch CPU)
- **Primary Training Checkpoint**: `backend/app/models/audiotag_model_v1.pt` (PyTorch 2.5.1+cu121 on NVIDIA RTX 3050 6GB Laptop GPU, Test Macro AUROC: 0.8989)
- **Zero Emojis Policy**: The user strictly enforces no emojis anywhere in the codebase or UI.

**18 Target Instruments**: accordion, bass, cello, clarinet, cymbals, drums, flute, guitar, mallet_percussion, mandolin, piano, saxophone, synthesizer, trombone, trumpet, ukulele, violin, voice

---

## Current Codebase State (Production-Hardened)

```
AudioTag-AI/
├── Dockerfile                    # Production container specification (Python 3.10-slim + ffmpeg)
├── render.yaml                   # Render Blueprint for automatic cloud deployments
├── Procfile                      # Process declaration for web deployment
├── requirements.txt              # Standard root dependencies (ONNX Runtime, PyTorch, FastAPI)
├── requirements-docker.txt       # Lean cloud container dependencies (<410MB image, sub-45s build)
├── templates/
│   ├── base.html                 # Editorial shell, topbar, zero-emoji UI
│   └── index.html                # Native web interface (Acoustic Studio & Dossier modal)
├── static/
│   ├── style.css                 # Editorial CSS system (Georgia serif + orange accents)
│   └── app.js                    # Web Audio API synthesizer, sorting & file upload
├── backend/
│   └── app/
│       ├── main.py               # FastAPI server (Restricted CORS, structured logging, v1.0.2)
│       ├── run.py                # Universal multi-port async runner (0.0.0.0:8000 and :10000)
│       ├── core/
│       │   ├── model_loader.py   # ONNX Runtime primary loader with lazy PyTorch fallback
│       │   └── inference.py      # Micro-batched (chunk 8) multi-label sigmoid inference
│       ├── routes/
│       │   └── analyze.py        # Streaming 64KB chunk disk uploads + 50MB MIME guard
│       ├── models/
│       │   ├── audiotag_model_v1.onnx  # Exported ONNX Runtime graph (3.11 ms latency, tracked in Git)
│       │   ├── audiotag_model_v1.pt    # PyTorch GPU AudioResNet-SE checkpoint (AUROC: 0.8989, gitignored)
│       │   └── audiotag_model_v1.keras # Legacy fallback checkpoint
│       └── utils/
│           └── preprocess.py     # 5.0s hop sliding-window + 45-window adaptive cap + native C FFmpeg
├── Scripts/
│   ├── setup_openmic.py          # OpenMIC-2018 dataset prepper
│   ├── train_openmic_gpu.py      # PyTorch GPU trainer (AMP FP16, pos_weight, SpecAugment)
│   ├── export_onnx.py            # Automated ONNX export, parity verification & benchmark
│   └── openmic-2018/             # Extracted dataset and cache files (cache_X.npy, cache_y.npy)
├── docs/
│   ├── CONCEPTS_AND_ALGORITHMS.md     # Exhaustive unit-by-unit concepts & algorithms guide
│   ├── DECISIONS_AND_ARCHITECTURE.md  # Chronicle of all technical choices and rationale
│   ├── INCIDENT_LOG.md                # Production incident, root cause & telemetry operational log
│   ├── MASTER_PLAN.md                 # System blueprint and roadmap
│   └── TECHNICAL_QA_DOSSIER.md        # Exhaustive 108-question technical interview guide
```

---

## Production Security and Hygiene Invariants

1. **CORS**: Restricted to `http://localhost:8000` and `http://127.0.0.1:8000` by default. Can be overridden using `AUDIOTAG_ALLOWED_ORIGINS` environment variable. Never enable `allow_credentials=True` with wildcard origins.
2. **Upload Guards & Zero-RAM Streaming**: `analyze.py` enforces a 50 MB limit, checks audio MIME/extension, and streams data in 64 KB chunks directly to disk to keep Python upload memory at ~64 KB instead of 50 MB.
3. **No Legacy NSynth Models**: Old models (`instrunet_model_v3.keras` and `instrunet_condition.keras`) have been permanently removed. Do NOT attempt to reference them.
4. **ONNX Runtime First, Lazy PyTorch**: `model_loader.py` exclusively serves `audiotag_model_v1.onnx` by default (~3.11 ms, ~40 MB RAM). PyTorch is wrapped in a lazy import factory so that it is never loaded in cloud containers unless explicitly requested (`AUDIOTAG_ENGINE=pytorch`), preventing out-of-memory errors on 512 MB instances.
5. **Full-Song 5.0-Second Sliding Window & Micro-Batching**: `preprocess.py` uses precomputed `MEL_BASIS` with single-pass STFT and 5.0-second sliding-window slicing with a 45-window adaptive safety ceiling. Inference is executed in micro-batches of 8 windows, bounding total process RAM to under 95 MB and executing full songs in ~3 seconds on Render Free Tier.
6. **Multi-Port Async Cloud Runner**: `run.py` listens concurrently on `0.0.0.0:8000`, `0.0.0.0:10000`, and dynamic `$PORT` within a single asyncio process, eliminating all port-binding mismatches and reverse proxy timeouts.
7. **No Emojis**: Maintain the editorial aesthetic: Georgia serif headlines, `#f7f8f5` paper background, `#18201d` dark ink, `#ff6b00` vibrant orange accent, and geometric/SVG icons.



