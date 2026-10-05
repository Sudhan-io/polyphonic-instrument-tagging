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
├── requirements.txt              # Standard root requirements (ONNX Runtime, PyTorch, FastAPI)
├── templates/
│   ├── base.html                 # Editorial shell, topbar, zero-emoji UI
│   └── index.html                # Native web interface (Acoustic Studio & Dossier modal)
├── static/
│   ├── style.css                 # Editorial CSS system (Georgia serif + orange accents)
│   └── app.js                    # Web Audio API synthesizer, sorting & file upload
├── backend/
│   └── app/
│       ├── main.py               # FastAPI server (Restricted CORS, structured logging)
│       ├── core/
│       │   ├── model_loader.py   # ONNX Runtime primary loader with PyTorch GPU fallback
│       │   └── inference.py      # 18-class multi-label sigmoid inference + base64 spectrogram
│       ├── routes/
│       │   └── analyze.py        # POST /analyze with 50MB upload cap & audio MIME guard
│       ├── models/
│       │   ├── audiotag_model_v1.onnx  # Exported ONNX Runtime graph (3.11 ms latency)
│       │   ├── audiotag_model_v1.pt    # PyTorch GPU AudioResNet-SE checkpoint (AUROC: 0.8989)
│       │   └── audiotag_model_v1.keras # Legacy fallback checkpoint
│       └── utils/
│           └── preprocess.py     # torchaudio/librosa Log-Mel spectrogram preprocessing
├── Scripts/
│   ├── setup_openmic.py          # OpenMIC-2018 dataset prepper
│   ├── train_openmic_gpu.py      # PyTorch GPU trainer (AMP FP16, pos_weight, SpecAugment)
│   ├── export_onnx.py            # Automated ONNX export, parity verification & benchmark
│   └── openmic-2018/             # Extracted dataset and cache files (cache_X.npy, cache_y.npy)
├── docs/
    ├── DECISIONS_AND_ARCHITECTURE.md  # Chronicle of all technical choices and rationale
    └── MASTER_PLAN.md                 # System blueprint and roadmap
```

---

## Production Security and Hygiene Invariants

1. **CORS**: Restricted to `http://localhost:8000` and `http://127.0.0.1:8000` by default. Can be overridden using `AUDIOTAG_ALLOWED_ORIGINS` environment variable. Never enable `allow_credentials=True` with wildcard origins.
2. **Upload Guards**: `analyze.py` enforces a 50 MB file size limit and checks for `audio/*` / `video/*` MIME prefixes before disk writes.
3. **No Legacy NSynth Models**: Old models (`instrunet_model_v3.keras` and `instrunet_condition.keras`) have been permanently removed. Do NOT attempt to reference them.
4. **PyTorch First**: `model_loader.py` exclusively serves `audiotag_model_v1.pt` onto CUDA if an NVIDIA GPU is available, CPU otherwise.
5. **No Emojis**: Maintain the editorial aesthetic: Georgia serif headlines, `#f7f8f5` paper background, `#18201d` dark ink, `#ff6b00` vibrant orange accent, and geometric/SVG icons.
