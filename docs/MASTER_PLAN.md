# AudioTag AI — Master Plan & Production Architecture Specification

> **For any developer or AI agent picking this up:** This file is the single source of truth for the project vision, end-to-end architecture, completed milestones, and engineering standards.

---

## 1. What AudioTag AI Is

AudioTag AI is an acoustic deep learning application that analyzes polyphonic, multi-instrument music tracks and simultaneously identifies every instrument playing in the mix with independent confidence scores.

A user uploads an audio file (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`). The system:
1. Normalizes and pads/trims audio to 10.0 seconds at 22,050 Hz Mono.
2. Generates a 128-band Log-Mel Spectrogram (fmax = 8,000 Hz, hop_length = 512).
3. Executes inference via an optimized ONNX Runtime graph (`audiotag_model_v1.onnx`) with sub-4ms latency.
4. Outputs independent sigmoid probabilities across **18 instrument classes**.
5. Renders a native editorial web studio with dynamic confidence bars, interactive instrument dossiers, and in-browser Web Audio API timbre synthesis.

This operates under a **Multi-Label Classification** paradigm: every instrument is an independent binary hypothesis, allowing simultaneous recognition of drums, bass, guitar, vocals, strings, brass, and synthesizers.

---

## 2. End-to-End System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        USER (Web Client)                               │
│                     Editorial Acoustic Studio                          │
│   • Drag & Drop Audio Upload (.wav, .mp3, .ogg, .flac, .m4a)          │
│   • Dynamic Threshold Filtering (0.10 - 0.90) in 0ms (Client-Side)     │
│   • 18 Instrument Confidence Bars + Dual Sorting Toolbar               │
│   • Interactive Instrument Dossier Modal (Metaphors & Mixing Tips)     │
│   • Web Audio API Synthetic Timbre Player (Zero Network Bandwidth)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP POST /analyze
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                                 │
│                      backend/app/main.py                               │
│                                                                        │
│  [Security Guards]                                                     │
│  • CORS: Restricted to local origins (Override via env)                │
│  • Upload Cap: 50 MB hard limit (HTTP 413)                             │
│  • MIME Validator: audio/* and video/* types only (HTTP 415)           │
│                                                                        │
│  [Inference Pipeline: backend/app/core/inference.py]                   │
│  1. Preprocessing: torchaudio / librosa → 128x128 Log-Mel Spectrogram │
│  2. Model Loader: Automatic Engine Resolution                          │
│     ├── Tier 1 (Active): ONNX Runtime (audiotag_model_v1.onnx)         │
│     ├── Tier 2: PyTorch AudioResNet-SE GPU (audiotag_model_v1.pt)     │
│     └── Tier 3: Keras Fallback (audiotag_model_v1.keras)               │
│  3. Multi-Label Sigmoid Activation                                     │
│  4. Spectrogram Base64 PNG Rendering (in-memory buffer)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                             JSON Response:
                     {
                       "status": "success",
                       "engine": "onnx",
                       "duration_seconds": 10.0,
                       "predictions": { "drums": 0.963, "guitar": 0.825, ... },
                       "detected": ["drums", "guitar", "bass", ...],
                       "spectrogram_base64": "iVBORw..."
                     }
```

---

## 3. Deep Learning Pipeline

### 3.1 Acoustic Feature Engineering

```
Input Audio File (.wav / .mp3 / .ogg / .flac)
         │
         ▼
torchaudio / librosa high-speed waveform loader
         │
         ▼
Resample to 22,050 Hz ──▶ Pad or Trim to exactly 10.0s (220,500 samples)
         │
         ▼
melspectrogram(n_mels=128, fmax=8000, hop_length=512) ──▶ power_to_db()
         │
         ▼
fix_length(size=128, axis=1) ──▶ Shape: (128, 128)
         │
         ▼
Min-Max Normalization to [0.0, 1.0]
         │
         ▼
Reshape to 4D Tensor: (Batch, 1, 128, 128) ──▶ Model Input
```

### 3.2 Neural Architecture: AudioResNet-SE

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AudioResNet-SE Engine                           │
│                                                                        │
│  Stem:                                                                 │
│    Conv2D(32 channels, 5x5 kernel, stride=1, padding=2)                │
│    BatchNorm2d ──▶ GELU ──▶ MaxPool2D(2, 2)                           │
│                                                                        │
│  Stage 1 (64 channels):                                                │
│    SEBasicBlock(stride=2) ──▶ SEBasicBlock(stride=1)                   │
│                                                                        │
│  Stage 2 (128 channels):                                               │
│    SEBasicBlock(stride=2) ──▶ SEBasicBlock(stride=1)                   │
│                                                                        │
│  Stage 3 (256 channels):                                               │
│    SEBasicBlock(stride=2) ──▶ SEBasicBlock(stride=1)                   │
│                                                                        │
│  Dual Global Pooling:                                                  │
│    AdaptiveAvgPool2D(1, 1) ⊕ AdaptiveMaxPool2D(1, 1) ──▶ 512 dims     │
│                                                                        │
│  Classification Head:                                                  │
│    Linear(512 → 256) ──▶ BatchNorm1d ──▶ GELU ──▶ Dropout(0.35)       │
│    Linear(256 → 18 logits)                                             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
       18 Independent Logits ──▶ Sigmoid Probability [0.0 - 1.0]
```

### 3.3 Training & Optimization Strategy
- **Dataset**: OpenMIC-2018 (20,000 real-world tracks, Free Music Archive).
- **Loss Function**: `BCEWithLogitsLoss` with dynamic positive class weighting (`pos_weight`). Eliminates majority-class bias (e.g., vocal dominance) and forces the network to learn rare acoustic classes (violin, flute, mandolin).
- **Precision**: Automatic Mixed Precision (`FP16` AMP) on NVIDIA RTX 3050 6GB Laptop GPU.
- **Regularization**: SpecAugment (frequency and time masking) + Dropout (0.35).
- **Test Metric**: **0.8989 Macro AUROC**.

### 3.4 Inference Optimization: ONNX Runtime
- Exported via `torch.onnx.export` with Opset 17 and dynamic batching.
- Model file: `backend/app/models/audiotag_model_v1.onnx` (`11.17 MB`).
- Operator fusion merges Conv2D, BatchNorm, and GELU into single execution kernels.
- **Latency**: **3.11 ms** per 10-second audio clip (**4.74x faster** than PyTorch CPU).
- **Numerical Parity**: Max probability delta against PyTorch is **`0.00000000`**.

---

## 4. The 18 Target Instrument Classes

| # | Instrument | Typical Frequency Range | Acoustic Family |
|---|---|---|---|
| 0 | Accordion | 60 Hz – 3.5 kHz | Free-reed aerophone |
| 1 | Bass | 40 Hz – 1 kHz | Electric / acoustic bass |
| 2 | Cello | 65 Hz – 2.5 kHz | Bowed string chordophone |
| 3 | Clarinet | 115 Hz – 3 kHz | Single-reed woodwind |
| 4 | Cymbals | 3 kHz – 16 kHz | Unpitched metallic percussion |
| 5 | Drums | 50 Hz – 8 kHz | Percussion kit |
| 6 | Flute | 250 Hz – 8 kHz | Non-reed aerophone |
| 7 | Guitar | 80 Hz – 5 kHz | Acoustic / electric guitar |
| 8 | Mallet Percussion | 100 Hz – 6 kHz | Marimba, vibraphone, xylophone |
| 9 | Mandolin | 196 Hz – 5 kHz | Plucked string chordophone |
| 10 | Piano | 27 Hz – 4.2 kHz | Acoustic / electric piano |
| 11 | Saxophone | 100 Hz – 8 kHz | Single-reed woodwind |
| 12 | Synthesizer | 20 Hz – 20 kHz | Electronic synthesis |
| 13 | Trombone | 80 Hz – 1 kHz | Low brass aerophone |
| 14 | Trumpet | 160 Hz – 8 kHz | High brass aerophone |
| 15 | Ukulele | 260 Hz – 4 kHz | Small chordophone |
| 16 | Violin | 196 Hz – 8 kHz | High bowed string chordophone |
| 17 | Voice | 85 Hz – 4 kHz | Lead / backing vocals |

---

## 5. Production Codebase Structure

```
AudioTag-AI/
├── requirements.txt              # Production dependencies (ONNX Runtime, PyTorch, FastAPI)
├── AGENT_CONTEXT.md              # Live agent developer context & invariants
├── README.md                     # Public documentation and setup guide
│
├── backend/
│   └── app/
│       ├── main.py               # FastAPI server (Restricted CORS, structured logging)
│       ├── core/
│       │   ├── model_loader.py   # Dual ONNX Runtime / PyTorch loader hierarchy
│       │   └── inference.py      # 18-label inference + spectrogram image generation
│       ├── routes/
│       │   └── analyze.py        # POST /analyze with 50MB upload cap & MIME guard
│       ├── utils/
│       │   └── preprocess.py     # torchaudio / librosa Log-Mel pipeline
│       └── models/
│           ├── audiotag_model_v1.onnx  # Primary ONNX graph (3.11 ms latency)
│           ├── audiotag_model_v1.pt    # PyTorch GPU checkpoint (0.8989 AUROC)
│           └── audiotag_model_v1.keras # Legacy fallback checkpoint
│
├── templates/
│   ├── base.html                 # Topbar navigation and layout shell
│   └── index.html                # Editorial acoustic studio & instrument dossier
│
├── static/
│   ├── app.js                    # Web Audio API synthesizer, sorting & file upload
│   └── style.css                 # Editorial design system (Georgia serif + orange accents)
│
├── Scripts/
│   ├── setup_openmic.py          # OpenMIC-2018 downloader & label extractor
│   ├── train_openmic_gpu.py      # PyTorch GPU training with mixed precision
│   ├── export_onnx.py            # Automated ONNX export & benchmark script
│   └── openmic-2018/             # Extracted dataset and cached feature arrays
│
└── docs/
    ├── DECISIONS_AND_ARCHITECTURE.md  # Comprehensive technical decisions log
    └── MASTER_PLAN.md                 # This system specification
```

---

## 6. Completed Milestones & Build Log

### Phase 0: Legacy Clean-Up & Problem Framing
- [x] Audited legacy NSynth code; proved that single-note monophonic datasets fail for real music.
- [x] Identified multi-label requirement and switched classification objective from Softmax to independent Sigmoids.
- [x] Purged legacy NSynth models (`instrunet_model_v3.keras`, `instrunet_condition.keras`).
- [x] Reclaimed **~2.74 GB** of storage by deleting obsolete models, raw archives, and stale scripts.

### Phase 1: Dataset Acquisition & Preprocessing
- [x] Downloaded and unpacked OpenMIC-2018 (20,000 real-world polyphonic tracks).
- [x] Constructed `clean_multilabels.json` label dictionary.
- [x] Pre-computed and cached Log-Mel Spectrogram features into `cache_X.npy` and `cache_y.npy` for sub-second data loading.

### Phase 2: Neural Architecture & Training
- [x] Designed `AudioResNet-SE` with 4 Squeeze-and-Excitation stages and dual GAP+GMP pooling.
- [x] Resolved Windows GPU bottleneck by migrating from CPU TensorFlow to PyTorch 2.5 + CUDA 12.1 on NVIDIA RTX 3050.
- [x] Implemented dynamic `pos_weight` frequency balancing to eliminate vocal dominance.
- [x] Achieved **0.8989 Test Macro AUROC**. Checkpoint saved to `audiotag_model_v1.pt`.

### Phase 3: Inference Engine Optimization (ONNX)
- [x] Converted PyTorch checkpoint to ONNX graph (`audiotag_model_v1.onnx`, Opset 17, dynamic batching).
- [x] Verified exact numerical parity (max probability difference: `0.00000000`).
- [x] Benchmarked latency: reduced CPU inference time from `14.74 ms` to **`3.11 ms`** (**`4.74x speedup`**).
- [x] Integrated ONNX Runtime into `backend/app/core/model_loader.py` with automatic fallback hierarchy.

### Phase 4: Production Web Application
- [x] Replaced slow Streamlit UI with a bespoke editorial acoustic web application.
- [x] Built responsive UI in Vanilla JavaScript with zero server roundtrips for threshold slider manipulation.
- [x] Enforced strict editorial style guide: Georgia serif headlines, warm paper background (`#f7f8f5`), dark ink text (`#18201d`), vibrant orange accents (`#ff6b00`), and zero emojis.
- [x] Integrated in-browser Web Audio API acoustic timbre synthesizer and 3-layer instrument dossier modals.
- [x] Added dual-mode sorting toolbar (Alphabetical vs. Detected Probability).

### Phase 5: Security Hardening & Observability
- [x] Restricted CORS policy to eliminate CSRF risks; removed wildcard origins with credentials.
- [x] Added 50 MB upload size guard and `audio/*` / `video/*` MIME type validation.
- [x] Replaced unformatted `print()` statements with structured, timestamped Python logging.
- [x] Tracked production ONNX model in Git for container builds while ignoring heavy raw checkpoints.

### Phase 6: Cloud Containerization & Audio Decoding Optimization
- [x] Isolated project from legacy history into dedicated repository `Sudhan-io/polyphonic-instrument-tagging`.
- [x] Authored production `Dockerfile` (Python 3.10-slim, native `ffmpeg` & `libsndfile1`, dynamic `$PORT` binding), `render.yaml`, and `Procfile`.
- [x] Successfully deployed live web service to Render: `https://polyphonic-instrument-tagging.onrender.com/`.
- [x] Diagnosed free-tier cloud CPU/memory bottlenecks on multi-minute audio files (Render 0.1 CPU core, 512MB RAM limit).
- [x] Implemented direct 10.0s container streaming decode (`librosa.load(..., duration=10.0)`), reducing decode from 3 minutes to 25 milliseconds (7,000x speedup).
- [x] Engineered lazy-loaded PyTorch factory in `model_loader.py`, cutting container RAM from ~450MB down to ~70MB and eliminating OOM crashes.
- [x] Extracted audio duration via container header reading (`soundfile.info`) instead of full file scanning.
- [x] Documented architectural roadmap for True Full-Song Time-Segmented Sliding Window Heatmap Analysis.

---

## 7. Current Project Status

- **Build Status**: **Production Ready & Deployed**
- **Public Cloud Service**: `https://polyphonic-instrument-tagging.onrender.com/`
- **Active Web Service**: `http://127.0.0.1:8000` (FastAPI + Uvicorn)
- **Active Inference Engine**: ONNX Runtime (`audiotag_model_v1.onnx`)
- **Active Training Pipeline**: PyTorch GPU (`Scripts/train_openmic_gpu.py`)
- **Documentation Status**: 100% synchronized across `README.md`, `AGENT_CONTEXT.md`, `docs/DECISIONS_AND_ARCHITECTURE.md`, and `docs/MASTER_PLAN.md`.

