# AudioTag AI — Multi-Label Music Tagging Engine

[![Live Demo](https://img.shields.io/badge/Render-Live_Demo-00E599.svg)](https://polyphonic-instrument-tagging.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.30-005CED.svg)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![Dataset](https://img.shields.io/badge/Dataset-OpenMIC--2018-purple.svg)](https://zenodo.org/record/1432913)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Live Web Application:** [https://polyphonic-instrument-tagging.onrender.com/](https://polyphonic-instrument-tagging.onrender.com/)  
> **Interactive API Documentation (Swagger):** [https://polyphonic-instrument-tagging.onrender.com/docs](https://polyphonic-instrument-tagging.onrender.com/docs)  
> **Service Health Endpoint:** [https://polyphonic-instrument-tagging.onrender.com/api/health](https://polyphonic-instrument-tagging.onrender.com/api/health)  
> **Technical Q&A & Interview Dossier (108 Questions):** [docs/TECHNICAL_QA_DOSSIER.md](docs/TECHNICAL_QA_DOSSIER.md)

AudioTag AI is an acoustic multi-label recognition system that analyzes complex polyphonic music recordings and simultaneously detects the presence of up to 18 instruments.


Unlike legacy single-instrument classifiers that assume only one instrument sounds at a time (single-label softmax), AudioTag AI formulates acoustic tagging as a multi-label classification problem. Powered by an AudioResNet-SE deep neural network exported to a kernel-fused **ONNX Runtime** engine, it computes independent sigmoid activations for each instrument class, resolving concurrent guitars, drums, bass, vocals, brass, and strings with sub-4ms latency.

---

## System Architecture

```
┌────────────────────────────────────────────────────────┐
│             Editorial Acoustic Web Client              │
│   FastAPI + Jinja2 + Vanilla CSS & JS + Web Audio API  │
│   • Drag & Drop Audio Analysis (WAV / MP3 / OGG / FLAC)│
│   • Panoramic Song Instrumentation Timeline Heatmap    │
│   • Interactive Window-by-Window Mix Inspector         │
│   • 18-Way Dynamic Confidence Bar Spectrum             │
│   • Dual-Mode Sorting: Alphabetical vs. Probability    │
│   • Log-Mel Spectrogram Visualization (128 x 128)      │
│   • Interactive Instrument Timbre Synthesis & Dossier  │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP POST /analyze
                            ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│                                                        │
│  1. Audio Input (.wav, .mp3, .ogg, .flac, .m4a)        │
│  2. File Guard: 50 MB upload limit & MIME validation   │
│  3. C-Level Audio Streaming (soundfile + soxr SIMD)    │
│  4. Single-Pass STFT with Cached 128-band MEL_BASIS    │
│  5. 2D Spectrogram Frame Slicing (128x128 windows)     │
│  6. Multi-Engine Vectorized Batch Inference:           │
│     ├── Tier 1 (Active): ONNX Runtime Graph            │
│     │   (audiotag_model_v1.onnx, 48ms for 71 windows)  │
│     ├── Tier 2 (Fallback): PyTorch GPU AudioResNet-SE  │
│     │   (audiotag_model_v1.pt, 0.8989 Test AUROC)      │
│     └── Tier 3 (Legacy): Keras Baseline Fallback       │
│  7. Dual Aggregation: Global Scores + Timeline Heatmap │
│                                                        │
│  Output: Detections + Timeline Matrix + Spectrogram    │
└────────────────────────────────────────────────────────┘
```


---

## Inference Acceleration & Performance Benchmarks

The trained PyTorch checkpoint was converted to an optimized ONNX computational graph using `torch.onnx.export` (Opset 17) with dynamic batching. ONNX Runtime applies operator fusion—merging adjacent Conv2D, BatchNorm, GELU, and Squeeze-and-Excitation layers into unified execution kernels.

### Benchmark Comparison (50 Iterations)

| Metric | Native PyTorch (CPU) | ONNX Runtime (CPU) | Performance Gain |
|---|---|---|---|
| **Average Latency** | 14.74 ms | **3.11 ms** | **4.74x faster** |
| **Numerical Parity** | Baseline | **0.00000000** | **Exact numerical match** |
| **Checkpoint Size** | 11.74 MB (`.pt`) | **11.17 MB** (`.onnx`) | **5% smaller** |
| **Execution Providers** | CUDA / CPU | `CUDAExecutionProvider`, `CPUExecutionProvider` | Dynamic cross-platform fallback |

### Dynamic Engine Hierarchy
The backend loader (`model_loader.py`) automatically routes requests through the fastest available engine:
1. **ONNX Runtime (Default)**: Executes `audiotag_model_v1.onnx` via C++ SIMD vectorization and fused kernels (~3.11 ms).
2. **Native PyTorch (Fallback or Direct)**: Can be explicitly selected by setting the environment variable `AUDIOTAG_ENGINE=pytorch`.
3. **Keras (Legacy Fallback)**: Reserved for environments missing both PyTorch and ONNX runtimes.

---

## Full-Song Vectorized Sliding-Window Analysis & Timeline Heatmap

AudioTag AI processes both short 10-second clips and entire 3-to-5 minute songs without degrading accuracy or exceeding cloud free-tier constraints (512 MB RAM, 0.1 CPU core):

1. **Precomputed Mel Filterbank (`MEL_BASIS`)**: The 128-band triangular Mel matrix is cached at server startup, accelerating Fourier transforms by **21.5x**.
2. **Single-Pass Full-Audio STFT**: Rather than running tens of separate STFT passes, a single Fourier transform is computed across the entire track in **~890 ms**.
3. **2D Spectrogram Frame Slicing**: Windows of width 128 frames (corresponding to 10 seconds of acoustic context) are extracted directly from the 2D spectrogram matrix (`mel_db[:, start_f:end_f]`) in **2.7 ms** without re-running Fourier transforms.
4. **Vectorized Batch ONNX Inference**: All windows are stacked into a 4D batch tensor `(N, 1, 128, 128)`. ONNX Runtime executes fused SIMD kernels across all 71 windows of a 3.5-minute song in **48.2 ms**.
5. **Interactive Timeline Heatmap**: The web client renders a visual timeline showing exactly when each instrument enters and leaves, with live scrubbing and real-time response to the cutoff slider.

| Stage | Latency (210s Track) | Peak Memory | Operational Mechanism |
|---|---|---|---|
| **Audio Loading & Resampling** | ~200 ms | 18.5 MB | Native C `soundfile` + SIMD `soxr.resample(quality='QQ')` |
| **Single-Pass STFT + Mel** | 895 ms | 4.6 MB | Cached `MEL_BASIS` matrix multiplication |
| **Spectrogram Window Slicing (71 windows)** | 2.7 ms | 1.4 MB | Strided 2D NumPy array slicing |
| **Batch ONNX Inference (71 windows)** | 48.2 ms | 1.4 MB | Dynamic batch execution on C++ ONNX Runtime |
| **Total Track Pipeline** | **~1.5 to 2.1 s** | **~95 MB** | **Fits comfortably within 0.1 CPU core and 512 MB RAM limit** |

---

## 18 Target Instrument Classes


| Instrument | Classification Type | Typical Frequency Range |
|---|---|---|
| Accordion | Free-reed aerophone | 60 Hz – 3.5 kHz |
| Bass | Electric / Acoustic Bass | 40 Hz – 1 kHz |
| Cello | Bowed string chordophone | 65 Hz – 2.5 kHz |
| Clarinet | Single-reed woodwind | 115 Hz – 3 kHz |
| Cymbals | Unpitched metallic percussion | 3 kHz – 16 kHz |
| Drums | Percussion kit | 50 Hz – 8 kHz |
| Flute | Non-reed aerophone | 250 Hz – 8 kHz |
| Guitar | Acoustic / Electric guitar | 80 Hz – 5 kHz |
| Mallet Percussion | Marimba, vibraphone, xylophone | 100 Hz – 6 kHz |
| Mandolin | Plucked string chordophone | 196 Hz – 5 kHz |
| Piano | Acoustic / Electric piano | 27 Hz – 4.2 kHz |
| Saxophone | Single-reed woodwind | 100 Hz – 8 kHz |
| Synthesizer | Electronic analog / digital synthesis | 20 Hz – 20 kHz |
| Trombone | Low brass aerophone | 80 Hz – 1 kHz |
| Trumpet | High brass aerophone | 160 Hz – 8 kHz |
| Ukulele | Small chordophone | 260 Hz – 4 kHz |
| Violin | High bowed string chordophone | 196 Hz – 8 kHz |
| Voice | Lead / Backing human vocals | 85 Hz – 4 kHz |

---

## Project Structure

```
AudioTag-AI/
├── Dockerfile                    # Production container specification (Python 3.10-slim + ffmpeg)
├── render.yaml                   # Render Blueprint for automatic cloud deployments
├── Procfile                      # Process declaration for web deployment
├── requirements.txt              # Core dependencies (ONNX Runtime, PyTorch, FastAPI, librosa)
├── AGENT_CONTEXT.md              # Architectural invariants and developer guide
├── README.md                     # Project overview and setup instructions
│
├── backend/
│   └── app/
│       ├── main.py               # FastAPI application with restricted CORS & logging
│       ├── core/
│       │   ├── model_loader.py   # ONNX Runtime primary loader with lazy PyTorch fallback
│       │   └── inference.py      # Batch sliding-window inference & timeline generation
│       ├── routes/
│       │   └── analyze.py        # POST /analyze with file size & MIME validation
│       ├── utils/
│       │   └── preprocess.py     # Vectorized single-pass STFT & sliding-window pipeline
│       └── models/
│           ├── audiotag_model_v1.onnx  # Exported ONNX Runtime graph (3.11 ms, tracked in Git)
│           ├── audiotag_model_v1.pt    # PyTorch AudioResNet-SE checkpoint (Macro AUROC: 0.8989)
│           └── audiotag_model_v1.keras # Legacy fallback checkpoint
│
├── templates/
│   ├── base.html                 # Shell layout and navigation
│   └── index.html                # Editorial acoustic studio, dossier & timeline heatmap
│
├── static/
│   ├── app.js                    # Web Audio API synthesizer, timeline heatmap & sorting
│   └── style.css                 # Editorial typography, timeline lanes & design system
│
├── Scripts/
│   ├── setup_openmic.py          # Resumable OpenMIC-2018 downloader & label generator
│   ├── train_openmic_gpu.py      # PyTorch GPU training with mixed precision & SE blocks
│   ├── export_onnx.py            # Automated ONNX export, parity verification & benchmark
│   └── openmic-2018/             # Extracted dataset and cached feature arrays
│
└── docs/
    ├── MASTER_PLAN.md            # Comprehensive project roadmap & milestones
    ├── DECISIONS_AND_ARCHITECTURE.md # Complete architectural decisions log
    └── TECHNICAL_QA_DOSSIER.md   # Exhaustive 50-Question Technical Interview & Systems Guide
```

---

## Deep Technical Q&A & Interview Dossier

For engineers, recruiters, and hiring managers seeking a granular breakdown of every technical decision, trade-off, and mathematical formula across this system, see the **[Technical Q&A & Interview Dossier](docs/TECHNICAL_QA_DOSSIER.md)**.

It contains **50 comprehensive, in-depth questions and answers (each 50+ words)** with exact benchmarks, mathematical proofs, and "why / why not" architectural analyses spanning:
- **Section 1: Problem Formulation & Core Paradigm** (Multi-label sigmoids, polyphonic frequency overlap, taxonomy selection)
- **Section 2: Neural Architecture & Deep Learning Design** (AudioResNet-SE, Squeeze-and-Excitation attention, Dual GAP+GMP, SpecAugment, AMP FP16)
- **Section 3: Loss Function, Optimization & Class Imbalance** (`BCEWithLogitsLoss` stability, `pos_weight` derivation, Macro vs Micro AUROC)
- **Section 4: Digital Signal Processing (DSP) & Acoustic Features** (22,050 Hz Nyquist, 128 Mel bands, Heisenberg uncertainty, precomputed `MEL_BASIS` 21x speedup, `soxr` SIMD)
- **Section 5: Full-Song Vectorized Sliding-Window Analysis** (Single-pass STFT, 2D matrix frame slicing, dynamic batch ONNX, interval merging)
- **Section 6: Inference Acceleration & ONNX Runtime** (Graph export, operator fusion, 3.11 ms latency, 4.74x speedup, `0.00000000` numerical parity)
- **Section 7: Cloud Deployment, Docker & Resource Optimization** (Render 512MB RAM optimization, lazy PyTorch loading, `ffmpeg`/`libsndfile1` containerization)
- **Section 8: Web Architecture, API Security & Editorial UX** (Bespoke FastAPI + Vanilla JS, Web Audio API synthesis, CSRF/DoS guards, Georgia serif design)

---


## Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Sudhan-io/polyphonic-instrument-tagging.git
cd polyphonic-instrument-tagging

# Install dependencies (includes onnx and onnxruntime)
pip install -r requirements.txt
```

If utilizing NVIDIA GPU acceleration:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run the Application

Start the FastAPI application:
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

- Web Interface: `http://localhost:8000`
- Interactive API Documentation: `http://localhost:8000/docs`
- Service Health Endpoint: `http://localhost:8000/api/health`

### 3. Model Export & Benchmarking (Optional)

To re-export the PyTorch checkpoint to ONNX and verify mathematical parity:
```bash
python Scripts/export_onnx.py
```
This runs numerical verification across 50 iterations, reports max delta, and benchmarks latency against native PyTorch.

### 4. Model Training (Optional)

To train or fine-tune AudioResNet-SE on OpenMIC-2018:
```bash
python Scripts/train_openmic_gpu.py --epochs 30 --batch_size 64
```
- Converts 20,000 clips to 128x128 Log-Mel spectrograms.
- Features are cached in `cache_X.npy` and `cache_y.npy` for rapid subsequent epochs.
- Incorporates Pos-Weighted BCE Loss, Mixed Precision (AMP), and Squeeze-and-Excitation blocks.
- Checkpoint is saved to `backend/app/models/audiotag_model_v1.pt`.

---

## REST API Specification

### `POST /analyze`

Analyzes an uploaded audio file (short excerpt or full song) and outputs independent confidence scores, detected instruments, and a window-by-window timeline heatmap.

**Headers:**
`Content-Type: multipart/form-data`

**Parameters:**
- `file`: Audio file (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`). Maximum file size: 50 MB.
- `threshold` *(optional query parameter, float, default `0.5`)*: Decision threshold for positive presence.

**Response (200 OK):**
```json
{
  "status": "success",
  "filename": "track.mp3",
  "duration_seconds": 210.0,
  "engine": "onnx",
  "predictions": {
    "accordion": 0.0124,
    "bass": 0.7412,
    "cello": 0.0381,
    "clarinet": 0.0815,
    "cymbals": 0.8921,
    "drums": 0.9634,
    "flute": 0.0092,
    "guitar": 0.8245,
    "mallet_percussion": 0.0211,
    "mandolin": 0.0187,
    "piano": 0.1142,
    "saxophone": 0.0652,
    "synthesizer": 0.4512,
    "trombone": 0.0278,
    "trumpet": 0.0489,
    "ukulele": 0.0103,
    "violin": 0.0573,
    "voice": 0.7891
  },
  "detected": ["drums", "cymbals", "guitar", "voice", "bass"],
  "threshold": 0.5,
  "is_full_song": true,
  "num_windows": 71,
  "timeline": [
    {
      "window_index": 0,
      "start": 0.0,
      "end": 2.97,
      "display": "0:00 - 0:02",
      "predictions": { "guitar": 0.8245, "drums": 0.1245 },
      "detected": ["guitar"]
    }
  ],
  "timeline_summary": {
    "drums": {
      "peak": 0.9634,
      "mean": 0.8412,
      "presence_percent": 82.5,
      "intervals": ["0:15 - 1:45", "2:05 - 3:30"]
    }
  },
  "spectrogram_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

---

## Cloud Deployment (Docker & Render)

AudioTag AI is containerized for zero-configuration cloud deployment:

- **Live Production URL:** [https://polyphonic-instrument-tagging.onrender.com/](https://polyphonic-instrument-tagging.onrender.com/)
- **Swagger Documentation:** [https://polyphonic-instrument-tagging.onrender.com/docs](https://polyphonic-instrument-tagging.onrender.com/docs)
- **Health Check Endpoint:** [https://polyphonic-instrument-tagging.onrender.com/api/health](https://polyphonic-instrument-tagging.onrender.com/api/health)

### Container Architecture
The repository includes a production [`Dockerfile`](Dockerfile), [`render.yaml`](render.yaml), and [`Procfile`](Procfile):
- **Base Image:** `python:3.10-slim`
- **Native Audio Codecs:** Pre-installs `ffmpeg` and `libsndfile1`
- **Execution Engine:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- **Model Inversion:** The 11.17 MB ONNX model graph is tracked in Git, ensuring cloud builds have the model ready immediately without S3/GCS download dependencies.

---

## Evaluation & Metrics

- **Loss Function:** Binary Cross-Entropy with Pos-Weight adjustment for label imbalance.
- **Evaluation Metric:** Multi-label Macro AUROC (Current checkpoint: **0.8989**).
- **Inference Latency:** **3.11 ms** per 10-second audio track via ONNX Runtime (**4.74x faster** than PyTorch CPU).
- **Full-Song Analysis:** **1.53 s** for a 3.5-minute song (71 sliding windows in **48.2 ms** batch inference).

---

## Technical Q&A & Interview Dossier (108 Questions)

An exhaustive, 108-question technical guide covering every aspect of the AudioTag AI engineering stack—including DSP mathematics, deep learning architectures, loss formulations, sliding-window analysis, memory profiling, ONNX acceleration, cloud containerization, security, and edge-case failure modes—is documented in the project repository:

**Read the Full Dossier:** [docs/TECHNICAL_QA_DOSSIER.md](docs/TECHNICAL_QA_DOSSIER.md)

### Dossier Thematic Sections:
1. **[Problem Framing, Music Information Retrieval (MIR) & Organology](docs/TECHNICAL_QA_DOSSIER.md#section-1-problem-framing-music-information-retrieval-mir--organology)**: Multi-label vs. multi-class, overlapping frequencies, 18 Hornbostel-Sachs classes, AMT vs. source separation, synth vs. acoustic timbre. (Q1 - Q10)
2. **[Neural Architecture & Deep Learning Engineering](docs/TECHNICAL_QA_DOSSIER.md#section-2-neural-architecture--deep-learning-engineering)**: AudioResNet-SE, Squeeze-and-Excitation attention mathematics, Dual Pooling (GAP+GMP), SpecAugment, AMP FP16, GELU vs. ReLU. (Q11 - Q22)
3. **[Loss Function, Optimization & Class Imbalance](docs/TECHNICAL_QA_DOSSIER.md#section-3-loss-function-optimization--class-imbalance)**: BCEWithLogitsLoss numerical stability, pos_weight derivation, Macro vs. Micro AUROC, AdamW vs. SGD, Cosine Annealing. (Q23 - Q32)
4. **[Digital Signal Processing (DSP) & Acoustic Features](docs/TECHNICAL_QA_DOSSIER.md#section-4-digital-signal-processing-dsp--acoustic-features)**: 22,050 Hz Nyquist sampling, 128 Mel bands, n_fft=2048 / hop_length=512 Heisenberg-Gabor tradeoff, precomputed MEL_BASIS, soxr SIMD decimation. (Q33 - Q46)
5. **[Full-Song Vectorized Sliding-Window Analysis Engine](docs/TECHNICAL_QA_DOSSIER.md#section-5-full-song-vectorized-sliding-window-analysis-engine)**: Single-pass STFT mathematical parity, 2D matrix frame slicing, 50% temporal overlap, peak vs. active-mean aggregation, interval merging. (Q47 - Q60)
6. **[Inference Acceleration & ONNX Runtime](docs/TECHNICAL_QA_DOSSIER.md#section-6-inference-acceleration--onnx-runtime)**: Operator fusion, dynamic batching, 3.11 ms single clip latency, 48.2 ms 71-window batch inference, lazy PyTorch loading. (Q61 - Q72)
7. **[Cloud Deployment, Docker & Resource Optimization](docs/TECHNICAL_QA_DOSSIER.md#section-7-cloud-deployment-docker--resource-optimization)**: Render 512 MB RAM budget, multi-stage Docker build, ffmpeg/libsndfile integration, horizontal scaling, zero-cost production. (Q73 - Q86)
8. **[Web Architecture, API Security & Editorial UX](docs/TECHNICAL_QA_DOSSIER.md#section-8-web-architecture-api-security--editorial-ux)**: 50 MB streaming guard, magic byte MIME sniffing, zero-latency client-side threshold scrubbing, Web Audio API synthesis. (Q87 - Q98)
9. **[Edge Cases, Failure Modes & Interview Curveballs](docs/TECHNICAL_QA_DOSSIER.md#section-9-edge-cases-failure-modes--interview-curveballs)**: Heavy distortion/overdrive, dense multi-timbral polyphony, extreme low-end masking, live microphone streaming, scaling to 10k RPM. (Q99 - Q108)