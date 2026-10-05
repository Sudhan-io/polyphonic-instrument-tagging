# AudioTag AI — Multi-Label Music Tagging Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.30-005CED.svg)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![Dataset](https://img.shields.io/badge/Dataset-OpenMIC--2018-purple.svg)](https://zenodo.org/record/1432913)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AudioTag AI is an acoustic multi-label recognition system that analyzes complex polyphonic music recordings and simultaneously detects the presence of up to 18 instruments.

Unlike legacy single-instrument classifiers that assume only one instrument sounds at a time (single-label softmax), AudioTag AI formulates acoustic tagging as a multi-label classification problem. Powered by an AudioResNet-SE deep neural network exported to a kernel-fused **ONNX Runtime** engine, it computes independent sigmoid activations for each instrument class, resolving concurrent guitars, drums, bass, vocals, brass, and strings with sub-4ms latency.

---

## System Architecture

```
┌────────────────────────────────────────────────────────┐
│             Editorial Acoustic Web Client              │
│   FastAPI + Jinja2 + Vanilla CSS & JS + Web Audio API  │
│   • Drag & Drop Audio Analysis (WAV / MP3 / OGG / FLAC)│
│   • 18-Way Dynamic Confidence Bar Breakdown            │
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
│  3. Resample & Normalize to 10.0s @ 22,050 Hz Mono     │
│  4. Log-Mel Spectrogram (128 bands, fmax=8000 Hz)      │
│  5. Normalization to [0, 1] → Input Tensor: (1,1,128,128)
│  6. Multi-Engine Inference Loader:                     │
│     ├── Tier 1 (Active): ONNX Runtime Graph (~3ms)     │
│     │   (audiotag_model_v1.onnx, Opset 17, Fused Ops)  │
│     ├── Tier 2 (Fallback): PyTorch GPU AudioResNet-SE  │
│     │   (audiotag_model_v1.pt, 0.8989 Test AUROC)      │
│     └── Tier 3 (Legacy): Keras Baseline Fallback       │
│  7. 18 Independent Sigmoid Probabilities               │
│                                                        │
│  Output: Multi-label detections + base64 spectrogram   │
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
├── requirements.txt              # Core dependencies (ONNX Runtime, PyTorch, FastAPI, librosa)
├── AGENT_CONTEXT.md              # Architectural invariants and developer guide
├── README.md                     # Project overview and setup instructions
│
├── backend/
│   └── app/
│       ├── main.py               # FastAPI application with restricted CORS & logging
│       ├── core/
│       │   ├── model_loader.py   # ONNX Runtime primary loader with PyTorch fallback
│       │   └── inference.py      # Multi-label inference & spectrogram generation
│       ├── routes/
│       │   └── analyze.py        # POST /analyze with file size & MIME validation
│       ├── utils/
│       │   └── preprocess.py     # High-speed torchaudio/librosa Log-Mel pipeline
│       └── models/
│           ├── audiotag_model_v1.onnx  # Exported ONNX Runtime graph (3.11 ms latency)
│           ├── audiotag_model_v1.pt    # PyTorch AudioResNet-SE checkpoint (Macro AUROC: 0.8989)
│           └── audiotag_model_v1.keras # Legacy fallback checkpoint
│
├── templates/
│   ├── base.html                 # Shell layout and navigation
│   └── index.html                # Editorial acoustic studio and taxonomy dossier
│
├── static/
│   ├── app.js                    # Web Audio API synthesizer, sorting & file upload
│   └── style.css                 # Editorial typography and design system
│
├── Scripts/
│   ├── setup_openmic.py          # Resumable OpenMIC-2018 downloader & label generator
│   ├── train_openmic_gpu.py      # PyTorch GPU training with mixed precision & SE blocks
│   ├── export_onnx.py            # Automated ONNX export, parity verification & benchmark
│   └── openmic-2018/             # Extracted dataset and cached feature arrays
│
└── docs/
    ├── MASTER_PLAN.md            # Comprehensive project roadmap & milestones
    └── DECISIONS_AND_ARCHITECTURE.md # Architectural decisions log
```

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

Analyzes an uploaded audio file and outputs independent confidence scores for each instrument.

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
  "duration_seconds": 10.0,
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
  "spectrogram_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

---

## Evaluation & Metrics

- **Loss Function:** Binary Cross-Entropy with Pos-Weight adjustment for label imbalance.
- **Evaluation Metric:** Multi-label Macro AUROC (Current checkpoint: **0.8989**).
- **Inference Latency:** **3.11 ms** per 10-second audio track via ONNX Runtime (**4.74x faster** than PyTorch CPU).