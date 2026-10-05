# AudioTag AI — Architectural Evolution, Design Decisions & Technical Rationales

> **Document Scope:** A complete, comprehensive technical chronicle explaining **what** was built, **why** it was built that way, **how** it works under the hood, and **why** specific technical choices were made across datasets, architectures, frameworks, inference runtimes, security hardening, and user interfaces.

---

## Table of Contents
1. [Executive Summary & Problem Definition](#1-executive-summary--problem-definition)
2. [Dataset Evolution: NSynth vs. OpenMIC-2018](#2-dataset-evolution-nsynth-vs-openmic-2018)
3. [The Multi-Label Paradigm Shift (Softmax vs. Sigmoid)](#3-the-multi-label-paradigm-shift-softmax-vs-sigmoid)
4. [Hardware Discovery: The Windows GPU Bottleneck & PyTorch Migration](#4-hardware-discovery-the-windows-gpu-bottleneck--pytorch-migration)
5. [The "Voice" Overfitting Problem & Class Imbalance Resolution](#5-the-voice-overfitting-problem--class-imbalance-resolution)
6. [Neural Architecture: AudioResNet with Squeeze-and-Excitation (SE) Attention](#6-neural-architecture-audioresnet-with-squeeze-and-excitation-se-attention)
7. [Inference Optimization: PyTorch to ONNX Runtime Engine](#7-inference-optimization-pytorch-to-onnx-runtime-engine)
8. [Interactive Acoustic Studio: Dossier, Timbre Synthesizer & Sorting](#8-interactive-acoustic-studio-dossier-timbre-synthesizer--sorting)
9. [Frontend Evolution: Streamlit to Native Editorial Web Application](#9-frontend-evolution-streamlit-to-native-editorial-web-application)
10. [Security & Production Hardening Audit](#10-security--production-hardening-audit)
11. [Summary of Key Technical Decisions](#11-summary-of-key-technical-decisions)

---

## 1. Executive Summary & Problem Definition

### The Goal
Build a deep learning system capable of listening to polyphonic, real-world music recordings (`.wav`, `.mp3`, `.ogg`, `.flac`) and **simultaneously isolating and tagging up to 18 instruments** playing in the mix.

### The Challenge
Most traditional machine learning tutorials treat instrument classification as a single-label, mutually exclusive problem: *"Is this track a piano OR a guitar?"* In real-world music, multiple instruments overlap in both the time and frequency domains. Drums, bass, guitars, synthesizers, and vocals share acoustic energy across overlapping harmonic spectra. Identifying them requires a **multi-label, multi-output acoustic intelligence engine**.

---

## 2. Dataset Evolution: NSynth vs. OpenMIC-2018

### Legacy State: NSynth (Milestones 1–4)
* **What it was:** Google Magenta's NSynth dataset containing isolated, single-note synthesized tones.
* **Why it failed for real music:**
  - NSynth samples are monophonic: one note played in isolation with zero background noise.
  - Models trained on NSynth completely broke down when exposed to real music with overlapping drums, vocals, and reverbs.
  - NSynth only covered 8 generic families (`brass`, `flute`, `guitar`, `keyboard`, `mallet`, `reed`, `string`, `vocal`).

### Current State: OpenMIC-2018
* **Source:** Free Music Archive (FMA) curated by Eric Humphrey et al.
* **Size:** 20,000 diverse, real-world 10-second audio excerpts.
* **Acoustic Characteristics:** Polyphonic, noisy, real recordings with studio mixes, crowd noise, electronic distortion, and live performances.
* **18 Target Classes:**
  `accordion`, `bass`, `cello`, `clarinet`, `cymbals`, `drums`, `flute`, `guitar`, `mallet_percussion`, `mandolin`, `piano`, `saxophone`, `synthesizer`, `trombone`, `trumpet`, `ukulele`, `violin`, `voice`.
* **Why chosen:** It is the premier academic benchmark for polyphonic music tagging.

---

## 3. The Multi-Label Paradigm Shift (Softmax vs. Sigmoid)

| Metric | Single-Label (Softmax) | Multi-Label (Sigmoid) — *AudioTag AI* |
|---|---|---|
| **Mathematical Formula** | $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | $\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$ |
| **Sum of Outputs** | Strictly equal to $1.0$ ($100\%$) | Can sum to any value ($0.0 \to 18.0$) |
| **Inter-Class Behavior** | Classes compete against each other | Each class is an independent binary question |
| **Loss Function** | `CategoricalCrossentropy` | `BinaryCrossentropy` / `BCEWithLogitsLoss` |
| **Musical Reality** | Assumes only one instrument exists | Reflects reality: Drums, Bass, and Vocals play together |

**Why Sigmoid:** By using 18 independent sigmoid activation units, the presence of Drums ($0.95$) does not penalize or reduce the score of Guitar ($0.88$) or Voice ($0.74$).

---

## 4. Hardware Discovery: The Windows GPU Bottleneck & PyTorch Migration

### The Hardware Audit
An audit of the host system revealed:
* **GPU:** **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (2,048 CUDA Cores, Ampere Architecture, Tensor Cores).
* **CPU:** **12th Gen Intel Core i5-12450HX** (8 Cores, 12 Threads).
* **RAM:** **16 GB**.

### The Windows TensorFlow Dead-End
* Starting with version 2.11, Google **deprecated native Windows GPU support** in TensorFlow. Running `pip install tensorflow` on Windows forces all computations onto the CPU.
* In our tests on the 12th Gen Intel CPU, TensorFlow took **~8 seconds per batch**, equating to **3 to 5 hours** for 20,000 tracks.

### The PyTorch Solution
* PyTorch (`v2.5.1+cu121`) has first-class native CUDA acceleration on Windows.
* Running on the RTX 3050 with Automatic Mixed Precision (`torch.amp.autocast`) cuts training time from **5 hours to ~10–15 minutes** (a **25x speedup**) while unlocking modern SOTA deep learning architectures.

---

## 5. The "Voice" Overfitting Problem & Class Imbalance Resolution

### Observation
Initial tests on instrumental and movie soundtrack songs consistently returned `"voice"` as the dominant prediction, even when no vocals were present.

### Root Causes
1. **Sanity-Check Checkpoint:** The initial model on disk was trained on only 10 dummy samples for 1 epoch. In those 10 tracks, only `voice` had positive labels, so the network learned to only fire the `voice` neuron.
2. **Dataset Imbalance:** In general music collections:
   - Common instruments (Voice, Drums, Guitar) appear in **30–45%** of recordings.
   - Rare instruments (Violin, Mandolin, Flute, Trombone) appear in only **4–8%** of recordings.
   - A naive model minimizes overall loss by always predicting common classes and ignoring rare classes.

### The Fix: Dynamic `pos_weight` Balancing
In `torch.nn.BCEWithLogitsLoss`, we calculate positive class weights:
$$\text{pos\_weight}_c = \frac{N - N_c}{N_c}$$
Where $N$ is total tracks ($20,000$) and $N_c$ is positive tracks for class $c$.
- For common classes (Voice, Drums), $\text{pos\_weight} \approx 2.0$.
- For rare classes (Violin, Cello, Flute), $\text{pos\_weight} \approx 12.0 - 20.0$.
When a rare instrument appears, the model receives a massive loss penalty for missing it, forcing the feature extractor to learn distinctive violin bow attacks, woodwind transients, and brass timbres.

---

## 6. Neural Architecture: AudioResNet with Squeeze-and-Excitation (SE) Attention

```
Input Audio (.wav / .mp3 / .ogg)
   │
   ▼
Resample to 22,050 Hz ──▶ Pad/Trim to 10.0s (220,500 samples)
   │
   ▼
128-Mel Filterbank Spectrogram (fmax = 8,000 Hz, hop_length = 512)
   │
   ▼
Power-to-dB Conversion ──▶ Normalization to [0, 1] ──▶ Shape: (Batch, 1, 128, 128)
   │
   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        AudioResNet-SE Engine                           │
│                                                                        │
│  [Stem: Conv2D(32, 5x5, s=1, p=2) + BatchNorm + GELU + MaxPool2D(2x2)] │
│                                                                        │
│  [Stage 1: 2× SEBasicBlock(64 dims, s=2, 1) + SE Channel Attention]    │
│                                                                        │
│  [Stage 2: 2× SEBasicBlock(128 dims, s=2, 1) + SE Channel Attention]   │
│                                                                        │
│  [Stage 3: 2× SEBasicBlock(256 dims, s=2, 1) + SE Channel Attention]   │
│                                                                        │
│  [Dual Pooling: GlobalAveragePooling2D ⊕ GlobalMaxPooling2D (512 dims)] │
│                                                                        │
│  [Classifier Head: Linear(512 → 256) ──▶ BatchNorm1d ──▶ GELU          │
│                    ──▶ Dropout(0.35) ──▶ Linear(256 → 18)]             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
       18 Independent Logits ──▶ Sigmoid Probability [0.0 - 1.0]
```

### Architectural Highlights
1. **Squeeze-and-Excitation (SE) Channel Attention:**
   Compresses 2D spatial feature maps via global average pooling, feeds them into a bottleneck MLP ($\text{Linear} \to \text{GELU} \to \text{Linear} \to \text{Sigmoid}$), and rescales channel weights dynamically. This allows the network to amplify frequency bands specific to active instruments.
2. **Dual Pooling (GAP $\oplus$ GMP):**
   Standard Global Average Pooling washes out sudden transients (drum snare hits, cymbal crashes). Global Max Pooling preserves peak acoustic spikes, while Average Pooling captures sustained instruments (violins, vocals, synth pads). Concatenating both yields a richer representation.
3. **SpecAugment Regularization:**
   Randomly masks horizontal frequency strips and vertical time bars during training, preventing overfitting to specific recording acoustics.

---

## 7. Inference Optimization: PyTorch to ONNX Runtime Engine

### Why ONNX?
While PyTorch is exceptional for model training and research, serving a raw `.pt` checkpoint in production incurs notable overhead:
1. **Framework Weight:** PyTorch requires loading large Python libraries, autograd memory structures, and JIT graph interpreters (~2.5 GB disk footprint).
2. **Unfused Kernel Passes:** PyTorch executes Conv2D, BatchNorm, GELU, and Squeeze-and-Excitation steps as separate, isolated GPU/CPU operations, causing repetitive memory reads and writes.
3. **High Latency on Edge / CPU:** Machines without high-end dedicated GPUs experience noticeable lag when running PyTorch CPU inference.

### The ONNX Solution
Using `torch.onnx.export` with Opset 17 and dynamic batching, `audiotag_model_v1.pt` was converted into `audiotag_model_v1.onnx` (`11.17 MB`).

**Key Architectural Optimizations in ONNX Runtime:**
- **Operator Fusion:** Merges adjacent operations (e.g. `Conv + BatchNorm + GELU` and linear projections) into fused execution kernels.
- **Constant Folding:** Pre-computes static weights, biases, and normalization parameters at export time.
- **Hardware Acceleration:** Seamlessly binds to `CUDAExecutionProvider` or optimized SIMD CPU execution (`CPUExecutionProvider` with AVX-512 / AVX2).

### Benchmark Comparison (50 Iterations)

| Metric | Native PyTorch | ONNX Runtime | Improvement |
|---|---|---|---|
| **Average Latency** | 14.74 ms | **3.11 ms** | **4.74x faster** |
| **Numerical Delta** | Baseline | **0.00000000** | **Exact numerical equivalence** |
| **Output File Size** | 11.74 MB (`.pt`) | **11.17 MB** (`.onnx`) | **5% smaller footprint** |
| **Execution Providers** | CUDA / CPU | `CUDAExecutionProvider`, `CPUExecutionProvider` | Dynamic cross-platform fallback |

### Dynamic Engine Hierarchy in `model_loader.py`
The production loader automatically manages engines:
1. **Tier 1 (Primary):** `audiotag_model_v1.onnx` via `onnxruntime` (ultra-low ~3ms latency).
2. **Tier 2 (Fallback / Direct):** `audiotag_model_v1.pt` via PyTorch CUDA/CPU (activated if ONNX is missing or if `AUDIOTAG_ENGINE="pytorch"` is set).
3. **Tier 3 (Degraded Fallback):** `audiotag_model_v1.keras` (legacy CPU-only).

---

## 8. Interactive Acoustic Studio: Dossier, Timbre Synthesizer & Sorting

### 1. Instrument Dossier Modal
Clicking on any instrument card opens an editorial intelligence modal that delivers a 3-layer sonic profile:
- **Everyday Sound Metaphors:** Relates abstract musical timbres to recognizable daily experiences (e.g., Flute to *“a sharp winter breeze whistling through an open window”*; Cello to *“a deep, warm human speaking voice rich with resonance”*).
- **Studio Mixing Fingerprint:** Professional audio engineering guidelines detailing fundamental frequencies, key harmonics, and standard EQ/compression techniques.
- **Iconic Listening Cues:** Notable songs and reference tracks where the instrument stands out distinctly in the mix.

### 2. Web Audio API Timbre Synthesizer
Rather than streaming heavy audio samples over the network, the client synthesizes instrument timbres natively in real time using the browser's Web Audio API:
- Configures custom harmonic oscillators (sine, triangle, sawtooth, square) at characteristic fundamental frequencies.
- Applies realistic Attack-Decay-Sustain-Release (ADSR) amplitude envelopes tailored to each instrument's acoustic dynamics (e.g., sharp percussive transients for drums vs. smooth exponential swells for flutes and strings).
- Zero server bandwidth; runs entirely on the client.

### 3. Dual-Mode Taxonomy Sorting
The taxonomy toolbar enables instant client-side toggling:
- **Alphabetical Mode:** Groups the 18 instruments alphabetically for standard reference.
- **Detected Probability Mode:** Dynamically re-orders instruments from highest confidence to lowest based on the current audio analysis, applying proportional opacity and color intensity styling.

---

## 9. Frontend Evolution: Streamlit to Native Editorial Web Application

### Why We Abandoned Streamlit
* **Streamlit Limitations:**
  - Re-executes the entire Python script on every user interaction (slider movement, button click).
  - High memory footprint and slow rendering over websockets.
  - Rigid, opinionated widget layouts that resist custom styling.
* **The New Native Web Architecture:**
  - Built with **FastAPI**, **Jinja2 templates**, and **Vanilla JavaScript**.
  - **Zero Server Roundtrips for UI updates:** When the user moves the detection threshold slider ($0.10 \to 0.90$), vanilla JS updates the active/inactive cards and counts instantly on the DOM in $0$ milliseconds.
  - **Editorial Design System:**
    - High-end editorial styling with `Georgia, serif` headlines (*"Music analyzed, layer by layer."*).
    - Warm paper background (`--paper: #f7f8f5;`), dark ink typography (`--ink: #18201d;`), and vibrant orange accents (`--coral: #ff6b00;`).
    - **Zero emojis**: Professional SVG icons and clean geometric badges throughout.
    - Integrated HTML5 audio preview, real-time Log-Mel spectrogram visualization, and JSON download.

---

## 10. Security & Production Hardening Audit

In preparation for production deployment, a full security and hygiene audit was conducted:

1. **CORS Policy Hardening:**
   - **Problem:** Wildcard origins (`allow_origins=["*"]`) combined with `allow_credentials=True` presented a CSRF attack surface and violated modern Fetch security specifications.
   - **Fix:** Restricted default origins to `http://localhost:8000` and `http://127.0.0.1:8000` with `allow_credentials=False`. Configurable in production via the `AUDIOTAG_ALLOWED_ORIGINS` environment variable.
2. **Payload Protection:**
   - **Upload File Cap:** Implemented a strict 50 MB hard limit on incoming audio files (`HTTP 413 Payload Too Large`), preventing denial-of-service memory exhaustion.
   - **MIME & Media Validation:** Enforced checks verifying that incoming payloads carry `audio/*` or `video/*` media types before allocating temporary disk buffers (`HTTP 415 Unsupported Media Type`).
3. **Structured Observability:**
   - Replaced unformatted `print()` statements across backend services with standard, timestamped `logging.getLogger` structured logging.
4. **Legacy Artifact Purge:**
   - Permanently deleted obsolete NSynth models (`instrunet_model_v3.keras`, `instrunet_condition.keras`).
   - Removed the raw OpenMIC dataset archive (`openmic-2018-v1.0.0.tgz`), reclaiming **2.62 GB** of disk space since clips and features are fully cached.
   - Removed legacy `streamlit_app_legacy.py` and stale CPU training scripts.
   - **Total Storage Reclaimed:** **~2.74 GB**.

---

## 11. Cloud Deployment, Free-Tier Resource Constraints & Audio Decoding Optimization

### The Cloud Deployment Architecture
To make AudioTag AI publicly accessible without local hosting requirements, the system was packaged for production deployment:
- **Repository Isolation:** Relocated from the legacy repository to a dedicated, professional repository: `Sudhan-io/polyphonic-instrument-tagging`.
- **Production Containerization ([Dockerfile](file:///d:/PROJECTS/AudioTag-AI/Dockerfile)):**
  - Base Image: `python:3.10-slim`
  - System Dependencies: Installs native Linux `ffmpeg` and `libsndfile1` packages required for decoding `.mp3`, `.ogg`, and `.flac` audio.
  - Process Execution: Uses `uvicorn` with dynamic environment port binding (`PORT=${PORT:-8000}`).
- **Model Inclusion in Version Control:**
  - Standard training checkpoints (`.pt` at 11.7MB and `.keras` at 80MB) remain `.gitignore`d.
  - The optimized ONNX graph (`audiotag_model_v1.onnx`, 11.17 MB) is explicitly tracked in Git, ensuring cloud build environments (Render, Railway, Fly.io) have the model immediately available without external S3/GCS downloads.
- **Public Service:** Deployed as a web service on Render at:
  `https://polyphonic-instrument-tagging.onrender.com/`

---

### Diagnosing the Cloud Free-Tier Bottleneck on Full Songs

When testing the live public deployment with a full 3.5-minute song (`End of Beginning - Djo Edit Audio.mp3`), the web interface appeared stuck in an indefinite "Analyzing Acoustic Mix..." state.

#### Root Cause 1: Unbounded Audio Resampling
In early revisions of `backend/app/utils/preprocess.py`, the waveform loading logic was:
```python
# Naive approach:
y, sr = librosa.load(file_path, sr=22050, mono=True)
if len(y) < N_SAMPLES:
    y = np.pad(y, (0, N_SAMPLES - len(y)))
else:
    y = y[:N_SAMPLES]  # Discard everything after 10.0 seconds
```
- A 3.5-minute song at 44.1 kHz stereo contains over **18 million audio samples**.
- `librosa.load` without a `duration` parameter decoded the entire file and executed high-order sinc interpolation (`librosa.resample`) across all 18 million samples.
- On a developer workstation (8-core Intel i5-12450HX), this took 1.5 seconds.
- On a cloud **Free Tier instance (0.1 fractional CPU core)**, computing an FFT sinc resample across 18 million samples required **over 3 minutes of 100% CPU lockup**.
- Crucially, the code was immediately slicing `y[:N_SAMPLES]` (the first 10 seconds)—meaning 95% of the heavy mathematical computation was performed on audio that was discarded immediately afterwards.

#### Root Cause 2: PyTorch Top-Level Memory Allocation
- The Render free tier provides a hard ceiling of **512 MB of RAM**.
- `import torch` at the module top level immediately allocates **~350 MB to 400 MB** of C++ runtime memory upon Python process launch.
- Combined with Uvicorn, FastAPI, Librosa, and NumPy, the container hovered near 480 MB, leaving virtually zero headroom for buffer allocations during audio decoding.

---

### The Two-Fold Optimization Fix (`commit 9cb9106`)

1. **Direct 10.0s Hardware Decode Cap ([preprocess.py](file:///d:/PROJECTS/AudioTag-AI/backend/app/utils/preprocess.py)):**
   - Configured the audio loader to read strictly the first 10.0 seconds directly from the media container:
     ```python
     y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
     ```
   - Skips reading the remaining 95% of the file entirely.
   - **Result:** Preprocessing time for full songs dropped from **3+ minutes down to 25 milliseconds** (a **7,000x speedup** on long audio files).

2. **Lazy-Loaded PyTorch Factory ([model_loader.py](file:///d:/PROJECTS/AudioTag-AI/backend/app/core/model_loader.py)):**
   - Wrapped the PyTorch `AudioResNetSE` class definition inside a lazy factory function `get_audio_resnet_class()`.
   - When the active engine is ONNX Runtime (`AUDIOTAG_ENGINE=onnx`), PyTorch is **never imported into RAM**.
   - **Result:** The active cloud container's memory footprint dropped from **~450 MB down to ~70 MB**, freeing up over 85% of memory and completely preventing out-of-memory container crashes.

3. **Header-Only Duration Extraction ([inference.py](file:///d:/PROJECTS/AudioTag-AI/backend/app/core/inference.py)):**
   - Replaced full-audio duration scanning with `soundfile.info(audio_path).duration`, reading container header metadata in 1 ms without touching raw audio streams.

---

### Understanding the 10-Second Window vs. True Full-Song Analysis

A critical architectural distinction must be made regarding the 10-second processing window:
- **The Model's Native Input Shape:** The neural network (`AudioResNet-SE`) was trained on OpenMIC-2018 benchmark tracks, which are standardized to 10-second excerpts. The input tensor is mathematically fixed at `(Batch, 1, 128, 128)` (128 Mel bands across 128 time frames).
- **No Model Degradation Occurred:** Restricting `librosa.load` to 10.0 seconds did not degrade model intelligence or accuracy; the previous code was already taking the first 10 seconds anyway (`y[:N_SAMPLES]`), but was wasting CPU power decoding the entire song beforehand.

#### The Roadmap: Time-Segmented Sliding Window Analysis
To analyze an entire 3–5 minute song comprehensively rather than inspecting a single 10-second excerpt, the system will implement **Time-Segmented Sliding Window Analysis**:
1. Slices the audio into sequential 10-second intervals (`[0:00-0:10]`, `[0:10-0:20]`, `[0:20-0:30]`, ...).
2. Executes batch inference through the ONNX Runtime graph (processing twenty 10s slices in **~60 ms total**).
3. Constructs an **Interactive Instrumentation Timeline Map**, displaying exactly when instruments enter and exit across the entire track (e.g. Drums entering at 0:15, Guitar solo at 1:40, Vocals stopping at 2:45).
4. Computes both global song-level averages and peak activation scores.

---

## 12. Summary of Key Technical Decisions

| Category | Selected Choice | Rejected Alternative | Key Technical Rationale |
|---|---|---|---|
| **Training Framework** | PyTorch 2.5 + CUDA 12.1 | TensorFlow 2.15 (CPU) | **25x faster training** on RTX 3050 Laptop GPU (12 mins vs 4 hours). |
| **Production Runtime** | **ONNX Runtime (Opset 17)** | Raw PyTorch Runtime | **4.74x lower latency (3.11 ms)**, operator fusion, zero Python overhead. |
| **Model Architecture** | AudioResNet-SE | Plain 4-Block CNN | Residual connections + SE Channel Attention isolate overlapping polyphonic timbres. |
| **Loss Function** | `BCEWithLogitsLoss` + `pos_weight` | Softmax / Unweighted BCE | Eliminates "voice" bias; forces network to learn rare acoustic classes (violin, flute). |
| **Pooling Strategy** | Dual GAP + GMP | GAP only | Preserves both sustained resonant bodies and sharp transient attack spikes. |
| **Web Client** | Native FastAPI + HTML5/CSS/JS | Streamlit | Instant DOM updates, zero emojis, editorial typography. |
| **Audio Synthesis** | Web Audio API (In-Browser) | Audio Sample Streaming | Zero network bandwidth; synthetic acoustic envelopes rendered client-side. |
| **API Security** | Restricted CORS + 50MB Cap | Wildcard CORS + Uncapped | Prevents CSRF vulnerability and DoS memory exhaustion. |
| **Audio Loading** | Capped 10s Direct Stream | Full-Song Resampling | Prevents 3-minute CPU lockups on 0.1 core cloud tiers for multi-minute songs. |
| **Memory Architecture**| Lazy PyTorch Import | Eager Top-Level Import | Reduces cloud container memory from ~450 MB to ~70 MB, preventing OOM. |

