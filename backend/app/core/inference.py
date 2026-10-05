import io
import base64
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from backend.app.core.model_loader import get_model
    from backend.app.utils.preprocess import preprocess_audio, audio_to_melspec
except ImportError:
    from app.core.model_loader import get_model
    from app.utils.preprocess import preprocess_audio, audio_to_melspec

INSTRUMENTS = [
    'accordion', 'bass', 'cello', 'clarinet', 'cymbals', 'drums', 'flute',
    'guitar', 'mallet_percussion', 'mandolin', 'piano', 'saxophone',
    'synthesizer', 'trombone', 'trumpet', 'ukulele', 'violin', 'voice'
]

DEFAULT_THRESHOLD = 0.5


def generate_spectrogram_image_base64(spec_2d):
    """Generate a clean base64 PNG of the 128x128 Log-Mel Spectrogram."""
    try:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100, facecolor="#111417")
        ax.set_facecolor("#111417")
        ax.imshow(spec_2d, origin="lower", aspect="auto", cmap="YlOrRd")
        ax.axis("off")
        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="#111417")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def predict_instruments(audio_path, threshold=DEFAULT_THRESHOLD):
    """
    Run multi-label inference on an audio file using ONNX Runtime (fastest), PyTorch, or Keras.
    Returns:
        dict containing predictions for all 18 instruments, detected list, and spectrogram image.
    """
    model, m_type = get_model()

    # Preprocess audio to (128, 128) - only reads first 10 seconds
    spec_2d = audio_to_melspec(audio_path)
    if spec_2d is None:
        raise ValueError(f"Failed to process audio file: {audio_path}")

    # Generate spectrogram base64 for frontend display
    spec_base64 = generate_spectrogram_image_base64(spec_2d)

    if model is None:
        raise RuntimeError("AudioTag AI model is not loaded. Train the model first.")

    # 1. ONNX Runtime Inference (ultra-low latency, ~3ms, no PyTorch overhead)
    if m_type == "onnx":
        x_np = spec_2d[np.newaxis, np.newaxis, ...].astype(np.float32)
        input_name = model.get_inputs()[0].name
        logits = model.run(None, {input_name: x_np})[0]
        raw_preds = (1.0 / (1.0 + np.exp(-logits)))[0]

    # 2. PyTorch AudioResNet-SE Inference (lazy import so torch doesn't eat RAM when onnx is used)
    elif m_type == "pytorch":
        import torch
        device = next(model.parameters()).device
        x_tensor = torch.tensor(spec_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            raw_preds = torch.sigmoid(logits)[0].cpu().numpy()

    # 3. Keras CNN Inference Fallback
    else:
        x_np = spec_2d[np.newaxis, ..., np.newaxis]
        raw_preds = model.predict(x_np, verbose=0)[0]

    predictions = {}
    detected = []

    for i, inst in enumerate(INSTRUMENTS):
        score = float(raw_preds[i])
        predictions[inst] = round(score, 4)
        if score >= threshold:
            detected.append(inst)

    # Sort detected instruments by confidence descending
    detected.sort(key=lambda inst: predictions[inst], reverse=True)

    # Audio duration - fast header inspection via soundfile first
    duration = 10.0
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        duration = float(info.duration)
    except Exception:
        try:
            duration = float(librosa.get_duration(path=audio_path))
        except Exception:
            duration = 10.0

    return {
        "status": "success",
        "duration_seconds": round(duration, 2),
        "engine": m_type,
        "predictions": predictions,
        "detected": detected,
        "threshold": threshold,
        "spectrogram_base64": spec_base64
    }