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
    from backend.app.utils.preprocess import preprocess_full_audio, audio_to_melspec
except ImportError:
    from app.core.model_loader import get_model
    from app.utils.preprocess import preprocess_full_audio, audio_to_melspec

INSTRUMENTS = [
    'accordion', 'bass', 'cello', 'clarinet', 'cymbals', 'drums', 'flute',
    'guitar', 'mallet_percussion', 'mandolin', 'piano', 'saxophone',
    'synthesizer', 'trombone', 'trumpet', 'ukulele', 'violin', 'voice'
]

DEFAULT_THRESHOLD = 0.5


def format_seconds_label(sec):
    """Format seconds into MM:SS display string."""
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


def merge_active_intervals(time_ranges, active_indices):
    """
    Merges contiguous or adjacent time-window intervals into readable format:
    e.g. ['0:00 - 0:45', '1:15 - 2:30']
    """
    if not active_indices:
        return []
    intervals = []
    start_sec = time_ranges[active_indices[0]]["start"]
    prev_end = time_ranges[active_indices[0]]["end"]

    for idx in active_indices[1:]:
        curr_start = time_ranges[idx]["start"]
        curr_end = time_ranges[idx]["end"]
        # If contiguous or overlapping window (within 0.25s tolerance)
        if curr_start <= prev_end + 0.25:
            prev_end = curr_end
        else:
            intervals.append(f"{format_seconds_label(start_sec)} - {format_seconds_label(prev_end)}")
            start_sec = curr_start
            prev_end = curr_end

    intervals.append(f"{format_seconds_label(start_sec)} - {format_seconds_label(prev_end)}")
    return intervals


def generate_spectrogram_image_base64(spec_2d):
    """Generate a clean base64 PNG of the 128x128 Log-Mel Spectrogram in ~1ms using Pillow."""
    try:
        from PIL import Image
        cmap = matplotlib.colormaps.get_cmap('YlOrRd')
        # Invert row order for origin='lower' acoustic display
        rgba = (cmap(spec_2d[::-1]) * 255).astype(np.uint8)
        img = Image.fromarray(rgba)
        img = img.resize((360, 180), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
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
    Run multi-label inference across an entire audio file (short clips or full songs)
    using vectorized single-pass STFT and batch ONNX Runtime acceleration.
    Returns:
        dict containing predictions, detected instruments, timeline heatmap, and spectrogram.
    """
    model, m_type = get_model()
    if model is None:
        raise RuntimeError("AudioTag AI model is not loaded. Train or export the model first.")

    # 1. High-speed single-pass audio decode & sliding-window extraction
    prep_data = preprocess_full_audio(audio_path)
    batch_tensor = prep_data["batch_tensor"]   # (N, 1, 128, 128) float32
    time_ranges = prep_data["time_ranges"]     # list of dicts
    total_duration = prep_data["total_duration"]
    overview_spec = prep_data["overview_spec"]

    # 2. Render overview spectrogram base64 for frontend UI
    spec_base64 = generate_spectrogram_image_base64(overview_spec)

    # 3. Vectorized Batch Inference
    # ONNX Runtime processes N windows in a single fused SIMD C++ call (~48ms for 70 windows)
    if m_type == "onnx":
        input_name = model.get_inputs()[0].name
        logits = model.run(None, {input_name: batch_tensor})[0]
        all_probs = 1.0 / (1.0 + np.exp(-logits))

    # PyTorch AudioResNet-SE Batch Inference (lazy import)
    elif m_type == "pytorch":
        import torch
        device = next(model.parameters()).device
        x_tensor = torch.tensor(batch_tensor, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            all_probs = torch.sigmoid(logits).cpu().numpy()

    # Keras CNN Inference Fallback
    else:
        k_in = batch_tensor.transpose(0, 2, 3, 1)
        all_probs = model.predict(k_in, verbose=0)

    num_windows = len(time_ranges)

    # 4. Construct Time-Segmented Window Timeline
    timeline = []
    for i in range(num_windows):
        w_preds = {}
        w_detected = []
        for c, inst in enumerate(INSTRUMENTS):
            score = round(float(all_probs[i, c]), 4)
            w_preds[inst] = score
            if score >= threshold:
                w_detected.append(inst)
        w_detected.sort(key=lambda inst: w_preds[inst], reverse=True)
        timeline.append({
            "window_index": i,
            "start": time_ranges[i]["start"],
            "end": time_ranges[i]["end"],
            "display": time_ranges[i]["display"],
            "predictions": w_preds,
            "detected": w_detected
        })

    # 5. Global Aggregation & Timeline Summary
    predictions = {}
    detected = []
    timeline_summary = {}

    for c, inst in enumerate(INSTRUMENTS):
        inst_scores = all_probs[:, c]
        peak = round(float(np.max(inst_scores)), 4)
        mean = round(float(np.mean(inst_scores)), 4)
        active_indices = [i for i, s in enumerate(inst_scores) if s >= threshold]
        presence_pct = round((len(active_indices) / float(num_windows)) * 100.0, 1)

        # Overall presence score: average confidence during active sections (or peak if single window)
        if active_indices:
            active_mean = round(float(np.mean(inst_scores[active_indices])), 4)
            overall_score = active_mean
            detected.append(inst)
        else:
            overall_score = peak

        predictions[inst] = overall_score
        timeline_summary[inst] = {
            "peak": peak,
            "mean": mean,
            "presence_percent": presence_pct,
            "intervals": merge_active_intervals(time_ranges, active_indices)
        }

    # Sort overall detected instruments by confidence descending
    detected.sort(key=lambda inst: predictions[inst], reverse=True)

    return {
        "status": "success",
        "duration_seconds": round(total_duration, 2),
        "engine": m_type,
        "predictions": predictions,
        "detected": detected,
        "threshold": threshold,
        "spectrogram_base64": spec_base64,
        "is_full_song": num_windows > 1,
        "num_windows": num_windows,
        "timeline": timeline,
        "timeline_summary": timeline_summary
    }