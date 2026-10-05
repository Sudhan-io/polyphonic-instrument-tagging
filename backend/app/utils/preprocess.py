import os
import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION = 10.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 220,500 samples (10.0s @ 22.05kHz)
N_MELS = 128
IMG_W = 128
FMAX = 8000


def load_audio_waveform(file_path):
    """
    High-speed audio waveform loader supporting .mp3, .wav, .ogg, .flac, .m4a.
    CRITICAL OPTIMIZATION: Only decodes the first 10.0 seconds directly from the file.
    Never decodes or resamples full 3-5 minute songs, preventing CPU/memory timeouts.
    """
    # 1. Fast path: load first 10 seconds directly with librosa at target sample rate
    try:
        y, _ = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=DURATION
        )
        return y
    except Exception:
        pass

    # 2. Secondary path: torchaudio with frame cap
    try:
        import torchaudio
        # Read at most ~12 seconds of frames to avoid loading large files
        info = torchaudio.info(file_path)
        max_frames = int(info.sample_rate * (DURATION + 1.0))
        waveform, sr = torchaudio.load(file_path, num_frames=max_frames)

        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        elif waveform.ndim > 1:
            waveform = waveform.squeeze(0)
        y = waveform.numpy()

        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        return y[:N_SAMPLES]
    except Exception:
        pass

    # 3. Final fallback: standard load with duration cap
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    return y


def audio_to_melspec(file_path_or_array, sr=None):
    """
    Convert audio file path or numpy waveform to a normalized Log-Mel spectrogram.
    Returns:
        mel_norm: numpy array of shape (128, 128), values in [0, 1]
    """
    if isinstance(file_path_or_array, str):
        y = load_audio_waveform(file_path_or_array)
    else:
        y = file_path_or_array
        if sr is not None and sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Ensure 1D mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Pad or trim to exactly 10 seconds (220,500 samples)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

    # Log-Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Fix time-axis width to IMG_W columns
    mel_db = librosa.util.fix_length(mel_db, size=IMG_W, axis=1)

    # Normalize to [0, 1]
    min_val = mel_db.min()
    max_val = mel_db.max()
    mel_norm = (mel_db - min_val) / (max_val - min_val + 1e-8)

    return mel_norm.astype(np.float32)


def preprocess_audio(file_path):
    """
    Preprocess an audio file into model input format: shape (1, 128, 128, 1).
    """
    spec = audio_to_melspec(file_path)
    return spec[np.newaxis, ..., np.newaxis]


def preprocess_audio_array(y, sr=SAMPLE_RATE):
    """
    Preprocess a raw waveform array into model input format: shape (1, 128, 128, 1).
    """
    spec = audio_to_melspec(y, sr=sr)
    return spec[np.newaxis, ..., np.newaxis]