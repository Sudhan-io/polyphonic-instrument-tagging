import os
import logging

logger = logging.getLogger("audiotag.model_loader")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_MODEL_PATH    = os.path.join(BASE_DIR, "models", "audiotag_model_v1.onnx")
PYTORCH_MODEL_PATH = os.path.join(BASE_DIR, "models", "audiotag_model_v1.pt")
KERAS_MODEL_PATH   = os.path.join(BASE_DIR, "models", "audiotag_model_v1.keras")

audiotag_model = None
model_type = None  # 'onnx', 'pytorch', or 'keras'


# ==============================================================================
# Lazy-loaded PyTorch AudioResNet-SE Architecture
# Importing torch consumes ~350MB of RAM. By defining this lazily, ONNX Runtime
# deployments (e.g. Render / Cloud containers) run in <100MB of RAM without OOM.
# ==============================================================================
def get_audio_resnet_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SqueezeExcitation(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.fc1 = nn.Linear(channels, max(channels // reduction, 8))
            self.fc2 = nn.Linear(max(channels // reduction, 8), channels)

        def forward(self, x):
            b, c, _, _ = x.shape
            w = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
            w = F.gelu(self.fc1(w))
            w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
            return x * w

    class SEBasicBlock(nn.Module):
        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.se = SqueezeExcitation(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
                )

        def forward(self, x):
            out = F.gelu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            out += self.shortcut(x)
            out = F.gelu(out)
            return out

    class AudioResNetSE(nn.Module):
        def __init__(self, num_classes=18):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.MaxPool2d(2, 2)
            )
            self.layer1 = nn.Sequential(
                SEBasicBlock(32, 64, stride=2),
                SEBasicBlock(64, 64, stride=1)
            )
            self.layer2 = nn.Sequential(
                SEBasicBlock(64, 128, stride=2),
                SEBasicBlock(128, 128, stride=1)
            )
            self.layer3 = nn.Sequential(
                SEBasicBlock(128, 256, stride=2),
                SEBasicBlock(256, 256, stride=1)
            )
            self.head = nn.Sequential(
                nn.Linear(256 * 2, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.35),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            out = self.stem(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            gap = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
            gmp = F.adaptive_max_pool2d(out, (1, 1)).view(out.size(0), -1)
            feat = torch.cat([gap, gmp], dim=1)
            logits = self.head(feat)
            return logits

    return AudioResNetSE


def AudioResNetSE(*args, **kwargs):
    """Factory proxy allowing 'from model_loader import AudioResNetSE' to work transparently."""
    cls = get_audio_resnet_class()
    return cls(*args, **kwargs)


def _load_onnx():
    """Attempt loading the optimized ONNX Runtime model."""
    if not os.path.exists(ONNX_MODEL_PATH):
        logger.warning("ONNX model file not found at: %s", ONNX_MODEL_PATH)
        return None
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        logger.info("Loading ONNX Runtime session from %s (providers=%s)", ONNX_MODEL_PATH, providers)
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        active_providers = session.get_providers()
        logger.info("ONNX Runtime session active (providers=%s)", active_providers)
        return session
    except Exception as e:
        logger.warning("Failed to initialize ONNX Runtime session: %s", e)
        return None


def _load_pytorch():
    """Attempt loading the native PyTorch GPU/CPU model (lazy)."""
    if not os.path.exists(PYTORCH_MODEL_PATH):
        return None
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading PyTorch AudioResNet-SE from %s onto %s", PYTORCH_MODEL_PATH, device)
        cls = get_audio_resnet_class()
        model = cls(num_classes=18).to(device)
        checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        logger.info("PyTorch AudioResNet-SE loaded successfully (device=%s)", device)
        return model
    except Exception as e:
        logger.warning("Failed to load PyTorch model: %s", e)
        return None


def _load_keras():
    """Attempt loading legacy Keras fallback."""
    if not os.path.exists(KERAS_MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        logger.info("Loading Keras fallback model from %s", KERAS_MODEL_PATH)
        model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        logger.warning("Keras fallback model loaded (degraded CPU-only performance)")
        return model
    except Exception as e:
        logger.error("Failed to load Keras fallback model: %s", e)
        return None


def load_models(force_reload=False):
    """
    Load AudioTag AI model hierarchy:
      1. ONNX Runtime (fastest kernel-fused graph, ~3ms latency, ~40MB RAM)
      2. PyTorch GPU AudioResNet-SE (0.8989 AUROC)
      3. Keras Fallback (degraded legacy)

    Set AUDIOTAG_ENGINE="pytorch" to force PyTorch over ONNX.
    """
    global audiotag_model, model_type

    if audiotag_model is not None and not force_reload:
        return audiotag_model, model_type

    preferred_engine = os.environ.get("AUDIOTAG_ENGINE", "onnx").lower()

    if preferred_engine == "pytorch":
        pt_model = _load_pytorch()
        if pt_model is not None:
            audiotag_model = pt_model
            model_type = "pytorch"
            return audiotag_model, model_type

    # 1. Try ONNX Runtime (default preferred, ultra-low memory)
    onnx_session = _load_onnx()
    if onnx_session is not None:
        audiotag_model = onnx_session
        model_type = "onnx"
        return audiotag_model, model_type

    # 2. Try PyTorch
    pt_model = _load_pytorch()
    if pt_model is not None:
        audiotag_model = pt_model
        model_type = "pytorch"
        return audiotag_model, model_type

    # 3. Try Keras
    keras_model = _load_keras()
    if keras_model is not None:
        audiotag_model = keras_model
        model_type = "keras"
        return audiotag_model, model_type

    logger.error("No trained model found at %s or %s", ONNX_MODEL_PATH, PYTORCH_MODEL_PATH)
    return None, None


def get_model():
    """Retrieve loaded model or attempt to load if uninitialized."""
    global audiotag_model, model_type
    if audiotag_model is None:
        load_models()
    return audiotag_model, model_type