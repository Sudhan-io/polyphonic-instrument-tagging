import os
import sys
import time
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.core.model_loader import AudioResNetSE

PT_MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "app", "models", "audiotag_model_v1.pt")
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "app", "models", "audiotag_model_v1.onnx")


def export_and_benchmark():
    print(f"Loading PyTorch checkpoint from: {PT_MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioResNetSE(num_classes=18).to(device)

    checkpoint = torch.load(PT_MODEL_PATH, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Move model to CPU for clean, portable ONNX graph export
    cpu_model = AudioResNetSE(num_classes=18)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.eval()

    dummy_input = torch.randn(1, 1, 128, 128, dtype=torch.float32)

    print(f"Exporting model to ONNX: {ONNX_MODEL_PATH} (opset 17)...")
    torch.onnx.export(
        cpu_model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "spectrogram": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print(f"Export complete. File size: {os.path.getsize(ONNX_MODEL_PATH) / (1024*1024):.2f} MB")

    # Validate with onnx package if available
    try:
        import onnx
        onnx_model = onnx.load(ONNX_MODEL_PATH)
        onnx.checker.check_model(onnx_model)
        print("ONNX graph validation: PASSED")
    except Exception as e:
        print(f"ONNX checker: {e}")

    # Validate numerical equivalence with ONNX Runtime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

        # Test dummy input
        test_np = np.random.randn(1, 1, 128, 128).astype(np.float32)

        # PyTorch prediction
        with torch.no_grad():
            pt_logits = cpu_model(torch.from_numpy(test_np)).numpy()
            pt_probs = 1.0 / (1.0 + np.exp(-pt_logits))

        # ONNX Runtime prediction
        ort_inputs = {session.get_inputs()[0].name: test_np}
        ort_logits = session.run(None, ort_inputs)[0]
        ort_probs = 1.0 / (1.0 + np.exp(-ort_logits))

        max_diff = np.max(np.abs(pt_probs - ort_probs))
        print(f"Numerical verification (max probability delta): {max_diff:.8f}")
        assert max_diff < 1e-4, f"Output divergence too large: {max_diff}"
        print("Verification: PyTorch and ONNX predictions are numerically identical.")

        # Latency benchmark (50 runs)
        print("\nBenchmarking latency over 50 iterations (CPU):")
        # PyTorch CPU
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                _ = cpu_model(torch.from_numpy(test_np))
        pt_time = (time.perf_counter() - start) / 50 * 1000

        # ONNX Runtime CPU
        start = time.perf_counter()
        for _ in range(50):
            _ = session.run(None, ort_inputs)
        ort_time = (time.perf_counter() - start) / 50 * 1000

        print(f"PyTorch CPU average latency:      {pt_time:.2f} ms")
        print(f"ONNX Runtime CPU average latency:  {ort_time:.2f} ms")
        print(f"Speedup:                          {pt_time / ort_time:.2f}x")

    except Exception as e:
        print(f"ONNX Runtime validation error: {e}")


if __name__ == "__main__":
    export_and_benchmark()
