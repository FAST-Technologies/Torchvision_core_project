from research.optimization_study.export_optimization.backend_registry import (
    get_registry,
)

registry = get_registry()
registry.print_status()
# ✅ onnx — ONNX Runtime (CPU/CUDA/TensorRT EP)
# ✅ torch_tensorrt — torch-tensorrt (официальный NVIDIA бэкенд)
# ❌ torch2trt — torch2trt (NVIDIA-AI-IOT, требует сборки)
# 🎯 Recommended: onnx
