"""
02_export_optimization — Экспорт классических методов сегментации
в оптимизированные форматы (ONNX, TensorRT) для ускорения инференса.

Доступные бэкенды:
- ONNX Runtime (CPU/CUDA/TensorRT EP) — ✅ Стабильный, кроссплатформенный
- torch-tensorrt — ✅ Официальный NVIDIA, требует TensorRT 10+
- torch2trt — ⚠️ Опционально, требует сборки из исходников

Пример:
    from optimization_study.02_export_optimization import ONNXOptimizer
    optimizer = ONNXOptimizer(segmenter)
    optimizer.export_method_to_onnx("sobel_edge", "sobel.onnx")
"""

from .utils import (
    get_available_providers,
    get_available_tensorrt_backends,
    format_time,
    format_speedup,
    save_benchmark_results,
)
from .onnx_converter import ONNXOptimizer
from .torch_tensorrt_converter import TorchTRTOptimizer

# Опциональный импорт с защитой
try:
    from .torch2trt_converter import Torch2TRTOptimizer

    TORCH2TRT_AVAILABLE = True
except ImportError:
    TORCH2TRT_AVAILABLE = False

__all__ = [
    "ONNXOptimizer",
    "TorchTRTOptimizer",
    "Torch2TRTOptimizer",
    "get_available_providers",
    "get_available_tensorrt_backends",
    "TORCH2TRT_AVAILABLE",
]

__version__ = "1.0.0"
