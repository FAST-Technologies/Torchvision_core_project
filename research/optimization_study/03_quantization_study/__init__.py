"""
03_quantization_study — Исследование влияния квантования
на скорость и точность классических методов сегментации.

Цель: Сравнить производительность и точность методов при использовании
разных схем квантования:
- FP32 (эталон) — полная точность
- FP16 (half precision) — половинная точность, ускорение на GPU
- INT8 (static/dynamic) — 8-битное квантование, максимальное ускорение на CPU

Пример:
    >>> from optimization_study.03_quantization_study import QuantizationBenchmark
    >>> benchmark = QuantizationBenchmark(segmenter, calibration_images)
    >>> results = benchmark.run_full_benchmark(["sobel_edge", "otsu_thresholding"])
    >>> benchmark.plot_tradeoff(results, output_dir="./results/")
"""

from .config import (
    QuantizationConfig,
    DEFAULT_CONFIG,
    QUANTIZATION_SCHEMES,
    SUPPORTED_DTYPES,
)
from .utils import (
    get_available_quantization_backends,
    format_speedup,
    format_size_reduction,
    compute_quantization_error,
    safe_quantize_tensor,
    is_method_quantizable,
)
from .quantizer import (
    QuantizedSegmenter,
    QuantizationWrapper,
    CalibrationDataset,
)
from .benchmark import QuantizationBenchmark
from .report_generator import QuantizationReportGenerator
from .visualization import QuantizationVisualizer

__all__ = [
    "QuantizationBenchmark",
    "QuantizedSegmenter",
    "QuantizationWrapper",
    "CalibrationDataset",
    "QuantizationConfig",
    "DEFAULT_CONFIG",
    "QuantizationReportGenerator",
    "QuantizationVisualizer",
    "get_available_quantization_backends",
    "is_method_quantizable",
    "format_speedup",
]

__version__ = "1.0.0"
