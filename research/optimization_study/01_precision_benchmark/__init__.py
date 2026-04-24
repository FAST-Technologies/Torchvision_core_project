"""
01_precision_benchmark — Исследование влияния точности вычислений
на скорость и качество сегментации изображений.

Цель: Сравнить производительность и точность классических методов
сегментации при использовании разных типов данных:
- torch.float32 (FP32) — эталонная точность
- torch.float16 (FP16) — половинная точность, ускорение на GPU
- torch.bfloat16 (BF16) — Brain Float, баланс точности/скорости
- (опционально) torch.int8 — квантование для инференса

Пример:
    >>> from optimization_study.01_precision_benchmark import PrecisionBenchmark
    >>> benchmark = PrecisionBenchmark(segmenter, image)
    >>> df = benchmark.run_full_benchmark(["sobel_edge", "otsu_thresholding"])
    >>> benchmark.plot_results(df, output_dir="./results/")
"""

from .config import PrecisionConfig, DEFAULT_CONFIG
from .utils import (
    get_available_dtypes,
    format_time,
    format_speedup,
    compute_pixel_agreement,
    compute_mse,
    safe_autocast_context,
)
from .precision_benchmark import PrecisionBenchmark
from .report_generator import ReportGenerator
from .visualization import PrecisionVisualizer

__all__ = [
    "PrecisionBenchmark",
    "PrecisionConfig",
    "DEFAULT_CONFIG",
    "ReportGenerator",
    "PrecisionVisualizer",
    "get_available_dtypes",
    "format_time",
    "format_speedup",
]

__version__ = "1.0.0"
