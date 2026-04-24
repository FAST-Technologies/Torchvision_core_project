"""
04_kernel_fusion — Исследование векторизации и объединения операций (Kernel Fusion)
для ускорения классических методов сегментации.

Цель: Уменьшение накладных расходов через:
- Fusion операций (объединение нескольких kernel'ов в один)
- Векторизацию вычислений (SIMD/SIMT)
- Custom CUDA kernels для критичных участков
- Graph-level оптимизации через torch.compile()

Пример:
    >>> from optimization_study.04_kernel_fusion import FusionOptimizer
    >>> optimizer = FusionOptimizer(segmenter)
    >>> fused_func = optimizer.fuse_method("sauvola_thresholding")
    >>> result = fused_func(image_tensor)
"""

from .config import FusionConfig, DEFAULT_CONFIG, FUSION_STRATEGIES
from .utils import (
    get_fusion_capabilities,
    is_vectorizable,
    estimate_kernel_launch_overhead,
    format_speedup,
    analyze_graph_complexity,
)
from .fusion_optimizer import FusionOptimizer, FusedOperation
from .graph_profiler import GraphProfiler, OperationNode
from .custom_kernels import CustomKernelRegistry, register_custom_kernel

__all__ = [
    "FusionOptimizer",
    "FusedOperation",
    "GraphProfiler",
    "OperationNode",
    "CustomKernelRegistry",
    "register_custom_kernel",
    "FusionConfig",
    "DEFAULT_CONFIG",
    "FUSION_STRATEGIES",
    "get_fusion_capabilities",
    "is_vectorizable",
    "format_speedup",
]

__version__ = "1.0.0"
