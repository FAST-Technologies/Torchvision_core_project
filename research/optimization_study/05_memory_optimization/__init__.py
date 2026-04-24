"""
05_memory_optimization — Исследование оптимизации использования памяти
для классических методов сегментации на PyTorch.

Цель: Снижение пикового потребления памяти и предотвращение утечек через:
- Кэширование повторяемых тензоров (ядра, буферы)
- Профилирование аллокаций в реальном времени
- Оптимизацию стратегии выделения/освобождения памяти
- Детектирование и устранение утечек памяти

Пример:
    >>> from optimization_study.05_memory_optimization import MemoryOptimizer
    >>> optimizer = MemoryOptimizer(segmenter)
    >>> report = optimizer.profile_method("sauvola_thresholding", image)
    >>> print(f"Peak memory: {report['peak_mb']:.2f} MB")
"""

from .config import MemoryConfig, DEFAULT_CONFIG, MEMORY_POLICIES
from .utils import (
    get_memory_backend_info,
    format_memory_size,
    estimate_tensor_size,
    detect_memory_leaks,
    is_inplace_safe,
)
from .caching_strategy import KernelCache, get_global_kernel_cache
from .memory_profiler import MemoryProfiler
from .memory_optimizer import MemoryOptimizer, OptimizationReport
from .allocation_tracker import AllocationTracker, AllocationSnapshot

__all__ = [
    "MemoryOptimizer",
    "MemoryProfiler",
    "KernelCache",
    "AllocationTracker",
    "OptimizationReport",
    "AllocationSnapshot",
    "MemoryConfig",
    "DEFAULT_CONFIG",
    "MEMORY_POLICIES",
    "get_global_kernel_cache",
    "get_memory_backend_info",
    "format_memory_size",
    "detect_memory_leaks",
]

__version__ = "1.0.0"
