"""
06_compilation_study — Исследование компиляционной оптимизации
для классических методов сегментации на базе PyTorch.

Цель: Ускорение инференса через:
- torch.jit (script/trace) — статическая компиляция графа
- torch.compile() (PyTorch 2.0+) — динамическая компиляция с авто-оптимизацией
- Graph capture & freezing — фиксация графа для продакшена
- Backend selection — выбор оптимального бэкенда (inductor, cudagraphs, etc.)

Пример:
    >>> from optimization_study.06_compilation_study import CompilationOptimizer
    >>> optimizer = CompilationOptimizer(segmenter)
    >>> compiled = optimizer.compile_method("sobel_edge", strategy="torch_compile")
    >>> result = optimizer.benchmark("sobel_edge", compiled)
"""

from .config import CompilationConfig, DEFAULT_CONFIG, COMPILATION_STRATEGIES
from .utils import (
    get_compilation_capabilities,
    is_graph_stable,
    estimate_compilation_overhead,
    format_speedup,
    analyze_graph_structure,
)
from .jit_compilation import JITCompiler, JITOptimizer
from .torch_compile_benchmark import TorchCompileOptimizer
from .graph_optimizer import GraphOptimizer, CompilationReport

__all__ = [
    "CompilationOptimizer",
    "JITCompiler",
    "JITOptimizer",
    "TorchCompileOptimizer",
    "GraphOptimizer",
    "CompilationReport",
    "CompilationConfig",
    "DEFAULT_CONFIG",
    "COMPILATION_STRATEGIES",
    "get_compilation_capabilities",
    "is_graph_stable",
    "format_speedup",
]

__version__ = "1.0.0"
