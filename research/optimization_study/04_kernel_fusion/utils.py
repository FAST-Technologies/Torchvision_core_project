"""
Вспомогательные утилиты для kernel fusion.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import warnings

from .config import FusionStrategy, FUSION_STRATEGIES


def get_fusion_capabilities(device: str = "cuda") -> Dict[str, bool]:
    """
    Проверяет возможности fusion на целевом устройстве.

    Returns:
        Dict с флагами доступных оптимизаций:
        {
            "torch_compile": bool,
            "custom_kernels": bool,
            "vectorization": bool,
            "graph_capture": bool,
        }
    """
    capabilities = {}

    # torch.compile (PyTorch >= 2.0)
    capabilities["torch_compile"] = (
        hasattr(torch, "compile") and torch.__version__ >= "2.0"
    )

    # Custom CUDA kernels (требует компилятора)
    try:
        from torch.utils.cpp_extension import load

        capabilities["custom_kernels"] = True
    except ImportError:
        capabilities["custom_kernels"] = False

    # Векторизация (поддержка на устройстве)
    if device == "cuda" and torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        # Векторизация лучше на современных GPU
        capabilities["vectorization"] = cc[0] >= 6
    else:
        # CPU векторизация через AVX/AVX2/AVX-512
        capabilities["vectorization"] = True

    # Graph capture для tracing
    capabilities["graph_capture"] = hasattr(torch.jit, "trace")

    return capabilities


def is_vectorizable(operation: str, tensor_shape: Tuple[int, ...]) -> bool:
    """
    Проверяет, можно ли векторизовать операцию.

    Args:
        operation: Название операции
        tensor_shape: Форма входного тензора

    Returns:
        bool: True если операция подходит для векторизации
    """
    # Операции, которые плохо векторизуются
    non_vectorizable = [
        "canny_edge",  # Сложный контроль потока
        "watershed",  # Итеративный алгоритм
        "random_walker",  # Решение СЛАУ
        "active_contour",  # Зависимость от предыдущей итерации
    ]

    if any(op in operation.lower() for op in non_vectorizable):
        return False

    # Векторизация эффективна для больших тензоров
    total_elements = np.prod(tensor_shape)
    if total_elements < 1024:
        return False  # Накладные расходы превысят выигрыш

    # Пороговые и градиентные методы — хорошо векторизуются
    vectorizable_patterns = [
        "threshold",
        "sobel",
        "prewitt",
        "scharr",
        "laplacian",
        "gaussian",
        "blur",
        "conv2d",
        "add",
        "mul",
        "sqrt",
        "abs",
    ]

    return any(pattern in operation.lower() for pattern in vectorizable_patterns)


def estimate_kernel_launch_overhead(device: str = "cuda") -> float:
    """
    Оценивает накладные расходы на запуск CUDA kernel.

    Returns:
        float: Ожидаемое время запуска kernel в мс
    """
    if device != "cuda" or not torch.cuda.is_available():
        return 0.01  # CPU overhead ~10 µs

    # Эмпирическая оценка: ~5-20 µs на kernel launch
    # Зависит от драйвера, GPU, загрузки
    cc = torch.cuda.get_device_capability()

    # Более новые GPU имеют меньший overhead
    if cc[0] >= 8:  # Ampere+
        return 0.005  # 5 µs
    elif cc[0] >= 7:  # Volta/Turing
        return 0.010  # 10 µs
    else:  # Pascal и старше
        return 0.020  # 20 µs


def format_speedup(ratio: float) -> str:
    """Форматирует коэффициент ускорения."""
    if ratio >= 10:
        return f"{ratio:.1f}×"
    elif ratio >= 2:
        return f"{ratio:.2f}×"
    else:
        return f"{ratio:.3f}×"


def analyze_graph_complexity(
    func: Callable, example_input: torch.Tensor
) -> Dict[str, Any]:
    """
    Анализирует сложность графа вычислений функции.

    Args:
        func: Функция для анализа
        example_input: Пример входного тензора

    Returns:
        Dict с метриками сложности:
        - num_operations: количество операций
        - num_parameters: количество параметров
        - memory_footprint: оценка использования памяти
        - fusion_potential: потенциал для fusion (0-1)
    """
    try:
        # Трассировка графа
        traced = torch.jit.trace(func, example_input, strict=False)
        graph = traced.graph

        # Подсчёт операций
        operations = list(graph.nodes())
        num_ops = len(operations)

        # Подсчёт параметров
        num_params = sum(
            1 for n in operations if n.kind() in ["prim::Param", "prim::Constant"]
        )

        # Оценка памяти (грубая)
        memory_estimate = example_input.element_size() * example_input.numel()
        for op in operations:
            if "conv" in op.kind().lower():
                memory_estimate *= 2  # Conv удваивает память
            elif "cat" in op.kind().lower() or "stack" in op.kind().lower():
                memory_estimate *= 1.5

        # Потенциал fusion: больше операций = больше возможностей
        fusion_potential = min(1.0, num_ops / 10)

        return {
            "num_operations": num_ops,
            "num_parameters": num_params,
            "memory_footprint_mb": memory_estimate / (1024**2),
            "fusion_potential": fusion_potential,
        }

    except Exception as e:
        warnings.warn(f"Graph analysis failed: {e}")
        return {
            "num_operations": 0,
            "num_parameters": 0,
            "memory_footprint_mb": 0,
            "fusion_potential": 0,
            "error": str(e),
        }


def get_fused_kernel_name(base_name: str, strategy: FusionStrategy) -> str:
    """Генерирует имя для fused kernel."""
    strategy_suffix = {
        FusionStrategy.GRAPH_FUSION: "fused_graph",
        FusionStrategy.MANUAL_FUSION: "fused_manual",
        FusionStrategy.CUSTOM_KERNEL: "fused_cuda",
        FusionStrategy.VECTORIZED: "fused_vectorized",
    }
    return f"{base_name}_{strategy_suffix.get(strategy, 'fused')}"
