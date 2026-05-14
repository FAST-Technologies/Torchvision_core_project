"""
Вспомогательные утилиты для компиляционной оптимизации.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import warnings
import time

from .config import CompilationStrategy


def get_compilation_capabilities(device: str = "cuda") -> Dict[str, Any]:
    """
    Проверяет доступные возможности компиляции.

    Returns:
        Dict с флагами:
        {
            "torch_version": str,
            "torch_compile_available": bool,
            "jit_available": bool,
            "cudagraphs_supported": bool,
            "inductor_available": bool,
        }
    """
    capabilities = {
        "torch_version": torch.__version__,
        "jit_available": hasattr(torch, "jit"),
        "torch_compile_available": hasattr(torch, "compile") and torch.__version__ >= "2.0",
        "cudagraphs_supported": False,
        "inductor_available": False,
    }

    if device == "cuda" and torch.cuda.is_available():
        # CUDA graphs требуют Ampere+ для полной поддержки
        cc = torch.cuda.get_device_capability()
        capabilities["cudagraphs_supported"] = cc[0] >= 7

        # Inductor доступен с PyTorch 2.0+
        if capabilities["torch_compile_available"]:
            try:
                from torch._inductor import config as inductor_config

                capabilities["inductor_available"] = True
            except ImportError:
                pass

    return capabilities


def is_graph_stable(func: Callable, example_input: torch.Tensor, n_tests: int = 3) -> bool:
    """
    Проверяет стабильность графа вычислений функции.

    Граф считается стабильным, если при нескольких прогонах
    не возникает динамических изменений структуры.

    Args:
        func: Функция для проверки
        example_input: Пример входа
        n_tests: Количество тестовых прогонов

    Returns:
        bool: True если граф стабилен
    """
    if not hasattr(torch, "jit"):
        return False

    try:
        # Пробуем трассировать граф
        traced = torch.jit.trace(func, example_input, check_trace=True)

        # Несколько прогонов для проверки стабильности
        for _ in range(n_tests):
            _ = traced(example_input)

        return True

    except Exception:
        return False


def estimate_compilation_overhead(compile_func: Callable, example_input: torch.Tensor, n_warmup: int = 3) -> float:
    """
    Оценивает накладные расходы на компиляцию.

    Args:
        compile_func: Функция компиляции
        example_input: Пример входа
        n_warmup: Прогревочные прогоны

    Returns:
        float: Время компиляции в секундах
    """
    # Прогрев
    for _ in range(n_warmup):
        _ = compile_func(example_input)

    # Замер
    start = time.perf_counter()
    compiled = compile_func(example_input)
    end = time.perf_counter()

    return end - start


def format_speedup(ratio: float) -> str:
    """Форматирует коэффициент ускорения."""
    if ratio >= 10:
        return f"{ratio:.1f}×"
    elif ratio >= 2:
        return f"{ratio:.2f}×"
    else:
        return f"{ratio:.3f}×"


def analyze_graph_structure(func: Callable, example_input: torch.Tensor) -> Dict[str, Any]:
    """
    Анализирует структуру графа вычислений.

    Args:
        func: Функция для анализа
        example_input: Пример входа

    Returns:
        Dict с метриками графа
    """
    try:
        # Трассировка графа
        traced = torch.jit.trace(func, example_input, strict=False)
        graph = traced.graph

        # Подсчёт операций
        operations = list(graph.nodes())
        op_types = {}
        for node in operations:
            op_type = node.kind()
            op_types[op_type] = op_types.get(op_type, 0) + 1

        # Оценка сложности
        num_ops = len(operations)
        num_params = sum(1 for n in operations if n.kind() in ["prim::Param", "prim::Constant"])

        # Потенциал для оптимизации
        fusion_candidates = sum(
            1 for op in op_types.keys() if any(k in op.lower() for k in ["conv", "add", "mul", "relu", "batchnorm"])
        )

        return {
            "num_operations": num_ops,
            "num_parameters": num_params,
            "operation_types": op_types,
            "fusion_candidates": fusion_candidates,
            "optimization_potential": min(1.0, fusion_candidates / max(1, num_ops)),
        }

    except Exception as e:
        warnings.warn(f"Graph analysis failed: {e}")
        return {
            "num_operations": 0,
            "num_parameters": 0,
            "operation_types": {},
            "fusion_candidates": 0,
            "optimization_potential": 0,
            "error": str(e),
        }
