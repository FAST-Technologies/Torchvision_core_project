"""
Вспомогательные утилиты для оптимизации памяти.
"""

import torch
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings

from .config import MemoryPolicy


def get_memory_backend_info(device: str = "cuda") -> Dict[str, Union[str, bool, int]]:
    """
    Возвращает информацию о бэкенде управления памятью.

    Returns:
        Dict с информацией:
        {
            "device": str,
            "cuda_available": bool,
            "pinned_memory_supported": bool,
            "max_memory_gb": float,
            "current_allocated_mb": float,
            "current_reserved_mb": float,
        }
    """
    info = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "pinned_memory_supported": False,
        "max_memory_gb": 0.0,
        "current_allocated_mb": 0.0,
        "current_reserved_mb": 0.0,
    }

    if device == "cuda" and torch.cuda.is_available():
        info["pinned_memory_supported"] = True
        info["max_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["current_allocated_mb"] = torch.cuda.memory_allocated(0) / 1e6
        info["current_reserved_mb"] = torch.cuda.memory_reserved(0) / 1e6

    return info


def format_memory_size(bytes_value: float) -> str:
    """Форматирует размер памяти в человекочитаемый вид."""
    if bytes_value < 1024:
        return f"{bytes_value:.0f} B"
    elif bytes_value < 1024**2:
        return f"{bytes_value / 1024:.2f} KB"
    elif bytes_value < 1024**3:
        return f"{bytes_value / 1024**2:.2f} MB"
    else:
        return f"{bytes_value / 1024**3:.2f} GB"


def estimate_tensor_size(tensor: torch.Tensor) -> Dict[str, float]:
    """
    Оценивает размер тензора в памяти.

    Returns:
        Dict с размерами:
        {
            "bytes": float,
            "mb": float,
            "elements": int,
            "element_size": int,
        }
    """
    return {
        "bytes": tensor.element_size() * tensor.nelement(),
        "mb": (tensor.element_size() * tensor.nelement()) / 1024**2,
        "elements": tensor.nelement(),
        "element_size": tensor.element_size(),
    }


def detect_memory_leaks(
    baseline_mb: float,
    current_mb: float,
    threshold_mb: float = 50.0,
    relative_threshold: float = 0.1,
) -> Tuple[bool, Dict[str, float]]:
    """
    Детектирует потенциальные утечки памяти.

    Args:
        baseline_mb: Базовое потребление памяти (МБ)
        current_mb: Текущее потребление (МБ)
        threshold_mb: Абсолютный порог утечки (МБ)
        relative_threshold: Относительный порог (доля от baseline)

    Returns:
        Tuple[bool, Dict]: (есть ли утечка, детали)
    """
    diff_mb = current_mb - baseline_mb
    relative_diff = diff_mb / baseline_mb if baseline_mb > 0 else 0

    is_leak = diff_mb > threshold_mb or relative_diff > relative_threshold

    return is_leak, {
        "baseline_mb": baseline_mb,
        "current_mb": current_mb,
        "diff_mb": diff_mb,
        "relative_diff_pct": relative_diff * 100,
        "threshold_mb": threshold_mb,
        "relative_threshold_pct": relative_threshold * 100,
    }


def is_inplace_safe(operation: str, tensor_shape: Tuple[int, ...]) -> bool:
    """
    Проверяет, безопасно ли использовать inplace-операции.

    Args:
        operation: Название операции
        tensor_shape: Форма тензора

    Returns:
        bool: True если inplace безопасен
    """
    # Операции, где inplace может сломать граф вычислений
    unsafe_inplace = [
        "conv2d",
        "conv_transpose2d",  # Свёртки
        "batch_norm",
        "layer_norm",  # Нормализации
        "softmax",
        "log_softmax",  # Активации с градиентами
    ]

    if any(op in operation.lower() for op in unsafe_inplace):
        return False

    # Inplace безопасен для больших тензоров (экономия памяти)
    total_elements = np.prod(tensor_shape)
    return total_elements > 100_000


def create_pinned_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Создаёт тензор в закреплённой (pinned) памяти для быстрого CPU↔GPU.

    Args:
        shape: Форма тензора
        dtype: Тип данных
        device: Устройство (должен быть "cpu" для pinned)

    Returns:
        torch.Tensor: Тензор в pinned memory
    """
    if device is None:
        device = torch.device("cpu")

    if device.type != "cpu":
        warnings.warn("Pinned memory only supported on CPU. Creating regular tensor.")
        return torch.empty(shape, dtype=dtype, device=device)

    return torch.empty(shape, dtype=dtype, device=device, pin_memory=True)


def clear_memory(device: str = "cuda", aggressive: bool = False):
    """
    Очистка памяти.

    Args:
        device: Устройство ("cuda" или "cpu")
        aggressive: Агрессивная очистка (сборка мусора)
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.reset_peak_memory_stats()

    if aggressive:
        gc.collect()
