"""
Вспомогательные утилиты для исследования точности.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, ContextManager
from contextlib import contextmanager

from .config import PRECISION_TO_DTYPE, is_dtype_supported


def get_available_dtypes(device: str = "cuda") -> List[str]:
    """
    Возвращает список доступных типов данных для устройства.

    Returns:
        List[str]: Доступные прецизионные режимы
    """
    available = []
    for name, dtype in PRECISION_TO_DTYPE.items():
        if is_dtype_supported(dtype, device):
            available.append(name)
    return available


def format_time(seconds: float) -> str:
    """Форматирует время в человекочитаемый вид."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.3f} ms"
    else:
        return f"{seconds:.3f} s"


def format_speedup(ratio: float) -> str:
    """Форматирует коэффициент ускорения."""
    if ratio >= 10:
        return f"{ratio:.1f}×"
    elif ratio >= 2:
        return f"{ratio:.2f}×"
    else:
        return f"{ratio:.3f}×"


def compute_pixel_agreement(
    mask_a: Union[torch.Tensor, np.ndarray],
    mask_b: Union[torch.Tensor, np.ndarray],
    tolerance: float = 1e-4,
) -> float:
    """
    Вычисляет процент совпадающих пикселей между двумя масками.

    Args:
        mask_a, mask_b: Маски для сравнения
        tolerance: Допустимая разница для "совпадения"

    Returns:
        float: Доля совпадающих пикселей [0, 1]
    """
    if isinstance(mask_a, torch.Tensor):
        mask_a = mask_a.cpu().numpy()
    if isinstance(mask_b, torch.Tensor):
        mask_b = mask_b.cpu().numpy()

    # Приводим к float для сравнения
    a = mask_a.astype(np.float32)
    b = mask_b.astype(np.float32)

    agreement = np.mean(np.abs(a - b) <= tolerance)
    return float(agreement)


def compute_mse(mask_a: Union[torch.Tensor, np.ndarray], mask_b: Union[torch.Tensor, np.ndarray]) -> float:
    """Вычисляет MSE между двумя масками."""
    if isinstance(mask_a, torch.Tensor):
        mask_a = mask_a.cpu().numpy()
    if isinstance(mask_b, torch.Tensor):
        mask_b = mask_b.cpu().numpy()

    mse = np.mean((mask_a.astype(float) - mask_b.astype(float)) ** 2)
    return float(mse)


@contextmanager
def safe_autocast_context(
    device_type: str,
    dtype: torch.dtype,
    enabled: bool = True,
    cpu_enabled: bool = False,
) -> ContextManager:
    """
    Безопасный контекст autocast с поддержкой CPU.

    Args:
        device_type: 'cuda' или 'cpu'
        dtype: Тип данных для autocast
        enabled: Включить ли autocast
        cpu_enabled: Разрешить ли autocast на CPU (экспериментально)

    Yields:
        Контекстный менеджер autocast или dummy-контекст
    """
    if not enabled:
        yield
        return

    if device_type == "cuda" and torch.cuda.is_available():
        # Стандартный autocast для CUDA
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            yield
    elif device_type == "cpu" and cpu_enabled:
        # Экспериментальный autocast для CPU (PyTorch >= 1.10)
        try:
            with torch.autocast(device_type="cpu", dtype=dtype, enabled=True):
                yield
        except (AttributeError, NotImplementedError):
            # Fallback: выполняем без autocast
            yield
    else:
        # Нет поддержки — выполняем без autocast
        yield


def convert_tensor_precision(
    tensor: torch.Tensor, target_dtype: torch.dtype, quantize_int8: bool = False
) -> torch.Tensor:
    """
    Конвертация тензора в целевой тип данных.

    Args:
        tensor: Исходный тензор
        target_dtype: Целевой dtype
        quantize_int8: Использовать ли квантование для INT8

    Returns:
        torch.Tensor: Конвертированный тензор
    """
    if target_dtype == torch.int8 and quantize_int8:
        # Квантование для INT8 (только для бенчмарка, де-квантуем обратно)
        scale = tensor.abs().max() / 127
        if scale < 1e-8:
            scale = torch.tensor(1.0, device=tensor.device)

        # Квантуем → де-квантуем для бенчмарка
        quantized = torch.quantize_per_tensor(tensor.float(), scale=scale.item(), zero_point=0, dtype=torch.qint8)
        return quantized.dequantize().to(target_dtype)

    # Простая конвертация для FP16/BF16
    return tensor.to(target_dtype)


def normalize_for_comparison(tensor: torch.Tensor) -> torch.Tensor:
    """
    Нормализует тензор для сравнения (приводит к [0, 1]).

    Полезно для сравнения результатов с разной нормализацией.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val - min_val < 1e-8:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val + 1e-8)
