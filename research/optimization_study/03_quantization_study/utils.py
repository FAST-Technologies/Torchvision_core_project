"""
Вспомогательные утилиты для исследования квантования.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings

from .config import SUPPORTED_DTYPES, QUANTIZATION_SCHEMES


def get_available_quantization_backends(device: str = "cpu") -> Dict[str, bool]:
    """
    Проверяет доступность бэкендов квантования.

    Returns:
        Dict с флагами доступности:
        {
            "fbgemm": bool,    # CPU backend (Intel)
            "qnnpack": bool,   # CPU backend (ARM)
            "xnnpack": bool,   # CPU backend (новый)
            "cuda": bool,      # Экспериментальная поддержка GPU
        }
    """
    backends = {}

    # Проверка доступных движков
    try:
        from torch.ao.quantization import get_default_qconfig

        backends["fbgemm"] = torch.backends.quantized.engine == "fbgemm"
        backends["qnnpack"] = torch.backends.quantized.engine == "qnnpack"
        backends["xnnpack"] = torch.backends.quantized.engine == "xnnpack"
    except:
        backends["fbgemm"] = False
        backends["qnnpack"] = False
        backends["xnnpack"] = False

    # CUDA поддержка (экспериментальная)
    backends["cuda"] = torch.cuda.is_available() and device == "cuda"

    return backends


def format_speedup(ratio: float) -> str:
    """Форматирует коэффициент ускорения."""
    if ratio >= 10:
        return f"{ratio:.1f}×"
    elif ratio >= 2:
        return f"{ratio:.2f}×"
    else:
        return f"{ratio:.3f}×"


def format_size_reduction(original_mb: float, quantized_mb: float) -> str:
    """Форматирует сокращение размера модели."""
    reduction = (1 - quantized_mb / original_mb) * 100
    return f"{reduction:.1f}% ({original_mb:.2f}MB → {quantized_mb:.2f}MB)"


def compute_quantization_error(
    reference: Union[torch.Tensor, np.ndarray],
    quantized: Union[torch.Tensor, np.ndarray],
    tolerance: float = 1e-3,
) -> Dict[str, float]:
    """
    Вычисляет метрики ошибки квантования.

    Args:
        reference: Референсный результат (FP32)
        quantized: Квантованный результат
        tolerance: Допустимая погрешность для "совпадения"

    Returns:
        Dict с метриками:
        - pixel_agreement: доля совпадающих пикселей
        - mse: среднеквадратичная ошибка
        - max_diff: максимальное отклонение
        - relative_error: относительная ошибка
    """
    # Конвертация к numpy
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    if isinstance(quantized, torch.Tensor):
        quantized = quantized.cpu().numpy()

    # Приведение к float для сравнения
    ref_f = reference.astype(np.float32)
    quant_f = quantized.astype(np.float32)

    # Метрики
    agreement = np.mean(np.abs(ref_f - quant_f) <= tolerance)
    mse = np.mean((ref_f - quant_f) ** 2)
    max_diff = np.max(np.abs(ref_f - quant_f))

    # Относительная ошибка (с защитой от деления на 0)
    ref_norm = np.linalg.norm(ref_f)
    rel_error = np.linalg.norm(ref_f - quant_f) / (ref_norm + 1e-8)

    return {
        "pixel_agreement": float(agreement),
        "mse": float(mse),
        "max_diff": float(max_diff),
        "relative_error": float(rel_error),
    }


def safe_quantize_tensor(
    tensor: torch.Tensor,
    scheme: QUANTIZATION_SCHEMES,
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
) -> torch.Tensor:
    """
    Безопасное квантование тензора с обработкой ошибок.

    Args:
        tensor: Входной тензор
        scheme: Схема квантования
        scale: Масштабный коэффициент (для manual quantization)
        zero_point: Точка нуля (для manual quantization)

    Returns:
        torch.Tensor: Квантованный тензор или оригинал при ошибке
    """
    try:
        if scheme == "fp32":
            return tensor.to(torch.float32)

        elif scheme == "fp16":
            return tensor.to(torch.float16)

        elif scheme in ["int8_dynamic", "int8_static"]:
            # Автоматическое вычисление scale/zero_point если не заданы
            if scale is None:
                scale = (tensor.max() - tensor.min()) / 255.0
            if zero_point is None:
                zero_point = int(-tensor.min() / scale)

            # Квантование
            quantized = torch.quantize_per_tensor(
                tensor,
                scale=float(scale),
                zero_point=int(zero_point),
                dtype=torch.qint8,
            )
            return quantized

        else:
            warnings.warn(f"Unknown quantization scheme: {scheme}")
            return tensor

    except Exception as e:
        warnings.warn(f"Quantization failed ({scheme}): {e}. Using FP32 fallback.")
        return tensor.to(torch.float32)


def is_method_quantizable(method_name: str, method_func: Callable) -> bool:
    """
    Проверяет, поддерживает ли метод квантование.

    Квантование не поддерживается для:
    - Методов с динамическим контролем потока
    - Методов с операциями без квантованных ядер
    - Методов, чувствительных к потере точности

    Args:
        method_name: Название метода
        method_func: Функция метода

    Returns:
        bool: True если метод можно квантовать
    """
    # Методы, которые обычно НЕ поддерживают квантование
    non_quantizable_patterns = [
        "active_contour",  # Итеративные вычисления с условиями
        "chan_vese",  # Энергетическая оптимизация
        "random_walker",  # Решение СЛАУ
        "phase_congruency",  # FFT и сложные операции
        "grabcut",  # GMM и итерации
    ]

    for pattern in non_quantizable_patterns:
        if pattern in method_name.lower():
            return False

    # Простые методы обычно поддерживают квантование
    quantizable_patterns = [
        "threshold",  # Пороговые методы
        "sobel",
        "prewitt",
        "scharr",  # Градиентные операторы
        "canny",  # Может работать с потерей точности
        "laplacian",
        "log",  # Линейные фильтры
    ]

    for pattern in quantizable_patterns:
        if pattern in method_name.lower():
            return True

    # По умолчанию — разрешаем, но с предупреждением
    return True


def estimate_model_size(model: torch.nn.Module, dtype: torch.dtype) -> float:
    """
    Оценивает размер модели в памяти для заданного типа данных.

    Returns:
        float: Размер в мегабайтах
    """
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    size_bytes = total_params * bytes_per_param
    return size_bytes / (1024**2)  # MB
