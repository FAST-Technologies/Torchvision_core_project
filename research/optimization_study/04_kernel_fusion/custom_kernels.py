"""
Custom CUDA kernels для критичных операций.

⚠️ Требует установленного CUDA toolkit и компилятора.
"""

import torch
from typing import Optional, Callable
import warnings

try:
    from torch.utils.cpp_extension import load

    CPP_EXTENSION_AVAILABLE = True
except ImportError:
    CPP_EXTENSION_AVAILABLE = False
    warnings.warn("cpp_extension not available. Custom kernels disabled.")


class CustomKernelRegistry:
    """Регистр custom CUDA kernels."""

    _kernels: dict = {}

    @classmethod
    def register(cls, name: str, kernel_func: Callable):
        """Регистрирует custom kernel."""
        cls._kernels[name] = kernel_func

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """Получает registered kernel."""
        return cls._kernels.get(name)


def register_custom_kernel(name: str):
    """Декоратор для регистрации custom kernel."""

    def decorator(func: Callable):
        CustomKernelRegistry.register(name, func)
        return func

    return decorator


# Пример: Custom kernel для fused Niblack (псевдокод)
@register_custom_kernel("fused_niblack_cuda")
def fused_niblack_cuda(
    gray: torch.Tensor,
    window_size: int,
    k: float,
) -> torch.Tensor:
    """
    Custom CUDA kernel для fused Niblack thresholding.

    Объединяет в одном kernel:
    1. Локальное среднее
    2. Локальный std
    3. Вычисление порога
    4. Бинаризация

    ⚠️ Это псевдокод — реальная реализация требует .cu файла.
    """
    if not CPP_EXTENSION_AVAILABLE:
        warnings.warn("Custom kernel not compiled, using fallback")
        return None

    # Реальная реализация загружается через load():
    # fused_module = load(
    #     name="fused_niblack",
    #     sources=["fused_niblack.cpp", "fused_niblack_kernel.cu"],
    #     verbose=True,
    # )
    # return fused_module.fused_niblack(gray, window_size, k)

    return None


def get_custom_kernel(
    method_name: str,
    original_func: Callable,
) -> Optional[Callable]:
    """
    Получает custom kernel для метода или fallback.

    Args:
        method_name: Название метода
        original_func: Оригинальная функция

    Returns:
        Custom kernel или None если недоступен
    """
    kernel_name = f"fused_{method_name}_cuda"
    kernel = CustomKernelRegistry.get(kernel_name)

    if kernel is None:
        return None

    try:
        return kernel
    except Exception as e:
        warnings.warn(f"Custom kernel execution failed: {e}")
        return None
