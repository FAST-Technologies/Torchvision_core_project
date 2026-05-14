# research/optimization_study/04_kernel_fusion/fusion_optimizer.py

# 🔬 Исследование 4: Векторизация и Kernel Fusion
# Цель: Уменьшение накладных расходов через fusion операций

"""
Основной класс для оптимизации через kernel fusion.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple, Optional, Callable, Any
import time
import warnings
import numpy as np

from .config import FusionConfig, DEFAULT_CONFIG, FusionStrategy, FUSION_STRATEGIES
from .utils import (
    get_fusion_capabilities,
    is_vectorizable,
    estimate_kernel_launch_overhead,
    format_speedup,
    analyze_graph_complexity,
    get_fused_kernel_name,
)


class FusedOperation:
    """
    Представление объединённой операции.

    Хранит информацию о:
    - Исходных операциях
    - Стратегии fusion
    - Скомпилированной функции
    - Метриках производительности
    """

    def __init__(
        self,
        name: str,
        original_ops: List[str],
        strategy: FusionStrategy,
        fused_func: Callable,
    ):
        self.name = name
        self.original_ops = original_ops
        self.strategy = strategy
        self.fused_func = fused_func
        self.metrics: Dict[str, float] = {}
        self.graph_info: Optional[Dict[str, Any]] = None

    def record_metrics(self, original_time: float, fused_time: float):
        """Записывает метрики производительности."""
        self.metrics["original_ms"] = original_time
        self.metrics["fused_ms"] = fused_time
        self.metrics["speedup"] = original_time / fused_time if fused_time > 0 else float("inf")

    def __call__(self, *args, **kwargs):
        """Вызов fused функции."""
        return self.fused_func(*args, **kwargs)


class FusionOptimizer:
    """
    Оптимизатор для объединения операций в TorchSegmenter.

    Поддерживает:
    - Graph fusion через torch.compile()
    - Manual fusion через ручное объединение операций
    - Custom CUDA kernels (опционально)
    - Векторизацию вычислений

    Пример:
        >>> optimizer = FusionOptimizer(segmenter)
        >>> fused = optimizer.fuse_method("sauvola_thresholding")
        >>> result = fused(image_tensor)
    """

    def __init__(
        self,
        segmenter: Any,
        config: Optional[FusionConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            segmenter: Экземпляр TorchSegmenter
            config: Конфигурация fusion
            device: Устройство для вычислений
        """
        self.segmenter = segmenter
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Кэш fused операций
        self.fused_operations: Dict[str, FusedOperation] = {}

        # Возможности устройства
        self.capabilities = get_fusion_capabilities(str(self.device.type))

        if self.config.verbose:
            print(f"⚡ FusionOptimizer initialized:")
            print(f"   Device: {self.device}")
            print(f"   Capabilities: {self.capabilities}")
            print(f"   Strategy: {self.config.strategy.name}")

    def _get_strategy_for_method(self, method_name: str) -> FusionStrategy:
        """Определяет стратегию fusion для метода."""
        # Проверка исключений
        if method_name in self.config.exclude_methods:
            return FusionStrategy.NONE

        # Проверка включений
        if self.config.include_methods and method_name not in self.config.include_methods:
            return FusionStrategy.NONE

        # Стратегия из конфигурации или по умолчанию
        return FUSION_STRATEGIES.get(method_name, FUSION_STRATEGIES.get("default", FusionStrategy.GRAPH_FUSION))

    def _fuse_graph_compilation(
        self,
        method_name: str,
        original_func: Callable,
        example_input: torch.Tensor,
    ) -> Callable:
        """
        Fusion через torch.compile() (graph-level оптимизация).

        Args:
            method_name: Название метода
            original_func: Оригинальная функция
            example_input: Пример входа для компиляции

        Returns:
            Callable: Скомпилированная функция
        """
        if not self.capabilities["torch_compile"]:
            warnings.warn("torch.compile not available, falling back to original")
            return original_func

        try:
            compiled_func = torch.compile(
                original_func,
                mode=self.config.compile_mode,
                fullgraph=self.config.fullgraph,
                dynamic=self.config.dynamic,
            )

            # "Прогрев" компиляции
            with torch.inference_mode():
                _ = compiled_func(example_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            if self.config.verbose:
                print(f"   ✓ Graph fusion applied via torch.compile()")

            return compiled_func

        except Exception as e:
            warnings.warn(f"Graph fusion failed: {e}. Using original function.")
            return original_func

    def _fuse_manual_niblack_sauvola(
        self,
        method_name: str,
        window_size: int,
        k: float,
        r: Optional[float] = None,  # Для Sauvola
    ) -> Callable:
        """
        Ручное объединение операций для методов типа Ниблака/Сауволы.

        Fused computation:
        1. local_mean = conv2d(gray, mean_kernel)
        2. local_mean_sq = conv2d(gray**2, mean_kernel)
        3. local_std = sqrt(local_mean_sq - local_mean**2)
        4. threshold = local_mean + k * local_std * (local_std / r)  # Sauvola
        5. return gray > threshold

        Все операции в одном kernel через vectorized ops.
        """
        padding = window_size // 2
        mean_kernel = torch.ones(1, 1, window_size, window_size, device=self.device) / (window_size**2)

        def fused_func(gray: torch.Tensor) -> torch.Tensor:
            # Убедимся, что вход 4D
            if gray.dim() == 2:
                gray = gray.unsqueeze(0).unsqueeze(0)
            elif gray.dim() == 3:
                gray = gray.unsqueeze(0)

            # Fused вычисления
            gray_sq = gray**2
            local_mean = F.conv2d(gray, mean_kernel, padding=padding)
            local_mean_sq = F.conv2d(gray_sq, mean_kernel, padding=padding)

            local_var = torch.clamp(local_mean_sq - local_mean**2, min=1e-8)
            local_std = torch.sqrt(local_var)

            if r is not None:  # Sauvola
                threshold = local_mean * (1 + k * (local_std / r - 1))
            else:  # Niblack
                threshold = local_mean + k * local_std

            return (gray > threshold).float()

        # Применяем torch.compile для дополнительного fusion
        if self.capabilities["torch_compile"]:
            fused_func = torch.compile(
                fused_func,
                mode="reduce-overhead",
                fullgraph=True,
            )

        return fused_func

    def _fuse_vectorized_gradient(
        self,
        method_name: str,
        kernel_x: torch.Tensor,
        kernel_y: torch.Tensor,
    ) -> Callable:
        """
        Векторизованное вычисление градиента (Sobel/Prewitt/Scharr).

        Объединяет:
        1. conv2d для Gx и Gy
        2. magnitude = sqrt(Gx**2 + Gy**2)
        3. normalization
        4. thresholding

        В одном kernel через fused ops.
        """

        def fused_func(gray: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
            if gray.dim() == 2:
                gray = gray.unsqueeze(0).unsqueeze(0)
            elif gray.dim() == 3:
                gray = gray.unsqueeze(0)

            # Векторизованные свёртки
            gx = F.conv2d(gray, kernel_x, padding=1)
            gy = F.conv2d(gray, kernel_y, padding=1)

            # Fused magnitude + normalization + thresholding
            magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
            if magnitude.max() > 0:
                magnitude = magnitude / magnitude.max()

            return (magnitude > threshold).float()

        if self.capabilities["torch_compile"]:
            fused_func = torch.compile(fused_func, mode="reduce-overhead")

        return fused_func

    def fuse_method(
        self,
        method_name: str,
        example_input: Optional[torch.Tensor] = None,
        **fusion_params,
    ) -> Optional[FusedOperation]:
        """
        Применяет fusion к указанному методу.

        Args:
            method_name: Название метода из segmenter.method_map
            example_input: Пример входа для компиляции
            **fusion_params: Дополнительные параметры fusion

        Returns:
            FusedOperation или None если fusion не применим
        """
        if method_name not in self.segmenter.method_map:
            raise ValueError(f"Method '{method_name}' not found")

        original_func = self.segmenter.method_map[method_name]
        strategy = self._get_strategy_for_method(method_name)

        if strategy == FusionStrategy.NONE:
            if self.config.verbose:
                print(f"   ✗ Fusion disabled for {method_name}")
            return None

        # Подготовка примера входа
        if example_input is None:
            example_input = torch.randn(1, 1, 512, 512, device=self.device)

        # Анализ графа
        if self.config.profile_graph:
            graph_info = analyze_graph_complexity(original_func, example_input)
            if self.config.verbose:
                print(
                    f"   📊 Graph: {graph_info['num_operations']} ops, "
                    f"fusion potential: {graph_info['fusion_potential']:.2f}"
                )
        else:
            graph_info = None

        # Применение стратегии fusion
        if strategy == FusionStrategy.GRAPH_FUSION:
            fused_func = self._fuse_graph_compilation(method_name, original_func, example_input)
        elif strategy == FusionStrategy.MANUAL_FUSION:
            # Специальные fused реализации для известных методов
            if "niblack" in method_name.lower():
                window_size = self.segmenter.params.get("window_size", 15)
                k = self.segmenter.params.get("k", -0.2)
                fused_func = self._fuse_manual_niblack_sauvola(method_name, window_size, k)
            elif "sauvola" in method_name.lower():
                window_size = self.segmenter.params.get("window_size", 15)
                k = self.segmenter.params.get("k", 0.5)
                r = self.segmenter.params.get("r", 128)
                fused_func = self._fuse_manual_niblack_sauvola(method_name, window_size, k, r)
            else:
                # Fallback на graph fusion
                fused_func = self._fuse_graph_compilation(method_name, original_func, example_input)

        elif strategy == FusionStrategy.VECTORIZED:
            # Векторизованные градиентные методы
            if "sobel" in method_name.lower():
                kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).view(1, 1, 3, 3)
                kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).view(1, 1, 3, 3)
                fused_func = self._fuse_vectorized_gradient(method_name, kernel_x, kernel_y)
            elif "prewitt" in method_name.lower():
                kernel_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], device=self.device).view(1, 1, 3, 3)
                kernel_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], device=self.device).view(1, 1, 3, 3)
                fused_func = self._fuse_vectorized_gradient(method_name, kernel_x, kernel_y)
            else:
                fused_func = self._fuse_graph_compilation(method_name, original_func, example_input)

        elif strategy == FusionStrategy.CUSTOM_KERNEL:
            # Custom CUDA kernels (опционально)
            if self.config.enable_custom_kernels and self.capabilities["custom_kernels"]:
                from .custom_kernels import get_custom_kernel

                fused_func = get_custom_kernel(method_name, original_func)
                if fused_func is None:
                    warnings.warn(f"Custom kernel not available, using graph fusion")
                    fused_func = self._fuse_graph_compilation(method_name, original_func, example_input)
            else:
                fused_func = self._fuse_graph_compilation(method_name, original_func, example_input)

        else:
            fused_func = original_func

        # Создание объекта FusedOperation
        fused_op = FusedOperation(
            name=get_fused_kernel_name(method_name, strategy),
            original_ops=[method_name],
            strategy=strategy,
            fused_func=fused_func,
        )
        fused_op.graph_info = graph_info

        # Кэширование
        self.fused_operations[method_name] = fused_op

        if self.config.verbose:
            print(f"   ✅ Fused: {fused_op.name} ({strategy.name})")

        return fused_op

    def benchmark_fusion(
        self,
        method_name: str,
        fused_op: Optional[FusedOperation] = None,
        example_input: Optional[torch.Tensor] = None,
        n_runs: int = 50,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Бенчмарк fused версии против оригинала.

        Args:
            method_name: Название метода
            fused_op: FusedOperation (если None — создаётся)
            example_input: Пример входа
            n_runs: Количество запусков
            warmup: Прогревочные запуски

        Returns:
            Dict с метриками производительности
        """
        if fused_op is None:
            fused_op = self.fuse_method(method_name, example_input)
            if fused_op is None:
                return {"error": "Fusion not applicable"}

        original_func = self.segmenter.method_map[method_name]

        if example_input is None:
            example_input = torch.randn(1, 1, 512, 512, device=self.device)

        def measure(func, inp, n, w):
            # Warm-up
            for _ in range(w):
                _ = func(inp)
                if inp.device.type == "cuda":
                    torch.cuda.synchronize()

            # Замер
            times = []
            with torch.inference_mode():
                for _ in range(n):
                    if inp.device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = func(inp)
                    if inp.device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)

            return np.mean(times), np.std(times)

        # Замер оригинала
        orig_mean, orig_std = measure(original_func, example_input, n_runs, warmup)
        # Замер fused версии
        fused_mean, fused_std = measure(fused_op.fused_func, example_input, n_runs, warmup)

        # Запись метрик
        fused_op.record_metrics(orig_mean, fused_mean)

        speedup = orig_mean / fused_mean if fused_mean > 0 else float("inf")

        return {
            "method": method_name,
            "strategy": fused_op.strategy.name,
            "original_mean_ms": orig_mean,
            "original_std_ms": orig_std,
            "fused_mean_ms": fused_mean,
            "fused_std_ms": fused_std,
            "speedup": speedup,
            "speedup_formatted": format_speedup(speedup),
        }

    def fuse_all_methods(
        self,
        methods: Optional[List[str]] = None,
        example_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, FusedOperation]:
        """
        Применяет fusion ко всем подходящим методам.

        Args:
            methods: Список методов (по умолчанию — все из config)
            example_input: Пример входа

        Returns:
            Dict {method_name: FusedOperation}
        """
        if methods is None:
            methods = [m for m in self.segmenter.method_map.keys() if m not in self.config.exclude_methods]

        if self.config.verbose:
            print(f"🔧 Fusing {len(methods)} methods...")

        results = {}
        for method_name in methods:
            try:
                fused_op = self.fuse_method(method_name, example_input)
                if fused_op is not None:
                    results[method_name] = fused_op
            except Exception as e:
                warnings.warn(f"Fusion failed for {method_name}: {e}")

        if self.config.verbose:
            print(f"✅ Fused {len(results)}/{len(methods)} methods")

        return results


# class FusedThresholding(nn.Module):
#     """Пример fusion для адаптивных пороговых методов"""

#     def __init__(self, window_size: int, k: float = -0.2):
#         super().__init__()
#         self.window_size = window_size
#         self.k = k
#         self.padding = window_size // 2

#         # Предвычисленное ядро для локального среднего
#         self.register_buffer(
#             "mean_kernel",
#             torch.ones(1, 1, window_size, window_size) / (window_size**2)
#         )
#         # Ядро для локального квадрата
#         self.register_buffer(
#             "square_kernel",
#             torch.ones(1, 1, window_size, window_size)
#         )

#     def forward(self, gray: torch.Tensor) -> torch.Tensor:
#         # Fused computation: mean, mean_sq, std, threshold, binarize
#         # Всё в одном kernel через torch.ops или custom CUDA kernel

#         # Локальное среднее
#         local_mean = F.conv2d(gray, self.mean_kernel, padding=self.padding)

#         # Локальный квадрат среднего (для std)
#         local_mean_sq = F.conv2d(
#             gray**2, self.mean_kernel, padding=self.padding
#         )

#         # Std = sqrt(E[X^2] - E[X]^2)
#         local_var = torch.clamp(local_mean_sq - local_mean**2, min=1e-8)
#         local_std = torch.sqrt(local_var)

#         # Порог Ниблака: T = μ + k·σ
#         threshold = local_mean + self.k * local_std

#         # Бинаризация в одном ops
#         return (gray > threshold).float()

#     # 🔥 Для ещё большего ускорения можно написать custom CUDA kernel:
#     # @torch.library.custom_op("mylib::fused_niblack", mutates_args=())
#     # def fused_niblack_kernel(gray: Tensor, window: int, k: float) -> Tensor:
#     #     ...
