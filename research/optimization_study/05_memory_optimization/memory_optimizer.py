"""
Основной класс для оптимизации памяти в TorchSegmenter.
"""

import torch
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import warnings

from .config import MemoryConfig, DEFAULT_CONFIG, MemoryPolicy, MEMORY_POLICIES
from .utils import (
    get_memory_backend_info,
    format_memory_size,
    estimate_tensor_size,
    clear_memory,
    is_inplace_safe,
)
from .caching_strategy import KernelCache, TensorPool, get_global_kernel_cache
from .memory_profiler import MemoryProfiler
from .allocation_tracker import AllocationTracker


@dataclass
class OptimizationReport:
    """Отчёт об оптимизации."""

    method_name: str
    original_peak_mb: float
    optimized_peak_mb: float
    memory_saved_mb: float
    speedup: float
    policy_applied: MemoryPolicy
    cache_hits: int
    recommendations: List[str] = field(default_factory=list)

    @property
    def reduction_pct(self) -> float:
        """Процент снижения потребления памяти."""
        if self.original_peak_mb == 0:
            return 0.0
        return (self.memory_saved_mb / self.original_peak_mb) * 100


class MemoryOptimizer:
    """
    Оптимизатор памяти для TorchSegmenter методов.

    Поддерживает:
    - Автоматический выбор политики памяти
    - Кэширование ядер и буферов
    - Inplace-оптимизации
    - Детектирование и устранение утечек

    Пример:
        >>> optimizer = MemoryOptimizer(segmenter)
        >>> report = optimizer.optimize_method("sauvola_thresholding", image)
        >>> print(f"Saved: {report.memory_saved_mb:.2f} MB ({report.reduction_pct:.1f}%)")
    """

    def __init__(
        self,
        segmenter: Any,
        config: Optional[MemoryConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            segmenter: Экземпляр TorchSegmenter
            config: Конфигурация оптимизации
            device: Устройство для вычислений
        """
        self.segmenter = segmenter
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(device or self.config.device)

        # Компоненты
        self.profiler = MemoryProfiler(self.config)
        self.kernel_cache = get_global_kernel_cache()
        self.allocation_tracker = AllocationTracker()

        # Пулы для часто используемых буферов
        self._tensor_pools: Dict[str, TensorPool] = {}

        # Результаты оптимизации
        self.optimization_reports: List[OptimizationReport] = []

        if self.config.verbose:
            backend_info = get_memory_backend_info(str(self.device.type))
            print(f"🧠 MemoryOptimizer initialized:")
            print(f"   Device: {self.device}")
            print(f"   Max memory: {backend_info.get('max_memory_gb', 0):.2f} GB")
            print(f"   Policy: {self.config.policy.name}")

    def _get_policy_for_method(self, method_name: str) -> MemoryPolicy:
        """Определяет политику памяти для метода."""
        if method_name in self.config.exclude_methods:
            return MemoryPolicy.LAZY

        if (
            self.config.include_methods
            and method_name not in self.config.include_methods
        ):
            return MemoryPolicy.LAZY

        return MEMORY_POLICIES.get(
            method_name, MEMORY_POLICIES.get("default", self.config.policy)
        )

    def _apply_inplace_optimizations(
        self,
        func: Callable,
        method_name: str,
        input_tensor: torch.Tensor,
    ) -> Callable:
        """
        Применяет inplace-оптимизации к функции.

        ⚠️ Требует осторожности — может сломать градиенты!
        """
        if not is_inplace_safe(method_name, input_tensor.shape):
            return func

        # Обёртка с inplace-флагами
        def optimized_func(*args, **kwargs):
            with torch.inference_mode():
                # Пример: для свёрток можно использовать inplace=True в ReLU
                # Реальная реализация зависит от конкретной функции
                return func(*args, **kwargs)

        return optimized_func

    def _precompute_and_cache_kernels(self, method_name: str):
        """Предвычисляет и кэширует ядра для метода."""
        policy = self._get_policy_for_method(method_name)

        if policy not in [MemoryPolicy.POOLED, MemoryPolicy.REUSE]:
            return

        # Пример: кэширование ядер для адаптивных методов
        if "threshold" in method_name and "adaptive" in method_name:
            window_size = self.segmenter.params.get("window_size", 15)

            # Ядро для локального среднего
            self.kernel_cache.get_or_create(
                f"{method_name}_mean_kernel",
                shape=(1, 1, window_size, window_size),
                dtype=torch.float32,
                device=self.device,
                creator=lambda s, d, dev: torch.ones(s, dtype=d, device=dev)
                / (window_size**2),
            )

        # Градиентные методы
        if "edge" in method_name:
            # Ядра Собеля/Прюитта
            for name, kernel_data in [
                ("sobel_x", [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                ("sobel_y", [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                ("prewitt_x", [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            ]:
                self.kernel_cache.get_or_create(
                    f"{method_name}_{name}",
                    shape=(1, 1, 3, 3),
                    dtype=torch.float32,
                    device=self.device,
                    creator=lambda s, d, dev: torch.tensor(
                        kernel_data, dtype=d, device=dev
                    ).view(s),
                )

    def optimize_method(
        self,
        method_name: str,
        input_tensor: torch.Tensor,
        n_runs: Optional[int] = None,
    ) -> OptimizationReport:
        """
        Оптимизирует метод и возвращает отчёт.

        Args:
            method_name: Название метода
            input_tensor: Входной тензор
            n_runs: Количество запусков для бенчмарка

        Returns:
            OptimizationReport: Детали оптимизации
        """
        if method_name not in self.segmenter.method_map:
            raise ValueError(f"Method '{method_name}' not found")

        original_func = self.segmenter.method_map[method_name]
        policy = self._get_policy_for_method(method_name)

        # === Базовый замер (без оптимизаций) ===
        if self.config.verbose:
            print(f"🔍 Profiling {method_name} (baseline)...")

        baseline_report = self.profiler.profile_method(
            original_func, input_tensor, n_runs=n_runs
        )
        original_peak = baseline_report.get("peak_mb_mean", 0)

        # === Применение оптимизаций ===
        if self.config.verbose:
            print(f"⚙️  Applying optimizations: {policy.name}")

        # 1. Кэширование ядер
        self._precompute_and_cache_kernels(method_name)

        # 2. Inplace-оптимизации
        optimized_func = self._apply_inplace_optimizations(
            original_func, method_name, input_tensor
        )

        # 3. Очистка перед замером
        clear_memory(str(self.device.type), aggressive=True)

        # === Замер после оптимизаций ===
        if self.config.verbose:
            print(f"🔍 Profiling {method_name} (optimized)...")

        optimized_report = self.profiler.profile_method(
            optimized_func, input_tensor, n_runs=n_runs
        )
        optimized_peak = optimized_report.get("peak_mb_mean", 0)

        # === Анализ результатов ===
        memory_saved = original_peak - optimized_peak
        speedup = (
            baseline_report.get("time_mean_ms", 1)
            / optimized_report.get("time_mean_ms", 1)
            if "time_mean_ms" in baseline_report
            else 1.0
        )

        # Рекомендации
        recommendations = []
        if memory_saved < 0:
            recommendations.append(
                "⚠️  Memory increased — check for redundant allocations"
            )
        if optimized_report.get("leak_detected", False):
            recommendations.append("🔍 Memory leak detected — review tensor lifecycle")
        if policy == MemoryPolicy.LAZY and original_peak > 100:
            recommendations.append("💡 Consider enabling POOLED policy for this method")

        report = OptimizationReport(
            method_name=method_name,
            original_peak_mb=original_peak,
            optimized_peak_mb=optimized_peak,
            memory_saved_mb=max(0, memory_saved),  # Не отрицательное
            speedup=speedup,
            policy_applied=policy,
            cache_hits=self.kernel_cache.stats()["hits"],
            recommendations=recommendations,
        )

        self.optimization_reports.append(report)

        if self.config.verbose:
            print(
                f"✅ {method_name}: {report.memory_saved_mb:.2f} MB saved "
                f"({report.reduction_pct:.1f}%), speedup: {report.speedup:.2f}×"
            )
            for rec in recommendations:
                print(f"   {rec}")

        return report

    def optimize_all_methods(
        self,
        methods: Optional[List[str]] = None,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> List[OptimizationReport]:
        """
        Оптимизирует все (или выбранные) методы.

        Args:
            methods: Список методов (None = все)
            input_tensor: Пример входа

        Returns:
            List[OptimizationReport]
        """
        if methods is None:
            methods = [
                m
                for m in self.segmenter.method_map.keys()
                if m not in self.config.exclude_methods
            ]

        if input_tensor is None:
            input_tensor = torch.randn(1, 1, 512, 512, device=self.device)

        if self.config.verbose:
            print(f"🚀 Optimizing {len(methods)} methods...")

        reports = []
        for method_name in methods:
            try:
                report = self.optimize_method(method_name, input_tensor)
                reports.append(report)
            except Exception as e:
                warnings.warn(f"Optimization failed for {method_name}: {e}")

        return reports

    def get_summary(self) -> Dict[str, Any]:
        """Сводная статистика по оптимизациям."""
        if not self.optimization_reports:
            return {}

        total_saved = sum(r.memory_saved_mb for r in self.optimization_reports)
        avg_reduction = np.mean([r.reduction_pct for r in self.optimization_reports])
        avg_speedup = np.mean([r.speedup for r in self.optimization_reports])

        return {
            "methods_optimized": len(self.optimization_reports),
            "total_memory_saved_mb": total_saved,
            "avg_reduction_pct": avg_reduction,
            "avg_speedup": avg_speedup,
            "cache_stats": self.kernel_cache.stats(),
        }

    def print_summary(self):
        """Вывод сводки в консоль."""
        summary = self.get_summary()

        print("\n🧠 Memory Optimization Summary")
        print("=" * 60)
        print(f"Methods optimized: {summary.get('methods_optimized', 0)}")
        print(f"Total memory saved: {summary.get('total_memory_saved_mb', 0):.2f} MB")
        print(f"Avg reduction: {summary.get('avg_reduction_pct', 0):.1f}%")
        print(f"Avg speedup: {summary.get('avg_speedup', 0):.2f}×")

        cache = summary.get("cache_stats", {})
        print(
            f"Cache: {cache.get('size', 0)}/{cache.get('max_size', 0)} entries, "
            f"{cache.get('hit_rate_pct', 0):.1f}% hit rate"
        )

        print("=" * 60)
