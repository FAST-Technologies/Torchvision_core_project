"""
Графовые оптимизации и анализ для компиляционного исследования.
"""

import torch
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import warnings

from .config import CompilationConfig, DEFAULT_CONFIG, CompilationStrategy
from .jit_compilation import JITOptimizer
from .torch_compile_benchmark import TorchCompileOptimizer


@dataclass
class CompilationReport:
    """Отчёт о компиляционной оптимизации."""

    method_name: str
    strategy: str
    original_time_ms: float
    compiled_time_ms: float
    compile_time_ms: float
    speedup: float
    graph_info: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)

    @property
    def speedup_formatted(self) -> str:
        if self.speedup >= 10:
            return f"{self.speedup:.1f}×"
        elif self.speedup >= 2:
            return f"{self.speedup:.2f}×"
        else:
            return f"{self.speedup:.3f}×"

    @property
    def is_worthwhile(self) -> bool:
        """Проверяет, стоит ли применять оптимизацию."""
        # Оптимизация имеет смысл если:
        # 1. Ускорение > 1.1×
        # 2. Время компиляции окупается за разумное число запусков
        return self.speedup > 1.1


class GraphOptimizer:
    """
    Комплексный оптимизатор графа вычислений.

    Объединяет:
    - torch.jit (script/trace)
    - torch.compile()
    - Graph analysis для принятия решений

    Пример:
        >>> optimizer = GraphOptimizer(segmenter)
        >>> report = optimizer.optimize_method("sobel_edge")
        >>> print(f"Speedup: {report.speedup_formatted}")
    """

    def __init__(
        self,
        segmenter: Any,
        config: Optional[CompilationConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            segmenter: Экземпляр сегментера
            config: Конфигурация
            device: Устройство
        """
        self.segmenter = segmenter
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(device or self.config.device)

        # Компоненты
        self.jit_optimizer = JITOptimizer(segmenter, config, device)
        self.compile_optimizer = TorchCompileOptimizer(segmenter, config, device)

        # Результаты
        self.reports: List[CompilationReport] = []

    def _get_strategy_for_method(self, method_name: str) -> CompilationStrategy:
        """Определяет стратегию компиляции для метода."""
        from .config import COMPILATION_STRATEGIES

        if method_name in self.config.exclude_methods:
            return CompilationStrategy.NONE

        if (
            self.config.include_methods
            and method_name not in self.config.include_methods
        ):
            return CompilationStrategy.NONE

        return COMPILATION_STRATEGIES.get(
            method_name, COMPILATION_STRATEGIES.get("default", self.config.strategy)
        )

    def optimize_method(
        self,
        method_name: str,
        example_input: Optional[torch.Tensor] = None,
    ) -> Optional[CompilationReport]:
        """
        Оптимизирует метод и возвращает отчёт.

        Args:
            method_name: Название метода
            example_input: Пример входа

        Returns:
            CompilationReport или None если оптимизация не применима
        """
        if method_name not in self.segmenter.method_map:
            raise ValueError(f"Method '{method_name}' not found")

        strategy = self._get_strategy_for_method(method_name)

        if strategy == CompilationStrategy.NONE:
            if self.config.verbose:
                print(f"   ✗ Compilation disabled for {method_name}")
            return None

        original_func = self.segmenter.method_map[method_name]

        # Подготовка входа
        if example_input is None:
            example_input = torch.randn(1, 1, 512, 512, device=self.device)

        # Анализ графа
        graph_info = None
        if self.config.verbose:
            from .utils import analyze_graph_structure

            graph_info = analyze_graph_structure(original_func, example_input)
            if self.config.verbose:
                print(
                    f"   📊 Graph: {graph_info['num_operations']} ops, "
                    f"optimization potential: {graph_info['optimization_potential']:.2f}"
                )

        # Замер оригинала
        start_time = time.perf_counter()
        with torch.inference_mode():
            for _ in range(self.config.warmup_runs):
                _ = original_func(example_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

        orig_times = []
        with torch.inference_mode():
            for _ in range(self.config.n_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = original_func(example_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                orig_times.append((time.perf_counter() - t0) * 1000)

        original_time = np.mean(orig_times)

        # Применение стратегии
        if strategy == CompilationStrategy.JIT_SCRIPT:
            compiled = self.jit_optimizer.compile_method(method_name, strategy="script")
            compile_time = (
                self.jit_optimizer.compiled_methods.get(method_name, {})
                .get("jit_script", {})
                .get("compile_time_s", 0)
                * 1000
            )

            compiled_time = self.jit_optimizer.benchmark(
                method_name, compiled, strategy="script", n_runs=self.config.n_runs
            )["compiled_mean_ms"]

        elif strategy == CompilationStrategy.JIT_TRACE:
            compiled = self.jit_optimizer.compile_method(method_name, strategy="trace")
            compile_time = (
                self.jit_optimizer.compiled_methods.get(method_name, {})
                .get("jit_trace", {})
                .get("compile_time_s", 0)
                * 1000
            )

            compiled_time = self.jit_optimizer.benchmark(
                method_name, compiled, strategy="trace", n_runs=self.config.n_runs
            )["compiled_mean_ms"]

        elif strategy == CompilationStrategy.TORCH_COMPILE:
            compiled = self.compile_optimizer.compile_method(
                method_name, mode=self.config.compile_mode
            )
            compile_time = (
                self.compile_optimizer.compiled_methods.get(method_name, {})
                .get(f"compile_{self.config.compile_mode}", {})
                .get("compile_time_s", 0)
                * 1000
            )

            compiled_time = self.compile_optimizer.benchmark(
                method_name,
                compiled,
                mode=self.config.compile_mode,
                n_runs=self.config.n_runs,
            )["compiled_mean_ms"]

        else:
            # Fallback
            compiled = original_func
            compile_time = 0
            compiled_time = original_time

        # Расчёт метрик
        speedup = original_time / compiled_time if compiled_time > 0 else float("inf")

        # Рекомендации
        recommendations = []
        if speedup < 1.1:
            recommendations.append("⚠️  Minimal speedup — consider skipping compilation")
        if compile_time > 1000:  # >1 sec compilation
            recommendations.append(
                "💡 Long compilation time — cache compiled model for reuse"
            )
        if strategy == CompilationStrategy.TORCH_COMPILE and speedup > 2.0:
            recommendations.append("✅ Excellent speedup — recommended for production")

        report = CompilationReport(
            method_name=method_name,
            strategy=strategy.name.lower(),
            original_time_ms=original_time,
            compiled_time_ms=compiled_time,
            compile_time_ms=compile_time,
            speedup=speedup,
            graph_info=graph_info,
            recommendations=recommendations,
        )

        self.reports.append(report)

        if self.config.verbose:
            print(
                f"✅ {method_name}: {report.speedup_formatted} speedup "
                f"(compile: {compile_time:.1f} ms)"
            )
            for rec in recommendations:
                print(f"   {rec}")

        return report

    def optimize_all_methods(
        self,
        methods: Optional[List[str]] = None,
        example_input: Optional[torch.Tensor] = None,
    ) -> List[CompilationReport]:
        """
        Оптимизирует все (или выбранные) методы.

        Args:
            methods: Список методов
            example_input: Пример входа

        Returns:
            List[CompilationReport]
        """
        if methods is None:
            methods = [
                m
                for m in self.segmenter.method_map.keys()
                if m not in self.config.exclude_methods
            ]

        if self.config.verbose:
            print(f"🚀 Optimizing {len(methods)} methods...")

        reports = []
        for method_name in methods:
            try:
                report = self.optimize_method(method_name, example_input)
                if report is not None:
                    reports.append(report)
            except Exception as e:
                warnings.warn(f"Optimization failed for {method_name}: {e}")

        return reports

    def get_summary(self) -> Dict[str, Any]:
        """Сводная статистика по оптимизациям."""
        if not self.reports:
            return {}

        valid_reports = [r for r in self.reports if r.is_worthwhile]

        return {
            "methods_optimized": len(self.reports),
            "methods_worthwhile": len(valid_reports),
            "avg_speedup": np.mean([r.speedup for r in self.reports]),
            "max_speedup": max(r.speedup for r in self.reports),
            "avg_compile_time_ms": np.mean([r.compile_time_ms for r in self.reports]),
            "total_time_saved_ms": sum(
                (r.original_time_ms - r.compiled_time_ms) * 100  # Условные 100 запусков
                for r in self.reports
            ),
        }

    def print_summary(self):
        """Вывод сводки в консоль."""
        summary = self.get_summary()

        print("\n🔧 Compilation Optimization Summary")
        print("=" * 60)
        print(f"Methods optimized: {summary.get('methods_optimized', 0)}")
        print(f"Methods worthwhile: {summary.get('methods_worthwhile', 0)}")
        print(f"Avg speedup: {summary.get('avg_speedup', 0):.2f}×")
        print(f"Max speedup: {summary.get('max_speedup', 0):.2f}×")
        print(f"Avg compile time: {summary.get('avg_compile_time_ms', 0):.1f} ms")
        print(
            f"Est. time saved (100 runs): {summary.get('total_time_saved_ms', 0):.1f} ms"
        )
        print("=" * 60)
