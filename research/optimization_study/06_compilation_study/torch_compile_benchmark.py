# research/optimization_study/06_compilation_study/torch_compile_benchmark.py
"""
Бенчмаркинг оптимизации через torch.compile() (PyTorch 2.0+).
"""
import torch
import time
import numpy as np
from typing import Dict, Callable, List, Optional, Any
import warnings

from .config import CompilationConfig, DEFAULT_CONFIG, CompilationStrategy
from .utils import (
    get_compilation_capabilities,
    format_speedup,
    analyze_graph_structure,
)


class TorchCompileOptimizer:
    """
    Оптимизатор через torch.compile() (PyTorch 2.0+).

    Поддерживает:
    - Разные режимы компиляции (default, reduce-overhead, max-autotune)
    - Выбор бэкенда (inductor, cudagraphs, aot_eager)
    - Оптимизации графа (fullgraph, dynamic shapes)
    - Профилирование времени компиляции и инференса

    Пример:
        >>> optimizer = TorchCompileOptimizer(segmenter)
        >>> compiled = optimizer.compile_method("sobel_edge", mode="reduce-overhead")
        >>> result = optimizer.benchmark("sobel_edge", compiled)
    """

    MODES = {
        "default": "balance",  # Баланс скорости компиляции/инференса
        "reduce-overhead": "reduce-overhead",  # Минимизация overhead запуска
        "max-autotune": "max-autotune",  # Максимальная оптимизация (долгая компиляция)
        "inductor": "inductor",  # Базовый бэкенд
        "cudagraphs": "cudagraphs",  # Графы CUDA для минимизации overhead
    }

    BACKENDS = ["inductor", "cudagraphs", "aot_eager", "onnxrt"]

    def __init__(
        self,
        segmenter: Any,
        config: Optional[CompilationConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            segmenter: Экземпляр сегментера
            config: Конфигурация компиляции
            device: Устройство для вычислений
        """
        self.segmenter = segmenter
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(device or self.config.device)

        # Проверка доступности torch.compile
        if not hasattr(torch, "compile") or torch.__version__ < "2.0":
            warnings.warn(
                "torch.compile() requires PyTorch >= 2.0. "
                f"Current version: {torch.__version__}"
            )
            self._compile_available = False
        else:
            self._compile_available = True

        # Кэш скомпилированных функций
        self.compiled_methods: Dict[str, Dict[str, Any]] = {}

        # Возможности устройства
        self.capabilities = get_compilation_capabilities(str(self.device.type))

        if self.config.verbose:
            print(f"🔧 TorchCompileOptimizer initialized:")
            print(f"   Device: {self.device}")
            print(f"   Torch version: {torch.__version__}")
            print(f"   torch.compile available: {self._compile_available}")
            print(
                f"   Available backends: {[b for b in self.BACKENDS if self.capabilities.get(f'{b}_available', True)]}"
            )

    def compile_method(
        self,
        method_name: str,
        mode: str = "reduce-overhead",
        backend: Optional[str] = None,
        fullgraph: Optional[bool] = None,  # True
        dynamic: Optional[bool] = None,  # False
        example_input: Optional[torch.Tensor] = None,
    ) -> Optional[Callable]:
        """
        Компилирует метод через torch.compile().

        Args:
            method_name: Название метода
            mode: Режим компиляции
            backend: Бэкенд компиляции
            fullgraph: Компилировать весь граф
            dynamic: Поддерживать динамические shapes
            example_input: Пример входа для "прогрева"

        Returns:
            Скомпилированная функция или None при ошибке
        """

        if not self._compile_available:
            warnings.warn("torch.compile not available. Using original function.")
            return self.segmenter.method_map[method_name]

        if method_name not in self.segmenter.method_map:
            raise ValueError(f"Method '{method_name}' not found")

        original_func = self.segmenter.method_map[method_name]

        # Параметры по умолчанию из config
        if backend is None:
            backend = self.config.backend
        if fullgraph is None:
            fullgraph = self.config.fullgraph
        if dynamic is None:
            dynamic = self.config.dynamic

        # Подготовка примера входа
        if example_input is None:
            example_input = torch.randn(1, 1, 512, 512, device=self.device)

        # Компиляция
        start_time = time.perf_counter()

        # Компиляция
        try:
            compiled_func = torch.compile(
                original_func,
                mode=self.MODES.get(mode, mode),
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend if backend in self.BACKENDS else "inductor",
            )

            # "Прогрев" компиляции — первый запуск включает JIT-компиляцию
            with torch.inference_mode():
                _ = compiled_func(example_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

            compile_time = time.perf_counter() - start_time

            # Кэширование
            if method_name not in self.compiled_methods:
                self.compiled_methods[method_name] = {}
            self.compiled_methods[method_name][f"compile_{mode}"] = {
                "func": compiled_func,
                "compile_time_s": compile_time,
                "mode": mode,
                "backend": backend,
            }

            if self.config.verbose:
                print(
                    f"✅ Compiled '{method_name}' via torch.compile "
                    f"(mode={mode}, backend={backend}, {compile_time*1000:.1f} ms)"
                )

            return compiled_func

        except Exception as e:
            warnings.warn(
                f"torch.compile failed for '{method_name}': {e}. "
                f"Using original function."
            )
            return

    def benchmark(
        self,
        method_name: str,
        compiled_func: Optional[Callable] = None,
        mode: str = "reduce-overhead",
        n_runs: Optional[int] = None,
        warmup: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Бенчмарк torch.compile-оптимизированной функции.

        Args:
            method_name: Название метода
            compiled_func: Скомпилированная функция
            mode: Режим компиляции
            n_runs: Количество запусков
            warmup: Прогревочные запуски

        Returns:
            Dict с метриками производительности
        """
        n_runs = n_runs or self.config.n_runs
        warmup = warmup or self.config.warmup_runs

        if compiled_func is None:
            compiled_func = self.compile_method(method_name, mode=mode)
            if compiled_func is None:
                return {"error": "Compilation failed"}

        original_func = self.segmenter.method_map[method_name]
        example_input = torch.randn(1, 1, 512, 512, device=self.device)

        def measure(func, inp, n, w):
            # Прогрев
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

        # Замер compiled-версии
        comp_mean, comp_std = measure(compiled_func, example_input, n_runs, warmup)

        speedup = orig_mean / comp_mean if comp_mean > 0 else float("inf")

        # Получаем время компиляции из кэша
        compile_info = self.compiled_methods.get(method_name, {}).get(
            f"compile_{mode}", {}
        )

        return {
            "method": method_name,
            "strategy": f"compile_{mode}",
            "original_mean_ms": orig_mean,
            "original_std_ms": orig_std,
            "compiled_mean_ms": comp_mean,
            "compiled_std_ms": comp_std,
            "speedup": speedup,
            "speedup_formatted": format_speedup(speedup),
            "compile_time_ms": compile_info.get("compile_time_s", 0) * 1000,
            "backend": compile_info.get("backend", "inductor"),
        }

    def benchmark_modes(
        self,
        method_name: str,
        modes: Optional[List[str]] = None,
        n_runs: int = 30,  # Меньше запусков т.к. компиляция долгая
    ) -> Dict[str, Dict[str, float]]:
        """
        Сравнение разных режимов компиляции.

        Args:
            method_name: Название метода
            modes: Список режимов для тестирования
            n_runs: Запусков на режим

        Returns:
            Dict {mode: benchmark_results}
        """
        if modes is None:
            modes = ["default", "reduce-overhead", "max-autotune"]

        results = {}

        for mode in modes:
            if self.config.verbose:
                print(f"   • Testing mode: {mode}...", end=" ")

            try:
                result = self.benchmark(method_name, mode=mode, n_runs=n_runs)
                results[mode] = result

                if self.config.verbose:
                    print(f"✓ {result['speedup_formatted']} speedup")

            except Exception as e:
                if self.config.verbose:
                    print(f"✗ {e}")
                warnings.warn(f"Benchmark failed for {method_name}/{mode}: {e}")
                results[mode] = {"error": str(e)}

        return results

    def benchmark_compile_modes(
        self, method_name: str, modes: List[str] = None, n_runs: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """Сравнение разных режимов компиляции"""

        if modes is None:
            modes = ["default", "reduce-overhead", "max-autotune"]

        results = {}
        original_func = self.segmenter.methods[method_name]
        example_input = torch.randn(1, 3, 512, 512, device=self.device)

        # Замер оригинала
        times_orig = self._measure_func(original_func, example_input, n_runs)
        results["original"] = {
            "mean_ms": np.mean(times_orig),
            "std_ms": np.std(times_orig),
            "min_ms": np.min(times_orig),
            "max_ms": np.max(times_orig),
        }

        # Замер компилированных версий
        for mode in modes:
            try:
                compiled = self.compile_method(method_name, mode=mode)
                times = self._measure_func(compiled, example_input, n_runs)

                results[mode] = {
                    "mean_ms": np.mean(times),
                    "std_ms": np.std(times),
                    "min_ms": np.min(times),
                    "max_ms": np.max(times),
                    "speedup": results["original"]["mean_ms"] / np.mean(times),
                }
            except Exception as e:
                print(f"⚠️  {method_name}/{mode}: {e}")
                results[mode] = {"error": str(e)}

        return results

    @staticmethod
    def _measure_func(func: Callable, inp: torch.Tensor, n: int) -> List[float]:
        times = []
        with torch.inference_mode():
            for _ in range(n):
                start = time.perf_counter()
                _ = func(inp)
                if inp.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        return times
