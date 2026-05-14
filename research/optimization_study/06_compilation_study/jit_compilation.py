# research/optimization_study/06_compilation_study/jit_compilation.py
"""
Компиляция через torch.jit для стабильных графов.
"""
import torch
from typing import Dict, Callable, Optional, Any, Tuple
import time
import warnings

from .config import CompilationConfig, DEFAULT_CONFIG
from .utils import (
    get_compilation_capabilities,
    is_graph_stable,
    format_speedup,
)


class JITCompiler:
    """
    Оптимизатор через torch.jit (script/trace/freeze).

    Поддерживает:
    - torch.jit.script — компиляция через AST-анализ
    - torch.jit.trace — компиляция через трассировку
    - torch.jit.freeze — фиксация графа для инференса
    - torch.jit.optimize_for_inference — оптимизации для продакшена

    Пример:
        >>> optimizer = JITOptimizer(segmenter)
        >>> compiled = optimizer.compile_method("sobel_edge", strategy="script")
        >>> result = optimizer.benchmark("sobel_edge", compiled)
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
            config: Конфигурация компиляции
            device: Устройство для вычислений
        """
        self.segmenter = segmenter
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(device or self.config.device)

        # Кэш скомпилированных функций
        self.compiled_methods: Dict[str, Dict[str, Any]] = {}

        # Возможности устройства
        self.capabilities = get_compilation_capabilities(str(self.device.type))

        if self.config.verbose:
            print(f"🔧 JITOptimizer initialized:")
            print(f"   Device: {self.device}")
            print(f"   Torch version: {self.capabilities['torch_version']}")
            print(f"   JIT available: {self.capabilities['jit_available']}")

    def _compile_script(
        self,
        func: Callable,
        example_input: torch.Tensor,
        optimize: bool = True,
    ) -> Any:
        """
        Компиляция через torch.jit.script.

        Args:
            func: Функция для компиляции
            example_input: Пример входа
            optimize: Применить оптимизации после компиляции

        Returns:
            Скомпилированная функция
        """
        try:
            # Компиляция через script
            compiled = torch.jit.script(func)

            # Оптимизации если нужно
            if optimize and self.config.freeze_graph:
                compiled = torch.jit.freeze(compiled)
                compiled = torch.jit.optimize_for_inference(compiled)

            return compiled

        except RuntimeError as e:
            warnings.warn(f"jit.script failed: {e}. Falling back to trace.")
            return self._compile_trace(func, example_input, optimize)

    def _compile_trace(
        self,
        func: Callable,
        example_input: torch.Tensor,
        optimize: bool = True,
        check_trace: bool = True,
    ) -> Any:
        """
        Компиляция через torch.jit.trace.

        Args:
            func: Функция для компиляции
            example_input: Пример входа
            optimize: Применить оптимизации
            check_trace: Проверять стабильность трассировки

        Returns:
            Скомпилированная функция
        """
        try:
            # Трассировка
            compiled = torch.jit.trace(
                func,
                example_input,
                check_trace=check_trace,
                strict=False,
            )

            # Оптимизации если нужно
            if optimize and self.config.freeze_graph:
                compiled = torch.jit.freeze(compiled)
                compiled = torch.jit.optimize_for_inference(compiled)

            return compiled

        except Exception as e:
            warnings.warn(f"jit.trace failed: {e}. Returning original function.")
            return func

    def compile_method(
        self,
        method_name: str,
        strategy: str = "script",  # "script" или "trace"
        example_input: Optional[torch.Tensor] = None,
    ) -> Optional[Callable]:
        """
        Компилирует указанный метод через torch.jit.

        Args:
            method_name: Название метода
            strategy: Стратегия ("script" или "trace")
            example_input: Пример входа для трассировки

        Returns:
            Скомпилированная функция или None при ошибке
        """
        if method_name not in self.segmenter.method_map:
            raise ValueError(f"Method '{method_name}' not found")

        if not self.capabilities["jit_available"]:
            warnings.warn("torch.jit not available. Using original function.")
            return self.segmenter.method_map[method_name]

        original_func = self.segmenter.method_map[method_name]

        # Подготовка примера входа
        if example_input is None:
            example_input = torch.randn(1, 1, 512, 512, device=self.device)

        # Компиляция
        start_time = time.perf_counter()

        if strategy == "script":
            compiled = self._compile_script(original_func, example_input, optimize=self.config.jit_optimize)
        elif strategy == "trace":
            compiled = self._compile_trace(original_func, example_input, optimize=self.config.jit_optimize)
        else:
            raise ValueError(f"Unknown JIT strategy: {strategy}")

        compile_time = time.perf_counter() - start_time

        # Кэширование
        if method_name not in self.compiled_methods:
            self.compiled_methods[method_name] = {}
        self.compiled_methods[method_name][f"jit_{strategy}"] = {
            "func": compiled,
            "compile_time_s": compile_time,
            "strategy": strategy,
        }
        if self.config.verbose:
            print(f"✅ Compiled '{method_name}' via jit.{strategy} " f"({compile_time*1000:.1f} ms)")

        return compiled

    def benchmark(
        self,
        method_name: str,
        compiled_func: Optional[Callable] = None,
        strategy: str = "script",
        n_runs: Optional[int] = None,
        warmup: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Бенчмарк JIT-компилированной функции.

        Args:
            method_name: Название метода
            compiled_func: Скомпилированная функция (если None — компилируем)
            strategy: Стратегия компиляции
            n_runs: Количество запусков
            warmup: Прогревочные запуски

        Returns:
            Dict с метриками производительности
        """
        n_runs = n_runs or self.config.n_runs
        warmup = warmup or self.config.warmup_runs

        if compiled_func is None:
            compiled_func = self.compile_method(method_name, strategy=strategy)
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

        # Замер JIT-версии
        jit_mean, jit_std = measure(compiled_func, example_input, n_runs, warmup)

        speedup = orig_mean / jit_mean if jit_mean > 0 else float("inf")

        return {
            "method": method_name,
            "strategy": f"jit_{strategy}",
            "original_mean_ms": orig_mean,
            "original_std_ms": orig_std,
            "compiled_mean_ms": jit_mean,
            "compiled_std_ms": jit_std,
            "speedup": speedup,
            "speedup_formatted": format_speedup(speedup),
        }

    @staticmethod
    def script_method(func: Callable, example_input: torch.Tensor, optimize: bool = True) -> torch.jit.ScriptFunction:
        """Компиляция через torch.jit.script"""

        # Для простых функций можно использовать script напрямую
        try:
            scripted = torch.jit.script(func)
        except RuntimeError:
            # Если script не работает, пробуем trace
            scripted = torch.jit.trace(func, example_input)

        if optimize:
            # Применяем оптимизации
            scripted = torch.jit.freeze(scripted)
            scripted = torch.jit.optimize_for_inference(scripted)

        return scripted

    @staticmethod
    def trace_method(func: Callable, example_input: torch.Tensor, check_trace: bool = True) -> torch.jit.TracedModule:
        """Компиляция через torch.jit.trace"""

        traced = torch.jit.trace(func, example_input, check_trace=check_trace, strict=False)

        return torch.jit.freeze(traced)

    @staticmethod
    def save_and_load(jit_module, path: str):
        """Сохранение и загрузка JIT-модуля"""
        jit_module.save(path)
        return torch.jit.load(path)
