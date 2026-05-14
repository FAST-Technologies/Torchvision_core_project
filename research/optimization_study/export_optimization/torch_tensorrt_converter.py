"""
torch-tensorrt конвертер для TorchSegmenter методов.

Использует официальный torch-tensorrt от NVIDIA.
Требует: tensorrt>=10.0, torch-tensorrt>=2.0
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import time
import warnings

from .utils import (
    warmup_inference,
    measure_inference,
    format_time,
    format_speedup,
)


class TorchTRTOptimizer:
    """
    Оптимизация через torch-tensorrt (официальный NVIDIA бэкенд).

    Преимущества:
    - Прямая интеграция с PyTorch
    - Поддержка FP16/INT8 квантования
    - Автоматическая оптимизация графа

    Пример:
        optimizer = TorchTRTOptimizer(segmenter)
        trt_func = optimizer.convert_method("sobel_edge", precision="fp16")
        results = optimizer.benchmark("sobel_edge", trt_func)
    """

    PRECISIONS = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        # INT8 требует калибровки — не включаем по умолчанию
    }

    def __init__(
        self,
        segmenter: Any,
        image_shape: Tuple[int, int, int] = (3, 512, 512),
        device: str = "cuda",
    ):
        """
        Args:
            segmenter: Экземпляр TorchSegmenter
            image_shape: (C, H, W) входного тензора
            device: Устройство (должен быть 'cuda')
        """
        if not torch.cuda.is_available():
            raise RuntimeError("torch-tensorrt requires CUDA")

        try:
            import torch_tensorrt
        except ImportError:
            raise ImportError("torch-tensorrt not installed. " "Install via: pip install torch-tensorrt")

        self.segmenter = segmenter
        self.image_shape = image_shape
        self.device = torch.device(device)
        self.compiled_methods: Dict[str, Any] = {}

        import torch_tensorrt

        self.trt = torch_tensorrt

    def convert_method(
        self,
        method_name: str,
        precision: str = "fp16",
        min_block_size: int = 5,
        max_block_size: int = -1,
        enabled_precisions: Optional[set] = None,
        verbose: bool = False,
    ):
        """
        Конвертация метода в torch-tensorrt engine.

        Args:
            method_name: Название метода из segmenter.method_map
            precision: Точность ('fp32', 'fp16')
            min_block_size: Мин. размер блока для TRT оптимизации
            max_block_size: Макс. размер (-1 = без ограничений)
            enabled_precisions: Явное указание точностей
            verbose: Выводить логи компиляции

        Returns:
            Callable: Оптимизированная функция
        """
        if method_name not in self.segmenter.method_map:
            raise ValueError(
                f"Method '{method_name}' not found. " f"Available: {list(self.segmenter.method_map.keys())}"
            )

        original_func = self.segmenter.method_map[method_name]

        # Пример входа
        example_input = torch.randn(1, *self.image_shape, device=self.device, dtype=torch.float32)

        # Настройки точности
        if enabled_precisions is None:
            target_dtype = self.PRECISIONS.get(precision, torch.float32)
            enabled_precisions = {target_dtype}

        # Компиляция
        try:
            compiled_func = self.trt.compile(
                original_func,
                inputs=[example_input],
                enabled_precisions=enabled_precisions,
                min_block_size=min_block_size,
                max_block_size=max_block_size,
                ir="torch",  # или "ts" для TorchScript
                debug=verbose,
            )

            self.compiled_methods[method_name] = compiled_func

            if verbose:
                print(f"✅ Compiled '{method_name}' with {precision}")

            return compiled_func

        except Exception as e:
            warnings.warn(
                f"⚠️ torch-tensorrt compilation failed for '{method_name}': {e}. " f"Falling back to original function."
            )
            return original_func

    def benchmark(
        self,
        method_name: str,
        compiled_func: Optional[Any] = None,
        n_runs: int = 100,
        n_warmup: int = 10,
        precision: str = "fp16",
    ) -> Dict[str, float]:
        """
        Бенчмарк оптимизированной функции.

        Args:
            method_name: Название метода
            compiled_func: Оптимизированная функция (если None — компилируем)
            n_runs: Количество запусков
            n_warmup: Прогревочные запуски
            precision: Точность для компиляции (если нужно)

        Returns:
            Dict со статистикой и speedup
        """
        original_func = self.segmenter.method_map[method_name]

        if compiled_func is None:
            compiled_func = self.convert_method(method_name, precision=precision)

        # Входной тензор
        dummy_input = torch.randn(1, *self.image_shape, device=self.device, dtype=torch.float32)

        # Замер оригинала
        warmup_inference(original_func, dummy_input, n_warmup)
        orig_stats = measure_inference(original_func, dummy_input, n_runs)

        # Замер TRT-версии
        warmup_inference(compiled_func, dummy_input, n_warmup)
        trt_stats = measure_inference(compiled_func, dummy_input, n_runs)

        # Результаты
        speedup = orig_stats["mean_ms"] / trt_stats["mean_ms"]

        return {
            "original_mean_ms": orig_stats["mean_ms"],
            "original_std_ms": orig_stats["std_ms"],
            "trt_mean_ms": trt_stats["mean_ms"],
            "trt_std_ms": trt_stats["std_ms"],
            "speedup": speedup,
            "speedup_formatted": format_speedup(speedup),
            "precision": precision,
        }

    def benchmark_precision_sweep(
        self,
        method_name: str,
        precisions: List[str] = None,
        n_runs: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """
        Сравнение разных режимов точности.

        Returns:
            Dict: {precision: benchmark_results}
        """
        if precisions is None:
            precisions = ["fp32", "fp16"]

        results = {}

        for precision in precisions:
            try:
                compiled = self.convert_method(method_name, precision=precision)
                results[precision] = self.benchmark(method_name, compiled, n_runs=n_runs, precision=precision)
            except Exception as e:
                warnings.warn(f"⚠️ {precision} failed: {e}")
                results[precision] = {"error": str(e)}

        return results
