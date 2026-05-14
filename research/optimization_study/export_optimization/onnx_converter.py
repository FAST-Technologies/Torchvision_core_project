# research/optimization_study/02_tensorrt_conversion/onnx_converter.py

"""
ONNX экспорт и бенчмаркинг для TorchSegmenter методов.
"""
import torch
import onnx
import onnxruntime as ort
from onnxsim import simplify
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable, Any
import time
import warnings

from .utils import (
    get_available_providers,
    warmup_inference,
    measure_inference,
    format_time,
    format_speedup,
)


class ONNXOptimizer:
    """
    Конвертация методов TorchSegmenter в ONNX формат.

    Поддерживает:
    - Экспорт с dynamic axes
    - Упрощение модели через onnx-simplifier
    - Бенчмаркинг на CPU/CUDA/TensorRT EP
    - Авто-детект доступных провайдеров

    Пример:
        optimizer = ONNXOptimizer(segmenter)
        path = optimizer.export_method_to_onnx("sobel_edge", "sobel.onnx")
        results = optimizer.benchmark_onnx_vs_torch("sobel_edge", path)
    """

    SUPPORTED_PROVIDERS = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

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
            device: Устройство для экспорта ('cuda' или 'cpu')
        """
        self.segmenter = segmenter
        self.image_shape = image_shape
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.onnx_models: Dict[str, str] = {}

        # Авто-детект провайдеров
        self.available_providers = [p for p in self.SUPPORTED_PROVIDERS if p in get_available_providers()]
        if not self.available_providers:
            warnings.warn("⚠️ No ONNX Runtime providers detected, using CPU")
            self.available_providers = ["CPUExecutionProvider"]

    def export_method_to_onnx(
        self,
        method_name: str,
        output_path: str,
        opset_version: int = 17,
        simplify_model: bool = True,
        dynamic_axes: Optional[Dict] = None,
        do_constant_folding: bool = True,
        verbose: bool = False,
    ) -> str:
        """
        Экспорт одного метода в ONNX формат.

        Args:
            method_name: Название метода из segmenter.method_map
            output_path: Путь для сохранения .onnx файла
            opset_version: Версия ONNX opset (рекомендуется 17+)
            simplify_model: Применить onnx-simplifier после экспорта
            dynamic_axes: Динамические оси для батч-инференса
            do_constant_folding: Оптимизация констант при экспорте
            verbose: Выводить детали экспорта

        Returns:
            str: Путь к сохранённому файлу

        Raises:
            ValueError: Если метод не найден
            RuntimeError: Если экспорт не удался
        """
        if method_name not in self.segmenter.method_map:
            raise ValueError(
                f"Method '{method_name}' not found in segmenter. "
                f"Available: {list(self.segmenter.method_map.keys())}"
            )

        original_func = self.segmenter.method_map[method_name]

        # Создаём пример входа
        dummy_input = torch.randn(1, *self.image_shape, device=self.device, dtype=torch.float32)

        # Настройки dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            }

        try:
            # Экспорт в ONNX
            torch.onnx.export(
                original_func,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                verbose=verbose,
            )

            # Опциональное упрощение модели
            if simplify_model:
                try:
                    onnx_model = onnx.load(output_path)
                    model_simp, check = simplify(onnx_model)
                    if check:
                        onnx.save(model_simp, output_path)
                        if verbose:
                            print(f"✅ Модель упрощена: {output_path}")
                except Exception as e:
                    warnings.warn(f"⚠️ onnx-simplifier failed: {e}")

            self.onnx_models[method_name] = output_path

            if verbose:
                # Проверка модели
                onnx.checker.check_model(onnx.load(output_path))
                print(f"✅ Экспорт успешен: {output_path}")
                print(f"   Input shape: {self.image_shape}")
                print(f"   Opset: {opset_version}")
                print(f"   Providers: {self.available_providers}")

            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to export '{method_name}' to ONNX: {e}") from e

    def create_session(
        self,
        onnx_path: str,
        provider: Optional[str] = None,
        enable_profiling: bool = False,
    ) -> ort.InferenceSession:
        """
        Создать ONNX Runtime сессию.

        Args:
            onnx_path: Путь к .onnx файлу
            provider: Провайдер ('CUDAExecutionProvider', etc.)
            enable_profiling: Включить профилирование

        Returns:
            ort.InferenceSession
        """
        # Выбор провайдера
        if provider is None:
            # Авто-выбор: TensorRT > CUDA > CPU
            for p in self.available_providers:
                if p in self.SUPPORTED_PROVIDERS:
                    provider = p
                    break

        if provider not in self.available_providers:
            warnings.warn(f"Provider '{provider}' not available, falling back to CPU")
            provider = "CPUExecutionProvider"

        # Настройки сессии
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if enable_profiling:
            sess_options.enable_profiling = True

        # Создание сессии
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=[provider],
        )

        return session

    def benchmark_onnx_vs_torch(
        self,
        method_name: str,
        onnx_path: str,
        n_runs: int = 100,
        n_warmup: int = 10,
        provider: Optional[str] = None,
        sync_cuda: bool = True,
    ) -> Dict[str, float]:
        """
        Сравнение скорости оригинала и ONNX-версии.

        Args:
            method_name: Название метода
            onnx_path: Путь к .onnx файлу
            n_runs: Количество запусков для замера
            n_warmup: Количество прогревочных запусков
            provider: ONNX Runtime провайдер
            sync_cuda: Синхронизировать CUDA для точных замеров

        Returns:
            Dict со статистикой:
            {
                "torch_mean_ms": ...,
                "onnx_mean_ms": ...,
                "speedup": ...,
                "onnx_provider": ...,
            }
        """

        # Оригинальная версия
        original_func = self.segmenter.method_map[method_name]

        # Входной тензор
        dummy_input = torch.randn(1, *self.image_shape, device=self.device, dtype=torch.float32)

        # === Замер оригинала (Torch) ===
        warmup_inference(original_func, dummy_input, n_warmup)
        torch_stats = measure_inference(original_func, dummy_input, n_runs, sync_cuda)

        # === Замер ONNX ===
        session = self.create_session(onnx_path, provider)
        input_name = session.get_inputs()[0].name

        # Конвертация входа для ONNX
        input_np = dummy_input.cpu().numpy()

        def onnx_func(_):
            return session.run(None, {input_name: input_np})

        warmup_inference(onnx_func, dummy_input, n_warmup)
        onnx_stats = measure_inference(onnx_func, dummy_input, n_runs, sync_cuda=False)

        # === Результаты ===
        speedup = torch_stats["mean_ms"] / onnx_stats["mean_ms"]

        return {
            # Torch статистика
            "torch_mean_ms": torch_stats["mean_ms"],
            "torch_std_ms": torch_stats["std_ms"],
            "torch_p95_ms": torch_stats["p95_ms"],
            # ONNX статистика
            "onnx_mean_ms": onnx_stats["mean_ms"],
            "onnx_std_ms": onnx_stats["std_ms"],
            "onnx_p95_ms": onnx_stats["p95_ms"],
            "onnx_provider": session.get_providers()[0],
            # Сравнение
            "speedup": speedup,
            "speedup_formatted": format_speedup(speedup),
        }

    def benchmark_all_providers(
        self,
        method_name: str,
        onnx_path: str,
        n_runs: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """
        Бенчмарк на всех доступных провайдерах.

        Returns:
            Dict: {provider_name: benchmark_results}
        """
        results = {}

        for provider in self.available_providers:
            try:
                results[provider] = self.benchmark_onnx_vs_torch(
                    method_name, onnx_path, n_runs=n_runs, provider=provider
                )
            except Exception as e:
                warnings.warn(f"⚠️ {provider} failed: {e}")
                results[provider] = {"error": str(e)}

        return results
