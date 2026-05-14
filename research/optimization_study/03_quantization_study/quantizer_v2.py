"""
Основной класс для квантования методов сегментации.
"""

import torch
from torch.ao.quantization import (
    QuantStub,
    DeQuantStub,
    prepare,
    convert,
    fuse_modules,
    QConfig,
    default_qconfig,
    default_dynamic_qconfig,
    get_default_qconfig,
)
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import time
import warnings

from .config import QuantizationConfig, DEFAULT_CONFIG, QUANTIZATION_SCHEMES
from .utils import (
    get_available_quantization_backends,
    compute_quantization_error,
    safe_quantize_tensor,
    is_method_quantizable,
    estimate_model_size,
)


class CalibrationDataset:
    """
    Датасет для калибровки квантованных моделей.

    Используется для сбора статистики активаций при статическом квантовании.
    """

    def __init__(
        self,
        images: List[np.ndarray],
        segmenter: Any,
        method_name: str,
        batch_size: int = 1,
    ):
        self.images = images
        self.segmenter = segmenter
        self.method_name = method_name
        self.batch_size = batch_size
        self.index = 0

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self.index >= len(self.images):
            raise StopIteration

        # Получаем батч изображений
        batch = []
        for _ in range(self.batch_size):
            if self.index >= len(self.images):
                break
            img = self.images[self.index]
            tensor = self.segmenter.preprocess_image(img)
            batch.append(tensor)
            self.index += 1

        if len(batch) == 1:
            return batch[0]
        return torch.stack(batch, dim=0)


class QuantizationWrapper(torch.nn.Module):
    """
    Обёртка для квантования произвольной функции сегментации.

    Добавляет QuantStub/DeQuantStub для входа/выхода
    и поддерживает слияние модулей для оптимизации.
    """

    def __init__(
        self,
        func: Callable,
        input_shape: Tuple[int, ...] = (1, 3, 512, 512),
        fuse_patterns: Optional[List[Tuple[str, ...]]] = None,
    ):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.func = func
        self.input_shape = input_shape
        self.fuse_patterns = fuse_patterns or []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Квантование входа
        x = self.quant(x)

        # Вызов оригинальной функции
        # Важно: функция должна работать с квантованными тензорами
        # или автоматически де-квантовать при необходимости
        try:
            result = self.func(x)
        except Exception as e:
            # Fallback: де-квантовать, выполнить, квантовать результат
            warnings.warn(f"Quantized execution failed: {e}. Using fallback.")
            x_f32 = self.dequant(x)
            result = self.func(x_f32)
            result = self.quant(result) if isinstance(result, torch.Tensor) else result

        # Де-квантование выхода
        if isinstance(result, torch.Tensor):
            result = self.dequant(result)

        return result

    def fuse(self) -> "QuantizationWrapper":
        """Слияние модулей для оптимизации."""
        if self.fuse_patterns:
            try:
                fused = fuse_modules(self, self.fuse_patterns, inplace=True)
                return fused
            except Exception as e:
                warnings.warn(f"Module fusion failed: {e}")
        return self


class QuantizedSegmenter:
    """
    Квантованная версия TorchSegmenter для бенчмаркинга.

    Поддерживает:
    - Динамическое квантование (без калибровки)
    - Статическое квантование (с калибровкой)
    - Полуточность (FP16)
    - Сравнение точности и скорости

    Пример:
        >>> segmenter = TorchSegmenter(method="sobel_edge")
        >>> quantizer = QuantizedSegmenter(segmenter)
        >>>
        >>> # Динамическое квантование (быстро, без калибровки)
        >>> quantized = quantizer.quantize_method("sobel_edge", scheme="int8_dynamic")
        >>>
        >>> # Статическое квантование (требует калибровки)
        >>> quantizer.calibrate("sobel_edge", calibration_images, steps=100)
        >>>
        >>> # Бенчмарк
        >>> results = quantizer.benchmark("sobel_edge", test_image)
        >>> print(f"Speedup: {results['speedup']:.2f}×")
    """

    def __init__(
        self,
        original_segmenter: Any,
        config: Optional[QuantizationConfig] = None,
    ):
        """
        Args:
            original_segmenter: Экземпляр TorchSegmenter
            config: Конфигурация квантования
        """
        self.original = original_segmenter
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(self.config.target_device)

        # Кэш квантованных методов
        self.quantized_methods: Dict[str, Dict[QUANTIZATION_SCHEMES, Any]] = {}

        # Статистика
        self.benchmark_results: List[Dict] = []

        if self.config.verbose:
            print(f"🔧 QuantizedSegmenter initialized:")
            print(f"   Device: {self.device}")
            print(f"   Available backends: {get_available_quantization_backends()}")

    def _get_qconfig(self, scheme: QUANTIZATION_SCHEMES) -> Optional[QConfig]:
        """Получает конфигурацию квантования для схемы."""
        if scheme == "fp32":
            return None
        elif scheme == "fp16":
            return None  # FP16 не использует QConfig
        elif scheme == "int8_dynamic":
            return default_dynamic_qconfig
        elif scheme == "int8_static":
            backend = torch.backends.quantized.engine
            return get_default_qconfig(backend)
        return None

    def quantize_method(
        self,
        method_name: str,
        scheme: QUANTIZATION_SCHEMES,
        calibration_data: Optional[List[np.ndarray]] = None,
    ) -> Any:
        """
        Квантует указанный метод.

        Args:
            method_name: Название метода из segmenter.method_map
            scheme: Схема квантования
            calibration_data: Данные для калибровки (только для static INT8)

        Returns:
            Квантованная функция или оригинал при ошибке
        """
        if method_name not in self.original.method_map:
            raise ValueError(f"Method '{method_name}' not found")

        if scheme not in self.config.schemes:
            raise ValueError(f"Scheme '{scheme}' not in config")

        # Проверка поддержки квантования
        original_func = self.original.method_map[method_name]
        if not is_method_quantizable(method_name, original_func):
            warnings.warn(f"Method '{method_name}' may not support quantization. " f"Proceeding with caution.")

        # Обработка по схемам
        if scheme == "fp32":
            # Без квантования — эталон
            quantized_func = original_func

        elif scheme == "fp16":
            # Полуточность через autocast
            def fp16_wrapper(x):
                with torch.autocast(device_type=str(self.device.type), dtype=torch.float16, enabled=True):
                    return original_func(x)

            quantized_func = fp16_wrapper

        elif scheme == "int8_dynamic":
            # Динамическое квантование (без калибровки)
            try:
                wrapper = QuantizationWrapper(original_func)
                wrapper.qconfig = self._get_qconfig(scheme)
                quantized_func = torch.ao.quantization.convert(wrapper.eval(), inplace=False)
            except Exception as e:
                warnings.warn(f"Dynamic quantization failed: {e}. Using FP32 fallback.")
                quantized_func = original_func

        elif scheme == "int8_static":
            # Статическое квантование (требует калибровки)
            if calibration_data is None:
                raise ValueError(
                    "Static quantization requires calibration_data. "
                    "Provide images for calibration or use 'int8_dynamic'."
                )

            # Калибровка
            self.calibrate(method_name, calibration_data)
            quantized_func = self.quantized_methods[method_name][scheme]

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # Кэширование
        if method_name not in self.quantized_methods:
            self.quantized_methods[method_name] = {}
        self.quantized_methods[method_name][scheme] = quantized_func

        if self.config.verbose:
            orig_size = estimate_model_size(original_func, torch.float32)
            quant_size = estimate_model_size(quantized_func, SUPPORTED_DTYPES[scheme])
            print(f"✅ Quantized '{method_name}' ({scheme}): " f"{format_size_reduction(orig_size, quant_size)}")

        return quantized_func

    def calibrate(
        self,
        method_name: str,
        calibration_data: List[np.ndarray],
        scheme: QUANTIZATION_SCHEMES = "int8_static",
        num_steps: Optional[int] = None,
    ) -> Any:
        """
        Калибровка метода для статического квантования.

        Args:
            method_name: Название метода
            calibration_data: Список изображений для калибровки
            scheme: Схема квантования
            num_steps: Количество шагов калибровки (по умолчанию из config)

        Returns:
            Квантованная функция
        """
        num_steps = num_steps or self.config.calibration_steps

        if method_name not in self.original.method_map:
            raise ValueError(f"Method '{method_name}' not found")

        original_func = self.original.method_map[method_name]

        # Подготовка обёртки
        wrapper = QuantizationWrapper(original_func)
        wrapper.qconfig = self._get_qconfig(scheme)
        prepared = torch.ao.quantization.prepare(wrapper.eval(), inplace=False)

        # Калибровочный проход
        with torch.no_grad():
            for i, img in enumerate(calibration_data[:num_steps]):
                try:
                    input_tensor = self.original.preprocess_image(img).to(self.device)
                    _ = prepared(input_tensor)
                except Exception as e:
                    warnings.warn(f"Calibration step {i} failed: {e}")
                    continue

        # Конвертация в квантованную версию
        quantized = torch.ao.quantization.convert(prepared, inplace=False)

        # Кэширование
        if method_name not in self.quantized_methods:
            self.quantized_methods[method_name] = {}
        self.quantized_methods[method_name][scheme] = quantized

        if self.config.verbose:
            print(f"✅ Calibrated '{method_name}' with {num_steps} samples")

        return quantized

    def benchmark_scheme(
        self,
        method_name: str,
        scheme: QUANTIZATION_SCHEMES,
        image: np.ndarray,
        n_runs: Optional[int] = None,
        warmup: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Бенчмарк одной схемы квантования.

        Args:
            method_name: Название метода
            scheme: Схема квантования
            image: Тестовое изображение
            n_runs: Количество запусков
            warmup: Прогревочные запуски

        Returns:
            Dict с метриками времени и точности
        """
        n_runs = n_runs or self.config.n_runs
        warmup = warmup or self.config.warmup_runs

        # Получение функции
        if scheme not in self.quantized_methods.get(method_name, {}):
            self.quantize_method(method_name, scheme)

        quant_func = self.quantized_methods[method_name][scheme]
        original_func = self.original.method_map[method_name]

        # Подготовка входа
        input_tensor = self.original.preprocess_image(image).to(self.device)

        # Прогрев
        for _ in range(warmup):
            _ = original_func(input_tensor)
            _ = quant_func(input_tensor)
            if self.config.sync_cuda and self.device.type == "cuda":
                torch.cuda.synchronize()

        # Замер оригинала
        times_orig = []
        with torch.inference_mode():
            for _ in range(n_runs):
                if self.config.sync_cuda and self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = original_func(input_tensor)
                if self.config.sync_cuda and self.device.type == "cuda":
                    torch.cuda.synchronize()
                times_orig.append((time.perf_counter() - start) * 1000)

        # Замер квантованной версии
        times_quant = []
        with torch.inference_mode():
            for _ in range(n_runs):
                if self.config.sync_cuda and self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = quant_func(input_tensor)
                if self.config.sync_cuda and self.device.type == "cuda":
                    torch.cuda.synchronize()
                times_quant.append((time.perf_counter() - start) * 1000)

        # Вычисление точности
        with torch.inference_mode():
            orig_mask = original_func(input_tensor)
            quant_mask = quant_func(input_tensor)

        accuracy = compute_quantization_error(orig_mask, quant_mask, tolerance=self.config.tolerance)

        # Статистика времени
        times_orig_np = np.array(times_orig)
        times_quant_np = np.array(times_quant)

        return {
            "method": method_name,
            "scheme": scheme,
            "device": str(self.device),
            # Время
            "original_mean_ms": float(np.mean(times_orig_np)),
            "original_std_ms": float(np.std(times_orig_np)),
            "quantized_mean_ms": float(np.mean(times_quant_np)),
            "quantized_std_ms": float(np.std(times_quant_np)),
            "speedup": float(np.mean(times_orig_np) / np.mean(times_quant_np)),
            # Точность
            **accuracy,
            # Размер
            "size_reduction": format_size_reduction(
                estimate_model_size(original_func, torch.float32),
                estimate_model_size(quant_func, SUPPORTED_DTYPES[scheme]),
            ),
        }

    def run_full_benchmark(
        self,
        methods: Optional[List[str]] = None,
        schemes: Optional[List[QUANTIZATION_SCHEMES]] = None,
        calibration_data: Optional[List[np.ndarray]] = None,
        test_image: Optional[np.ndarray] = None,
    ) -> List[Dict[str, float]]:
        """
        Полный бенчмарк всех комбинаций метод × схема.

        Args:
            methods: Список методов (по умолчанию — все из config)
            schemes: Список схем квантования
            calibration_data: Данные для калибровки статического INT8
            test_image: Изображение для бенчмарка

        Returns:
            List[Dict] с результатами
        """
        # Определение списков
        if methods is None:
            methods = self.config.include_methods or [
                m for m in self.original.method_map.keys() if m not in self.config.exclude_methods
            ]

        if schemes is None:
            schemes = self.config.schemes

        if self.config.verbose:
            print(f"🚀 Running quantization benchmark:")
            print(f"   Methods: {len(methods)}")
            print(f"   Schemes: {schemes}")
            print(f"   Device: {self.device}")

        results = []

        for method_name in methods:
            if method_name not in self.original.method_map:
                warnings.warn(f"⚠️ Method '{method_name}' not found, skipping")
                continue

            for scheme in schemes:
                try:
                    if self.config.verbose:
                        print(f"   • {method_name} / {scheme}...", end=" ")

                    # Калибровка если нужно
                    if scheme == "int8_static" and calibration_data:
                        self.calibrate(method_name, calibration_data, scheme)

                    # Бенчмарк
                    result = self.benchmark_scheme(method_name, scheme, test_image)
                    results.append(result)

                    if self.config.verbose:
                        speedup = result["speedup"]
                        agree = result["pixel_agreement"]
                        print(f"✓ {format_speedup(speedup)} speedup, " f"{agree*100:.2f}% agreement")

                except Exception as e:
                    if self.config.verbose:
                        print(f"✗ {e}")
                    warnings.warn(f"⚠️ {method_name}/{scheme}: {e}")
                    continue

        self.benchmark_results = results
        return results
