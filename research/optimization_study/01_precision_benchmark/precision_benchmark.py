# research/optimization_study/01_precision_benchmark/precision_benchmark.py

# 🔬 Исследование 1: Влияние точности вычислений (Precision Study)
# Цель: Сравнить скорость/точность для разных типов данных

"""
Основной класс для бенчмаркинга точности вычислений.
"""

import torch
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings

from .config import (
    PrecisionConfig,
    is_dtype_supported,
    PRECISION_TO_DTYPE,
    DEFAULT_CONFIG,
)
from .utils import (
    get_available_dtypes,
    format_time,
    format_speedup,
    compute_pixel_agreement,
    compute_mse,
    safe_autocast_context,
    convert_tensor_precision,
    normalize_for_comparison,
)
from segmenters.TorchSegmenter import TorchSegmenter


class PrecisionBenchmark:
    """
    Бенчмарк для сравнения скорости и точности классических методов
    сегментации при использовании разных типов данных.

    Поддерживаемые прецизионные режимы:
    - fp32 (float32): Эталонная точность, полная совместимость
    - fp16 (float16): Половинная точность, ускорение на современных GPU
    - bf16 (bfloat16): Brain Float, баланс точности/скорости (Ampere+)
    - int8 (int8): Квантование (экспериментально, для инференса)

    Пример использования:
        >>> segmenter = TorchSegmenter(method="sobel_edge")
        >>> image = load_image("test.jpg")
        >>> benchmark = PrecisionBenchmark(segmenter, image)
        >>>
        >>> # Замер для одного метода
        >>> result = benchmark.benchmark_method("sobel_edge", "fp16")
        >>> print(f"FP16 speed: {result['mean_ms']:.3f} ms")
        >>>
        >>> # Полный бенчмарк
        >>> df = benchmark.run_full_benchmark(
        ...     methods=["sobel_edge", "otsu_thresholding"],
        ...     precisions=["fp32", "fp16"]
        ... )
        >>> benchmark.plot_results(df, output_dir="./results/")
    """

    PRECISIONS = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,  # Требует Ampere+ GPU
        "int8": torch.int8,  # Только для квантованных моделей
    }

    def __init__(
        self,
        segmenter: Any,
        image: np.ndarray,
        config: Optional[PrecisionConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            segmenter: Экземпляр TorchSegmenter
            image: Исходное изображение (numpy array или PIL)
            config: Конфигурация бенчмарка (по умолчанию DEFAULT_CONFIG)
            device: Устройство для вычислений ('cuda' или 'cpu')
        """
        self.segmenter = segmenter
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Предобработка изображения один раз
        self.reference_input = segmenter.preprocess_image(image).to(self.device)

        # Кэш для reference масок (FP32)
        self._reference_masks: Dict[str, torch.Tensor] = {}

        # Результаты
        self.results: List[Dict[str, Any]] = []

        if self.config.verbose:
            print(f"🔧 PrecisionBenchmark initialized:")
            print(f"   Device: {self.device}")
            print(
                f"   Available precisions: {get_available_dtypes(str(self.device.type))}"
            )
            print(f"   Image shape: {self.reference_input.shape}")

    def _get_reference_mask(self, method_name: str) -> torch.Tensor:
        """Получает или вычисляет reference маску (FP32) для метода."""
        if method_name not in self._reference_masks:
            with torch.inference_mode():
                self._reference_masks[method_name] = self.segmenter.methods[
                    method_name
                ](self.reference_input).clone()
        return self._reference_masks[method_name]

    def _convert_precision(
        self, tensor: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Конвертация тензора в нужный тип данных"""
        if dtype == torch.int8:
            # Для INT8 нужна квантовка
            return torch.quantize_per_tensor(
                tensor.float(), scale=1.0 / 128, zero_point=0, dtype=torch.qint8
            ).dequantize()  # Для бенчмарка де-квантуем
        return tensor.to(dtype)

    def benchmark_method(
        self,
        method_name: str,
        precision: str,
        n_runs: Optional[int] = None,  # 50
        warmup: Optional[int] = None,  # 10
    ) -> Dict[str, float]:
        """
        Замер времени выполнения для одного метода и точности.

        Args:
            method_name: Название метода из segmenter.method_map
            precision: Прецизионный режим ('fp32', 'fp16', 'bf16', 'int8')
            n_runs: Количество запусков (переопределяет config)
            warmup: Количество прогревочных запусков

        Returns:
            Dict со статистикой времени выполнения
        """

        n_runs = n_runs or self.config.n_runs
        warmup = warmup or self.config.warmup_runs

        dtype = PRECISION_TO_DTYPE[precision]

        # Подготовка входа
        input_tensor = convert_tensor_precision(
            self.reference_input.clone(), dtype, quantize_int8=(precision == "int8")
        )

        # Проверка поддержки dtype
        if not is_dtype_supported(dtype, str(self.device.type)):
            raise RuntimeError(
                f"Precision '{precision}' ({dtype}) not supported on {self.device}"
            )

        # Warm-up
        for _ in range(warmup):
            with torch.inference_mode():
                with safe_autocast_context(
                    device_type=str(self.device.type),
                    dtype=dtype,
                    enabled=self.config.enable_autocast,
                    cpu_enabled=self.config.autocast_cpu_enabled,
                ):
                    _ = self.segmenter.methods[method_name](input_tensor)
                    if self.config.sync_cuda and self.device.type == "cuda":
                        torch.cuda.synchronize()

        # Основной замер
        times: List[float] = []

        with torch.inference_mode():
            for _ in range(n_runs):
                start = time.perf_counter()

                with safe_autocast_context(
                    device_type=str(self.device.type),
                    dtype=dtype,
                    enabled=self.config.enable_autocast,
                    cpu_enabled=self.config.autocast_cpu_enabled,
                ):
                    _ = self.segmenter.methods[method_name](input_tensor)

                if self.config.sync_cuda and self.device.type == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms

        # Статистика
        times_np = np.array(times)
        return {
            "mean_ms": float(np.mean(times_np)),
            "median_ms": float(np.median(times_np)),
            "std_ms": float(np.std(times_np)),
            "min_ms": float(np.min(times_np)),
            "max_ms": float(np.max(times_np)),
            "p95_ms": float(np.percentile(times_np, 95)),
            "p99_ms": float(np.percentile(times_np, 99)),
        }

    def benchmark_accuracy(
        self, method_name: str, precision: str, reference_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Оценка потери точности при смене формата данных.

        Args:
            method_name: Название метода
            precision: Тестируемый прецизионный режим
            reference_mask: Reference маска (если None — вычисляется)

        Returns:
            Dict с метриками согласия с reference
        """

        if reference_mask is None:
            reference_mask = self._get_reference_mask(method_name)

        dtype = PRECISION_TO_DTYPE[precision]
        input_tensor = convert_tensor_precision(
            self.reference_input.clone(), dtype, quantize_int8=(precision == "int8")
        )

        with torch.inference_mode():
            # with torch.autocast(
            #     device_type="cuda",
            #     dtype=dtype if dtype != torch.int8 else torch.float16,
            #     enabled=(precision != "fp32")
            # ):
            #     pred_mask = self.segmenter.methods[method_name](
            #         self._convert_precision(input_tensor, dtype)
            #     )
            with safe_autocast_context(
                device_type=str(self.device.type),
                dtype=dtype,
                enabled=self.config.enable_autocast,
                cpu_enabled=self.config.autocast_cpu_enabled,
            ):
                pred_mask = self.segmenter.methods[method_name](input_tensor)

        # Нормализация для сравнения (если нужно)
        pred_norm = normalize_for_comparison(pred_mask)
        ref_norm = normalize_for_comparison(reference_mask)

        # Метрики согласия
        agreement = compute_pixel_agreement(
            pred_norm, ref_norm, tolerance=self.config.tolerance
        )
        mse = compute_mse(pred_norm, ref_norm)
        max_diff = float(torch.max(torch.abs(pred_norm - ref_norm)).item())

        return {
            "pixel_agreement": agreement,
            "mse": mse,
            "max_diff": max_diff,
            "mean_diff": float(torch.mean(torch.abs(pred_norm - ref_norm)).item()),
        }

    def run_full_benchmark(
        self,
        methods: Optional[List[str]] = None,
        precisions: Optional[List[str]] = None,
        include_accuracy: bool = True,
    ) -> pd.DataFrame:
        """
        Полный бенчмарк всех комбинаций метод × точность.

        Args:
            methods: Список методов (по умолчанию — все из config)
            precisions: Список прецизионных режимов
            include_accuracy: Включать ли замеры точности

        Returns:
            pd.DataFrame с результатами
        """
        # Определение списков
        if methods is None:
            methods = self.config.include_methods or [
                m
                for m in self.segmenter.method_map.keys()
                if m not in self.config.exclude_methods
            ]

        if precisions is None:
            precisions = [
                p
                for p in self.config.precisions
                if is_dtype_supported(PRECISION_TO_DTYPE[p], str(self.device.type))
            ]

        if self.config.verbose:
            print(f"🚀 Running full benchmark:")
            print(f"   Methods: {len(methods)}")
            print(f"   Precisions: {precisions}")
            print(f"   Runs per config: {self.config.n_runs}")

        rows = []
        for method_name in methods:
            if method_name not in self.segmenter.method_map:
                warnings.warn(f"⚠️ Method '{method_name}' not found, skipping")
                continue

            # Reference для точности (вычисляем один раз)
            reference_mask = None
            if include_accuracy:
                reference_mask = self._get_reference_mask(method_name)

            for precision in precisions:
                try:
                    if self.config.verbose:
                        print(f"   • {method_name} / {precision}...", end=" ")

                    # Замер времени
                    timing = self.benchmark_method(method_name, precision)

                    # Замер точности
                    accuracy = {}
                    if (
                        include_accuracy
                        and precision != self.config.reference_precision
                    ):
                        accuracy = self.benchmark_accuracy(
                            method_name, precision, reference_mask
                        )

                    row = {
                        "method": method_name,
                        "precision": precision,
                        "device": str(self.device),
                        **timing,
                        **accuracy,
                    }
                    rows.append(row)

                    if self.config.verbose:
                        speed = format_time(timing["median_ms"] / 1000)
                        print(f"✓ {speed}")

                except Exception as e:
                    if self.config.verbose:
                        print(f"✗ {e}")
                    warnings.warn(f"⚠️ {method_name}/{precision}: {e}")
                    continue

        # Создание DataFrame
        df = pd.DataFrame(rows)

        # Добавление speedup относительно reference precision
        if not df.empty and self.config.reference_precision in df["precision"].values:
            for method in df["method"].unique():
                ref_time = df[
                    (df["method"] == method)
                    & (df["precision"] == self.config.reference_precision)
                ]["median_ms"].values

                if len(ref_time) > 0:
                    mask = df["method"] == method
                    df.loc[mask, "speedup_vs_reference"] = (
                        ref_time[0] / df.loc[mask, "median_ms"]
                    )

        # Сохранение результатов
        if self.config.save_raw_data:
            output_path = Path(self.config.output_dir) / "raw_results.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            if self.config.verbose:
                print(f"💾 Results saved: {output_path}")

        self.results = rows
        return df

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерация сводной статистики по результатам.

        Args:
            df: DataFrame с результатами бенчмарка

        Returns:
            Dict с агрегированной статистикой
        """
        if df.empty:
            return {}

        summary = {
            "total_methods": df["method"].nunique(),
            "total_precisions": df["precision"].nunique(),
            "total_runs": len(df),
        }

        # Средняя скорость по прецизионным режимам
        speed_by_precision = df.groupby("precision")["median_ms"].mean()
        summary["avg_time_by_precision"] = speed_by_precision.to_dict()

        # Максимальный speedup
        if "speedup_vs_reference" in df.columns:
            max_speedup = df["speedup_vs_reference"].max()
            best_config = df.loc[df["speedup_vs_reference"].idxmax()]
            summary["best_speedup"] = {
                "value": float(max_speedup),
                "method": best_config["method"],
                "precision": best_config["precision"],
            }

        # Средняя точность по прецизионным режимам
        if "pixel_agreement" in df.columns:
            accuracy_by_precision = df.groupby("precision")["pixel_agreement"].mean()
            summary["avg_agreement_by_precision"] = accuracy_by_precision.to_dict()

        return summary

    def print_summary(self, df: pd.DataFrame):
        """Вывод сводки в консоль."""
        summary = self.get_summary(df)

        print("\n📊 Precision Benchmark Summary")
        print("=" * 60)
        print(f"Methods tested: {summary.get('total_methods', 0)}")
        print(f"Precisions tested: {summary.get('total_precisions', 0)}")
        print(f"Total configurations: {summary.get('total_runs', 0)}")

        if "avg_time_by_precision" in summary:
            print("\n⏱️  Average time by precision:")
            for prec, time_ms in summary["avg_time_by_precision"].items():
                print(f"   {prec:6s}: {format_time(time_ms / 1000)}")

        if "best_speedup" in summary:
            best = summary["best_speedup"]
            print(f"\n🚀 Best speedup: {format_speedup(best['value'])}")
            print(f"   Method: {best['method']}")
            print(f"   Precision: {best['precision']}")

        if "avg_agreement_by_precision" in summary:
            print("\n🎯 Average pixel agreement (vs FP32):")
            for prec, agree in summary["avg_agreement_by_precision"].items():
                status = "✅" if agree > 0.999 else "⚠️" if agree > 0.99 else "❌"
                print(f"   {status} {prec:6s}: {agree*100:.3f}%")

        print("=" * 60)
