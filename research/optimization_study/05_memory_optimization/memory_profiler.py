# research/optimization_study/05_memory_optimization/memory_profiler.py

"""
Профилирование памяти для TorchSegmenter методов.
"""
import torch
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple
import time
import gc
import warnings

from .config import MemoryConfig, DEFAULT_CONFIG
from .utils import (
    get_memory_backend_info,
    format_memory_size,
    estimate_tensor_size,
    detect_memory_leaks,
    clear_memory,
)


class MemoryProfiler:
    """
    Профилировщик потребления памяти.

    Поддерживает:
    - Замеры allocated/reserved/peak памяти
    - Детектирование утечек
    - Сравнение до/после оптимизации
    - Временные профили аллокаций

    Пример:
        >>> profiler = MemoryProfiler()
        >>> report = profiler.profile_method(
        ...     func=segmenter.method_map["sauvola_thresholding"],
        ...     input_tensor=image_tensor,
        ...     n_runs=20
        ... )
        >>> print(f"Peak: {report['peak_mb']:.2f} MB")
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device(self.config.device)
        self.profiles: List[Dict[str, Any]] = []

    @staticmethod
    def get_gpu_memory_mb() -> float:
        """Возвращает использованную GPU память в МБ"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0

    @staticmethod
    def get_gpu_memory_reserved_mb() -> float:
        """Возвращает зарезервированную GPU память в МБ"""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024**2
        return 0.0

    @staticmethod
    def reset_memory_stats():
        """Сброс статистики памяти"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

    def _get_memory_stats(self) -> Dict[str, float]:
        """Получает текущую статистику памяти."""
        stats = {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_mb": 0.0,
        }

        if self.device.type == "cuda" and torch.cuda.is_available():
            stats["allocated_mb"] = torch.cuda.memory_allocated(self.device) / 1e6
            stats["reserved_mb"] = torch.cuda.memory_reserved(self.device) / e6
            stats["peak_mb"] = torch.cuda.max_memory_allocated(self.device) / 1e6

        return stats

    def _reset_stats(self):
        """Сбрасывает статистику памяти."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        if self.config.clear_cache_between_runs:
            clear_memory(
                device=self.config.device,
                aggressive=self.config.collect_garbage,
            )

    def profile_method(
        self,
        func: Callable,
        input_tensor: torch.Tensor,
        n_runs: Optional[int] = None,  # 10
        warmup: Optional[int] = None,  # 3
        track_allocations: bool = True,
    ) -> Dict[str, Any]:
        """
        Профилирование памяти для одного метода.

        Args:
            func: Функция для профилирования
            input_tensor: Входной тензор
            n_runs: Количество запусков
            warmup: Прогревочные запуски
            track_allocations: Отслеживать аллокации в реальном времени

        Returns:
            Dict с метриками памяти
        """

        # self.reset_memory_stats()
        n_runs = n_runs or self.config.n_runs
        warmup = warmup or self.config.warmup_runs

        # Сброс статистики
        self._reset_stats()

        # Warm-up
        for _ in range(warmup):
            _ = func(input_tensor)
            if self.config.sync_cuda and self.device.type == "cuda":
                torch.cuda.synchronize()

        # # Замеры
        # memory_allocated: List[float] = []
        # memory_reserved: List[float] = []

        # with torch.inference_mode():
        #     for _ in range(n_runs):
        #         before_alloc = self.get_gpu_memory_mb()
        #         before_res = self.get_gpu_memory_reserved_mb()

        #         _ = func(input_tensor)

        #         if input_tensor.device.type == "cuda":
        #             torch.cuda.synchronize()

        #         after_alloc = self.get_gpu_memory_mb()
        #         after_res = self.get_gpu_memory_reserved_mb()

        #         memory_allocated.append(after_alloc - before_alloc)
        #         memory_reserved.append(after_res - before_res)

        # return {
        #     "alloc_mean_mb": np.mean(memory_allocated),
        #     "alloc_std_mb": np.std(memory_allocated),
        #     "reserved_mean_mb": np.mean(memory_reserved),
        #     "reserved_std_mb": np.std(memory_reserved),
        #     "peak_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        # }
        # Базовые замеры
        baseline = self._get_memory_stats()

        # Основной замер
        alloc_deltas: List[float] = []
        reserved_deltas: List[float] = []
        peak_values: List[float] = []

        for run in range(n_runs):
            if self.config.clear_cache_between_runs and run > 0:
                self._reset_stats()

            before = self._get_memory_stats()

            # Выполнение
            result = func(input_tensor)

            if self.config.sync_cuda and self.device.type == "cuda":
                torch.cuda.synchronize()

            after = self._get_memory_stats()

            alloc_deltas.append(after["allocated_mb"] - before["allocated_mb"])
            reserved_deltas.append(after["reserved_mb"] - before["reserved_mb"])
            peak_values.append(after["peak_mb"])

        # Финальные замеры
        final = self._get_memory_stats()

        # Детект утечек
        leak_detected, leak_info = detect_memory_leaks(
            baseline["allocated_mb"],
            final["allocated_mb"],
            threshold_mb=self.config.leak_threshold_mb,
        )

        # Статистика
        alloc_np = np.array(alloc_deltas)
        reserved_np = np.array(reserved_deltas)
        peak_np = np.array(peak_values)

        report = {
            # Базовые метрики
            "baseline_allocated_mb": baseline["allocated_mb"],
            "baseline_reserved_mb": baseline["reserved_mb"],
            "final_allocated_mb": final["allocated_mb"],
            "final_reserved_mb": final["reserved_mb"],
            # Дельты (потребление за вызов)
            "alloc_delta_mean_mb": float(np.mean(alloc_np)),
            "alloc_delta_std_mb": float(np.std(alloc_np)),
            "reserved_delta_mean_mb": float(np.mean(reserved_np)),
            # Пиковое потребление
            "peak_mb_mean": float(np.mean(peak_np)),
            "peak_mb_max": float(np.max(peak_np)),
            "peak_mb_min": float(np.min(peak_np)),
            # Утечки
            "leak_detected": leak_detected,
            "leak_info": leak_info,
            # Мета
            "n_runs": n_runs,
            "device": str(self.device),
            "input_size_mb": estimate_tensor_size(input_tensor)["mb"],
        }

        # Сохранение профиля
        self.profiles.append(report)

        return report

    def compare_methods(
        self,
        methods: Dict[str, Callable],
        input_tensor: torch.Tensor,
        n_runs: int = 10,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Сравнительное профилирование нескольких методов.

        Args:
            methods: Dict {name: func}
            input_tensor: Входной тензор
            n_runs: Запусков на метод

        Returns:
            Dict {method_name: profile_report}
        """
        results = {}
        for name, func in methods.items():
            if self.config.verbose:
                print(f"🔍 Profiling {name}...")

            try:
                report = self.profile_method(func, input_tensor, n_runs=n_runs)
                results[name] = report
            except Exception as e:
                warnings.warn(f"Profiling failed for {name}: {e}")
                results[name] = {"error": str(e)}

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Сводная статистика по всем профилям."""
        if not self.profiles:
            return {}

        # Группировка по устройству
        by_device: Dict[str, List[Dict]] = {}
        for p in self.profiles:
            dev = p.get("device", "unknown")
            by_device.setdefault(dev, []).append(p)

        summary = {
            "total_profiles": len(self.profiles),
            "devices": list(by_device.keys()),
        }

        # Агрегация по устройствам
        for dev, profiles in by_device.items():
            peaks = [p["peak_mb_mean"] for p in profiles if "peak_mb_mean" in p]
            leaks = sum(1 for p in profiles if p.get("leak_detected", False))

            summary[f"{dev}_avg_peak_mb"] = np.mean(peaks) if peaks else 0
            summary[f"{dev}_leak_count"] = leaks

        return summary

    def print_summary(self):
        """Вывод сводки в консоль."""
        summary = self.get_summary()

        print("\n🧠 Memory Profiling Summary")
        print("=" * 60)
        print(f"Total profiles: {summary.get('total_profiles', 0)}")
        print(f"Devices: {summary.get('devices', [])}")

        for dev in summary.get("devices", []):
            peak = summary.get(f"{dev}_avg_peak_mb", 0)
            leaks = summary.get(f"{dev}_leak_count", 0)
            print(f"   {dev}: avg peak = {peak:.2f} MB, leaks = {leaks}")

        print("=" * 60)
