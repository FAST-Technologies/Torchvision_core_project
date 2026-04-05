# utils/threshold_warmup.py

# Импорт основных библиотек
import numpy as np
from typing import (
    TypedDict,
    List,
    Dict,
    Any,
)
import time


class WarmupMetrics(TypedDict):
    mean_ms: float
    std_ms: float
    n_runs: int  # или total: int


class SizeResults(TypedDict):
    sizes: Dict[str, WarmupMetrics]


class PatternResults(TypedDict):
    patterns: Dict[str, WarmupMetrics]


class ThresholdWarmUp:
    """
    Специализированный warm-up для пороговых и граничных методов.
    Оптимизирует кэширование гистограмм и свёрток.
    """

    @staticmethod
    def warmup_threshold_methods(
        segmenters_dict: Dict[str, Any],
        image_sizes: List[tuple] = [(128, 128), (256, 256), (512, 512)],
        n_runs_per_size: int = 2,
    ) -> Dict[str, SizeResults]:
        """
        Прогрев пороговых методов на изображениях разного размера.

        Args:
            segmenters_dict: {name: segmenter}
            image_sizes: Список размеров для тестирования
            n_runs_per_size: Количество прогонов на каждый размер

        Returns:
            Dict со статистикой warm-up
        """
        threshold_methods: List[str] = [
            "global_threshold",
            "otsu",
            "adaptive_threshold",
            "niblack",
            "sauvola",
        ]

        results: Dict[str, SizeResults] = {}

        print("\n🔥 WARM-UP ПОРОГОВЫХ МЕТОДОВ")
        print("=" * 60)

        for name, segmenter in segmenters_dict.items():
            is_threshold = any(tm in name.lower() for tm in threshold_methods)
            if not is_threshold:
                continue
            method_results: SizeResults = {"sizes": {}}
            for size in image_sizes:
                # Создаём тестовое изображение
                img: np.ndarray = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

                times: List[float] = []
                for _ in range(n_runs_per_size):
                    start = time.perf_counter()
                    try:
                        if hasattr(segmenter, "segment"):
                            segmenter.segment(img)
                        times.append(time.perf_counter() - start)
                    except Exception:
                        times.append(float("inf"))

                method_results["sizes"][str(size)] = WarmupMetrics(
                    mean_ms=float(np.mean(times) * 1000),
                    std_ms=float(np.std(times) * 1000),
                    n_runs=len(times),
                )
            results[name] = method_results
            print(f"✅ {name}: {method_results['sizes']}")
        return results

    @staticmethod
    def warmup_edge_methods(
        segmenters_dict: Dict[str, Any],
        edge_patterns: List[str] = ["horizontal", "vertical", "diagonal", "noise"],
    ) -> Dict[str, PatternResults]:
        """
        Прогрев граничных методов на различных паттернах.
        """
        edge_methods = ["sobel", "canny", "laplacian", "prewitt"]
        results: Dict[str, PatternResults] = {}

        print("\n🔥 WARM-UP ГРАНИЧНЫХ МЕТОДОВ")
        print("=" * 60)

        for name, segmenter in segmenters_dict.items():
            is_edge = any(em in name.lower() for em in edge_methods)
            if not is_edge:
                continue

            method_results: PatternResults = {"patterns": {}}

            for pattern in edge_patterns:
                img: np.ndarray = ThresholdWarmUp._create_edge_pattern(
                    256, 256, pattern
                )

                times: List[float] = []
                for _ in range(3):
                    start = time.perf_counter()
                    try:
                        if hasattr(segmenter, "segment"):
                            segmenter.segment(img)
                        times.append(time.perf_counter() - start)
                    except Exception:
                        times.append(float("inf"))

                method_results["patterns"][pattern] = WarmupMetrics(  # type: ignore[typeddict-item]
                    mean_ms=float(np.mean(times) * 1000),
                    std_ms=float(np.std(times) * 1000),
                    n_runs=len(times),
                )

            results[name] = method_results
            print(f"✅ {name}: {method_results['patterns']}")

        return results

    @staticmethod
    def _create_edge_pattern(h: int, w: int, pattern: str) -> np.ndarray:
        """Создаёт тестовые паттерны для граничных методов."""
        img: np.ndarray = np.zeros((h, w), dtype=np.uint8)

        if pattern == "horizontal":
            img[(h // 2 - 5) : (h // 2 + 5), :] = 255
        elif pattern == "vertical":
            img[:, (w // 2 - 5) : (w // 2 + 5)] = 255
        elif pattern == "diagonal":
            for i in range(min(h, w) - 1):
                img[i, i] = 255
                img[i, i + 1] = 255
                img[i + 1, i] = 255
        elif pattern == "noise":
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        else:
            # Default: checkerboard
            img[::2, ::2, :] = 255
            img[1::2, 1::2, :] = 255

        return img
