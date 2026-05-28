# utils/threshold_warmup.py

"""Специализированный warm-up для пороговых и граничных методов сегментации.

Оптимизирует кэширование гистограмм и свёрточных ядер.

Предназначен для:
- Прогрева кэшей гистограмм и свёрточных ядер перед бенчмарком.
- Оценки стабильности производительности на разных размерах/паттернах.
- Выявления методов с аномальным временем первого запуска (JIT, CUDA init).

Ключевые особенности:
- ✅ Специализация: отдельные методы для threshold и edge detection
- ✅ Масштабный анализ: тестирование на разных размерах изображений
- ✅ Паттерн-тестирование: горизонталь, вертикаль, диагональ, шум
- ✅ Типизированные результаты: WarmupMetrics, SizeResults, PatternResults
- ✅ Устойчивость к ошибкам: сбойный прогон не прерывает цикл
- ✅ Точные замеры: time.perf_counter() для высокой точности

Типичный workflow:
```python
from utils.threshold_warmup import ThresholdWarmUp
from segmenters.SklearnSegmenter import SklearnSegmenter

# 1. Подготовка пороговых методов
threshold_segmenters = {
    "otsu": SklearnSegmenter("otsu_thresholding"),
    "adaptive": SklearnSegmenter("adaptive_thresholding"),
}

# 2. Прогрев на разных размерах
threshold_results = ThresholdWarmUp.warmup_threshold_methods(
    segmenters_dict=threshold_segmenters,
    image_sizes=[(256, 256), (512, 512)],
    n_runs_per_size=3
)

# 3. Подготовка граничных методов
edge_segmenters = {
    "sobel": SklearnSegmenter("sobel_edge"),
    "canny": SklearnSegmenter("canny_edge"),
}

# 4. Прогрев на разных паттернах
edge_results = ThresholdWarmUp.warmup_edge_methods(
    segmenters_dict=edge_segmenters,
    edge_patterns=["horizontal", "vertical", "noise"],
    n_runs_per_pattern=3
)

# 5. Анализ результатов
for method, data in threshold_results.items():
    for size, metrics in data["sizes"].items():
        print(f"{method} @ {size}: {metrics['mean_ms']:.2f}ms ± {metrics['std_ms']:.2f}ms")
```

Note:
    - Методы фильтруются по ключевым словам в имени (регистронезависимо).
    - Для паттерна "noise" генерируется цветное изображение (H×W×3), остальные — grayscale (H×W).
    - При ошибке в прогоне время записывается как `inf`, но цикл продолжается.
    - Доступ к `segmenter.params["execution_info"]` возможен после каждого прогона.
    - Результаты возвращаются в миллисекундах для удобства интерпретации.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций
import time
import numpy as np
from typing import TypedDict, List, Dict, Any, Tuple, TypeAlias

import torch

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & TYPEDDICTS
# ──────────────────────────────────────────────────────────────────────
SegmenterLike: TypeAlias = Any  # Объект с методом .segment(image) -> np.ndarray
"""Тип используемого сегментатора, dtype=Any."""

EdgePatterns: List[str] = ["horizontal", "vertical", "diagonal", "noise"]
"""Тип граничного паттерна, dtype=List[str]."""


# ──────────────────────────────────────────────────────────────────────
class WarmupMetrics(TypedDict):
    """Структура метрик warm-up для одного теста.

    Attributes:
        mean_ms: Среднее время выполнения в миллисекундах.
        std_ms: Стандартное отклонение времени в миллисекундах.
        n_runs: Количество успешных прогонов.
    """

    mean_ms: float
    std_ms: float
    n_runs: int

# ──────────────────────────────────────────────────────────────────────
class SizeResults(TypedDict):
    """Результаты warm-up по различным размерам изображений.

    Attributes:
        sizes: Словарь `{размер_строкой: WarmupMetrics}`.
    """

    sizes: Dict[str, WarmupMetrics]


# ──────────────────────────────────────────────────────────────────────
class PatternResults(TypedDict):
    """Результаты warm-up по различным паттернам изображений.

    Attributes:
        patterns: Словарь `{имя_паттерна: WarmupMetrics}`.
    """

    patterns: Dict[str, WarmupMetrics]


# ──────────────────────────────────────────────────────────────────────
class ThresholdWarmUp:
    """Специализированный warm-up для пороговых и граничных методов сегментации.

    Оптимизирует кэширование гистограмм и свёрток.

    Предназначен для:
    - Прогрева кэшей гистограмм и свёрточных ядер перед бенчмарком.
    - Оценки стабильности производительности на разных размерах/паттернах.
    - Выявления методов с аномальным временем первого запуска (JIT, CUDA init).

    Особенности:
    - Использует `time.perf_counter()` для точных замеров.
    - Обрабатывает исключения: сбойный прогон не останавливает весь warm-up.
    - Возвращает типизированные результаты для дальнейшего анализа.

    Example:
        ```python
        results = ThresholdWarmUp.warmup_threshold_methods(
            segmenters_dict={"otsu": otsu_segmenter},
            image_sizes=[(256, 256), (512, 512)],
            n_runs_per_size=3,
        )
        print(results["otsu"]["sizes"]["(256, 256)"]["mean_ms"])  # 12.34
        ```
    """

    @staticmethod
    def warmup_threshold_methods(
        segmenters_dict: Dict[str, SegmenterLike],
        image_sizes: List[Tuple[int, int]] = [(128, 128), (256, 256), (512, 512)],
        n_runs_per_size: int = 2,
        use_return_info: bool = False,
        reset_cuda_after_method: bool = True,
    ) -> Dict[str, SizeResults]:
        """Прогрев пороговых методов на изображениях разного размера.

        Для каждого метода, содержащего ключевые слова ("global_threshold", "otsu", ...):
        1. Генерирует случайное изображение указанного размера.
        2. Выполняет `n_runs_per_size` прогонов с замером времени.
        3. Рассчитывает среднее и стандартное отклонение времени выполнения.

        Args:
            segmenters_dict: Словарь `{имя_метода: экземпляр_сегментера}`.
            image_sizes: Список кортежей `(высота, ширина)` для тестирования.
            n_runs_per_size: Количество прогонов на каждый размер изображения.

        Returns:
            Dict[str, SizeResults]: Результаты по методам:
            ```python
            {
                "otsu_method": {
                    "sizes": {
                        "(256, 256)": {"mean_ms": 12.34, "std_ms": 1.23, "n_runs": 2},
                        "(512, 512)": {"mean_ms": 45.67, "std_ms": 3.45, "n_runs": 2},
                    }
                }
            }
            ```

        Note:
            - Методы фильтруются по наличию ключевых слов в имени (регистронезависимо).
            - При ошибке в прогоне время записывается как `inf`, но цикл продолжается.
        """
        threshold_keywords: List[str] = [
            "global_thresholding",
            "global_threshold",
            "otsu_thresholding",
            "otsu",
            "adaptive_thresholding",
            "adaptive_threshold",
            "threshold_niblack",
            "niblack",
            "threshold_sauvola",
            "sauvola",
            "threshold_bernsen",
            "bernsen",
            "threshold_phansalkar",
            "phansalkar",
            "threshold_percentile",
            "percentile",
            "threshold_kittler",
            "kittler",
            "threshold_entropy",
            "kapur",
            "threshold_triangle",
            "triangle",
            "threshold_multi_otsu",
            "multi_otsu",
            "threshold_local_contrast",
            "local_contrast",
        ]

        results: Dict[str, SizeResults] = {}

        print("\n🔥 WARM-UP ПОРОГОВЫХ МЕТОДОВ")
        print("=" * 60)

        for name, segmenter in segmenters_dict.items():
            name_lower = name.lower()
            is_threshold: bool = any(kw in name_lower for kw in threshold_keywords)
            # exclude_keywords = ["kmeans", "dbscan", "meanshift", "neural", "segformer"]
            # if any(excl in name_lower for excl in exclude_keywords):
            #     is_threshold = False
            if not is_threshold:
                continue
            method_results: SizeResults = {"sizes": {}}
            for size in image_sizes:
                # Создаём тестовое изображение
                img: np.ndarray = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

                times: List[float] = []
                for _ in range(n_runs_per_size):
                    start: float = time.perf_counter()
                    try:
                        if use_return_info and hasattr(segmenter, "segment"):
                            _, info = segmenter.segment(img, return_info=True)
                        elif hasattr(segmenter, "segment"):
                            segmenter.segment(img)
                            info = segmenter.params.get("execution_info", {})
                        times.append(time.perf_counter() - start)
                    except Exception:
                        times.append(float("inf"))

                exec_info = segmenter.params.get("execution_info")
                if exec_info:
                    print(
                        f"   📊 {name}: {exec_info.get('method', 'N/A')} — {exec_info.get('execution_time', 0)*1000:.2f}ms"
                    )
                else:
                    # Пытаемся получить из атрибута .info (для OpenCV/Sklearn)
                    exec_info = getattr(segmenter, "info", None)
                    if exec_info:
                        print(
                            f"   📊 {name}: {exec_info.get('method', 'N/A')} — {exec_info.get('execution_time', 0)*1000:.2f}ms"
                        )

                method_results["sizes"][str(size)] = WarmupMetrics(
                    mean_ms=float(np.mean(times) * 1000),
                    std_ms=float(np.std(times) * 1000),
                    n_runs=len(times),
                )
                print("\n⏳ Пауза 10 секунд перед запуском бенчмарка...")
                print("   (нажмите Ctrl+C для отмены, если нужно)")
                try:
                    time.sleep(10)
                except KeyboardInterrupt:
                    logger.warning("\n⚠️  Бенчмарк пропущен по запросу пользователя")
            results[name] = method_results
            print(f"✅ {name}: {method_results['sizes']}")

            if reset_cuda_after_method and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass

        return results

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def warmup_edge_methods(
        segmenters_dict: Dict[str, SegmenterLike],
        edge_patterns: List[str] = EdgePatterns,
        n_runs_per_pattern: int = 3,
        use_return_info: bool = False,
        reset_cuda_after_method: bool = True,
    ) -> Dict[str, PatternResults]:
        """Прогрев граничных методов на различных тестовых паттернах.

        Для каждого метода, содержащего ключевые слова ("sobel", "canny", ...):
        1. Генерирует изображение с указанным паттерном границ.
        2. Выполняет `n_runs_per_pattern` прогонов с замером времени.
        3. Рассчитывает статистику выполнения.

        Args:
            segmenters_dict: Словарь `{имя_метода: экземпляр_сегментера}`.
            edge_patterns: Список имён паттернов для генерации тестовых изображений.
            n_runs_per_pattern: Количество прогонов на каждый паттерн.

        Returns:
            Dict[str, PatternResults]: Результаты по методам:
            ```python
            {
                "sobel_method": {
                    "patterns": {
                        "horizontal": {"mean_ms": 8.12, "std_ms": 0.45, "n_runs": 3},
                        "vertical": {"mean_ms": 8.34, "std_ms": 0.52, "n_runs": 3},
                    }
                }
            }
            ```

        Note:
            - Паттерны генерируются через `_create_edge_pattern()`.
            - Для паттерна "noise" создаётся цветное изображение (H×W×3), остальные — градации серого (H×W).
        """
        edge_keywords: List[str] = [
            "sobel_edge",
            "sobel",
            "canny_edge",
            "canny",
            "prewitt_edge",
            "prewitt",
            "scharr_edge",
            "scharr",
            "roberts_cross_edge",
            "roberts",
            "laplacian_edge",
            "laplacian",
            "log_edge",
            "log",
            "dog_edge",
            "dog",
            "marr_hildreth_edge",
            "marr_hildreth",
            "gradient_magnitude",
            "gradient",
            "phase_congruency_edge",
            "phase_congruency",
        ]
        results: Dict[str, PatternResults] = {}

        print("\n🔥 WARM-UP ГРАНИЧНЫХ МЕТОДОВ")
        print("=" * 60)

        for name, segmenter in segmenters_dict.items():
            name_lower = name.lower()
            is_edge: bool = any(em in name_lower for em in edge_keywords)
            # exclude_keywords = ["kmeans", "dbscan", "meanshift", "neural", "segformer"]
            # if any(excl in name_lower for excl in exclude_keywords):
            #     is_edge = False
            if not is_edge:
                continue

            method_results: PatternResults = {"patterns": {}}

            for pattern in edge_patterns:
                img: np.ndarray = ThresholdWarmUp._create_edge_pattern(256, 256, pattern)

                times: List[float] = []
                for _ in range(n_runs_per_pattern):
                    start: float = time.perf_counter()
                    try:
                        if use_return_info and hasattr(segmenter, "segment"):
                            _, info = segmenter.segment(img, return_info=True)
                        elif hasattr(segmenter, "segment"):
                            segmenter.segment(img)
                            info = segmenter.params.get("execution_info", {})
                        times.append(time.perf_counter() - start)
                    except Exception:
                        times.append(float("inf"))

                method_results["patterns"][pattern] = WarmupMetrics(  # type: ignore[typeddict-item]
                    mean_ms=float(np.mean(times) * 1000),
                    std_ms=float(np.std(times) * 1000),
                    n_runs=len(times),
                )

            # Доступ к метаданным выполнения
            exec_info = segmenter.params.get("execution_info")
            if exec_info:
                print(
                    f"   📊 {name}: {exec_info.get('method', 'N/A')} — {exec_info.get('execution_time', 0)*1000:.2f}ms"
                )
            else:
                # Пытаемся получить из атрибута .info (для OpenCV/Sklearn)
                exec_info = getattr(segmenter, "info", None)
                if exec_info:
                    print(
                        f"   📊 {name}: {exec_info.get('method', 'N/A')} — {exec_info.get('execution_time', 0)*1000:.2f}ms"
                    )
            results[name] = method_results
            print(f"✅ {name}: {method_results['patterns']}")

            if reset_cuda_after_method and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass

        return results

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _create_edge_pattern(h: int, w: int, pattern: str) -> np.ndarray:
        """Создаёт тестовые паттерны для граничных методов.

        Поддерживаемые паттерны:
        - `"horizontal"`: Горизонтальная белая полоса по центру.
        - `"vertical"`: Вертикальная белая полоса по центру.
        - `"diagonal"`: Диагональная линия из единичных пикселей.
        - `"noise"`: Случайный цветной шум (для тестирования устойчивости).
        - Другие значения: Шахматная доска (черно-белая).

        Args:
            h: Высота изображения.
            w: Ширина изображения.
            pattern: Имя паттерна.

        Returns:
            np.ndarray:
            - Для `"noise"`: массив формы `(h, w, 3)`, dtype `uint8`.
            - Для остальных: массив формы `(h, w)`, dtype `uint8` (градации серого).
        """
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
