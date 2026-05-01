# utils/warmup.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import time
import numpy as np
import torch
from typing import (
    TypedDict,
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Literal,
)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & TYPEDDICTS
# ──────────────────────────────────────────────────────────────────────
SegmenterLike = Any  # Объект с методом .segment() или .segment_with_mask()
ImagePattern = Literal["gradient", "noise", "checkerboard", "circles"]


class WarmupStats(TypedDict):
    """
    Статистика warm-up для одного метода.

    Attributes:
        method: Имя метода.
        n_runs: Количество успешных прогонов.
        median_time_ms: Медианное время выполнения (мс).
        mean_time_ms: Среднее время выполнения (мс).
        std_time_ms: Стандартное отклонение времени (мс).
        min_time_ms: Минимальное время (мс).
        max_time_ms: Максимальное время (мс).
    """

    method: str
    n_runs: int
    median_time_ms: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float


class SegmentationWarmUp:
    """
    Универсальный warm-up для классических и нейросетевых методов сегментации.

    Предназначен для:
    - Прогрева кэшей, JIT-компиляции и CUDA kernels перед бенчмарком.
    - Оценки стабильности времени выполнения (mean/std/min/max).
    - Автоматической адаптации под CPU/CUDA устройства.

    Особенности:
    - Поддержка различных тестовых паттернов: `gradient`, `noise`, `checkerboard`, `circles`.
    - Специальный CUDA warm-up с синхронизацией и очисткой кэша.
    - Обработка исключений: сбойный прогон не останавливает весь warm-up.
    - Возврат типизированной статистики для дальнейшего анализа.

    Example:
        ```python
        warmup = SegmentationWarmUp(n_warmup_runs=5, image_size=(256, 256))
        stats = warmup.warmup_segmenter(
            segmenter=my_segmenter,
            method_name="otsu",
            verbose=True,
        )
        print(f"Mean time: {stats['mean_time_ms']:.2f}ms ± {stats['std_time_ms']:.2f}ms")
        ```
    """

    def __init__(
        self,
        n_warmup_runs: int = 10,
        image_size: Tuple[int, int] = (256, 256),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Инициализация утилиты warm-up.

        Args:
            n_warmup_runs: Количество прогонов для каждого метода.
            image_size: Размер тестовых изображений `(высота, ширина)`.
            device: Устройство для вычислений (`"cuda"` или `"cpu"`).
        """
        self.n_warmup_runs: int = n_warmup_runs
        self.image_size: Tuple[int, int] = image_size
        self.device: str = device
        self.warmup_results: Dict[str, List[float]] = {}

    def create_test_image(self, pattern: ImagePattern = "gradient") -> np.ndarray:
        """
        Создаёт тестовое изображение для warm-up.

        Поддерживаемые паттерны:
        - `"gradient"`: Градиент по горизонтали/вертикали (для пороговых методов).
        - `"noise"`: Случайный цветной шум (для граничных методов).
        - `"checkerboard"`: Шахматная доска (для детекции контуров).
        - `"circles"`: Концентрические круги (для тестирования кривизны границ).

        Args:
            pattern: Тип паттерна.

        Returns:
            np.ndarray: Изображение формы `(H, W, 3)`, dtype `uint8`, диапазон `[0, 255]`.
        """
        h, w = self.image_size

        if pattern == "gradient":
            # Градиент для тестирования пороговых методов
            img: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :, 0] = np.tile(np.linspace(0, 255, w), (h, 1)).astype(np.uint8)
            img[:, :, 1] = np.tile(
                np.linspace(0, 255, h).reshape(-1, 1), (1, w)
            ).astype(np.uint8)
            img[:, :, 2] = (img[:, :, 0] + img[:, :, 1]) // 2

        elif pattern == "noise":
            # Шум для тестирования граничных методов
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)  # type: ignore[assignment]

        elif pattern == "checkerboard":
            # Шахматная доска для тестирования контуров
            img = np.zeros((h, w, 3), dtype=np.uint8)
            square_size = 16
            for i in range(0, h, square_size):
                for j in range(0, w, square_size):
                    if (i // square_size + j // square_size) % 2 == 0:
                        img[i : (i + square_size), j : (j + square_size)] = 255

        elif pattern == "circles":
            # Круги для тестирования детекции границ
            img = np.ones((h, w, 3), dtype=np.uint8) * 255
            center_y, center_x = h // 2, w // 2
            for radius in [20, 40, 60, 80]:
                y, x = np.ogrid[:h, :w]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
                img[mask] = 0
        else:
            # Fallback: случайный шум
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)  # type: ignore[assignment]
        return img

    def warmup_segmenter(
        self,
        segmenter: SegmenterLike,
        method_name: str,
        real_image: Optional[np.ndarray] = None,
        verbose: bool = True,
        use_real_image: bool = False,
    ) -> WarmupStats:
        """
        Прогрев конкретного сегментера.

        Логика:
        1. Выбирает изображение: `real_image` (если задано и `use_real_image=True`) или сгенерированное.
        2. Выполняет `n_warmup_runs` прогонов с замером времени через `time.perf_counter()`.
        3. Для CUDA-сегментеров вызывает `_warmup_cuda()` для дополнительной синхронизации.
        4. Рассчитывает статистику: mean, std, min, max, median.

        Args:
            segmenter: Экземпляр сегментера с методом `.segment()` или `.segment_with_mask()`.
            method_name: Имя метода для логирования и ключа в результатах.
            real_image: Реальное изображение для тестирования (опционально).
            verbose: Если `True`, выводит прогресс в консоль.
            use_real_image: Если `True`, использует `real_image` вместо сгенерированного.

        Returns:
            WarmupStats: Словарь со статистикой времени выполнения (в миллисекундах).

        Raises:
            AttributeError: Если у сегментера нет ни `.segment()`, ни `.segment_with_mask()`.
        """
        # Выбор изображения
        if use_real_image and real_image is not None:
            image: np.ndarray = real_image
        else:
            image = self.create_test_image(pattern="gradient")

        warmup_times: List[float] = []

        if verbose:
            print(f"🔥 Warm-up: {method_name} ({self.n_warmup_runs} runs)")

        for i in range(self.n_warmup_runs):
            start_time: float = time.perf_counter()
            try:
                result: SegmenterLike
                if hasattr(segmenter, "segment_with_mask"):
                    result, mask = segmenter.segment_with_mask(image)
                elif hasattr(segmenter, "segment"):
                    result = segmenter.segment(image)
                else:
                    raise AttributeError(
                        "Segmenter must have 'segment' or 'segment_with_mask' method"
                    )
                print(result)
                end_time: float = time.perf_counter()
                warmup_times.append(end_time - start_time)
                if verbose and i == 0:
                    print(f"   ✅ Run 1: {warmup_times[-1] * 1000:.2f}ms")
            except Exception as e:
                print(f"   ❌ Warm-up failed: {e}")
                warmup_times.append(float("inf"))
                break

        # Специальный warm-up для Torch методов (CUDA)
        if hasattr(segmenter, "device") and "cuda" in str(segmenter.device).lower():
            self._warmup_cuda(segmenter, image, verbose)
        self.warmup_results[method_name] = warmup_times

        stats: WarmupStats = {
            "method": method_name,
            "n_runs": len(warmup_times),
            "median_time_ms": (
                float(np.median(warmup_times) * 1000) if warmup_times else float("inf")
            ),
            "mean_time_ms": (
                float(np.mean(warmup_times) * 1000) if warmup_times else float("inf")
            ),
            "std_time_ms": (
                float(np.std(warmup_times) * 1000) if warmup_times else float("inf")
            ),
            "min_time_ms": (
                np.min(warmup_times) * 1000 if warmup_times else float("inf")
            ),
            "max_time_ms": (
                np.max(warmup_times) * 1000 if warmup_times else float("inf")
            ),
        }

        if verbose:
            print(
                f"   📊 Mean: {stats['mean_time_ms']:.2f}ms ± {stats['std_time_ms']:.2f}ms"
            )

        return stats

    def _warmup_cuda(
        self,
        segmenter: SegmenterLike,
        image: np.ndarray,
        verbose: bool = True,
    ) -> None:
        """
        Специальный warm-up для CUDA kernels (Torch-сегментеры).

        Выполняет:
        1. `torch.cuda.synchronize()` перед прогонами.
        2. Дополнительные `n_warmup_runs` вызовов без замера времени.
        3. `torch.cuda.synchronize()` и `torch.cuda.empty_cache()` после.

        Это необходимо для:
        - Инициализации CUDA context и выделения памяти.
        - Прогрева JIT-компилированных ядер (если используются).
        - Стабилизации времени выполнения для последующих бенчмарков.

        Args:
            segmenter: Экземпляр Torch-сегментера.
            image: Тестовое изображение.
            verbose: Если `True`, выводит статус в консоль.
        """
        if verbose:
            print("   🔥 CUDA warm-up...")

        # Синхронизация перед warm-up
        torch.cuda.synchronize()

        # Несколько дополнительных прогонов для CUDA
        for _ in range(self.n_warmup_runs):
            try:
                if hasattr(segmenter, "segment_with_mask"):
                    segmenter.segment_with_mask(image)
                elif hasattr(segmenter, "segment"):
                    segmenter.segment(image)
            except Exception:
                pass

        # Синхронизация после warm-up
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if verbose:
            print("   ✅ CUDA kernels warmed up")

    def warmup_all_segmenters(
        self,
        segmenters_dict: Dict[str, SegmenterLike],
        image: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Прогрев всех сегментеров в словаре.

        Args:
            segmenters_dict: Словарь `{имя_метода: экземпляр_сегментера}`.
            image: Общее тестовое изображение для всех методов (опционально).
            verbose: Если `True`, выводит прогресс в консоль.

        Returns:
            Dict[str, Any]: Результаты по методам:
            - При успехе: `WarmupStats` (см. `warmup_segmenter()`).
            - При ошибке: `{"error": str}`.
        """
        all_results: Dict[str, Any] = {}

        print("\n" + "=" * 60)
        print("🔥 WARM-UP ВСЕХ МЕТОДОВ СЕГМЕНТАЦИИ")
        print("=" * 60)

        for name, segmenter in segmenters_dict.items():
            try:
                stats: WarmupStats = self.warmup_segmenter(
                    segmenter, name, image, verbose
                )
                all_results[name] = stats
            except Exception as e:
                print(f"   ❌ {name}: {e}")
                all_results[name] = {"error": str(e)}

        print("\n" + "=" * 60)
        print("✅ WARM-UP ЗАВЕРШЁН")
        print("=" * 60)
        return all_results

    def get_warmup_summary(self) -> str:
        """
        Возвращает текстовую сводку результатов warm-up.

        Returns:
            str: Форматированный отчёт с методом, средним временем и стандартным отклонением.
        """
        if not self.warmup_results:
            return "No warm-up results available"

        lines: List[str] = ["\n📊 WARM-UP SUMMARY", "=" * 60]

        for method, times in self.warmup_results.items():
            if isinstance(times, dict) and "error" in times:
                lines.append(f"❌ {method}: {times['error']}")
            else:
                mean_ms: float = float(np.mean(times) * 1000)
                std_ms: float = float(np.std(times) * 1000)
                lines.append(f"✅ {method}: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
        return "\n".join(lines)
