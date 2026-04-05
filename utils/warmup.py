# utils/warmup.py

# Импорт основных библиотек
import time
import numpy as np
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)
import torch


class SegmentationWarmUp:
    """
    Техника warm-up для классических методов сегментации.
    Прогревает кэши, JIT-компиляцию и CUDA kernels перед бенчмарком.
    """

    def __init__(
        self,
        n_warmup_runs: int = 10,
        image_size: Tuple[int, int] = (256, 256),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.n_warmup_runs = n_warmup_runs
        self.image_size = image_size
        self.device = device
        self.warmup_results: Dict[str, List[float]] = {}

    def create_test_image(self, pattern: str = "gradient") -> np.ndarray:
        """
        Создаёт тестовое изображение для warm-up.

        Args:
            pattern: Тип паттерна ("gradient", "noise", "checkerboard", "circles")

        Returns:
            np.ndarray: Тестовое изображение (H, W, 3) uint8
        """
        h, w = self.image_size

        if pattern == "gradient":
            # Градиент для тестирования пороговых методов
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :, 0] = np.tile(np.linspace(0, 255, w), (h, 1)).astype(np.uint8)
            img[:, :, 1] = np.tile(
                np.linspace(0, 255, h).reshape(-1, 1), (1, w)
            ).astype(np.uint8)
            img[:, :, 2] = (img[:, :, 0] + img[:, :, 1]) // 2

        elif pattern == "noise":
            # Шум для тестирования граничных методов
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        elif pattern == "checkerboard":
            # Шахматная доска для тестирования контуров
            img = np.zeros((h, w, 3), dtype=np.uint8)
            square_size = 16
            for i in range(0, h, square_size):
                for j in range(0, w, square_size):
                    if (i // square_size + j // square_size) % 2 == 0:
                        img[i:(i + square_size), j:(j + square_size)] = 255

        elif pattern == "circles":
            # Круги для тестирования детекции границ
            img = np.ones((h, w, 3), dtype=np.uint8) * 255
            center_y, center_x = h // 2, w // 2
            for radius in [20, 40, 60, 80]:
                y, x = np.ogrid[:h, :w]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
                img[mask] = 0
        else:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return img

    def warmup_segmenter(
        self,
        segmenter: Any,
        method_name: str,
        real_image: Optional[np.ndarray] = None,
        verbose: bool = True,
        use_real_image: bool = False,
    ) -> Dict[str, float]:
        """
        Прогрев конкретного сегментера.

        Args:
            segmenter: Экземпляр сегментера (OpenCV/Sklearn/Torch)
            method_name: Имя метода для логирования
            image: Тестовое изображение (если None, создаётся автоматически)
            verbose: Логировать ли процесс

        Returns:
            Dict с временем warm-up и статистикой
        """
        if use_real_image and real_image is not None:
            image = real_image
        else:
            image = self.create_test_image(pattern="gradient")

        warmup_times = []

        if verbose:
            print(f"🔥 Warm-up: {method_name} ({self.n_warmup_runs} runs)")

        for i in range(self.n_warmup_runs):
            start_time = time.perf_counter()
            try:
                if hasattr(segmenter, "segment_with_mask"):
                    result, mask = segmenter.segment_with_mask(image)
                elif hasattr(segmenter, "segment"):
                    result = segmenter.segment(image)
                else:
                    raise AttributeError(
                        "Segmenter must have 'segment' or 'segment_with_mask' method"
                    )
                print(result)
                end_time = time.perf_counter()
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

        stats = {
            "method": method_name,
            "n_runs": len(warmup_times),
            "median_time_ms": (
                np.median(warmup_times) * 1000 if warmup_times else float("inf")
            ),
            "mean_time_ms": (
                np.mean(warmup_times) * 1000 if warmup_times else float("inf")
            ),
            "std_time_ms": (
                np.std(warmup_times) * 1000 if warmup_times else float("inf")
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
        self, segmenter: Any, image: np.ndarray, verbose: bool = True
    ) -> None:
        """
        Специальный warm-up для CUDA kernels (Torch сегментеры).
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
        segmenters_dict: Dict[str, Any],
        image: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Прогрев всех сегментеров в словаре.

        Args:
            segmenters_dict: {name: segmenter_object}
            image: Тестовое изображение
            verbose: Логирование

        Returns:
            Dict с результатами warm-up для каждого метода
        """
        all_results = {}

        print("\n" + "=" * 60)
        print("🔥 WARM-UP ВСЕХ МЕТОДОВ СЕГМЕНТАЦИИ")
        print("=" * 60)

        for name, segmenter in segmenters_dict.items():
            try:
                stats = self.warmup_segmenter(segmenter, name, image, verbose)
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
        """
        if not self.warmup_results:
            return "No warm-up results available"

        lines = ["\n📊 WARM-UP SUMMARY", "=" * 60]

        for method, times in self.warmup_results.items():
            if isinstance(times, dict) and "error" in times:
                lines.append(f"❌ {method}: {times['error']}")
            else:
                mean_ms = np.mean(times) * 1000
                std_ms = np.std(times) * 1000
                lines.append(f"✅ {method}: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
        return "\n".join(lines)
