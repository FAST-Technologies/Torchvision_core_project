# segmenters/OpenCVSegmenter.py

# Импорт основных библиотек
from segmenters.BaseSegmenter import BaseSegmenter

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
import warnings
from collections import deque
from scipy import ndimage
from scipy.ndimage import gaussian_filter, laplace
import time

from sklearn.cluster import (
    DBSCAN,
)


class OpenCVSegmenter(BaseSegmenter):
    """
    Класс для реализации методов сегментации изображений с использованием чистого OpenCV.
    Поддерживает как классические методы (пороговые, граничные), так и методы на основе кластеризации,
    активных контуров и графов.
    """

    def __init__(self, method: str = "global_thresholding", **kwargs) -> None:
        super().__init__()
        self.method: str = method
        self.raw_params: Dict[str, Any] = kwargs
        self.params = self._adapt_params(kwargs.copy())
        self._needs_gray: bool = method in [
            "global_thresholding",
            "adaptive_thresholding",
            "otsu_thresholding",
            "region_growing",
            "split_and_merge",
            "sobel_edge",
            "canny_edge",
            "active_contour",
            "gvf_contour",
            "watershed",
            "meanshift",
            "grabcut",
            "floodfill",
            "morphological_snakes",
            "chan_vese",
            "threshold_niblack",
            "threshold_sauvola",
            "random_walker",
            "threshold_bernsen",
            "threshold_phansalkar",
            "threshold_kittler_illingworth",
            "threshold_entropy_kapur",
            "threshold_triangle",
            "threshold_multi_otsu",
            "threshold_percentile",
            "threshold_local_contrast",
            "prewitt_edge",
            "scharr_edge",
            "roberts_cross_edge",
            "log_edge",
            "dog_edge",
            "marr_hildreth_edge",
            "gradient_magnitude_direction",
            "phase_congruency_edge",
        ]
        self._setup_methods()

    def _adapt_params(self, params: dict) -> dict:
        """Конвертирует значения и приводит имена параметров к стандарту OpenCV."""
        adapted = params.copy()

        # Конвертация значений (0.0-1.0 -> 0-255) ---
        # Параметры, которые точно являются порогами яркости
        intensity_params = ["threshold", "low", "high", "t1", "t2"]
        for key in intensity_params:
            if key in adapted:
                val = adapted[key]
                if isinstance(val, (int, float)) and 0.0 <= val <= 1.0:
                    adapted[key] = int(val * 255)

        # Параметры, зависящие от интенсивности (смещения)
        offset_params = ["C", "tolerance", "c", "k"]
        for key in offset_params:
            if key in adapted:
                val = adapted[key]
                if isinstance(val, (int, float)) and 0.0 <= abs(val) <= 1.0:
                    adapted[key] = int(val * 255)

        mapping = {}
        if self.method == "grabcut":
            mapping = {"n_iterations": "iterations"}
        elif self.method == "dbscan_segmentation":
            mapping = {"epsilon": "eps", "min_points": "min_samples"}
        elif self.method == "kmeans_segmentation":
            mapping = {"n_clusters": "k"}
        elif self.method == "adaptive_thresholding":
            pass

        final_params = {}
        for key, value in adapted.items():
            new_key = mapping.get(key, key)
            final_params[new_key] = value

        return final_params

    def _setup_methods(self, **kwargs) -> None:
        """Регистрация всех доступных методов сегментации."""
        self.methods: Dict[str, Callable[..., np.ndarray]] = {
            # ============ ПОРОГОВЫЕ МЕТОДЫ СЕГМЕНТАЦИИ ============
            "global_thresholding": self._opencv_global_thresholding,
            "adaptive_thresholding": self._opencv_adaptive_thresholding,
            "otsu_thresholding": self._opencv_otsu_thresholding,
            "threshold_niblack": self._opencv_threshold_niblack,
            "threshold_sauvola": self._opencv_threshold_sauvola,
            "threshold_bernsen": self._opencv_threshold_bernsen,
            "threshold_phansalkar": self._opencv_threshold_phansalkar,
            "threshold_kittler_illingworth": self._opencv_threshold_kittler_illingworth,
            "threshold_entropy_kapur": self._opencv_threshold_entropy_kapur,
            "threshold_triangle": self._opencv_threshold_triangle,
            "threshold_multi_otsu": self._opencv_threshold_multi_otsu,
            "threshold_percentile": self._opencv_threshold_percentile,
            "threshold_local_contrast": self._opencv_threshold_local_contrast,
            # ============ КРАЕВЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "sobel_edge": self._opencv_sobel_edge,
            "canny_edge": self._opencv_canny_edge,
            "prewitt_edge": self._opencv_prewitt_edge,
            "scharr_edge": self._opencv_scharr_edge,
            "roberts_cross_edge": self._opencv_roberts_cross_edge,
            "log_edge": self._opencv_log_edge,
            "dog_edge": self._opencv_dog_edge,
            "marr_hildreth_edge": self._opencv_marr_hildreth_edge,
            "gradient_magnitude_direction": self._opencv_gradient_magnitude_direction,
            "phase_congruency_edge": self._opencv_phase_congruency_edge,
            # ============ РЕГИОНАЛЬНЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "region_growing": self._opencv_region_growing,
            "split_and_merge": self._opencv_split_and_merge,
            "floodfill": self._opencv_floodfill,
            # ============ КЛАСТЕРИЗАЦИЯ ============
            "kmeans_segmentation": self._opencv_kmeans_segmentation,
            "dbscan_segmentation": self._opencv_dbscan_segmentation,
            "meanshift": self._opencv_meanshift,
            # ============ АКТИВНЫЕ КОНТУРЫ ==========
            "active_contour": self._opencv_active_contour,
            "gvf_contour": self._opencv_gvf_contour,
            "morphological_snakes": self._opencv_morphological_snakes,
            "chan_vese": self._opencv_chan_vese,
            # ============ WATERSHED И ГРАФОВЫЕ ============
            "watershed": self._opencv_watershed,
            "random_walker": self._opencv_random_walker,
            # ============ SUPER-PIXEL МЕТОДЫ ===========
            "quickshift": self._opencv_quickshift,
            "slic": self._opencv_slic,
            "felzenszwalb": self._opencv_felzenszwalb,
            # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============
            "grabcut": self._opencv_grabcut,
        }

        if self.method not in self.methods:
            raise ValueError(
                f"Неизвестный метод: {self.method}. "
                f"Доступные методы: {list(self.methods.keys())}"
            )

    def _log_info(
        self, method_name: str, exec_time: float, params: Dict[str, Any]
    ) -> None:
        """Вспомогательный метод для логирования информации о выполнении."""
        self.info = {
            "method": method_name,
            "parameters": params,
            "execution_time": exec_time,
        }

    def segment(  # type: ignore[override]
        self, image: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Основной метод сегментации.

        Args:
            image: Входное изображение (RGB, grayscale или любой формат)

        Returns:
            np.ndarray: Бинарная маска сегментации (0-255)
        """
        img_array: np.ndarray = self.preprocess_image(image, as_gray=self._needs_gray)
        # print(f"Image after OpenCV preprocessing: {image}")

        mask = self.methods[self.method](img_array, **kwargs)
        # print(f"Mask after OpenCV segment: {mask}")
        return mask

    def segment_with_mask(  # type: ignore[override]
        self, image: np.ndarray, alpha: float = 0.9, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сегментация с возвратом визуализации и маски.

        Args:
            image: Входное изображение

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Визуализация: исходное изображение с наложенной маской (0–255, RGB).
                - Маска: бинарная маска (0–255, grayscale).
        """
        image = self.preprocess_image(image)
        # print(f"Image after OpenCV preprocessing with mask: {image}")
        mask: np.ndarray = self.segment(image, **kwargs)

        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

        # Создаем визуализацию
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = image.copy()

        overlay[mask > 127] = [255, 0, 0]
        result = cv2.addWeighted(
            overlay,
            alpha,
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image,
            1 - alpha,
            0,
        )

        # print(f"Mask after OpenCV segment_with_mask: {mask}")
        # print(f"Result after OpenCV segment_with_mask: {result}")
        return result, mask

    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    # ============ ПОРОГОВЫЕ МЕТОДЫ ============

    def _opencv_global_thresholding(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Глобальная пороговая сегментация.

        Применяет фиксированный порог ко всему изображению.
        Все пиксели яркостью выше порога становятся белыми (объект), остальные — черными (фон).

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            Бинарная маска (0/255).
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # print(f"Gray after OpenCV_thresholding_global: {gray}")

        start_time = time.time()

        threshold: float = self.params.get("threshold", 127)
        _, mask = cv2.threshold(gray, float(threshold), 255.0, cv2.THRESH_BINARY)

        exec_time = time.time() - start_time

        info = {
            "method": "global_thresholding_opencv",
            "parameters": {"threshold": threshold, **kwargs},
            "execution_time": exec_time,
        }

        # print(f"Mask after OpenCV_thresholding_global: {mask}")
        print(f"Info after OpenCV_thresholding_global: {info}")

        return mask

    def _opencv_adaptive_thresholding(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Адаптивная пороговая сегментация (Gaussian).

        Вычисляет локальный порог для каждой области изображения.
        Особенно эффективна при неравномерном освещении.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # print(f"Gray after OpenCV_thresholding_adaptive: {gray}")

        start_time = time.time()

        block_size = self.params.get("block_size", 11)
        C = self.params.get("C", 2)

        if block_size % 2 == 0:
            block_size += 1

        mask = cv2.adaptiveThreshold(
            gray,
            255.0,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C,
        )

        exec_time = time.time() - start_time

        info = {
            "method": "adaptive_thresholding_opencv",
            "parameters": {"block_size": block_size, "C": C, **kwargs},
            "execution_time": exec_time,
        }

        # print(f"Mask after OpenCV_thresholding_adaptive: {mask}")
        print(f"Info after OpenCV_thresholding_adaptive: {info}")
        return mask

    def _opencv_otsu_thresholding(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Автоматическая бинаризация по методу Оцу.

        Находит оптимальный порог, максимизирующий межклассовую дисперсию между фоном и объектом.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # print(f"Gray after OpenCV_thresholding_otsu: {gray}")
        start_time = time.time()

        _, mask = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        exec_time = time.time() - start_time

        info = {
            "method": "otsu_thresholding_opencv",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }

        # print(f"Mask after OpenCV_thresholding_otsu: {mask}")
        print(f"Info after OpenCV_thresholding_otsu: {info}")
        return mask

    def _opencv_threshold_niblack(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Адаптивная пороговая обработка по Ниблаку.

        Порог вычисляется как: T = μ + k·σ, где μ и σ — локальное среднее и СКО.
        Хорошо работает на изображениях с шумом и градиентом освещения.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # print(f"Gray after OpenCV_thresholding_niblack: {gray}")

        start_time = time.time()

        window_size = self.params.get("window_size", 15)
        k = self.params.get("k", -0.2)

        # Вычисляем локальное среднее и стандартное отклонение
        mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
        mean_sq = cv2.boxFilter(
            gray.astype(np.float32) ** 2, cv2.CV_32F, (window_size, window_size)
        )
        std = np.sqrt(mean_sq - mean**2)

        # Или
        # mean = cv2.blur(gray, (window_size, window_size))
        # std = np.sqrt(cv2.boxFilter(gray.astype(float)**2, -1, (window_size, window_size)) - mean**2)

        # Вычисляем порог
        threshold = mean + k * std
        mask = (gray.astype(np.float32) > threshold).astype(np.uint8) * 255

        exec_time = time.time() - start_time

        info = {
            "method": "niblack_thresholding_opencv",
            "parameters": {"window_size": window_size, "k": k, **kwargs},
            "execution_time": exec_time,
        }

        # print(f"Mask after OpenCV_thresholding_niblack: {mask}")
        print(f"Info after OpenCV_thresholding_niblack: {info}")

        return mask

    def _opencv_threshold_sauvola(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Улучшенная адаптивная пороговая обработка по Сауволе.

        Порог: T = μ·(1 + k·(σ/R - 1)), где R — динамический диапазон (обычно 128).
        Лучше Ниблака при очень низком контрасте.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # print(f"Gray after OpenCV_thresholding_sauvola: {gray}")

        start_time = time.time()

        window_size = self.params.get("window_size", 15)
        k = self.params.get("k", 0.5)
        r = self.params.get("r", 128)

        # Вычисляем локальное среднее и стандартное отклонение
        mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
        mean_sq = cv2.boxFilter(
            gray.astype(np.float32) ** 2, cv2.CV_32F, (window_size, window_size)
        )
        std = np.sqrt(mean_sq - mean**2)

        # mean = cv2.blur(gray, (window_size, window_size))
        # std = np.sqrt(cv2.boxFilter(gray.astype(float)**2, -1, (window_size, window_size)) - mean**2)

        # Вычисляем порог
        threshold = mean * (1 + k * (std / r - 1))
        mask = (gray.astype(np.float32) > threshold).astype(np.uint8) * 255

        exec_time = time.time() - start_time

        info = {
            "method": "sauvola_thresholding_opencv",
            "parameters": {"window_size": window_size, "k": k, "r": r, **kwargs},
            "execution_time": exec_time,
        }

        # print(f"Mask after OpenCV_thresholding_sauvola: {mask}")
        print(f"Info after OpenCV_thresholding_sauvola: {info}")

        return mask

    def _opencv_threshold_bernsen(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Пороговая обработка по методу Бернсена.

        Локальный адаптивный порог на основе контраста в окне.
        T = (min + max) / 2, если контраст > порог, иначе фон.

        Args:
            img: Входное изображение (grayscale)
            window_size: Размер окна для локального анализа (по умолчанию 15)
            contrast_threshold: Минимальный контраст для разделения (по умолчанию 25)

        Returns:
            np.ndarray: Бинарная маска (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        window_size: int = self.params.get("window_size", 15)
        contrast_threshold: int = self.params.get("contrast_threshold", 25)

        if window_size % 2 == 0:
            window_size += 1

        # Вычисляем локальные min и max
        min_filter = cv2.erode(gray, np.ones((window_size, window_size), np.uint8))
        max_filter = cv2.dilate(gray, np.ones((window_size, window_size), np.uint8))

        # Контраст
        contrast = max_filter - min_filter

        # Порог Бернсена
        threshold = (
            min_filter.astype(np.float32) + max_filter.astype(np.float32)
        ) / 2.0
        # Бинаризация
        mask = np.zeros_like(gray, dtype=np.uint8)
        high_contrast = contrast >= contrast_threshold
        mask[high_contrast] = (gray[high_contrast] > threshold[high_contrast]).astype(
            np.uint8
        ) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "bernsen_thresholding_opencv",
            exec_time,
            {
                "window_size": window_size,
                "contrast_threshold": contrast_threshold,
                **kwargs,
            },
        )

        return mask

    def _opencv_threshold_phansalkar(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Пороговая обработка по методу Фансалкара.

        Улучшенная версия Ниблака для документов с низким контрастом.
        T = μ + k·σ·(σ/R) + m·(μ/128 - 1), где μ, σ — локальные среднее и СКО.

        Проверить T = μ·[1 + p·(σ/R - 1)]

        Args:
            img: Входное изображение (grayscale)
            window_size: Размер окна (по умолчанию 15)
            k: Параметр чувствительности (по умолчанию 0.25)
            r: Динамический диапазон (по умолчанию 0.5)
            m: Параметр смещения (по умолчанию 0.5)

        Returns:
            np.ndarray: Бинарная маска (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        window_size: int = self.params.get("window_size", 15)
        k: float = self.params.get("k", 0.25)
        r: float = self.params.get("r", 0.5)
        m: float = self.params.get("m", 0.5)

        if window_size % 2 == 0:
            window_size += 1

        # Локальные статистики
        mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
        mean_sq = cv2.boxFilter(
            gray.astype(np.float32) ** 2, cv2.CV_32F, (window_size, window_size)
        )
        std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
        # Порог Фансалкара
        threshold = mean + k * std * (std / r) + m * (mean / 128.0 - 1)

        mask = (gray.astype(np.float32) > threshold).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "phansalkar_thresholding_opencv",
            exec_time,
            {"window_size": window_size, "k": k, "r": r, "m": m, **kwargs},
        )

        return mask

    def _opencv_threshold_kittler_illingworth(
        self, img: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Пороговая обработка по методу Киттлера-Иллингуорта.

        Минимизация ошибки классификации на основе гистограммы.
        Предполагает бимодальное распределение интенсивностей.

        Args:
            img: Входное изображение (grayscale)
            num_bins: Количество бинов гистограммы (по умолчанию 256)

        Returns:
            np.ndarray: Бинарная маска (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        num_bins: int = self.params.get("num_bins", 256)

        # Гистограмма
        hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))
        hist = hist.astype(np.float64)

        # Нормализация
        hist = hist / hist.sum()

        # Кумулятивные суммы
        cum_hist = np.cumsum(hist)
        cum_mean = np.cumsum(hist * np.arange(num_bins))

        total_mean = cum_mean[-1]
        min_error = np.inf
        best_threshold = 128
        # Поиск оптимального порога
        for t in range(1, num_bins - 1):
            if cum_hist[t] < 1e-6 or (1 - cum_hist[t]) < 1e-6:
                continue

            # Статистики для фона и объекта
            w0 = cum_hist[t]
            w1 = 1 - w0
            mu0 = cum_mean[t] / w0
            mu1 = (total_mean - cum_mean[t]) / w1

            # Дисперсии
            cum_mean_sq = np.cumsum(hist * np.arange(num_bins) ** 2)
            sigma0_sq = cum_mean_sq[t] / w0 - mu0**2
            sigma1_sq = (cum_mean_sq[-1] - cum_mean_sq[t]) / w1 - mu1**2

            if sigma0_sq <= 1e-6 or sigma1_sq <= 1e-6:
                continue

            # Критерий Киттлера-Иллингуорта
            error = (
                w0 * np.log(sigma0_sq)
                + w1 * np.log(sigma1_sq)
                - 2 * (w0 * np.log(w0) + w1 * np.log(w1))
            )

            if error < min_error:
                min_error = error
                best_threshold = t

        # Бинаризация
        _, mask = cv2.threshold(gray, float(best_threshold), 255.0, cv2.THRESH_BINARY)

        exec_time: float = time.time() - start_time
        self._log_info(
            "kittler_illingworth_thresholding_opencv",
            exec_time,
            {"num_bins": num_bins, "optimal_threshold": best_threshold, **kwargs},
        )

        return mask

    def _opencv_threshold_entropy_kapur(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Пороговая обработка на основе энтропии Капура.

        Максимизация суммы энтропий фона и объекта.

        Args:
            img: Входное изображение (grayscale)
            num_bins: Количество бинов гистограммы (по умолчанию 256)

        Returns:
            np.ndarray: Бинарная маска (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        num_bins: int = self.params.get("num_bins", 256)

        # Гистограмма
        hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))
        hist = hist.astype(np.float64) + 1e-10  # Избегаем log(0)
        hist = hist / hist.sum()

        # Кумулятивная гистограмма и энтропия
        cum_hist = np.cumsum(hist)
        cum_entropy = np.cumsum(-hist * np.log(hist))

        max_entropy = -np.inf
        best_threshold = 128

        # Поиск порога
        for t in range(1, num_bins - 1):
            if cum_hist[t] < 1e-6 or (1 - cum_hist[t]) < 1e-6:
                continue
            # Энтропия фона
            h0 = cum_entropy[t] / cum_hist[t] + np.log(cum_hist[t])
            # Энтропия объекта
            h1 = (cum_entropy[-1] - cum_entropy[t]) / (1 - cum_hist[t]) + np.log(
                1 - cum_hist[t]
            )

            total_entropy = h0 + h1
            if total_entropy > max_entropy:
                max_entropy = total_entropy
                best_threshold = t

        # Бинаризация
        _, mask = cv2.threshold(gray, float(best_threshold), 255.0, cv2.THRESH_BINARY)

        exec_time: float = time.time() - start_time
        self._log_info(
            "entropy_kapur_thresholding_opencv",
            exec_time,
            {"num_bins": num_bins, "optimal_threshold": best_threshold, **kwargs},
        )

        return mask

    def _opencv_threshold_triangle(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Пороговая обработка треугольным методом.

        Геометрический метод для бимодальных гистограмм.
        Находит порог как точку максимального расстояния от линии пик-минимум.

        Args:
            img: Входное изображение (grayscale)
            num_bins: Количество бинов гистограммы (по умолчанию 256)

        Returns:
            np.ndarray: Бинарная маска (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        num_bins: int = self.params.get("num_bins", 256)

        # Гистограмма
        hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))

        # Находим пик гистограммы
        peak_idx: int = int(np.argmax(hist))

        # Линия от пика до конца диапазона
        y_peak = hist[peak_idx]
        y_end = hist[-1]
        # Уравнение линии: y = mx + b
        m = (
            (y_end - y_peak) / (num_bins - 1 - peak_idx)
            if (num_bins - 1 != peak_idx)
            else 0
        )

        # Находим точку максимального расстояния
        max_dist: float = 0.0
        best_threshold: int = peak_idx

        for t in range(int(peak_idx) + 1, int(num_bins)):
            # Расстояние от точки до линии
            y_line = y_peak + m * (t - peak_idx)
            dist = abs(hist[t] - y_line) / np.sqrt(1 + m**2)

            if dist > max_dist:
                max_dist = float(dist)
                best_threshold = int(t)

        # Бинаризация
        _, mask = cv2.threshold(gray, float(best_threshold), 255.0, cv2.THRESH_BINARY)

        exec_time: float = time.time() - start_time
        self._log_info(
            "triangle_thresholding_opencv",
            exec_time,
            {"num_bins": num_bins, "optimal_threshold": best_threshold, **kwargs},
        )

        return mask

    def _opencv_threshold_multi_otsu(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Многоуровневая пороговая обработка по методу Оцу.

        Расширение метода Оцу для нескольких порогов.

        Args:
            img: Входное изображение (grayscale)
            n_thresholds: Количество порогов (по умолчанию 2)
            num_bins: Количество бинов гистограммы (по умолчанию 256)

        Returns:
            np.ndarray: Бинарная маска (0/255) - объект = самый яркий класс
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        n_thresholds: int = self.params.get("n_thresholds", 2)
        num_bins: int = self.params.get("num_bins", 256)

        # Гистограмма
        hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))
        hist = hist.astype(np.float64)

        if n_thresholds == 1:
            # Обычный Оцу
            _, mask = cv2.threshold(
                gray, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return mask

        # Упрощённый поиск порогов (для 2 порогов)
        if n_thresholds == 2:
            best_var = -np.inf
            best_t1, best_t2 = 64, 192

            cum_sum = np.cumsum(hist)
            cum_mean = np.cumsum(hist * np.arange(num_bins))
            total = cum_sum[-1]
            total_mean = cum_mean[-1] / total

            for t1 in range(1, num_bins - 2):
                for t2 in range(t1 + 1, num_bins - 1):
                    # Класс 0: [0, t1)
                    w0 = cum_sum[t1] / total
                    m0 = cum_mean[t1] / cum_sum[t1] if cum_sum[t1] > 0 else 0

                    # Класс 1: [t1, t2)
                    w1 = (cum_sum[t2] - cum_sum[t1]) / total
                    m1 = (
                        (cum_mean[t2] - cum_mean[t1]) / (cum_sum[t2] - cum_sum[t1])
                        if (cum_sum[t2] > cum_sum[t1])
                        else 0
                    )

                    # Класс 2: [t2, 256)
                    w2 = (total - cum_sum[t2]) / total
                    m2 = (
                        (total_mean * total - cum_mean[t2]) / (total - cum_sum[t2])
                        if (total > cum_sum[t2])
                        else 0
                    )

                    # Межклассовая дисперсия
                    var_between = (
                        w0 * (m0 - total_mean) ** 2
                        + w1 * (m1 - total_mean) ** 2
                        + w2 * (m2 - total_mean) ** 2
                    )

                    if var_between > best_var:
                        best_var = var_between
                        best_t1, best_t2 = t1, t2

            # Бинаризация: объект = самый яркий класс
            mask = (gray >= best_t2).astype(np.uint8) * 255

            exec_time: float = time.time() - start_time
            self._log_info(
                "multi_otsu_thresholding_opencv",
                exec_time,
                {
                    "n_thresholds": n_thresholds,
                    "thresholds": [best_t1, best_t2],
                    **kwargs,
                },
            )
            return mask

        # Для >2 порогов используем рекурсивный Оцу (упрощённо)
        thresholds: List[float] = []
        current_gray = gray.copy()

        for _ in range(n_thresholds):
            _, thresh = cv2.threshold(
                current_gray, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            thresholds.append(float(thresh))
            current_gray = cv2.threshold(
                current_gray, float(thresh), 255.0, cv2.THRESH_BINARY_INV
            )[1]

        # Используем последний порог для бинаризации
        _, mask = cv2.threshold(gray, float(thresholds[-1]), 255.0, cv2.THRESH_BINARY)

        exec_time = time.time() - start_time
        self._log_info(
            "multi_otsu_thresholding_opencv",
            exec_time,
            {"n_thresholds": n_thresholds, "thresholds": thresholds, **kwargs},
        )

        return mask

    def _opencv_threshold_percentile(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Процентильная пороговая обработка.

        Порог выбирается как заданный процентиль распределения интенсивностей гистограммы.

        Args:
            img: Входное изображение (grayscale)
            percentile: Процентиль для порога (по умолчанию 90)

        Returns:
            np.ndarray: Бинарная маска (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        percentile: float = self.params.get("percentile", 90)

        # Вычисляем процентиль
        threshold = np.percentile(gray.astype(np.float32), percentile)

        # Бинаризация
        _, mask = cv2.threshold(gray, int(threshold), 255.0, cv2.THRESH_BINARY)

        exec_time: float = time.time() - start_time
        self._log_info(
            "percentile_thresholding_opencv",
            exec_time,
            {"percentile": percentile, "threshold": int(threshold), **kwargs},
        )

        return mask

    def _opencv_threshold_local_contrast(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Пороговая обработка на основе локального контраста.

        Пиксель считается объектом, если его интенсивность значительно
        отличается от локального среднего.
        T = μ + k·(max-min) в окне

        Args:
            img: Входное изображение (grayscale)
            window_size: Размер окна для локального анализа (по умолчанию 15)
            contrast_factor: Коэффициент контраста (по умолчанию 0.1)

        Returns:
            np.ndarray: Бинарная маска (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        window_size: int = self.params.get("window_size", 15)
        contrast_factor: float = self.params.get("contrast_factor", 0.1)

        if window_size % 2 == 0:
            window_size += 1

        # Локальное среднее
        local_mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))

        # Локальный контраст (разница от среднего)
        local_contrast = np.abs(gray.astype(np.float32) - local_mean)

        # Глобальный порог контраста
        global_contrast_threshold = np.percentile(
            local_contrast, 100 * (1 - contrast_factor)
        )
        # Бинаризация по контрасту
        mask = (local_contrast > global_contrast_threshold).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "local_contrast_thresholding_opencv",
            exec_time,
            {"window_size": window_size, "contrast_factor": contrast_factor, **kwargs},
        )

        return mask

    # ============ МЕТОДЫ НА ОСНОВЕ КРАЕВ ============
    def _opencv_sobel_edge(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Обнаружение границ оператором Собеля.

        Вычисляет градиент интенсивности по горизонтали и вертикали, затем объединяет их.
        Применяется порог к величине градиента для получения бинарной маски границ.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска границ (0/255, dtype=np.uint8).
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # print(f"Gray after OpenCV_sobel_edge: {gray}")

        start_time = time.time()

        threshold = self.params.get("threshold", 50)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        # Или
        # magnitude = cv2.magnitude(sobelx, sobely)
        # sobel_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # _, mask = cv2.threshold(sobel_norm.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)

        _, mask = cv2.threshold(
            magnitude.astype(np.float32), float(threshold), 255.0, cv2.THRESH_BINARY
        )  # type: ignore[call-overload]

        exec_time = time.time() - start_time

        info = {
            "method": "sobel_edge_opencv",
            "parameters": {"threshold": threshold, **kwargs},
            "execution_time": exec_time,
        }

        # print(f"Mask after OpenCV_sobel_edge: {mask}")
        print(f"Info after OpenCV_sobel_edge: {info}")
        return mask.astype(np.uint8)

    def _opencv_canny_edge(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Обнаружение границ оператором Кэнни.

        Многоэтапный алгоритм: сглаживание, вычисление градиента, подавление немаксимумов,
        двойная пороговая фильтрация и отслеживание связных границ.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска границ (0/255, dtype=np.uint8).
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        print(f"Gray after OpenCV_canny_edge: {gray}")

        start_time = time.time()

        low = self.params.get("low", 50)
        high = self.params.get("high", 150)

        mask = cv2.Canny(gray, low, high)

        exec_time = time.time() - start_time

        info = {
            "method": "canny_edge_opencv",
            "parameters": {"low": low, "high": high, **kwargs},
            "execution_time": exec_time,
        }

        print(f"Mask after OpenCV_canny_edge: {mask}")
        print(f"Info after OpenCV_canny_edge: {info}")
        return mask

    def _opencv_prewitt_edge(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Обнаружение границ оператором Превитта.

        Вычисляет градиент с использованием ядер 3×3 (использование весов [1,1,1]).
        Менее чувствителен к шуму, чем Собель.

        Args:
            img: Входное изображение (grayscale)
            threshold: Порог для бинаризации градиента (по умолчанию 50)
            direction: Направление градиента ('x', 'y', 'both') (по умолчанию 'both')

        Returns:
            np.ndarray: Бинарная маска границ (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        threshold: int = self.params.get("threshold", 50)
        direction: str = self.params.get("direction", "both")

        # Ядра Превитта
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

        if direction in ["x", "both"]:
            grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
        else:
            grad_x = np.zeros_like(gray, dtype=np.float32)
        if direction in ["y", "both"]:
            grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
        else:
            grad_y = np.zeros_like(gray, dtype=np.float32)

        # Магнитуда градиента
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-8))

        # Бинаризация
        _, mask = cv2.threshold(magnitude, float(threshold), 255.0, cv2.THRESH_BINARY)  # type: ignore[call-overload]

        exec_time: float = time.time() - start_time
        self._log_info(
            "prewitt_edge_opencv",
            exec_time,
            {"threshold": threshold, "direction": direction, **kwargs},
        )

        return mask

    def _opencv_scharr_edge(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Обнаружение границ оператором Шара.

        Улучшенная версия Собеля с лучшей точностью вычисления градиента.
        Использует оптимизированные ядра 3×3.

        Args:
            img: Входное изображение (grayscale)
            threshold: Порог для бинаризации градиента (по умолчанию 50)
            direction: Направление градиента ('x', 'y', 'both') (по умолчанию 'both')

        Returns:
            np.ndarray: Бинарная маска границ (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        threshold: int = self.params.get("threshold", 50)
        direction: str = self.params.get("direction", "both")

        # Ядра Шара (более точные, чем Собель)
        if direction in ["x", "both"]:
            grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        else:
            grad_x = np.zeros_like(gray, dtype=np.float64)

        if direction in ["y", "both"]:
            grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        else:
            grad_y = np.zeros_like(gray, dtype=np.float64)

        # Магнитуда градиента
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-8))
        # Бинаризация
        _, mask = cv2.threshold(magnitude, float(threshold), 255.0, cv2.THRESH_BINARY)  # type: ignore[call-overload]

        exec_time: float = time.time() - start_time
        self._log_info(
            "scharr_edge_opencv",
            exec_time,
            {"threshold": threshold, "direction": direction, **kwargs},
        )

        return mask

    def _opencv_roberts_cross_edge(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Обнаружение границ оператором Робертса (Cross).

        Простой оператор для обнаружения диагональных границ.
        Использует ядра 2×2 для вычисления градиента.
        Диагональные разности
        [[+1,0],[0,-1]]
        [[0,+1],[-1,0]]

        Args:
            img: Входное изображение (grayscale)
            threshold: Порог для бинаризации градиента (по умолчанию 50)

        Returns:
            np.ndarray: Бинарная маска границ (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        threshold: int = self.params.get("threshold", 50)

        # Ядра Робертса (2×2)
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

        # Магнитуда градиента
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-8))

        # Бинаризация
        _, mask = cv2.threshold(magnitude, float(threshold), 255.0, cv2.THRESH_BINARY)  # type: ignore[call-overload]

        exec_time: float = time.time() - start_time
        self._log_info(
            "roberts_cross_edge_opencv", exec_time, {"threshold": threshold, **kwargs}
        )

        return mask

    def _opencv_log_edge(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Обнаружение границ Лапласианом Гауссиана (LoG / Laplacian of Gaussian).

        Применяет Гауссово размытие, затем Лапласиан.
        Границы обнаруживаются по пересечению нуля (zero-crossing).
        ∇²(G * I) — детекция по нулевым пересечениям

        Args:
            img: Входное изображение (grayscale)
            sigma: Стандартное отклонение Гаусса (по умолчанию 1.0)
            kernel_size: Размер ядра Лапласиана (по умолчанию 5)
            threshold: Порог для бинаризации (по умолчанию 10)

        Returns:
            np.ndarray: Бинарная маска границ (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        sigma: float = self.params.get("sigma", 1.0)
        kernel_size: int = self.params.get("kernel_size", 5)
        threshold: int = self.params.get("threshold", 10)

        if kernel_size % 2 == 0:
            kernel_size += 1

        # Гауссово размытие
        blurred = gaussian_filter(gray.astype(np.float32), sigma=sigma)

        # Лапласиан
        laplacian = laplace(blurred)
        # Zero-crossing detection
        # zero_crossing = np.zeros_like(laplacian, dtype=np.uint8)

        # for i in range(1, laplacian.shape[0] - 1):
        #     for j in range(1, laplacian.shape[1] - 1):
        #         neighborhood = laplacian[i - 1 : i + 2, j - 1 : j + 2]
        #         if np.min(neighborhood) < 0 < np.max(neighborhood):
        #             if (
        #                 np.abs(np.min(neighborhood)) + np.abs(np.max(neighborhood))
        #                 > threshold
        #             ):
        #                 zero_crossing[i, j] = 255

        # exec_time: float = time.time() - start_time
        # self._log_info(
        #     "log_edge_opencv",
        #     exec_time,
        #     {
        #         "sigma": sigma,
        #         "kernel_size": kernel_size,
        #         "threshold": threshold,
        #         **kwargs,
        #     },
        # )

        # Векторизованное zero-crossing: соседние пиксели имеют противоположные знаки
        sign = np.sign(laplacian)
        zc_h = sign[:, :-1] * sign[:, 1:] < 0  # горизонтальное пересечение
        zc_v = sign[:-1, :] * sign[1:, :] < 0  # вертикальное пересечение
        zero_crossing_bool = np.zeros_like(laplacian, dtype=bool)
        zero_crossing_bool[:, :-1] |= zc_h
        zero_crossing_bool[:-1, :] |= zc_v
        # Фильтр по амплитуде (отсекаем слабые пересечения)
        abs_lap = np.abs(laplacian)
        zero_crossing = (zero_crossing_bool & (abs_lap > threshold)).astype(
            np.uint8
        ) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "log_edge_opencv",
            exec_time,
            {
                "sigma": sigma,
                "kernel_size": kernel_size,
                "threshold": threshold,
                **kwargs,
            },
        )

        return zero_crossing

    def _opencv_dog_edge(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Обнаружение границ разностью Гауссианов (DoG / Difference of Gaussians).

        Аппроксимация LoG через разность двух Гауссианов с разными σ.
        Аппроксимация LoG: G₁*I - G₂*I
        Эффективно для обнаружения границ разного масштаба.

        Args:
            img: Входное изображение (grayscale)
            sigma1: Стандартное отклонение первого Гаусса (по умолчанию 1.0)
            sigma2: Стандартное отклонение второго Гаусса (по умолчанию 2.0)
            threshold: Порог для бинаризации (по умолчанию 10)

        Returns:
            np.ndarray: Бинарная маска границ (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        sigma1: float = self.params.get("sigma1", 1.0)
        sigma2: float = self.params.get("sigma2", 2.0)
        threshold: int = self.params.get("threshold", 10)

        # Применяем два Гауссовых фильтра
        g1 = gaussian_filter(gray.astype(np.float32), sigma=sigma1)
        g2 = gaussian_filter(gray.astype(np.float32), sigma=sigma2)

        # Разность Гауссианов
        dog = g1 - g2
        # # Zero-crossing detection
        # zero_crossing = np.zeros_like(dog, dtype=np.uint8)

        # for i in range(1, dog.shape[0] - 1):
        #     for j in range(1, dog.shape[1] - 1):
        #         neighborhood = dog[i - 1 : i + 2, j - 1 : j + 2]
        #         if np.min(neighborhood) < 0 < np.max(neighborhood):
        #             if (
        #                 np.abs(np.min(neighborhood)) + np.abs(np.max(neighborhood))
        #                 > threshold
        #             ):
        #                 zero_crossing[i, j] = 255

        # Векторизованное zero-crossing
        sign = np.sign(dog)
        zc_h = sign[:, :-1] * sign[:, 1:] < 0
        zc_v = sign[:-1, :] * sign[1:, :] < 0
        zero_crossing_bool = np.zeros_like(dog, dtype=bool)
        zero_crossing_bool[:, :-1] |= zc_h
        zero_crossing_bool[:-1, :] |= zc_v
        abs_dog = np.abs(dog)
        zero_crossing = (zero_crossing_bool & (abs_dog > threshold)).astype(
            np.uint8
        ) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "dog_edge_opencv",
            exec_time,
            {"sigma1": sigma1, "sigma2": sigma2, "threshold": threshold, **kwargs},
        )

        return zero_crossing

    def _opencv_marr_hildreth_edge(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Обнаружение границ методом Марра-Хилдрета.

        Комбинация Гауссова размытия и Лапласиана с zero-crossing.
        Нулевые пересечения LoG с порогом.
        Классический метод для обнаружения границ.

        Args:
            img: Входное изображение (grayscale)
            sigma: Стандартное отклонение Гаусса (по умолчанию 1.0)
            threshold: Порог для отсечения слабого zero-crossing (по умолчанию 10)

        Returns:
            np.ndarray: Бинарная маска границ (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        sigma: float = self.params.get("sigma", 1.0)
        threshold: int = self.params.get("threshold", 10)

        # Лапласиан Гауссиана через OpenCV
        laplacian = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), sigma), cv2.CV_64F)

        # # Zero-crossing detection
        # zero_crossing = np.zeros_like(laplacian, dtype=np.uint8)
        # for i in range(1, laplacian.shape[0] - 1):
        #     for j in range(1, laplacian.shape[1] - 1):
        #         neighborhood = laplacian[i - 1 : i + 2, j - 1 : j + 2]
        #         if np.min(neighborhood) < 0 < np.max(neighborhood):
        #             if (
        #                 np.abs(np.min(neighborhood)) + np.abs(np.max(neighborhood))
        #                 > threshold
        #             ):
        #                 zero_crossing[i, j] = 255
        # Векторизованное zero-crossing
        sign = np.sign(laplacian)
        zc_h = sign[:, :-1] * sign[:, 1:] < 0
        zc_v = sign[:-1, :] * sign[1:, :] < 0
        zero_crossing_bool = np.zeros_like(laplacian, dtype=bool)
        zero_crossing_bool[:, :-1] |= zc_h
        zero_crossing_bool[:-1, :] |= zc_v
        abs_lap = np.abs(laplacian)
        zero_crossing = (zero_crossing_bool & (abs_lap > threshold)).astype(
            np.uint8
        ) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "marr_hildreth_edge_opencv",
            exec_time,
            {"sigma": sigma, "threshold": threshold, **kwargs},
        )

        return zero_crossing

    def _opencv_gradient_magnitude_direction(
        self, img: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Обнаружение границ через магнитуду и направление градиента.

        Вычисляет градиент с направлением и позволяет фильтрацию по углу.

        Args:
            img: Входное изображение (grayscale)
            threshold: Порог магнитуды (по умолчанию 50)
            angle_range: Диапазон углов для фильтрации в градусах (по умолчанию None - все углы)

        Returns:
            np.ndarray: Бинарная маска границ (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        threshold: int = self.params.get("threshold", 50)
        angle_range: Optional[Tuple[float, float]] = self.params.get(
            "angle_range", None
        )

        # Градиенты Собеля
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Магнитуда и направление
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x) * 180 / np.pi  # В градусах
        # Фильтрация по магнитуде
        mask = (magnitude > threshold).astype(np.uint8) * 255

        # Опциональная фильтрация по направлению
        if angle_range is not None:
            angle_mask = (
                (direction >= angle_range[0]) & (direction <= angle_range[1])
            ) | (
                (direction + 180 >= angle_range[0])
                & (direction + 180 <= angle_range[1])
            )
            mask = mask & (angle_mask.astype(np.uint8) * 255)

        exec_time: float = time.time() - start_time
        self._log_info(
            "gradient_magnitude_direction_opencv",
            exec_time,
            {"threshold": threshold, "angle_range": angle_range, **kwargs},
        )

        return mask

    def _opencv_phase_congruency_edge(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Обнаружение границ через фазовую конгруэнтность (упрощённая реализация).

        Метод, инвариантный к изменению контраста и яркости.
        Основан на согласованности фаз Фурье-компонент.

        Примечание: Полная реализация требует pyphase или аналогичной библиотеки.
        Здесь используется аппроксимация через много-масштабные градиенты.

        Args:
            img: Входное изображение (grayscale)
            nscales: Количество масштабов (по умолчанию 4)
            threshold: Порог для бинаризации (по умолчанию 0.2)

        Returns:
            np.ndarray: Бинарная маска границ (0/255)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        start_time: float = time.time()

        nscales: int = self.params.get("nscales", 4)
        threshold: float = self.params.get("threshold", 0.2)

        # Аппроксимация фазовой конгруэнтности через много-масштабные градиенты
        pc_map = np.zeros_like(gray, dtype=np.float32)

        for scale in range(nscales):
            sigma = 2**scale
            # Гауссово размытие на текущем масштабе
            blurred = gaussian_filter(gray.astype(np.float32), sigma=sigma)
            # Градиенты
            grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

            # Магнитуда
            mag = np.sqrt(grad_x**2 + grad_y**2)

            # Нормализация и добавление к карте
            mag_norm = mag / (np.max(mag) + 1e-8)
            pc_map += mag_norm

        # Усреднение по масштабам
        pc_map /= nscales

        # Нормализация к [0, 1]
        pc_map = (pc_map - np.min(pc_map)) / (np.max(pc_map) - np.min(pc_map) + 1e-8)

        # Бинаризация
        mask = (pc_map > threshold).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "phase_congruency_edge_opencv",
            exec_time,
            {"nscales": nscales, "threshold": threshold, **kwargs},
        )

        return mask

    # ============ РЕГИОНАЛЬНЫЕ МЕТОДЫ ============
    def _opencv_region_growing(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Сегментация методом Region Growing (роста регионов).

        Начинает с заданной точки (или центра) и рекурсивно добавляет соседние пиксели,
        интенсивность которых отличается от средней интенсивности региона не более чем на допуск.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска выращенного региона.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        start_time = time.time()

        h, w = gray.shape

        seed = self.params.get("seed", (w // 2, h // 2))
        tolerance = self.params.get("tolerance", 25)

        if seed is None or not (0 <= seed[0] < w and 0 <= seed[1] < h):
            seed = (w // 2, h // 2)  # (x, y)

        mask: np.ndarray = np.zeros((h, w), dtype=np.uint8)
        visited: np.ndarray = np.zeros((h, w), dtype=bool)

        queue = deque([seed])
        start_value = float(gray[seed[1], seed[0]])

        while queue:
            x, y = queue.popleft()

            if x < 0 or x >= w or y < 0 or y >= h or visited[y, x]:
                continue

            visited[y, x] = True

            if abs(int(gray[y, x]) - int(start_value)) <= tolerance:
                mask[y, x] = 255

                # 8-связность
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                            queue.append((nx, ny))

        exec_time = time.time() - start_time

        info = {
            "method": "region_growing_opencv",
            "parameters": {"seed": seed, "tolerance": tolerance, **kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_region_growing: {info}")

        return mask

    def _opencv_split_and_merge(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Рекурсивный алгоритм разделения и слияния регионов.

        Рекурсивно делит изображение на квадранты до тех пор, пока дисперсия внутри региона
        не станет меньше заданного порога. Затем объединяет похожие соседние регионы.
        Возвращает маску второго по величине региона (предполагаемый объект).

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        start_time = time.time()

        h, w = gray.shape
        threshold = self.params.get("threshold", 20)
        min_size = self.params.get("min_size", 50)

        def region_stats(region):
            pixels = [gray[y, x] for x, y in region]
            if not pixels:
                return 0, 0
            mean = np.mean(pixels)
            std = np.std(pixels)
            return mean, std

        def split(region, min_sz, thresh):
            if len(region) <= min_sz:
                return [region]

            mean, std = region_stats(region)
            if std < thresh:
                return [region]

            x_coords = [p[0] for p in region]
            y_coords = [p[1] for p in region]

            x_mid = (min(x_coords) + max(x_coords)) // 2
            y_mid = (min(y_coords) + max(y_coords)) // 2

            quadrants = [
                [(x, y) for x, y in region if x <= x_mid and y <= y_mid],
                [(x, y) for x, y in region if x > x_mid and y <= y_mid],
                [(x, y) for x, y in region if x <= x_mid and y > y_mid],
                [(x, y) for x, y in region if x > x_mid and y > y_mid],
            ]

            result = []
            for quad in quadrants:
                result.extend(split(quad, min_sz, thresh))
            return result

        def merge(regions, thresh):
            merged = []
            used = [False] * len(regions)

            for i, reg1 in enumerate(regions):
                if used[i]:
                    continue

                current = reg1.copy()
                mean1, _ = region_stats(reg1)

                for j, reg2 in enumerate(regions[i + 1 :], i + 1):
                    if used[j]:
                        continue

                    mean2, _ = region_stats(reg2)
                    if abs(mean1 - mean2) < thresh:
                        current.extend(reg2)
                        used[j] = True

                merged.append(current)
                used[i] = True

            return merged

        # Начальный регион - все изображение
        initial = [(x, y) for y in range(h) for x in range(w)]

        # Split фаза
        regions = split(initial, min_size, threshold)

        # Merge фаза
        regions = merge(regions, threshold)

        # Создаем маску (второй по величине регион)
        if len(regions) > 1:
            sizes = [len(r) for r in regions]
            idx = np.argsort(sizes)[-2]  # Второй по величине
            mask = np.zeros((h, w), dtype=np.uint8)
            for x, y in regions[idx]:
                mask[y, x] = True

            exec_time = time.time() - start_time

            info = {
                "method": "split_and_merge_opencv",
                "parameters": {"threshold": threshold, "min_size": min_size, **kwargs},
                "execution_time": exec_time,
            }
            print(f"Info after OpenCV_split_and_merge: {info}")

            return mask.astype(np.uint8) * 255

        return np.zeros((h, w), dtype=np.uint8)

    def _opencv_floodfill(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Сегментация методом заливки (Flood Fill).

        Начиная с заданной точки, рекурсивно заполняет все связанные пиксели,
        интенсивность которых отличается от исходной не более чем на допуск.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8) залитой области.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        start_time = time.time()

        h, w = gray.shape

        # Начальная точка
        seed = self.params.get("seed", None)
        if seed is None:
            seed = (w // 2, h // 2)

        # Создаем маску
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Параметры заливки
        tolerance = self.params.get("tolerance", 20)
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE

        # Применяем floodfill
        cv2.floodFill(
            gray.copy(), mask, seed, 255, (tolerance,) * 3, (tolerance,) * 3, flags
        )

        # Извлекаем маску
        mask_final = mask[1:-1, 1:-1] * 255

        # Опционально: заполняем дыры
        mask_final = ndimage.binary_fill_holes(mask_final > 0).astype(np.uint8) * 255

        exec_time = time.time() - start_time

        info = {
            "method": "floodfill_opencv",
            "parameters": {"seed": seed, "tolerance": tolerance, **kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_floodfill: {info}")

        return mask_final

    # ============ КЛАСТЕРИЗАЦИЯ ============
    def _opencv_kmeans_segmentation(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Сегментация методом K-Means кластеризации.

        Группирует пиксели по цветовому признаку в K кластеров.
        Самый крупный кластер считается фоном; остальные — объектами.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        start_time = time.time()

        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3).astype(np.float32)

        k = self.params.get("k", 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        best_labels = np.zeros((pixels.shape[0],), dtype=np.int32)
        compactness, labels, centers = cv2.kmeans(
            pixels, k, best_labels, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        labels = labels.reshape(h, w)

        # Находим самый большой кластер
        unique, counts = np.unique(labels, return_counts=True)
        bg_label = unique[np.argmax(counts)]

        mask = (labels != bg_label).astype(np.uint8) * 255

        exec_time = time.time() - start_time

        info = {
            "method": "kmeans_opencv",
            "parameters": {"k": k, **kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_kmeans: {info}")

        return mask

    def _opencv_dbscan_segmentation(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Сегментация методом DBSCAN кластеризации.

        Группирует пиксели на основе плотности в пространстве цветовых признаков.
        Шумовые пиксели (метка -1) исключаются. Самый крупный кластер считается фоном.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        start_time = time.time()

        h, w = img.shape[:2]

        # Для больших изображений уменьшаем разрешение чтобы DBSCAN не завис
        scale = 1.0
        if h * w > 80000:
            scale = np.sqrt(80000.0 / (h * w))
            small = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            small = img

        sh, sw = small.shape[:2]

        if len(small.shape) == 3:
            pixels = small.reshape(-1, 3).astype(np.float32) / 255.0
        else:
            pixels = small.reshape(-1, 1).astype(np.float32) / 255.0

        eps = self.params.get("eps", 0.05)
        min_samples = self.params.get("min_samples", 5)

        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(pixels)

        # Маска: всё кроме шума (-1) и самого большого кластера (фон)
        labels_2d = labels.reshape(sh, sw)
        valid = labels[labels != -1]
        if len(valid) > 0:
            unique, counts = np.unique(valid, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            mask_small = ((labels_2d != bg_label) & (labels_2d != -1)).astype(
                np.uint8
            ) * 255
        else:
            mask_small = np.zeros((sh, sw), dtype=np.uint8)

        # Восстанавливаем исходный размер
        if scale < 1.0:
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = mask_small

        exec_time = time.time() - start_time

        info = {
            "method": "dbscan_opencv",
            "parameters": {"min_samples": min_samples, "eps": eps, **kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_dbscan: {info}")

        return mask

    def _opencv_meanshift(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Сегментация методом MeanShift.

        Итеративно сдвигает каждый пиксель к локальному центру масс в пространстве признаков
        (цвет + координаты). Результатом является кластеризация пикселей по плотности.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8). Самый крупный кластер — фон.
        """
        start_time = time.time()

        spatial_radius = self.params.get("spatial_radius", 60)
        color_radius = self.params.get("color_radius", 60)
        max_level = self.params.get("max_level", 1)

        # Применяем MeanShift
        shifted = cv2.pyrMeanShiftFiltering(
            img, spatial_radius, color_radius, max_level
        )

        # Конвертируем в grayscale и пороговую обработку
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        exec_time = time.time() - start_time

        info = {
            "method": "meanshift_opencv",
            "parameters": {
                "spatial_radius": spatial_radius,
                "color_radius": color_radius,
                "max_level": max_level,
                **kwargs,
            },
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_meanshift: {info}")

        return mask

    # ============ АКТИВНЫЕ КОНТУРЫ ============

    def _opencv_active_contour(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Сегментация активными контурами (Snakes).

        Инициализирует замкнутый контур (обычно окружность) и деформирует его под действием
        внутренних (упругость, жесткость) и внешних (притяжение к границам) сил до равновесия.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8) внутри замкнутого контура.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        start_time = time.time()

        h, w = gray.shape

        # Начальный контур (окружность)
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y) // 2

        # Создаем маску контура
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        # Применяем морфологические операции для имитации активного контура
        kernel = np.ones((5, 5), np.uint8)
        iterations = self.params.get("iterations", 10)
        for _ in range(iterations):
            edges = cv2.Canny(gray, 100, 200)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        exec_time = time.time() - start_time

        info = {
            "method": "active_contour_opencv",
            "parameters": {"iterations": iterations, **kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_active_contour: {info}")

        return mask

    def _opencv_gvf_contour(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Сегментация на основе Gradient Vector Flow (GVF).

        Вычисляет векторное поле, распространяющее информацию о градиентах по всему изображению.
        Это позволяет контуру "чувствовать" границы даже на расстоянии. Маска строится по величине GVF.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        start_time = time.time()

        # Вычисляем градиенты
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Матрица границ
        edges = cv2.Canny(gray, 100, 200)

        # Распространение градиентов
        u = grad_x.copy()
        v = grad_y.copy()

        mu = self.params.get("mu", 0.1)
        iterations = self.params.get("iterations", 50)

        for _ in range(iterations):
            laplacian_u = cv2.Laplacian(u, cv2.CV_64F)
            laplacian_v = cv2.Laplacian(v, cv2.CV_64F)

            edge_weight = edges.astype(np.float64) / 255.0

            u = u + mu * laplacian_u - edge_weight * (u - grad_x)
            v = v + mu * laplacian_v - edge_weight * (v - grad_y)

        # Величина GVF
        gvf_mag = np.sqrt(u**2 + v**2)
        gvf_mag = np.uint8(255 * gvf_mag / np.max(gvf_mag))

        _, mask = cv2.threshold(gvf_mag, 50.0, 255.0, cv2.THRESH_BINARY)  # type: ignore[call-overload]
        mask = ndimage.binary_fill_holes(mask > 0).astype(np.uint8) * 255

        exec_time = time.time() - start_time

        info = {
            "method": "gvf_opencv",
            "parameters": {"iterations": iterations, "mu": mu, **kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_gvf_contour: {info}")

        return mask

    def _opencv_morphological_snakes(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Сегментация морфологическими змеями.

        Итеративно расширяет или сужает бинарную маску на основе величины градиента.
        Области с низким градиентом "поглощаются", с высоким — отбрасываются.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        start_time = time.time()

        h, w = gray.shape

        # Начальная маска (окружность в центре)
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 4, 255, -1)

        iterations = self.params.get("iterations", 50)
        kernel = np.ones((3, 3), np.uint8)

        for _ in range(iterations):
            # Градиент изображения
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_mag = np.uint8(255 * grad_mag / np.max(grad_mag))

            _, grad_binary = cv2.threshold(grad_mag, 50.0, 255.0, cv2.THRESH_BINARY)  # type: ignore[call-overload]

            # Расширение/сужение на основе градиента
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(grad_binary))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        exec_time = time.time() - start_time

        info = {
            "method": "morphological_snakes_opencv",
            "parameters": {"iterations": iterations, **kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_morphological_snakes: {info}")

        return mask

    def _opencv_chan_vese(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Модель Chan-Vese — активные контуры без градиентов.

        Энергетическая модель, которая разделяет изображение на две области с минимальной
        внутрирегиональной дисперсией. Подходит для объектов без четких границ.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска: 255 — внутренняя область контура.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        start_time = time.time()

        h, w = gray.shape

        # Начальная маска (центральная область)
        mask = np.zeros((h, w), np.uint8)
        cv2.rectangle(mask, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), 255, -1)

        iterations = self.params.get("max_iter", 100)
        mu = self.params.get("mu", 0.25)

        for _ in range(iterations):
            # Вычисляем средние значения внутри и снаружи маски
            inside_mean = (
                float(np.mean(gray[mask > 0].astype(np.float32)))
                if np.any(mask > 0)
                else 0.0
            )
            outside_mean = (
                float(np.mean(gray[mask == 0].astype(np.float32)))
                if np.any(mask == 0)
                else 0.0
            )

            # Обновляем маску на основе разности с средними
            diff_inside = np.abs(gray.astype(float) - inside_mean)
            diff_outside = np.abs(gray.astype(float) - outside_mean)

            new_mask = np.zeros_like(mask)
            new_mask[diff_inside < diff_outside] = 255

            # Сглаживание
            kernel = np.ones((3, 3), np.uint8)
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel)

            mask = new_mask

        exec_time = time.time() - start_time

        info = {
            "method": "chan_vese_opencv",
            "parameters": {"mu": mu, "iterations": iterations, **kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_chan_vese: {info}")

        return mask

    # ============ WATERSHED И ГРАФОВЫЕ ============

    def _opencv_watershed(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Сегментация методом водораздела (Watershed).

        Использует морфологические операции и преобразование расстояния для выделения
        надежных маркеров переднего плана и фона. Алгоритм "затопляет" изображение от маркеров,
        формируя границы между объектами.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8) всех сегментированных объектов.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        start_time = time.time()

        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Бинаризация
        _, binary = cv2.threshold(
            gray, 0.0, 255.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Морфологические операции
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Фон
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Преобразование расстояния
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg_raw = cv2.threshold(
            dist_transform, float(0.7 * dist_transform.max()), 255.0, 0
        )
        sure_fg: np.ndarray = sure_fg_raw.astype(np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Маркеры
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Watershed
        if len(img.shape) == 3:
            markers = cv2.watershed(img, markers)
        else:
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(color_img, markers)

        mask = (markers > 1).astype(np.uint8) * 255

        exec_time = time.time() - start_time

        info = {
            "method": "watershed_opencv",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_watershed: {info}")

        return mask

    def _opencv_random_walker(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Сегментация методом Random Walker.

        На основе маркеров решается задача на графе: каждый пиксель принадлежит тому
        маркеру, до которого случайное блуждание доходит быстрее.
        Использует skimage.segmentation.random_walker (корректная реализация).

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска переднего плана.
        """
        from skimage.segmentation import random_walker as sk_random_walker

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        start_time = time.time()

        h, w = gray.shape
        gray_norm = gray.astype(np.float32) / 255.0

        # Создаём маркеры: 1=фон (углы), 2=объект (центр)
        markers = np.zeros((h, w), dtype=np.int32)
        markers[(h // 4) : (3 * h // 4), (w // 4) : (3 * w // 4)] = 2
        corner_size = min(h, w) // 8
        markers[:corner_size, :corner_size] = 1
        markers[:corner_size, -corner_size:] = 1
        markers[-corner_size:, :corner_size] = 1
        markers[-corner_size:, -corner_size:] = 1

        beta = self.params.get("beta", 130)
        mode = self.params.get("mode", "cg_j")

        try:
            labels = sk_random_walker(gray_norm, markers, beta=beta, mode=mode)
            mask = (labels == 2).astype(np.uint8) * 255
        except Exception as e:
            warnings.warn(f"Random Walker failed: {e}. Falling back to Watershed.")
            # Fallback: используем Watershed с теми же маркерами
            color_img = (
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
            )
            ws_markers = markers.copy().astype(np.int32)
            cv2.watershed(color_img, ws_markers)
            mask = (ws_markers == 2).astype(np.uint8) * 255

        exec_time = time.time() - start_time
        info = {
            "method": "random_walker_opencv",
            "parameters": {"beta": beta, "mode": mode, **kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_random_walker: {info}")

        return mask

    # ============ SUPER-PIXEL МЕТОДЫ ============
    def _opencv_quickshift(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Quickshift сегментация — mode-seeking алгоритм в пространстве (цвет + координаты).

        Использует skimage.segmentation.quickshift.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8). Самый крупный сегмент — фон.
        """
        from skimage.segmentation import quickshift as sk_quickshift

        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.time()

        kernel_size = self.params.get("kernel_size", 3)
        max_dist = self.params.get("max_dist", 6)
        ratio = self.params.get("ratio", 0.5)
        sigma = self.params.get("sigma", 0.0)

        img_float = img_rgb.astype(np.float32) / 255.0

        segments = sk_quickshift(
            img_float,
            kernel_size=kernel_size,
            max_dist=max_dist,
            ratio=ratio,
            sigma=sigma,
        )

        unique, counts = np.unique(segments, return_counts=True)
        bg_label = unique[np.argmax(counts)]
        mask = (segments != bg_label).astype(np.uint8) * 255

        exec_time = time.time() - start_time
        info = {
            "method": "quickshift_opencv",
            "parameters": {
                "kernel_size": kernel_size,
                "max_dist": max_dist,
                "ratio": ratio,
                "sigma": sigma,
                **kwargs,
            },
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_quickshift: {info}")

        return mask

    def _opencv_slic(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        SLIC (Simple Linear Iterative Clustering) — суперпиксельная сегментация.

        Группирует пиксели в компактные, однородные регионы (суперпиксели) на основе пространственной
        и цветовой близости. Самый крупный суперпиксель считается фоном.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8): 255 — все суперпиксели, кроме фона.
        """
        # Приводим к BGR для ximgproc
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = (
                img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            )

        start_time = time.time()

        region_size = self.params.get("region_size", 20)
        ruler = self.params.get("ruler", 10.0)
        num_iterations = self.params.get("num_iterations", 10)

        if not hasattr(cv2, "ximgproc"):
            warnings.warn(
                "cv2.ximgproc не доступен. Установите opencv-contrib-python "
                "или используйте альтернативный метод.",
                RuntimeWarning,
            )
            # Fallback: простая сетка или возвращаем исходное
            return (
                np.zeros_like(img)
                if len(img.shape) == 2
                else np.zeros(img.shape[:2], dtype=np.uint8)
            )

        try:
            # cv2.ximgproc доступен в opencv-contrib-python
            slic = cv2.ximgproc.createSuperpixelSLIC(
                img_bgr,
                algorithm=cv2.ximgproc.SLIC,
                region_size=region_size,
                ruler=ruler,
            )
            slic.iterate(num_iterations)
            labels = slic.getLabels()  # (H, W) int32
        except AttributeError:
            # Fallback: ximgproc не установлен — используем KMeans как аппроксимацию
            warnings.warn(
                "cv2.ximgproc не доступен. Используем KMeans как аппроксимацию SLIC."
            )
            return self._opencv_kmeans_segmentation(img, **kwargs)

        # Находим самый большой суперпиксель (фон)
        unique, counts = np.unique(labels, return_counts=True)
        bg_label = unique[np.argmax(counts)]
        mask = (labels != bg_label).astype(np.uint8) * 255

        exec_time = time.time() - start_time
        info = {
            "method": "slic_opencv",
            "parameters": {
                "region_size": region_size,
                "ruler": ruler,
                "num_iterations": num_iterations,
                **kwargs,
            },
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_slic: {info}")

        return mask

    def _opencv_felzenszwalb(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Алгоритм Felzenszwalb — иерархическая сегментация на основе графов.

        Реализован через skimage.segmentation.felzenszwalb (оригинальный алгоритм).
        Строит сегментацию на основе минимального остовного дерева, начиная с мелких регионов и объединяя их, если внутреннее различие
        меньше межрегионального. Очень эффективен для выделения объектов разного масштаба.

        Args:
            img: Входное изображение (RGB).

        Returns:
            Бинарная маска: 255 — все регионы, кроме самого крупного (фона).
        """
        from skimage.segmentation import felzenszwalb as sk_felzenszwalb

        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.time()

        scale = self.params.get("scale", 100)
        sigma = self.params.get("sigma", 0.8)
        min_size = self.params.get("min_size", 50)

        # Нормализуем к float [0,1] для skimage
        img_float = img_rgb.astype(np.float32) / 255.0

        segments = sk_felzenszwalb(
            img_float, scale=scale, sigma=sigma, min_size=min_size
        )

        # Находим самый большой сегмент (фон) и создаём маску
        unique, counts = np.unique(segments, return_counts=True)
        bg_label = unique[np.argmax(counts)]
        mask = (segments != bg_label).astype(np.uint8) * 255

        exec_time = time.time() - start_time
        info = {
            "method": "felzenszwalb_opencv",
            "parameters": {
                "scale": scale,
                "sigma": sigma,
                "min_size": min_size,
                **kwargs,
            },
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_felzenszwalb: {info}")

        return mask

    # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============
    # def _opencv_grabcut(
    #     self,
    #     img: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Интерактивная сегментация GrabCut.

    #     Использует прямоугольник для инициализации фона и переднего плана.
    #     Строит модели цветового распределения (GMM) и уточняет границы итеративно.

    #     Args:
    #         img: Входное изображение (RGB).

    #     Returns:
    #         np.ndarray: Бинарная маска (0/255, dtype=np.uint8) переднего плана.
    #     """
    #     h, w = img.shape[:2]

    #     # Создаем маску
    #     mask = np.zeros((h, w), np.uint8)

    #     # Прямоугольник для инициализации (центр изображения)
    #     rect = self.params.get('rect', (w//4, h//4, w//2, h//2))

    #     # Временные массивы
    #     bgd_model = np.zeros((1, 65), np.float64)
    #     fgd_model = np.zeros((1, 65), np.float64)

    #     # Применяем GrabCut
    #     cv2.grabCut(img, mask, rect, bgd_model, fgd_model,
    #                self.params.get('iterations', 5), cv2.GC_INIT_WITH_RECT)

    #     # Создаем финальную маску
    #     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    #     result_mask = mask2 * 255

    #     return result_mask

    def _opencv_grabcut(self, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        Интерактивная сегментация GrabCut.

        Использует прямоугольник для инициализации фона и переднего плана.
        Строит модели цветового распределения (GMM) и уточняет границы итеративно.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8) переднего плана.
        """
        start_time = time.time()
        # Параметры
        rect = self.params.get("rect", None)
        iter_count = self.params.get("num_iterations", 10)

        # Создаем маску и модель
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)  # type: ignore[var-annotated]
        fgd_model = np.zeros((1, 65), dtype=np.float64)  # type: ignore[var-annotated]

        # Если прямоугольник не задан, используем центральную часть
        if rect is None:
            h, w = img.shape[:2]
            rect = (int(w * 0.25), int(h * 0.25), int(w * 0.5), int(h * 0.5))

        # Применяем GrabCut
        mask, bgd_model, fgd_model = cv2.grabCut(  # type: ignore[assignment]
            img, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT
        )

        # Создаем финальную маску (0-255)
        mask_final: np.ndarray = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Опционально: применение морфологических операций для улучшения результата
        kernel = np.ones((3, 3), np.uint8)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)  # type: ignore[assignment]
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=2)  # type: ignore[assignment]

        exec_time = time.time() - start_time

        info = {
            "method": "grabcut_opencv",
            "parameters": {"iterations": iter_count, "rect": rect, **kwargs},
            "execution_time": exec_time,
        }
        print(f"Info after OpenCV_grabcut: {info}")

        return mask_final.astype(np.uint8)
