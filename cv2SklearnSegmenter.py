# cv2SklearnSegmenter.py

# Импорт основных библиотек

from BaseSegmenter import BaseSegmenter

from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload
)
import numpy as np
from PIL import Image
from collections import deque
from scipy import ndimage
import warnings

import torch
import cv2

from skimage import segmentation, feature, measure, morphology
from skimage import segmentation as skseg
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from skimage.draw import polygon
from skimage.feature import canny
from skimage.segmentation import chan_vese, random_walker, slic as sk_slic

class CV2SklearnSegmenter(BaseSegmenter):
    """
    Класс для реализации методов сегментации изображений на основе библиотек OpenCV и scikit-learn.
    Поддерживает как классические методы (пороговые, граничные), так и методы на основе кластеризации,
    активных контуров и графов.
    """
    def __init__(
        self, 
        method: str = "global_thresholding", 
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.method: str = method
        self.params: Dict[str, Any] = kwargs
        self._setup_method()

        # Определяем, нужны ли изображения в градациях серого для данного метода
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
            "random_walker"
        ]
    
    def _setup_method(self) -> None:
        """Регистрация всех доступных методов сегментации."""
        self.method_map: Dict[str, Any] = {
            # ============ ПОРОГОВЫЕ МЕТОДЫ СЕГМЕНТАЦИИ ============
            "global_thresholding": self._global_thresholding,
            "adaptive_thresholding": self._adaptive_thresholding,
            "otsu_thresholding": self._otsu_thresholding,
            "threshold_niblack": self._threshold_niblack,
            "threshold_sauvola": self._threshold_sauvola,

            # ============ КРАЕВЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "sobel_edge": self._sobel_edge,
            "canny_edge": self._canny_edge,

            # ============ РЕГИОНАЛЬНЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "region_growing": self._region_growing,
            "split_and_merge": self._split_and_merge,
            "floodfill": self._floodfill,

            # ============ КЛАСТЕРИЗАЦИЯ ============
            "kmeans_segmentation": self._kmeans_segmentation,
            "dbscan_segmentation": self._dbscan_segmentation,
            "meanshift": self._meanshift,

            # ============ АКТИВНЫЕ КОНТУРЫ ============
            "active_contour": self._active_contour,
            "gvf_contour": self._gvf_contour,
            "morphological_snakes": self._morphological_snakes,
            "chan_vese": self._chan_vese,

            # ============ WATERSHED И ГРАФОВЫЕ ============
            "watershed": self._watershed,
            "random_walker": self._random_walker,

            # ============ SUPER-PIXEL МЕТОДЫ ===========
            "quickshift": self._quickshift,
            "slic": self._slic,
            "felzenszwalb": self._felzenszwalb,

            # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============
            "grabcut": self._grabcut
        }
        
        if self.method not in self.method_map:
            raise ValueError(f"Неизвестный метод: {self.method}. "
                           f"Доступные методы: {list(self.method_map.keys())}")
        
        self._segment_func = self.method_map[self.method]
    
    def segment(
        self, 
        image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """
        Выполняет сегментацию изображения и возвращает бинарную маску.

        Args:
            image: Входное изображение (путь, массив, PIL или тензор).

        Returns:
            np.ndarray: Бинарная маска (0–255, dtype=np.uint8), где 255 — объект.
        """
        img_array: np.ndarray = self.preprocess_image(
            image, 
            as_gray=self._needs_gray
        )
        img_array: np.ndarray = self.preprocess_image(
            image, 
            as_gray=self._needs_gray
        )
        return self._segment_func(img_array)
    
    def segment_with_mask(
        self, 
        image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполняет сегментацию и возвращает визуализацию + маску.

        Args:
            image: Входное изображение.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Визуализация: исходное изображение с наложенной красной маской (0–255, RGB).
                - Маска: бинарная маска (0–255, grayscale).
        """
        img_array: np.ndarray = self.preprocess_image(image)
        mask = self._segment_func(img_array)

        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        
        # Создаем визуализацию
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            img_array = img_array.copy()
        
        overlay = img_array
        overlay[mask > 0] = [255, 0, 0] # Красный цвет для маски
        result = cv2.addWeighted(img_array, 0.1, overlay, 0.9, 0)
        return result, mask
    
    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    # ============ ПОРОГОВЫЕ МЕТОДЫ ============
    
    def _global_thresholding(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        threshold = self.params.get('threshold', 127)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return mask
    
    def _adaptive_thresholding(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        block_size = self.params.get('block_size', 11)
        c = self.params.get('C', 2)

        if block_size % 2 == 0:
            block_size += 1
        
        mask = cv2.adaptiveThreshold(
            gray, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 
            block_size, 
            c
        )
        return mask
    
    def _otsu_thresholding(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Автоматическая бинаризация по методу Оцу.

        Находит оптимальный порог, максимизирующий межклассовую дисперсию между фоном и объектом.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    
    def _threshold_niblack(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        window_size = self.params.get('window_size', 15)
        k = self.params.get('k', 0.2)
        
        # Вычисляем среднее и стандартное отклонение в окне
        mean = cv2.blur(gray, (window_size, window_size))
        std = np.sqrt(cv2.boxFilter(gray.astype(float)**2, -1, (window_size, window_size)) - mean**2)
        
        # Вычисляем порог
        threshold = mean + k * std
        
        # Бинаризация
        mask = (gray > threshold).astype(np.uint8) * 255
        
        return mask

    def _threshold_sauvola(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        window_size = self.params.get('window_size', 15)
        k = self.params.get('k', 0.2)
        r = self.params.get('r', 128)
        
        # Вычисляем среднее и стандартное отклонение в окне
        mean = cv2.blur(gray, (window_size, window_size))
        std = np.sqrt(cv2.boxFilter(gray.astype(float)**2, -1, (window_size, window_size)) - mean**2)
        
        # Вычисляем порог
        threshold = mean * (1 + k * (std / r - 1))
        
        # Бинаризация
        mask = (gray > threshold).astype(np.uint8) * 255
        
        return mask
    
    # ============ МЕТОДЫ НА ОСНОВЕ КРАЕВ ============
    def _sobel_edge(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        threshold = self.params.get('threshold', 50)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel_norm = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
        # # Оптимальный способ: cv2.magnitude (быстрый и точный)
        # sobel_mag = cv2.magnitude(sobelx, sobely)
    
        # # Нормализация
        # sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
        _, mask = cv2.threshold(sobel_norm.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _canny_edge(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        low = self.params.get('low', 50)
        high = self.params.get('high', 150)
        edges = cv2.Canny(gray, low, high)
        return edges
    
    # ============ РЕГИОНАЛЬНЫЕ МЕТОДЫ ============
    
    def _region_growing(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        seed = self.params.get('seed', None)
        tolerance = self.params.get('tolerance', 15)
        
        h, w = gray.shape
        
        # Если seed не указан, используем центр изображения
        if seed is None or not (0 <= seed[0] < w and 0 <= seed[1] < h):
            seed = (w // 2, h // 2) # (x, y)
        
        region_mask: np.ndarray = np.zeros_like(gray, dtype=bool)
        visited: np.ndarray = np.zeros_like(gray, dtype=bool)
        queue = deque([seed])
        
        region_mean = float(gray[seed[1], seed[0]])
        
        while queue:
            x, y = queue.popleft()
            if not (0 <= x < w and 0 <= y < h) or visited[y, x]:
                continue
            visited[y, x] = True
            pixel_value = gray[y, x]
            
            # Проверяем сходство со средним значением региона
            if abs(pixel_value - region_mean) <= tolerance:
                region_mask[y, x] = 255
                
                # Добавляем соседей 8-связности
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                            queue.append((nx, ny))
        
        return region_mask
    
    def _split_and_merge(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        min_region_size = self.params.get('min_region_size', 50)
        threshold = self.params.get('threshold', 20)
        
        h, w = gray.shape
        
        def region_properties(region):
            if len(region) == 0:
                return 0, 0
            intensities = [gray[y, x] for x, y in region]
            return np.mean(intensities), np.std(intensities)
        
        def split(region, min_size, thresh):
            if len(region) <= min_size:
                return [region]
            
            mean_intensity, std_intensity = region_properties(region)
            if std_intensity < thresh:
                return [region]
            
            x_coords = [p[0] for p in region]
            y_coords = [p[1] for p in region]
            
            x_mid = (min(x_coords) + max(x_coords)) // 2
            y_mid = (min(y_coords) + max(y_coords)) // 2
            
            subregions = [
                [(x, y) for x, y in region if x <= x_mid and y <= y_mid],
                [(x, y) for x, y in region if x > x_mid and y <= y_mid],
                [(x, y) for x, y in region if x <= x_mid and y > y_mid],
                [(x, y) for x, y in region if x > x_mid and y > y_mid]
            ]
            
            result = []
            for subregion in subregions:
                result.extend(split(subregion, min_size, thresh))
            return result
        
        def merge(regions, thresh):
            merged = []
            used = [False] * len(regions)
            
            for i, region1 in enumerate(regions):
                if used[i]:
                    continue
                current_merge = region1.copy()
                mean1, std1 = region_properties(region1)
                
                for j, region2 in enumerate(regions[i+1:], i+1):
                    if used[j]:
                        continue
                    mean2, std2 = region_properties(region2)
                    if abs(mean1 - mean2) < thresh:
                        current_merge.extend(region2)
                        used[j] = True
                
                merged.append(current_merge)
                used[i] = True
            
            return merged
        
        # Начинаем с целого изображения как одного региона
        initial_region = [(x, y) for x in range(w) for y in range(h)]
        
        # Фаза разделения
        regions = split(initial_region, min_region_size, threshold)
        
        # Фаза слияния
        regions = merge(regions, threshold)
        
        # Создаем маску (берем самый большой регион после фона)
        region_sizes = [len(region) for region in regions]
        if len(region_sizes) > 1:
            foreground_idx = np.argsort(region_sizes)[-2]
            mask = np.zeros_like(gray, dtype=bool)
            for x, y in regions[foreground_idx]:
                mask[y, x] = True
            return mask.astype(np.uint8) * 255
        else:
            return np.zeros_like(gray, dtype=np.uint8)
        
    def _floodfill(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Сегментация методом заливки (Flood Fill).

        Начиная с заданной точки, рекурсивно заполняет все связанные пиксели,
        интенсивность которых отличается от исходной не более чем на допуск.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8) залитой области.
        """
        # Параметры
        seed = self.params.get('seed', None)
        tolerance = self.params.get('tolerance', 20)
        
        h, w = img.shape[:2]
        
        # Если seed не указан, используем центр изображения
        if seed is None:
            seed = (w // 2, h // 2)  # (x, y)
        
        # Создаем маску для floodfill
        mask = np.zeros((h+2, w+2), dtype=np.uint8)
        
        # Определяем цвет заполнения
        new_val = (255, 255, 255)
        
        # Параметры floodfill
        lo_diff = (tolerance, tolerance, tolerance)
        up_diff = (tolerance, tolerance, tolerance)
        
        # Применяем floodfill
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
        
        try:
            # Создаем копию изображения
            img_copy = img.copy()
            
            # Запускаем floodfill
            cv2.floodFill(img_copy, mask, seed, new_val, lo_diff, up_diff, flags)
            
            # Извлекаем маску
            mask_final = mask[1:-1, 1:-1] * 255
            
            # Опционально: заполняем дыры
            mask_final = ndimage.binary_fill_holes(mask_final > 0).astype(np.uint8) * 255
            
            return mask_final
            
        except Exception as e:
            warnings.warn(f"FloodFill failed: {e}. Using fallback.")
            # Резервный вариант
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return mask
        
    # ============ КЛАСТЕРИЗАЦИЯ ============
    
    def _kmeans_segmentation(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Сегментация методом K-Means кластеризации.

        Группирует пиксели по цветовому признаку в K кластеров.
        Самый крупный кластер считается фоном; остальные — объектами.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        k = self.params.get('k', 3)
        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
        # kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pixels)
        labels = kmeans.labels_.reshape(h, w)
         # Выбираем самый крупный кластер как фон
        unique, counts = np.unique(labels, return_counts=True)
        bg_label = unique[np.argmax(counts)]
        mask = (labels != bg_label).astype(np.uint8) * 255
        return mask
    
    def _dbscan_segmentation(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Сегментация методом DBSCAN кластеризации.

        Группирует пиксели на основе плотности. Пиксели, не принадлежащие ни одному кластеру (шум),
        исключаются. Самый крупный кластер считается фоном.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        eps = self.params.get('eps', 10)
        min_samples = self.params.get('min_samples', 100)
        
        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3)
        
        # db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
        # labels = db.labels_.reshape(h, w)
        
        # mask = (labels != -1) & (labels != 0)
        # return mask.astype(np.uint8) * 255
        scale = 0.5
        if h * w > 100000:
            small_h, small_w = int(h * scale), int(w * scale)
            img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            pixels = img_small.reshape(-1, 3)
        else:
            pixels = img.reshape(-1, 3)
        
        try:
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(pixels)
            labels = db.labels_
            
            if h * w > 100000:
                # Интерполируем обратно
                labels_2d = labels.reshape(small_h, small_w)
                labels_2d = cv2.resize(labels_2d.astype(np.float32), (w, h), 
                                      interpolation=cv2.INTER_NEAREST).astype(int)
            else:
                labels_2d = labels.reshape(h, w)
            
            # Создаем маску (все кроме шума)
            mask = (labels_2d != -1).astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"DBSCAN failed: {e}. Using fallback.")
            # Резервный вариант
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _meanshift(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Сегментация методом MeanShift.

        Итеративно сдвигает каждый пиксель к локальному центру масс в пространстве признаков
        (цвет + координаты). Результатом является кластеризация пикселей по плотности.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8). Самый крупный кластер — фон.
        """
        
        # Параметры
        bandwidth = self.params.get('bandwidth', 0.5)
        
        h, w = img.shape[:2]
        
        # Уменьшаем разрешение для скорости
        scale = 0.5
        if h * w > 100000:
            small_h, small_w = int(h * scale), int(w * scale)
            img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            pixels = img_small.reshape(-1, 3)
        else:
            pixels = img.reshape(-1, 3)
        
        try:
            # Применяем MeanShift
            meanshift = MeanShift(bandwidth=bandwidth, n_jobs=-1)
            labels = meanshift.fit_predict(pixels)
            
            if h * w > 100000:
                # Интерполируем обратно
                labels_2d = labels.reshape(small_h, small_w)
                labels_2d = cv2.resize(labels_2d.astype(np.float32), (w, h), 
                                      interpolation=cv2.INTER_NEAREST)
            else:
                labels_2d = labels.reshape(h, w)
            
            # Находим самый большой кластер (предположительно фон)
            unique, counts = np.unique(labels, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            
            # Создаем маску
            mask = (labels_2d != bg_label).astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"MeanShift failed: {e}. Using fallback.")
            # Резервный вариант
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    

    # ============ АКТИВНЫЕ КОНТУРЫ ============
    def _active_contour(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        gray_norm = gray.astype(float) / 255.0
        
        h, w = gray_norm.shape
        center_y, center_x = h // 2, w // 2
        # radius = min(center_x, center_y) * 0.7 # 70% от центра до края
        
        # s = np.linspace(0, 2 * np.pi, 400)
        radius = min(center_x, center_y) * 0.5  # 50% от центра до края
            
        s = np.linspace(0, 2 * np.pi, 100)
        r = center_y + radius * np.sin(s)
        c = center_x + radius * np.cos(s)
        init = np.array([r, c]).T

        alpha = self.params.get('alpha', 0.01)    # elasticity
        beta = self.params.get('beta', 0.1)       # rigidity
        gamma = self.params.get('gamma', 0.001)   # time step
        max_iterations = self.params.get('max_iterations', 2000)
        
        try:
            snake = segmentation.active_contour(
                gray_norm, 
                init,
                alpha=alpha,    # elasticity
                beta=beta,      # rigidity
                gamma=gamma,   # time step
                w_line=0,      # attract to light
                w_edge=1,      # attract to edges
                max_num_iter=max_iterations
            )
            # Создаем маску из контура
            mask = np.zeros_like(gray, dtype=np.uint8)
            # Заполняем контур
            rr, cc = polygon(snake[:, 0], snake[:, 1], gray.shape)
            mask[rr, cc] = 255
            
            return mask
        except Exception as e:
            warnings.warn(f"Active contour failed: {e}. Using fallback.")
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return mask
            # return np.zeros_like(gray, dtype=np.uint8)
    
    def _gvf_contour(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        gray_norm = gray.astype(float) / 255.0

        # Вычисляем градиенты
        grad_y, grad_x = np.gradient(gray_norm)
        
        # Вычисляем внешние силы (edge map)
        edge_map = grad_x**2 + grad_y**2
        edge_map = edge_map / (edge_map.max() + 1e-8)
        
        # Инициализируем GVF поле
        u = grad_x * edge_map
        v = grad_y * edge_map
        
        # Параметры GVF
        mu = self.params.get('mu', 0.2)
        iterations = self.params.get('iterations', 100)
        
        # Итеративно вычисляем GVF поле
        for _ in range(iterations):
            u_new = u + mu * cv2.Laplacian(u, -1) - edge_map * (u - grad_x)
            v_new = v + mu * cv2.Laplacian(v, -1) - edge_map * (v - grad_y)
            u, v = u_new, v_new
        
        # Вычисляем величину GVF
        gvf_magnitude = np.sqrt(u**2 + v**2)
        
        # Нормализуем и пороговое разделение
        gvf_norm = (gvf_magnitude / (gvf_magnitude.max() + 1e-8) * 255).astype(np.uint8)
        _, mask = cv2.threshold(gvf_norm, 50, 255, cv2.THRESH_BINARY)
        
        # Заполняем дыры
        mask = ndimage.binary_fill_holes(mask > 0).astype(np.uint8) * 255
        
        return mask

    def _morphological_snakes(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Сегментация морфологическими змеями.

        Итеративно расширяет или сужает бинарную маску на основе величины градиента.
        Области с низким градиентом "поглощаются", с высоким — отбрасываются.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Создаем начальную маску (окружность в центре)
            h, w = gray.shape
            center_y, center_x = h // 2, w // 2
            radius = min(center_x, center_y) // 2
            
            # Создаем начальную маску
            mask = np.zeros((h, w), dtype=bool)
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask[dist_from_center <= radius] = True
            
            # Параметры morphological snakes
            iterations = self.params.get('iterations', 100)
            smoothing = self.params.get('smoothing', 1)
            threshold = self.params.get('threshold', 0.5)
            
            # Применяем morphological snakes
            for _ in range(iterations):
                # Вычисляем градиент
                grad_y, grad_x = np.gradient(gray.astype(float))
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                # Нормализуем градиент
                grad_mag = grad_mag / (grad_mag.max() + 1e-8)
                
                # Расширяем или сужаем маску в зависимости от градиента
                expansion = grad_mag < threshold
                erosion = grad_mag > threshold
                
                mask[expansion] = True
                mask[erosion] = False
                
                # Сглаживание маски
                if smoothing > 0:
                    mask = morphology.binary_closing(mask, morphology.disk(smoothing))
                    mask = morphology.binary_opening(mask, morphology.disk(smoothing))
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"Morphological snakes failed: {e}. Using fallback.")
            # Резервный вариант
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return mask
        
    def _chan_vese(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Модель Chan-Vese — активные контуры без градиентов.

        Энергетическая модель, которая разделяет изображение на две области с минимальной
        внутрирегиональной дисперсией. Подходит для объектов без четких границ.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска: 255 — внутренняя область контура.
        """
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Нормализуем изображение
            gray_norm = gray.astype(float) / 255.0
            
            # Параметры Chan-Vese
            mu = self.params.get('mu', 0.25)
            lambda1 = self.params.get('lambda1', 1.0)
            lambda2 = self.params.get('lambda2', 1.0)
            tol = self.params.get('tol', 1e-3)
            max_iter = self.params.get('max_iter', 100)
            
            # Инициализируем контур (все изображение)
            init_level_set = np.ones(gray_norm.shape, dtype=np.float64)
            
            # Применяем Chan-Vese
            segmentation = chan_vese(
                gray_norm,
                mu=mu,
                lambda1=lambda1,
                lambda2=lambda2,
                tol=tol,
                max_num_iter=max_iter,
                init_level_set=init_level_set
            )
            
            # Создаем маску
            mask = (segmentation > 0.5).astype(np.uint8) * 255
            
            return mask
            
        except Exception as e:
            warnings.warn(f"Chan-Vese failed: {e}. Using fallback.")
            return self._otsu_thresholding(img)
        
    # ============ WATERSHED И ГРАФОВЫЕ ============
    def _watershed(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
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
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # # Вычисляем градиент
        # gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3)))
        
        # # Маркеры
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # kernel = np.ones((3, 3), np.uint8)
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg, sure_fg)
        
        # _, markers = cv2.connectedComponents(sure_fg)
        # markers = markers + 1
        # markers[unknown == 255] = 0
        
        # markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), markers)
        # mask = (markers > 1).astype(np.uint8) * 255
        
        # return mask

        # Гауссово размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Бинаризация с помощью Otsu
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Морфологические операции
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Определяем область фона
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Преобразование расстояния
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Маркеры для watershed
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Применяем watershed
        if len(img.shape) == 3:
            markers = cv2.watershed(img, markers)
        else:
            markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), markers)
        
        # Создаем маску (все что не фон и не граница)
        mask = (markers > 1).astype(np.uint8) * 255
        
        return mask
    
    def _random_walker(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Сегментация методом Random Walker.

        На основе маркеров (пользовательских или автоматических) решается задача на графе:
        каждый пиксель "принадлежит" тому маркеру, до которого "случайное блуждание" короче.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска переднего плана.
        """
        try:
            
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Создаем маркеры
            markers = np.zeros_like(gray, dtype=np.int8)
            
            rect = self.params.get('rect')
            h, w = gray.shape
            if rect is not None:
                x, y, rw, rh = rect
                markers[y:y+rh, x:x+rw] = 1
                markers[y+1:y+rh-1, x+1:x+rw-1] = 2
            else:
                markers[h//4:3*h//4, w//4:3*w//4] = 2
                markers[0:h//8, 0:w//8] = 1
                markers[7*h//8:, 7*w//8:] = 1
            
            # Применяем Random Walker
            labels = random_walker(gray, markers)
            
            # Создаем маску (все что не фон)
            mask = (labels == 2).astype(np.uint8) * 255
            
            return mask
            
        except Exception as e:
            warnings.warn(f"Random Walker failed: {e}. Using fallback.")
            return self._otsu_thresholding(img)
        
    # ============ SUPER-PIXEL МЕТОДЫ ============
    def _quickshift(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Сегментация методом Quickshift (реализована через MeanShift как аналог).

        Находит моды в плотности распределения пикселей в пространстве признаков.
        Группирует пиксели, принадлежащие одной моде.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8). Самый крупный кластер — фон.
        """
        try:
            h, w = img.shape[:2]
            pixels = img.reshape(-1, 3)
            
            # Применяем MeanShift (как аналог Quickshift)
            bandwidth = self.params.get('bandwidth', 0.5)
            meanshift = MeanShift(bandwidth=bandwidth, n_jobs=-1)
            labels = meanshift.fit_predict(pixels)
            labels_2d = labels.reshape(h, w)
            
            # Находим самый большой кластер
            unique, counts = np.unique(labels, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            
            # Создаем маску
            mask = (labels_2d != bg_label).astype(np.uint8) * 255
            
            return mask
            
        except Exception as e:
            warnings.warn(f"Quickshift failed: {e}. Using fallback.")
            return self._kmeans_segmentation(img)
    
    def _slic(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        SLIC (Simple Linear Iterative Clustering) — суперпиксельная сегментация.

        Группирует пиксели в компактные, однородные регионы (суперпиксели) на основе пространственной
        и цветовой близости. Самый крупный суперпиксель считается фоном.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8): 255 — все суперпиксели, кроме фона.
        """
        try:
            # Параметры SLIC
            n_segments = self.params.get('n_segments', 100)
            compactness = self.params.get('compactness', 10.0)
            
            # Применяем SLIC
            segments = skseg.slic(img, n_segments=n_segments, compactness=compactness)
            
            # Находим самый большой сегмент
            unique, counts = np.unique(segments, return_counts=True)
            bg_segment = unique[np.argmax(counts)]
            
            # Создаем маску
            mask = (segments != bg_segment).astype(np.uint8) * 255
            
            return mask
            
        except Exception as e:
            warnings.warn(f"SLIC failed: {e}. Using fallback.")
            return self._kmeans_segmentation(img)
        
    def _slic(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        SLIC (Simple Linear Iterative Clustering) — суперпиксельная сегментация.

        Группирует пиксели в компактные, однородные регионы (суперпиксели) на основе пространственной
        и цветовой близости. Самый крупный суперпиксель считается фоном.

        Args:
            img: Входное изображение (RGB).

        Returns:
            Бинарная маска: 255 — все суперпиксели, кроме фона.
        """
        try:
            
            if len(img.shape) == 3:
                img_rgb = img
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Параметры
            n_segments = self.params.get('n_segments', 100)
            compactness = self.params.get('compactness', 10.0)
            max_iter = self.params.get('max_iter', 10)

            # Применяем SLIC
            segments = sk_slic(
                img_rgb,
                n_segments=n_segments,
                compactness=compactness,
                max_num_iter=max_iter,
                enforce_connectivity=True,
                start_label=0
            )

            # Находим самый большой сегмент — считаем его фоном
            unique, counts = np.unique(segments, return_counts=True)
            if len(unique) > 0:
                bg_label = unique[np.argmax(counts)]
                mask_np = (segments != bg_label).astype(np.uint8) * 255
            else:
                mask_np = np.zeros_like(segments, dtype=np.uint8)

            return mask_np

        except Exception as e:
            warnings.warn(f"SLIC failed: {e}. Using fallback to KMeans.")
            return self._kmeans_segmentation(img)
    
    def _felzenszwalb(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Алгоритм Felzenszwalb — иерархическая сегментация на основе графов.

        Строит сегментацию, начиная с мелких регионов и объединяя их, если внутреннее различие
        меньше межрегионального. Очень эффективен для выделения объектов разного масштаба.

        Args:
            img: Входное изображение (RGB).

        Returns:
            Бинарная маска: 255 — все регионы, кроме самого крупного (фона).
        """
        try:

            if len(img.shape) == 3:
                img_rgb = img
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Параметры
            scale = self.params.get('scale', 100)
            sigma = self.params.get('sigma', 0.8)
            min_size = self.params.get('min_size', 50)
            
            # Применяем Felzenszwalb
            segments = skseg.felzenszwalb(img, 
                                          scale=scale, 
                                          sigma=sigma, 
                                          min_size=min_size)
            
            # Находим самый большой сегмент
            unique, counts = np.unique(segments, return_counts=True)
            if len(unique) > 0:
                bg_label = unique[np.argmax(counts)]
                mask_np = (segments != bg_label).astype(np.uint8) * 255
            else:
                mask_np = np.zeros_like(segments, dtype=np.uint8)
            
            return mask_np
            
        except Exception as e:
            warnings.warn(f"Felzenszwalb failed: {e}. Using fallback.")
            return self._kmeans_segmentation(img)

    # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============
    def _grabcut(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """
        Интерактивная сегментация GrabCut.

        Использует прямоугольник для инициализации фона и переднего плана.
        Строит модели цветового распределения (GMM) и уточняет границы итеративно.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8) переднего плана.
        """
        # Параметры
        rect = self.params.get('rect', None)
        iter_count = self.params.get('iterations', 10)
        
        # Создаем маску и модель
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        # Если прямоугольник не задан, используем центральную часть
        if rect is None:
            h, w = img.shape[:2]
            rect = (int(w*0.25), int(h*0.25), int(w*0.5), int(h*0.5))
        
        # Применяем GrabCut
        mask, bgd_model, fgd_model = cv2.grabCut(
            img, mask, rect, bgd_model, fgd_model, 
            iter_count, cv2.GC_INIT_WITH_RECT
        )
        
        # Создаем финальную маску (0-255)
        mask_final = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        
        # Опционально: применение морфологических операций для улучшения результата
        kernel = np.ones((3, 3), np.uint8)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return mask_final