# OpenCVSegmenter.py

# Импорт основных библиотек

import cv2
import numpy as np
from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)
import warnings
from collections import deque
from scipy import ndimage
import math

class OpenCVSegmenter:
    """
    Класс для реализации сегментации изображений с использованием чистого OpenCV.
    """  
    def __init__(
        self, 
        method: str = "global_thresholding", 
        **kwargs
    ) -> None:
        self.method: str = method
        self.params: Dict[str, Any] = kwargs
        self._setup_methods()
    
    def _setup_methods(self) -> None:
        """Регистрация всех доступных методов сегментации."""
        self.methods: Dict[str, np.ndarray] = {
            # ============ ПОРОГОВЫЕ МЕТОДЫ СЕГМЕНТАЦИИ ============
            "global_thresholding": self._opencv_global_thresholding,
            "adaptive_thresholding": self._opencv_adaptive_thresholding,
            "otsu_thresholding": self._opencv_otsu_thresholding,
            "threshold_niblack": self._opencv_threshold_niblack,
            "threshold_sauvola": self._opencv_threshold_sauvola,

             # ============ КРАЕВЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "sobel_edge": self._opencv_sobel_edge,
            "canny_edge": self._opencv_canny_edge,

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
             raise ValueError(f"Неизвестный метод: {self.method}. "
                           f"Доступные методы: {list(self.methods.keys())}")
    
    def segment(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Основной метод сегментации.
        
        Args:
            image: Входное изображение (RGB, grayscale или любой формат)
        
        Returns:
            np.ndarray: Бинарная маска сегментации (0-255)
        """
        return self.methods[self.method](image)
    
    def segment_with_mask(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сегментация с возвратом визуализации и маски.
        
        Args:
            image: Входное изображение
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Визуализация и маска
        """
        mask = self.segment(image)
        
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
        
        overlay[mask > 0] = [255, 0, 0]
        result = cv2.addWeighted(overlay, 0.9, 
                                cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape)==2 else image, 
                                0.1, 0)
        
        return result, mask
    
    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    # ============ ПОРОГОВЫЕ МЕТОДЫ ============
    
    def _opencv_global_thresholding(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Глобальная пороговая сегментация"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        threshold = self.params.get('threshold', 127)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return mask
    
    def _opencv_adaptive_thresholding(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Адаптивная пороговая сегментация"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        block_size = self.params.get('block_size', 11)
        C = self.params.get('C', 2)
        
        if block_size % 2 == 0:
            block_size += 1
        
        mask = cv2.adaptiveThreshold(gray, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block_size, C)
        return mask
    
    def _opencv_otsu_thresholding(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Метод Оцу"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    
    def _opencv_threshold_niblack(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Порог Ниблака"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        window_size = self.params.get('window_size', 15)
        k = self.params.get('k', -0.2)
        
        # Вычисляем локальное среднее и стандартное отклонение
        mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
        mean_sq = cv2.boxFilter(gray.astype(np.float32)**2, cv2.CV_32F, (window_size, window_size))
        std = np.sqrt(mean_sq - mean**2)
        
        # Вычисляем порог
        threshold = mean + k * std
        mask = (gray.astype(np.float32) > threshold).astype(np.uint8) * 255
        
        return mask
    
    def _opencv_threshold_sauvola(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Порог Сауволы"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        window_size = self.params.get('window_size', 15)
        k = self.params.get('k', 0.5)
        r = self.params.get('r', 128)
        
        # Вычисляем локальное среднее и стандартное отклонение
        mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
        mean_sq = cv2.boxFilter(gray.astype(np.float32)**2, cv2.CV_32F, (window_size, window_size))
        std = np.sqrt(mean_sq - mean**2)
        
        # Вычисляем порог
        threshold = mean * (1 + k * (std / r - 1))
        mask = (gray.astype(np.float32) > threshold).astype(np.uint8) * 255
        
        return mask
    
    # ============ МЕТОДЫ НА ОСНОВЕ КРАЕВ ============
    def _opencv_sobel_edge(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Детектор границ Собеля"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        threshold = self.params.get('threshold', 50)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        _, mask = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        return mask
    
    def _opencv_canny_edge(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Детектор границ Кэнни"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        low = self.params.get('low', 50)
        high = self.params.get('high', 150)
        
        mask = cv2.Canny(gray, low, high)
        return mask
    
    # ============ РЕГИОНАЛЬНЫЕ МЕТОДЫ ============
    def _opencv_region_growing(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Region Growing сегментация"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        h, w = gray.shape
        seed = self.params.get('seed', (w//2, h//2))
        tolerance = self.params.get('tolerance', 15)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)
        
        queue = deque([seed])
        start_value = gray[seed[1], seed[0]]
        
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
                        queue.append((x + dx, y + dy))
        
        return mask
    
    def _opencv_split_and_merge(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Split and Merge алгоритм"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        h, w = gray.shape
        threshold = self.params.get('threshold', 20)
        min_size = self.params.get('min_size', 50)
        
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
                [(x, y) for x, y in region if x > x_mid and y > y_mid]
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
                
                for j, reg2 in enumerate(regions[i+1:], i+1):
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
                mask[y, x] = 255
            return mask
        
        return np.zeros((h, w), dtype=np.uint8)
    
    def _opencv_floodfill(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """FloodFill сегментация"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Начальная точка
        seed = self.params.get('seed', (w//2, h//2))
        
        # Создаем маску
        mask = np.zeros((h+2, w+2), np.uint8)
        
        # Параметры заливки
        tolerance = self.params.get('tolerance', 20)
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
        
        # Применяем floodfill
        cv2.floodFill(gray.copy(), mask, seed, 255, 
                     (tolerance,)*3, (tolerance,)*3, flags)
        
        return mask[1:-1, 1:-1] * 255

    # ============ КЛАСТЕРИЗАЦИЯ ============
    def _opencv_kmeans_segmentation(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """K-Means кластеризация на OpenCV"""
        if len(img.shape) == 2:
            # Если grayscale, конвертируем в BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3).astype(np.float32)
        
        k = self.params.get('k', 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        labels = labels.reshape(h, w)
        
        # Находим самый большой кластер
        unique, counts = np.unique(labels, return_counts=True)
        bg_label = unique[np.argmax(counts)]
        
        mask = (labels != bg_label).astype(np.uint8) * 255
        return mask
    
    def _opencv_dbscan_segmentation(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """DBSCAN на OpenCV (упрощенная реализация)"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Используем упрощенный подход на основе расстояния
        binary = np.zeros_like(gray, dtype=np.uint8)
        binary[gray > 127] = 255
        
        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray, dtype=np.uint8)
        min_area = self.params.get('min_area', 100)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        return mask
    
    def _opencv_meanshift(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """MeanShift сегментация"""
        spatial_radius = self.params.get('spatial_radius', 60)
        color_radius = self.params.get('color_radius', 60)
        max_level = self.params.get('max_level', 1)
        
        # Применяем MeanShift
        shifted = cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius, max_level)
        
        # Конвертируем в grayscale и пороговую обработку
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
    
    # ============ АКТИВНЫЕ КОНТУРЫ ============
    
    def _opencv_active_contour(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Активные контуры на OpenCV"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Начальный контур (окружность)
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y) // 2
        
        # Создаем маску контура
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Применяем морфологические операции для имитации активного контура
        kernel = np.ones((5, 5), np.uint8)
        for _ in range(self.params.get('iterations', 10)):
            edges = cv2.Canny(gray, 100, 200)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _opencv_gvf_contour(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Gradient Vector Flow (упрощенная версия)"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Вычисляем градиенты
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Матрица границ
        edges = cv2.Canny(gray, 100, 200)
        
        # Распространение градиентов
        u = grad_x.copy()
        v = grad_y.copy()
        
        mu = self.params.get('mu', 0.1)
        iterations = self.params.get('iterations', 50)
        
        for _ in range(iterations):
            laplacian_u = cv2.Laplacian(u, cv2.CV_64F)
            laplacian_v = cv2.Laplacian(v, cv2.CV_64F)
            
            edge_weight = edges.astype(np.float64) / 255.0
            
            u = u + mu * laplacian_u - edge_weight * (u - grad_x)
            v = v + mu * laplacian_v - edge_weight * (v - grad_y)
        
        # Величина GVF
        gvf_mag = np.sqrt(u**2 + v**2)
        gvf_mag = np.uint8(255 * gvf_mag / np.max(gvf_mag))
        
        _, mask = cv2.threshold(gvf_mag, 50, 255, cv2.THRESH_BINARY)
        return mask
    
    def _opencv_morphological_snakes(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Морфологические змеи"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Начальная маска (окружность в центре)
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (w//2, h//2), min(w, h)//4, 255, -1)
        
        iterations = self.params.get('iterations', 50)
        kernel = np.ones((3, 3), np.uint8)
        
        for _ in range(iterations):
            # Градиент изображения
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_mag = np.uint8(255 * grad_mag / np.max(grad_mag))
            
            _, grad_binary = cv2.threshold(grad_mag, 50, 255, cv2.THRESH_BINARY)
            
            # Расширение/сужение на основе градиента
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(grad_binary))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _opencv_chan_vese(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Chan-Vese активные контуры без градиентов"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Начальная маска (центральная область)
        mask = np.zeros((h, w), np.uint8)
        cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
        
        iterations = self.params.get('iterations', 100)
        mu = self.params.get('mu', 0.25)
        
        for _ in range(iterations):
            # Вычисляем средние значения внутри и снаружи маски
            inside_mean = np.mean(gray[mask > 0]) if np.any(mask > 0) else 0
            outside_mean = np.mean(gray[mask == 0]) if np.any(mask == 0) else 0
            
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
        
        return mask

    # ============ WATERSHED И ГРАФОВЫЕ ============
    
    def _opencv_watershed(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Watershed алгоритм"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Морфологические операции
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Фон
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Преобразование расстояния
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
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
        return mask
    
    def _opencv_random_walker(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Random Walker (упрощенная версия)"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Создаем маркеры
        markers = np.zeros((h, w), dtype=np.int32)
        
        # Центральная область - объект
        cv2.rectangle(markers, (w//4, h//4), (3*w//4, 3*h//4), 2, -1)
        
        # Углы - фон
        corner_size = min(h, w) // 8
        cv2.rectangle(markers, (0, 0), (corner_size, corner_size), 1, -1)
        cv2.rectangle(markers, (w-corner_size, 0), (w, corner_size), 1, -1)
        cv2.rectangle(markers, (0, h-corner_size), (corner_size, h), 1, -1)
        cv2.rectangle(markers, (w-corner_size, h-corner_size), (w, h), 1, -1)
        
        # Применяем Watershed с маркерами
        if len(img.shape) == 3:
            markers = cv2.watershed(img, markers)
        else:
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(color_img, markers)
        
        mask = (markers == 2).astype(np.uint8) * 255
        return mask

    # ============ SUPER-PIXEL МЕТОДЫ ============
    def _opencv_quickshift(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Quickshift (упрощенная версия на основе superpixels)"""
        # Используем простую кластеризацию по цвету
        return self._kmeans_segmentation(img)
    
    def _opencv_slic(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """SLIC superpixels (упрощенная реализация)"""
        h, w = img.shape[:2]
        
        # Разбиваем изображение на регионы
        region_size = self.params.get('region_size', 20)
        ruler = self.params.get('ruler', 10.0)
        
        # Создаем сетку суперпикселей
        mask = np.zeros((h, w), np.uint8)
        
        for y in range(0, h, region_size):
            for x in range(0, w, region_size):
                # Простая цветовая кластеризация в регионе
                region = img[y:min(y+region_size, h), x:min(x+region_size, w)]
                if len(region) == 0:
                    continue
                
                if len(region.shape) == 3:
                    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                else:
                    region_gray = region
                
                _, region_mask = cv2.threshold(region_gray, 0, 255, 
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mask[y:min(y+region_size, h), x:min(x+region_size, w)] = region_mask
        
        return mask
    
    def _opencv_felzenszwalb(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Felzenszwalb (графовая сегментация - упрощенная версия)"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Применяем несколько порогов для создания иерархической сегментации
        thresholds = [50, 100, 150]
        masks = []
        
        for thresh in thresholds:
            _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            masks.append(mask)
        
        # Комбинируем маски
        combined = np.zeros_like(gray, dtype=np.uint8)
        for mask in masks:
            combined = cv2.bitwise_or(combined, mask)
        
        return combined
    
    # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============
    def _opencv_grabcut(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """GrabCut сегментация"""
        h, w = img.shape[:2]
        
        # Создаем маску
        mask = np.zeros((h, w), np.uint8)
        
        # Прямоугольник для инициализации (центр изображения)
        rect = self.params.get('rect', (w//4, h//4, w//2, h//2))
        
        # Временные массивы
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Применяем GrabCut
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 
                   self.params.get('iterations', 5), cv2.GC_INIT_WITH_RECT)
        
        # Создаем финальную маску
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result_mask = mask2 * 255
        
        return result_mask