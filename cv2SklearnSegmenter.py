# cv2SklearnSegmenter.py
from BaseSegmenter import BaseSegmenter
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Dict, Any
from collections import deque
from scipy import ndimage
from skimage import segmentation, feature, measure
import warnings


class CV2SklearnSegmenter(BaseSegmenter):
    """Класс для методов сегментации с использованием CV2 и Sklearn"""
    
    def __init__(self, 
                 method: str = "global_thresholding", 
                 **kwargs
    ) -> None:
        super().__init__()
        self.method: str = method
        self.params: Dict[str, Any] = kwargs
        self._setup_method()

        self._needs_gray = method in [
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
            "morphological_snakes"
        ]
    
    def _setup_method(self) -> None:
        """Настройка выбранного метода"""
        method_map: Dict[str, Any] = {
            "global_thresholding": self._global_thresholding,
            "adaptive_thresholding": self._adaptive_thresholding,
            "otsu_thresholding": self._otsu_thresholding,
            "region_growing": self._region_growing,
            "split_and_merge": self._split_and_merge,
            "sobel_edge": self._sobel_edge,
            "canny_edge": self._canny_edge,
            "kmeans_segmentation": self._kmeans_segmentation,
            "dbscan_segmentation": self._dbscan_segmentation,
            "active_contour": self._active_contour,
            "gvf_contour": self._gvf_contour,
            "watershed": self._watershed,
            "meanshift": self._meanshift,
            "grabcut": self._grabcut,
            "floodfill": self._floodfill,
            "morphological_snakes": self._morphological_snakes,
            "quickshift": self._quickshift,
            "slic": self._slic,
            "felzenszwalb": self._felzenszwalb
        }
        
        if self.method not in method_map:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        self._segment_func = method_map[self.method]
    
    def segment(self, 
                image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """Сегментация изображения"""
        img_array: np.ndarray = self.preprocess_image(image, 
                                                      as_gray=self._needs_gray)
        return self._segment_func(img_array)
    
    def segment_with_mask(self, 
                          image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Сегментация с возвратом маски и обработанного изображения"""
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
        
        overlay = img_array.copy()
        overlay[mask > 0] = [255, 0, 0] # Красный цвет для маски
        result = cv2.addWeighted(img_array, 0.5, overlay, 0.5, 0)
        
        return result, mask
    
    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    
    def _global_thresholding(self, 
                             img: np.ndarray
    ) -> np.ndarray:
        """Глобальная пороговая обработка - возвращает маску 0-255"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        threshold = self.params.get('threshold', 127)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return mask
    
    def _adaptive_thresholding(self, 
                               img: np.ndarray
    ) -> np.ndarray:
        """Адаптивная пороговая обработка - возвращает маску 0-255"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        block_size = self.params.get('block_size', 11)
        c = self.params.get('C', 2)

        if block_size % 2 == 0:
            block_size += 1
        
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
        return mask
    
    def _otsu_thresholding(self, 
                           img: np.ndarray
    ) -> np.ndarray:
        """Метод Оцу"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    
    def _region_growing(self, 
                        img: np.ndarray
    ) -> np.ndarray:
        """Region Growing - возвращает маску 0-255"""
        
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
        
        region_mask = np.zeros_like(gray, dtype=bool)
        visited = np.zeros_like(gray, dtype=bool)
        queue = deque([seed])
        
        region_mean = float(gray[seed[1], seed[0]])
        region_pixels = 1
        
        while queue:
            x, y = queue.popleft()
            
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            
            if visited[y, x]:
                continue
            
            visited[y, x] = True
            pixel_value = gray[y, x]
            
            # Проверяем сходство со средним значением региона
            if abs(pixel_value - region_mean) <= tolerance:
                # # Добавляем пиксель в регион
                # region_mask[y, x] = True
                # # Обновляем среднее значение региона
                # region_mean = (region_mean * region_pixels + pixel_value) / (region_pixels + 1)
                # region_pixels += 1
                
                # # Добавляем соседей 4-связности
                # neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                # queue.extend(neighbors)
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
    
    # def _split_and_merge(self, 
    #                      img: np.ndarray
    # ) -> np.ndarray:
    #     """Split-and-Merge - возвращает маску 0-255"""
    #     if len(img.shape) == 3:
    #         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     else:
    #         gray = img
        
    #     min_region_size = self.params.get('min_region_size', 50)
    #     threshold = self.params.get('threshold', 20)
        
    #     h, w = gray.shape
        
    #     def region_properties(region):
    #         if len(region) == 0:
    #             return 0, 0
    #         intensities = [gray[y, x] for x, y in region]
    #         return np.mean(intensities), np.std(intensities)
        
    #     def split(region, min_size, thresh):
    #         if len(region) <= min_size:
    #             return [region]
            
    #         mean_intensity, std_intensity = region_properties(region)
    #         if std_intensity < thresh:
    #             return [region]
            
    #         x_coords = [p[0] for p in region]
    #         y_coords = [p[1] for p in region]
            
    #         x_mid = (min(x_coords) + max(x_coords)) // 2
    #         y_mid = (min(y_coords) + max(y_coords)) // 2
            
    #         subregions = [
    #             [(x, y) for x, y in region if x <= x_mid and y <= y_mid],
    #             [(x, y) for x, y in region if x > x_mid and y <= y_mid],
    #             [(x, y) for x, y in region if x <= x_mid and y > y_mid],
    #             [(x, y) for x, y in region if x > x_mid and y > y_mid]
    #         ]
            
    #         result = []
    #         for subregion in subregions:
    #             result.extend(split(subregion, min_size, thresh))
    #         return result
        
    #     def merge(regions, thresh):
    #         merged = []
    #         used = [False] * len(regions)
            
    #         for i, region1 in enumerate(regions):
    #             if used[i]:
    #                 continue
    #             current_merge = region1.copy()
    #             mean1, std1 = region_properties(region1)
                
    #             for j, region2 in enumerate(regions[i+1:], i+1):
    #                 if used[j]:
    #                     continue
    #                 mean2, std2 = region_properties(region2)
    #                 if abs(mean1 - mean2) < thresh:
    #                     current_merge.extend(region2)
    #                     used[j] = True
                
    #             merged.append(current_merge)
    #             used[i] = True
            
    #         return merged
        
    #     # Начинаем с целого изображения как одного региона
    #     initial_region = [(x, y) for x in range(w) for y in range(h)]
        
    #     # Фаза разделения
    #     regions = split(initial_region, min_region_size, threshold)
        
    #     # Фаза слияния
    #     regions = merge(regions, threshold)
        
    #     # Создаем маску (берем самый большой регион после фона)
    #     region_sizes = [len(region) for region in regions]
    #     if len(region_sizes) > 1:
    #         foreground_idx = np.argsort(region_sizes)[-2]
    #         mask = np.zeros_like(gray, dtype=bool)
    #         for x, y in regions[foreground_idx]:
    #             mask[y, x] = True
    #         return mask.astype(np.uint8) * 255
    #     else:
    #         return np.zeros_like(gray, dtype=np.uint8)
    def _split_and_merge(self, 
                         img: np.ndarray
    ) -> np.ndarray:
        """Split-and-Merge - возвращает маску 0-255"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        min_region_size = self.params.get('min_region_size', 50)
        threshold = self.params.get('threshold', 20)
        
        h, w = gray.shape
        
        def recursive_split(region, min_size, thresh):
            """Рекурсивное разделение региона"""
            y, x, h_r, w_r = region
            
            if h_r <= min_size or w_r <= min_size:
                return [region]
            
            region_pixels = gray[y:y+h_r, x:x+w_r]
            if region_pixels.std() < thresh:
                return [region]
            
            h_half, w_half = h_r // 2, w_r // 2
            
            subregions = [
                (y, x, h_half, w_half),
                (y, x + w_half, h_half, w_r - w_half),
                (y + h_half, x, h_r - h_half, w_half),
                (y + h_half, x + w_half, h_r - h_half, w_r - w_half)
            ]
            
            result = []
            for subregion in subregions:
                result.extend(recursive_split(subregion, min_size, thresh))
            return result
        
        def merge_regions(regions, thresh):
            """Объединение похожих регионов"""
            if len(regions) <= 1:
                return regions
            
            merged = []
            used = [False] * len(regions)
            
            for i, (y1, x1, h1, w1) in enumerate(regions):
                if used[i]:
                    continue
                
                current_region = [y1, x1, h1, w1]
                region1_mean = gray[y1:y1+h1, x1:x1+w1].mean()
                
                for j, (y2, x2, h2, w2) in enumerate(regions[i+1:], i+1):
                    if used[j]:
                        continue
                    
                    region2_mean = gray[y2:y2+h2, x2:x2+w2].mean()
                    
                    if abs(region1_mean - region2_mean) < thresh:
                        # Объединяем регионы
                        new_y = min(y1, y2)
                        new_x = min(x1, x2)
                        new_h = max(y1+h1, y2+h2) - new_y
                        new_w = max(x1+w1, x2+w2) - new_x
                        current_region = [new_y, new_x, new_h, new_w]
                        used[j] = True
                
                merged.append(tuple(current_region))
                used[i] = True
            
            return merged
        
        # Начинаем с целого изображения
        initial_region = (0, 0, h, w)
        regions = recursive_split(initial_region, min_region_size, threshold)
        regions = merge_regions(regions, threshold)
        
        # Создаем маску (выбираем второй по величине регион)
        if len(regions) > 1:
            # Сортируем по размеру
            region_sizes = [(i, r[2] * r[3]) for i, r in enumerate(regions)]
            region_sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Выбираем второй по величине регион (предположительно объект)
            idx = region_sizes[1][0]
            mask = np.zeros((h, w), dtype=np.uint8)
            y, x, h_r, w_r = regions[idx]
            mask[y:y+h_r, x:x+w_r] = 255
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        
        return mask
    
    def _sobel_edge(self, 
                    img: np.ndarray
    ) -> np.ndarray:
        """Оператор Собеля - возвращает маску 0-255"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        threshold = self.params.get('threshold', 50)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        # mask = (sobel > threshold)
        # return mask.astype(np.uint8) * 255
        sobel_norm = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
        _, mask = cv2.threshold(sobel_norm.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _canny_edge(self, 
                    img: np.ndarray
    ) -> np.ndarray:
        """Оператор Кэнни - возвращает маску 0-255"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        low = self.params.get('low', 50)
        high = self.params.get('high', 150)
        edges = cv2.Canny(gray, low, high)
        return edges
    
    def _kmeans_segmentation(self, 
                             img: np.ndarray
    ) -> np.ndarray:
        """K-Means кластеризация - возвращает маску 0-255"""
        from sklearn.cluster import KMeans
        
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
    
    def _dbscan_segmentation(self, 
                             img: np.ndarray
    ) -> np.ndarray:
        """DBSCAN кластеризация - возвращает маску 0-255"""
        from sklearn.cluster import DBSCAN
        
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
    
    def _active_contour(self, 
                        img: np.ndarray
    ) -> np.ndarray:
        """Active Contour (Snakes) - возвращает маску 0-255"""
        from skimage import segmentation
        
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
            from skimage.draw import polygon
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
    
    def _gvf_contour(self, 
                     img: np.ndarray
    ) -> np.ndarray:
        """Gradient Vector Flow - возвращает маску 0-255"""
        
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
    
    def _watershed(self, 
                   img: np.ndarray
    ) -> np.ndarray:
        """Watershed сегментация - возвращает маску 0-255"""
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
    
    def _meanshift(self, 
                   img: np.ndarray
    ) -> np.ndarray:
        """MeanShift сегментация - возвращает маску 0-255"""
        from sklearn.cluster import MeanShift
        
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
    
    def _grabcut(self, 
                 img: np.ndarray
    ) -> np.ndarray:
        """GrabCut сегментация - возвращает маску 0-255"""
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
    
    def _floodfill(self, 
                   img: np.ndarray
    ) -> np.ndarray:
        """FloodFill сегментация - возвращает маску 0-255"""
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
    
    def _morphological_snakes(self, 
                              img: np.ndarray
    ) -> np.ndarray:
        """Morphological Snakes - возвращает маску 0-255"""
        try:
            from skimage import morphology
            
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
    
    def _quickshift(self, 
                    img: np.ndarray
    ) -> np.ndarray:
        """Quickshift сегментация - возвращает маску 0-255"""
        try:
            from sklearn.cluster import MeanShift  # Используем MeanShift как аналог
            
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
    
    def _slic(self, 
              img: np.ndarray
    ) -> np.ndarray:
        """SLIC сегментация - возвращает маску 0-255"""
        try:
            from skimage import segmentation as skseg
            
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
    
    def _felzenszwalb(self, 
                      img: np.ndarray
    ) -> np.ndarray:
        """Felzenszwalb сегментация - возвращает маску 0-255"""
        try:
            from skimage import segmentation as skseg
            
            # Параметры
            scale = self.params.get('scale', 100)
            sigma = self.params.get('sigma', 0.8)
            min_size = self.params.get('min_size', 50)
            
            # Применяем Felzenszwalb
            segments = skseg.felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
            
            # Находим самый большой сегмент
            unique, counts = np.unique(segments, return_counts=True)
            bg_segment = unique[np.argmax(counts)]
            
            # Создаем маску
            mask = (segments != bg_segment).astype(np.uint8) * 255
            
            return mask
            
        except Exception as e:
            warnings.warn(f"Felzenszwalb failed: {e}. Using fallback.")
            return self._kmeans_segmentation(img)