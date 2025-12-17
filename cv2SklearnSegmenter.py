from BaseSegmenter import BaseSegmenter
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple

class CV2SklearnSegmenter(BaseSegmenter):
    """Класс для методов сегментации с использованием CV2 и Sklearn"""
    
    def __init__(self, 
                 method: str = "global_thresholding", 
                 **kwargs
    ) -> None:
        super().__init__()
        self.method = method
        self.params = kwargs
        self._setup_method()
    
    def _setup_method(self) -> None:
        """Настройка выбранного метода"""
        method_map = {
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
        }
        
        if self.method not in method_map:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        self._segment_func = method_map[self.method]
    
    def segment(self, 
                image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """Сегментация изображения"""
        img_array = self.preprocess_image(image)
        return self._segment_func(img_array)
    
    def segment_with_mask(self, 
                          image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Сегментация с возвратом маски и обработанного изображения"""
        img_array = self.preprocess_image(image)
        mask = self._segment_func(img_array)
        
        # Создаем визуализацию
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        overlay = img_array.copy()
        overlay[mask > 0] = [255, 0, 0]
        result = cv2.addWeighted(img_array, 0.5, overlay, 0.5, 0)
        
        return result, mask
    
    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    
    def _global_thresholding(self, 
                             img: np.ndarray
    ) -> np.ndarray:
        """Глобальная пороговая обработка"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        threshold = self.params.get('threshold', 127)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return (mask > 0).astype(np.uint8) * 255
    
    def _adaptive_thresholding(self, 
                               img: np.ndarray
    ) -> np.ndarray:
        """Адаптивная пороговая обработка"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        block_size = self.params.get('block_size', 11)
        c = self.params.get('C', 2)
        
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
        return (mask > 0).astype(np.uint8) * 255
    
    def _otsu_thresholding(self, 
                           img: np.ndarray
    ) -> np.ndarray:
        """Метод Оцу"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (mask > 0).astype(np.uint8) * 255
    
    def _region_growing(self, 
                        img: np.ndarray
    ) -> np.ndarray:
        """Region Growing"""
        from collections import deque
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        seed = self.params.get('seed', None)
        tolerance = self.params.get('tolerance', 15)
        
        h, w = gray.shape
        
        if seed is None:
            seed = (w // 2, h // 2)
        
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
            
            if abs(pixel_value - region_mean) <= tolerance:
                region_mask[y, x] = True
                region_mean = (region_mean * region_pixels + pixel_value) / (region_pixels + 1)
                region_pixels += 1
                
                neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                queue.extend(neighbors)
        
        return region_mask.astype(np.uint8) * 255
    
    def _split_and_merge(self, 
                         img: np.ndarray
    ) -> np.ndarray:
        """Split-and-Merge"""
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
        
        initial_region = [(x, y) for x in range(w) for y in range(h)]
        regions = split(initial_region, min_region_size, threshold)
        regions = merge(regions, threshold)
        
        region_sizes = [len(region) for region in regions]
        if len(region_sizes) > 1:
            foreground_idx = np.argsort(region_sizes)[-2]
            mask = np.zeros_like(gray, dtype=bool)
            for x, y in regions[foreground_idx]:
                mask[y, x] = True
            return mask.astype(np.uint8) * 255
        else:
            return np.zeros_like(gray, dtype=np.uint8)
    
    def _sobel_edge(self, 
                    img: np.ndarray
    ) -> np.ndarray:
        """Оператор Собеля"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        threshold = self.params.get('threshold', 50)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        mask = (sobel > threshold)
        return mask.astype(np.uint8) * 255
    
    def _canny_edge(self, 
                    img: np.ndarray
    ) -> np.ndarray:
        """Оператор Кэнни"""
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
        """K-Means кластеризация"""
        from sklearn.cluster import KMeans
        
        k = self.params.get('k', 3)
        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
        labels = kmeans.labels_.reshape(h, w)
        
        unique, counts = np.unique(labels, return_counts=True)
        bg_label = unique[np.argmax(counts)]
        mask = (labels != bg_label)
        return mask.astype(np.uint8) * 255
    
    def _dbscan_segmentation(self, 
                             img: np.ndarray
    ) -> np.ndarray:
        """DBSCAN кластеризация"""
        from sklearn.cluster import DBSCAN
        
        eps = self.params.get('eps', 10)
        min_samples = self.params.get('min_samples', 100)
        
        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3)
        
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
        labels = db.labels_.reshape(h, w)
        
        mask = (labels != -1) & (labels != 0)
        return mask.astype(np.uint8) * 255
    
    def _active_contour(self, 
                        img: np.ndarray
    ) -> np.ndarray:
        """Active Contour (Snakes)"""
        from skimage import segmentation
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        gray_norm = gray.astype(float) / 255.0
        
        h, w = gray_norm.shape
        center_y, center_x = h // 2, w // 2
        radius = min(center_x, center_y) * 0.7
        
        s = np.linspace(0, 2 * np.pi, 400)
        r = center_y + radius * np.sin(s)
        c = center_x + radius * np.cos(s)
        init = np.array([r, c]).T
        
        try:
            snake = segmentation.active_contour(
                gray_norm, init,
                alpha=0.01, beta=0.1, gamma=0.001,
                w_line=0, w_edge=1, max_iterations=2000
            )
            
            mask = np.zeros_like(gray, dtype=bool)
            from skimage.draw import polygon
            rr, cc = polygon(snake[:, 0], snake[:, 1], gray.shape)
            mask[rr, cc] = True
            
            return mask.astype(np.uint8) * 255
        except:
            return np.zeros_like(gray, dtype=np.uint8)
    
    def _gvf_contour(self, 
                     img: np.ndarray
    ) -> np.ndarray:
        """Gradient Vector Flow"""
        from skimage import feature
        from scipy import ndimage
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        gray_norm = gray.astype(float) / 255.0
        edges = feature.canny(gray_norm, sigma=1.5)
        mask = ndimage.binary_fill_holes(edges)
        
        return mask.astype(np.uint8) * 255
    
    def _watershed(self, 
                   img: np.ndarray
    ) -> np.ndarray:
        """Watershed сегментация"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Вычисляем градиент
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3)))
        
        # Маркеры
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), markers)
        mask = (markers > 1).astype(np.uint8) * 255
        
        return mask