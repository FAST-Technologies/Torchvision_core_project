# segmentation_comparator.py

# Импорт основных библиотек
import numpy as np
from typing import Union, Tuple, Dict, Any, List
from PIL import Image
import cv2
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    jaccard_score, confusion_matrix
)
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import (
    segmentation as skseg,
    filters,
    feature,
    measure,
    morphology
)
from skimage.segmentation import (
    felzenszwalb, slic, quickshift, watershed,
    random_walker, chan_vese, morphological_geodesic_active_contour
)
from skimage.filters import (
    threshold_otsu, threshold_niblack, threshold_sauvola,
    sobel, scharr, prewitt, roberts
)
from skimage.feature import canny
from skimage.color import label2rgb
import pandas as pd
import time
import os


class SegmentationComparator:
    """
    Класс для сравнительного тестирования сегментационных методов.
    Использует готовые реализации из scikit-image и scikit-learn
    для валидации кастомных реализаций.
    """
    
    def __init__(self):
        self.results = {}
        self.metrics_history = {}
        self.reference_methods = {}
        
    def segment_with_sklearn(
        self, 
        image: np.ndarray,
        method: str,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Сегментация с использованием scikit-learn методов.
        
        Args:
            image: Входное изображение (RGB или grayscale)
            method: Метод сегментации
            **kwargs: Параметры метода
        
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Маска и информация о методе
        """
        if method == "kmeans":
            return self._sklearn_kmeans(image, **kwargs)
        elif method == "dbscan":
            return self._sklearn_dbscan(image, **kwargs)
        elif method == "meanshift":
            return self._sklearn_meanshift(image, **kwargs)
        elif method == "gmm":
            return self._sklearn_gmm(image, **kwargs)
        else:
            raise ValueError(f"Неизвестный sklearn метод: {method}")
    
    def segment_with_skimage(
        self,
        image: np.ndarray,
        method: str,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Сегментация с использованием scikit-image методов.
        
        Args:
            image: Входное изображение
            method: Метод сегментации
            **kwargs: Параметры метода
        
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Маска и информация
        """
        method_map = {
            "felzenszwalb": self._skimage_felzenszwalb,
            "slic": self._skimage_slic,
            "quickshift": self._skimage_quickshift,
            "watershed": self._skimage_watershed,
            "random_walker": self._skimage_random_walker,
            "chan_vese": self._skimage_chan_vese,
            "active_contour": self._skimage_active_contour,
            "morphological_snakes": self._skimage_morphological_snakes,
            "threshold_otsu": self._skimage_threshold_otsu,
            "threshold_niblack": self._skimage_threshold_niblack,
            "threshold_sauvola": self._skimage_threshold_sauvola,
            "sobel": self._skimage_sobel,
            "canny": self._skimage_canny
        }
        
        if method not in method_map:
            raise ValueError(f"Неизвестный skimage метод: {method}")
        
        return method_map[method](image, **kwargs)
    
    # ============ SKLEARN МЕТОДЫ ============
    
    def _sklearn_kmeans(
        self,
        image: np.ndarray,
        n_clusters: int = 3,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """KMeans кластеризация с sklearn"""
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3)
        else:
            pixels = image.reshape(-1, 1)
        
        start_time = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        labels = kmeans.fit_predict(pixels)
        exec_time = time.time() - start_time
        
        labels_2d = labels.reshape(h, w)
        
        # Находим самый большой кластер как фон
        unique, counts = np.unique(labels, return_counts=True)
        bg_label = unique[np.argmax(counts)]
        
        mask = (labels_2d != bg_label).astype(np.uint8) * 255
        
        info = {
            'method': 'sklearn_kmeans',
            'parameters': {'n_clusters': n_clusters, **kwargs},
            'execution_time': exec_time,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }
        
        return mask, info
    
    def _sklearn_dbscan(
        self,
        image: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """DBSCAN кластеризация с sklearn"""
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3)
        else:
            pixels = image.reshape(-1, 1)
        
        # Масштабирование для больших изображений
        scale_factor = 0.5 if h * w > 100000 else 1.0
        
        if scale_factor < 1.0:
            small_h, small_w = int(h * scale_factor), int(w * scale_factor)
            small_image = cv2.resize(image, (small_w, small_h))
            pixels = small_image.reshape(-1, 3)
        
        start_time = time.time()
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = dbscan.fit_predict(pixels)
        exec_time = time.time() - start_time
        
        if scale_factor < 1.0:
            labels_2d_small = labels.reshape(small_h, small_w)
            labels_2d = cv2.resize(labels_2d_small.astype(np.float32),
                                  (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(int)
        else:
            labels_2d = labels.reshape(h, w)
        
        # Создаем маску (все кроме шума)
        mask = (labels_2d != -1).astype(np.uint8) * 255
        
        info = {
            'method': 'sklearn_dbscan',
            'parameters': {'eps': eps, 'min_samples': min_samples, **kwargs},
            'execution_time': exec_time,
            'n_clusters': len(np.unique(labels[labels != -1])),
            'n_noise': np.sum(labels == -1)
        }
        
        return mask, info
    
    def _sklearn_meanshift(
        self,
        image: np.ndarray,
        bandwidth: float = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """MeanShift кластеризация с sklearn"""
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3)
        else:
            pixels = image.reshape(-1, 1)
        
        # Масштабирование для производительности
        scale_factor = 0.3 if h * w > 50000 else 1.0
        
        if scale_factor < 1.0:
            small_h, small_w = int(h * scale_factor), int(w * scale_factor)
            small_image = cv2.resize(image, (small_w, small_h))
            pixels = small_image.reshape(-1, 3)
        
        start_time = time.time()
        meanshift = MeanShift(bandwidth=bandwidth, **kwargs)
        labels = meanshift.fit_predict(pixels)
        exec_time = time.time() - start_time
        
        if scale_factor < 1.0:
            labels_2d_small = labels.reshape(small_h, small_w)
            labels_2d = cv2.resize(labels_2d_small.astype(np.float32),
                                  (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(int)
        else:
            labels_2d = labels.reshape(h, w)
        
        # Находим самый большой кластер как фон
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) > 0:
            bg_label = unique[np.argmax(counts)]
            mask = (labels_2d != bg_label).astype(np.uint8) * 255
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        
        info = {
            'method': 'sklearn_meanshift',
            'parameters': {'bandwidth': bandwidth, **kwargs},
            'execution_time': exec_time,
            'n_clusters': len(unique),
            'cluster_centers': meanshift.cluster_centers_
        }
        
        return mask, info
    
    def _sklearn_gmm(
        self,
        image: np.ndarray,
        n_components: int = 3,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Gaussian Mixture Models с sklearn"""
        h, w = image.shape[:2]
        
        if len(image.shape) == 3:
            pixels = image.reshape(-1, 3)
        else:
            pixels = image.reshape(-1, 1)
        
        start_time = time.time()
        gmm = GaussianMixture(n_components=n_components,
                            random_state=42,
                            **kwargs)
        labels = gmm.fit_predict(pixels)
        exec_time = time.time() - start_time
        
        labels_2d = labels.reshape(h, w)
        
        # Находим самый большой кластер как фон
        unique, counts = np.unique(labels, return_counts=True)
        bg_label = unique[np.argmax(counts)]
        
        mask = (labels_2d != bg_label).astype(np.uint8) * 255
        
        info = {
            'method': 'sklearn_gmm',
            'parameters': {'n_components': n_components, **kwargs},
            'execution_time': exec_time,
            'converged': gmm.converged_,
            'lower_bound': gmm.lower_bound_
        }
        
        return mask, info
    
    # ============ SKIMAGE МЕТОДЫ ============
    
    def _skimage_felzenszwalb(
        self,
        image: np.ndarray,
        scale: float = 100,
        sigma: float = 0.8,
        min_size: int = 50,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Алгоритм Felzenszwalb"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        start_time = time.time()
        segments = felzenszwalb(gray,
                               scale=scale,
                               sigma=sigma,
                               min_size=min_size,
                               **kwargs)
        exec_time = time.time() - start_time
        
        # Находим самый большой сегмент
        unique, counts = np.unique(segments, return_counts=True)
        if len(unique) > 0:
            bg_label = unique[np.argmax(counts)]
            mask = (segments != bg_label).astype(np.uint8) * 255
        else:
            mask = np.zeros_like(segments, dtype=np.uint8)
        
        info = {
            'method': 'skimage_felzenszwalb',
            'parameters': {'scale': scale, 'sigma': sigma, 'min_size': min_size, **kwargs},
            'execution_time': exec_time,
            'n_segments': len(unique)
        }
        
        return mask, info
    
    def _skimage_slic(
        self,
        image: np.ndarray,
        n_segments: int = 100,
        compactness: float = 10.0,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """SLIC суперпиксели"""
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        start_time = time.time()
        segments = slic(image_rgb,
                       n_segments=n_segments,
                       compactness=compactness,
                       **kwargs)
        exec_time = time.time() - start_time
        
        # Находим самый большой сегмент
        unique, counts = np.unique(segments, return_counts=True)
        if len(unique) > 0:
            bg_label = unique[np.argmax(counts)]
            mask = (segments != bg_label).astype(np.uint8) * 255
        else:
            mask = np.zeros_like(segments, dtype=np.uint8)
        
        info = {
            'method': 'skimage_slic',
            'parameters': {'n_segments': n_segments, 'compactness': compactness, **kwargs},
            'execution_time': exec_time,
            'n_segments': len(unique)
        }
        
        return mask, info
    
    def _skimage_quickshift(
        self,
        image: np.ndarray,
        kernel_size: float = 3,
        max_dist: float = 6,
        ratio: float = 0.5,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quickshift сегментация"""
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        start_time = time.time()
        segments = quickshift(image_rgb,
                             kernel_size=kernel_size,
                             max_dist=max_dist,
                             ratio=ratio,
                             **kwargs)
        exec_time = time.time() - start_time
        
        # Находим самый большой сегмент
        unique, counts = np.unique(segments, return_counts=True)
        if len(unique) > 0:
            bg_label = unique[np.argmax(counts)]
            mask = (segments != bg_label).astype(np.uint8) * 255
        else:
            mask = np.zeros_like(segments, dtype=np.uint8)
        
        info = {
            'method': 'skimage_quickshift',
            'parameters': {'kernel_size': kernel_size, 'max_dist': max_dist, 'ratio': ratio, **kwargs},
            'execution_time': exec_time,
            'n_segments': len(unique)
        }
        
        return mask, info
    
    def _skimage_watershed(
        self,
        image: np.ndarray,
        markers: int = 10,
        compactness: float = 0.001,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Watershed сегментация"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Вычисляем градиент
        gradient = filters.sobel(gray)
        
        start_time = time.time()
        
        # Автоматические маркеры
        from skimage.feature import peak_local_max
        from scipy import ndimage as ndi
        
        # Расстояние до фона
        distance = ndi.distance_transform_edt(gray > filters.threshold_otsu(gray))
        
        # Находим локальные максимумы
        local_max = peak_local_max(distance, 
                                  min_distance=20, 
                                  labels=gray > filters.threshold_otsu(gray))
        
        mask_peaks = np.zeros(distance.shape, dtype=bool)
        mask_peaks[tuple(local_max.T)] = True
        
        markers_ws, _ = ndi.label(mask_peaks)
        
        segments = watershed(gradient, markers_ws)
        exec_time = time.time() - start_time
        
        # Создаем маску
        mask = (segments > 1).astype(np.uint8) * 255
        
        info = {
            'method': 'skimage_watershed',
            'parameters': {'markers': markers, 'compactness': compactness, **kwargs},
            'execution_time': exec_time,
            'n_segments': len(np.unique(segments))
        }
        
        return mask, info
    
    def _skimage_random_walker(
        self,
        image: np.ndarray,
        markers: np.ndarray = None,
        beta: float = 130,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Random Walker сегментация"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Нормализуем
        gray_norm = gray.astype(float) / 255.0
        
        # Создаем маркеры если не предоставлены
        if markers is None:
            h, w = gray.shape
            markers = np.zeros_like(gray, dtype=np.uint8)
            markers[h//4:3*h//4, w//4:3*w//4] = 2  # Предполагаемый объект
            markers[0:h//8, 0:w//8] = 1  # Фон
            markers[7*h//8:, 7*w//8:] = 1  # Фон
        
        start_time = time.time()
        labels = random_walker(gray_norm, markers, beta=beta, **kwargs)
        exec_time = time.time() - start_time
        
        # Создаем маску
        mask = (labels == 2).astype(np.uint8) * 255
        
        info = {
            'method': 'skimage_random_walker',
            'parameters': {'beta': beta, **kwargs},
            'execution_time': exec_time
        }
        
        return mask, info
    
    def _skimage_chan_vese(
        self,
        image: np.ndarray,
        mu: float = 0.25,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 100,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Chan-Vese активные контуры"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Нормализуем
        gray_norm = gray.astype(float) / 255.0
        
        start_time = time.time()
        segmentation = chan_vese(gray_norm,
                                mu=mu,
                                lambda1=lambda1,
                                lambda2=lambda2,
                                tol=tol,
                                max_num_iter=max_iter,
                                **kwargs)
        exec_time = time.time() - start_time
        
        mask = segmentation.astype(np.uint8) * 255
        
        info = {
            'method': 'skimage_chan_vese',
            'parameters': {
                'mu': mu, 'lambda1': lambda1, 'lambda2': lambda2,
                'tol': tol, 'max_iter': max_iter, **kwargs
            },
            'execution_time': exec_time,
            'converged': exec_time < max_iter * 0.1  # Эвристика сходимости
        }
        
        return mask, info
    
    def _skimage_active_contour(
        self,
        image: np.ndarray,
        alpha: float = 0.01,
        beta: float = 0.1,
        gamma: float = 0.001,
        w_edge: float = 1,
        w_line: float = 0,
        max_iter: int = 1000,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Активные контуры"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray_norm = gray.astype(float) / 255.0
        
        h, w = gray_norm.shape
        
        # Инициализируем контур (окружность)
        s = np.linspace(0, 2*np.pi, 100)
        r = h/2 + h/4 * np.sin(s)
        c = w/2 + w/4 * np.cos(s)
        init = np.array([r, c]).T
        
        start_time = time.time()
        try:
            snake = skseg.active_contour(gray_norm,
                                        init,
                                        alpha=alpha,
                                        beta=beta,
                                        gamma=gamma,
                                        w_edge=w_edge,
                                        w_line=w_line,
                                        max_num_iter=max_iter,
                                        **kwargs)
            
            # Создаем маску из контура
            mask = np.zeros_like(gray, dtype=np.uint8)
            from skimage.draw import polygon
            rr, cc = polygon(snake[:, 0], snake[:, 1], gray.shape)
            mask[rr, cc] = 255
            
            success = True
        except Exception as e:
            warnings.warn(f"Active contour failed: {e}")
            mask = np.zeros_like(gray, dtype=np.uint8)
            success = False
        
        exec_time = time.time() - start_time
        
        info = {
            'method': 'skimage_active_contour',
            'parameters': {
                'alpha': alpha, 'beta': beta, 'gamma': gamma,
                'w_edge': w_edge, 'w_line': w_line, 'max_iter': max_iter, **kwargs
            },
            'execution_time': exec_time,
            'success': success
        }
        
        return mask, info
    
    def _skimage_morphological_snakes(
        self,
        image: np.ndarray,
        iterations: int = 100,
        smoothing: int = 1,
        threshold: float = 0.5,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Морфологические змеи"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray_norm = gray.astype(float) / 255.0
        
        h, w = gray_norm.shape
        
        # Инициализируем контур (окружность)
        init_ls = np.zeros((h, w))
        center_y, center_x = h // 2, w // 2
        radius = min(center_x, center_y) // 2
        y, x = np.ogrid[:h, :w]
        init_ls[(x - center_x)**2 + (y - center_y)**2 <= radius**2] = 1
        
        start_time = time.time()
        try:
            ls = morphological_geodesic_active_contour(gray_norm,
                                                      iterations,
                                                      init_level_set=init_ls,
                                                      smoothing=smoothing,
                                                      threshold=threshold,
                                                      **kwargs)
            mask = (ls > 0.5).astype(np.uint8) * 255
            success = True
        except Exception as e:
            warnings.warn(f"Morphological snakes failed: {e}")
            mask = np.zeros_like(gray, dtype=np.uint8)
            success = False
        
        exec_time = time.time() - start_time
        
        info = {
            'method': 'skimage_morphological_snakes',
            'parameters': {
                'iterations': iterations,
                'smoothing': smoothing,
                'threshold': threshold,
                **kwargs
            },
            'execution_time': exec_time,
            'success': success
        }
        
        return mask, info
    
    def _skimage_threshold_otsu(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Порог Оцу"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        start_time = time.time()
        thresh = threshold_otsu(gray)
        mask = (gray > thresh).astype(np.uint8) * 255
        exec_time = time.time() - start_time
        
        info = {
            'method': 'skimage_threshold_otsu',
            'parameters': kwargs,
            'execution_time': exec_time,
            'threshold': thresh
        }
        
        return mask, info
    
    def _skimage_threshold_niblack(
        self,
        image: np.ndarray,
        window_size: int = 15,
        k: float = 0.2,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Порог Ниблака"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        start_time = time.time()
        thresh = threshold_niblack(gray,
                                  window_size=window_size,
                                  k=k,
                                  **kwargs)
        mask = (gray > thresh).astype(np.uint8) * 255
        exec_time = time.time() - start_time
        
        info = {
            'method': 'skimage_threshold_niblack',
            'parameters': {'window_size': window_size, 'k': k, **kwargs},
            'execution_time': exec_time
        }
        
        return mask, info
    
    def _skimage_threshold_sauvola(
        self,
        image: np.ndarray,
        window_size: int = 15,
        k: float = 0.2,
        r: float = 128,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Порог Сауволы"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        start_time = time.time()
        thresh = threshold_sauvola(gray,
                                  window_size=window_size,
                                  k=k,
                                  r=r,
                                  **kwargs)
        mask = (gray > thresh).astype(np.uint8) * 255
        exec_time = time.time() - start_time
        
        info = {
            'method': 'skimage_threshold_sauvola',
            'parameters': {'window_size': window_size, 'k': k, 'r': r, **kwargs},
            'execution_time': exec_time
        }
        
        return mask, info
    
    def _skimage_sobel(
        self,
        image: np.ndarray,
        threshold: float = 0.1,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Детектор границ Собеля"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray_norm = gray.astype(float) / 255.0
        
        start_time = time.time()
        edges = sobel(gray_norm)
        mask = (edges > threshold).astype(np.uint8) * 255
        exec_time = time.time() - start_time
        
        info = {
            'method': 'skimage_sobel',
            'parameters': {'threshold': threshold, **kwargs},
            'execution_time': exec_time,
            'edge_max': edges.max()
        }
        
        return mask, info
    
    def _skimage_canny(
        self,
        image: np.ndarray,
        sigma: float = 1.0,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Детектор границ Кэнни"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray_norm = gray.astype(float) / 255.0
        
        start_time = time.time()
        edges = canny(gray_norm,
                     sigma=sigma,
                     low_threshold=low_threshold,
                     high_threshold=high_threshold,
                     **kwargs)
        mask = edges.astype(np.uint8) * 255
        exec_time = time.time() - start_time
        
        info = {
            'method': 'skimage_canny',
            'parameters': {
                'sigma': sigma,
                'low_threshold': low_threshold,
                'high_threshold': high_threshold,
                **kwargs
            },
            'execution_time': exec_time
        }
        
        return mask, info
    
    # ============ МЕТРИКИ КАЧЕСТВА ============
    
    def compute_metrics(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray,
        method1_name: str = "Method1",
        method2_name: str = "Method2"
    ) -> Dict[str, float]:
        """
        Вычисляет метрики сходства между двумя масками.
        
        Args:
            mask1: Первая маска (0-255)
            mask2: Вторая маска (0-255)
            method1_name: Имя первого метода
            method2_name: Имя второго метода
        
        Returns:
            Dict[str, float]: Словарь с метриками
        """
        # Бинаризируем маски
        mask1_bin = (mask1 > 127).astype(np.uint8).flatten()
        mask2_bin = (mask2 > 127).astype(np.uint8).flatten()
        
        metrics = {}
        
        # Основные метрики
        try:
            metrics['accuracy'] = accuracy_score(mask1_bin, mask2_bin)
            metrics['precision'] = precision_score(mask1_bin, mask2_bin, zero_division=0)
            metrics['recall'] = recall_score(mask1_bin, mask2_bin, zero_division=0)
            metrics['f1_score'] = f1_score(mask1_bin, mask2_bin, zero_division=0)
            metrics['jaccard'] = jaccard_score(mask1_bin, mask2_bin, zero_division=0)
        except Exception as e:
            warnings.warn(f"Error computing metrics: {e}")
            metrics.update({
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'jaccard': 0.0
            })
        
        # Дополнительные метрики
        intersection = np.sum(mask1_bin & mask2_bin)
        union = np.sum(mask1_bin | mask2_bin)
        total_pixels = len(mask1_bin)
        
        metrics['dice_coefficient'] = (2 * intersection) / (np.sum(mask1_bin) + np.sum(mask2_bin) + 1e-8)
        metrics['intersection_over_union'] = intersection / (union + 1e-8)
        metrics['pixel_agreement'] = np.sum(mask1_bin == mask2_bin) / total_pixels
        
        # Площади масок
        metrics[f'{method1_name}_area'] = np.sum(mask1_bin)
        metrics[f'{method2_name}_area'] = np.sum(mask2_bin)
        metrics['area_difference'] = abs(metrics[f'{method1_name}_area'] - metrics[f'{method2_name}_area'])
        metrics['area_ratio'] = min(metrics[f'{method1_name}_area'], metrics[f'{method2_name}_area']) / \
                               (max(metrics[f'{method1_name}_area'], metrics[f'{method2_name}_area']) + 1e-8)
        
        # Матрица ошибок
        tn, fp, fn, tp = confusion_matrix(mask1_bin, mask2_bin, labels=[0, 1]).ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
        
        return metrics
    
    def compare_methods(
        self,
        image: np.ndarray,
        method1: str,
        method2: str,
        method1_type: str = "skimage",  # "skimage" или "sklearn"
        method2_type: str = "sklearn",
        method1_params: Dict[str, Any] = None,
        method2_params: Dict[str, Any] = None,
        save_comparison: bool = True,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Сравнивает две реализации методов сегментации.
        
        Args:
            image: Входное изображение
            method1: Имя первого метода
            method2: Имя второго метода
            method1_type: Тип первого метода
            method2_type: Тип второго метода
            method1_params: Параметры первого метода
            method2_params: Параметры второго метода
            save_comparison: Сохранять ли визуализацию
            output_path: Путь для сохранения
        
        Returns:
            Dict[str, Any]: Результаты сравнения
        """
        method1_params = method1_params or {}
        method2_params = method2_params or {}
        
        # Сегментация первым методом
        if method1_type == "skimage":
            mask1, info1 = self.segment_with_skimage(image, method1, **method1_params)
        elif method1_type == "sklearn":
            mask1, info1 = self.segment_with_sklearn(image, method1, **method1_params)
        else:
            raise ValueError(f"Неизвестный тип метода: {method1_type}")
        
        # Сегментация вторым методом
        if method2_type == "skimage":
            mask2, info2 = self.segment_with_skimage(image, method2, **method2_params)
        elif method2_type == "sklearn":
            mask2, info2 = self.segment_with_sklearn(image, method2, **method2_params)
        else:
            raise ValueError(f"Неизвестный тип метода: {method2_type}")
        
        # Вычисляем метрики
        metrics = self.compute_metrics(mask1, mask2, f"{method1_type}_{method1}", f"{method2_type}_{method2}")
        
        # Сохраняем результаты
        result_key = f"{method1_type}_{method1}_vs_{method2_type}_{method2}"
        self.results[result_key] = {
            'mask1': mask1,
            'mask2': mask2,
            'info1': info1,
            'info2': info2,
            'metrics': metrics
        }
        
        # Визуализация
        if save_comparison:
            self.visualize_comparison(
                image, mask1, mask2,
                info1, info2, metrics,
                method1_name=f"{method1_type} {method1}",
                method2_name=f"{method2_type} {method2}",
                output_path=output_path
            )
        
        return self.results[result_key]
    
    def visualize_comparison(
        self,
        image: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
        info1: Dict[str, Any],
        info2: Dict[str, Any],
        metrics: Dict[str, float],
        method1_name: str = "Method 1",
        method2_name: str = "Method 2",
        output_path: str = None
    ) -> None:
        """
        Визуализирует сравнение двух методов.
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Оригинальное изображение
        if len(image.shape) == 2:
            axes[0, 0].imshow(image, cmap='gray')
        else:
            axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Маска 1
        axes[0, 1].imshow(mask1, cmap='gray')
        time1 = info1.get('execution_time', 0)
        axes[0, 1].set_title(f"{method1_name}\nTime: {time1:.3f}s")
        axes[0, 1].axis('off')
        
        # Маска 2
        axes[0, 2].imshow(mask2, cmap='gray')
        time2 = info2.get('execution_time', 0)
        axes[0, 2].set_title(f"{method2_name}\nTime: {time2:.3f}s")
        axes[0, 2].axis('off')
        
        # Разность масок
        diff = np.abs(mask1.astype(float) - mask2.astype(float))
        axes[0, 3].imshow(diff, cmap='hot')
        axes[0, 3].set_title("Difference")
        axes[0, 3].axis('off')
        
        # Наложение масок на изображение
        if len(image.shape) == 2:
            overlay1 = np.stack([image] * 3, axis=-1)
            overlay2 = np.stack([image] * 3, axis=-1)
        else:
            overlay1 = image.copy()
            overlay2 = image.copy()
        
        overlay1[mask1 > 0] = [255, 0, 0]  # Красный
        overlay2[mask2 > 0] = [0, 255, 0]  # Зеленый
        
        axes[1, 0].imshow(overlay1)
        axes[1, 0].set_title(f"{method1_name} Overlay")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(overlay2)
        axes[1, 1].set_title(f"{method2_name} Overlay")
        axes[1, 1].axis('off')
        
        # Комбинированное наложение
        combined = image.copy() if len(image.shape) == 3 else np.stack([image] * 3, axis=-1)
        combined[mask1 > 0] = [255, 0, 0]  # Красный для метода 1
        combined[mask2 > 0] = [0, 255, 0]  # Зеленый для метода 2
        
        # Желтый для пересечения
        intersection = (mask1 > 0) & (mask2 > 0)
        combined[intersection] = [255, 255, 0]
        
        axes[1, 2].imshow(combined)
        axes[1, 2].set_title("Combined Overlay\n(Red: Method1, Green: Method2, Yellow: Both)")
        axes[1, 2].axis('off')
        
        # Текстовые метрики
        axes[1, 3].axis('off')
        text_str = (
            f"Metrics Comparison:\n"
            f"Accuracy: {metrics.get('accuracy', 0):.3f}\n"
            f"Precision: {metrics.get('precision', 0):.3f}\n"
            f"Recall: {metrics.get('recall', 0):.3f}\n"
            f"F1-Score: {metrics.get('f1_score', 0):.3f}\n"
            f"Jaccard: {metrics.get('jaccard', 0):.3f}\n"
            f"Dice: {metrics.get('dice_coefficient', 0):.3f}\n"
            f"IoU: {metrics.get('intersection_over_union', 0):.3f}\n"
            f"Pixel Agreement: {metrics.get('pixel_agreement', 0):.3f}"
        )
        
        axes[1, 3].text(0.1, 0.5, text_str, fontsize=10,
                       verticalalignment='center',
                       transform=axes[1, 3].transAxes)
        
        plt.suptitle(f"Segmentation Methods Comparison: {method1_name} vs {method2_name}", fontsize=14)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Визуализация сохранена: {output_path}")
        
        plt.show()
    
    def batch_comparison(
        self,
        image: np.ndarray,
        methods_config: List[Dict[str, Any]],
        reference_method: str = "skimage_felzenszwalb",
        save_results: bool = True,
        output_dir: str = "comparison_results"
    ) -> pd.DataFrame:
        """
        Пакетное сравнение нескольких методов с референсным.
        
        Args:
            image: Входное изображение
            methods_config: Конфигурация методов для сравнения
            reference_method: Референсный метод
            save_results: Сохранять ли результаты
            output_dir: Директория для сохранения
        
        Returns:
            pd.DataFrame: DataFrame с результатами сравнения
        """
        import os
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        # Получаем референсную маску
        ref_mask, ref_info = self.segment_with_skimage(image, reference_method)
        
        comparison_results = []
        
        for config in methods_config:
            method_name = config.get('name')
            method_type = config.get('type', 'skimage')
            method_params = config.get('params', {})
            
            try:
                if method_type == "skimage":
                    test_mask, test_info = self.segment_with_skimage(
                        image, method_name, **method_params)
                elif method_type == "sklearn":
                    test_mask, test_info = self.segment_with_sklearn(
                        image, method_name, **method_params)
                else:
                    continue
                
                # Вычисляем метрики
                metrics = self.compute_metrics(ref_mask, test_mask, 
                                             f"Reference_{reference_method}",
                                             f"Test_{method_name}")
                
                # Сохраняем результаты
                result = {
                    'method': method_name,
                    'type': method_type,
                    **metrics,
                    'test_time': test_info.get('execution_time', 0),
                    'ref_time': ref_info.get('execution_time', 0),
                    'parameters': str(method_params)
                }
                
                comparison_results.append(result)
                
                # Сохраняем визуализацию
                if save_results:
                    output_path = os.path.join(output_dir, f"comparison_{method_name}.jpg")
                    self.visualize_comparison(
                        image, ref_mask, test_mask,
                        ref_info, test_info, metrics,
                        method1_name=f"Reference: {reference_method}",
                        method2_name=f"Test: {method_name}",
                        output_path=output_path
                    )
                
                print(f"✅ Сравнение {method_name}: F1={metrics.get('f1_score', 0):.3f}")
                
            except Exception as e:
                print(f"❌ Ошибка при тестировании {method_name}: {e}")
                continue
        
        # Создаем DataFrame
        df = pd.DataFrame(comparison_results)
        
        if save_results and not df.empty:
            csv_path = os.path.join(output_dir, "comparison_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"📊 Результаты сохранены в CSV: {csv_path}")
            
            # Создаем сводную визуализацию
            self._create_summary_visualization(df, output_dir)
        
        return df
    
    def _create_summary_visualization(
        self,
        df: pd.DataFrame,
        output_dir: str
    ) -> None:
        """Создает сводную визуализацию результатов сравнения."""
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # График 1: Метрики качества
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'jaccard']
        metrics_data = df[metrics_to_plot].mean()
        
        axes[0, 0].bar(range(len(metrics_data)), metrics_data.values)
        axes[0, 0].set_xticks(range(len(metrics_data)))
        axes[0, 0].set_xticklabels(metrics_data.index, rotation=45)
        axes[0, 0].set_title('Average Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        # График 2: Время выполнения
        if 'test_time' in df.columns and 'ref_time' in df.columns:
            methods = df['method'].tolist()
            test_times = df['test_time'].tolist()
            ref_time = df['ref_time'].iloc[0] if len(df) > 0 else 0
            
            x = np.arange(len(methods))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, test_times, width, label='Test Methods')
            axes[0, 1].bar(x[-1] + width/2, ref_time, width, label='Reference', alpha=0.7)
            axes[0, 1].set_xlabel('Methods')
            axes[0, 1].set_ylabel('Execution Time (s)')
            axes[0, 1].set_title('Execution Time Comparison')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(methods, rotation=45)
            axes[0, 1].legend()
        
        # График 3: Площадь масок
        area_cols = [col for col in df.columns if 'area' in col.lower() and 'difference' not in col.lower()]
        if len(area_cols) >= 2:
            area_data = df[area_cols].mean()
            axes[1, 0].bar(range(len(area_data)), area_data.values)
            axes[1, 0].set_xticks(range(len(area_data)))
            axes[1, 0].set_xticklabels(area_data.index, rotation=45)
            axes[1, 0].set_title('Average Mask Areas')
            axes[1, 0].set_ylabel('Pixels')
        
        # График 4: Корреляционная матрица метрик
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('Correlation Matrix')
        axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
        axes[1, 1].set_yticklabels(corr_matrix.columns, fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.suptitle('Segmentation Methods Comparison Summary', fontsize=16)
        plt.tight_layout()
        
        summary_path = os.path.join(output_dir, "comparison_summary.jpg")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Сводная визуализация сохранена: {summary_path}")


# Пример использования
def example_usage() -> Tuple[SegmentationComparator, pd.DataFrame]:
    """Пример использования класса для сравнения методов"""
    import cv2
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Загрузка тестового изображения
    url = "https://i.pinimg.com/736x/f7/5a/f2/f75af26820b50c24600f50f3998eb02f.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_np = np.array(img)
    
    # Создаем компаратор
    comparator = SegmentationComparator()
    
    # 1. Простое сравнение двух методов
    print("1. Сравнение двух методов (skimage vs sklearn)...")
    result = comparator.compare_methods(
        img_np,
        method1="kmeans",
        method2="felzenszwalb",
        method1_type="sklearn",
        method2_type="skimage",
        method1_params={"n_clusters": 3},
        method2_params={"scale": 100, "sigma": 0.8},
        output_path="kmeans_vs_felzenszwalb.jpg"
    )
    
    # 2. Пакетное сравнение нескольких методов
    print("\n2. Пакетное сравнение методов...")
    methods_config = [
        {"name": "kmeans", "type": "sklearn", "params": {"n_clusters": 3}},
        {"name": "dbscan", "type": "sklearn", "params": {"eps": 0.5, "min_samples": 5}},
        {"name": "gmm", "type": "sklearn", "params": {"n_components": 3}},
        {"name": "slic", "type": "skimage", "params": {"n_segments": 100}},
        {"name": "quickshift", "type": "skimage", "params": {"kernel_size": 3}},
        {"name": "watershed", "type": "skimage", "params": {}},
        {"name": "random_walker", "type": "skimage", "params": {"beta": 130}},
        {"name": "chan_vese", "type": "skimage", "params": {"max_iter": 100}},
        {"name": "threshold_otsu", "type": "skimage", "params": {}},
        {"name": "canny", "type": "skimage", "params": {"sigma": 1.0}}
    ]
    
    
    df_results = comparator.batch_comparison(
        img_np,
        methods_config=methods_config,
        reference_method="felzenszwalb",
        output_dir="batch_comparison_results"
    )
    
    # Выводим результаты
    if not df_results.empty:
        print("\nРезультаты сравнения:")
        print(df_results[['method', 'f1_score', 'accuracy', 'test_time']].sort_values('f1_score', ascending=False))
    
    return comparator, df_results


# Интеграция с вашими классами
def compare_with_custom_method(
    custom_segmenter, 
    comparator: SegmentationComparator,
    image: np.ndarray,
    custom_method_name: str = "Custom",
    reference_method: str = "felzenszwalb",
    reference_type: str = "skimage",
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Сравнение кастомного сегментатора с референсным методом.
    
    Args:
        custom_segmenter: Ваш кастомный сегментатор
        comparator: Объект SegmentationComparator
        image: Входное изображение
        custom_method_name: Имя кастомного метода
        reference_method: Референсный метод
        reference_type: Тип референсного метода
        save_results: Сохранять ли результаты
    
    Returns:
        Dict[str, Any]: Результаты сравнения
    """
    # Получаем маску от кастомного сегментатора
    if hasattr(custom_segmenter, 'segment'):
        custom_mask = custom_segmenter.segment(image)
    elif hasattr(custom_segmenter, 'segment_with_mask'):
        _, custom_mask = custom_segmenter.segment_with_mask(image)
    else:
        raise ValueError("Кастомный сегментатор должен иметь метод segment или segment_with_mask")
    
    # Получаем референсную маску
    if reference_type == "skimage":
        ref_mask, ref_info = comparator.segment_with_skimage(image, reference_method)
    elif reference_type == "sklearn":
        ref_mask, ref_info = comparator.segment_with_sklearn(image, reference_method)
    else:
        raise ValueError(f"Неизвестный тип референсного метода: {reference_type}")
    
    # Вычисляем метрики
    metrics = comparator.compute_metrics(ref_mask, custom_mask, 
                                       f"Reference_{reference_method}",
                                       f"Custom_{custom_method_name}")
    
    # Создаем информацию о кастомном методе
    custom_info = {
        'method': f'custom_{custom_method_name}',
        'parameters': {},
        'execution_time': 0.0  # Можно замерять время если нужно
    }
    
    # Визуализируем сравнение
    if save_results:
        output_path = f"custom_vs_{reference_method}.jpg"
        comparator.visualize_comparison(
            image, ref_mask, custom_mask,
            ref_info, custom_info, metrics,
            method1_name=f"Reference: {reference_method}",
            method2_name=f"Custom: {custom_method_name}",
            output_path=output_path
        )
    
    # Сохраняем результаты
    result = {
        'custom_mask': custom_mask,
        'reference_mask': ref_mask,
        'custom_info': custom_info,
        'reference_info': ref_info,
        'metrics': metrics
    }
    
    # Выводим метрики
    print(f"\nСравнение {custom_method_name} с {reference_method}:")
    print(f"  F1-Score: {metrics.get('f1_score', 0):.3f}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
    print(f"  Precision: {metrics.get('precision', 0):.3f}")
    print(f"  Recall: {metrics.get('recall', 0):.3f}")
    print(f"  Jaccard: {metrics.get('jaccard', 0):.3f}")
    
    return result


if __name__ == "__main__":
    # Запуск примера
    comparator, results = example_usage()
    
    print("\n✅ Тестирование завершено!")
    print("Результаты сохранены в папках:")
    print("  - batch_comparison_results/ (пакетное сравнение)")
    print("  - kmeans_vs_felzenszwalb.jpg (простое сравнение)")