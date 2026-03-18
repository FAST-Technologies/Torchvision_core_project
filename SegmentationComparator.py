# SegmentationComparator.py

# Импорт основных библиотек

import os
import time
import warnings
from PIL import Image
import requests
from io import BytesIO
from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)
from datetime import datetime
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy import ndimage as ndi

import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    jaccard_score, confusion_matrix
)
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from skimage import (
    segmentation as skseg,
    filters,
    feature,
    measure,
    morphology
)
from skimage.color import label2rgb
from skimage.draw import polygon
from skimage.filters import (
    threshold_otsu, threshold_niblack, threshold_sauvola,
    sobel, scharr, prewitt, roberts
)
from skimage.feature import canny, peak_local_max
from skimage.segmentation import (
    felzenszwalb, slic, quickshift, watershed,
    random_walker, chan_vese, morphological_geodesic_active_contour
)

class SegmentationComparator:
    """
    Класс для сравнительного тестирования сегментационных методов.
    Использует готовые реализации из scikit-image и scikit-learn
    для валидации кастомных реализаций.
    """
    
    def __init__(self):
        self.results = {}
        
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
        
        labels_2d = labels.reshape(h, w)
        
        # Находим самый большой кластер как фон
        unique, counts = np.unique(labels, return_counts=True)
        bg_label = unique[np.argmax(counts)]
        
        mask = (labels_2d != bg_label).astype(np.uint8) * 255
        
        exec_time = time.time() - start_time
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
        output_dir: str = "./data/comparison_results"
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

    def matrix_comparison(
        self,
        image: np.ndarray,
        methods_config: List[Dict[str, Any]],
        reference_method: Optional[str] = None,
        comparison_type: str = "all_vs_all",  # "all_vs_all", "all_vs_ref", "pairwise"
        save_results: bool = True,
        output_dir: str = "./data/matrix_comparison_results"
    ) -> Dict[str, Any]:
        """
        Матричное сравнение всех методов между собой.
        
        Args:
            image: Входное изображение
            methods_config: Конфигурация методов
            reference_method: Референсный метод (если None - сравнение всех со всеми)
            comparison_type: Тип сравнения
                - "all_vs_all": все методы сравниваются со всеми (N x N матрица)
                - "all_vs_ref": все методы сравниваются с референсным
                - "pairwise": сравнение всех возможных пар (без дубликатов)
            save_results: Сохранять ли результаты
            output_dir: Директория для сохранения
        
        Returns:
            Dict[str, Any]: Результаты сравнения
        """
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir: str = os.path.join(output_dir, f"comparison_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Генерируем имена методов для удобства
        method_names = []
        method_objects = {}
        
        for config in methods_config:
            method_name = config.get('name')
            method_type = config.get('type', 'skimage')
            method_params = config.get('params', {})
            
            full_name: str = f"{method_type}_{method_name}"
            method_names.append(full_name)
            method_objects[full_name] = {
                'type': method_type,
                'name': method_name,
                'params': method_params
            }
        
        # Выполняем сегментацию всеми методами
        print(f"Выполняем сегментацию {len(method_names)} методами...")
        masks = {}
        execution_times = {}
        method_infos = {}
        
        for full_name in method_names:
            config = method_objects[full_name]
            
            try:
                if config['type'] == "skimage":
                    mask, info = self.segment_with_skimage(
                        image, config['name'], **config['params'])
                elif config['type'] == "sklearn":
                    mask, info = self.segment_with_sklearn(
                        image, config['name'], **config['params'])
                else:
                    continue
                
                masks[full_name] = mask
                execution_times[full_name] = info.get('execution_time', 0)
                method_infos[full_name] = info
                
                print(f"  ✅ {full_name}: {execution_times[full_name]:.3f}s")
                
            except Exception as e:
                print(f"  ❌ {full_name}: {e}")
                # Создаем пустую маску для методов с ошибкой
                if len(image.shape) == 3:
                    h, w = image.shape[:2]
                else:
                    h, w = image.shape
                masks[full_name] = np.zeros((h, w), dtype=np.uint8)
                execution_times[full_name] = 0
                method_infos[full_name] = {'error': str(e)}
        
        # Выбираем стратегию сравнения
        if comparison_type == "all_vs_ref" and reference_method:
            # Все методы сравниваем с референсным
            comparison_pairs = [(reference_method, other) for other in method_names 
                              if other != reference_method]
            ref_name = reference_method
        elif comparison_type == "pairwise":
            # Все возможные пары без дубликатов
            comparison_pairs = list(itertools.combinations(method_names, 2))
            ref_name = None
        else:  # "all_vs_all"
            # Полная матрица N x N (включая сравнение с самим собой)
            comparison_pairs = [(m1, m2) for m1 in method_names 
                              for m2 in method_names]
            ref_name = None
        
        # Выполняем сравнения
        print(f"\nВыполняем сравнение {len(comparison_pairs)} пар...")
        comparison_results = []
        
        for i, (method1, method2) in enumerate(comparison_pairs):
            if method1 not in masks or method2 not in masks:
                continue
            
            mask1 = masks[method1]
            mask2 = masks[method2]
            
            try:
                # Вычисляем метрики
                metrics = self.compute_metrics(mask1, mask2, method1, method2)
                
                result = {
                    'method1': method1,
                    'method2': method2,
                    **metrics,
                    'time1': execution_times.get(method1, 0),
                    'time2': execution_times.get(method2, 0),
                    'time_diff': abs(execution_times.get(method1, 0) - 
                                   execution_times.get(method2, 0))
                }
                
                comparison_results.append(result)
                
                # Прогресс
                if (i + 1) % 10 == 0:
                    print(f"  Обработано {i + 1}/{len(comparison_pairs)} пар...")
                    
            except Exception as e:
                print(f"  Ошибка сравнения {method1} vs {method2}: {e}")
        
        # Создаем DataFrame
        df_comparisons = pd.DataFrame(comparison_results)
        
        if save_results:
            # Сохраняем все маски
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            
            for name, mask in masks.items():
                mask_path = os.path.join(masks_dir, f"{name}_mask.png")
                plt.imsave(mask_path, mask, cmap='gray')
            
            # Сохраняем все изображения
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Оригинал
            if len(image.shape) == 2:
                plt.imsave(os.path.join(images_dir, "original.png"), 
                          image, cmap='gray')
            else:
                plt.imsave(os.path.join(images_dir, "original.png"), image)
            
            # Наложения
            for name, mask in masks.items():
                if len(image.shape) == 2:
                    overlay = np.stack([image] * 3, axis=-1)
                else:
                    overlay = image.copy()
                
                overlay[mask > 127] = [255, 0, 0]  # Красный
                overlay_path = os.path.join(images_dir, f"{name}_overlay.png")
                plt.imsave(overlay_path, overlay)
            
            # Сохраняем результаты
            self._save_matrix_results(df_comparisons, masks, method_infos, 
                                     output_dir, comparison_type, ref_name)
        
        return {
            'df_comparisons': df_comparisons,
            'masks': masks,
            'execution_times': execution_times,
            'method_infos': method_infos
        }
    
    def _save_matrix_results(
        self,
        df_comparisons: pd.DataFrame,
        masks: Dict[str, np.ndarray],
        method_infos: Dict[str, Any],
        output_dir: str,
        comparison_type: str,
        reference_method: Optional[str] = None
    ):
        """Сохраняет результаты матричного сравнения."""
        
        # 1. Сохраняем DataFrame
        csv_path = os.path.join(output_dir, "comparisons.csv")
        df_comparisons.to_csv(csv_path, index=False)
        print(f"📊 CSV с результатами: {csv_path}")
        
        # 2. Сводная таблица метрик
        summary_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'jaccard']
        
        if comparison_type == "all_vs_ref" and reference_method:
            # Средние метрики по сравнению с референсом
            ref_comparisons = df_comparisons[df_comparisons['method1'] == reference_method]
            if not ref_comparisons.empty:
                summary_df = ref_comparisons[['method2'] + summary_metrics].copy()
                summary_df = summary_df.rename(columns={'method2': 'method'})
                summary_df = summary_df.sort_values('f1_score', ascending=False)
                
                summary_path = os.path.join(output_dir, "summary_vs_ref.csv")
                summary_df.to_csv(summary_path, index=False)
                
                print(f"📋 Сводная таблица (vs {reference_method}): {summary_path}")
        
        # 3. Матрицы сравнения
        methods = sorted(list(masks.keys()))
        n_methods = len(methods)
        
        # Создаем матрицы для каждой метрики
        for metric in ['f1_score', 'accuracy', 'jaccard']:
            if metric not in df_comparisons.columns:
                continue
            
            # Создаем матрицу N x N
            matrix = np.zeros((n_methods, n_methods))
            
            for i, m1 in enumerate(methods):
                for j, m2 in enumerate(methods):
                    if i == j:
                        matrix[i, j] = 1.0  # Само с собой - идеальное совпадение
                    else:
                        # Ищем сравнение в DataFrame
                        mask = ((df_comparisons['method1'] == m1) & 
                               (df_comparisons['method2'] == m2)) | \
                               ((df_comparisons['method1'] == m2) & 
                               (df_comparisons['method2'] == m1))
                        
                        if mask.any():
                            matrix[i, j] = df_comparisons.loc[mask, metric].values[0]
                        else:
                            matrix[i, j] = np.nan
            
            # Визуализируем матрицу
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Сокращаем имена методов для подписей
            short_names = [name[:15] + "..." if len(name) > 15 else name 
                          for name in methods]
            
            im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_xticks(np.arange(n_methods))
            ax.set_yticks(np.arange(n_methods))
            ax.set_xticklabels(short_names, rotation=45, ha='right')
            ax.set_yticklabels(short_names)
            
            # Добавляем значения в ячейки
            for i in range(n_methods):
                for j in range(n_methods):
                    if not np.isnan(matrix[i, j]):
                        text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                                     ha="center", va="center", 
                                     color="black" if matrix[i, j] < 0.7 else "white",
                                     fontsize=8)
            
            ax.set_title(f"Матрица сравнения: {metric.upper()}", fontsize=14)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            
            matrix_path = os.path.join(output_dir, f"{metric}_matrix.png")
            plt.savefig(matrix_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Матрица {metric}: {matrix_path}")
        
        # 4. Визуализация всех масок
        self._visualize_all_masks(masks, output_dir)
        
        # 5. Создаем HTML отчет
        self._create_html_report(df_comparisons, masks, method_infos, 
                                output_dir, comparison_type, reference_method)
    
    def _visualize_all_masks(
        self,
        masks: Dict[str, np.ndarray],
        output_dir: str
    ):
        """Визуализирует все маски в одной фигуре."""
        methods = list(masks.keys())
        n_methods = len(methods)
        
        # Определяем размер сетки
        n_cols = 4
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
        axes = axes.flatten()
        
        for i, (name, mask) in enumerate(masks.items()):
            ax = axes[i]
            ax.imshow(mask, cmap='gray')
            ax.set_title(f"{name}", fontsize=10)
            ax.axis('off')
        
        # Скрываем пустые оси
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle("Все маски сегментации", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        all_masks_path = os.path.join(output_dir, "all_masks.png")
        plt.savefig(all_masks_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"🖼️ Все маски: {all_masks_path}")
    
    def _create_html_report(
        self,
        df_comparisons: pd.DataFrame,
        masks: Dict[str, np.ndarray],
        method_infos: Dict[str, Any],
        output_dir: str,
        comparison_type: str,
        reference_method: Optional[str] = None
    ):
        """Создает HTML отчет с результатами."""
        
        html_path = os.path.join(output_dir, "report.html")
        
        # Статистика по методам
        methods_stats = []
        for name, mask in masks.items():
            mask_binary = mask > 127
            area = np.sum(mask_binary)
            total_pixels = mask.size
            coverage = area / total_pixels * 100
            
            methods_stats.append({
                'method': name,
                'area': area,
                'coverage': f"{coverage:.1f}%",
                'pixels': f"{area:,}",
                'time': method_infos.get(name, {}).get('execution_time', 0)
            })
        
        # Топ методов по F1 (если есть референс)
        if reference_method and 'f1_score' in df_comparisons.columns:
            ref_df = df_comparisons[df_comparisons['method1'] == reference_method]
            if not ref_df.empty:
                top_methods = ref_df.nlargest(5, 'f1_score')[['method2', 'f1_score']]
                top_methods_html = top_methods.to_html(index=False, 
                                                      float_format=lambda x: f"{x:.3f}")
            else:
                top_methods_html = "<p>Нет данных</p>"
        else:
            top_methods_html = "<p>Сравнение всех со всеми</p>"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Отчет сравнения методов сегментации</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                    .metric-card {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                    .highlight {{ background-color: #e6f7ff; }}
                </style>
            </head>
            <body>
                <h1>📊 Отчет сравнения методов сегментации</h1>
                
                <div class="summary">
                    <h2>Общая информация</h2>
                    <p><strong>Дата:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Всего методов:</strong> {len(masks)}</p>
                    <p><strong>Тип сравнения:</strong> {comparison_type}</p>
                    <p><strong>Референсный метод:</strong> {reference_method if reference_method else 'Нет (все со всеми)'}</p>
                </div>
                
                <h2>📈 Матрицы сравнения</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>F1-Score матрица</h3>
                        <img src="f1_score_matrix.png" alt="F1 Matrix">
                    </div>
                    <div class="metric-card">
                        <h3>Accuracy матрица</h3>
                        <img src="accuracy_matrix.png" alt="Accuracy Matrix">
                    </div>
                    <div class="metric-card">
                        <h3>Все маски</h3>
                        <img src="all_masks.png" alt="All Masks">
                    </div>
                </div>
                
                <h2>🏆 Топ методов</h2>
                {top_methods_html}
                
                <h2>📋 Статистика методов</h2>
                <table>
                    <tr>
                        <th>Метод</th>
                        <th>Площадь маски</th>
                        <th>Покрытие</th>
                        <th>Время (с)</th>
                    </tr>
            """)
            
            for stat in sorted(methods_stats, key=lambda x: x['area'], reverse=True):
                f.write(f"""
                    <tr>
                        <td>{stat['method']}</td>
                        <td>{stat['pixels']}</td>
                        <td>{stat['coverage']}</td>
                        <td>{stat['time']:.3f}</td>
                    </tr>
                """)
            
            f.write("""
                </table>
                
                <h2>🔗 Быстрые ссылки</h2>
                <ul>
                    <li><a href="comparisons.csv">CSV с результатами сравнения</a></li>
                    <li><a href="masks/">Папка с масками</a></li>
                    <li><a href="images/">Папка с изображениями</a></li>
                </ul>
                
                <footer>
                    <p>Сгенерировано автоматически с помощью SegmentationComparator</p>
                </footer>
            </body>
            </html>
            """)
        
        print(f"📄 HTML отчет: {html_path}")


# Пример использования
def example_usage() -> Tuple[SegmentationComparator, pd.DataFrame]:
    """Пример использования класса для сравнения методов"""
    
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
        output_dir="./data/batch_comparison_results"
    )
    
    # Выводим результаты
    if not df_results.empty:
        print("\nРезультаты сравнения:")
        print(df_results[['method', 'f1_score', 'accuracy', 'test_time']].sort_values('f1_score', ascending=False))
    
    return comparator, df_results

def comprehensive_comparison_example():
    """Пример полного сравнения всех методов"""
    
    url = "https://i.pinimg.com/736x/17/66/c4/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_np = np.array(img)
    
    # Конфигурация ВСЕХ методов
    all_methods_config = [
        # sklearn методы
        {"name": "kmeans", "type": "sklearn", "params": {"n_clusters": 3}},
        {"name": "dbscan", "type": "sklearn", "params": {"eps": 0.5, "min_samples": 5}},
        {"name": "meanshift", "type": "sklearn", "params": {"bandwidth": 0.5}},
        {"name": "gmm", "type": "sklearn", "params": {"n_components": 3}},
        
        # skimage методы сегментации
        {"name": "felzenszwalb", "type": "skimage", "params": {"scale": 100, "sigma": 0.8}},
        {"name": "slic", "type": "skimage", "params": {"n_segments": 100}},
        {"name": "quickshift", "type": "skimage", "params": {"kernel_size": 3}},
        {"name": "watershed", "type": "skimage", "params": {}},
        {"name": "random_walker", "type": "skimage", "params": {"beta": 130}},
        {"name": "chan_vese", "type": "skimage", "params": {"max_iter": 100}},
        {"name": "active_contour", "type": "skimage", "params": {"max_iter": 100}},
        
        # skimage пороговые методы
        {"name": "threshold_otsu", "type": "skimage", "params": {}},
        {"name": "threshold_niblack", "type": "skimage", "params": {"window_size": 25}},
        {"name": "threshold_sauvola", "type": "skimage", "params": {"window_size": 25}},
        
        # skimage детекторы границ
        {"name": "sobel", "type": "skimage", "params": {"threshold": 0.1}},
        {"name": "canny", "type": "skimage", "params": {"sigma": 1.0}},
    ]
    
    # Создаем компаратор
    comparator = SegmentationComparator()
    
    print("=" * 60)
    print("ПОЛНОЕ МАТРИЧНОЕ СРАВНЕНИЕ ВСЕХ МЕТОДОВ")
    print("=" * 60)
    
    # Вариант 1: Сравнение всех со всеми
    print("\n1. Сравнение всех методов со всеми...")
    results_all = comparator.matrix_comparison(
        img_np,
        methods_config=all_methods_config,
        comparison_type="all_vs_all",
        output_dir="./data/all_vs_all_comparison"
    )
    
    # Вариант 2: Все методы vs референс (например, felzenszwalb)
    print("\n2. Все методы vs референс (felzenszwalb)...")
    results_vs_ref = comparator.matrix_comparison(
        img_np,
        methods_config=all_methods_config,
        reference_method="skimage_felzenszwalb",
        comparison_type="all_vs_ref",
        output_dir="./data/all_vs_felzenszwalb"
    )
    
    # Вариант 3: Только попарное сравнение
    print("\n3. Попарное сравнение всех методов...")
    results_pairwise = comparator.matrix_comparison(
        img_np,
        methods_config=all_methods_config[:8],  # Берем первые 8 для скорости
        comparison_type="pairwise",
        output_dir="./data/pairwise_comparison"
    )
    
    # Анализ результатов
    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    if 'df_comparisons' in results_all:
        df_all = results_all['df_comparisons']
        
        # Находим наиболее похожие методы
        print("\nСамые похожие пары методов (F1 > 0.9):")
        high_similarity = df_all[df_all['f1_score'] > 0.9]
        
        if not high_similarity.empty:
            # Исключаем сравнение с самим собой
            high_similarity = high_similarity[high_similarity['method1'] != high_similarity['method2']]
            top_pairs = high_similarity.nlargest(10, 'f1_score')[['method1', 'method2', 'f1_score']]
            print(top_pairs.to_string(index=False))
        else:
            print("Нет пар с F1 > 0.9")
        
        # Находим наиболее разные методы
        print("\nСамые разные пары методов (F1 < 0.3):")
        low_similarity = df_all[df_all['f1_score'] < 0.3]
        
        if not low_similarity.empty:
            low_similarity = low_similarity[low_similarity['method1'] != low_similarity['method2']]
            bottom_pairs = low_similarity.nsmallest(10, 'f1_score')[['method1', 'method2', 'f1_score']]
            print(bottom_pairs.to_string(index=False))
    
    return comparator, results_all, results_vs_ref


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
        output_path = f"./data/custom_vs_{reference_method}.jpg"
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

    comparator, results_all, results_vs_ref = comprehensive_comparison_example()
    
    print("\n✅ Сравнение завершено!")
    print("Результаты сохранены в папках:")
    print("  - all_vs_all_comparison/")
    print("  - all_vs_felzenszwalb/")
    print("  - pairwise_comparison/")