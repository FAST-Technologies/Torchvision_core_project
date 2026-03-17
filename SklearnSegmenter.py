# SklearnSegmenter.py

# Импорт основных библиотек

from BaseSegmenter import BaseSegmenter
from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)
import numpy as np
import warnings
from collections import deque
import math
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Импорт scikit-learn компонентов
from sklearn.cluster import (
    KMeans, DBSCAN, MeanShift, OPTICS, 
    AgglomerativeClustering, SpectralClustering, 
    Birch, MiniBatchKMeans
)
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    pairwise_distances
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PolynomialFeatures
)
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from scipy import ndimage, signal, sparse
from scipy.sparse.linalg import eigsh
from skimage.util import img_as_float, img_as_ubyte

import skimage
from skimage import (
    filters, segmentation, morphology, measure,
    feature, color, exposure, transform, util,
    restoration, graph, draw
)
from skimage.color import label2rgb
from skimage.draw import polygon
from skimage.feature import canny
from skimage.filters import (
    threshold_otsu, threshold_local, threshold_niblack,
    threshold_sauvola, gaussian, sobel, prewitt, roberts,
    scharr, laplace, farid, butterworth
)
# from skimage.future import graph
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk, square, dilation, erosion, opening, closing,
    white_tophat, black_tophat, skeletonize, thin,
    remove_small_objects, remove_small_holes
)
from skimage.segmentation import (
    felzenszwalb, slic, quickshift, watershed,
    random_walker, active_contour, morphological_chan_vese,
    morphological_geodesic_active_contour, mark_boundaries
)
from skimage.util import img_as_ubyte, img_as_float
SKIMAGE_AVAILABLE = True

class SklearnSegmenter(BaseSegmenter):
    """
    Класс для сегментации изображений с использованием scikit-learn и scikit-image.
    Все реализации сделаны без использования OpenCV или специализированных
    библиотек для обработки изображений. Поддерживает как классические методы (пороговые, граничные), так и методы на основе кластеризации,
    активных контуров и графов.
    """
    def __init__(
        self, 
        method: str = "kmeans", 
        **kwargs
    ) -> None:
        super().__init__()
        self.method: str = method
        self.params: Dict[str, Any] = kwargs
        self._setup_methods()
        self._scaler: StandardScaler = StandardScaler()
    
    def _setup_methods(self) -> None:
        """Регистрация всех доступных методов сегментации."""
        self.methods: Dict[str, np.ndarray] = {
            # ============ ПОРОГОВЫЕ МЕТОДЫ СЕГМЕНТАЦИИ ============
            "global_thresholding": self._sklearn_global_thresholding,
            "adaptive_thresholding": self._sklearn_adaptive_thresholding,
            "otsu_thresholding": self._sklearn_otsu_thresholding,
            "threshold_niblack": self._sklearn_threshold_niblack,
            "threshold_sauvola": self._sklearn_threshold_sauvola,

            # ============ КРАЕВЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "sobel_edge": self._sklearn_sobel_edge,
            "canny_edge": self._sklearn_canny_edge,

            # ============ РЕГИОНАЛЬНЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "region_growing": self._sklearn_region_growing,
            "split_and_merge": self._sklearn_split_and_merge,
            "floodfill": self._sklearn_floodfill,

            # ============ КЛАСТЕРИЗАЦИЯ ============
            "kmeans_segmentation": self._sklearn_kmeans_segmentation,
            "dbscan_segmentation": self._sklearn_dbscan_segmentation,
            "meanshift": self._sklearn_meanshift,
            
            # ============ АКТИВНЫЕ КОНТУРЫ ============
            "active_contour": self._sklearn_active_contour,
            "gvf_contour": self._sklearn_gvf_contour,
            "morphological_snakes": self._sklearn_morphological_snakes,
            "chan_vese": self._sklearn_chan_vese,
            
            # ============ WATERSHED И ГРАФОВЫЕ ============
            "watershed": self._sklearn_watershed,
            "random_walker": self._sklearn_random_walker,
            
            # ============ SUPER-PIXEL МЕТОДЫ ============
            "quickshift": self._sklearn_quickshift,
            "slic": self._sklearn_slic,
            "felzenszwalb": self._sklearn_felzenszwalb,
            
            # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============
            "grabcut": self._sklearn_grabcut,
            
            # ============ МЕТОДЫ ИЗ SKLEARN ============
            "kmeans": self._sklearn_kmeans,
            "dbscan": self._sklearn_dbscan,
            "gmm": self._sklearn_gmm,
            "optics": self._sklearn_optics,
            "agglomerative": self._sklearn_agglomerative,
            "spectral": self._sklearn_spectral,
            "birch": self._sklearn_birch,
            "mini_batch_kmeans": self._sklearn_mini_batch_kmeans,

            # Классификация (для сегментации)
            "random_forest": self._sklearn_random_forest,
            "svm": self._sklearn_svm,
            "logistic_regression": self._sklearn_logistic_regression,
            "knn": self._sklearn_knn,
            "decision_tree": self._sklearn_decision_tree,
            "mlp": self._sklearn_mlp,
            "naive_bayes": self._sklearn_naive_bayes,
            "lda": self._sklearn_lda,
            "qda": self._sklearn_qda,
            
            # Обнаружение аномалий
            "isolation_forest": self._sklearn_isolation_forest,
            "local_outlier_factor": self._sklearn_local_outlier_factor,
            "one_class_svm": self._sklearn_one_class_svm,
            "elliptic_envelope": self._sklearn_elliptic_envelope,
            
            # Методы разложения
            "pca_segmentation": self._sklearn_pca_segmentation,
            "nmf_segmentation": self._sklearn_nmf_segmentation,
            "ica_segmentation": self._sklearn_ica_segmentation,
            
            # Методы многообразия
            "tsne_segmentation": self._sklearn_tsne_segmentation,
            "isomap_segmentation": self._sklearn_isomap_segmentation,
            "spectral_embedding": self._sklearn_spectral_embedding,
            
            # Комбинированные методы
            "ensemble_clustering": self._sklearn_ensemble_clustering,
            "hierarchical_kmeans": self._sklearn_hierarchical_kmeans,
            "pca_kmeans": self._sklearn_pca_kmeans,
            
            # Специальные методы для изображений
            "superpixel_clustering": self._sklearn_superpixel_clustering,
            "color_spatial_clustering": self._sklearn_color_spatial_clustering,
            "texture_clustering": self._sklearn_texture_clustering,
            
            # Байесовские методы
            "bayesian_gmm": self._sklearn_bayesian_gmm,
            "variational_gmm": self._sklearn_variational_gmm,
            
            # Методы плотности
            "density_based": self._sklearn_density_based,
            "hdbscan_emulation": self._sklearn_hdbscan_emulation,
            
            # Методы на основе графов
            "graph_clustering": self._sklearn_graph_clustering,
            "modularity_clustering": self._sklearn_modularity_clustering,
            
            # Методы с обучением
            "self_training": self._sklearn_self_training,
            "semi_supervised": self._sklearn_semi_supervised,
            
            # Методы на основе расстояния
            "distance_matrix": self._sklearn_distance_matrix,
            "affinity_propagation": self._sklearn_affinity_propagation,
        }
        
        self.methods: Dict[str, callable] = self.methods
        
        if self.method not in self.methods:
            raise ValueError(f"Неизвестный метод: {self.method}. "
                           f"Доступные методы: {list(self.methods.keys())}")

    def _normalize_image(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """Нормализация изображения к [0, 1] для skimage"""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)
    
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
        # # Конвертируем в RGB если нужно
        # if len(image.shape) == 2:
        #     # Grayscale to RGB
        #     image_rgb = np.stack([image] * 3, axis=-1)
        # elif image.shape[2] == 4:
        #     # RGBA to RGB
        #     image_rgb = image[:, :, :3]
        # else:
        #     image_rgb = image
        
        mask = self.methods[self.method](image)
        
        # Гарантируем правильный формат вывода
        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        if self.method in ['canny_edge', 'sobel_edge']:
            pass  # Не применяем постобработку
        elif self.params.get('postprocess', True):
            mask = self._postprocess_mask(mask)
            
        return mask
    
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
        
        # Создаем визуализацию
        if len(image.shape) == 2:
            overlay = np.stack([image] * 3, axis=-1)
        else:
            overlay = image.copy()
        
        # Красный цвет для маски
        overlay[mask > 127] = [255, 0, 0]
        
        # Смешиваем с оригиналом
        if len(image.shape) == 2:
            original_rgb = np.stack([image] * 3, axis=-1)
        else:
            original_rgb = image
        
        # Смешивание
        alpha = 0.7
        beta = 0.3
        result = (alpha * overlay + beta * original_rgb).astype(np.uint8)
        
        return result, mask
    
    # ============ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ============
    
    def _extract_features(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Извлечение признаков из изображения для scikit-learn методов.
        
        Args:
            image: Входное изображение RGB
        
        Returns:
            np.ndarray: Матрица признаков (n_samples x n_features)
        """
        h, w = image.shape[:2]
        
        # Базовые цветовые признаки
        if len(image.shape) == 3:
            if SKIMAGE_AVAILABLE:
                gray = color.rgb2gray(image)
                color_features = image.reshape(-1, 3).astype(np.float32)
            else:
                gray = np.mean(image, axis=2)
                color_features = image.reshape(-1, 3).astype(np.float32)
        else:
            gray = image
            color_features = image.reshape(-1, 1).astype(np.float32)
        
        # Пространственные признаки
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        spatial_features = np.stack([
            x_coords.ravel() / w,
            y_coords.ravel() / h,
            (x_coords.ravel() * y_coords.ravel()) / (w * h)
        ], axis=1)
        
        # Текстура (упрощенная)
        if SKIMAGE_AVAILABLE:
            grad_x = filters.sobel_h(gray)
            grad_y = filters.sobel_v(gray)
        else:
            # Реализация Собеля на numpy
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            grad_x = signal.convolve2d(gray, kernel_x, mode='same', boundary='symm')
            grad_y = signal.convolve2d(gray, kernel_y, mode='same', boundary='symm')
        
        texture_features = np.stack([
            grad_x.ravel(),
            grad_y.ravel(),
            np.sqrt(grad_x.ravel()**2 + grad_y.ravel()**2)
        ], axis=1)
        
        # Комбинируем все признаки
        features = np.hstack([
            color_features,
            spatial_features,
            texture_features,
        ])
        
        # Масштабирование
        if features.shape[0] > 0:
            features = self._scaler.fit_transform(features)
        
        return features
    
    def _create_mask_from_labels(
        self, 
        labels: np.ndarray, 
        shape: Tuple
    ) -> np.ndarray:
        """
        Создание бинарной маски из меток кластеризации.
        
        Args:
            labels: Метки кластеров
            shape: Форма выходной маски
        
        Returns:
            np.ndarray: Бинарная маска
        """
        labels_2d = labels.reshape(shape)
        unique_labels = np.unique(labels)
        
        # Исключаем шум (-1) если есть
        valid_labels = unique_labels[unique_labels != -1]
        
        if len(valid_labels) == 0:
            return np.zeros(shape, dtype=bool)
        
        # Находим самый большой кластер как фон
        label_sizes = [np.sum(labels_2d == label) for label in valid_labels]
        bg_label = valid_labels[np.argmax(label_sizes)]
        
        # Создаем маску (все кроме фона)
        mask = labels_2d != bg_label
        
        return mask
    
    def _postprocess_mask(
        self, 
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Постобработка маски для улучшения качества.
        
        Args:
            mask: Исходная маска
        
        Returns:
            np.ndarray: Улучшенная маска
        """
        binary = mask > 127
        
        # Морфологические операции
        if SKIMAGE_AVAILABLE:
            # Удаление мелких объектов
            binary = remove_small_objects(binary, min_size=self.params.get('min_area', 100))
            
            # Заполнение дыр
            binary = remove_small_holes(binary, area_threshold=self.params.get('min_area', 100))
            
            # Морфологические операции
            selem = disk(2)
            binary = closing(binary, selem)
            binary = opening(binary, selem)
        else:
            # Простая реализация на numpy
            # Удаление мелких объектов
            labeled, num_features = ndimage.label(binary)
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            for i, size in enumerate(sizes):
                if size < self.params.get('min_area', 100):
                    binary[labeled == i + 1] = False
        
        return binary.astype(np.uint8) * 255
    
    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    # ============ ПОРОГОВЫЕ МЕТОДЫ ============
    
    def _sklearn_global_thresholding(
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        threshold = self.params.get('threshold', 0.5)
        mask = gray > threshold
        
        return (mask * 255).astype(np.uint8)
    
    def _sklearn_adaptive_thresholding(
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        block_size = self.params.get('block_size', 11)
        C = self.params.get('C', 2)
        adaptive_thresh = threshold_local(gray, block_size=block_size, offset=C/255.0)
        mask = gray > adaptive_thresh
        
        return (mask * 255).astype(np.uint8)
    
    def _sklearn_otsu_thresholding(
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        thresh = threshold_otsu(gray)
        mask = gray > thresh
        
        return (mask * 255).astype(np.uint8)
    
    def _sklearn_threshold_niblack(
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        window_size = self.params.get('window_size', 15)
        k = self.params.get('k', -0.2)
        thresh = threshold_niblack(gray, window_size=window_size, k=k)
        mask = gray > thresh
        return (mask * 255).astype(np.uint8)
    
    def _sklearn_threshold_sauvola(
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        window_size = self.params.get('window_size', 15)
        k = self.params.get('k', 0.5)
        
        # Порог Сауволы из scikit-image
        thresh = threshold_sauvola(gray, window_size=window_size, k=k)
        mask = gray > thresh
        
        return (mask * 255).astype(np.uint8)
    
    # ============ МЕТОДЫ НА ОСНОВЕ КРАЕВ ============
    
    def _sklearn_sobel_edge(
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        threshold = self.params.get('threshold', 0.1)
        edges = sobel(gray)
        
        # Нормализация и порог
        if edges.max() > 0:
            edges = edges / edges.max()
        
        mask = edges > threshold
        
        return (mask * 255).astype(np.uint8)
    
    def _sklearn_canny_edge(
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        
        # 1. Нормализация к [0, 1]
        # gray = self._normalize_image(gray)
        if gray.dtype == np.uint8:
            gray = gray.astype(np.float32) / 255.0
        
        sigma = self.params.get('sigma', 1.0)
        low_threshold = self.params.get('low', 0.1)
        high_threshold = self.params.get('high', 0.3)
        use_quantiles = self.params.get('use_quantiles', False)
        
        # 3. Включаем квантили!
        use_quantiles = False 
        
        mask = feature.canny(
            gray, 
            sigma=sigma, 
            low_threshold=low_threshold, 
            high_threshold=high_threshold,
            use_quantiles=use_quantiles
        )
        print(f"DEBUG: sigma={sigma}, low={low_threshold}, high={high_threshold}, quantiles={use_quantiles}")
        print(f"DEBUG: Image range: [{gray.min():.4f}, {gray.max():.4f}]")
        
        return (mask * 255).astype(np.uint8)
        
    # ============ РЕГИОНАЛЬНЫЕ МЕТОДЫ ============
    
    def _sklearn_region_growing(
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)

        h, w = gray.shape
        seed = self.params.get('seed', (w//2, h//2))
        tolerance = self.params.get('tolerance', 0.1)
        
        mask = np.zeros((h, w), dtype=bool)
        visited = np.zeros((h, w), dtype=bool)
        
        queue = deque([seed])
        start_value = gray[seed[1], seed[0]]
        
        while queue:
            x, y = queue.popleft()
            
            if x < 0 or x >= w or y < 0 or y >= h or visited[y, x]:
                continue
            
            visited[y, x] = True
            
            if abs(float(gray[y, x]) - float(start_value)) <= tolerance:
                mask[y, x] = True
                
                # 8-связность
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                            queue.append((nx, ny))
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_split_and_merge(
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        h, w = gray.shape
        threshold = self.params.get('threshold', 0.1)
        min_size = self.params.get('min_size', 50)
        
        # Начальный регион
        regions = [gray.copy()]
        region_masks = [np.ones((h, w), dtype=bool)]
        
        # Простая реализация Split
        def split_region(region, mask):
            h_reg, w_reg = region.shape
            if h_reg * w_reg <= min_size:
                return [(region, mask)]
            
            mean = np.mean(region[mask])
            std = np.std(region[mask])
            
            if std < threshold:
                return [(region, mask)]
            
            # Разделяем на 4 части
            h_mid = h_reg // 2
            w_mid = w_reg // 2
            
            sub_regions = []
            sub_masks = []
            
            for i in range(2):
                for j in range(2):
                    r_start = i * h_mid
                    r_end = (i+1) * h_mid if i == 0 else h_reg
                    c_start = j * w_mid
                    c_end = (j+1) * w_mid if j == 0 else w_reg
                    
                    sub_region = region[r_start:r_end, c_start:c_end]
                    sub_mask = mask[r_start:r_end, c_start:c_end]
                    
                    if sub_mask.any():
                        sub_regions.append(sub_region)
                        sub_masks.append(sub_mask)
            
            # Рекурсивно разделяем
            result = []
            for sub_region, sub_mask in zip(sub_regions, sub_masks):
                result.extend(split_region(sub_region, sub_mask))
            
            return result
        
        # Split фаза
        split_results = split_region(gray, np.ones((h, w), dtype=bool))
        
        # Merge фаза (упрощенная)
        mask = np.zeros((h, w), dtype=np.uint8)
        if split_results:
            # Берем самый контрастный регион
            max_contrast = -1
            best_region_mask = None
            
            for region, region_mask in split_results:
                if region_mask.sum() > 0:
                    region_values = region[region_mask]
                    contrast = np.std(region_values)
                    
                    if contrast > max_contrast:
                        max_contrast = contrast
                        best_region_mask = region_mask
            
            if best_region_mask is not None:
                # Находим координаты маски в оригинальном изображении
                # (упрощенно - предполагаем, что регионы не перекрываются)
                mask[:best_region_mask.shape[0], :best_region_mask.shape[1]] = \
                    best_region_mask.astype(np.uint8)
        
        return mask.astype(np.uint8) * 255
    
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        h, w = gray.shape
        threshold = self.params.get('threshold', 0.1)
        min_size = self.params.get('min_size', 50)
        
        # Рекурсивная функция split
        def split_region(x1, y1, x2, y2):
            if (x2 - x1) * (y2 - y1) <= min_size:
                return [(x1, y1, x2, y2)]
            
            # Вычисляем статистики региона
            region = gray[y1:y2, x1:x2]
            mean = np.mean(region)
            std = np.std(region)
            
            if std < threshold:
                return [(x1, y1, x2, y2)]
            
            # Разделяем регион на 4 части
            x_mid = (x1 + x2) // 2
            y_mid = (y1 + y2) // 2
            
            regions = []
            regions.extend(split_region(x1, y1, x_mid, y_mid))
            regions.extend(split_region(x_mid, y1, x2, y_mid))
            regions.extend(split_region(x1, y_mid, x_mid, y2))
            regions.extend(split_region(x_mid, y_mid, x2, y2))
            
            return regions
        
        # Split фаза
        regions = split_region(0, 0, w, h)
        
        # Merge фаза (упрощенная)
        mask = np.zeros((h, w), dtype=bool)
        if len(regions) > 1:
            # Берем второй по величине регион
            sizes = [(y2-y1)*(x2-x1) for x1, y1, x2, y2 in regions]
            sorted_indices = np.argsort(sizes)[::-1]
            
            # Второй по величине регион
            if len(sorted_indices) > 1:
                idx = sorted_indices[1]
                x1, y1, x2, y2 = regions[idx]
                mask[y1:y2, x1:x2] = True
        
        return mask.astype(np.uint8) * 255
    
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
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        h, w = gray.shape
        seed = self.params.get('seed', (w//2, h//2))
        tolerance = self.params.get('tolerance', 0.1)
        mask = segmentation.flood(gray, seed, tolerance=tolerance)
        return mask.astype(np.uint8) * 255

    # ============ КЛАСТЕРИЗАЦИЯ ============
    
    def _sklearn_kmeans_segmentation(
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
        if len(img.shape) == 2:
            # Если grayscale, конвертируем в 3 канала
            img = np.stack([img] * 3, axis=-1)
        
        h, w = img.shape[:2]
        
        # Преобразуем изображение в массив пикселей
        pixels = img.reshape(-1, 3)
        
        # Применяем K-Means
        k = self.params.get('k', 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Находим самый большой кластер (фон)
        unique_labels, counts = np.unique(labels, return_counts=True)
        bg_label = unique_labels[np.argmax(counts)]
        
        # Создаем маску (все кроме фона)
        mask = (labels != bg_label).reshape(h, w)
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_kmeans(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """K-Means из sklearn с извлечением признаков."""
        features = self._extract_features(img)
        h, w = img.shape[:2]
        
        n_clusters = self.params.get('n_clusters', 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_dbscan_segmentation(
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
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        h, w = gray.shape
        
        # Извлекаем признаки (пиксель + координаты)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack([
            gray.ravel(),
            x_coords.ravel() / w,
            y_coords.ravel() / h
        ])
        
        # Применяем DBSCAN
        eps = self.params.get('eps', 0.1)
        min_samples = self.params.get('min_samples', 10)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        
        # Создаем маску (исключаем шум -1)
        mask = (labels != -1).reshape(h, w)
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_dbscan(
        self, 
        img: np.ndarray
    ) -> np.ndarray:
        """DBSCAN из sklearn."""
        features = self._extract_features(img)
        h, w = img.shape[:2]
        
        eps = self.params.get('eps', 0.5)
        min_samples = self.params.get('min_samples', 5)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        
        # Создаем маску (исключаем шум)
        mask = (labels != -1).reshape(h, w)
        
        return mask.astype(np.uint8) * 255
    
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
        if len(img.shape) == 3:
            img_rgb = img
        else:
            img_rgb = np.stack([img] * 3, axis=-1)
        
        h, w = img_rgb.shape[:2]
        
        # Для производительности сэмплируем пиксели
        sample_size = min(1000, h * w)
        indices = np.random.choice(h * w, sample_size, replace=False)
        
        pixels = img_rgb.reshape(-1, 3)[indices]
        coords = np.column_stack([
            indices // w,
            indices % w
        ]) / [h, w]
        
        features = np.hstack([pixels / 255.0, coords])
        
        # Применяем MeanShift
        bandwidth = self.params.get('bandwidth', 0.2)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        labels = ms.fit_predict(features)
        
        # Интерполируем метки обратно на все пиксели
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, labels)
        
        all_coords = np.column_stack([
            np.repeat(np.arange(h), w),
            np.tile(np.arange(w), h)
        ]) / [h, w]
        
        all_pixels = img_rgb.reshape(-1, 3) / 255.0
        all_features = np.hstack([all_pixels, all_coords])
        
        all_labels = knn.predict(all_features)
        
        # Находим самый большой кластер
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        bg_label = unique_labels[np.argmax(counts)]
        
        mask = (all_labels != bg_label).reshape(h, w)
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_meanshift(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        MeanShift кластеризация для сегментации.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Ограничиваем размер для производительности
        max_samples = 5000
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            sample_features = features[indices]
            use_sampling = True
        else:
            sample_features = features
            use_sampling = False
        
        # Параметры MeanShift
        bandwidth = self.params.get('bandwidth', None)
        if bandwidth is None:
            # Автоматическая оценка bandwidth
            bandwidth = self._estimate_meanshift_bandwidth(sample_features)
        
        # Применяем MeanShift
        meanshift = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=True,
            min_bin_freq=1,
            cluster_all=True,
            n_jobs=-1
        )
        
        if use_sampling:
            meanshift.fit(sample_features)
            # Предсказываем для всех точек
            labels = meanshift.predict(features)
        else:
            labels = meanshift.fit_predict(features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        mask = self._postprocess_mask(mask)
        
        return mask.astype(np.uint8) * 255
    
    # ============ АКТИВНЫЕ КОНТУРЫ ============
    
    def _active_contour(self, img: np.ndarray) -> np.ndarray:
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        h, w = gray.shape
        
        # Создаем начальный контур (окружность)
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y) // 2
        
        s = np.linspace(0, 2 * np.pi, 100)
        # r = center_y + radius * np.sin(s)
        # c = center_x + radius * np.cos(s)
        # init = np.array([r, c]).T
        init = np.array([center_x + radius * np.cos(s),
                         center_y + radius * np.sin(s)]).T
        
        # Параметры активного контура
        alpha = self.params.get('alpha', 0.015) # elasticity (0.01)
        beta = self.params.get('beta', 10) # rigidity (0.1)
        gamma = self.params.get('gamma', 0.001) # time step (0.001)
        max_iterations = self.params.get('max_iterations', 2000)
        
        # Применяем активный контур
        snake = active_contour(
            gaussian(gray, 3, preserve_range=False),
            init, 
            alpha=alpha, 
            beta=beta, 
            gamma=gamma, 
            w_line=0,      # attract to light
            w_edge=1,      # attract to edges
            max_num_iter=max_iterations
        )
        
        # Создаем маску из контура
        mask = np.zeros((h, w), dtype=bool)
        rr, cc = polygon(snake[:, 1], snake[:, 0], mask.shape)
        mask[rr, cc] = True
        
        return mask.astype(np.uint8) * 255
    
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        # Вычисляем градиенты
        # grad_y, grad_x = np.gradient(gray_norm)
        grad_x = filters.sobel_h(gray)
        grad_y = filters.sobel_v(gray)

        # Вычисляем внешние силы (edge map)
        edge_map = grad_x**2 + grad_y**2
        edge_map = edge_map / (edge_map.max() + 1e-8)
        
        # Применяем GVF
        mu = self.params.get('mu', 0.1) # 0.2
        iterations = self.params.get('iterations', 50) # 100
        
        # Инициализируем GVF поле
        u = grad_x.copy() * edge_map
        v = grad_y.copy() * edge_map
         
        for _ in range(iterations):
            laplacian_u = filters.laplace(u)
            laplacian_v = filters.laplace(v)
            
            u = u + mu * laplacian_u - edge_map * (u - grad_x)
            v = v + mu * laplacian_v - edge_map * (v - grad_y)
        
        # Вычисляем величину GVF
        gvf_mag = np.sqrt(u**2 + v**2)

        # Нормализуем и пороговое разделение
        mask = gvf_mag > np.percentile(gvf_mag, 70)
        
        return mask.astype(np.uint8) * 255
    
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
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        # Применяем морфологические активные контуры
        init_level_set = np.zeros(gray.shape, dtype=np.int8)
        h, w = gray.shape
        init_level_set[h//4:3*h//4, w//4:3*w//4] = 1
        
        smoothing = self.params.get('smoothing', 1)
        threshold = self.params.get('threshold', 0.5)
        iterations = self.params.get('iterations', 50)
        
        mask = morphological_geodesic_active_contour(
            gray, iterations, init_level_set,
            smoothing=smoothing, threshold=threshold,
            balloon=1
        )
        
        return mask.astype(np.uint8) * 255
    
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
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        # Начальная маска
        init_level_set = np.zeros(gray.shape, dtype=np.int8)
        h, w = gray.shape
        init_level_set[h//4:3*h//4, w//4:3*w//4] = 1
        
        # Параметры Chan-Vese
        mu = self.params.get('mu', 0.25)
        lambda1 = self.params.get('lambda1', 1.0)
        lambda2 = self.params.get('lambda2', 1.0)
        tol = self.params.get('tol', 1e-3)
        max_iter = self.params.get('max_iter', 100)
        iterations = self.params.get('iterations', 100)
        
        # Применяем метод Chan-Vese
        mask = morphological_chan_vese(
            gray, 
            iterations, 
            init_level_set,
            smoothing=1, 
            lambda1=lambda1,
            lambda2=lambda2
        )

        # segmentation = chan_vese(
        #         gray_norm,
        #         mu=mu,
        #         lambda1=lambda1,
        #         lambda2=lambda2,
        #         tol=tol,
        #         max_num_iter=max_iter,
        #         init_level_set=init_level_set
        #     )
        # mask = (segmentation > 0.5).astype(np.uint8) * 255
        
        return mask.astype(np.uint8) * 255
    
    # ============ WATERSHED И ГРАФОВЫЕ МЕТОДЫ ============
    
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
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        # Маркеры для watershed
        markers = np.zeros_like(gray, dtype=np.uint8)
        h, w = gray.shape
        
        # Создаем маркеры
        markers[gray < np.percentile(gray, 25)] = 1
        markers[gray > np.percentile(gray, 75)] = 2
        
        # Применяем watershed
        segmentation = watershed(gray, markers)
        
        # Бинаризуем (все кроме фона)
        mask = segmentation == 2
        
        return mask.astype(np.uint8) * 255
    
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
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)
        
        # Создаем маркеры
        markers = np.zeros(gray.shape, dtype=np.uint8)
        h, w = gray.shape
        
        # Центральная область - объект
        markers[h//4:3*h//4, w//4:3*w//4] = 2
        
        # Углы - фон
        corner_size = min(h, w) // 8
        markers[:corner_size, :corner_size] = 1
        markers[:corner_size, -corner_size:] = 1
        markers[-corner_size:, :corner_size] = 1
        markers[-corner_size:, -corner_size:] = 1
        
        # Применяем Random Walker
        labels = random_walker(gray, markers, beta=10, mode='cg_mg')
        
        # Бинаризуем
        mask = labels == 2
        
        return mask.astype(np.uint8) * 255
    
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
        segments = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
        
        # Находим самый большой суперпиксель
        unique_labels, counts = np.unique(segments, return_counts=True)
        bg_label = unique_labels[np.argmax(counts)]
        
        # Создаем маску (все кроме фона)
        mask = segments != bg_label
        
        return mask.astype(np.uint8)
    
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
        n_segments = self.params.get('n_segments', 100)
        compactness = self.params.get('compactness', 10)
        
        # Применяем SLIC
        segments = slic(img, n_segments=n_segments, compactness=compactness)
        
        # Находим самый большой суперпиксель
        unique_labels, counts = np.unique(segments, return_counts=True)
        bg_label = unique_labels[np.argmax(counts)]
        
        # Создаем маску
        mask = segments != bg_label
        
        return mask.astype(np.uint8) * 255
    
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
        scale = self.params.get('scale', 100)
        sigma = self.params.get('sigma', 0.5)
        min_size = self.params.get('min_size', 50)
        
        # Применяем Felzenszwalb
        segments = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
        
        # Находим самый большой регион
        unique_labels, counts = np.unique(segments, return_counts=True)
        if len(unique_labels) > 0:
            bg_label = unique_labels[np.argmax(counts)]
            mask_np = (segments != bg_label).astype(np.uint8) * 255
        else:
            mask_np = np.zeros_like(segments, dtype=np.uint8)
        
        return mask_np
    
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
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        
        h, w = img.shape[:2]
        
        # Создаем начальную маску
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Прямоугольник в центре
        rect = self.params.get('rect', (w//4, h//4, w//2, h//2))
        x, y, w_rect, h_rect = rect
        
        mask[y:y+h_rect, x:x+w_rect] = 3  # Вероятный передний план
        
        # Углы - определенный фон
        corner_size = min(h, w) // 8
        mask[:corner_size, :corner_size] = 0  # Определенный фон
        mask[:corner_size, -corner_size:] = 0
        mask[-corner_size:, :corner_size] = 0
        mask[-corner_size:, -corner_size:] = 0
        
        # Используем Random Forest для имитации GrabCut
        # Подготовка данных
        pixels = img.reshape(-1, 3)
        coords = np.column_stack([
            np.repeat(np.arange(h), w),
            np.tile(np.arange(w), h)
        ])
        
        features = np.hstack([pixels / 255.0, coords / [h, w]])
        
        # Выбираем пиксели для обучения
        train_mask = (mask.ravel() == 0) | (mask.ravel() == 3)
        X_train = features[train_mask]
        y_train = mask.ravel()[train_mask]
        y_train = (y_train == 3).astype(int)  # 0 - фон, 1 - передний план
        
        # Обучаем Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        
        # Предсказываем для всех пикселей
        labels = rf.predict(features)
        
        mask_result = labels.reshape(h, w)
        
        return mask_result.astype(np.uint8)
    
    # ============ SKLEARN МЕТОДЫ ============
    
    def _sklearn_gmm(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Gaussian Mixture Models для сегментации.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        n_components = self.params.get('n_components', 3)
        if n_components <= 0:
            n_components = self._estimate_gmm_components(features)
        
        # Тип ковариационной матрицы
        covariance_type = self.params.get('covariance_type', 'full')
        
        # Применяем GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=1e-3,
            reg_covar=1e-6,
            max_iter=100,
            n_init=1,
            init_params='kmeans',
            random_state=42
        )
        
        labels = gmm.fit_predict(features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))      
        return mask.astype(np.uint8)
    
    def _sklearn_optics(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        OPTICS кластеризация для сегментации.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Ограничиваем размер для производительности
        max_samples = 2000
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            sample_features = features[indices]
            use_sampling = True
        else:
            sample_features = features
            use_sampling = False
        
        # Параметры OPTICS
        min_samples = self.params.get('min_samples', 5)
        xi = self.params.get('xi', 0.05)
        min_cluster_size = self.params.get('min_cluster_size', 0.1)
        
        # Применяем OPTICS
        optics = OPTICS(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_method='xi',
            algorithm='auto',
            leaf_size=30,
            n_jobs=-1
        )
        
        if use_sampling:
            optics.fit(sample_features)
            # Интерполируем метки
            labels = self._interpolate_labels(sample_features, features, 
                                            optics.labels_)
        else:
            labels = optics.fit_predict(features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        
        return mask.astype(np.uint8)
    
    def _sklearn_agglomerative(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Agglomerative (иерархическая) кластеризация.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Ограничиваем размер
        max_samples = 1000
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            sample_features = features[indices]
            use_sampling = True
        else:
            sample_features = features
            use_sampling = False
        
        # Параметры
        n_clusters = self.params.get('n_clusters', 3)
        linkage = self.params.get('linkage', 'ward')
        affinity = self.params.get('affinity', 'euclidean')
        
        # Применяем Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity,
            compute_full_tree='auto'
        )
        
        if use_sampling:
            clustering.fit(sample_features)
            labels = self._interpolate_labels(sample_features, features,
                                            clustering.labels_)
        else:
            labels = clustering.fit_predict(features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        
        return mask.astype(np.uint8)
    
    def _sklearn_spectral(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Spectral Clustering для сегментации.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Ограничиваем размер (Spectral Clustering требователен к памяти)
        max_samples = 1000
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            sample_features = features[indices]
            use_sampling = True
        else:
            sample_features = features
            use_sampling = False
        
        # Параметры
        n_clusters = self.params.get('n_clusters', 3)
        affinity = self.params.get('affinity', 'nearest_neighbors')
        n_neighbors = self.params.get('n_neighbors', 10)
        
        # Применяем Spectral Clustering
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_solver='arpack',
            random_state=42,
            n_jobs=-1
        )
        
        if use_sampling:
            spectral.fit(sample_features)
            labels = self._interpolate_labels(sample_features, features,
                                            spectral.labels_)
        else:
            labels = spectral.fit_predict(features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        return mask.astype(np.uint8)
    
    def _sklearn_birch(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Параметры BIRCH
        n_clusters = self.params.get('n_clusters', 3)
        threshold = self.params.get('threshold', 0.5)
        branching_factor = self.params.get('branching_factor', 50)
        
        # Применяем BIRCH
        birch = Birch(
            n_clusters=n_clusters,
            threshold=threshold,
            branching_factor=branching_factor
        )
        
        labels = birch.fit_predict(features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        return mask
    
    def _sklearn_mini_batch_kmeans(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Mini-Batch K-Means для больших изображений.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Параметры
        n_clusters = self.params.get('n_clusters', 3)
        batch_size = self.params.get('batch_size', 100)
        
        # Применяем Mini-Batch K-Means
        mbkmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            init='k-means++',
            n_init=3,
            max_iter=100,
            random_state=42
        )
        
        labels = mbkmeans.fit_predict(features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        return mask.astype(np.uint8)
    
    # ============ МЕТОДЫ КЛАССИФИКАЦИИ ДЛЯ СЕГМЕНТАЦИИ ============
    
    # def _sklearn_random_forest(
    #     self, 
    #     image: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Random Forest для сегментации (полу-контролируемый подход).
        
    #     Args:
    #         image: Входное изображение
        
    #     Returns:
    #         Бинарная маска сегментации
    #     """
    #     features = self._extract_features(image)
    #     h, w = image.shape[:2]
        
    #     # Автоматически создаем метки для обучения
    #     # (предполагаем, что центральная область - объект, края - фон)
    #     labels_train = np.zeros(features.shape[0])
        
    #     # Создаем маску для обучения
    #     train_mask = np.zeros((h, w), dtype=bool)
        
    #     # Центральная область - объект (класс 1)
    #     center_h, center_w = h // 2, w // 2
    #     obj_size = min(h, w) // 4
    #     cv2.rectangle(train_mask, 
    #                  (center_w - obj_size, center_h - obj_size),
    #                  (center_w + obj_size, center_h + obj_size),
    #                  True, -1)
        
    #     # Углы - фон (класс 0)
    #     corner_size = min(h, w) // 8
    #     corners = [
    #         (0, 0, corner_size, corner_size),
    #         (w - corner_size, 0, w, corner_size),
    #         (0, h - corner_size, corner_size, h),
    #         (w - corner_size, h - corner_size, w, h)
    #     ]
        
    #     for x1, y1, x2, y2 in corners:
    #         train_mask[y1:y2, x1:x2] = True
        
    #     # Преобразуем в метки
    #     labels_train = train_mask.ravel().astype(int)
        
    #     # Выбираем только помеченные пиксели
    #     train_indices = np.where(labels_train >= 0)[0]
    #     X_train = features[train_indices]
    #     y_train = labels_train[train_indices]
        
    #     # Обучаем Random Forest
    #     rf = RandomForestClassifier(
    #         n_estimators=self.params.get('n_estimators', 100),
    #         max_depth=self.params.get('max_depth', None),
    #         min_samples_split=self.params.get('min_samples_split', 2),
    #         min_samples_leaf=self.params.get('min_samples_leaf', 1),
    #         random_state=42,
    #         n_jobs=-1
    #     )
        
    #     rf.fit(X_train, y_train)
        
    #     # Предсказываем для всего изображения
    #     labels = rf.predict(features)
        
    #     # Создаем маску
    #     mask = labels.reshape(h, w).astype(np.uint8) * 255
    #     mask = self._postprocess_mask(mask)
        
    #     return mask
    def _sklearn_random_forest(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """Random Forest для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Создаем метки для обучения
        labels_train = -np.ones(h * w)
        
        # Центральная область - объект
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Маска объекта
        obj_mask = ((x_coords - center_w)**2 + (y_coords - center_h)**2) <= obj_size**2
        labels_train[obj_mask.ravel()] = 1
        
        # Углы - фон
        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h)
        ]
        
        for x1, y1, x2, y2 in corners:
            labels_train[y_coords[y1:y2, x1:x2].ravel()] = 0
        
        # Обучаем Random Forest
        train_indices = labels_train >= 0
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(features[train_indices], labels_train[train_indices])
        
        # Предсказываем
        labels = rf.predict(features)
        mask = labels.reshape(h, w)
        
        return mask.astype(np.uint8)
    
    def _sklearn_svm(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Support Vector Machine для сегментации.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Создаем метки для обучения (как в Random Forest)
        labels_train = np.zeros(features.shape[0])
        train_mask = np.zeros((h, w), dtype=bool)
        
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        cv2.rectangle(train_mask, 
                     (center_w - obj_size, center_h - obj_size),
                     (center_w + obj_size, center_h + obj_size),
                     True, -1)
        
        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h)
        ]
        
        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True
        
        labels_train = train_mask.ravel().astype(int)
        train_indices = np.where(labels_train >= 0)[0]
        X_train = features[train_indices]
        y_train = labels_train[train_indices]
        
        # Обучаем SVM
        svm = SVC(
            C=self.params.get('C', 1.0),
            kernel=self.params.get('kernel', 'rbf'),
            gamma=self.params.get('gamma', 'scale'),
            probability=True,
            random_state=42
        )
        
        svm.fit(X_train, y_train)
        
        # Предсказываем
        labels = svm.predict(features)
        
        # Создаем маску
        mask = labels.reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)
        
        return mask
    
    def _sklearn_logistic_regression(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Logistic Regression для сегментации.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Создаем метки для обучения
        labels_train = np.zeros(features.shape[0])
        train_mask = np.zeros((h, w), dtype=bool)
        
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        cv2.rectangle(train_mask, 
                     (center_w - obj_size, center_h - obj_size),
                     (center_w + obj_size, center_h + obj_size),
                     True, -1)
        
        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h)
        ]
        
        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True
        
        labels_train = train_mask.ravel().astype(int)
        train_indices = np.where(labels_train >= 0)[0]
        X_train = features[train_indices]
        y_train = labels_train[train_indices]
        
        # Обучаем Logistic Regression
        lr = LogisticRegression(
            penalty='l2',
            C=self.params.get('C', 1.0),
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        lr.fit(X_train, y_train)
        
        # Предсказываем
        labels = lr.predict(features)
        
        # Создаем маску
        mask = labels.reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)
        
        return mask
    
    def _sklearn_knn(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        K-Nearest Neighbors для сегментации.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Создаем метки для обучения
        labels_train = np.zeros(features.shape[0])
        train_mask = np.zeros((h, w), dtype=bool)
        
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        cv2.rectangle(train_mask, 
                     (center_w - obj_size, center_h - obj_size),
                     (center_w + obj_size, center_h + obj_size),
                     True, -1)
        
        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h)
        ]
        
        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True
        
        labels_train = train_mask.ravel().astype(int)
        train_indices = np.where(labels_train >= 0)[0]
        X_train = features[train_indices]
        y_train = labels_train[train_indices]
        
        # Обучаем KNN
        knn = KNeighborsClassifier(
            n_neighbors=self.params.get('n_neighbors', 5),
            weights=self.params.get('weights', 'uniform'),
            algorithm='auto',
            leaf_size=30,
            n_jobs=-1
        )
        
        knn.fit(X_train, y_train)
        
        # Предсказываем
        labels = knn.predict(features)
        
        # Создаем маску
        mask = labels.reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)
        
        return mask
    
    # ============ МЕТОДЫ ОБНАРУЖЕНИЯ АНОМАЛИЙ ============
    
    def _sklearn_isolation_forest(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Isolation Forest для сегментации (объект как аномалия).
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Применяем Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=self.params.get('n_estimators', 100),
            contamination=self.params.get('contamination', 'auto'),
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        
        # Предсказываем аномалии (-1 - аномалия, 1 - норма)
        labels = iso_forest.fit_predict(features)
        
        # Преобразуем в маску (аномалии = объект)
        mask = (labels == -1).reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)
        
        return mask
    
    def _sklearn_local_outlier_factor(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Local Outlier Factor для сегментации.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Ограничиваем размер для производительности
        max_samples = 2000
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            sample_features = features[indices]
            use_sampling = True
        else:
            sample_features = features
            use_sampling = False
        
        # Применяем LOF
        lof = LocalOutlierFactor(
            n_neighbors=self.params.get('n_neighbors', 20),
            contamination=self.params.get('contamination', 'auto'),
            novelty=False,
            n_jobs=-1
        )
        
        if use_sampling:
            labels_sample = lof.fit_predict(sample_features)
            labels = self._interpolate_labels(sample_features, features,
                                            labels_sample, method='knn')
        else:
            labels = lof.fit_predict(features)
        
        # Преобразуем в маску
        mask = (labels == -1).reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)
        
        return mask
    
    def _sklearn_one_class_svm(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        One-Class SVM для сегментации.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Обучаем на центральной области (предполагаем, что это фон)
        center_mask = np.zeros((h, w), dtype=bool)
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        
        cv2.rectangle(center_mask,
                     (center_w - obj_size, center_h - obj_size),
                     (center_w + obj_size, center_h + obj_size),
                     True, -1)
        
        # Выбираем пиксели фона (все кроме центра)
        background_mask = ~center_mask
        X_train = features[background_mask.ravel()]
        
        # Обучаем One-Class SVM
        oc_svm = OneClassSVM(
            kernel=self.params.get('kernel', 'rbf'),
            gamma=self.params.get('gamma', 'auto'),
            nu=self.params.get('nu', 0.1)
        )
        
        oc_svm.fit(X_train)
        
        # Предсказываем для всех пикселей
        labels = oc_svm.predict(features)
        
        # Преобразуем в маску (объект = -1)
        mask = (labels == -1).reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)
        
        return mask
    
    # ============ МЕТОДЫ РАЗЛОЖЕНИЯ ============
    
    def _sklearn_pca_segmentation(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        PCA-based сегментация.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Применяем PCA
        n_components = self.params.get('n_components', 3)
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(features)
        
        # Кластеризуем в новом пространстве
        n_clusters = self.params.get('n_clusters', 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(transformed)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        
        return mask.astype(np.uint8)
    
    def _sklearn_nmf_segmentation(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Non-negative Matrix Factorization для сегментации.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Убедимся, что данные неотрицательные
        features_nonneg = features - features.min()
        
        # Применяем NMF
        n_components = self.params.get('n_components', 3)
        nmf = NMF(n_components=n_components, init='random', random_state=42)
        
        # Преобразуем данные
        transformed = nmf.fit_transform(features_nonneg)
        
        # Кластеризуем
        n_clusters = self.params.get('n_clusters', 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(transformed)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        
        return mask.astype(np.uint8)
    
    def _sklearn_tsne_segmentation(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        t-SNE для сегментации (визуализация + кластеризация).
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Ограничиваем размер для производительности
        max_samples = 1000
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            sample_features = features[indices]
            use_sampling = True
        else:
            sample_features = features
            use_sampling = False
        
        # Применяем t-SNE
        perplexity = self.params.get('perplexity', 30)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        
        if use_sampling:
            transformed_sample = tsne.fit_transform(sample_features)
            
            # Интерполируем для всех точек
            knn = KNeighborsRegressor(n_neighbors=5)
            knn.fit(sample_features, transformed_sample)
            transformed = knn.predict(features)
        else:
            transformed = tsne.fit_transform(features)
        
        # Кластеризуем
        n_clusters = self.params.get('n_clusters', 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(transformed)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        return mask.astype(np.uint8)
    
    # ============ КОМБИНИРОВАННЫЕ МЕТОДЫ ============
    
    def _sklearn_ensemble_clustering(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Ensemble clustering (комбинация нескольких методов).
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Создаем несколько кластеризаций
        clusterings = []
        
        # 1. K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusterings.append(kmeans.fit_predict(features))
        
        # 2. Agglomerative
        agglomerative = AgglomerativeClustering(n_clusters=3)
        clusterings.append(agglomerative.fit_predict(features))
        
        # 3. GMM
        gmm = GaussianMixture(n_components=3, random_state=42)
        clusterings.append(gmm.fit_predict(features))
        
        # Создаем матрицу согласованности
        n_samples = features.shape[0]
        consensus = np.zeros((n_samples, n_samples))
        
        for labels in clusterings:
            # Матрица совпадений
            match_matrix = labels[:, None] == labels[None, :]
            consensus += match_matrix.astype(float)
        
        # Нормализуем
        consensus /= len(clusterings)
        
        # Кластеризуем матрицу согласованности
        n_clusters = self.params.get('n_clusters', 2)
        spectral = SpectralClustering(n_clusters=n_clusters, 
                                    affinity='precomputed',
                                    random_state=42)
        
        labels = spectral.fit_predict(consensus)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        return mask.astype(np.uint8)
    
    def _sklearn_color_spatial_clustering(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Кластеризация с учетом цвета и пространственных координат.
        
        Args:
            image: Входное изображение
        
        Returns:
            Бинарная маска сегментации
        """
        h, w = image.shape[:2]
        
        # Создаем признаки: цвет + пространственные координаты
        if len(image.shape) == 3:
            color = image.reshape(-1, 3).astype(np.float32) / 255.0
        else:
            color = image.reshape(-1, 1).astype(np.float32) / 255.0
        
        # Пространственные координаты
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        spatial = np.stack([
            x_coords.ravel() / w,
            y_coords.ravel() / h
        ], axis=1)
        
        # Веса для баланса цвета и пространства
        color_weight = self.params.get('color_weight', 0.7)
        spatial_weight = self.params.get('spatial_weight', 0.3)
        
        # Комбинируем признаки
        features = np.hstack([
            color * color_weight,
            spatial * spatial_weight
        ])
        
        # Масштабируем
        features = self._scaler.fit_transform(features)
        
        # Кластеризуем
        n_clusters = self.params.get('n_clusters', 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        return mask.astype(np.uint8)
    
    # ============ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ АВТОМАТИЧЕСКОЙ НАСТРОЙКИ ============
    
    def _estimate_optimal_clusters(
        self, 
        features: np.ndarray, 
        max_k: int = 10
    ) -> int:
        """Оценка оптимального числа кластеров методом локтя."""
        if features.shape[0] < 10:
            return 2
        
        inertias = []
        k_range = range(1, min(max_k, features.shape[0]) + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        # Находим точку перегиба
        if len(inertias) >= 3:
            # Вычисляем вторую производную
            derivatives = np.diff(inertias)
            second_derivatives = np.diff(derivatives)
            
            if len(second_derivatives) > 0:
                elbow_point = np.argmax(np.abs(second_derivatives)) + 2
            else:
                elbow_point = 2
        else:
            elbow_point = 2
        
        return max(2, min(elbow_point, max_k))
    
    def _estimate_dbscan_params(
        self, 
        features: np.ndarray
    ) -> Tuple[float, int]:
        """Автоматическая оценка параметров DBSCAN."""
        # Используем метод k-distance graph
        
        n_samples = min(1000, features.shape[0])
        sample_indices = np.random.choice(features.shape[0], n_samples, replace=False)
        sample_features = features[sample_indices]
        
        # Вычисляем расстояния до k-го соседа
        k = min(10, n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(sample_features)
        distances, _ = nbrs.kneighbors(sample_features)
        
        k_distances = distances[:, -1]
        k_distances_sorted = np.sort(k_distances)
        
        # Находим точку перегиба
        gradients = np.diff(k_distances_sorted)
        inflection_idx = np.argmax(gradients) + 1
        
        eps = float(k_distances_sorted[inflection_idx] * 1.1)
        min_samples = max(2 * features.shape[1], 5)
        
        return eps, int(min_samples)
    
    def _estimate_meanshift_bandwidth(
        self, 
        features: np.ndarray
    ) -> float:
        """Оценка bandwidth для MeanShift."""
        # Используем квантильный метод
        
        n_samples = min(500, features.shape[0])
        sample_indices = np.random.choice(features.shape[0], n_samples, replace=False)
        sample_features = features[sample_indices]
        
        nbrs = NearestNeighbors(n_neighbors=5).fit(sample_features)
        distances, _ = nbrs.kneighbors(sample_features)
        avg_distances = distances.mean(axis=1)
        
        quantile = self.params.get('quantile', 0.3)
        bandwidth = float(np.percentile(avg_distances, quantile * 100))
        
        return max(bandwidth, 0.1)
    
    def _estimate_gmm_components(
        self, 
        features: np.ndarray, 
        max_components: int = 10
    ) -> int:
        """Оценка числа компонент для GMM с помощью BIC."""
        if features.shape[0] < 20:
            return 2
        
        bics = []
        n_components_range = range(1, min(max_components, features.shape[0] // 10))
        
        for n in n_components_range:
            try:
                gmm = GaussianMixture(n_components=n, random_state=42)
                gmm.fit(features)
                bics.append(gmm.bic(features))
            except:
                bics.append(np.inf)
        
        if bics:
            optimal_n = n_components_range[np.argmin(bics)]
        else:
            optimal_n = 2
        
        return max(2, optimal_n)
    
    def _interpolate_labels(self, train_features: np.ndarray, 
                          test_features: np.ndarray, 
                          train_labels: np.ndarray,
                          method: str = 'knn') -> np.ndarray:
        """
        Интерполяция меток с обучающего набора на тестовый.
        
        Args:
            train_features: Признаки обучающей выборки
            test_features: Признаки тестовой выборки
            train_labels: Метки обучающей выборки
            method: Метод интерполяции ('knn' или 'nearest')
        
        Returns:
            Интерполированные метки
        """
        if method == 'knn':
            # Используем KNN для интерполяции
            knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            knn.fit(train_features, train_labels)
            test_labels = knn.predict(test_features)
        else:
            # Ближайший сосед
            nbrs = NearestNeighbors(n_neighbors=1).fit(train_features)
            _, indices = nbrs.kneighbors(test_features)
            test_labels = train_labels[indices.flatten()]
        
        return test_labels
    
    # ============ ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ (для полноты) ============
    
    def _sklearn_decision_tree(self, image: np.ndarray) -> np.ndarray:
        """Decision Tree для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Создаем метки для обучения
        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]
        
        X_train = features[train_indices]
        y_train = labels_train[train_indices]
        
        # Обучаем Decision Tree
        dt = DecisionTreeClassifier(
            max_depth=self.params.get('max_depth', None),
            min_samples_split=self.params.get('min_samples_split', 2),
            random_state=42
        )
        
        dt.fit(X_train, y_train)
        labels = dt.predict(features)
        
        mask = labels.reshape(h, w).astype(np.uint8) * 255
        return self._postprocess_mask(mask)
    
    def _sklearn_mlp(self, image: np.ndarray) -> np.ndarray:
        """Multi-layer Perceptron для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]
        
        X_train = features[train_indices]
        y_train = labels_train[train_indices]
        
        # Обучаем MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=self.params.get('hidden_layer_sizes', (100, 50)),
            activation='relu',
            solver='adam',
            random_state=42,
            max_iter=500
        )
        
        mlp.fit(X_train, y_train)
        labels = mlp.predict(features)
        
        mask = labels.reshape(h, w).astype(np.uint8) * 255
        return self._postprocess_mask(mask)
    
    def _sklearn_naive_bayes(self, image: np.ndarray) -> np.ndarray:
        """Naive Bayes для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]
        
        X_train = features[train_indices]
        y_train = labels_train[train_indices]
        
        # Обучаем Gaussian Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        labels = nb.predict(features)
        
        mask = labels.reshape(h, w).astype(np.uint8) * 255
        return self._postprocess_mask(mask)
    
    def _sklearn_lda(self, image: np.ndarray) -> np.ndarray:
        """Linear Discriminant Analysis для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]
        
        X_train = features[train_indices]
        y_train = labels_train[train_indices]
        
        # Обучаем LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        labels = lda.predict(features)
        
        mask = labels.reshape(h, w).astype(np.uint8) * 255
        return self._postprocess_mask(mask)
    
    def _create_training_labels(self, h: int, w: int) -> np.ndarray:
        """Создание меток для обучения."""
        labels_train = -np.ones(h * w)  # -1 означает непомеченный
        
        # Создаем маску для обучения
        train_mask = np.zeros((h, w), dtype=bool)
        
        # Центральная область - объект (класс 1)
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        cv2.rectangle(train_mask, 
                     (center_w - obj_size, center_h - obj_size),
                     (center_w + obj_size, center_h + obj_size),
                     True, -1)
        
        # Углы - фон (класс 0)
        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h)
        ]
        
        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True
        
        labels_train = train_mask.ravel().astype(int)
        # Непомеченные пиксели остаются -1
        
        return labels_train
    
    # ============ ОСТАЛЬНЫЕ МЕТОДЫ (заглушки для полноты API) ============
    
    def _sklearn_elliptic_envelope(self, image: np.ndarray) -> np.ndarray:
        """Elliptic Envelope для сегментации."""
        return self._sklearn_isolation_forest(image)
    
    def _sklearn_bayesian_gmm(self, image: np.ndarray) -> np.ndarray:
        """Bayesian GMM для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        bgmm = BayesianGaussianMixture(
            n_components=self.params.get('n_components', 10),
            random_state=42
        )
        
        labels = bgmm.fit_predict(features)
        mask = self._create_mask_from_labels(labels, (h, w))
        return self._postprocess_mask(mask)
    
    def _sklearn_ica_segmentation(self, image: np.ndarray) -> np.ndarray:
        """ICA для сегментации."""
        return self._sklearn_pca_segmentation(image)
    
    def _sklearn_isomap_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Isomap для сегментации."""
        return self._sklearn_tsne_segmentation(image)
    
    def _sklearn_spectral_embedding(self, image: np.ndarray) -> np.ndarray:
        """Spectral Embedding для сегментации."""
        return self._sklearn_spectral(image)
    
    def _sklearn_variational_gmm(self, image: np.ndarray) -> np.ndarray:
        """Variational GMM для сегментации."""
        return self._sklearn_bayesian_gmm(image)
    
    def _sklearn_density_based(self, image: np.ndarray) -> np.ndarray:
        """Density-based кластеризация."""
        return self._sklearn_dbscan(image)
    
    def _sklearn_hdbscan_emulation(self, image: np.ndarray) -> np.ndarray:
        """Эмуляция HDBSCAN."""
        return self._sklearn_optics(image)
    
    def _sklearn_graph_clustering(self, image: np.ndarray) -> np.ndarray:
        """Graph-based кластеризация."""
        return self._sklearn_spectral(image)
    
    def _sklearn_modularity_clustering(self, image: np.ndarray) -> np.ndarray:
        """Modularity-based кластеризация."""
        return self._sklearn_spectral(image)
    
    def _sklearn_self_training(self, image: np.ndarray) -> np.ndarray:
        """Self-training для сегментации."""
        return self._sklearn_random_forest(image)
    
    def _sklearn_semi_supervised(self, image: np.ndarray) -> np.ndarray:
        """Semi-supervised сегментация."""
        return self._sklearn_random_forest(image)
    
    def _sklearn_distance_matrix(self, image: np.ndarray) -> np.ndarray:
        """Distance matrix-based кластеризация."""
        return self._sklearn_spectral(image)
    
    def _sklearn_affinity_propagation(self, image: np.ndarray) -> np.ndarray:
        """Affinity Propagation."""
        features = self._extract_features(image)
        h, w = image.shape[:2]
        
        # Используем K-Means как альтернативу (Affinity Propagation требователен)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(features)
        
        mask = self._create_mask_from_labels(labels, (h, w))
        return self._postprocess_mask(mask)
    
    def _sklearn_qda(self, image: np.ndarray) -> np.ndarray:
        """Quadratic Discriminant Analysis."""
        return self._sklearn_lda(image)
    
    def _sklearn_texture_clustering(self, image: np.ndarray) -> np.ndarray:
        """Текстурная кластеризация."""
        return self._sklearn_color_spatial_clustering(image)
    
    def _sklearn_superpixel_clustering(self, image: np.ndarray) -> np.ndarray:
        """Кластеризация суперпикселей."""
        return self._sklearn_color_spatial_clustering(image)
    
    def _sklearn_hierarchical_kmeans(self, image: np.ndarray) -> np.ndarray:
        """Иерархический K-Means."""
        return self._sklearn_agglomerative(image)
    
    def _sklearn_pca_kmeans(self, image: np.ndarray) -> np.ndarray:
        """PCA + K-Means."""
        return self._sklearn_pca_segmentation(image)
    
    def _sklearn_gmm(self, img: np.ndarray) -> np.ndarray:
        """Gaussian Mixture Models."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Подготовка признаков
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack([
            gray.ravel(),
            x_coords.ravel() / w,
            y_coords.ravel() / h
        ])
        
        # Применяем GMM
        n_components = self.params.get('n_components', 3)
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_agglomerative(self, img: np.ndarray) -> np.ndarray:
        """Agglomerative Clustering."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Для производительности сэмплируем
        sample_size = min(1000, h * w)
        indices = np.random.choice(h * w, sample_size, replace=False)
        
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack([
            gray.ravel()[indices],
            x_coords.ravel()[indices] / w,
            y_coords.ravel()[indices] / h
        ])
        
        # Применяем Agglomerative Clustering
        n_clusters = self.params.get('n_clusters', 3)
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels_sample = agg.fit_predict(features)
        
        # Интерполируем на все пиксели
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, labels_sample)
        
        all_features = np.column_stack([
            gray.ravel(),
            x_coords.ravel() / w,
            y_coords.ravel() / h
        ])
        
        labels = knn.predict(all_features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_spectral(self, img: np.ndarray) -> np.ndarray:
        """Spectral Clustering."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Для производительности - сильное сэмплирование
        sample_size = min(500, h * w)
        indices = np.random.choice(h * w, sample_size, replace=False)
        
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack([
            gray.ravel()[indices],
            x_coords.ravel()[indices] / w,
            y_coords.ravel()[indices] / h
        ])
        
        # Применяем Spectral Clustering
        n_clusters = self.params.get('n_clusters', 3)
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
        labels_sample = spectral.fit_predict(features)
        
        # Интерполируем
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, labels_sample)
        
        all_features = np.column_stack([
            gray.ravel(),
            x_coords.ravel() / w,
            y_coords.ravel() / h
        ])
        
        labels = knn.predict(all_features)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_isolation_forest(self, img: np.ndarray) -> np.ndarray:
        """Isolation Forest для сегментации."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Подготовка признаков
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack([
            gray.ravel(),
            x_coords.ravel() / w,
            y_coords.ravel() / h
        ])
        
        # Применяем Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        labels = iso_forest.fit_predict(features)
        
        # Аномалии = объект
        mask = (labels == -1).reshape(h, w)
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_random_forest(self, img: np.ndarray) -> np.ndarray:
        """Random Forest для сегментации."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Создаем метки для обучения
        labels_train = -np.ones(h * w)
        train_mask = np.zeros((h, w), dtype=bool)
        
        # Центральная область - объект
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        train_mask[center_h-obj_size:center_h+obj_size, 
                  center_w-obj_size:center_w+obj_size] = True
        labels_train[train_mask.ravel()] = 1
        
        # Углы - фон
        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h)
        ]
        
        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True
            labels_train[train_mask[y1:y2, x1:x2].ravel()] = 0
        
        # Подготовка признаков
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack([
            gray.ravel(),
            x_coords.ravel() / w,
            y_coords.ravel() / h
        ])
        
        # Обучаем Random Forest
        train_indices = labels_train >= 0
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(features[train_indices], labels_train[train_indices])
        
        # Предсказываем
        labels = rf.predict(features)
        mask = labels.reshape(h, w)
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_svm(self, img: np.ndarray) -> np.ndarray:
        """SVM для сегментации."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Создаем метки для обучения (как в Random Forest)
        labels_train = -np.ones(h * w)
        train_mask = np.zeros((h, w), dtype=bool)
        
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        train_mask[center_h-obj_size:center_h+obj_size, 
                  center_w-obj_size:center_w+obj_size] = True
        labels_train[train_mask.ravel()] = 1
        
        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h)
        ]
        
        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True
            labels_train[train_mask[y1:y2, x1:x2].ravel()] = 0
        
        # Подготовка признаков
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack([
            gray.ravel(),
            x_coords.ravel() / w,
            y_coords.ravel() / h
        ])
        
        # Обучаем SVM
        train_indices = labels_train >= 0
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(features[train_indices], labels_train[train_indices])
        
        # Предсказываем
        labels = svm.predict(features)
        mask = labels.reshape(h, w)
        
        return mask.astype(np.uint8) * 255
    
    def _sklearn_pca_segmentation(self, img: np.ndarray) -> np.ndarray:
        """PCA-based сегментация."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Подготовка признаков
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack([
            gray.ravel(),
            x_coords.ravel() / w,
            y_coords.ravel() / h
        ])
        
        # Применяем PCA
        pca = PCA(n_components=2, random_state=42)
        transformed = pca.fit_transform(features)
        
        # Кластеризуем в новом пространстве
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(transformed)
        
        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        
        return mask.astype(np.uint8) * 255
    
    # ============ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ============
    
    def _create_mask_from_labels(self, labels: np.ndarray, shape: Tuple) -> np.ndarray:
        """Создание бинарной маски из меток."""
        labels_2d = labels.reshape(shape)
        unique_labels = np.unique(labels)
        
        # Исключаем шум (-1) если есть
        valid_labels = unique_labels[unique_labels != -1]
        
        if len(valid_labels) == 0:
            return np.zeros(shape, dtype=bool)
        
        # Находим самый большой кластер как фон
        label_sizes = [np.sum(labels_2d == label) for label in valid_labels]
        bg_label = valid_labels[np.argmax(label_sizes)]
        
        # Создаем маску (все кроме фона)
        mask = labels_2d != bg_label
        
        return mask

    # ============================================================================
    # ДОБАВИТЬ/ОБНОВИТЬ МЕТОДЫ В КЛАСС SklearnSegmenter:
    # ============================================================================

    def _sklearn_floodfill(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        FloodFill сегментация (scikit-image).
        Заполняет область, начиная с заданной точки, пока интенсивность
        отличается не более чем на допуск.
        
        Args:
            img: Входное изображение (RGB или grayscale)
        
        Returns:
            Бинарная маска (0-255)
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img
            
            h, w = gray.shape
            seed = self.params.get('seed', (w//2, h//2))
            tolerance = self.params.get('tolerance', 0.1)
            
            # Используем flood из skimage
            mask = segmentation.flood(gray, seed_point=seed[::-1], tolerance=tolerance)
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"FloodFill failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)


    def _sklearn_active_contour(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Active Contour (Snakes) сегментация.
        Инициализирует замкнутый контур и деформирует его под действием
        внутренних и внешних сил до равновесия.
        
        Args:
            img: Входное изображение
        
        Returns:
            Бинарная маска внутри контура
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img
            
            gray_norm = img_as_float(gray)
            h, w = gray_norm.shape
            
            # Создаём начальный контур (окружность в центре)
            center_x, center_y = w // 2, h // 2
            radius = min(center_x, center_y) // 2
            s = np.linspace(0, 2 * np.pi, 100)
            init = np.array([
                center_x + radius * np.cos(s),
                center_y + radius * np.sin(s)
            ]).T
            
            # Параметры
            alpha = self.params.get('alpha', 0.015)
            beta = self.params.get('beta', 10)
            gamma = self.params.get('gamma', 0.001)
            w_edge = self.params.get('w_edge', 1)
            w_line = self.params.get('w_line', 0)
            max_iter = self.params.get('max_iter', 1000)
            
            # Применяем active_contour
            snake = active_contour(
                gaussian(gray_norm, 3),
                init,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                w_edge=w_edge,
                w_line=w_line,
                max_num_iter=max_iter
            )
            
            # Создаём маску из контура
            mask = np.zeros((h, w), dtype=bool)
            rr, cc = polygon(snake[:, 1], snake[:, 0], mask.shape)
            mask[rr, cc] = True
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"Active Contour failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)


    def _sklearn_gvf_contour(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Gradient Vector Flow (GVF) контуры.
        Вычисляет векторное поле, распространяющее информацию о градиентах.
        
        Args:
            img: Входное изображение
        
        Returns:
            Бинарная маска
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img
            
            gray_norm = img_as_float(gray)
            
            # Вычисляем градиенты
            grad_x = filters.sobel_h(gray_norm)
            grad_y = filters.sobel_v(gray_norm)
            
            # Параметры GVF
            mu = self.params.get('mu', 0.1)
            iterations = self.params.get('iterations', 50)
            
            # Итеративно вычисляем GVF поле
            u = grad_x.copy()
            v = grad_y.copy()
            
            for _ in range(iterations):
                laplacian_u = filters.laplace(u)
                laplacian_v = filters.laplace(v)
                edge_map = grad_x**2 + grad_y**2
                
                u = u + mu * laplacian_u - edge_map * (u - grad_x)
                v = v + mu * laplacian_v - edge_map * (v - grad_y)
            
            # Величина GVF
            gvf_mag = np.sqrt(u**2 + v**2)
            
            # Пороговое разделение
            threshold = np.percentile(gvf_mag, 70)
            mask = gvf_mag > threshold
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"GVF Contour failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)


    def _sklearn_morphological_snakes(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Морфологические змеи (Morphological Geodesic Active Contours).
        Итеративно расширяет/сужает маску на основе градиента изображения.
        
        Args:
            img: Входное изображение
        
        Returns:
            Бинарная маска
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img
            
            gray_norm = img_as_float(gray)
            h, w = gray_norm.shape
            
            # Начальный уровень (прямоугольник в центре)
            init_level_set = np.zeros(gray_norm.shape, dtype=np.int8)
            init_level_set[h//4:3*h//4, w//4:3*w//4] = 1
            
            # Параметры
            iterations = self.params.get('iterations', 50)
            smoothing = self.params.get('smoothing', 1)
            threshold = self.params.get('threshold', 0.5)
            balloon = self.params.get('balloon', 1)
            
            # Применяем морфологические змеи
            mask = morphological_geodesic_active_contour(
                gray_norm,
                iterations,
                init_level_set=init_level_set,
                smoothing=smoothing,
                threshold=threshold,
                balloon=balloon
            )
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"Morphological Snakes failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)


    def _sklearn_chan_vese(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Chan-Vese активные контуры (без градиентов).
        Энергетическая модель, разделяющая изображение на две области
        с минимальной внутрирегиональной дисперсией.
        
        Args:
            img: Входное изображение
        
        Returns:
            Бинарная маска
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img
            
            gray_norm = img_as_float(gray)
            h, w = gray_norm.shape
            
            # Начальный уровень
            init_level_set = np.zeros(gray_norm.shape, dtype=np.int8)
            init_level_set[h//4:3*h//4, w//4:3*w//4] = 1
            
            # Параметры
            mu = self.params.get('mu', 0.25)
            lambda1 = self.params.get('lambda1', 1.0)
            lambda2 = self.params.get('lambda2', 1.0)
            tol = self.params.get('tol', 1e-3)
            max_iter = self.params.get('max_iter', 100)
            
            # Применяем Chan-Vese
            mask = morphological_chan_vese(
                gray_norm,
                max_iter,
                init_level_set=init_level_set,
                smoothing=1,
                lambda1=lambda1,
                lambda2=lambda2
            )
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"Chan-Vese failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)


    def _sklearn_watershed(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Watershed (водораздел) сегментация.
        Использует маркеры для разделения объектов на основе градиента.
        
        Args:
            img: Входное изображение
        
        Returns:
            Бинарная маска
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img
            
            gray_norm = img_as_float(gray)
            
            # Создаём маркеры
            markers = np.zeros_like(gray, dtype=np.uint8)
            h, w = gray.shape
            
            # Маркеры на основе процентилей
            markers[gray < np.percentile(gray, 25)] = 1  # Фон
            markers[gray > np.percentile(gray, 75)] = 2  # Объект
            
            # Вычисляем градиент
            gradient = filters.sobel(gray_norm)
            
            # Применяем watershed
            segmentation = watershed(gradient, markers=markers)
            
            # Создаём маску (объект = маркер 2)
            mask = segmentation == 2
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"Watershed failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)


    def _sklearn_random_walker(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Random Walker сегментация.
        На основе маркеров решает задачу на графе о принадлежности пикселей.
        
        Args:
            img: Входное изображение
        
        Returns:
            Бинарная маска
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img
            
            gray_norm = img_as_float(gray)
            h, w = gray.shape
            
            # Создаём маркеры
            markers = np.zeros(gray.shape, dtype=np.uint8)
            
            # Центральная область - объект (маркер 2)
            markers[h//4:3*h//4, w//4:3*w//4] = 2
            
            # Углы - фон (маркер 1)
            corner_size = min(h, w) // 8
            markers[:corner_size, :corner_size] = 1
            markers[:corner_size, -corner_size:] = 1
            markers[-corner_size:, :corner_size] = 1
            markers[-corner_size:, -corner_size:] = 1
            
            # Параметры
            beta = self.params.get('beta', 130)
            mode = self.params.get('mode', 'cg_mg')
            
            # Применяем Random Walker
            labels = random_walker(gray_norm, markers, beta=beta, mode=mode)
            
            # Создаём маску (объект = маркер 2)
            mask = labels == 2
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"Random Walker failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)


    def _sklearn_quickshift(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Quickshift сегментация.
        Mode-seeking алгоритм для сегментации в пространстве признаков.
        
        Args:
            img: Входное изображение (RGB)
        
        Returns:
            Бинарная маска
        """
        try:
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            
            # Параметры
            kernel_size = self.params.get('kernel_size', 3)
            max_dist = self.params.get('max_dist', 6)
            ratio = self.params.get('ratio', 0.5)
            
            # Применяем Quickshift
            segments = quickshift(
                img_rgb,
                kernel_size=kernel_size,
                max_dist=max_dist,
                ratio=ratio
            )
            
            # Находим самый большой сегмент (фон)
            unique, counts = np.unique(segments, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            
            # Создаём маску (все кроме фона)
            mask = segments != bg_label
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"Quickshift failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)


    def _sklearn_slic(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        SLIC (Simple Linear Iterative Clustering).
        Суперпиксельная сегментация на основе пространственной и цветовой близости.
        
        Args:
            img: Входное изображение (RGB)
        
        Returns:
            Бинарная маска
        """
        try:
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            
            # Параметры
            n_segments = self.params.get('n_segments', 100)
            compactness = self.params.get('compactness', 10.0)
            max_iter = self.params.get('max_iter', 10)
            enforce_connectivity = self.params.get('enforce_connectivity', True)
            
            # Применяем SLIC
            segments = slic(
                img_rgb,
                n_segments=n_segments,
                compactness=compactness,
                max_num_iter=max_iter,
                enforce_connectivity=enforce_connectivity,
                start_label=0
            )
            
            # Находим самый большой сегмент (фон)
            unique, counts = np.unique(segments, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            
            # Создаём маску (все кроме фона)
            mask = segments != bg_label
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"SLIC failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)


    def _sklearn_felzenszwalb(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Felzenszwalb сегментация.
        Графовая сегментация на основе минимального остовного дерева.
        
        Args:
            img: Входное изображение (RGB)
        
        Returns:
            Бинарная маска
        """
        try:
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            
            # Параметры
            scale = self.params.get('scale', 100)
            sigma = self.params.get('sigma', 0.8)
            min_size = self.params.get('min_size', 50)
            
            # Применяем Felzenszwalb
            segments = felzenszwalb(
                img_rgb,
                scale=scale,
                sigma=sigma,
                min_size=min_size
            )
            
            # Находим самый большой сегмент (фон)
            unique, counts = np.unique(segments, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            
            # Создаём маску (все кроме фона)
            mask = segments != bg_label
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"Felzenszwalb failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)


    def _sklearn_grabcut(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        GrabCut сегментация (эмуляция через Random Forest).
        Имитация интерактивной сегментации с использованием маркеров.
        
        Args:
            img: Входное изображение (RGB)
        
        Returns:
            Бинарная маска переднего плана
        """
        try:
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            
            h, w = img_rgb.shape[:2]
            
            # Создаём начальную маску с маркерами
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Прямоугольник в центре - вероятный передний план
            rect = self.params.get('rect', (w//4, h//4, w//2, h//2))
            x, y, rw, rh = rect
            mask[y:y+rh, x:x+rw] = 3  # Вероятный передний план
            
            # Углы - определённый фон
            corner_size = min(h, w) // 8
            mask[:corner_size, :corner_size] = 0
            mask[:corner_size, -corner_size:] = 0
            mask[-corner_size:, :corner_size] = 0
            mask[-corner_size:, -corner_size:] = 0
            
            # Подготовка признаков
            pixels = img_rgb.reshape(-1, 3).astype(np.float32) / 255.0
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            coords = np.column_stack([
                x_coords.ravel() / w,
                y_coords.ravel() / h
            ])
            features = np.hstack([pixels, coords])
            
            # Выбираем пиксели для обучения
            train_mask = (mask.ravel() == 0) | (mask.ravel() == 3)
            X_train = features[train_mask]
            y_train = (mask.ravel()[train_mask] == 3).astype(int)
            
            # Обучаем Random Forest
            n_estimators = self.params.get('n_estimators', 50)
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            # Предсказываем для всех пикселей
            labels = rf.predict(features)
            mask_result = labels.reshape(h, w)
            
            return mask_result.astype(np.uint8) * 255
            
        except Exception as e:
            warnings.warn(f"GrabCut failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)


# Пример использования SklearnSegmenter
def demo_sklearn_segmenter() -> SklearnSegmenter:
    """Демонстрация работы SklearnSegmenter."""
    if not SKIMAGE_AVAILABLE:
        print("scikit-image не установлен. Установите: pip install scikit-image")
        return
    
    # Загрузка тестового изображения
    url = "https://upload.wikimedia.org/wikipedia/commons/7/7d/Colorful_spring_garden.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_np = np.array(img)

    # Создаем тестовое изображение
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    radius = h // 4
    
    # Круг на фоне градиента
    circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    gradient = np.linspace(0, 1, w).reshape(1, -1).repeat(h, axis=0)
    test_image = gradient.copy()
    test_image[circle] = 0.8
    
    print("Демонстрация SklearnSegmenter")
    print("=" * 50)
    
    # Тестируем разные методы
    methods_to_test = [
        ("global_thresholding", {"threshold": 0.5}),
        ("otsu_thresholding", {}),
        ("canny_edge", {"sigma": 1.0}),
        ("kmeans_segmentation", {"k": 2}),
        ("watershed", {}),
        ("slic", {"n_segments": 50}),
        ("kmeans", {"n_clusters": 3}),
        ("dbscan", {"eps": "auto", "min_samples": "auto"}),
        ("meanshift", {"bandwidth": None}),
        ("gmm", {"n_components": 3}),
        ("random_forest", {"n_estimators": 50}),
        ("svm", {"C": 1.0, "kernel": "rbf"}),
        ("isolation_forest", {"n_estimators": 100}),
        ("pca_segmentation", {"n_components": 3}),
        ("color_spatial_clustering", {"color_weight": 0.7, "spatial_weight": 0.3}),
    ]
    
    fig, axes = plt.subplots(2, 8, figsize=(16, 8))
    axes = axes.flatten()
    
    # Оригинальное изображение
    # axes[0].imshow(img_np)
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    for idx, (method_name, params) in enumerate(methods_to_test):
        try:
            print(f"Тестирование метода: {method_name}")
            
            # Создаем сегментатор
            segmenter = SklearnSegmenter(method=method_name, **params)
            
            # Выполняем сегментацию
            # mask = segmenter.segment(img_np)
            mask = segmenter.segment(test_image)
            
            # Отображаем результат
            axes[idx + 1].imshow(mask, cmap='gray')
            axes[idx + 1].set_title(f"{method_name}")
            axes[idx + 1].axis('off')
            
            print(f"  ✓ Успешно: маска размера {mask.shape}, "
                  f"уникальных значений: {np.unique(mask)}")
            
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            axes[idx + 1].axis('off')
            axes[idx + 1].set_title(f"{method_name}\n(ошибка)")
    
    plt.suptitle("SklearnSegmenter: Разные методы сегментации", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Демонстрация segment_with_mask
    print("\nДемонстрация segment_with_mask:")
    segmenter = SklearnSegmenter(method="kmeans", n_clusters=3)
    visualization, mask = segmenter.segment_with_mask(img_np)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')
    
    axes[2].imshow(visualization)
    axes[2].set_title("Visualization")
    axes[2].axis('off')
    
    plt.suptitle("SklearnSegmenter.segment_with_mask()", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return segmenter


if __name__ == "__main__":
    segmenter = demo_sklearn_segmenter()
    print("\n✅ Демонстрация завершена успешно!")