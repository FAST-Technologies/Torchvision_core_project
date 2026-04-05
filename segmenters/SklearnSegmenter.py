# segmenters/SklearnSegmenter.py

# Импорт основных библиотек
from segmenters.BaseSegmenter import BaseSegmenter
from typing import Union, Tuple, Dict, Any, Optional, Callable, Literal
import numpy as np
import warnings
from collections import deque
from PIL import Image
import time

# Импорт scikit-learn компонентов
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    MeanShift,
    OPTICS,
    AgglomerativeClustering,
    SpectralClustering,
    Birch,
    MiniBatchKMeans,
)
from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    LocalOutlierFactor,
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    StandardScaler,
)
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter, laplace, sobel, prewitt
from skimage.util import img_as_float, img_as_ubyte

# Импорт scikit-image компонентов
import skimage
from skimage import (
    filters,
    segmentation,
    feature,
    color,
)
from skimage.draw import polygon
from skimage.filters import (
    threshold_otsu,
    threshold_local,
    threshold_niblack,
    threshold_sauvola,
    gaussian,
    sobel,
    prewitt,
    roberts,
    scharr,
    laplace,
    farid,
    butterworth,
)
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk,
    square,
    dilation,
    erosion,
    opening,
    closing,
    white_tophat,
    black_tophat,
    skeletonize,
    thin,
    remove_small_objects,
    remove_small_holes,
)
from skimage.segmentation import (
    felzenszwalb,
    slic,
    quickshift,
    watershed,
    random_walker,
    active_contour,
    morphological_chan_vese,
    morphological_geodesic_active_contour,
    mark_boundaries,
)
from skimage.util import img_as_ubyte, img_as_float
import cv2
import torch
from typing_extensions import TypeAlias

SKIMAGE_AVAILABLE = True

# Определение типов для изображений
ImagePath: TypeAlias = str
NumpyImage: TypeAlias = np.ndarray
PILImage: TypeAlias = Image.Image
TorchImage: TypeAlias = torch.Tensor
ImageInput: TypeAlias = Union[ImagePath, NumpyImage, PILImage, TorchImage]

# Типы для цветовых пространств
ColorSpace = Literal["RGB", "BGR", "GRAY", "L"]
ColorChannel = Literal[1, 3]
OverlayColor: TypeAlias = Tuple[int, int, int]

# Типы для масок
Mask: TypeAlias = np.ndarray
BinaryMask: TypeAlias = np.ndarray  # shape: (H, W), dtype: uint8, значения: 0 или 255
ProbabilityMask: TypeAlias = np.ndarray  # shape: (H, W), dtype: float32, значения: 0-1

# Тип для метрик
MetricsDict: TypeAlias = Dict[str, float]


class SklearnSegmenter(BaseSegmenter):
    """
    Класс для сегментации изображений с использованием scikit-learn и scikit-image.
    Все реализации сделаны без использования OpenCV или специализированных
    библиотек для обработки изображений. Поддерживает как классические методы (пороговые, граничные), так и методы на основе кластеризации,
    активных контуров и графов.
    """

    def __init__(self, method: str = "global_thresholding", **kwargs) -> None:
        super().__init__()
        self.method: str = method
        self.params: Dict[str, Any] = kwargs
        self.model_name: str = f"Sklearn_{method}"
        self._setup_methods()
        self._scaler: StandardScaler = StandardScaler()

        self._needs_gray: bool = method in [
            "global_thresholding",
            "adaptive_thresholding",
            "otsu_thresholding",
            "threshold_niblack",
            "threshold_sauvola",
            "threshold_bernsen",
            "threshold_phansalkar",
            "threshold_kittler_illingworth",
            "threshold_entropy_kapur",
            "threshold_triangle",
            "threshold_multi_otsu",
            "threshold_percentile",
            "threshold_local_contrast",
            "region_growing",
            "split_and_merge",
            "sobel_edge",
            "canny_edge",
            "prewitt_edge",
            "scharr_edge",
            "roberts_cross_edge",
            "log_edge",
            "dog_edge",
            "marr_hildreth_edge",
            "gradient_magnitude_direction",
            "phase_congruency_edge",
            "active_contour",
            "gvf_contour",
            "watershed",
            "meanshift",
            "grabcut",
            "floodfill",
            "morphological_snakes",
            "chan_vese",
            "random_walker",
        ]

    def _setup_methods(self) -> None:
        """
        Сегментация с использованием scikit методов.

        Args:
            **kwargs: Параметры метода
        """
        self.methods: Dict[str, Callable[..., Tuple[np.ndarray, Dict[str, Any]]]] = {
            # ============ ПОРОГОВЫЕ МЕТОДЫ СЕГМЕНТАЦИИ ============
            "global_thresholding": self._sklearn_global_thresholding,
            "adaptive_thresholding": self._sklearn_adaptive_thresholding,
            "otsu_thresholding": self._sklearn_otsu_thresholding,
            "threshold_niblack": self._sklearn_threshold_niblack,
            "threshold_sauvola": self._sklearn_threshold_sauvola,
            "threshold_bernsen": self._sklearn_threshold_bernsen,
            "threshold_phansalkar": self._sklearn_threshold_phansalkar,
            "threshold_kittler_illingworth": self._sklearn_threshold_kittler_illingworth,
            "threshold_entropy_kapur": self._sklearn_threshold_entropy_kapur,
            "threshold_triangle": self._sklearn_threshold_triangle,
            "threshold_multi_otsu": self._sklearn_threshold_multi_otsu,
            "threshold_percentile": self._sklearn_threshold_percentile,
            "threshold_local_contrast": self._sklearn_threshold_local_contrast,
            # ============ КРАЕВЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "sobel_edge": self._sklearn_sobel_edge,
            "canny_edge": self._sklearn_canny_edge,
            "prewitt_edge": self._sklearn_prewitt_edge,
            "scharr_edge": self._sklearn_scharr_edge,
            "roberts_cross_edge": self._sklearn_roberts_cross_edge,
            "log_edge": self._sklearn_log_edge,
            "dog_edge": self._sklearn_dog_edge,
            "marr_hildreth_edge": self._sklearn_marr_hildreth_edge,
            "gradient_magnitude_direction": self._sklearn_gradient_magnitude_direction,
            "phase_congruency_edge": self._sklearn_phase_congruency_edge,
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

        if self.method not in self.methods:
            raise ValueError(
                f"Неизвестный метод: {self.method}. "
                f"Доступные методы: {list(self.methods.keys())}"
            )

    def _get_param(self, key: str, default: Any, **kwargs) -> Any:
        """
        Универсальный геттер параметров: приоритет kwargs > self.params > default
        """
        if key in kwargs:
            return kwargs[key]
        if key in self.params:
            return self.params[key]
        return default

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Нормализация изображения к [0, 1] для skimage"""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)

    def _ensure_uint8_mask(self, mask: np.ndarray) -> np.ndarray:
        """Гарантирует возврат маски в uint8 [0, 255]"""
        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                return (mask * 255).astype(np.uint8)
            return mask.astype(np.uint8)
        return mask

    def preprocess_image(
        self,
        image: ImageInput,
        as_gray: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
    ) -> NumpyImage:
        """
        Переопределенная предобработка для гарантии использования BT.601.
        """
        result: NumpyImage = super().preprocess_image(
            image, as_gray=False, target_size=target_size, normalize=False
        )

        # Конвертация в серый, используя BT.601
        if as_gray and len(result.shape) == 3:
            result = self._to_gray_bt601(result)
            if result.max() <= 1.0:
                result = (result * 255).astype(np.uint8)

        if normalize and result.dtype != np.float32:
            result = result.astype(np.float32) / 255.0
        return result

    def segment(  # type: ignore[override]
        self, image: ImageInput, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Основной метод сегментации.

        Args:
            image: Входное изображение (RGB, grayscale или любой формат)

        Returns:
            np.ndarray: Бинарная маска сегментации (0-255)
        """
        try:
            # image может быть str/PIL/Tensor, preprocess_image обработает это
            img_processed: np.ndarray = self.preprocess_image(
                image, as_gray=self._needs_gray
            )

            if self.method not in self.methods:
                raise ValueError(f"Метод {self.method} не реализован")

            mask, info = self.methods[self.method](img_processed, **kwargs)

            mask = self._ensure_uint8_mask(mask)
            if self.params.get("postprocess", True) and self.method not in [
                "canny_edge",
                "sobel_edge",
            ]:
                mask = self._postprocess_mask(mask)

            return mask
        except Exception as e:
            warnings.warn(
                f"Ошибка в методе {self.method}: {e}. Возвращаем пустую маску.",
                RuntimeWarning,
            )
            h, w = img_processed.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

    def segment_and_evaluate(  # type: ignore[override]
        self,
        image: ImageInput,
        gt_mask: BinaryMask,  # имя как в базе
        threshold: float = 0.5,
        **segment_kwargs: Any,  # имя как в базе
    ) -> Tuple[MetricsDict, BinaryMask]:  # типы как в базе
        """
        Сегментация с немедленным вычислением метрик.

        Args:
            image: Входное изображение
            ground_truth: Ground truth маска
            threshold: Порог для бинаризации

        Returns:
            Tuple[Dict[str, float], np.ndarray]: Метрики и предсказанная маска
        """
        from metrics.SegmentationMetrics import SegmentationMetrics

        # Выполняем сегментацию
        img_array: np.ndarray = self.preprocess_image(image)
        mask = self.segment(img_array, **segment_kwargs)  # type: ignore[arg-type]

        # Вычисляем метрики
        metrics = SegmentationMetrics.calculate_all_metrics(
            pred_mask=mask,
            gt_mask=gt_mask,
            threshold=threshold,
            include_hausdorff=True,
        )

        return metrics, mask

    def segment_with_mask(  # type: ignore[override]
        self, image: np.ndarray, alpha: float = 0.9, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сегментация с возвратом визуализации и маски.

        Args:
            image: Входное изображение

        Returns:
            Tuple[np.ndarray, np.ndarray]: Визуализация и маска
        """
        image = self.preprocess_image(image)
        print(f"Image after sklearn preprocessing with mask: {image}")
        mask = self.segment(image, **kwargs)

        # Создаем визуализацию
        if len(image.shape) == 2:
            overlay = np.stack([image] * 3, axis=-1)
            original_rgb = np.stack([image] * 3, axis=-1)
        else:
            overlay = image.copy()
            original_rgb = image

        # Красный цвет для маски
        overlay[mask > 127] = [255, 0, 0]

        # Смешивание
        result = (alpha * overlay + (1 - alpha) * original_rgb).astype(np.uint8)

        print(f"Mask after sklearn segment_with_mask: {mask}")
        print(f"Result after sklearn segment_with_mask: {result}")

        return result, mask

    # ============ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ============

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
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
        spatial_features = np.stack(
            [
                x_coords.ravel() / w,
                y_coords.ravel() / h,
                (x_coords.ravel() * y_coords.ravel()) / (w * h),
            ],
            axis=1,
        )

        # Текстура (упрощенная)
        if SKIMAGE_AVAILABLE:
            grad_x = filters.sobel_h(gray)
            grad_y = filters.sobel_v(gray)
        else:
            # Реализация Собеля на numpy
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            grad_x = signal.convolve2d(gray, kernel_x, mode="same", boundary="symm")
            grad_y = signal.convolve2d(gray, kernel_y, mode="same", boundary="symm")

        texture_features = np.stack(
            [
                grad_x.ravel(),
                grad_y.ravel(),
                np.sqrt(grad_x.ravel() ** 2 + grad_y.ravel() ** 2),
            ],
            axis=1,
        )

        # Комбинируем все признаки
        features = np.hstack(
            [
                color_features,
                spatial_features,
                texture_features,
            ]
        )

        # Масштабирование
        if features.shape[0] > 0:
            features = self._scaler.fit_transform(features)

        return features

    def _to_gray_bt601(self, img: np.ndarray) -> np.ndarray:
        """Конвертация в серый по стандарту BT.601 (как в OpenCV/Torch)."""
        if len(img.shape) != 3 or img.shape[2] != 3:
            return img
        # Веса ITU-R BT.601
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    def _create_mask_from_labels(self, labels: np.ndarray, shape: Tuple) -> np.ndarray:
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
        label_sizes = [np.sum(labels_2d == label1) for label1 in valid_labels]
        bg_label = valid_labels[np.argmax(label_sizes)]

        # Создаем маску (все кроме фона) — uint8 0/255
        mask = (labels_2d != bg_label).astype(np.uint8) * 255

        return mask

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
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
            binary = remove_small_objects(
                binary, min_size=self.params.get("min_area", 100)
            )

            # Заполнение дыр
            binary = remove_small_holes(
                binary, area_threshold=self.params.get("min_area", 100)
            )

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
                if size < self.params.get("min_area", 100):
                    binary[labeled == i + 1] = False

        return binary.astype(np.uint8) * 255

    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    # ============ ПОРОГОВЫЕ МЕТОДЫ ============

    def _sklearn_global_thresholding(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Глобальная пороговая сегментация.

        Применяет фиксированный порог ко всему изображению.
        Все пиксели яркостью выше порога становятся белыми (объект), остальные — черными (фон).

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            Бинарная маска (0/255).
        """
        gray = img

        # Нормализация к [0, 1] для корректной работы порога 0.5
        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0
        else:
            gray = gray.astype(np.float32)
        # print(f"Gray after Sklearn_thresholding_global: {gray}")

        # gray = self._normalize_for_skimage(img) if img.max() > 1.0 else img.astype(np.float32)

        start_time = time.time()
        threshold = self.params.get("threshold", 0.5, **kwargs)
        mask = gray > threshold
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "global_thresholding_sklearn",
            "parameters": kwargs,
            "execution_time": exec_time,
            "threshold": threshold,
        }

        # print(f"Mask after Sklearn_thresholding_global: {mask}")
        # print(f"Info after Sklearn_thresholding_global: {info}")

        return mask, info

    def _sklearn_adaptive_thresholding(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Адаптивная пороговая сегментация (Gaussian).

        Вычисляет локальный порог для каждой области изображения.
        Особенно эффективна при неравномерном освещении.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        gray = img

        # Нормализация к [0, 1] для корректной работы порога 0.5
        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0
        else:
            gray = gray.astype(np.float32)

        # print(f"Gray after Sklearn_thresholding_adaptive: {gray}")

        start_time = time.time()
        block_size = self.params.get("block_size", 11)
        C = self.params.get("C", 2)
        adaptive_thresh = threshold_local(
            gray, block_size=block_size, offset=C / 255.0, method="gaussian"
        )
        mask = gray > adaptive_thresh
        mask = (mask * 255).astype(np.uint8)
        exec_time = time.time() - start_time

        info = {
            "method": "adaptive_thresholding_sklearn",
            "parameters": kwargs,
            "execution_time": exec_time,
            "threshold": adaptive_thresh,
        }

        # print(f"Mask after Sklearn_thresholding_adaptive: {mask}")
        # print(f"Info after Sklearn_thresholding_adaptive: {info}")

        return mask, info

    def _sklearn_otsu_thresholding(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Автоматическая бинаризация по методу Оцу.

        Находит оптимальный порог, максимизирующий межклассовую дисперсию между фоном и объектом.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        gray = img

        # Нормализация к [0, 1] для корректной работы порога 0.5
        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0
        else:
            gray = gray.astype(np.float32)

        # print(f"Gray after Sklearn_thresholding_otsu: {gray}")

        start_time = time.time()
        thresh = threshold_otsu(gray)
        mask = gray > thresh

        mask = (mask * 255).astype(np.uint8)
        exec_time = time.time() - start_time

        info = {
            "method": "otsu_thresholding_sklearn",
            "parameters": kwargs,
            "execution_time": exec_time,
            "threshold": thresh,
        }

        # print(f"Mask after Sklearn_thresholding_otsu: {mask}")
        # print(f"Info after Sklearn_thresholding_otsu: {info}")

        return mask, info

    def _sklearn_threshold_niblack(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Адаптивная пороговая обработка по Ниблаку.

        Порог вычисляется как: T = μ + k·σ, где μ и σ — локальное среднее и СКО.
        Хорошо работает на изображениях с шумом и градиентом освещения.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        gray = img

        # Нормализация к [0, 1] для корректной работы порога 0.5
        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0
        else:
            gray = gray.astype(np.float32)

        # print(f"Gray after Sklearn_thresholding_niblack: {gray}")

        start_time = time.time()
        window_size = self.params.get("window_size", 15)
        k = self.params.get("k", -0.2)
        # func_kwargs = {key: val for key, val in kwargs.items() if key not in ['window_size', 'k']}
        thresh = threshold_niblack(gray, window_size=window_size, k=k)
        mask = gray > thresh

        mask = (mask * 255).astype(np.uint8)
        exec_time = time.time() - start_time

        info = {
            "method": "niblack_thresholding_sklearn",
            "parameters": {"window_size": window_size, "k": k, **kwargs},
            "execution_time": exec_time,
            "threshold": thresh,
        }

        # print(f"Mask after Sklearn_thresholding_niblack: {mask}")
        # print(f"Info after Sklearn_thresholding_niblack: {info}")

        return mask, info

    def _sklearn_threshold_sauvola(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Улучшенная адаптивная пороговая обработка по Сауволе.

        Порог: T = μ·(1 + k·(σ/R - 1)), где R — динамический диапазон (обычно 128).
        Лучше Ниблака при очень низком контрасте.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        gray = img

        # Нормализация к [0, 1] для корректной работы порога 0.5
        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0
        else:
            gray = gray.astype(np.float32)

        # print(f"Gray after Sklearn_thresholding_sauvola: {gray}")

        start_time = time.time()
        window_size = self.params.get("window_size", 15)
        k = self.params.get("k", 0.2)
        r = self.params.get("r", 128)

        # Порог Сауволы из scikit-image
        # func_kwargs = {key: val for key, val in kwargs.items() if key not in ['window_size', 'k', 'r']}
        thresh = threshold_sauvola(gray, window_size=window_size, k=k, r=r)
        mask = gray > thresh

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "sauvola_thresholding_sklearn",
            "parameters": {"window_size": window_size, "k": k, "r": r, **kwargs},
            "execution_time": exec_time,
            "threshold": thresh,
        }

        # print(f"Mask after Sklearn_thresholding_sauvola: {mask}")
        # print(f"Info after Sklearn_thresholding_sauvola: {info}")

        return mask, info

    def _sklearn_threshold_bernsen(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Пороговая обработка по методу Бернсена.
        Локальный адаптивный порог на основе контраста в окне.
        T = (min + max) / 2, если контраст > порог, иначе фон.
        """
        start_time = time.time()
        window_size = self.params.get("window_size", 15)
        contrast_threshold = self.params.get("contrast_threshold", 0.1)

        # h, w = img.shape
        # pad = window_size // 2
        # img_padded = np.pad(img, pad, mode='reflect')
        # mask = np.zeros_like(img)

        # for i in range(h):
        #     for j in range(w):
        #         window = img_padded[i:i+window_size, j:j+window_size]
        #         local_min = np.min(window)
        #         local_max = np.max(window)
        #         contrast = local_max - local_min

        #         if contrast > contrast_threshold:
        #             threshold = (local_min + local_max) / 2
        #             mask[i, j] = 1 if img[i, j] > threshold else 0
        #         else:
        #             mask[i, j] = 0  # Фон при низком контрасте

        from scipy.ndimage import minimum_filter, maximum_filter

        local_min = minimum_filter(img, size=window_size)
        local_max = maximum_filter(img, size=window_size)
        local_contrast = local_max - local_min

        threshold_map = (local_min + local_max) / 2.0
        mask = np.where(local_contrast > contrast_threshold, img > threshold_map, False)

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "threshold_bernsen_sklearn",
            "parameters": {
                "window_size": window_size,
                "contrast_threshold": contrast_threshold,
                **kwargs,
            },
            "execution_time": exec_time,
        }
        return mask, info

    def _sklearn_threshold_phansalkar(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Пороговая обработка по методу Фансалкара.
        Улучшенная версия Ниблака для документов с низким контрастом.
        T = μ + k·σ·(σ/R) + m·(μ/128 - 1)
        """
        start_time = time.time()
        window_size = self.params.get("window_size", 15)
        k = self.params.get("k", 0.25)
        r = self.params.get("r", 0.5)
        m = self.params.get("m", 0.5)

        # h, w = img.shape
        # pad = window_size // 2
        # img_padded = np.pad(img, pad, mode='reflect')
        # mask = np.zeros_like(img)

        # for i in range(h):
        #     for j in range(w):
        #         window = img_padded[i:i+window_size, j:j+window_size]
        #         local_mean = np.mean(window)
        #         local_std = np.std(window)

        #         # Порог Фансалкара
        #         threshold = local_mean + k * local_std * (local_std / r) + m * (local_mean / 128 - 1)
        #         mask[i, j] = 1 if img[i, j] > threshold else 0

        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(img, size=window_size)
        local_sq_mean = uniform_filter(img**2, size=window_size)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))

        # Адаптированная формула для диапазона [0, 1]
        # Порог Фансалкара
        threshold_map = (
            local_mean + k * local_std * (local_std / r) + m * (local_mean / 0.5 - 1)
        )
        mask = img > threshold_map

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "threshold_phansalkar_sklearn",
            "parameters": {
                "window_size": window_size,
                "k": k,
                "r": r,
                "m": m,
                **kwargs,
            },
            "execution_time": exec_time,
        }
        return mask, info

    def _sklearn_threshold_kittler_illingworth(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Пороговая обработка по методу Киттлера-Иллингуорта.
        Минимизация ошибки классификации на основе гистограммы.
        """
        start_time = time.time()
        num_bins = self.params.get("num_bins", 256)

        # Гистограмма изображения
        hist, bin_edges = np.histogram(img.flatten(), bins=num_bins, range=(0, 1))
        hist = hist.astype(np.float64)

        # Нормализация гистограммы
        hist = hist / hist.sum()

        # Кумулятивные суммы
        cum_sum = np.cumsum(hist)
        cum_mean = np.cumsum(hist * np.arange(num_bins) / num_bins)

        total_mean = cum_mean[-1]
        cum_mean_sq = np.cumsum(hist * (np.arange(num_bins) / num_bins) ** 2)
        min_error = np.inf
        best_threshold = 0.5

        # Поиск оптимального порога
        for t in range(1, num_bins - 1):
            if cum_sum[t] < 1e-6 or (1 - cum_sum[t]) < 1e-6:
                continue

            # Статистики для фона и объекта
            w0 = cum_sum[t]
            w1 = 1 - w0
            mu0 = cum_mean[t] / w0
            mu1 = (total_mean - cum_mean[t]) / w1

            # Дисперсии
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
                best_threshold = t / num_bins

        mask = img > best_threshold
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "threshold_kittler_illingworth_sklearn",
            "parameters": {"num_bins": num_bins, **kwargs},
            "execution_time": exec_time,
            "threshold": best_threshold,
        }
        return mask, info

    def _sklearn_threshold_entropy_kapur(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Пороговая обработка на основе энтропии Капура.
        Максимизация суммы энтропий фона и объекта.
        """
        start_time = time.time()
        num_bins = self.params.get("num_bins", 256)

        # Гистограмма изображения
        hist, bin_edges = np.histogram(img.flatten(), bins=num_bins, range=(0, 1))
        hist = hist.astype(np.float64) + 1e-10  # Избегаем log(0)
        hist = hist / hist.sum()

        # Кумулятивная гистограмма и энтропия
        cum_hist = np.cumsum(hist)
        cum_entropy = np.cumsum(-hist * np.log(hist))

        max_entropy = -np.inf
        best_threshold = 0.5

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
                best_threshold = t / num_bins

        mask = img > best_threshold
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "threshold_entropy_kapur_sklearn",
            "parameters": {"num_bins": num_bins, **kwargs},
            "execution_time": exec_time,
            "threshold": best_threshold,
        }
        return mask, info

    def _sklearn_threshold_triangle(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Пороговая обработка треугольным методом.
        Геометрический метод для бимодальных гистограмм.
        Находит порог как точку максимального расстояния от линии пик-минимум.
        """
        start_time = time.time()
        num_bins = self.params.get("num_bins", 256)

        # Гистограмма изображения
        hist, bin_edges = np.histogram(img.flatten(), bins=num_bins, range=(0, 1))

        # Находим пик гистограммы
        peak_idx = np.argmax(hist)

        # Линия от пика до конца диапазона
        # x = np.arange(num_bins)
        y_peak = hist[peak_idx]
        y_end = hist[-1]

        # Уравнение линии: y = mx + b
        if num_bins - 1 != peak_idx:
            m = (y_end - y_peak) / (num_bins - 1 - peak_idx)
        else:
            m = 0

        # Находим точку максимального расстояния
        max_dist = 0
        best_threshold = peak_idx / num_bins

        for t in range(peak_idx + 1, num_bins):
            # Расстояние от точки до линии
            y_line = y_peak + m * (t - peak_idx)
            dist = abs(hist[t] - y_line) / np.sqrt(1 + m**2)

            if dist > max_dist:
                max_dist = dist
                best_threshold = t / num_bins
        mask = img > best_threshold
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "threshold_triangle_sklearn",
            "parameters": {"num_bins": num_bins, **kwargs},
            "execution_time": exec_time,
            "threshold": best_threshold,
        }
        return mask, info

    # def _sklearn_threshold_multi_otsu(
    #     self,
    #     img: np.ndarray,
    #     n_thresholds: int = 2,
    #     num_bins: int = 256,
    #     **kwargs
    # ) -> Tuple[np.ndarray, Dict[str, Any]]:
    #     """
    #     Многоуровневая пороговая обработка по методу Оцу.
    #     Расширение метода Оцу для нескольких порогов.
    #     """
    #     # Гистограмма изображения
    #     hist, bin_edges = np.histogram(img.flatten(), bins=num_bins, range=[0, 1])
    #     hist = hist.astype(np.float64)

    #     if n_thresholds == 1:
    #         # Обычный Оцу
    #         thresh = threshold_otsu(img)
    #         return (img > thresh).astype(np.float32)

    #     # Упрощённый поиск порогов (для 2 порогов)
    #     if n_thresholds == 2:
    #         best_var = -np.inf
    #         best_t1, best_t2 = num_bins // 3, 2 * num_bins // 3

    #         cum_sum = np.cumsum(hist)
    #         cum_mean = np.cumsum(hist * np.arange(num_bins) / num_bins)
    #         total = cum_sum[-1]
    #         total_mean = cum_mean[-1] / total

    #         for t1 in range(1, num_bins - 2):
    #             for t2 in range(t1 + 1, num_bins - 1):
    #                 # Класс 0: [0, t1)
    #                 w0 = cum_sum[t1] / total
    #                 m0 = cum_mean[t1] / cum_sum[t1] if cum_sum[t1] > 0 else 0

    #                 # Класс 1: [t1, t2)
    #                 w1 = (cum_sum[t2] - cum_sum[t1]) / total
    #                 m1 = (cum_mean[t2] - cum_mean[t1]) / (cum_sum[t2] - cum_sum[t1]) if (cum_sum[t2] > cum_sum[t1]) else 0

    #                 # Класс 2: [t2, num_bins)
    #                 w2 = (total - cum_sum[t2]) / total
    #                 m2 = (total_mean * total - cum_mean[t2]) / (total - cum_sum[t2]) if (total > cum_sum[t2]) else 0

    #                 # Межклассовая дисперсия
    #                 var_between = (w0 * (m0 - total_mean)**2 +
    #                               w1 * (m1 - total_mean)**2 +
    #                               w2 * (m2 - total_mean)**2)

    #                 if var_between > best_var:
    #                     best_var = var_between
    #                     best_t1, best_t2 = t1, t2

    #         # Бинаризация: объект = самый яркий класс
    #         best_threshold = best_t2 / num_bins
    #         return (img > best_threshold).astype(np.float32)

    #     # Для >2 порогов используем рекурсивный Оцу
    #     thresholds = []
    #     current_img = img.copy()

    #     for _ in range(n_thresholds):
    #         thresh = threshold_otsu(current_img)
    #         thresholds.append(thresh)
    #         current_img = current_img[current_img <= thresh]
    #         if len(current_img) == 0:
    #             break

    #     # Используем последний порог для бинаризации
    #     if thresholds:
    #         return (img > thresholds[-1]).astype(np.float32)
    #     else:
    #         return np.zeros_like(img)
    def _sklearn_threshold_multi_otsu(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.time()
        n_classes = self.params.get("n_thresholds", 2) + 1

        # Используем нативную реализацию skimage (быстрее и стабильнее кастомной рекурсии)
        from skimage.filters import threshold_multiotsu as threshold_multi_otsu

        thresholds = threshold_multi_otsu(img, classes=n_classes)

        # Для бинарной маски берем порог, отделяющий самый яркий класс
        best_threshold = thresholds[-1] if len(thresholds) > 0 else np.mean(thresholds)
        mask = img > best_threshold

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "threshold_multi_otsu_sklearn",
            "parameters": {"n_thresholds": len(thresholds), **kwargs},
            "execution_time": exec_time,
            "thresholds": thresholds.tolist(),
        }
        return mask, info

    def _sklearn_threshold_percentile(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Процентильная пороговая обработка.
        Порог выбирается как заданный процентиль распределения интенсивностей.
        """
        start_time = time.time()
        percentile = self.params.get("percentile", 90)
        threshold = self.params.get("threshold", 0.1)

        mask = img > threshold

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "threshold_percentile_sklearn",
            "parameters": {"percentile": percentile, **kwargs},
            "execution_time": exec_time,
            "threshold": threshold,
        }
        return mask, info

    def _sklearn_threshold_local_contrast(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Пороговая обработка на основе локального контраста.
        Пиксель считается объектом, если его интенсивность значительно
        отличается от локального среднего.
        """
        start_time = time.time()
        window_size = self.params.get("window_size", 15)
        contrast_factor = self.params.get("contrast_factor", 0.1)

        # h, w = img.shape
        # pad = window_size // 2
        # img_padded = np.pad(img, pad, mode='reflect')
        # mask = np.zeros_like(img)

        # # Вычисляем локальное среднее
        # local_mean = np.zeros_like(img)
        # for i in range(h):
        #     for j in range(w):
        #         window = img_padded[i:i+window_size, j:j+window_size]
        #         local_mean[i, j] = np.mean(window)
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(img, size=window_size)

        # Вычисляем локальный контраст
        local_contrast = np.abs(img - local_mean)
        # Глобальный порог контраста
        global_contrast_threshold = np.percentile(
            local_contrast, 100 * (1 - contrast_factor)
        )

        mask = local_contrast > global_contrast_threshold

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "threshold_local_contrast_sklearn",
            "parameters": {
                "window_size": window_size,
                "contrast_factor": contrast_factor,
                **kwargs,
            },
            "execution_time": exec_time,
        }
        return mask, info

    # ============ МЕТОДЫ НА ОСНОВЕ КРАЕВ ============

    def _sklearn_sobel_edge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Обнаружение границ оператором Собеля.

        Вычисляет градиент интенсивности по горизонтали и вертикали, затем объединяет их.
        Применяется порог к величине градиента для получения бинарной маски границ.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска границ (0/255, dtype=np.uint8).
        """
        gray = img

        # Нормализация к [0, 1] для корректной работы порога 0.5
        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0
        else:
            gray = gray.astype(np.float32)

        # print(f"Gray after Sklearn_sobel_edge: {gray}")

        start_time = time.time()
        threshold = self.params.get("threshold", 0.1)
        magnitude = sobel(gray)

        # Нормализация и порог
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()

        mask = magnitude > threshold

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "sobel_edge_sklearn",
            "parameters": {"threshold": threshold, **kwargs},
            "execution_time": exec_time,
            "threshold": threshold,
        }

        # print(f"Mask after Sklearn_sobel_edge: {mask}")
        # print(f"Info after Sklearn_sobel_edge: {info}")

        return mask, info

    def _sklearn_canny_edge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Обнаружение границ оператором Кэнни.

        Многоэтапный алгоритм: сглаживание, вычисление градиента, подавление немаксимумов,
        двойная пороговая фильтрация и отслеживание связных границ.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска границ (0/255, dtype=np.uint8).
        """
        gray = img

        # Нормализация к [0, 1] для корректной работы порога 0.5
        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0
        else:
            gray = gray.astype(np.float32)

        print(f"Gray after Sklearn_canny_edge: {gray}")

        start_time = time.time()

        sigma = self.params.get("sigma", 1.0)
        low_threshold = self.params.get("low", 0.1)
        high_threshold = self.params.get("high", 0.3)
        use_quantiles = self.params.get("use_quantiles", False)

        # 3. Включаем квантили!
        use_quantiles = False

        start_time = time.time()
        mask = feature.canny(
            gray,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            use_quantiles=use_quantiles,
        )
        exec_time = time.time() - start_time
        print(
            f"DEBUG: sigma={sigma}, low={low_threshold}, high={high_threshold}, quantiles={use_quantiles}"
        )
        print(f"DEBUG: Image range: [{gray.min():.4f}, {gray.max():.4f}]")

        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "canny_edge_sklearn",
            "parameters": {
                "sigma": sigma,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "use_quantiles": use_quantiles,
                **kwargs,
            },
            "execution_time": exec_time,
        }
        print(f"Mask after Sklearn_canny_edge: {mask}")
        print(f"Info after Sklearn_canny_edge: {info}")
        return mask, info

    def _sklearn_prewitt_edge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Обнаружение границ оператором Превитта.
        Вычисляет градиент с использованием ядер 3×3.
        Менее чувствителен к шуму, чем Собель.
        """
        start_time = time.time()
        threshold = self.params.get("threshold", 0.1)
        magnitude = prewitt(img)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        mask = magnitude > threshold

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "prewitt_edge_sklearn",
            "parameters": {"threshold": threshold, **kwargs},
            "execution_time": exec_time,
        }
        return mask, info

    def _sklearn_scharr_edge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Обнаружение границ оператором Шара.
        Улучшенная версия Собеля с лучшей точностью вычисления градиента.
        Использует оптимизированные ядра 3×3.
        """
        start_time = time.time()
        threshold = self.params.get("threshold", 0.1)

        # # Ядра Шара
        # kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32) / 16
        # kernel_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32) / 16

        # # Свёртка
        # gx = ndimage.convolve(img, kernel_x, mode='reflect')
        # gy = ndimage.convolve(img, kernel_y, mode='reflect')

        # # Магнитуда градиента
        # magnitude = np.sqrt(gx**2 + gy**2)
        # if magnitude.max() > 0:
        #     magnitude = magnitude / magnitude.max()

        magnitude = filters.scharr(img)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        mask = magnitude > threshold

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "scharr_edge_sklearn",
            "parameters": {"threshold": threshold, **kwargs},
            "execution_time": exec_time,
        }
        return mask, info

    def _sklearn_roberts_cross_edge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Обнаружение границ оператором Робертса (Cross).
        Простой оператор для обнаружения диагональных границ.
        Использует ядра 2×2 для вычисления градиента.
        """
        start_time = time.time()
        threshold = self.params.get("threshold", 0.1)

        # # Ядра Робертса
        # kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        # kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        # # Свёртка
        # gx = ndimage.convolve(img, kernel_x, mode='reflect')
        # gy = ndimage.convolve(img, kernel_y, mode='reflect')

        # # Магнитуда градиента
        # magnitude = np.sqrt(gx**2 + gy**2)
        # if magnitude.max() > 0:
        #     magnitude = magnitude / magnitude.max()

        magnitude = filters.roberts(img)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        mask = magnitude > threshold

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "roberts_cross_edge_sklearn",
            "parameters": {"threshold": threshold, **kwargs},
            "execution_time": exec_time,
        }
        return mask, info

    def _sklearn_log_edge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Обнаружение границ Лапласианом Гауссиана (LoG).
        Применяет Гауссово размытие, затем Лапласиан.
        Границы обнаруживаются по пересечению нуля (zero-crossing).
        """
        start_time = time.time()
        sigma = self.params.get("sigma", 1.0)
        threshold = self.params.get("threshold", 0.01)
        # Гауссово размытие
        img_blurred = gaussian_filter(img, sigma=sigma)

        # Лапласиан
        laplacian = laplace(img_blurred)

        # # Zero-crossing detection
        # zero_crossing = np.zeros_like(img)
        # h, w = img.shape

        # for i in range(1, h - 1):
        #     for j in range(1, w - 1):
        #         neighborhood = laplacian[i-1:i+2, j-1:j+2]
        #         if np.min(neighborhood) < 0 < np.max(neighborhood):
        #             if np.abs(np.min(neighborhood)) + np.abs(np.max(neighborhood)) > threshold:
        #                 zero_crossing[i, j] = 1

        # return zero_crossing.astype(np.float32)

        # Векторизованный поиск zero-crossing
        zc = (
            (laplacian > 0) & (np.roll(laplacian, 1, axis=0) < 0)
            | (laplacian < 0) & (np.roll(laplacian, 1, axis=0) > 0)
            | (laplacian > 0) & (np.roll(laplacian, 1, axis=1) < 0)
            | (laplacian < 0) & (np.roll(laplacian, 1, axis=1) > 0)
        )
        mask = zc & (np.abs(laplacian) > threshold)

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "log_edge_sklearn",
            "parameters": {"sigma": sigma, "threshold": threshold, **kwargs},
            "execution_time": exec_time,
        }
        return mask, info

    def _sklearn_dog_edge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Обнаружение границ разностью Гауссианов (DoG).
        Аппроксимация LoG через разность двух Гауссианов с разными σ.
        Эффективно для обнаружения границ разного масштаба.
        """
        start_time = time.time()
        sigma1 = self.params.get("sigma1", 1.0)
        sigma2 = self.params.get("sigma2", 2.0)
        threshold = self.params.get("threshold", 0.01)

        # Два Гауссовых фильтра
        g1 = gaussian_filter(img, sigma=sigma1)
        g2 = gaussian_filter(img, sigma=sigma2)

        # Разность Гауссианов
        dog = g1 - g2

        # # Zero-crossing detection
        # zero_crossing = np.zeros_like(img)
        # h, w = img.shape

        # for i in range(1, h - 1):
        #     for j in range(1, w - 1):
        #         neighborhood = dog[i-1:i+2, j-1:j+2]
        #         if np.min(neighborhood) < 0 < np.max(neighborhood):
        #             if np.abs(np.min(neighborhood)) + np.abs(np.max(neighborhood)) > threshold:
        #                 zero_crossing[i, j] = 1

        # return zero_crossing.astype(np.float32)
        zc = (
            (dog > 0) & (np.roll(dog, 1, axis=0) < 0)
            | (dog < 0) & (np.roll(dog, 1, axis=0) > 0)
            | (dog > 0) & (np.roll(dog, 1, axis=1) < 0)
            | (dog < 0) & (np.roll(dog, 1, axis=1) > 0)
        )
        mask = zc & (np.abs(dog) > threshold)

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "dog_edge_sklearn",
            "parameters": {
                "sigma1": sigma1,
                "sigma2": sigma2,
                "threshold": threshold,
                **kwargs,
            },
            "execution_time": exec_time,
        }
        return mask, info

    def _sklearn_marr_hildreth_edge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Обнаружение границ методом Марра-Хилдрета.
        Комбинация Гауссова размытия и Лапласиана с zero-crossing.
        Классический метод для обнаружения границ.
        """
        return self._sklearn_log_edge(img, **kwargs)

    def _sklearn_gradient_magnitude_direction(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Обнаружение границ через магнитуду и направление градиента.
        Вычисляет градиент с направлением и позволяет фильтрацию по углу.
        """
        start_time = time.time()
        threshold = self.params.get("threshold", 0.1)
        angle_range = self.params.get("angle_range", None)

        # Градиенты Собеля
        # gx = sobel(img, axis=1)  # Горизонтальный градиент
        # gy = sobel(img, axis=0)  # Вертикальный градиент

        gx = filters.sobel_h(img)
        gy = filters.sobel_v(img)

        # Магнитуда и направление
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi  # В градусах

        # Нормализация магнитуды
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()

        # Фильтрация по направлению (если указано)
        if angle_range is not None:
            angle_mask = (
                (direction >= angle_range[0]) & (direction <= angle_range[1])
            ) | (
                (direction + 180 >= angle_range[0])
                & (direction + 180 <= angle_range[1])
            )
            magnitude = magnitude * angle_mask

        mask = magnitude > threshold
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)
        info = {
            "method": "gradient_magnitude_direction_sklearn",
            "parameters": {
                "threshold": threshold,
                "angle_range": angle_range,
                **kwargs,
            },
            "execution_time": exec_time,
        }
        return mask, info

    # def _sklearn_phase_congruency_edge(
    #     self,
    #     img: np.ndarray,
    #     **kwargs
    # ) -> Tuple[np.ndarray, Dict[str, Any]]:
    #     """
    #     Обнаружение границ через фазовую конгруэнтность.
    #     Метод, инвариантный к изменению контраста и яркости.
    #     Основан на согласованности фаз Фурье-компонент.
    #     Упрощённая реализация через много-масштабные градиенты.
    #     """
    #     start_time = time.time()
    #     nscales = self.params.get('nscales', 4)
    #     threshold = self.params.get('threshold', 0.5)

    #     # h, w = img.shape

    #     # # Много-масштабные градиенты
    #     # phase_congruency = np.zeros_like(img)

    #     # for scale in range(nscales):
    #     #     sigma = 2 ** scale

    #     #     # Гауссово размытие на текущем масштабе
    #     #     img_blurred = gaussian_filter(img, sigma=sigma)

    #     #     # Градиенты
    #     #     gx = sobel(img_blurred, axis=1)
    #     #     gy = sobel(img_blurred, axis=0)

    #     #     # Магнитуда
    #     #     mag = np.sqrt(gx**2 + gy**2)
    #     #     if mag.max() > 0:
    #     #         mag = mag / mag.max()

    #     #     phase_congruency += mag
    #     phase_congruency = np.zeros_like(img)
    #     for scale in range(nscales):
    #         sigma = 2 ** scale

    #         # Гауссово размытие на текущем масштабе
    #         blurred = gaussian_filter(img, sigma=sigma)

    #         # Градиенты
    #         gx = filters.sobel_h(blurred)
    #         gy = filters.sobel_v(blurred)

    #         # Магнитуда
    #         mag = np.sqrt(gx**2 + gy**2)
    #         if mag.max() > 0:
    #             mag = mag / mag.max()
    #         phase_congruency += mag

    #     # Усреднение по масштабам
    #     phase_congruency = phase_congruency / nscales

    #     mask = phase_congruency > threshold

    #     exec_time = time.time() - start_time
    #     mask = (mask * 255).astype(np.uint8)
    #     info = {'method': 'phase_congruency_edge_sklearn', 'parameters': {'nscales': nscales, 'threshold': threshold, **kwargs}, 'execution_time': exec_time}
    #     return mask, info
    def _sklearn_phase_congruency_edge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Настоящая фазовая конгруэнтность (Kovesi's algorithm).
        Инвариантна к освещению и контрасту. Обнаруживает края через
        выравнивание фаз Фурье-компонент в пространстве изображений.
        """
        start_time = time.time()

        # ============ ПАРАМЕТРЫ ============
        nscales = self.params.get("nscales", 4)
        norientations = self.params.get("norientations", 4)
        min_wavelength = self.params.get("min_wavelength", 3)
        mult = self.params.get("mult", 2.0)
        sigma_onf = self.params.get("sigma_onf", 0.55)
        k_noise = self.params.get("k_noise", 2.0)
        cutoff_pc = self.params.get("cutoff_pc", 0.5)
        epsilon = 1e-6

        # Нормализация
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        rows, cols = img.shape
        img_fft = np.fft.fft2(img)

        # ============ ЧАСТОТНАЯ СЕТКА ============
        x = np.linspace(-0.5, 0.5, cols)
        y = np.linspace(-0.5, 0.5, rows)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(-Y, X)
        R[0, 0] = 1e-10  # Защита от деления на 0

        # Аккумуляторы
        sum_even = np.zeros((rows, cols), dtype=np.float32)
        sum_odd = np.zeros((rows, cols), dtype=np.float32)
        sum_amp = np.zeros((rows, cols), dtype=np.float32)
        noise_energy = np.zeros((rows, cols), dtype=np.float32)

        orientations = np.linspace(0, np.pi, norientations, endpoint=False)

        for scale in range(nscales):
            wavelength = min_wavelength * (mult**scale)
            fo = 1.0 / wavelength
            # sigma_f = fo * sigma_onf

            # Радиальная часть Log-Gabor
            log_gabor = np.exp(-0.5 * (np.log(R / fo) / np.log(sigma_onf)) ** 2)
            log_gabor[0, 0] = 0.0  # DC компонента = 0

            for orient_idx, angle in enumerate(orientations):
                # Угловая часть (Гауссов разброс)
                angular_spread = 1.57 / norientations
                d_theta = np.abs(Theta - angle)
                d_theta = np.minimum(d_theta, 2 * np.pi - d_theta)
                angular = np.exp(-0.5 * (d_theta / angular_spread) ** 2)

                # Полный фильтр в частотной области (сдвинутый)
                filter_f = log_gabor * angular
                filter_f = np.fft.ifftshift(filter_f)  # Готовим к умножению с fft2

                # Свёртка в частотной области
                response = np.fft.ifft2(img_fft * filter_f)
                even_resp = np.real(response)
                odd_resp = np.imag(response)

                # Амплитуда отклика
                amp = np.sqrt(even_resp**2 + odd_resp**2)

                # Оценка шума (MAD) для текущего фильтра
                # noise_amp ≈ 2 * median(|response|) / 0.6745
                med = np.median(np.abs(amp))
                noise_est = 2 * (med / 0.6745)

                # Накопление
                sum_even += even_resp
                sum_odd += odd_resp
                sum_amp += amp
                noise_energy += noise_est**2

        # ============ ВЫЧИСЛЕНИЕ PHASE CONGRUENCY ============
        local_energy = np.sqrt(sum_even**2 + sum_odd**2)

        # Компенсация шума (Kovesi)
        T = noise_energy * k_noise
        pc_map = np.maximum(local_energy - T, 0) / (sum_amp + epsilon)

        # Ограничение [0, 1]
        pc_map = np.clip(pc_map, 0, 1)

        # Бинаризация
        mask = pc_map > cutoff_pc

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "phase_congruency_edge_sklearn",
            "parameters": {
                "nscales": nscales,
                "norientations": norientations,
                "min_wavelength": min_wavelength,
                "mult": mult,
                "sigma_onf": sigma_onf,
                "k_noise": k_noise,
                "cutoff_pc": cutoff_pc,
                **kwargs,
            },
            "execution_time": exec_time,
            "mean_pc": float(np.mean(pc_map[mask > 0])) if mask.any() else 0.0,
        }
        return mask, info

    # ============ РЕГИОНАЛЬНЫЕ МЕТОДЫ ============

    def _sklearn_region_growing(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

        h, w = gray.shape
        seed = self.params.get("seed", (w // 2, h // 2))
        tolerance = self.params.get("tolerance", 0.1)

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

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "region_growing_sklearn",
            "parameters": {"seed": seed, "tolerance": tolerance, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_split_and_merge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

        h, w = gray.shape
        threshold = self.params.get("threshold", 0.1)
        min_size = self.params.get("min_size", 50)

        # Начальный регион
        # regions = [gray.copy()]
        # region_masks = [np.ones((h, w), dtype=bool)]

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
                    r_end = (i + 1) * h_mid if i == 0 else h_reg
                    c_start = j * w_mid
                    c_end = (j + 1) * w_mid if j == 0 else w_reg

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
                mask[: best_region_mask.shape[0], : best_region_mask.shape[1]] = (
                    best_region_mask.astype(np.uint8)
                )

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "split_and_merge_sklearn",
            "parameters": {"threshold": threshold, "min_size": min_size, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _split_and_merge(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

        h, w = gray.shape
        threshold = self.params.get("threshold", 0.1)
        min_size = self.params.get("min_size", 50)

        # Рекурсивная функция split
        def split_region(x1, y1, x2, y2):
            if (x2 - x1) * (y2 - y1) <= min_size:
                return [(x1, y1, x2, y2)]

            # Вычисляем статистики региона
            region = gray[y1:y2, x1:x2]
            # mean = np.mean(region)
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
            sizes = [(y2 - y1) * (x2 - x1) for x1, y1, x2, y2 in regions]
            sorted_indices = np.argsort(sizes)[::-1]

            # Второй по величине регион
            if len(sorted_indices) > 1:
                idx = sorted_indices[1]
                x1, y1, x2, y2 = regions[idx]
                mask[y1:y2, x1:x2] = True

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "split_and_merge_sklearn",
            "parameters": {"threshold": threshold, "min_size": min_size, **kwargs},
            "execution_time": exec_time,
        }
        return mask, info

    def _floodfill(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()
        h, w = gray.shape
        seed = self.params.get("seed", (w // 2, h // 2))
        tolerance = self.params.get("tolerance", 0.15)
        mask = segmentation.flood(gray, seed, tolerance=tolerance)

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "floodfill_sklearn",
            "parameters": {"seed": seed, "tolarance": tolerance, **kwargs},
            "execution_time": exec_time,
        }
        return mask, info

    # ============ КЛАСТЕРИЗАЦИЯ ============

    def _sklearn_kmeans_segmentation(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

        # Преобразуем изображение в массив пикселей
        pixels = img.reshape(-1, 3)

        # Применяем K-Means
        k = self.params.get("k", 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)

        # Находим самый большой кластер (фон)
        unique_labels, counts = np.unique(labels, return_counts=True)
        bg_label = unique_labels[np.argmax(counts)]

        # Создаем маску (все кроме фона)
        mask = (labels != bg_label).reshape(h, w)
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "kmeans_sklearn",
            "parameters": {"k": k, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_kmeans(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """K-Means из sklearn с извлечением признаков."""
        features = self._extract_features(img)
        h, w = img.shape[:2]

        start_time = time.time()

        n_clusters = self.params.get("n_clusters", 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
        labels = kmeans.fit_predict(features)

        unique, counts = np.unique(labels, return_counts=True)
        # bg_label = unique[np.argmax(counts)]

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "kmeans_sklearn",
            "parameters": {"n_clusters": n_clusters, **kwargs},
            "execution_time": exec_time,
            "cluster_centers": kmeans.cluster_centers_,
            "inertia": kmeans.inertia_,
        }

        return mask, info

    def _sklearn_dbscan_segmentation(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        start_time = time.time()

        h, w = gray.shape

        # Извлекаем признаки (пиксель + координаты)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack(
            [gray.ravel(), x_coords.ravel() / w, y_coords.ravel() / h]
        )

        # Применяем DBSCAN
        eps = self.params.get("eps", 0.1)
        min_samples = self.params.get("min_samples", 10)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = dbscan.fit_predict(features)

        # Создаем маску (исключаем шум -1)
        mask = (labels != -1).reshape(h, w)
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "dbscan_sklearn",
            "parameters": {"eps": eps, "min_samples": min_samples, **kwargs},
            "execution_time": exec_time,
            "n_clusters": len(np.unique(labels[labels != -1])),
            "n_noise": np.sum(labels == -1),
        }

        return mask, info

    def _sklearn_dbscan(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """DBSCAN из sklearn."""
        features = self._extract_features(img)
        h, w = img.shape[:2]
        start_time = time.time()

        eps = self.params.get("eps", 0.5)
        min_samples = self.params.get("min_samples", 5)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = dbscan.fit_predict(features)

        # Создаем маску (исключаем шум)
        mask = (labels != -1).reshape(h, w)
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "dbscan_sklearn",
            "parameters": {"eps": eps, "min_samples": min_samples, **kwargs},
            "execution_time": exec_time,
            "n_clusters": len(np.unique(labels[labels != -1])),
            "n_noise": np.sum(labels == -1),
        }

        return mask, info

    def _meanshift(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        start_time = time.time()

        # Для производительности сэмплируем пиксели
        sample_size = min(1000, h * w)
        indices = np.random.choice(h * w, sample_size, replace=False)

        pixels = img_rgb.reshape(-1, 3)[indices]
        coords = np.column_stack([indices // w, indices % w]) / [h, w]

        features = np.hstack([pixels / 255.0, coords])

        # Применяем MeanShift
        bandwidth = self.params.get("bandwidth", 0.5)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        labels = ms.fit_predict(features)

        # Интерполируем метки обратно на все пиксели
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, labels)

        all_coords = np.column_stack(
            [np.repeat(np.arange(h), w), np.tile(np.arange(w), h)]
        ) / [h, w]

        all_pixels = img_rgb.reshape(-1, 3) / 255.0
        all_features = np.hstack([all_pixels, all_coords])

        all_labels = knn.predict(all_features)

        # Находим самый большой кластер
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        bg_label = unique_labels[np.argmax(counts)]

        mask = (all_labels != bg_label).reshape(h, w)
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "meanshift_sklearn",
            "parameters": {"bandwidth": bandwidth, **kwargs},
            "execution_time": exec_time,
        }
        return mask, info

    def _sklearn_meanshift(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        MeanShift кластеризация для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

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
        bandwidth = self.params.get("bandwidth", None)
        if bandwidth is None:
            # Автоматическая оценка bandwidth
            bandwidth = self._estimate_meanshift_bandwidth(sample_features)

        # Применяем MeanShift
        meanshift = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=True,
            min_bin_freq=1,
            cluster_all=True,
            n_jobs=-1,
            **kwargs,
        )

        if use_sampling:
            meanshift.fit(sample_features)
            # Предсказываем для всех точек
            labels = meanshift.predict(features)
        else:
            labels = meanshift.fit_predict(features)

        # Создаем маску
        unique, counts = np.unique(labels, return_counts=True)
        mask = self._create_mask_from_labels(labels, (h, w))
        mask = self._postprocess_mask(mask)
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "meanshift_sklearn",
            "parameters": {"bandwidth": bandwidth, **kwargs},
            "execution_time": exec_time,
            "n_clusters": len(unique),
            "cluster_centers": meanshift.cluster_centers_,
        }

        return mask, info

    # ============ АКТИВНЫЕ КОНТУРЫ ============

    def _active_contour(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

        # Создаем начальный контур (окружность)
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y) // 2

        s = np.linspace(0, 2 * np.pi, 100)
        # r = center_y + radius * np.sin(s)
        # c = center_x + radius * np.cos(s)
        # init = np.array([r, c]).T
        init = np.array(
            [center_x + radius * np.cos(s), center_y + radius * np.sin(s)]
        ).T

        # Параметры активного контура
        alpha = self.params.get("alpha", 0.015)  # elasticity (0.01)
        beta = self.params.get("beta", 10)  # rigidity (0.1)
        gamma = self.params.get("gamma", 0.001)  # time step (0.001)
        max_iterations = self.params.get("max_iterations", 2000)
        w_edge = self.params.get("w_edge", 1)
        w_line = self.params.get("w_line", 0)

        # Применяем активный контур
        snake = active_contour(
            gaussian(gray, 3, preserve_range=False),
            init,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            w_line=w_line,  # attract to light
            w_edge=w_edge,  # attract to edges
            max_num_iter=max_iterations,
            **kwargs,
        )

        # Создаем маску из контура
        mask = np.zeros((h, w), dtype=bool)
        rr, cc = polygon(snake[:, 1], snake[:, 0], mask.shape)
        mask[rr, cc] = True

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "active_contour_sklearn",
            "parameters": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "w_edge": w_edge,
                "w_line": w_line,
                "max_iterations": max_iterations,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _gvf_contour(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

        # Вычисляем градиенты
        # grad_y, grad_x = np.gradient(gray_norm)
        grad_x = filters.sobel_h(gray)
        grad_y = filters.sobel_v(gray)

        # Вычисляем внешние силы (edge map)
        edge_map = grad_x**2 + grad_y**2
        edge_map = edge_map / (edge_map.max() + 1e-8)

        # Применяем GVF
        mu = self.params.get("mu", 0.1)  # 0.2
        iterations = self.params.get("iterations", 50)  # 100

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
        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "gvf_contour_sklearn",
            "parameters": {"mu": mu, "iterations": iterations, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _morphological_snakes(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

        # Применяем морфологические активные контуры
        init_level_set = np.zeros(gray.shape, dtype=np.int8)
        h, w = gray.shape
        init_level_set[(h // 4):(3 * h // 4), (w // 4):(3 * w // 4)] = 1

        smoothing = self.params.get("smoothing", 1)
        threshold = self.params.get("threshold", 0.5)
        iterations = self.params.get("iterations", 50)

        mask = morphological_geodesic_active_contour(
            gray,
            iterations,
            init_level_set,
            smoothing=smoothing,
            threshold=threshold,
            balloon=1,
            **kwargs,
        )

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "morphological_snakes_sklearn",
            "parameters": {
                "smoothing": smoothing,
                "threshold": threshold,
                "iterations": iterations,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _chan_vese(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

        # Начальная маска
        init_level_set = np.zeros(gray.shape, dtype=np.int8)
        h, w = gray.shape
        init_level_set[(h // 4):(3 * h // 4), (w // 4):(3 * w // 4)] = 1

        # Параметры Chan-Vese
        mu = self.params.get("mu", 0.25)
        lambda1 = self.params.get("lambda1", 1.0)
        lambda2 = self.params.get("lambda2", 1.0)
        tol = self.params.get("tol", 1e-3)
        max_iter = self.params.get("max_iter", 100)
        iterations = self.params.get("iterations", 100)

        # Применяем метод Chan-Vese
        mask = morphological_chan_vese(
            gray,
            iterations,
            init_level_set,
            smoothing=1,
            lambda1=lambda1,
            lambda2=lambda2,
            **kwargs,
        )

        # segmentation = chan_vese(
        #         gray_norm,
        #         mu=mu,
        #         lambda1=lambda1,
        #         lambda2=lambda2,
        #         tol=tol,
        #         max_num_iter=max_iter,
        #         init_level_set=init_level_set,
        # **kwargs
        #     )
        # mask = (segmentation > 0.5).astype(np.uint8) * 255

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "chan_vese_sklearn",
            "parameters": {
                "mu": mu,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "tol": tol,
                "max_iter": max_iter,
                "iterations": iterations,
                **kwargs,
            },
            "execution_time": exec_time,
            "converged": exec_time < max_iter * 0.1,
        }

        return mask, info

    # ============ WATERSHED И ГРАФОВЫЕ МЕТОДЫ ============

    def _watershed(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

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

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "watershed_sklearn",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _random_walker(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

        # Создаем маркеры
        markers = np.zeros(gray.shape, dtype=np.uint8)
        h, w = gray.shape

        # Центральная область - объект
        markers[(h // 4):(3 * h // 4), (w // 4):(3 * w // 4)] = 2

        # Углы - фон
        corner_size = min(h, w) // 8
        markers[:corner_size, :corner_size] = 1
        markers[:corner_size, -corner_size:] = 1
        markers[-corner_size:, :corner_size] = 1
        markers[-corner_size:, -corner_size:] = 1

        # Применяем Random Walker
        beta = self.params.get("beta", 10)
        labels = random_walker(gray, markers, beta=beta, mode="cg_mg")

        # Бинаризуем
        mask = labels == 2

        exec_time = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info = {
            "method": "random_walker_sklearn",
            "parameters": {"beta": beta, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    # ============ SUPER-PIXEL МЕТОДЫ ============
    def _quickshift(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Сегментация методом Quickshift (реализована через MeanShift как аналог).

        Находит моды в плотности распределения пикселей в пространстве признаков.
        Группирует пиксели, принадлежащие одной моде.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8). Самый крупный кластер — фон.
        """
        start_time = time.time()

        kernel_size = self.params.get("kernel_size", 3)
        max_dist = self.params.get("max_dist", 6)
        ratio = self.params.get("ratio", 0.5)

        segments = quickshift(
            img, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio, **kwargs
        )

        # Находим самый большой суперпиксель
        unique_labels, counts = np.unique(segments, return_counts=True)
        if len(unique_labels) > 0:
            bg_label = unique_labels[np.argmax(counts)]
            mask = (segments != bg_label).astype(np.uint8) * 255
        else:
            mask = np.zeros_like(segments, dtype=np.uint8)

        exec_time = time.time() - start_time

        info = {
            "method": "quickshift_sklearn",
            "parameters": {
                "kernel_size": kernel_size,
                "max_dist": max_dist,
                "ratio": ratio,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _slic(self, img: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        SLIC (Simple Linear Iterative Clustering) — суперпиксельная сегментация.

        Группирует пиксели в компактные, однородные регионы (суперпиксели) на основе пространственной
        и цветовой близости. Самый крупный суперпиксель считается фоном.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8): 255 — все суперпиксели, кроме фона.
        """
        start_time = time.time()

        n_segments = self.params.get("n_segments", 100)
        compactness = self.params.get("compactness", 10)

        # Применяем SLIC
        segments = slic(img, n_segments=n_segments, compactness=compactness, **kwargs)

        # Находим самый большой суперпиксель
        unique_labels, counts = np.unique(segments, return_counts=True)
        if len(unique_labels) > 0:
            bg_label = unique_labels[np.argmax(counts)]
            mask = (segments != bg_label).astype(np.uint8) * 255
        else:
            mask = np.zeros_like(segments, dtype=np.uint8)

        exec_time = time.time() - start_time

        info = {
            "method": "slic_sklearn",
            "parameters": {
                "n_segments": n_segments,
                "compactness": compactness,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _felzenszwalb(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Алгоритм Felzenszwalb — иерархическая сегментация на основе графов.

        Строит сегментацию, начиная с мелких регионов и объединяя их, если внутреннее различие
        меньше межрегионального. Очень эффективен для выделения объектов разного масштаба.

        Args:
            img: Входное изображение (RGB).

        Returns:
            Бинарная маска: 255 — все регионы, кроме самого крупного (фона).
        """
        start_time = time.time()

        scale = self.params.get("scale", 100)
        sigma = self.params.get("sigma", 0.5)
        min_size = self.params.get("min_size", 50)

        # Применяем Felzenszwalb
        segments = felzenszwalb(
            img, scale=scale, sigma=sigma, min_size=min_size, **kwargs
        )

        # Находим самый большой регион
        unique_labels, counts = np.unique(segments, return_counts=True)
        if len(unique_labels) > 0:
            bg_label = unique_labels[np.argmax(counts)]
            mask_np = (segments != bg_label).astype(np.uint8) * 255
        else:
            mask_np = np.zeros_like(segments, dtype=np.uint8)

        exec_time = time.time() - start_time

        info = {
            "method": "felzenszwalb_sklearn",
            "parameters": {
                "scale": scale,
                "sigma": sigma,
                "min_size": min_size,
                **kwargs,
            },
            "execution_time": exec_time,
            "n_segments": len(unique_labels),
        }

        return mask_np, info

    # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============

    def _grabcut(self, img: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        start_time = time.time()

        # Создаем начальную маску
        mask = np.zeros((h, w), dtype=np.uint8)

        # Прямоугольник в центре
        rect = self.params.get("rect", (w // 4, h // 4, w // 2, h // 2))
        x, y, w_rect, h_rect = rect

        mask[y:(y + h_rect), x:(x + w_rect)] = 3  # Вероятный передний план

        # Углы - определенный фон
        corner_size = min(h, w) // 8
        mask[:corner_size, :corner_size] = 0  # Определенный фон
        mask[:corner_size, -corner_size:] = 0
        mask[-corner_size:, :corner_size] = 0
        mask[-corner_size:, -corner_size:] = 0

        # Используем Random Forest для имитации GrabCut
        # Подготовка данных
        pixels = img.reshape(-1, 3)
        coords = np.column_stack([np.repeat(np.arange(h), w), np.tile(np.arange(w), h)])

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

        exec_time = time.time() - start_time
        mask = mask_result.astype(np.uint8)

        info = {
            "method": "grabcut_sklearn",
            "parameters": {"rect": rect, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    # ============ SKLEARN МЕТОДЫ ============

    def _sklearn_gmm(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Gaussian Mixture Models для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        n_components = self.params.get("n_components", 3)
        if n_components <= 0:
            n_components = self._estimate_gmm_components(features)

        # Тип ковариационной матрицы
        covariance_type = self.params.get("covariance_type", "full")

        # Применяем GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=1e-3,
            reg_covar=1e-6,
            max_iter=100,
            n_init=1,
            init_params="kmeans",
            random_state=42,
            **kwargs,
        )

        labels = gmm.fit_predict(features)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "gmm_sklearn",
            "parameters": {
                "n_components": n_components,
                "covariance_type": covariance_type,
                **kwargs,
            },
            "execution_time": exec_time,
            "converged": gmm.converged_,
            "lower_bound": gmm.lower_bound_,
        }

        return mask, info

    def _sklearn_optics(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        OPTICS кластеризация для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

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
        min_samples = self.params.get("min_samples", 5)
        xi = self.params.get("xi", 0.05)
        min_cluster_size = self.params.get("min_cluster_size", 0.1)

        # Применяем OPTICS
        optics = OPTICS(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_method="xi",
            algorithm="auto",
            leaf_size=30,
            n_jobs=-1,
        )

        if use_sampling:
            optics.fit(sample_features)
            # Интерполируем метки
            labels = self._interpolate_labels(sample_features, features, optics.labels_)
        else:
            labels = optics.fit_predict(features)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "optics_sklearn",
            "parameters": {
                "min_samples": min_samples,
                "xi": xi,
                "min_cluster_size": min_cluster_size,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_agglomerative(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Agglomerative (иерархическая) кластеризация.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

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
        n_clusters = self.params.get("n_clusters", 3)
        linkage = self.params.get("linkage", "ward")
        affinity = self.params.get("affinity", "euclidean")

        # Применяем Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity,
            compute_full_tree="auto",
        )

        if use_sampling:
            clustering.fit(sample_features)
            labels = self._interpolate_labels(
                sample_features, features, clustering.labels_
            )
        else:
            labels = clustering.fit_predict(features)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "agglomerative_sklearn",
            "parameters": {
                "n_clusters": n_clusters,
                "linkage": linkage,
                "affinity": affinity,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_spectral(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Spectral Clustering для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

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
        n_clusters = self.params.get("n_clusters", 3)
        affinity = self.params.get("affinity", "nearest_neighbors")
        n_neighbors = self.params.get("n_neighbors", 10)

        # Применяем Spectral Clustering
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_solver="arpack",
            random_state=42,
            n_jobs=-1,
        )

        if use_sampling:
            spectral.fit(sample_features)
            labels = self._interpolate_labels(
                sample_features, features, spectral.labels_
            )
        else:
            labels = spectral.fit_predict(features)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "spectral_sklearn",
            "parameters": {
                "n_clusters": n_clusters,
                "n_neighbors": n_neighbors,
                "affinity": affinity,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_birch(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Параметры BIRCH
        n_clusters = self.params.get("n_clusters", 3)
        threshold = self.params.get("threshold", 0.5)
        branching_factor = self.params.get("branching_factor", 50)

        # Применяем BIRCH
        birch = Birch(
            n_clusters=n_clusters,
            threshold=threshold,
            branching_factor=branching_factor,
        )

        labels = birch.fit_predict(features)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time

        info = {
            "method": "birch_sklearn",
            "parameters": {
                "n_clusters": n_clusters,
                "threshold": threshold,
                "branching_factor": branching_factor,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_mini_batch_kmeans(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Mini-Batch K-Means для больших изображений.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Параметры
        n_clusters = self.params.get("n_clusters", 3)
        batch_size = self.params.get("batch_size", 100)

        # Применяем Mini-Batch K-Means
        mbkmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            init="k-means++",
            n_init=3,
            max_iter=100,
            random_state=42,
        )

        labels = mbkmeans.fit_predict(features)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "mini_batch_kmeans_sklearn",
            "parameters": {
                "n_clusters": n_clusters,
                "batch_size": batch_size,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

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
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Random Forest для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Создаем метки для обучения
        labels_train = -np.ones(h * w)

        # Центральная область - объект
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Маска объекта
        obj_mask = (
            (x_coords - center_w) ** 2 + (y_coords - center_h) ** 2
        ) <= obj_size**2
        labels_train[obj_mask.ravel()] = 1

        # Углы - фон
        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h),
        ]

        for x1, y1, x2, y2 in corners:
            labels_train[y_coords[y1:y2, x1:x2].ravel()] = 0

        # Обучаем Random Forest
        train_indices = labels_train >= 0
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(features[train_indices], labels_train[train_indices])

        # Предсказываем
        labels = rf.predict(features)
        mask = (labels.reshape(h, w) > 0).astype(np.uint8) * 255

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "random_forest_sklearn",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_svm(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Support Vector Machine для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Создаем метки для обучения (как в Random Forest)
        labels_train = np.zeros(features.shape[0])
        train_mask = np.zeros((h, w), dtype=bool)

        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        cv2.rectangle(
            train_mask,
            (center_w - obj_size, center_h - obj_size),
            (center_w + obj_size, center_h + obj_size),
            True,
            -1,
        )

        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h),
        ]

        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True

        labels_train = train_mask.ravel().astype(int)
        train_indices = np.where(labels_train >= 0)[0]
        X_train = features[train_indices]
        y_train = labels_train[train_indices]

        C = (self.params.get("C", 1.0),)
        kernel = (self.params.get("kernel", "rbf"),)
        gamma = (self.params.get("gamma", "scale"),)

        # Обучаем SVM
        svm = SVC(C, kernel, gamma, probability=True, random_state=42)

        svm.fit(X_train, y_train)

        # Предсказываем
        labels = svm.predict(features)

        # Создаем маску
        mask = labels.reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)

        exec_time = time.time() - start_time

        info = {
            "method": "svm_sklearn",
            "parameters": {"C": C, "kernel": kernel, "gamma": gamma, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_logistic_regression(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Logistic Regression для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Создаем метки для обучения
        labels_train = np.zeros(features.shape[0])
        train_mask = np.zeros((h, w), dtype=bool)

        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        cv2.rectangle(
            train_mask,
            (center_w - obj_size, center_h - obj_size),
            (center_w + obj_size, center_h + obj_size),
            True,
            -1,
        )

        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h),
        ]

        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True

        labels_train = train_mask.ravel().astype(int)
        train_indices = np.where(labels_train >= 0)[0]
        X_train = features[train_indices]
        y_train = labels_train[train_indices]

        C = (self.params.get("C", 1.0),)

        # Обучаем Logistic Regression
        lr = LogisticRegression(
            penalty="l2",
            C=self.params.get("C", 1.0),
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )

        lr.fit(X_train, y_train)

        # Предсказываем
        labels = lr.predict(features)

        # Создаем маску
        mask = labels.reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)

        exec_time = time.time() - start_time

        info = {
            "method": "logistic_regression_sklearn",
            "parameters": {"C": C, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_knn(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        K-Nearest Neighbors для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Создаем метки для обучения
        labels_train = np.zeros(features.shape[0])
        train_mask = np.zeros((h, w), dtype=bool)

        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        cv2.rectangle(
            train_mask,
            (center_w - obj_size, center_h - obj_size),
            (center_w + obj_size, center_h + obj_size),
            True,
            -1,
        )

        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h),
        ]

        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True

        labels_train = train_mask.ravel().astype(int)
        train_indices = np.where(labels_train >= 0)[0]
        X_train = features[train_indices]
        y_train = labels_train[train_indices]

        n_neighbors = (self.params.get("n_neighbors", 5),)
        weights = (self.params.get("weights", "uniform"),)

        # Обучаем KNN
        knn = KNeighborsClassifier(
            n_neighbors, weights, algorithm="auto", leaf_size=30, n_jobs=-1
        )

        knn.fit(X_train, y_train)

        # Предсказываем
        labels = knn.predict(features)

        # Создаем маску
        mask = labels.reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)

        exec_time = time.time() - start_time

        info = {
            "method": "knn_sklearn",
            "parameters": {"n_neighbors": n_neighbors, "weights": weights, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    # ============ МЕТОДЫ ОБНАРУЖЕНИЯ АНОМАЛИЙ ============

    def _sklearn_isolation_forest(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Isolation Forest для сегментации (объект как аномалия).

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        n_estimators = (self.params.get("n_estimators", 100),)
        contamination = (self.params.get("contamination", "auto"),)

        # Применяем Isolation Forest
        iso_forest = IsolationForest(
            n_estimators, contamination, max_samples="auto", random_state=42, n_jobs=-1
        )

        # Предсказываем аномалии (-1 - аномалия, 1 - норма)
        labels = iso_forest.fit_predict(features)

        # Преобразуем в маску (аномалии = объект)
        mask = (labels == -1).reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)

        exec_time = time.time() - start_time

        info = {
            "method": "isolation_forest_sklearn",
            "parameters": {
                "n_estimators": n_estimators,
                "contamination": contamination,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_local_outlier_factor(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Local Outlier Factor для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Ограничиваем размер для производительности
        max_samples = 2000
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            sample_features = features[indices]
            use_sampling = True
        else:
            sample_features = features
            use_sampling = False

        n_neighbors = (self.params.get("n_neighbors", 20),)
        contamination = (self.params.get("contamination", "auto"),)

        # Применяем LOF
        lof = LocalOutlierFactor(n_neighbors, contamination, novelty=False, n_jobs=-1)

        if use_sampling:
            labels_sample = lof.fit_predict(sample_features)
            labels = self._interpolate_labels(
                sample_features, features, labels_sample, method="knn"
            )
        else:
            labels = lof.fit_predict(features)

        # Преобразуем в маску
        mask = (labels == -1).reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)

        exec_time = time.time() - start_time

        info = {
            "method": "local_outlier_factor_sklearn",
            "parameters": {
                "n_neighbors": n_neighbors,
                "contamination": contamination,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_one_class_svm(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        One-Class SVM для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Обучаем на центральной области (предполагаем, что это фон)
        center_mask = np.zeros((h, w), dtype=bool)
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4

        cv2.rectangle(
            center_mask,
            (center_w - obj_size, center_h - obj_size),
            (center_w + obj_size, center_h + obj_size),
            True,
            -1,
        )

        # Выбираем пиксели фона (все кроме центра)
        background_mask = ~center_mask
        X_train = features[background_mask.ravel()]

        kernel = (self.params.get("kernel", "rbf"),)
        gamma = (self.params.get("gamma", "auto"),)
        nu = self.params.get("nu", 0.1)

        # Обучаем One-Class SVM
        oc_svm = OneClassSVM(kernel, gamma, nu)

        oc_svm.fit(X_train)

        # Предсказываем для всех пикселей
        labels = oc_svm.predict(features)

        # Преобразуем в маску (объект = -1)
        mask = (labels == -1).reshape(h, w).astype(np.uint8) * 255
        mask = self._postprocess_mask(mask)

        exec_time = time.time() - start_time

        info = {
            "method": "one_class_svm_sklearn",
            "parameters": {"kernel": kernel, "gamma": gamma, "nu": nu, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    # ============ МЕТОДЫ РАЗЛОЖЕНИЯ ============

    def _sklearn_pca_segmentation(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        PCA-based сегментация.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Применяем PCA
        n_components = self.params.get("n_components", 3)
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(features)

        # Кластеризуем в новом пространстве
        n_clusters = self.params.get("n_clusters", 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(transformed)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "pca_segmentation_sklearn",
            "parameters": {
                "n_components": n_components,
                "n_clusters": n_clusters,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_nmf_segmentation(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Non-negative Matrix Factorization для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Убедимся, что данные неотрицательные
        features_nonneg = features - features.min()

        # Применяем NMF
        n_components = self.params.get("n_components", 3)
        nmf = NMF(n_components=n_components, init="random", random_state=42)

        # Преобразуем данные
        transformed = nmf.fit_transform(features_nonneg)

        # Кластеризуем
        n_clusters = self.params.get("n_clusters", 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(transformed)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "nmf_segmentation_sklearn",
            "parameters": {
                "n_components": n_components,
                "n_clusters": n_clusters,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_tsne_segmentation(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        t-SNE для сегментации (визуализация + кластеризация).

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

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
        perplexity = self.params.get("perplexity", 30)
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
        n_clusters = self.params.get("n_clusters", 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(transformed)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "tsne_segmentation_sklearn",
            "parameters": {
                "perplexity": perplexity,
                "n_clusters": n_clusters,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    # ============ КОМБИНИРОВАННЫЕ МЕТОДЫ ============

    def _sklearn_ensemble_clustering(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Ensemble clustering (комбинация нескольких методов).

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

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
        n_clusters = self.params.get("n_clusters", 2)
        spectral = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=42
        )

        labels = spectral.fit_predict(consensus)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "ensemble_clustering_sklearn",
            "parameters": {"n_clusters": n_clusters, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_color_spatial_clustering(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Кластеризация с учетом цвета и пространственных координат.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = image.shape[:2]

        start_time = time.time()

        # Создаем признаки: цвет + пространственные координаты
        if len(image.shape) == 3:
            color = image.reshape(-1, 3).astype(np.float32) / 255.0
        else:
            color = image.reshape(-1, 1).astype(np.float32) / 255.0

        # Пространственные координаты
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        spatial = np.stack([x_coords.ravel() / w, y_coords.ravel() / h], axis=1)

        # Веса для баланса цвета и пространства
        color_weight = self.params.get("color_weight", 0.7)
        spatial_weight = self.params.get("spatial_weight", 0.3)

        # Комбинируем признаки
        features = np.hstack([color * color_weight, spatial * spatial_weight])

        # Масштабируем
        features = self._scaler.fit_transform(features)

        # Кластеризуем
        n_clusters = self.params.get("n_clusters", 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8)

        info = {
            "method": "color_spatial_clustering_sklearn",
            "parameters": {
                "n_clusters": n_clusters,
                "color_weight": color_weight,
                "spatial_weight": spatial_weight,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    # ============ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ АВТОМАТИЧЕСКОЙ НАСТРОЙКИ ============

    def _estimate_optimal_clusters(
        self, features: np.ndarray, max_k: int = 10, **kwargs
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
        elbow_point: int = 2
        if len(inertias) >= 3:
            # Вычисляем вторую производную
            derivatives = np.diff(inertias)
            second_derivatives = np.diff(derivatives)

            if len(second_derivatives) > 0:
                elbow_point = int(np.argmax(np.abs(second_derivatives))) + 2

        return max(2, min(elbow_point, max_k))

    def _estimate_dbscan_params(
        self, features: np.ndarray, **kwargs
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

    def _estimate_meanshift_bandwidth(self, features: np.ndarray, **kwargs) -> float:
        """Оценка bandwidth для MeanShift."""
        # Используем квантильный метод

        n_samples = min(500, features.shape[0])
        sample_indices = np.random.choice(features.shape[0], n_samples, replace=False)
        sample_features = features[sample_indices]

        nbrs = NearestNeighbors(n_neighbors=5).fit(sample_features)
        distances, _ = nbrs.kneighbors(sample_features)
        avg_distances = distances.mean(axis=1)

        quantile = self.params.get("quantile", 0.3)
        bandwidth = float(np.percentile(avg_distances, quantile * 100))

        return max(bandwidth, 0.1)

    def _estimate_gmm_components(
        self, features: np.ndarray, max_components: int = 10, **kwargs
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
            except Exception:
                bics.append(np.inf)

        if bics:
            optimal_n = n_components_range[np.argmin(bics)]
        else:
            optimal_n = 2

        return max(2, optimal_n)

    def _interpolate_labels(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray,
        train_labels: np.ndarray,
        method: str = "knn",
        **kwargs,
    ) -> np.ndarray:
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
        if method == "knn":
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

    def _sklearn_decision_tree(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decision Tree для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Создаем метки для обучения
        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]

        X_train = features[train_indices]
        y_train = labels_train[train_indices]

        max_depth = (self.params.get("max_depth", None),)
        min_samples_split = (self.params.get("min_samples_split", 2),)

        # Обучаем Decision Tree
        dt = DecisionTreeClassifier(max_depth, min_samples_split, random_state=42)

        dt.fit(X_train, y_train)
        labels = dt.predict(features)

        mask = labels.reshape(h, w).astype(np.uint8) * 255

        exec_time = time.time() - start_time
        mask = self._postprocess_mask(mask)
        info = {
            "method": "decision_tree_sklearn",
            "parameters": {
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_mlp(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Multi-layer Perceptron для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]

        X_train = features[train_indices]
        y_train = labels_train[train_indices]
        hidden_layer_sizes = self.params.get("hidden_layer_sizes", (100, 50))

        # Обучаем MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            random_state=42,
            max_iter=500,
        )

        mlp.fit(X_train, y_train)
        labels = mlp.predict(features)

        mask = labels.reshape(h, w).astype(np.uint8) * 255
        exec_time = time.time() - start_time
        mask = self._postprocess_mask(mask)

        info = {
            "method": "mlp_sklearn",
            "parameters": {"hidden_layer_sizes": hidden_layer_sizes, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_naive_bayes(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Naive Bayes для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]

        X_train = features[train_indices]
        y_train = labels_train[train_indices]

        # Обучаем Gaussian Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        labels = nb.predict(features)

        mask = labels.reshape(h, w).astype(np.uint8) * 255
        exec_time = time.time() - start_time
        mask = self._postprocess_mask(mask)

        info = {
            "method": "naive_bayes_sklearn",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_lda(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Linear Discriminant Analysis для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]

        X_train = features[train_indices]
        y_train = labels_train[train_indices]

        # Обучаем LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        labels = lda.predict(features)

        mask = labels.reshape(h, w).astype(np.uint8) * 255
        exec_time = time.time() - start_time
        mask = self._postprocess_mask(mask)

        info = {
            "method": "lda_sklearn",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _create_training_labels(self, h: int, w: int, **kwargs) -> np.ndarray:
        """Создание меток для обучения."""
        labels_train = -np.ones(h * w)  # -1 означает непомеченный

        # Создаем маску для обучения
        train_mask = np.zeros((h, w), dtype=bool)

        # Центральная область - объект (класс 1)
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        cv2.rectangle(
            train_mask,
            (center_w - obj_size, center_h - obj_size),
            (center_w + obj_size, center_h + obj_size),
            True,
            -1,
        )

        # Углы - фон (класс 0)
        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h),
        ]

        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True

        labels_train = train_mask.ravel().astype(int)
        # Непомеченные пиксели остаются -1

        return labels_train

    # ============ ОСТАЛЬНЫЕ МЕТОДЫ (заглушки для полноты API) ============

    def _sklearn_elliptic_envelope(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Elliptic Envelope для сегментации."""
        return self._sklearn_isolation_forest(image)

    def _sklearn_bayesian_gmm(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Bayesian GMM для сегментации."""
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        n_components = (self.params.get("n_components", 10),)
        bgmm = BayesianGaussianMixture(n_components=n_components, random_state=42)

        labels = bgmm.fit_predict(features)
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = self._postprocess_mask(mask)

        info = {
            "method": "bayesian_gmm_sklearn",
            "parameters": {"n_components": n_components, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_ica_segmentation(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ICA для сегментации."""
        return self._sklearn_pca_segmentation(image)

    def _sklearn_isomap_segmentation(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Isomap для сегментации."""
        return self._sklearn_tsne_segmentation(image)

    def _sklearn_spectral_embedding(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Spectral Embedding для сегментации."""
        return self._sklearn_spectral(image)

    def _sklearn_variational_gmm(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Variational GMM для сегментации."""
        return self._sklearn_bayesian_gmm(image)

    def _sklearn_density_based(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Density-based кластеризация."""
        return self._sklearn_dbscan(image)

    def _sklearn_hdbscan_emulation(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Эмуляция HDBSCAN."""
        return self._sklearn_optics(image)

    def _sklearn_graph_clustering(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Graph-based кластеризация."""
        return self._sklearn_spectral(image)

    def _sklearn_modularity_clustering(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Modularity-based кластеризация."""
        return self._sklearn_spectral(image)

    def _sklearn_self_training(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Self-training для сегментации."""
        return self._sklearn_random_forest(image)

    def _sklearn_semi_supervised(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Semi-supervised сегментация."""
        return self._sklearn_random_forest(image)

    def _sklearn_distance_matrix(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Distance matrix-based кластеризация."""
        return self._sklearn_spectral(image)

    def _sklearn_affinity_propagation(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Affinity Propagation."""
        features = self._extract_features(image)
        h, w = image.shape[:2]

        start_time = time.time()

        # Используем K-Means как альтернативу (Affinity Propagation требователен)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(features)

        mask = self._create_mask_from_labels(labels, (h, w))
        exec_time = time.time() - start_time
        mask = self._postprocess_mask(mask)

        info = {
            "method": "affinity_propagation_sklearn",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_qda(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quadratic Discriminant Analysis."""
        return self._sklearn_lda(image)

    def _sklearn_texture_clustering(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Текстурная кластеризация."""
        return self._sklearn_color_spatial_clustering(image)

    def _sklearn_superpixel_clustering(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Кластеризация суперпикселей."""
        return self._sklearn_color_spatial_clustering(image)

    def _sklearn_hierarchical_kmeans(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Иерархический K-Means."""
        return self._sklearn_agglomerative(image)

    def _sklearn_pca_kmeans(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """PCA + K-Means."""
        return self._sklearn_pca_segmentation(image)

    def _sklearn_gmm_vers2(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Gaussian Mixture Models."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        h, w = gray.shape

        start_time = time.time()

        # Подготовка признаков
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack(
            [gray.ravel(), x_coords.ravel() / w, y_coords.ravel() / h]
        )

        # Применяем GMM
        n_components = self.params.get("n_components", 3)
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(features)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8) * 255

        info = {
            "method": "gmm_sklearn",
            "parameters": {"n_components": n_components, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_agglomerative_vers2(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Agglomerative Clustering."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        h, w = gray.shape

        start_time = time.time()

        # Для производительности сэмплируем
        sample_size = min(1000, h * w)
        indices = np.random.choice(h * w, sample_size, replace=False)

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack(
            [
                gray.ravel()[indices],
                x_coords.ravel()[indices] / w,
                y_coords.ravel()[indices] / h,
            ]
        )

        # Применяем Agglomerative Clustering
        n_clusters = self.params.get("n_clusters", 3)
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels_sample = agg.fit_predict(features)

        # Интерполируем на все пиксели
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, labels_sample)

        all_features = np.column_stack(
            [gray.ravel(), x_coords.ravel() / w, y_coords.ravel() / h]
        )

        labels = knn.predict(all_features)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8) * 255

        info = {
            "method": "agglomerative_sklearn",
            "parameters": {"n_clusters": n_clusters, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_spectral_vers2(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Spectral Clustering."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        h, w = gray.shape

        start_time = time.time()

        # Для производительности - сильное сэмплирование
        sample_size = min(500, h * w)
        indices = np.random.choice(h * w, sample_size, replace=False)

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack(
            [
                gray.ravel()[indices],
                x_coords.ravel()[indices] / w,
                y_coords.ravel()[indices] / h,
            ]
        )

        # Применяем Spectral Clustering
        n_clusters = self.params.get("n_clusters", 3)
        spectral = SpectralClustering(
            n_clusters=n_clusters, affinity="nearest_neighbors"
        )
        labels_sample = spectral.fit_predict(features)

        # Интерполируем
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, labels_sample)

        all_features = np.column_stack(
            [gray.ravel(), x_coords.ravel() / w, y_coords.ravel() / h]
        )

        labels = knn.predict(all_features)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8) * 255

        info = {
            "method": "spectral_sklearn",
            "parameters": {"n_clusters": n_clusters, **kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_isolation_forest_vers2(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Isolation Forest для сегментации."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        h, w = gray.shape

        start_time = time.time()

        # Подготовка признаков
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack(
            [gray.ravel(), x_coords.ravel() / w, y_coords.ravel() / h]
        )

        # Применяем Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        labels = iso_forest.fit_predict(features)

        # Аномалии = объект
        mask = (labels == -1).reshape(h, w)

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8) * 255

        info = {
            "method": "isolation_forest_sklearn",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_random_forest_vers2(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Random Forest для сегментации."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        h, w = gray.shape

        start_time = time.time()

        # Создаем метки для обучения
        labels_train = -np.ones(h * w)
        train_mask = np.zeros((h, w), dtype=bool)

        # Центральная область - объект
        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        train_mask[
            (center_h - obj_size):(center_h + obj_size),
            (center_w - obj_size):(center_w + obj_size),
        ] = True
        labels_train[train_mask.ravel()] = 1

        # Углы - фон
        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h),
        ]

        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True
            labels_train[train_mask[y1:y2, x1:x2].ravel()] = 0

        # Подготовка признаков
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack(
            [gray.ravel(), x_coords.ravel() / w, y_coords.ravel() / h]
        )

        # Обучаем Random Forest
        train_indices = labels_train >= 0
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(features[train_indices], labels_train[train_indices])

        # Предсказываем
        labels = rf.predict(features)
        mask = labels.reshape(h, w)

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8) * 255

        info = {
            "method": "random_forest_sklearn",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_svm_vers2(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """SVM для сегментации."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        h, w = gray.shape

        start_time = time.time()

        # Создаем метки для обучения (как в Random Forest)
        labels_train = -np.ones(h * w)
        train_mask = np.zeros((h, w), dtype=bool)

        center_h, center_w = h // 2, w // 2
        obj_size = min(h, w) // 4
        train_mask[
            (center_h - obj_size):(center_h + obj_size),
            (center_w - obj_size):(center_w + obj_size),
        ] = True
        labels_train[train_mask.ravel()] = 1

        corner_size = min(h, w) // 8
        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h),
        ]

        for x1, y1, x2, y2 in corners:
            train_mask[y1:y2, x1:x2] = True
            labels_train[train_mask[y1:y2, x1:x2].ravel()] = 0

        # Подготовка признаков
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack(
            [gray.ravel(), x_coords.ravel() / w, y_coords.ravel() / h]
        )

        # Обучаем SVM
        train_indices = labels_train >= 0
        svm = SVC(kernel="rbf", probability=True, random_state=42)
        svm.fit(features[train_indices], labels_train[train_indices])

        # Предсказываем
        labels = svm.predict(features)
        mask = labels.reshape(h, w)

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8) * 255

        info = {
            "method": "svm_sklearn",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    def _sklearn_pca_segmentation_vers2(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """PCA-based сегментация."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        h, w = gray.shape

        start_time = time.time()

        # Подготовка признаков
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack(
            [gray.ravel(), x_coords.ravel() / w, y_coords.ravel() / h]
        )

        # Применяем PCA
        pca = PCA(n_components=2, random_state=42)
        transformed = pca.fit_transform(features)

        # Кластеризуем в новом пространстве
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(transformed)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8) * 255

        info = {
            "method": "psa_segmentation_sklearn",
            "parameters": {**kwargs},
            "execution_time": exec_time,
        }

        return mask, info

    # ============ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ============

    def _create_mask_from_labels_vers2(
        self, labels: np.ndarray, shape: Tuple, **kwargs
    ) -> np.ndarray:
        """Создание бинарной маски из меток."""
        labels_2d = labels.reshape(shape)
        unique_labels = np.unique(labels)

        # Исключаем шум (-1) если есть
        valid_labels = unique_labels[unique_labels != -1]

        if len(valid_labels) == 0:
            return np.zeros(shape, dtype=bool)

        # Находим самый большой кластер как фон
        label_sizes = [np.sum(labels_2d == label1) for label1 in valid_labels]
        bg_label = valid_labels[np.argmax(label_sizes)]

        # Создаем маску (все кроме фона)
        mask = labels_2d != bg_label

        return mask

    # ============================================================================
    # ДОБАВИТЬ/ОБНОВИТЬ МЕТОДЫ В КЛАСС SklearnSegmenter:
    # ============================================================================

    def _sklearn_floodfill(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

            start_time = time.time()
            seed = self.params.get("seed", (w // 2, h // 2))
            tolerance = self.params.get("tolerance", 0.1)

            # Используем flood из skimage
            mask = segmentation.flood(gray, seed_point=seed[::-1], tolerance=tolerance)

            exec_time = time.time() - start_time
            mask = mask.astype(np.uint8) * 255

            info = {
                "method": "floodfill_sklearn",
                "parameters": {"seed": seed, "tolerance": tolerance, **kwargs},
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"FloodFill failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    def _sklearn_active_contour(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            start_time = time.time()

            # Создаём начальный контур (окружность в центре)
            center_x, center_y = w // 2, h // 2
            radius = min(center_x, center_y) // 2
            s = np.linspace(0, 2 * np.pi, 100)
            init = np.array(
                [center_x + radius * np.cos(s), center_y + radius * np.sin(s)]
            ).T

            # Параметры
            alpha = self.params.get("alpha", 0.015)
            beta = self.params.get("beta", 10)
            gamma = self.params.get("gamma", 0.001)
            w_edge = self.params.get("w_edge", 1)
            w_line = self.params.get("w_line", 0)
            max_iter = self.params.get("max_iter", 1000)

            # Применяем active_contour
            snake = active_contour(
                gaussian(gray_norm, 3),
                init,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                w_edge=w_edge,
                w_line=w_line,
                max_num_iter=max_iter,
            )

            # Создаём маску из контура
            mask = np.zeros((h, w), dtype=bool)
            rr, cc = polygon(snake[:, 1], snake[:, 0], mask.shape)
            mask[rr, cc] = True

            exec_time = time.time() - start_time
            mask = mask.astype(np.uint8) * 255

            info = {
                "method": "active_contour_sklearn",
                "parameters": {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "w_edge": w_edge,
                    "w_line": w_line,
                    "max_iter": max_iter,
                    **kwargs,
                },
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"Active Contour failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    def _sklearn_gvf_contour(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            start_time = time.time()

            # Вычисляем градиенты
            grad_x = filters.sobel_h(gray_norm)
            grad_y = filters.sobel_v(gray_norm)

            # Параметры GVF
            mu = self.params.get("mu", 0.1)
            iterations = self.params.get("iterations", 50)

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

            exec_time = time.time() - start_time
            mask = mask.astype(np.uint8) * 255

            info = {
                "method": "gvf_sklearn",
                "parameters": {"mu": mu, "iterations": iterations, **kwargs},
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"GVF Contour failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    def _sklearn_morphological_snakes(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            start_time = time.time()

            # Начальный уровень (прямоугольник в центре)
            init_level_set = np.zeros(gray_norm.shape, dtype=np.int8)
            init_level_set[(h // 4):(3 * h // 4), (w // 4):(3 * w // 4)] = 1

            # Параметры
            iterations = self.params.get("iterations", 50)
            smoothing = self.params.get("smoothing", 1)
            threshold = self.params.get("threshold", 0.5)
            balloon = self.params.get("balloon", 1)

            # Применяем морфологические змеи
            mask = morphological_geodesic_active_contour(
                gray_norm,
                iterations,
                init_level_set=init_level_set,
                smoothing=smoothing,
                threshold=threshold,
                balloon=balloon,
            )

            exec_time = time.time() - start_time
            mask = mask.astype(np.uint8) * 255

            info = {
                "method": "morphological_snakes_sklearn",
                "parameters": {
                    "smoothing": smoothing,
                    "threshold": threshold,
                    "smoothing": smoothing,
                    "balloon": balloon,
                    **kwargs,
                },
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"Morphological Snakes failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    def _sklearn_chan_vese(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            start_time = time.time()

            # Начальный уровень
            init_level_set = np.zeros(gray_norm.shape, dtype=np.int8)
            init_level_set[(h // 4):(3 * h // 4), (w // 4):(3 * w // 4)] = 1

            # Параметры
            mu = self.params.get("mu", 0.25)
            lambda1 = self.params.get("lambda1", 1.0)
            lambda2 = self.params.get("lambda2", 1.0)
            tol = self.params.get("tol", 1e-3)
            max_iter = self.params.get("max_iter", 100)

            # Применяем Chan-Vese
            mask = morphological_chan_vese(
                gray_norm,
                max_iter,
                init_level_set=init_level_set,
                smoothing=1,
                lambda1=lambda1,
                lambda2=lambda2,
            )
            exec_time = time.time() - start_time
            mask = mask.astype(np.uint8) * 255

            info = {
                "method": "chan_vese_sklearn",
                "parameters": {
                    "mu": mu,
                    "lambda1": lambda1,
                    "lambda2": lambda2,
                    "tol": tol,
                    "max_iter": max_iter,
                    **kwargs,
                },
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"Chan-Vese failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    def _sklearn_watershed(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            start_time = time.time()

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
            exec_time = time.time() - start_time
            mask = mask.astype(np.uint8) * 255

            info = {
                "method": "watershed_sklearn",
                "parameters": {**kwargs},
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"Watershed failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    def _sklearn_random_walker(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            start_time = time.time()

            # Создаём маркеры
            markers = np.zeros(gray.shape, dtype=np.uint8)

            # Центральная область - объект (маркер 2)
            markers[(h // 4):(3 * h // 4), (w // 4):(3 * w // 4)] = 2

            # Углы - фон (маркер 1)
            corner_size = min(h, w) // 8
            markers[:corner_size, :corner_size] = 1
            markers[:corner_size, -corner_size:] = 1
            markers[-corner_size:, :corner_size] = 1
            markers[-corner_size:, -corner_size:] = 1

            # Параметры
            beta = self.params.get("beta", 130)
            mode = self.params.get("mode", "cg_mg")

            # Применяем Random Walker
            labels = random_walker(gray_norm, markers, beta=beta, mode=mode)

            # Создаём маску (объект = маркер 2)
            mask = labels == 2
            exec_time = time.time() - start_time
            mask = mask.astype(np.uint8) * 255

            info = {
                "method": "random_walker_sklearn",
                "parameters": {"beta": beta, "mode": mode, **kwargs},
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"Random Walker failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    def _sklearn_quickshift(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            start_time = time.time()

            # Параметры
            kernel_size = self.params.get("kernel_size", 3)
            max_dist = self.params.get("max_dist", 6)
            ratio = self.params.get("ratio", 0.5)

            # Применяем Quickshift
            segments = quickshift(
                img_rgb, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio
            )

            # Находим самый большой сегмент (фон)
            unique, counts = np.unique(segments, return_counts=True)
            bg_label = unique[np.argmax(counts)]

            # Создаём маску (все кроме фона)
            mask = segments != bg_label
            exec_time = time.time() - start_time
            mask = mask.astype(np.uint8) * 255

            info = {
                "method": "quickshift_sklearn",
                "parameters": {
                    "kernel_size": kernel_size,
                    "max_dist": max_dist,
                    "ratio": ratio,
                    **kwargs,
                },
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"Quickshift failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)

    def _sklearn_slic(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            start_time = time.time()

            # Параметры
            n_segments = self.params.get("n_segments", 100)
            compactness = self.params.get("compactness", 10.0)
            max_iter = self.params.get("max_iter", 10)
            enforce_connectivity = self.params.get("enforce_connectivity", True)

            # Применяем SLIC
            segments = slic(
                img_rgb,
                n_segments=n_segments,
                compactness=compactness,
                max_num_iter=max_iter,
                enforce_connectivity=enforce_connectivity,
                start_label=0,
            )

            # Находим самый большой сегмент (фон)
            unique, counts = np.unique(segments, return_counts=True)
            bg_label = unique[np.argmax(counts)]

            # Создаём маску (все кроме фона)
            mask = segments != bg_label
            exec_time = time.time() - start_time
            mask = mask.astype(np.uint8) * 255

            info = {
                "method": "slic_sklearn",
                "parameters": {
                    "n_segments": n_segments,
                    "compactness": compactness,
                    "max_iter": max_iter,
                    "enforce_connectivity": enforce_connectivity,
                    **kwargs,
                },
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"SLIC failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)

    def _sklearn_felzenszwalb(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            start_time = time.time()

            # Параметры
            scale = self.params.get("scale", 100)
            sigma = self.params.get("sigma", 0.8)
            min_size = self.params.get("min_size", 50)

            # Применяем Felzenszwalb
            segments = felzenszwalb(
                img_rgb, scale=scale, sigma=sigma, min_size=min_size
            )

            # Находим самый большой сегмент (фон)
            unique, counts = np.unique(segments, return_counts=True)
            bg_label = unique[np.argmax(counts)]

            # Создаём маску (все кроме фона)
            mask = segments != bg_label
            exec_time = time.time() - start_time
            mask = mask.astype(np.uint8) * 255

            info = {
                "method": "felzenszwalb_sklearn",
                "parameters": {
                    "scale": scale,
                    "sigma": sigma,
                    "min_size": min_size,
                    **kwargs,
                },
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"Felzenszwalb failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)

    def _sklearn_grabcut(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
            start_time = time.time()

            # Создаём начальную маску с маркерами
            mask = np.zeros((h, w), dtype=np.uint8)

            # Прямоугольник в центре - вероятный передний план
            rect = self.params.get("rect", (w // 4, h // 4, w // 2, h // 2))
            x, y, rw, rh = rect
            mask[y:(y + rh), x:(x + rw)] = 3  # Вероятный передний план

            # Углы - определённый фон
            corner_size = min(h, w) // 8
            mask[:corner_size, :corner_size] = 0
            mask[:corner_size, -corner_size:] = 0
            mask[-corner_size:, :corner_size] = 0
            mask[-corner_size:, -corner_size:] = 0

            # Подготовка признаков
            pixels = img_rgb.reshape(-1, 3).astype(np.float32) / 255.0
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            coords = np.column_stack([x_coords.ravel() / w, y_coords.ravel() / h])
            features = np.hstack([pixels, coords])

            # Выбираем пиксели для обучения
            train_mask = (mask.ravel() == 0) | (mask.ravel() == 3)
            X_train = features[train_mask]
            y_train = (mask.ravel()[train_mask] == 3).astype(int)

            # Обучаем Random Forest
            n_estimators = self.params.get("n_estimators", 50)
            rf = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)

            # Предсказываем для всех пикселей
            labels = rf.predict(features)
            mask_result = labels.reshape(h, w)
            exec_time = time.time() - start_time
            mask = mask_result.astype(np.uint8) * 255

            info = {
                "method": "grabcut_sklearn",
                "parameters": {"rect": rect, "n_estimators": n_estimators, **kwargs},
                "execution_time": exec_time,
            }

            return mask, info

        except Exception as e:
            warnings.warn(f"GrabCut failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)


# methods_to_test_sklearn = [
#         ("global_thresholding", {"threshold": 0.5}),
#         ("otsu_thresholding", {}),
#         ("canny_edge", {"sigma": 1.0}),
#         ("kmeans_segmentation", {"k": 2}),
#         ("watershed", {}),
#         ("slic", {"n_segments": 50}),
#         ("kmeans", {"n_clusters": 3}),
#         ("dbscan", {"eps": "auto", "min_samples": "auto"}),
#         ("meanshift", {"bandwidth": None}),
#         ("gmm", {"n_components": 3}),
#         ("random_forest", {"n_estimators": 50}),
#         ("svm", {"C": 1.0, "kernel": "rbf"}),
#         ("isolation_forest", {"n_estimators": 100}),
#         ("pca_segmentation", {"n_components": 3}),
#         ("color_spatial_clustering", {"color_weight": 0.7, "spatial_weight": 0.3}),
#     ]
