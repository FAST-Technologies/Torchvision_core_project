# segmenters/SklearnSegmenter.py

"""Класс для сегментации изображений с использованием scikit-learn и scikit-image.

Поддерживает 80+ методов сегментации, сгруппированных по 10 категориям:
1. **Пороговые методы** (13): глобальный порог, Оцу, адаптивные методы
2. **Детекторы границ** (10): Собель, Кэнни, Лапласиан, фазовая конгруэнтность
3. **Региональные методы** (3): рост регионов, split-and-merge, floodfill
4. **Кластеризация** (15+): K-Means, DBSCAN, GMM, Spectral, OPTICS
5. **Активные контуры** (4): Snakes, GVF, морфологические змеи, Чан-Везе
6. **Watershed и графовые** (2): классический watershed, random walker
7. **Суперпиксели** (3): SLIC, Felzenszwalb, QuickShift
8. **ML-классификаторы** (10+): Random Forest, SVM, MLP, Naive Bayes, LDA
9. **Обнаружение аномалий** (4): Isolation Forest, LOF, One-Class SVM
10. **Разложение и многообразия** (6): PCA, NMF, t-SNE, Isomap

Все методы возвращают:
- `segment()`: бинарную маску `MaskArray` (0/255)
- `segment_with_mask()`: кортеж `(визуализация, маска)`

Quick Start:
```python
from segmenters.SklearnSegmenter import SklearnSegmenter

# Пороговая сегментация
segmenter = SklearnSegmenter("otsu_thresholding")
mask = segmenter.segment("image.jpg")

# K-Means кластеризация
segmenter = SklearnSegmenter("kmeans_segmentation", k=3)
overlay, mask = segmenter.segment_with_mask(image, alpha=0.7)

# С метриками (при наличии GT)
metrics, pred = segmenter.segment_and_evaluate(image, gt_mask=ground_truth)
print(f"IoU: {metrics['iou']:.3f}")
```

Attributes:
    method (str): Название текущего метода сегментации.
    params (Dict[str, Any]): Словарь параметров метода.
    model_name (str): Уникальное имя модели для логирования.
    methods (Dict[str, SegmentationFunc]): Словарь зарегистрированных методов.
    _scaler (StandardScaler): Скалер для нормализации признаков.
    _needs_gray (bool): Флаг необходимости конвертации в grayscale.

Note:
    - Для методов кластеризации используется автоматическое извлечение признаков:
        цвет (RGB/Lab) + пространственные координаты + текстура (градиенты).
    - Все изображения нормализуются к [0, 1] для совместимости с scikit-image.
    - Методы, требующие обучения (Random Forest, SVM), используют автоматическую
        генерацию меток: центр = объект, углы = фон.
    - Для больших изображений (>1000×1000) методы кластеризации используют
        сэмплирование для ускорения.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

from segmenters.BaseSegmenter import BaseSegmenter
from typing import List, Union, Tuple, Dict, Any, Optional, Callable, Literal, cast
import numpy as np
import numpy.typing as npt
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
from scipy.ndimage import gaussian_filter
from skimage.util import img_as_float

# Импорт scikit-image компонентов
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
    laplace,
)
from skimage.morphology import (
    disk,
    opening,
    closing,
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
)
import cv2
import torch
from typing_extensions import TypeAlias

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

SKIMAGE_AVAILABLE = True

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
# Типы для изображений
ImageArray = npt.NDArray[np.uint8]
"""Тип для входного изображения (RGB или grayscale), dtype=uint8."""

GrayImage = npt.NDArray[np.uint8]
"""Тип для grayscale изображения, форма (H, W), dtype=uint8."""

MaskArray = npt.NDArray[np.uint8]
"""Тип для бинарной маски, форма (H, W), dtype=uint8, значения {0, 255}."""

FloatArray = npt.NDArray[np.float32]
"""Тип для массивов с плавающей точкой, форма (H, W) или (N, D)."""

NormalizedArray = npt.NDArray[np.float32]
"""Тип для нормализованных изображений [0, 1], dtype=float32."""

ImagePath: TypeAlias = str
NumpyImage: TypeAlias = np.ndarray
PILImage: TypeAlias = Image.Image
TorchImage: TypeAlias = torch.Tensor
ImageInput: TypeAlias = Union[ImagePath, NumpyImage, PILImage, TorchImage]
"""Поддерживаемые форматы входного изображения."""

# Типы для цветовых пространств
ColorSpace = Literal["RGB", "BGR", "GRAY", "LAB"]
"""Поддерживаемые цветовые пространства."""
ColorChannel = Literal[1, 3]
OverlayColor: TypeAlias = Tuple[int, int, int]

# Типы для масок
Mask: TypeAlias = np.ndarray
BinaryMask: TypeAlias = np.ndarray  # shape: (H, W), dtype: uint8, значения: 0 или 255
ProbabilityMask: TypeAlias = np.ndarray  # shape: (H, W), dtype: float32, значения: 0-1

# Типы для метаданных
MetricsDict = Dict[str, float]
"""Словарь метрик сегментации."""

SegmentationInfo = Dict[str, Any]
"""Метаданные выполнения метода сегментации."""

# Тип для функции сегментации
SegmentationFunc = Callable[[ImageArray, Any], Tuple[MaskArray, SegmentationInfo]]
"""Сигнатура функции сегментации."""

# Generic type для кластеров
ClusterLabels = npt.NDArray[np.int32]
"""Метки кластеров, форма (N,) или (H, W), dtype=int32."""


# ──────────────────────────────────────────────────────────────────────
class SklearnSegmenter(BaseSegmenter):
    """Класс для сегментации изображений с использованием scikit-learn и scikit-image.

    Поддерживает 80+ методов сегментации, сгруппированных по категориям:
    1. **Пороговые методы** (13 вариантов): глобальный порог, Оцу, адаптивные методы
    2. **Детекторы границ** (10 вариантов): Собель, Кэнни, Лапласиан, фазовая конгруэнтность
    3. **Региональные методы** (3 варианта): рост регионов, split-and-merge, floodfill
    4. **Кластеризация** (15+ вариантов): K-Means, DBSCAN, GMM, Spectral, OPTICS
    5. **Активные контуры** (4 варианта): Snakes, GVF, морфологические змеи, Чан-Везе
    6. **Watershed и графовые** (2 варианта): классический watershed, random walker
    7. **Суперпиксели** (3 варианта): SLIC, Felzenszwalb, QuickShift
    8. **ML-классификаторы** (10+ вариантов): Random Forest, SVM, MLP, Naive Bayes
    9. **Обнаружение аномалий** (4 варианта): Isolation Forest, LOF, One-Class SVM
    10. **Разложение и многообразия** (6 вариантов): PCA, NMF, t-SNE, Isomap

    Все методы возвращают:
    - `segment()`: бинарную маску `MaskArray` (0/255)
    - `segment_with_mask()`: кортеж `(визуализация, маска)`

    Attributes:
        method (str): Название текущего метода сегментации.
        params (Dict[str, Any]): Словарь параметров метода.
        model_name (str): Уникальное имя модели для логирования.
        methods (Dict[str, SegmentationFunc]): Словарь зарегистрированных методов.
        _scaler (StandardScaler): Скалер для нормализации признаков.
        _needs_gray (bool): Флаг необходимости конвертации в grayscale.

    Example:
        ```python
        from segmenters.SklearnSegmenter import SklearnSegmenter
        import cv2

        # Загрузка изображения
        image = cv2.imread("sample.jpg")

        # Сегментация методом Оцу
        segmenter = SklearnSegmenter("otsu_thresholding")
        mask = segmenter.segment(image)

        # K-Means кластеризация с 3 кластерами
        segmenter = SklearnSegmenter("kmeans_segmentation", k=3)
        mask = segmenter.segment(image)

        # С возвратом визуализации
        overlay, mask = segmenter.segment_with_mask(image, alpha=0.7)

        # Получение метрик при наличии GT
        from metrics.SegmentationMetrics import SegmentationMetrics
        metrics, pred_mask = segmenter.segment_and_evaluate(
            image, gt_mask=ground_truth, threshold=0.5
        )
        print(f"IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}")
        ```

    Note:
        - Для методов кластеризации используется автоматическое извлечение признаков:
          цвет (RGB/Lab) + пространственные координаты + текстура (градиенты).
        - Все изображения нормализуются к [0, 1] для совместимости с scikit-image.
        - Методы, требующие обучения (Random Forest, SVM), используют автоматическую
          генерацию меток: центр = объект, углы = фон.
        - Для больших изображений (>1000×1000) методы кластеризации используют
          сэмплирование для ускорения.
    """

    def __init__(self, method: str = "global_thresholding", **kwargs: Any) -> None:
        """Инициализация класса SklearnSegmenter."""
        super().__init__()
        self.method: str = method
        self.params: Dict[str, Any] = kwargs
        self.info: SegmentationInfo = {}
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

    # ──────────────────────────────────────────────────────────────────────
    def _setup_methods(self) -> None:
        """Регистрация всех доступных методов сегментации в словаре `self.methods`.

        Методы сгруппированы по категориям:
        - Пороговые (13): от простого глобального до сложных адаптивных
        - Граничные (10): от классических операторов до фазовой конгруэнтности
        - Региональные (3): рост регионов, split-and-merge, floodfill
        - Кластеризация (15+): K-Means, DBSCAN, GMM, Spectral, OPTICS, Birch
        - Активные контуры (4): Snakes, GVF, морфологические змеи, Чан-Везе
        - Watershed (2): классический и random walker
        - Суперпиксели (3): SLIC, Felzenszwalb, QuickShift
        - ML-классификаторы (10+): Random Forest, SVM, MLP, Naive Bayes, LDA
        - Обнаружение аномалий (4): Isolation Forest, LOF, One-Class SVM
        - Разложение (6): PCA, NMF, t-SNE, Isomap, Spectral Embedding

        Raises:
            ValueError: Если `self.method` не найден в зарегистрированных методах.

        Note:
            Все методы должны соответствовать сигнатуре:
            ```python
            def method_name(self, img: ImageArray, **kwargs) -> Tuple[MaskArray, SegmentationInfo]:
                ...
                return mask, info
            ```
        """
        self.methods: Dict[str, Callable[..., Tuple[MaskArray, SegmentationInfo]]] = {
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
            available: List[str] = list(self.methods.keys())
            raise ValueError(
                f"Неизвестный метод: {self.method}. " f"Доступные методы: {available}"
            )

    # ──────────────────────────────────────────────────────────────────────
    def _log_info(
        self,
        method_name: str,
        exec_time: float,
        params: Dict[str, Any],
        **extra: Any,
    ) -> SegmentationInfo:
        """Вспомогательный метод для логирования информации о выполнении.

        Сохраняет метаданные выполнения в атрибут `self.info` и возвращает словарь.

        Args:
            method_name: Название выполненного метода (например, "global_thresholding_sklearn").
            exec_time: Время выполнения в секундах.
            params: Словарь использованных параметров метода.
            **extra: Дополнительные поля для метаданных (опционально).

        Returns:
            SegmentationInfo: Словарь с полными метаданными выполнения.

        Note:
            Метод вызывается автоматически в конце каждого приватного метода
            сегментации (например, `_sklearn_otsu_thresholding`).
            Результат можно использовать как возвращаемый `info`.

        Example:
            ```python
            exec_time = time.time() - start_time
            return mask, self._log_info(
                "otsu_sklearn",
                exec_time,
                {"threshold": thresh},
                histogram_stats={"mean": mean_val}
            )
            ```
        """
        self.info = {
            "method": method_name,
            "parameters": params,
            "execution_time": exec_time,
            **extra,  # Дополнительные поля
        }
        return self.info

    # ──────────────────────────────────────────────────────────────────────
    def _get_param(self, key: str, default: Any, **kwargs: Any) -> Any:
        """Универсальный геттер параметров с приоритетом: kwargs > self.params > default.

        Позволяет гибко переопределять параметры метода при вызове сегментации,
        сохраняя значения по умолчанию из `self.params` и fallback на `default`.

        Args:
            key: Имя искомого параметра.
            default: Значение по умолчанию, если параметр отсутствует.
            **kwargs: Дополнительные аргументы, переданные при вызове метода.

        Returns:
            Any: Значение параметра или `default`, если параметр не найден.

        Note:
            - Используется во всех внутренних методах для безопасного доступа к параметрам.
            - Гарантирует, что `kwargs` всегда имеют высший приоритет.

        Example:
            ```python
            # kwargs имеет приоритет над self.params
            thr = self._get_param("threshold", 0.5, threshold=0.7)  # вернет 0.7
            # Если kwargs пуст, берется из self.params или default
            thr = self._get_param("threshold", 0.5)  # вернет self.params.get("threshold", 0.5)
            ```
        """
        if key in kwargs:
            return kwargs[key]
        if key in self.params:
            return self.params[key]
        return default

    # ──────────────────────────────────────────────────────────────────────
    def _normalize_image(self, img: np.ndarray) -> NormalizedArray:
        """Нормализация изображения к диапазону [0, 1].

        Преобразует входной массив в `float32` и масштабирует пиксели из `[0, 255]` в `[0, 1]`,
        если изображение имеет тип `uint8`. Необходимо для совместимости с алгоритмами scikit-image.

        Args:
            img: Входное изображение формы `(H, W)` или `(H, W, C)`, dtype=uint8 или float.

        Returns:
            NormalizedArray: Нормализованное изображение, dtype=float32, диапазон [0, 1].

        Note:
            - Если изображение уже `float`, оно просто копируется в `float32` без масштабирования.
            - Не обрабатывает каналы по отдельности; применяется глобально ко всему массиву.
            - Используется как вспомогательный шаг перед пороговыми и кластеризационными методами.

        Example:
            ```python
            img_uint8 = cv2.imread("sample.jpg")
            img_norm = self._normalize_image(img_uint8)
            print(img_norm.max())  # 1.0
            ```
        """
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    def _ensure_uint8_mask(self, mask: np.ndarray) -> MaskArray:
        """Гарантирует возврат бинарной маски в формате uint8 [0, 255].

        Автоматически конвертирует логические или float маски в `uint8`,
        масштабируя значения из `[0, 1]` в `[0, 255]` при необходимости.

        Args:
            mask: Исходная маска формы `(H, W)`, dtype=bool, float32 или uint8.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, значения {0, 255}.

        Note:
            - Используется во всех публичных методах сегментации для унификации вывода.
            - Безопасно обрабатывает маски с плавающей точкой, где `1.0` считается объектом.
            - Не изменяет маску, если она уже имеет корректный тип `uint8`.

        Example:
            ```python
            bool_mask = np.array([[True, False], [False, True]])
            mask = self._ensure_uint8_mask(bool_mask)  # [[255, 0], [0, 255]]
            ```
        """
        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                return (mask * 255).astype(np.uint8)
            return mask.astype(np.uint8)
        return mask

    # ──────────────────────────────────────────────────────────────────────
    def preprocess_image(
        self,
        image: ImageInput,
        as_gray: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
    ) -> NumpyImage:
        """Предобработка входного изображения с гарантией использования ITU-R BT.601.

        Переопределяет базовый метод для строгого контроля цветового пространства
        и форматирования данных перед сегментацией. Автоматически обрабатывает
        пути, PIL, Tensor и numpy массивы.

        Args:
            image: Входное изображение (путь, PIL, Tensor или np.ndarray).
            as_gray: Флаг конвертации в оттенки серого. По умолчанию False.
            target_size: Целевой размер `(width, height)`. По умолчанию None.
            normalize: Флаг нормализации к [0, 1]. По умолчанию False.

        Returns:
            NumpyImage: Обработанное изображение формы `(H, W)` или `(H, W, 3)`,
                        dtype=uint8 или float32.

        Note:
            - Конвертация в серый использует веса BT.601: `Y = 0.299·R + 0.587·G + 0.114·B`.
            - При `as_gray=True` и `normalize=True` возвращается `float32` в диапазоне `[0, 1]`.
            - Размер `target_size` применяется до конвертации цвета для сохранения производительности.

        Example:
            ```python
            img = self.preprocess_image("photo.jpg", as_gray=True, target_size=(512, 512))
            print(img.shape, img.dtype)  # (512, 512) uint8
            ```
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

    # ──────────────────────────────────────────────────────────────────────
    def segment(  # type: ignore[override]
        self, image: ImageInput, **kwargs: Any
    ) -> MaskArray:
        """Основной метод сегментации изображения.

        Выполняет предобработку, вызывает зарегистрированный алгоритм из `self.methods`,
        постобрабатывает результат и возвращает унифицированную бинарную маску.

        Args:
            image: Входное изображение (любой поддерживаемый формат).
            **kwargs: Дополнительные параметры для переопределения настроек метода.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                       где 255 = объект, 0 = фон.

        Note:
            - Автоматически применяет постобработку (удаление шума, заполнение дыр),
              если `self.params.get("postprocess", True)` и метод не является детектором границ.
            - При ошибке выполнения возвращает пустую маску и выводит `RuntimeWarning`.
            - Гарантирует стабильный интерфейс для всех 80+ методов сегментации.

        Example:
            ```python
            segmenter = SklearnSegmenter("otsu_thresholding")
            mask = segmenter.segment("document.png")
            # mask.shape == (H, W), mask.dtype == np.uint8, values ∈ {0, 255}
            ```
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

    # ──────────────────────────────────────────────────────────────────────
    def segment_and_evaluate(  # type: ignore[override]
        self,
        image: ImageInput,
        gt_mask: BinaryMask,  # имя как в базе
        threshold: float = 0.5,
        **segment_kwargs: Any,  # имя как в базе
    ) -> Tuple[MetricsDict, BinaryMask]:  # типы как в базе
        """Сегментация с немедленным вычислением метрик качества.

        Выполняет сегментацию и сравнивает результат с ground truth маской,
        возвращая словарь метрик и предсказанную маску для быстрой валидации.

        Args:
            image: Входное изображение.
            gt_mask: Ground truth маска формы `(H, W)`, dtype=uint8 или bool.
            threshold: Порог для бинаризации вероятностных масок [0.0, 1.0]. По умолчанию 0.5.
            **segment_kwargs: Дополнительные параметры для метода сегментации.

        Returns:
            Tuple[MetricsDict, BinaryMask]:
            - `metrics`: Словарь с метриками (IoU, Dice, Precision, Recall, Hausdorff и др.).
            - `mask`: Предсказанная бинарная маска.

        Note:
            - Требует установленного модуля `metrics.SegmentationMetrics`.
            - Метрики вычисляются после постобработки маски.
            - Hausdorff distance включён по умолчанию (`include_hausdorff=True`).

        Example:
            ```python
            metrics, mask = segmenter.segment_and_evaluate(
                image="sample.jpg",
                gt_mask=ground_truth,
                threshold=0.5
            )
            print(f"IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}")
            ```
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

    # ──────────────────────────────────────────────────────────────────────
    def segment_with_mask(  # type: ignore[override]
        self, image: ImageInput, alpha: float = 0.9, **kwargs: Any
    ) -> Tuple[ImageArray, MaskArray]:
        """Сегментация с возвратом визуализации и бинарной маски.

        Создаёт наложение маски на оригинальное изображение с настраиваемой прозрачностью
        для визуального контроля качества сегментации и презентации результатов.

        Args:
            image: Входное изображение.
            alpha: Коэффициент наложения маски [0.0, 1.0].
                   1.0 = только маска (красный), 0.0 = только оригинал. По умолчанию 0.9.
            **kwargs: Дополнительные параметры для метода сегментации.

        Returns:
            Tuple[ImageArray, MaskArray]:
            - `overlay`: Визуализация формы `(H, W, 3)`, dtype=uint8, RGB.
            - `mask`: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Маска накладывается красным цветом `[255, 0, 0]` для пикселей > 127.
            - Grayscale изображения автоматически конвертируются в 3-канальные для наложения.
            - Полезно для отладки, анализа ошибок и подготовки датасетов.

        Example:
            ```python
            segmenter = SklearnSegmenter("kmeans_segmentation", k=3)
            overlay, mask = segmenter.segment_with_mask(image, alpha=0.7)
            cv2.imwrite("result_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            ```
        """
        image = self.preprocess_image(image)
        # print(f"Image after sklearn preprocessing with mask: {image}")
        mask: MaskArray = self.segment(image, **kwargs)

        # Создаем визуализацию
        if len(image.shape) == 2:
            overlay: npt.NDArray = np.stack([image] * 3, axis=-1)
            original_rgb: npt.NDArray = np.stack([image] * 3, axis=-1)
        else:
            overlay = image.copy()
            original_rgb = image

        # Красный цвет для маски
        overlay[mask > 127] = [255, 0, 0]

        # Смешивание
        result = (alpha * overlay + (1 - alpha) * original_rgb).astype(np.uint8)
        # print(f"Mask after sklearn segment_with_mask: {mask}")
        # print(f"Result after sklearn segment_with_mask: {result}")
        return result, mask

    # ============ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _extract_features(self, image: ImageArray) -> FloatArray:
        """Извлечение признаков из изображения для scikit-learn методов.

        Извлекает три группы признаков:
        1. **Цветовые**: интенсивность (grayscale) или 3 канала (RGB/Lab).
        2. **Пространственные**: нормализованные координаты (x, y) и их произведение.
        3. **Текстурные**: градиенты по осям и магнитуда (через Sobel).

        Args:
            image: Входное изображение формы `(H, W)` или `(H, W, 3)`, dtype=uint8.

        Returns:
            FloatArray: Матрица признаков формы `(N, D)`, где:
                       - `N = H × W` — количество пикселей
                       - `D = 8` — количество признаков (3 цвета + 3 пространственных + 2 текстуры)

        Note:
            - Признаки автоматически масштабируются через `StandardScaler`.
            - Для больших изображений (>1000×1000) рассмотрите сэмплирование.
            - Текстура вычисляется через градиенты Собеля; для более сложных
              текстур можно добавить LBP, Gabor фильтры.
        """
        h: int = image.shape[0]
        w: int = image.shape[1]

        # Базовые цветовые признаки
        if len(image.shape) == 3:
            if SKIMAGE_AVAILABLE:
                gray: NormalizedArray = color.rgb2gray(image).astype(np.float32)
                color_features: FloatArray = image.reshape(-1, 3).astype(np.float32)
            else:
                gray = np.mean(image, axis=2).astype(np.float32)
                color_features = image.reshape(-1, 3).astype(np.float32)
        else:
            gray = image.astype(np.float32)
            color_features = image.reshape(-1, 1).astype(np.float32)

        # Пространственные признаки
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        spatial_features: FloatArray = np.stack(
            [
                x_coords.ravel() / w,
                y_coords.ravel() / h,
                (x_coords.ravel() * y_coords.ravel()) / (w * h),
            ],
            axis=1,
        ).astype(np.float32)

        # Текстура (упрощенная)
        if SKIMAGE_AVAILABLE:
            grad_x: FloatArray = filters.sobel_h(gray).ravel()
            grad_y: FloatArray = filters.sobel_v(gray).ravel()
        else:
            # Реализация Собеля на numpy
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            grad_x = signal.convolve2d(
                gray, kernel_x, mode="same", boundary="symm"
            ).ravel()
            grad_y = signal.convolve2d(
                gray, kernel_y, mode="same", boundary="symm"
            ).ravel()

        texture_features: FloatArray = np.stack(
            [grad_x, grad_y, np.sqrt(grad_x**2 + grad_y**2)],
            axis=1,
        ).astype(np.float32)

        # Комбинирование всех признаков
        features: FloatArray = np.hstack(
            [color_features, spatial_features, texture_features]
        )

        # Масштабирование
        if features.shape[0] > 0:
            features = self._scaler.fit_transform(features)

        return features

    # ──────────────────────────────────────────────────────────────────────
    def _to_gray_bt601(self, img: ImageArray) -> GrayImage:
        """Конвертация в серый по стандарту ITU-R BT.601 (как в OpenCV/Torch).

        Формула:
        ```
        Y = 0.299·R + 0.587·G + 0.114·B
        ```

        Args:
            img: Входное изображение формы `(H, W, 3)`, dtype=uint8.

        Returns:
            GrayImage: Grayscale изображение формы `(H, W)`, dtype=uint8.

        Note:
            - Веса соответствуют стандарту BT.601 для совместимости с OpenCV.
            - Для современных дисплеев можно использовать BT.709 (0.2126, 0.7152, 0.0722).
        """
        if len(img.shape) != 3 or img.shape[2] != 3:
            return img
        # Веса ITU-R BT.601
        gray: GrayImage = (
            0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        ).astype(np.uint8)

        return gray

    # ──────────────────────────────────────────────────────────────────────
    def _create_mask_from_labels(
        self,
        labels: ClusterLabels,
        shape: Tuple[int, int],
    ) -> MaskArray:
        """Создание бинарной маски из меток кластеризации.

        Алгоритм:
        1. Преобразование плоских меток в 2D форму.
        2. Исключение шумовых меток (-1) если есть.
        3. Определение фона как самого крупного кластера.
        4. Создание маски: всё кроме фона = объект.

        Args:
            labels: Метки кластеров формы `(N,)` или `(H, W)`, dtype=int32.
            shape: Целевая форма маски `(H, W)`.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255}.
        """
        labels_2d: ClusterLabels = labels.reshape(shape)

        # Исключение шума (-1) если есть
        unique: npt.NDArray[np.int32] = np.unique(labels_2d)
        valid_labels: npt.NDArray[np.int32] = unique[unique != -1]

        if len(valid_labels) == 0:
            return np.zeros(shape, dtype=np.uint8)

        # Нахождение самого крупного кластера как фона
        label_sizes: List[int] = [
            int(np.sum(labels_2d == label_val)) for label_val in valid_labels
        ]
        bg_label: int = int(valid_labels[np.argmax(label_sizes)])

        # Создание маски: всё кроме фона = объект
        mask: MaskArray = (labels_2d != bg_label).astype(np.uint8) * 255

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _postprocess_mask(self, mask: np.ndarray) -> MaskArray:
        """Постобработка маски для улучшения качества.

        Args:
            mask: Исходная маска

        Returns:
            np.ndarray: Улучшенная маска
        """
        binary: npt.NDArray[np.bool_] = mask > 127

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
            labeled: npt.NDArray[np.int32]
            num_features: int
            labeled, num_features = ndimage.label(binary)
            sizes: npt.NDArray[np.float64] = ndimage.sum(
                binary, labeled, range(1, num_features + 1)
            )
            for i, size in enumerate(sizes):
                if size < self.params.get("min_area", 100):
                    binary[labeled == i + 1] = False

        return binary.astype(np.uint8) * 255

    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    # ============ ПОРОГОВЫЕ МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_global_thresholding(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Глобальная пороговая сегментация.

        Применяет фиксированный порог яркости ко всему изображению:
        ```
        mask[x, y] = 255 if img[x, y] > threshold else 0
        ```

        Метод особенно эффективен для:
        - Изображений с равномерным освещением
        - Документов с чётким контрастом текста
        - Быстрой предварительной сегментации

        Алгоритм:
        1. Нормализация изображения к `[0, 1]` при необходимости.
        2. Побитовое сравнение каждого пикселя с порогом: `mask = img > threshold`.
        3. Преобразование булевой маски к `uint8 {0, 255}`.

        Args:
            img: Входное изображение. Поддерживаются форматы:
                 - Grayscale: `(H, W)`, dtype=uint8 или float32 [0,1]
                 - RGB: `(H, W, 3)`, dtype=uint8 (автоматически конвертируется)
            **kwargs: Дополнительные параметры:
                     - `threshold` (float): Порог [0.0, 1.0]. По умолчанию 0.5.

        Returns:
            Tuple[MaskArray, SegmentationInfo]:
                - `mask`: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255}.
                - `info`: Словарь с метаданными выполнения.

        Note:
            - Простой и быстрый метод, но чувствителен к неравномерному освещению.
            - Для адаптивного порога используйте `_sklearn_adaptive_thresholding`.
            - Порог 0.5 соответствует 128 для 8-битных изображений.

        Example:
            ```python
            # Базовое использование
            segmenter = SklearnSegmenter("global_thresholding", threshold=0.5)
            mask, info = segmenter._sklearn_global_thresholding(image)

            # Для тёмных объектов на светлом фоне
            segmenter = SklearnSegmenter("global_thresholding", threshold=0.3)
            mask, _ = segmenter.segment(image)
            ```
        """
        # Нормализация к [0, 1] для корректной работы порога
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)

        start_time: float = time.time()

        # Получение параметра с типизацией
        threshold: float = float(self.params.get("threshold", 0.5))

        # Применение порога
        mask_bool: npt.NDArray[np.bool_] = gray > threshold
        mask: MaskArray = (mask_bool * 255).astype(np.uint8)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "global_thresholding_sklearn",
            exec_time,
            {"threshold": threshold, **kwargs},
            threshold_applied=threshold,
        )

        # print(f"Mask after Sklearn_thresholding_global: {mask}")
        # print(f"Info after Sklearn_thresholding_global: {info}")

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_adaptive_thresholding(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Адаптивная пороговая сегментация (Gaussian).

        Вычисляет локальный порог для каждой области изображения на основе
        взвешенной суммы соседних пикселей (гауссово ядро).
        ```
        T(x, y) = mean(neighbors) - C
        ```
        Алгоритм:
        1. Нормализация к `[0, 1]`.
        2. Свёртка с гауссовым окном `block_size × block_size`.
        3. Вычитание константы `C` и сравнение с исходным изображением.
        4. Бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `block_size` (int, нечётный, ≥3), `C` (float, смещение).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Идеальна для документов с градиентом освещения.
            - `block_size` должен быть нечётным; чётные автоматически корректируются.
            - Отрицательный `C` делает порог строже.

        Example:
            ```python
            segmenter = SklearnSegmenter("adaptive_thresholding", block_size=15, C=2)
            mask, _ = segmenter.segment(document_image)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)

        start_time: float = time.time()
        block_size: int = int(self.params.get("block_size", 11))
        C: float = float(self.params.get("C", 2))

        adaptive_thresh: npt.NDArray[np.float64] = threshold_local(
            gray, block_size=block_size, offset=C / 255.0, method="gaussian"
        )
        mask_bool: npt.NDArray[np.bool_] = gray > adaptive_thresh
        mask: MaskArray = (mask_bool * 255).astype(np.uint8)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "adaptive_thresholding_sklearn",
            exec_time,
            {"block_size": block_size, "C": C, **kwargs},
            threshold_applied=(
                adaptive_thresh.tolist()
                if hasattr(adaptive_thresh, "tolist")
                else float(adaptive_thresh.mean())
            ),
        )

        # print(f"Mask after Sklearn_thresholding_adaptive: {mask}")
        # print(f"Info after Sklearn_thresholding_adaptive: {info}")

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_otsu_thresholding(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Автоматическая бинаризация по методу Оцу.

        Находит оптимальный порог, максимизирующий межклассовую дисперсию:
        ```
        σ_B² = w₀·w₁·(μ₀ - μ₁)²
        ```
        Алгоритм:
        1. Построение гистограммы интенсивностей.
        2. Перебор всех возможных порогов.
        3. Выбор порога с максимальной `σ_B²`.
        4. Бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: Не используются.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Эффективен для бимодальных гистограмм.
            - Требует предварительного преобразования в grayscale.
            - Может давать артефакты на унимодальных распределениях.

        Example:
            ```python
            segmenter = SklearnSegmenter("otsu_thresholding")
            mask, _ = segmenter.segment(gray_image)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)

        # print(f"Gray after Sklearn_thresholding_otsu: {gray}")

        start_time: float = time.time()
        thresh: float = float(threshold_otsu(gray))

        mask_bool: npt.NDArray[np.bool_] = gray > thresh
        mask: MaskArray = (mask_bool * 255).astype(np.uint8)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "otsu_thresholding_sklearn", exec_time, kwargs, threshold=thresh
        )

        # print(f"Mask after Sklearn_thresholding_otsu: {mask}")
        # print(f"Info after Sklearn_thresholding_otsu: {info}")

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_threshold_niblack(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Адаптивная пороговая обработка по Ниблаку.

        Порог вычисляется локально: `T = μ + k·σ`.
        Алгоритм:
        1. Вычисление локального среднего `μ` и СКО `σ` в окне.
        2. Применение формулы Ниблака.
        3. Сравнение пикселей с локальным порогом.

        Args:
            img: Входное изображение.
            **kwargs: `window_size` (int), `k` (float).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Чувствителен к шуму; при сильном шуме используйте Sauvola.
            - `k < 0` лучше для тёмных объектов на светлом фоне.

        Example:
            ```python
            segmenter = SklearnSegmenter("threshold_niblack", window_size=15, k=-0.2)
            mask, _ = segmenter.segment(noisy_doc)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)

        # print(f"Gray after Sklearn_thresholding_niblack: {gray}")

        start_time: float = time.time()
        window_size: int = int(self.params.get("window_size", 15))
        if window_size % 2 == 0:
            window_size += 1
        k: float = float(self.params.get("k", -0.2))
        # func_kwargs = {key: val for key, val in kwargs.items() if key not in ['window_size', 'k']}
        thresh: npt.NDArray[np.float64] = threshold_niblack(
            gray, window_size=window_size, k=k
        )
        mask: MaskArray = ((gray > thresh) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "niblack_thresholding_sklearn",
            exec_time,
            {"window_size": window_size, "k": k, **kwargs},
            threshold=thresh,
        )

        # print(f"Mask after Sklearn_thresholding_niblack: {mask}")
        # print(f"Info after Sklearn_thresholding_niblack: {info}")

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_threshold_sauvola(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Улучшенная адаптивная пороговая обработка по Сауволе.

        Формула: `T = μ·(1 + k·(σ/R - 1))`, где `R` — динамический диапазон.
        Алгоритм:
        1. Локальное вычисление `μ` и `σ`.
        2. Нормализация контраста через `R`.
        3. Адаптивная бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `window_size`, `k`, `r`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Устойчивее Ниблака при низком контрасте.
            - Для нормализованных изображений `[0,1]` используйте `r=0.5`.

        Example:
            ```python
            segmenter = SklearnSegmenter("threshold_sauvola", k=0.2, r=128.0)
            mask, _ = segmenter.segment(low_contrast_image)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)

        # print(f"Gray after Sklearn_thresholding_sauvola: {gray}")

        start_time: float = time.time()
        window_size: int = int(self.params.get("window_size", 15))
        if window_size % 2 == 0:
            window_size += 1
        k: float = float(self.params.get("k", 0.2))
        r: float = float(self.params.get("r", 128.0))

        # Порог Сауволы из scikit-image
        # func_kwargs = {key: val for key, val in kwargs.items() if key not in ['window_size', 'k', 'r']}
        thresh: npt.NDArray[np.float64] = threshold_sauvola(
            gray, window_size=window_size, k=k, r=r
        )
        mask: MaskArray = ((gray > thresh) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "sauvola_thresholding_sklearn",
            exec_time,
            {"window_size": window_size, "k": k, "r": r, **kwargs},
            threshold=thresh,
        )

        # print(f"Mask after Sklearn_thresholding_sauvola: {mask}")
        # print(f"Info after Sklearn_thresholding_sauvola: {info}")

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_threshold_bernsen(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Пороговая обработка по методу Бернсена.

        `T = (min + max)/2`, применяется только если `max - min > contrast_threshold`.
        Алгоритм:
        1. Локальные минимум и максимум в окне.
        2. Проверка контраста.
        3. Бинаризация по среднему при достаточном контрасте.

        Args:
            img: Входное изображение.
            **kwargs: `window_size`, `contrast_threshold`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Быстрее статистических методов (не требует вычисления СКО).
            - Чувствителен к выбросам в окне.

        Example:
            ```python
            segmenter = SklearnSegmenter("threshold_bernsen", contrast_threshold=25)
            mask, _ = segmenter.segment(high_contrast_image)
            ```
        """
        gray: npt.NDArray[np.float32] = img.astype(np.float32)
        start_time: float = time.time()
        window_size: int = int(self.params.get("window_size", 15))
        if window_size % 2 == 0:
            window_size += 1
        contrast_threshold: float = float(self.params.get("contrast_threshold", 0.1))

        img_range = img.max() - img.min()
        is_normalized = img_range <= 1.0

        if not is_normalized and contrast_threshold <= 1.0:
            # Если изображение [0-255], а порог [0-1], масштабируем
            contrast_threshold = contrast_threshold * 255

        from scipy.ndimage import minimum_filter, maximum_filter

        local_min: npt.NDArray[np.float32] = minimum_filter(
            gray, size=window_size, mode="reflect"
        )
        local_max: npt.NDArray[np.float32] = maximum_filter(
            gray, size=window_size, mode="reflect"
        )
        local_contrast: npt.NDArray[np.float32] = local_max - local_min
        threshold_map: npt.NDArray[np.float32] = (local_min + local_max) / 2.0
        mask_bool: npt.NDArray[np.bool_] = np.where(
            local_contrast > contrast_threshold, img > threshold_map, False
        )

        mask: MaskArray = (mask_bool * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "threshold_bernsen_sklearn",
            exec_time,
            {
                "window_size": window_size,
                "contrast_threshold": contrast_threshold,
                **kwargs,
            },
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_threshold_phansalkar(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Пороговая обработка по методу Фансалкара.

        Формула: `T = μ + k·σ·(σ/R) + m·(μ/R - 1)`.
        Алгоритм:
        1. Локальные статистики `μ`, `σ`.
        2. Коррекция порога через дисперсию и среднее.
        3. Бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `window_size`, `k`, `r`, `m`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Идеален для медицинских снимков с низким контрастом.
            - Параметр `m` корректирует смещение в зависимости от яркости.

        Example:
            ```python
            segmenter = SklearnSegmenter("threshold_phansalkar", k=0.25, m=0.5)
            mask, _ = segmenter.segment(medical_image)
            ```
        """
        gray: npt.NDArray[np.float32] = img.astype(np.float32)
        start_time: float = time.time()
        window_size: int = int(self.params.get("window_size", 15))
        if window_size % 2 == 0:
            window_size += 1
        k: float = float(self.params.get("k", 0.25))
        r: float = float(self.params.get("r", 128.0))
        m: float = float(self.params.get("m", 0.5))

        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(gray, size=window_size)
        local_sq_mean = uniform_filter(gray**2, size=window_size)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))

        # Адаптированная формула для диапазона [0, 1]
        # Порог Фансалкара
        img_range = img.max() - img.min()
        is_normalized = img_range <= 1.0

        # R = половина от максимального значения диапазона
        R = 0.5 if is_normalized else 128.0

        # ЕДИНАЯ ФОРМУЛА: T = μ + k·σ·(σ/R) + m·(μ/R - 1)
        threshold_map = (
            local_mean + k * local_std * (local_std / R) + m * (local_mean / R - 1)
        )
        mask: MaskArray = ((gray > threshold_map) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "threshold_phansalkar_sklearn",
            exec_time,
            {"window_size": window_size, "k": k, "r": r, "m": m, **kwargs},
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_threshold_kittler_illingworth(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Пороговая обработка по методу Киттлера-Иллингуорта.

        Минимизация ошибки классификации на основе гистограммы:
        ```
        J(t) = w₀·log(σ₀²) + w₁·log(σ₁²) - 2·[w₀·log(w₀) + w₁·log(w₁)]
        ```
        Алгоритм:
        1. Построение нормализованной гистограммы.
        2. Перебор порогов и вычисление `J(t)`.
        3. Выбор порога с минимальной ошибкой.

        Args:
            img: Входное изображение.
            **kwargs: `num_bins` (int).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Предполагает гауссовость классов.
            - Быстрее Оцу за счёт аналитического критерия.

        Example:
            ```python
            segmenter = SklearnSegmenter("threshold_kittler_illingworth", num_bins=256)
            mask, _ = segmenter.segment(document)
            ```
        """
        gray: npt.NDArray[np.float32] = img.astype(np.float32)
        start_time: float = time.time()
        num_bins: int = int(self.params.get("num_bins", 256))

        img_min = gray.min()
        img_max = gray.max()
        img_range = img_max - img_min
        is_normalized = img_range <= 1.0

        # Гистограмма изображения
        if is_normalized:
            hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 1.0))
            bin_edges = np.linspace(0, 1, num_bins + 1)
        else:
            hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))
            bin_edges = np.linspace(0, 256, num_bins + 1)

        hist = hist.astype(np.float64) + 1e-10

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
                best_bin = t

        best_threshold = bin_edges[best_bin]
        mask: MaskArray = ((gray > best_threshold) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "threshold_kittler_illingworth_sklearn",
            exec_time,
            {"num_bins": num_bins, **kwargs},
            threshold=best_threshold,
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_threshold_entropy_kapur(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Пороговая обработка на основе энтропии Капура.

        Максимизация суммы энтропий фона и объекта:
        ```
        H(t) = H₀(t) + H₁(t)
        ```
        Алгоритм:
        1. Нормализация гистограммы.
        2. Кумулятивное вычисление энтропии.
        3. Поиск максимума суммарной энтропии.

        Args:
            img: Входное изображение.
            **kwargs: `num_bins`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Эффективен для изображений с чётким информационным разделением.
            - Защита от `log(0)` добавлена автоматически.

        Example:
            ```python
            segmenter = SklearnSegmenter("threshold_entropy_kapur")
            mask, _ = segmenter.segment(information_rich_image)
            ```
        """
        gray: npt.NDArray[np.float32] = img.astype(np.float32)
        start_time: float = time.time()
        num_bins: int = int(self.params.get("num_bins", 256))

        # Гистограмма изображения
        img_min = img.min()
        img_max = img.max()
        img_range = img_max - img_min
        is_normalized = img_range <= 1.0

        # Гистограмма изображения
        if is_normalized:
            hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 1.0))
            bin_edges = np.linspace(0, 1, num_bins + 1)
        else:
            hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))
            bin_edges = np.linspace(0, 256, num_bins + 1)
        hist = (hist.astype(np.float64) + 1e-10) / (hist.sum() + num_bins * 1e-10)

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
                best_bin = t

        best_threshold = bin_edges[best_bin]
        mask: MaskArray = ((gray > best_threshold) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "threshold_entropy_kapur_sklearn",
            exec_time,
            {"num_bins": num_bins, **kwargs},
            threshold=best_threshold,
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_threshold_triangle(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Пороговая обработка треугольным методом.

        Геометрический поиск порога как точки максимального расстояния от линии пик-минимум.
        Алгоритм:
        1. Нахождение пика гистограммы.
        2. Построение линии до конца диапазона.
        3. Поиск максимального перпендикулярного расстояния.

        Args:
            img: Входное изображение.
            **kwargs: `num_bins`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Работает лучше для асимметричных гистограмм с длинным хвостом.
            - Очень быстрый, но не подходит для симметричных бимодальных распределений.

        Example:
            ```python
            segmenter = SklearnSegmenter("threshold_triangle")
            mask, _ = segmenter.segment(text_on_white_bg)
            ```
        """
        gray: npt.NDArray[np.float32] = img.astype(np.float32)
        start_time: float = time.time()
        num_bins: int = int(self.params.get("num_bins", 256))

        # Гистограмма изображения
        img_min = img.min()
        img_max = img.max()
        img_range = img_max - img_min
        is_normalized = img_range <= 1.0

        # Гистограмма изображения
        if is_normalized:
            hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 1.0))
            bin_edges = np.linspace(0, 1, num_bins + 1)
        else:
            hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))
            bin_edges = np.linspace(0, 256, num_bins + 1)

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
        max_dist = 0.0
        best_threshold = peak_idx / num_bins

        for t in range(peak_idx + 1, num_bins):
            # Расстояние от точки до линии
            y_line = y_peak + m * (t - peak_idx)
            dist = abs(hist[t] - y_line) / np.sqrt(1 + m**2)

            if dist > max_dist:
                max_dist = dist
                best_bin = t

        best_threshold = bin_edges[best_bin]
        mask: MaskArray = ((gray > best_threshold) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "threshold_triangle_sklearn",
            exec_time,
            {"num_bins": num_bins, **kwargs},
            threshold=best_threshold,
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_threshold_multi_otsu(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Многоуровневая пороговая обработка по методу Оцу.

        Расширение Оцу для `n_thresholds` порогов, разделяющих изображение на `n+1` классов.
        Алгоритм:
        1. Построение гистограммы.
        2. Полный перебор пар порогов (для 2 порогов) или рекурсия (>2).
        3. Бинаризация по самому яркому классу.

        Args:
            img: Входное изображение.
            **kwargs: `n_thresholds`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Для `n>2` используется упрощённый рекурсивный подход.
            - Возвращает бинарную маску, выделяя самый яркий класс.

        Example:
            ```python
            segmenter = SklearnSegmenter("threshold_multi_otsu", n_thresholds=2)
            mask, _ = segmenter.segment(multi_tissue_image)
            ```
        """
        start_time: float = time.time()
        n_classes: int = int(self.params.get("n_thresholds", 2)) + 1

        # Используем нативную реализацию skimage (быстрее и стабильнее кастомной рекурсии)
        from skimage.filters import threshold_multiotsu as threshold_multi_otsu

        thresholds: npt.NDArray[np.float64] = threshold_multi_otsu(
            img, classes=n_classes
        )

        # Для бинарной маски берем порог, отделяющий самый яркий класс
        best_threshold: float = (
            float(thresholds[-1]) if len(thresholds) > 0 else float(np.mean(img))
        )
        mask: MaskArray = ((img.astype(np.float32) > best_threshold) * 255).astype(
            np.uint8
        )
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "threshold_multi_otsu_sklearn",
            exec_time,
            {"n_thresholds": len(thresholds), **kwargs},
            thresholds=thresholds.tolist(),
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_threshold_percentile(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Процентильная пороговая обработка.

        Порог выбирается как заданный процентиль распределения интенсивностей.
        Алгоритм:
        1. Вычисление `np.percentile(img, percentile)`.
        2. Сравнение пикселей с порогом.
        3. Бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `percentile` (float ∈ [0,100]).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Автоматически адаптируется к контрасту сцены.
            - `percentile=90` отсекает 10% самых ярких пикселей.

        Example:
            ```python
            segmenter = SklearnSegmenter("threshold_percentile", percentile=90)
            mask, _ = segmenter.segment(bright_objects_image)
            ```
        """
        start_time: float = time.time()
        percentile: float = float(self.params.get("percentile", 90))
        threshold: float = float(np.percentile(img, percentile))
        mask: MaskArray = ((img.astype(np.float32) > threshold) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "threshold_percentile_sklearn",
            exec_time,
            {"percentile": percentile, **kwargs},
            threshold=threshold,
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_threshold_local_contrast(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Пороговая обработка на основе локального контраста.

        Пиксель считается объектом, если его интенсивность значительно отличается от локального среднего.
        Алгоритм:
        1. Вычисление локального среднего через `uniform_filter`.
        2. Расчёт абсолютной разницы `|img - local_mean|`.
        3. Глобальный порог контраста через процентиль.
        4. Бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `window_size`, `contrast_factor`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Эффективен для текстурных изображений.
            - Не зависит от абсолютной яркости, только от локальной изменчивости.

        Example:
            ```python
            segmenter = SklearnSegmenter("threshold_local_contrast", contrast_factor=0.1)
            mask, _ = segmenter.segment(texture_image)
            ```
        """
        start_time: float = time.time()
        window_size: int = int(self.params.get("window_size", 15))
        contrast_factor: float = float(self.params.get("contrast_factor", 0.1))

        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(img.astype(np.float32), size=window_size)
        local_contrast = np.abs(img.astype(np.float32) - local_mean)

        # Глобальный порог контраста
        global_contrast_threshold: float = float(
            np.percentile(local_contrast, 100 * (1 - contrast_factor))
        )

        mask: MaskArray = ((local_contrast > global_contrast_threshold) * 255).astype(
            np.uint8
        )
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "threshold_local_contrast_sklearn",
            exec_time,
            {"window_size": window_size, "contrast_factor": contrast_factor, **kwargs},
        )
        return mask, info

    # ============ МЕТОДЫ НА ОСНОВЕ КРАЕВ ============
    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_sobel_edge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Обнаружение границ оператором Собеля.

        Вычисляет аппроксимацию градиента через свёртку с ядрами:
        ```
        G_x = [[-1,0,1],[-2,0,2],[-1,0,1]], G_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
        |G| = sqrt(G_x² + G_y²)
        ```
        Алгоритм:
        1. Нормализация к `[0,1]`.
        2. Свёртка с ядрами Собеля.
        3. Нормализация магнитуды и пороговая бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `threshold` (float ∈ [0,1]).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска границ и метаданные.

        Note:
            - Чувствителен к шуму; предварительно применяйте размытие.
            - Возвращает только контуры.

        Example:
            ```python
            segmenter = SklearnSegmenter("sobel_edge", threshold=0.1)
            edges, _ = segmenter.segment(gray_image)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)

        # print(f"Gray after Sklearn_sobel_edge: {gray}")

        start_time: float = time.time()
        threshold: float = float(self.params.get("threshold", 0.1))
        magnitude: npt.NDArray[np.float64] = sobel(gray)

        # Нормализация и порог
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()

        mask: MaskArray = ((magnitude > threshold) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "sobel_edge_sklearn", exec_time, {"threshold": threshold, **kwargs}
        )

        # print(f"Mask after Sklearn_sobel_edge: {mask}")
        # print(f"Info after Sklearn_sobel_edge: {info}")

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_canny_edge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Обнаружение границ оператором Кэнни.

        Многоэтапный алгоритм: сглаживание → градиент → немаксимумы → гистерезис.
        Алгоритм:
        1. Гауссово размытие (`sigma`).
        2. Вычисление градиента.
        3. Подавление немаксимумов.
        4. Двойная пороговая фильтрация и отслеживание связности.

        Args:
            img: Входное изображение.
            **kwargs: `sigma`, `low`, `high`, `use_quantiles`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска границ и метаданные.

        Note:
            - Рекомендуется `high ≈ 2–3 × low`.
            - Лучший компромисс между точностью и шумом.

        Example:
            ```python
            segmenter = SklearnSegmenter("canny_edge", low=0.1, high=0.3)
            edges, _ = segmenter.segment(image)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)

        print(f"Gray after Sklearn_canny_edge: {gray}")

        start_time: float = time.time()
        sigma: float = float(self.params.get("sigma", 1.0))
        low_threshold: float = float(self.params.get("low", 0.1))
        high_threshold: float = float(self.params.get("high", 0.3))
        use_quantiles: bool = bool(self.params.get("use_quantiles", False))

        edges_bool: npt.NDArray[np.bool_] = feature.canny(
            gray,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            use_quantiles=use_quantiles,
        )
        mask: MaskArray = (edges_bool * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        print(
            f"DEBUG: sigma={sigma}, low={low_threshold}, high={high_threshold}, quantiles={use_quantiles}"
        )
        print(f"DEBUG: Image range: [{gray.min():.4f}, {gray.max():.4f}]")

        info: SegmentationInfo = self._log_info(
            "canny_edge_sklearn",
            exec_time,
            {
                "sigma": sigma,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "use_quantiles": use_quantiles,
                **kwargs,
            },
            edge_count=int(np.sum(edges_bool)),
        )
        print(f"Mask after Sklearn_canny_edge: {mask}")
        print(f"Info after Sklearn_canny_edge: {info}")
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_prewitt_edge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Обнаружение границ оператором Превитта.

        Использует равные веса `[1,1,1]` в ядрах, менее чувствителен к шуму, чем Собель.
        Алгоритм:
        1. Свёртка с ядрами Превитта.
        2. Расчёт магнитуды и нормализация.
        3. Пороговая бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `threshold`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска границ и метаданные.

        Note:
            - Быстрее Собеля за счёт простых весов.
            - Хорош для предварительной фильтрации шума.

        Example:
            ```python
            segmenter = SklearnSegmenter("prewitt_edge", threshold=0.1)
            edges, _ = segmenter.segment(noisy_image)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)
        start_time: float = time.time()
        threshold: float = float(self.params.get("threshold", 0.1))
        magnitude: npt.NDArray[np.float64] = prewitt(gray)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        mask: MaskArray = ((magnitude > threshold) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "prewitt_edge_sklearn", exec_time, {"threshold": threshold, **kwargs}
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_scharr_edge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Обнаружение границ оператором Шара.

        Улучшенная версия Собеля с оптимизированными весами для минимизации ошибки аппроксимации.
        Алгоритм:
        1. Применение `filters.scharr()`.
        2. Нормализация магнитуды.
        3. Бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `threshold`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска границ и метаданные.

        Note:
            - Более точен для тонких границ.
            - Вычислительно сопоставим с Собелем.

        Example:
            ```python
            segmenter = SklearnSegmenter("scharr_edge", threshold=0.1)
            edges, _ = segmenter.segment(fine_edges_image)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)
        start_time: float = time.time()
        threshold: float = float(self.params.get("threshold", 0.1))

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

        magnitude: npt.NDArray[np.float64] = filters.scharr(gray)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        mask: MaskArray = ((magnitude > threshold) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "scharr_edge_sklearn", exec_time, {"threshold": threshold, **kwargs}
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_roberts_cross_edge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Обнаружение границ оператором Робертса.

        Простой оператор 2×2 для диагональных границ:
        ```
        G = sqrt([I(x+1,y+1)-I(x,y)]² + [I(x+1,y)-I(x,y+1)]²)
        ```
        Алгоритм:
        1. Свёртка с ядрами Робертса.
        2. Расчёт магнитуды и бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `threshold`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска границ и метаданные.

        Note:
            - Очень быстрый, но крайне чувствителен к шуму.
            - Обнаруживает преимущественно диагонали.

        Example:
            ```python
            segmenter = SklearnSegmenter("roberts_cross_edge", threshold=0.1)
            edges, _ = segmenter.segment(diagonal_pattern)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)
        start_time: float = time.time()
        threshold: float = float(self.params.get("threshold", 0.1))

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

        magnitude: npt.NDArray[np.float64] = filters.roberts(gray)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        mask: MaskArray = ((magnitude > threshold) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "roberts_cross_edge_sklearn", exec_time, {"threshold": threshold, **kwargs}
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_log_edge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Обнаружение границ Лапласианом Гауссиана (LoG).

        `LoG = ∇²[G(σ)*I]`. Границы находятся по zero-crossing.
        Алгоритм:
        1. Гауссово размытие.
        2. Применение Лапласиана.
        3. Векторизованный поиск нулевых пересечений.
        4. Фильтрация по амплитуде.

        Args:
            img: Входное изображение.
            **kwargs: `sigma`, `threshold`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска границ и метаданные.

        Note:
            - `sigma` контролирует масштаб детектируемых границ.
            - Может давать разрывные контуры.

        Example:
            ```python
            segmenter = SklearnSegmenter("log_edge", sigma=1.0, threshold=0.01)
            edges, _ = segmenter.segment(smooth_transitions)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)
        start_time: float = time.time()
        sigma: float = float(self.params.get("sigma", 1.0))
        threshold: float = float(self.params.get("threshold", 0.01))
        # Гауссово размытие
        img_blurred = gaussian_filter(gray, sigma=sigma)

        # Лапласиан
        laplacian: npt.NDArray[np.float64] = laplace(img_blurred)

        # Векторизованный поиск zero-crossing
        zc = (
            (laplacian > 0) & (np.roll(laplacian, 1, axis=0) < 0)
            | (laplacian < 0) & (np.roll(laplacian, 1, axis=0) > 0)
            | (laplacian > 0) & (np.roll(laplacian, 1, axis=1) < 0)
            | (laplacian < 0) & (np.roll(laplacian, 1, axis=1) > 0)
        )
        mask: MaskArray = ((zc & (np.abs(laplacian) > threshold)) * 255).astype(
            np.uint8
        )
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "log_edge_sklearn",
            exec_time,
            {"sigma": sigma, "threshold": threshold, **kwargs},
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_dog_edge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Обнаружение границ разностью Гауссианов (DoG).

        Аппроксимация LoG: `DoG = G(σ₁)*I - G(σ₂)*I`.
        Алгоритм:
        1. Два Гауссовых фильтра с `σ₁` и `σ₂`.
        2. Вычисление разности.
        3. Zero-crossing detection и пороговая фильтрация.

        Args:
            img: Входное изображение.
            **kwargs: `sigma1`, `sigma2`, `threshold`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска границ и метаданные.

        Note:
            - Эффективна для multi-scale анализа.
            - `σ₂ > σ₁` обязательно.

        Example:
            ```python
            segmenter = SklearnSegmenter("dog_edge", sigma1=1.0, sigma2=2.0)
            edges, _ = segmenter.segment(multi_scale_image)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)
        start_time: float = time.time()
        sigma1: float = float(self.params.get("sigma1", 1.0))
        sigma2: float = float(self.params.get("sigma2", 2.0))
        threshold: float = float(self.params.get("threshold", 0.01))

        # Два Гауссовых фильтра
        g1 = gaussian_filter(gray, sigma=sigma1)
        g2 = gaussian_filter(gray, sigma=sigma2)

        # Разность Гауссианов
        dog = g1 - g2

        zc = (
            (dog > 0) & (np.roll(dog, 1, axis=0) < 0)
            | (dog < 0) & (np.roll(dog, 1, axis=0) > 0)
            | (dog > 0) & (np.roll(dog, 1, axis=1) < 0)
            | (dog < 0) & (np.roll(dog, 1, axis=1) > 0)
        )
        mask: MaskArray = ((zc & (np.abs(dog) > threshold)) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "dog_edge_sklearn",
            exec_time,
            {"sigma1": sigma1, "sigma2": sigma2, "threshold": threshold, **kwargs},
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_marr_hildreth_edge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Обнаружение границ методом Марра-Хилдрета.

        Классический метод: LoG + zero-crossing. Делегируется `_sklearn_log_edge`.

        Args/Returns/Note: Идентичны `log_edge`.

        Example:
            ```python
            segmenter = SklearnSegmenter("marr_hildreth_edge", sigma=1.5)
            edges, _ = segmenter.segment(image)
            ```
        """
        return self._sklearn_log_edge(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_gradient_magnitude_direction(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Границы через магнитуду и направление градиента.

        Позволяет фильтрацию по углу: `θ = arctan2(Gy, Gx)`.
        Алгоритм:
        1. Градиенты Собеля.
        2. Расчёт магнитуды и направления.
        3. Опциональная фильтрация по `angle_range`.
        4. Бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `threshold`, `angle_range` (Tuple[float, float]).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска границ и метаданные.

        Note:
            - Углы в градусах: 0°=вправо, 90°=вверх.
            - Учитывает симметрию `θ` и `θ+180°`.

        Example:
            ```python
            segmenter = SklearnSegmenter("gradient_magnitude_direction", angle_range=(80, 100))
            edges, _ = segmenter.segment(vertical_edges)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)
        start_time: float = time.time()
        threshold: float = float(self.params.get("threshold", 0.1))
        angle_range: Optional[Tuple[float, float]] = self.params.get(
            "angle_range", None
        )

        # Градиенты Собеля
        # gx = sobel(img, axis=1)  # Горизонтальный градиент
        # gy = sobel(img, axis=0)  # Вертикальный градиент

        gx = filters.sobel_h(gray)
        gy = filters.sobel_v(gray)

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

        mask: MaskArray = ((magnitude > threshold) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "gradient_magnitude_direction_sklearn",
            exec_time,
            {"threshold": threshold, "angle_range": angle_range, **kwargs},
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_phase_congruency_edge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Фазовая конгруэнтность (Kovesi's algorithm).

        Инвариантна к освещению и контрасту. Обнаруживает границы через выравнивание фаз Фурье-компонент.
        Алгоритм:
        1. FFT изображения.
        2. Построение Log-Gabor фильтров по масштабам и ориентациям.
        3. Свёртка в частотной области, накопление even/odd откликов.
        4. Компенсация шума через MAD.
        5. Нормализация и бинаризация.

        Args:
            img: Входное изображение.
            **kwargs: `nscale`, `norientations`, `min_wavelength`, `mult`, `sigma_onf`, `k_noise`, `cutoff_pc`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска границ и метаданные.

        Note:
            - Вычислительно интенсивный, но очень устойчив к шуму.
            - Идеален для медицинских/спутниковых снимков.

        Example:
            ```python
            segmenter = SklearnSegmenter("phase_congruency_edge", nscale=4, threshold=0.5)
            edges, _ = segmenter.segment(illumination_varying_image)
            ```
        """
        gray: NormalizedArray
        if img.max() > 1.0:
            gray = img.astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32)
        start_time: float = time.time()

        # ============ ПАРАМЕТРЫ ============
        nscale = int(self.params.get("nscale", 4))
        norientations = int(self.params.get("norientations", 4))
        min_wavelength = int(self.params.get("min_wavelength", 3))
        mult = float(self.params.get("mult", 2.0))
        sigma_onf = float(self.params.get("sigma_onf", 0.55))
        k_noise = float(self.params.get("k_noise", 2.0))
        cutoff = float(self.params.get("threshold", 0.5))
        epsilon = 1e-6

        rows, cols = gray.shape
        img_fft = np.fft.fft2(gray)

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

        for scale in range(nscale):
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
        mask: MaskArray = ((pc_map > cutoff) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "phase_congruency_edge_sklearn",
            exec_time,
            {
                "nscale": nscale,
                "norientations": norientations,
                "min_wavelength": min_wavelength,
                "mult": mult,
                "sigma_onf": sigma_onf,
                "k_noise": k_noise,
                "cutoff": cutoff,
                **kwargs,
            },
            mean_pc=float(np.mean(pc_map[mask > 0])) if mask.any() else 0.0,
        )
        return mask, info

    # ============ РЕГИОНАЛЬНЫЕ МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_region_growing(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация методом роста регионов (Region Growing).

        Алгоритм итеративно расширяет область от заданного семени, добавляя соседние пиксели,
        интенсивность которых отличается от начального значения не более чем на `tolerance`.
        Использует 8-связность для плавного захвата сложных форм.

        Алгоритм:
        1. Выбор начальной точки `seed` (по умолчанию — центр изображения).
        2. Инициализация очереди и маски посещённых пикселей.
        3. Пока очередь не пуста:
        - Извлечь пиксель `(x, y)`.
        - Если `|I(x,y) - I(seed)| <= tolerance`, добавить к региону.
        - Добавить 8-связных соседей в очередь.
        4. Возврат бинарной маски захваченной области.

        Метод особенно эффективен для:
        - Изображений с однородными областями и чёткими внутренними границами
        - Интерактивной сегментации с указанием точки внутри объекта
        - Медицинских снимков с выделением конкретных анатомических структур

        Args:
            img: Входное изображение `(H, W)` или `(H, W, 3)`, dtype=uint8.
            **kwargs: Дополнительные параметры:
                - `seed` (Tuple[int, int] | None): Координаты семени `(x, y)`. По умолчанию `(W//2, H//2)`.
                - `tolerance` (float): Максимальное отклонение интенсивности [0.0, 1.0]. По умолчанию 0.1.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Бинарная маска и метаданные.

        Note:
            - Чувствителен к выбору семени; если семя попадает на фон, будет захвачен фон.
            - Для нормализованных изображений `[0,1]` `tolerance` интерпретируется в абсолютных единицах.
            - Вычислительная сложность: O(N), где N — количество пикселей в регионе.
            - Не требует обучения, полностью детерминирован.

        Example:
            ```python
            # Сегментация объекта в центре
            segmenter = SklearnSegmenter("region_growing", tolerance=0.1)
            mask, _ = segmenter.segment(image)

            # С явным указанием семени
            segmenter = SklearnSegmenter("region_growing", seed=(200, 300), tolerance=0.05)
            mask, _ = segmenter.segment(image)
            ```
        """
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)

        start_time: float = time.time()
        h, w = gray.shape
        seed: Tuple[int, int] = self.params.get("seed", (w // 2, h // 2))
        tolerance: float = float(self.params.get("tolerance", 0.1))

        mask_bool = np.zeros((h, w), dtype=bool)
        visited = np.zeros((h, w), dtype=bool)

        queue = deque([seed])
        start_value = float(gray[seed[1], seed[0]])

        while queue:
            x, y = queue.popleft()

            if x < 0 or x >= w or y < 0 or y >= h or visited[y, x]:
                continue

            visited[y, x] = True

            if abs(float(gray[y, x]) - float(start_value)) <= tolerance:
                mask_bool[y, x] = True

                # 8-связность
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                            queue.append((nx, ny))

        mask: MaskArray = (mask_bool * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "region_growing_sklearn",
            exec_time,
            {"seed": seed, "tolerance": tolerance, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_split_and_merge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Рекурсивный алгоритм разделения и слияния регионов.

        Делит изображение на квадранты до тех пор, пока дисперсия интенсивности
        внутри региона не станет меньше `threshold`. Затем объединяет соседние
        однородные регионы и возвращает маску второго по величине кластера (объект).

        Алгоритм:
        1. Рекурсивное разделение (Split): если `std(region) > threshold`, делить на 4 квадранта.
        2. Слияние (Merge): объединять соседние регионы с близкими средними значениями.
        3. Сортировка регионов по площади, выбор 2-го по величине.
        4. Создание бинарной маски.

        Метод особенно эффективен для:
        - Спутниковых снимков с чёткими границами ландшафтов
        - Индустриальных изображений с однородными зонами
        - Задач, где объект занимает среднюю площадь относительно фона

        Args:
            img: Входное изображение.
            **kwargs: `threshold` (float), `min_size` (int).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Возвращается именно 2-й по величине регион, так как самый крупный обычно является фоном.
            - Экспоненциальная сложность в худшем случае, но на практике работает быстро для чётких границ.
            - `min_size` ограничивает глубину рекурсии, предотвращая переобучение на шуме.

        Example:
            ```python
            segmenter = SklearnSegmenter("split_and_merge", threshold=0.1, min_size=50)
            mask, _ = segmenter.segment(landscape_image)
            ```
        """
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img
        gray = self._normalize_image(gray)

        start_time: float = time.time()

        h, w = gray.shape
        threshold: float = float(self.params.get("threshold", 0.1))
        min_size: int = int(self.params.get("min_size", 50))

        # Начальный регион
        # regions = [gray.copy()]
        # region_masks = [np.ones((h, w), dtype=bool)]

        # Простая реализация Split
        def split_region(region, mask):
            h_reg, w_reg = region.shape
            if h_reg * w_reg <= min_size:
                return [(region, mask)]

            # mean = np.mean(region[mask])
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

        exec_time: float = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info: SegmentationInfo = self._log_info(
            "split_and_merge_sklearn",
            exec_time,
            {"threshold": threshold, "min_size": min_size, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _split_and_merge(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Рекурсивный алгоритм разделения и слияния регионов.

        Делит изображение на квадранты до тех пор, пока дисперсия интенсивности
        внутри региона не станет меньше `threshold`. Затем объединяет соседние
        однородные регионы и возвращает маску второго по величине кластера (объект).

        Алгоритм:
        1. Рекурсивное разделение (Split): если `std(region) > threshold`, делить на 4 квадранта.
        2. Слияние (Merge): объединять соседние регионы с близкими средними значениями.
        3. Сортировка регионов по площади, выбор 2-го по величине.
        4. Создание бинарной маски.

        Метод особенно эффективен для:
        - Спутниковых снимков с чёткими границами ландшафтов
        - Индустриальных изображений с однородными зонами
        - Задач, где объект занимает среднюю площадь относительно фона

        Args:
            img: Входное изображение.
            **kwargs: `threshold` (float), `min_size` (int).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Возвращается именно 2-й по величине регион, так как самый крупный обычно является фоном.
            - Экспоненциальная сложность в худшем случае, но на практике работает быстро для чётких границ.
            - `min_size` ограничивает глубину рекурсии, предотвращая переобучение на шуме.

        Example:
            ```python
            segmenter = SklearnSegmenter("split_and_merge", threshold=0.1, min_size=50)
            mask, _ = segmenter.segment(landscape_image)
            ```
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

        exec_time: float = time.time() - start_time
        mask = (mask * 255).astype(np.uint8)

        info: SegmentationInfo = self._log_info(
            "split_and_merge_sklearn",
            exec_time,
            {"threshold": threshold, "min_size": min_size, **kwargs},
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _floodfill(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация методом заливки (Flood Fill).

        Заполняет связную область, начиная с `seed`, пока разница интенсивности
        между текущим пикселем и семенем не превысит `tolerance`.
        Использует оптимизированную реализацию из `skimage.segmentation.flood`.

        Алгоритм:
        1. Преобразование в grayscale.
        2. Вызов `flood()` с указанным допуском.
        3. Возврат булевой маски, конвертированной в `{0, 255}`.

        Метод особенно эффективен для:
        - Быстрого выделения однородных объектов по точке внутри
        - Предобработки для watershed или активных контуров
        - Интерактивных редакторов и инструментов разметки

        Args:
            img: Входное изображение.
            **kwargs: `seed` (Tuple[int, int]), `tolerance` (float ∈ [0, 1]).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `tolerance` интерпретируется относительно диапазона `[0, 1]`.
            - Автоматически обрабатывает 4- и 8-связность.
            - Быстрее ручного queue-based region growing за счёт C-бэкенда skimage.

        Example:
            ```python
            segmenter = SklearnSegmenter("floodfill", seed=(150, 200), tolerance=0.15)
            mask, _ = segmenter.segment(image)
            ```
        """
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        if gray.max() > 1.0:
            gray /= 255.0

        start_time: float = time.time()
        h, w = gray.shape
        seed: Tuple[int, int] = self.params.get("seed", (w // 2, h // 2))
        tolerance: float = float(self.params.get("tolerance", 0.15))
        mask_bool = segmentation.flood(gray, seed, tolerance=tolerance)

        exec_time: float = time.time() - start_time
        mask: MaskArray = (mask_bool * 255).astype(np.uint8)

        info: SegmentationInfo = self._log_info(
            "floodfill_sklearn",
            exec_time,
            {"seed": seed, "tolerance": tolerance, **kwargs},
        )
        return mask, info

    # ============ КЛАСТЕРИЗАЦИЯ ============
    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_kmeans_segmentation(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация методом K-Means кластеризации.

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

        start_time: float = time.time()

        # Преобразуем изображение в массив пикселей
        pixels = img.reshape(-1, 3)
        # features = self._extract_features(img)

        # Применяем K-Means
        k = int(self.params.get("k", 3))
        random_state = 42
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(pixels)

        # Находим самый большой кластер (фон)
        unique_labels, counts = np.unique(labels, return_counts=True)
        bg_label = unique_labels[np.argmax(counts)]

        # Создаем маску (все кроме фона)
        mask_raw = (labels != bg_label).reshape(h, w)
        exec_time: float = time.time() - start_time
        mask: MaskArray = (mask_raw * 255).astype(np.uint8)

        info: SegmentationInfo = self._log_info(
            "kmeans_segmentation_sklearn",
            exec_time,
            {
                "k": k,
                "random_state": random_state,
                #  "n_init": n_init,
                **kwargs,
            },
            inertia=float(kmeans.inertia_),
            n_iter=int(kmeans.n_iter_),
            cluster_sizes=counts.tolist(),
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_kmeans(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """K-Means из sklearn с извлечением признаков."""
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        k = int(self.params.get("k", 3))
        random_state = 42
        n_init = 10
        kmeans = KMeans(
            n_clusters=k, random_state=random_state, n_init=n_init, **kwargs
        )
        labels = kmeans.fit_predict(features)

        # Создаем маску
        mask_raw = self._create_mask_from_labels(labels, (h, w))
        exec_time: float = time.time() - start_time
        mask: MaskArray = (mask_raw * 255).astype(np.uint8)

        info: SegmentationInfo = self._log_info(
            "kmeans_segmentation_sklearn",
            exec_time,
            {"k": k, "random_state": random_state, "n_init": n_init, **kwargs},
            inertia=float(kmeans.inertia_),
            n_iter=int(kmeans.n_iter_),
            cluster_sizes=kmeans.tolist(),
            cluster_centers=kmeans.cluster_centers_,
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_dbscan_segmentation(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация методом DBSCAN кластеризации.

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
        start_time: float = time.time()

        h, w = gray.shape

        # Извлекаем признаки (пиксель + координаты)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.column_stack(
            [gray.ravel(), x_coords.ravel() / w, y_coords.ravel() / h]
        )

        # Применяем DBSCAN
        eps = float(self.params.get("eps", 0.1))
        min_samples = int(self.params.get("min_samples", 10))

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = dbscan.fit_predict(features)

        # Создаем маску (исключаем шум -1)
        mask: MaskArray = ((labels != -1).reshape(h, w) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "dbscan_sklearn",
            exec_time,
            {"eps": eps, "min_samples": min_samples, **kwargs},
            n_clusters=int(len(np.unique(labels[labels != -1]))),
            n_noise=int(np.sum(labels == -1)),
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_dbscan(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """DBSCAN из sklearn."""
        features = self._extract_features(img)
        h, w = img.shape[:2]
        start_time: float = time.time()

        eps = float(self.params.get("eps", 0.5))
        min_samples = int(self.params.get("min_samples", 5))

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = dbscan.fit_predict(features)

        # Создаем маску (исключаем шум)
        mask: MaskArray = ((labels != -1).reshape(h, w) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "dbscan_sklearn",
            exec_time,
            {"eps": eps, "min_samples": min_samples, **kwargs},
            n_clusters=int(len(np.unique(labels[labels != -1]))),
            n_noise=int(np.sum(labels == -1)),
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _meanshift(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация методом MeanShift.

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
        start_time: float = time.time()

        # Для производительности сэмплируем пиксели
        sample_size = min(1000, h * w)
        indices = np.random.choice(h * w, sample_size, replace=False)

        pixels = img_rgb.reshape(-1, 3)[indices]
        coords = np.column_stack([indices // w, indices % w]) / [h, w]

        features = np.hstack([pixels / 255.0, coords])

        # Применяем MeanShift
        bandwidth = float(self.params.get("bandwidth", 0.5))
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
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

        mask_raw = (all_labels != bg_label).reshape(h, w)
        exec_time: float = time.time() - start_time
        mask: MaskArray = (mask_raw * 255).astype(np.uint8)

        info: SegmentationInfo = self._log_info(
            "meanshift_sklearn",
            exec_time,
            {"bandwidth": bandwidth, **kwargs},
            n_clusters=len(unique_labels),
            cluster_centers=ms.cluster_centers_,
        )
        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_meanshift(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Meanshift кластеризация для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

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
        bandwidth = float(self.params.get("bandwidth", 0.5))
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
        maskd = self._create_mask_from_labels(labels, (h, w))
        mask_raw = self._postprocess_mask(maskd)
        exec_time: float = time.time() - start_time
        mask: MaskArray = (mask_raw * 255).astype(np.uint8)

        info: SegmentationInfo = self._log_info(
            "meanshift_sklearn",
            exec_time,
            {"bandwidth": bandwidth, **kwargs},
            n_clusters=len(unique),
            cluster_centers=meanshift.cluster_centers_,
        )

        return mask, info

    # ============ АКТИВНЫЕ КОНТУРЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _active_contour(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация активными контурами (Snakes).

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

        start_time: float = time.time()

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
        alpha = float(self.params.get("alpha", 0.015))  # elasticity (0.01)
        beta = float(self.params.get("beta", 10))  # rigidity (0.1)
        gamma = float(self.params.get("gamma", 0.001))  # time step (0.001)
        max_iterations = int(self.params.get("max_iterations", 2000))
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
        mask_bool = np.zeros((h, w), dtype=bool)
        rr, cc = polygon(snake[:, 1], snake[:, 0], mask_bool.shape)
        mask_bool[rr, cc] = True
        mask: MaskArray = (mask_bool * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "active_contour_sklearn",
            exec_time,
            {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "w_edge": w_edge,
                "w_line": w_line,
                "max_iterations": max_iterations,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _gvf_contour(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация на основе Gradient Vector Flow (GVF).

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

        start_time: float = time.time()

        # Вычисляем градиенты
        # grad_y, grad_x = np.gradient(gray_norm)
        grad_x = filters.sobel_h(gray)
        grad_y = filters.sobel_v(gray)

        # Вычисляем внешние силы (edge map)
        edge_map = grad_x**2 + grad_y**2
        edge_map = edge_map / (edge_map.max() + 1e-8)

        # Применяем GVF
        mu = float(self.params.get("mu", 0.1))  # 0.2
        iterations = int(self.params.get("iterations", 50))  # 100

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
        mask_raw = gvf_mag > np.percentile(gvf_mag, 70)
        exec_time: float = time.time() - start_time
        mask: MaskArray = (mask_raw * 255).astype(np.uint8)
        info: SegmentationInfo = self._log_info(
            "gvf_contour_sklearn",
            exec_time,
            {"mu": mu, "iterations": iterations, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _morphological_snakes(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация морфологическими змеями.

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

        start_time: float = time.time()

        # Применяем морфологические активные контуры
        init_level_set = np.zeros(gray.shape, dtype=np.int8)
        h, w = gray.shape
        init_level_set[(h // 4) : (3 * h // 4), (w // 4) : (3 * w // 4)] = 1

        smoothing = self.params.get("smoothing", 1)
        threshold = self.params.get("threshold", 0.5)
        iterations = self.params.get("iterations", 50)
        balloon = 1
        mask_raw = morphological_geodesic_active_contour(
            gray,
            iterations,
            init_level_set,
            smoothing=smoothing,
            threshold=threshold,
            balloon=balloon,
            **kwargs,
        )

        exec_time: float = time.time() - start_time
        mask: MaskArray = (mask_raw * 255).astype(np.uint8)

        info: SegmentationInfo = self._log_info(
            "morphological_snakes_sklearn",
            exec_time,
            {
                "smoothing": smoothing,
                "threshold": threshold,
                "iterations": iterations,
                "balloon": balloon,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _chan_vese(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Модель Chan-Vese — активные контуры без градиентов.

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

        start_time: float = time.time()

        # Начальная маска
        init_level_set = np.zeros(gray.shape, dtype=np.int8)
        h, w = gray.shape
        init_level_set[(h // 4) : (3 * h // 4), (w // 4) : (3 * w // 4)] = 1

        # Параметры Chan-Vese
        mu = self.params.get("mu", 0.25)
        lambda1 = self.params.get("lambda1", 1.0)
        lambda2 = self.params.get("lambda2", 1.0)
        tol = self.params.get("tol", 1e-3)
        max_iter = self.params.get("max_iter", 100)
        iterations = self.params.get("iterations", 100)

        # Применяем метод Chan-Vese
        mask_raw = morphological_chan_vese(
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

        exec_time: float = time.time() - start_time
        mask: MaskArray = (mask_raw * 255).astype(np.uint8)

        info: SegmentationInfo = self._log_info(
            "chan_vese_sklearn",
            exec_time,
            {
                "mu": mu,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "tol": tol,
                "max_iter": max_iter,
                "iterations": iterations,
                **kwargs,
            },
            converged=exec_time < max_iter * 0.1,
        )

        return mask, info

    # ============ WATERSHED И ГРАФОВЫЕ МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _watershed(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация методом водораздела (Watershed).

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

        start_time: float = time.time()

        # Маркеры для watershed
        markers = np.zeros_like(gray, dtype=np.uint8)
        h, w = gray.shape

        # Создаем маркеры
        markers[gray < np.percentile(gray, 25)] = 1
        markers[gray > np.percentile(gray, 75)] = 2

        # Применяем watershed
        segmentation = watershed(filters.sobel(gray), markers=markers)
        mask: MaskArray = ((segmentation == 2) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info("watershed_sklearn", exec_time, kwargs)

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _random_walker(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация методом Random Walker.

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

        start_time: float = time.time()

        # Создаем маркеры
        markers = np.zeros(gray.shape, dtype=np.uint8)
        h, w = gray.shape

        # Центральная область - объект
        markers[(h // 4) : (3 * h // 4), (w // 4) : (3 * w // 4)] = 2

        # Углы - фон
        corner_size = min(h, w) // 8
        markers[:corner_size, :corner_size] = 1
        markers[:corner_size, -corner_size:] = 1
        markers[-corner_size:, :corner_size] = 1
        markers[-corner_size:, -corner_size:] = 1

        # Применяем Random Walker
        beta = float(self.params.get("beta", 10))
        mode = str(self.params.get("mode", "cg_mg"))
        labels = random_walker(gray, markers, beta=beta, mode=mode)

        # Бинаризуем
        mask: MaskArray = ((labels == 2) * 255).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "random_walker_sklearn", exec_time, {"beta": beta, "mode": mode, **kwargs}
        )

        return mask, info

    # ============ SUPER-PIXEL МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _quickshift(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация методом Quickshift (реализована через MeanShift как аналог).

        Находит моды в плотности распределения пикселей в пространстве признаков.
        Группирует пиксели, принадлежащие одной моде.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8). Самый крупный кластер — фон.
        """
        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img
        start_time: float = time.time()

        kernel_size = int(self.params.get("kernel_size", 3))
        max_dist = float(self.params.get("max_dist", 6))
        ratio = float(self.params.get("ratio", 0.5))

        segments = quickshift(
            img_rgb, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio, **kwargs
        )

        # Находим самый большой суперпиксель
        unique_labels, counts = np.unique(segments, return_counts=True)
        if len(unique_labels) > 0:
            bg_label = unique_labels[np.argmax(counts)]
            mask: MaskArray = (segments != bg_label).astype(np.uint8) * 255
        else:
            mask = np.zeros_like(segments, dtype=np.uint8)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "quickshift_sklearn",
            exec_time,
            {
                "kernel_size": kernel_size,
                "max_dist": max_dist,
                "ratio": ratio,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _slic(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """SLIC (Simple Linear Iterative Clustering) — суперпиксельная сегментация.

        Группирует пиксели в компактные, однородные регионы (суперпиксели) на основе пространственной
        и цветовой близости. Самый крупный суперпиксель считается фоном.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8): 255 — все суперпиксели, кроме фона.
        """
        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img
        start_time: float = time.time()

        n_segments = int(self.params.get("n_segments", 100))
        compactness = float(self.params.get("compactness", 10.0))

        # Применяем SLIC
        segments = slic(
            img_rgb, n_segments=n_segments, compactness=compactness, **kwargs
        )

        # Находим самый большой суперпиксель
        unique_labels, counts = np.unique(segments, return_counts=True)
        if len(unique_labels) > 0:
            bg_label = unique_labels[np.argmax(counts)]
            mask: MaskArray = (segments != bg_label).astype(np.uint8) * 255
        else:
            mask = np.zeros_like(segments, dtype=np.uint8)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "slic_sklearn",
            exec_time,
            {
                "n_segments": n_segments,
                "compactness": compactness,
                # "max_iter": max_iter,
                # "enforce_connectivity": enforce_connectivity,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _felzenszwalb(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Алгоритм Felzenszwalb — иерархическая сегментация на основе графов.

        Строит сегментацию, начиная с мелких регионов и объединяя их, если внутреннее различие
        меньше межрегионального. Очень эффективен для выделения объектов разного масштаба.

        Args:
            img: Входное изображение (RGB).

        Returns:
            Бинарная маска: 255 — все регионы, кроме самого крупного (фона).
        """
        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img
        start_time: float = time.time()

        scale = float(self.params.get("scale", 100))
        sigma = float(self.params.get("sigma", 0.5))
        min_size = int(self.params.get("min_size", 50))

        # Применяем Felzenszwalb
        segments = felzenszwalb(
            img_rgb, scale=scale, sigma=sigma, min_size=min_size, **kwargs
        )

        # Находим самый большой регион
        unique_labels, counts = np.unique(segments, return_counts=True)
        if len(unique_labels) > 0:
            bg_label = unique_labels[np.argmax(counts)]
            mask_np: MaskArray = (segments != bg_label).astype(np.uint8) * 255
        else:
            mask_np = np.zeros_like(segments, dtype=np.uint8)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "felzenszwalb_sklearn",
            exec_time,
            {"scale": scale, "sigma": sigma, "min_size": min_size, **kwargs},
            n_segments=len(unique_labels),
        )

        return mask_np, info

    # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────
    def _grabcut(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Интерактивная сегментация GrabCut.

        Алгоритм на основе графов и гауссовых смесей (GMM):
        1. Инициализация прямоугольником (фон/объект).
        2. Построение моделей цвета для фона и объекта.
        3. Итеративная оптимизация через минимизацию энергии.
        4. Финальная бинаризация.

        Требует указания прямоугольника, содержащего объект.

        Args:
            img: Входное изображение (RGB/BGR/GRAY).
            **kwargs: Дополнительные параметры:
                - `rect` (Tuple[int, int, int, int]): Прямоугольник (x, y, w, h).
                По умолчанию: центральная область 50%×50%.
                - `num_iterations` (int): Количество итераций оптимизации.
                По умолчанию 5.

        Returns:
            Tuple[MaskArray, SegmentationInfo]:
                - `mask`: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255}.
                - `info`: Словарь с метаданными выполнения.

        Note:
            - Метод чувствителен к качеству инициализации (прямоугольника).
            - Больше итераций → точнее результат, но медленнее.
            - После выполнения рекомендуется морфологическая пост-обработка.

        Example:
            ```python
            # Прямоугольник: (x=100, y=100, w=200, h=200)
            segmenter = SklearnSegmenter("grabcut", rect=(100, 100, 200, 200), num_iterations=10)
            mask, _ = segmenter.segment(image)
            ```
        """
        # Конвертация в BGR для OpenCV
        if len(img.shape) == 2:
            img_bgr: ImageArray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        elif img.shape[2] == 4:  # RGBA → BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR).astype(np.uint8)
        elif img.shape[2] == 3 and img.max() <= 1.0:  # float RGB → uint8 BGR
            img_bgr = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR).astype(np.uint8)
        elif img.shape[2] == 3:  # RGB → BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
        else:
            img_bgr = img

        h, w = img_bgr.shape[:2]
        start_time: float = time.time()

        # Получение параметров
        rect: Tuple[int, int, int, int] = self.params.get(
            "rect", (w // 4, h // 4, w // 2, h // 2)
        )
        num_iterations: int = int(self.params.get("num_iterations", 5))

        # Инициализация маски: 0=определённый фон, 2=вероятный объект
        mask_init: npt.NDArray[np.uint8] = np.zeros((h, w), dtype=np.uint8)
        x, y, rw, rh = rect
        # Ограничиваем прямоугольник границами изображения
        x, y = max(0, x), max(0, y)
        rw, rh = min(rw, w - x), min(rh, h - y)
        mask_init[y : (y + rh), x : (x + rw)] = cv2.GC_PR_FGD  # Вероятный передний план

        # Временные массивы для GMM моделей (обязательные аргументы cv2.grabCut)
        bgd_model: npt.NDArray[np.float64] = np.zeros((1, 65), dtype=np.float64)
        fgd_model: npt.NDArray[np.float64] = np.zeros((1, 65), dtype=np.float64)

        # Выполнение GrabCut
        mask_grabcut: npt.NDArray[np.int32]
        mask_grabcut, bgd_model, fgd_model = cv2.grabCut(  # type: ignore[assignment]
            img_bgr,
            mask_init,
            rect,
            bgd_model,
            fgd_model,
            num_iterations,
            cv2.GC_INIT_WITH_RECT,
        )

        # Финальная маска: GC_FGD и GC_PR_FGD = объект (255), остальное = фон (0)
        mask_bool: npt.NDArray[np.bool_] = (mask_grabcut == cv2.GC_FGD) | (
            mask_grabcut == cv2.GC_PR_FGD
        )
        mask: MaskArray = (mask_bool * 255).astype(np.uint8)

        # Пост-обработка: морфологические операции для сглаживания границ
        kernel: npt.NDArray[np.uint8] = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # type: ignore[assignment]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # type: ignore[assignment]

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = {
            "method": "grabcut_sklearn",
            "parameters": {
                "rect": rect,
                "num_iterations": num_iterations,
                **kwargs,
            },
            "execution_time": exec_time,
        }

        return mask, info

    # ============ SKLEARN МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_gmm(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Gaussian Mixture Models для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(img)
        h, w = img.shape[:2]

        start_time: float = time.time()

        n_components = int(self.params.get("n_components", 3))
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
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "gmm_sklearn",
            exec_time,
            {
                "n_components": n_components,
                "covariance_type": covariance_type,
                **kwargs,
            },
            converged=gmm.converged_,
            lower_bound=gmm.lower_bound_,
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_optics(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """OPTICS кластеризация для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        features = self._extract_features(img)
        h, w = img.shape[:2]

        start_time: float = time.time()

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
        min_samples = int(self.params.get("min_samples", 5))
        xi = float(self.params.get("xi", 0.05))
        min_cluster_size = float(self.params.get("min_cluster_size", 0.1))

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
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)

        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "optics_sklearn",
            exec_time,
            {
                "min_samples": min_samples,
                "xi": xi,
                "min_cluster_size": min_cluster_size,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_agglomerative(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Agglomerative (иерархическая) кластеризация.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

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
        n_clusters = int(self.params.get("n_clusters", 3))
        linkage = str(self.params.get("linkage", "ward"))
        affinity = str(self.params.get("affinity", "euclidean"))

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
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "agglomerative_sklearn",
            exec_time,
            {
                "n_clusters": n_clusters,
                "linkage": linkage,
                "affinity": affinity,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_spectral(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Spectral Clustering для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()

        features = self._extract_features(img)

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
        n_clusters = int(self.params.get("n_clusters", 3))
        affinity = str(self.params.get("affinity", "nearest_neighbors"))
        n_neighbors = int(self.params.get("n_neighbors", 10))

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
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "spectral_sklearn",
            exec_time,
            {
                "n_clusters": n_clusters,
                "n_neighbors": n_neighbors,
                "affinity": affinity,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_birch(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        # Параметры BIRCH
        n_clusters = int(self.params.get("n_clusters", 3))
        threshold = float(self.params.get("threshold", 0.5))
        branching_factor = int(self.params.get("branching_factor", 50))

        # Применяем BIRCH
        birch = Birch(
            n_clusters=n_clusters,
            threshold=threshold,
            branching_factor=branching_factor,
        )

        labels = birch.fit_predict(features)

        # Создаем маску
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "birch_sklearn",
            exec_time,
            {
                "n_clusters": n_clusters,
                "threshold": threshold,
                "branching_factor": branching_factor,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_mini_batch_kmeans(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Mini-Batch K-Means для больших изображений.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        # Параметры
        n_clusters = int(self.params.get("n_clusters", 3))
        batch_size = int(self.params.get("batch_size", 100))

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
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "mini_batch_kmeans_sklearn",
            exec_time,
            {"n_clusters": n_clusters, "batch_size": batch_size, **kwargs},
        )

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

    # ──────────────────────────────────────────────────────────────────────
    # ============ КЛАССИФИКАТОРЫ ДЛЯ СЕГМЕНТАЦИИ ============
    def _sklearn_random_forest(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Random Forest для сегментации (полу-автоматический).

        Обучает ансамбль деревьев на автоматически сгенерированных метках:
        центр = объект, углы = фон. Использует признаки цвета, координат и текстуры.

        Алгоритм:
        1. Генерация обучающих меток `y ∈ {0, 1}`.
        2. Извлечение признаков `X` (цвет + пространство + градиенты).
        3. Обучение `RandomForestClassifier(n_estimators=50)`.
        4. Предсказание и конвертация в маску.

        Метод особенно эффективен для:
        - Задач с нечёткими границами, но различимыми текстурными паттернами
        - Предобработки для глубокого обучения (pseudo-labeling)
        - Устойчивой сегментации при умеренном шуме

        Args:
            img: Входное изображение.
            **kwargs: (параметры RF настраиваются через self.params)

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Автоматические метки работают хорошо, если объект в центре.
            - `n_estimators` по умолчанию 50; увеличение повышает точность, но замедляет.
            - Не требует ручной разметки; полностью автономен.

        Example:
            ```python
            segmenter = SklearnSegmenter("random_forest")
            mask, _ = segmenter.segment(texture_varying_image)
            ```
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

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
        mask: MaskArray = (labels.reshape(h, w) > 0).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "random_forest_sklearn", exec_time, kwargs
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_svm(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Support Vector Machine для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()

        features = self._extract_features(img)

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
        mask_raw = labels.reshape(h, w).astype(np.uint8) * 255
        mask: MaskArray = self._postprocess_mask(mask_raw)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "svm_sklearn",
            exec_time,
            {"C": C, "kernel": kernel, "gamma": gamma, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_logistic_regression(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Logistic Regression для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

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
        mask_raw = labels.reshape(h, w).astype(np.uint8) * 255
        mask: MaskArray = self._postprocess_mask(mask_raw)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "logistic_regression_sklearn", exec_time, {"C": C, **kwargs}
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_knn(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """K-Nearest Neighbors для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

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
        mask_raw = labels.reshape(h, w).astype(np.uint8) * 255
        mask: MaskArray = self._postprocess_mask(mask_raw)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "knn_sklearn",
            exec_time,
            {"n_neighbors": n_neighbors, "weights": weights, **kwargs},
        )

        return mask, info

    # ============ МЕТОДЫ ОБНАРУЖЕНИЯ АНОМАЛИЙ ============
    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_isolation_forest(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Isolation Forest для сегментации (объект как аномалия).

        Строит изолирующие деревья. Пиксели, требующие меньше разбиений
        для изоляции, считаются аномалиями (объектами).

        Алгоритм:
        1. Извлечение признаков `(L,a,b,x,y, gradients)`.
        2. Обучение `IsolationForest(contamination='auto')`.
        3. Предсказание: `-1` = аномалия (объект), `1` = норма (фон).
        4. Конвертация и постобработка.

        Метод особенно эффективен для:
        - Обнаружения дефектов, пятен, инородных объектов
        - Изображений, где объект резко отличается от фона по статистике
        - Задач без ground truth и предварительных меток

        Args:
            img: Входное изображение.
            **kwargs: `n_estimators`, `contamination`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `contamination` оценивает долю объекта; `'auto'` подходит для большинства.
            - Чувствителен к выбросам; для шумных изображений используйте сглаживание.
            - Быстрый и масштабируемый; O(N log N).

        Example:
            ```python
            segmenter = SklearnSegmenter("isolation_forest", n_estimators=100)
            mask, _ = segmenter.segment(defect_detection_image)
            ```
        """
        h, w = img.shape[:2]
        features = self._extract_features(img)

        start_time: float = time.time()

        n_estimators = int(self.params.get("n_estimators", 100))
        contamination = str(self.params.get("contamination", "auto"))

        # Применяем Isolation Forest
        iso_forest = IsolationForest(
            n_estimators, contamination, max_samples="auto", random_state=42, n_jobs=-1
        )

        # Предсказываем аномалии (-1 - аномалия, 1 - норма)
        labels = iso_forest.fit_predict(features)

        # Преобразуем в маску (аномалии = объект)
        mask_raw = (labels == -1).reshape(h, w).astype(np.uint8) * 255
        mask: MaskArray = self._postprocess_mask(mask_raw)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "isolation_forest_sklearn",
            exec_time,
            {"n_estimators": n_estimators, "contamination": contamination, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_local_outlier_factor(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Local Outlier Factor для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        # Ограничиваем размер для производительности
        max_samples = 2000
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            sample_features = features[indices]
            use_sampling = True
        else:
            sample_features = features
            use_sampling = False

        n_neighbors = int(self.params.get("n_neighbors", 20))
        contamination = str(self.params.get("contamination", "auto"))

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
        mask_raw = (labels == -1).reshape(h, w).astype(np.uint8) * 255
        mask: MaskArray = self._postprocess_mask(mask_raw)

        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "local_outlier_factor_sklearn",
            exec_time,
            {"n_neighbors": n_neighbors, "contamination": contamination, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_one_class_svm(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """One-Class SVM для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

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
        mask_raw = (labels == -1).reshape(h, w).astype(np.uint8) * 255
        mask: MaskArray = self._postprocess_mask(mask_raw)

        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "one_class_svm_sklearn",
            exec_time,
            {"kernel": kernel, "gamma": gamma, "nu": nu, **kwargs},
        )

        return mask, info

    # ============ МЕТОДЫ РАЗЛОЖЕНИЯ ============
    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_pca_segmentation(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """PCA-based сегментация.

        Снижает размерность признаков до `n_components`, сохраняя максимальную дисперсию.
        Затем применяет K-Means в сжатом пространстве для выделения кластеров.

        Алгоритм:
        1. Извлечение признаков.
        2. Применение `PCA(n_components)`.
        3. Кластеризация `KMeans(n_clusters)` в PCA-пространстве.
        4. Создание маски из меток.

        Метод особенно эффективен для:
        - Изображений с высокой корреляцией каналов (мульти-спектральные)
        - Снижения шума перед кластеризацией
        - Ускорения работы на больших данных

        Args:
            img: Входное изображение.
            **kwargs: `n_components`, `n_clusters`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `n_components=2` или `3` достаточно для визуализации/кластеризации.
            - PCA необратим; некоторые мелкие детали могут быть потеряны.
            - Отлично комбинируется с `nmf_segmentation` для неотрицательных данных.

        Example:
            ```python
            segmenter = SklearnSegmenter("pca_segmentation", n_components=3, n_clusters=2)
            mask, _ = segmenter.segment(multispectral_image)
            ```
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        # Применяем PCA
        n_components = int(self.params.get("n_components", 3))
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(features)

        # Кластеризуем в новом пространстве
        n_clusters = int(self.params.get("n_clusters", 2))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(transformed)

        # Создаем маску
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)

        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "pca_segmentation_sklearn",
            exec_time,
            {"n_components": n_components, "n_clusters": n_clusters, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_nmf_segmentation(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Non-negative Matrix Factorization для сегментации.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        # Убедимся, что данные неотрицательные
        features_nonneg = features - features.min()

        # Применяем NMF
        n_components = int(self.params.get("n_components", 3))
        nmf = NMF(n_components=n_components, init="random", random_state=42)

        # Преобразуем данные
        transformed = nmf.fit_transform(features_nonneg)

        # Кластеризуем
        n_clusters = int(self.params.get("n_clusters", 2))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(transformed)

        # Создаем маску
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)

        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "nmf_segmentation_sklearn",
            exec_time,
            {"n_components": n_components, "n_clusters": n_clusters, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_tsne_segmentation(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """t-SNE для сегментации (визуализация + кластеризация).

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

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
        perplexity = int(self.params.get("perplexity", 30))
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
        n_clusters = int(self.params.get("n_clusters", 2))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(transformed)

        # Создаем маску
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)

        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "tsne_segmentation_sklearn",
            exec_time,
            {"perplexity": perplexity, "n_clusters": n_clusters, **kwargs},
        )

        return mask, info

    # ============ КОМБИНИРОВАННЫЕ МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_ensemble_clustering(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Ensemble clustering (комбинация нескольких методов).

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

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
        n_clusters = int(self.params.get("n_clusters", 2))
        spectral = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=42
        )

        labels = spectral.fit_predict(consensus)

        # Создаем маску
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "ensemble_clustering_sklearn",
            exec_time,
            {"n_clusters": n_clusters, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_color_spatial_clustering(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Кластеризация с учетом цвета и пространственных координат.

        Args:
            image: Входное изображение

        Returns:
            Бинарная маска сегментации
        """
        h, w = img.shape[:2]

        start_time: float = time.time()

        # Создаем признаки: цвет + пространственные координаты
        if len(img.shape) == 3:
            color = img.reshape(-1, 3).astype(np.float32) / 255.0
        else:
            color = img.reshape(-1, 1).astype(np.float32) / 255.0

        # Пространственные координаты
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        spatial = np.stack([x_coords.ravel() / w, y_coords.ravel() / h], axis=1)

        # Веса для баланса цвета и пространства
        color_weight = float(self.params.get("color_weight", 0.7))
        spatial_weight = float(self.params.get("spatial_weight", 0.3))

        # Комбинируем признаки
        features = np.hstack([color * color_weight, spatial * spatial_weight])

        # Масштабируем
        features = self._scaler.fit_transform(features)

        # Кластеризуем
        n_clusters = int(self.params.get("n_clusters", 3))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        # Создаем маску
        mask: MaskArray = self._create_mask_from_labels(labels, (h, w)).astype(np.uint8)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "color_spatial_clustering_sklearn",
            exec_time,
            {
                "n_clusters": n_clusters,
                "color_weight": color_weight,
                "spatial_weight": spatial_weight,
                **kwargs,
            },
        )

        return mask, info

    # ============ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ АВТОМАТИЧЕСКОЙ НАСТРОЙКИ ============
    # ──────────────────────────────────────────────────────────────────────
    def _estimate_optimal_clusters(
        self, features: FloatArray, max_k: int = 10, **kwargs: Any
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

    # ──────────────────────────────────────────────────────────────────────
    def _estimate_dbscan_params(
        self, features: FloatArray, **kwargs: Any
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

    # ──────────────────────────────────────────────────────────────────────
    def _estimate_meanshift_bandwidth(
        self, features: FloatArray, **kwargs: Any
    ) -> float:
        """Оценка bandwidth для MeanShift."""
        # Используем квантильный метод

        n_samples = min(500, features.shape[0])
        sample_indices = np.random.choice(features.shape[0], n_samples, replace=False)
        sample_features = features[sample_indices]

        nbrs = NearestNeighbors(n_neighbors=5).fit(sample_features)
        distances, _ = nbrs.kneighbors(sample_features)
        avg_distances = distances.mean(axis=1)

        quantile = float(self.params.get("quantile", 0.3))
        bandwidth = float(np.percentile(avg_distances, quantile * 100))

        return max(bandwidth, 0.1)

    # ──────────────────────────────────────────────────────────────────────
    def _estimate_gmm_components(
        self, features: FloatArray, max_components: int = 10, **kwargs: Any
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

    # ──────────────────────────────────────────────────────────────────────
    def _interpolate_labels(
        self,
        train_features: FloatArray,
        test_features: FloatArray,
        train_labels: ClusterLabels,
        method: str = "knn",
        **kwargs: Any,
    ) -> ClusterLabels:
        """Интерполяция меток с обучающего набора на тестовый.

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

        return cast(ClusterLabels, test_labels)

    # ============ ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ (для полноты) ============
    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_decision_tree(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Decision Tree для сегментации."""
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        # Создаем метки для обучения
        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]

        X_train = features[train_indices]
        y_train = labels_train[train_indices]

        max_depth = int(self.params.get("max_depth", 1))
        min_samples_split = int(self.params.get("min_samples_split", 2))

        # Обучаем Decision Tree
        dt = DecisionTreeClassifier(max_depth, min_samples_split, random_state=42)

        dt.fit(X_train, y_train)
        labels = dt.predict(features)

        mask_raw = labels.reshape(h, w).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time
        mask: MaskArray = self._postprocess_mask(mask_raw)
        info: SegmentationInfo = self._log_info(
            "decision_tree_sklearn",
            exec_time,
            {"max_depth": max_depth, "min_samples_split": min_samples_split, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_mlp(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Multi-layer Perceptron для сегментации."""
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

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

        mask_raw = labels.reshape(h, w).astype(np.uint8) * 255
        mask: MaskArray = self._postprocess_mask(mask_raw)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "mlp_sklearn",
            exec_time,
            {"hidden_layer_sizes": hidden_layer_sizes, **kwargs},
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_naive_bayes(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Naive Bayes для сегментации."""
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]

        X_train = features[train_indices]
        y_train = labels_train[train_indices]

        # Обучаем Gaussian Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        labels = nb.predict(features)

        mask_raw = labels.reshape(h, w).astype(np.uint8) * 255
        mask: MaskArray = self._postprocess_mask(mask_raw)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "naive_bayes_sklearn", exec_time, kwargs
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_lda(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Linear Discriminant Analysis для сегментации."""
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        labels_train = self._create_training_labels(h, w)
        train_indices = np.where(labels_train >= 0)[0]

        X_train = features[train_indices]
        y_train = labels_train[train_indices]

        # Обучаем LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        labels = lda.predict(features)

        mask_raw = labels.reshape(h, w).astype(np.uint8) * 255
        mask: MaskArray = self._postprocess_mask(mask_raw)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info("lda_sklearn", exec_time, kwargs)

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _create_training_labels(self, h: int, w: int, **kwargs: Any) -> np.ndarray:
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
    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_elliptic_envelope(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Elliptic Envelope для сегментации."""
        return self._sklearn_isolation_forest(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_bayesian_gmm(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Bayesian GMM для сегментации."""
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        n_components = (self.params.get("n_components", 10),)
        bgmm = BayesianGaussianMixture(n_components=n_components, random_state=42)

        labels = bgmm.fit_predict(features)
        mask_raw = self._create_mask_from_labels(labels, (h, w))
        mask: MaskArray = self._postprocess_mask(mask_raw)
        exec_time: float = time.time() - start_time

        info: SegmentationInfo = self._log_info(
            "bayesian_gmm_sklearn", exec_time, {"n_components": n_components, **kwargs}
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_ica_segmentation(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """ICA для сегментации."""
        return self._sklearn_pca_segmentation(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_isomap_segmentation(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Isomap для сегментации."""
        return self._sklearn_tsne_segmentation(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_spectral_embedding(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Spectral Embedding для сегментации."""
        return self._sklearn_spectral(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_variational_gmm(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Variational GMM для сегментации."""
        return self._sklearn_bayesian_gmm(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_density_based(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        return self._sklearn_dbscan(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_hdbscan_emulation(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Эмуляция HDBSCAN."""
        return self._sklearn_optics(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_graph_clustering(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Graph-based кластеризация."""
        return self._sklearn_spectral(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_modularity_clustering(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Modularity-based кластеризация."""
        return self._sklearn_spectral(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_self_training(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Self-training для сегментации."""
        return self._sklearn_random_forest(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_semi_supervised(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Semi-supervised сегментация."""
        return self._sklearn_random_forest(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_distance_matrix(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Distance matrix-based кластеризация."""
        return self._sklearn_spectral(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_affinity_propagation(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Affinity Propagation."""
        h, w = img.shape[:2]

        start_time: float = time.time()
        features = self._extract_features(img)

        # Используем K-Means как альтернативу (Affinity Propagation требователен)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(features)

        mask_raw = self._create_mask_from_labels(labels, (h, w))
        mask: MaskArray = self._postprocess_mask(mask_raw)
        exec_time: float = time.time() - start_time
        info: SegmentationInfo = self._log_info(
            "affinity_propagation_sklearn", exec_time, kwargs
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_qda(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Quadratic Discriminant Analysis."""
        return self._sklearn_lda(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_texture_clustering(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Текстурная кластеризация."""
        return self._sklearn_color_spatial_clustering(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_superpixel_clustering(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Кластеризация суперпикселей."""
        return self._sklearn_color_spatial_clustering(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_hierarchical_kmeans(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Иерархический K-Means."""
        return self._sklearn_agglomerative(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_pca_kmeans(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """PCA + K-Means."""
        return self._sklearn_pca_segmentation(img, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_gmm_vers2(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Gaussian Mixture Models."""
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img

        h, w = gray.shape

        start_time: float = time.time()

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

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_agglomerative_vers2(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
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

        info: SegmentationInfo = self._log_info(
            "agglomerative_sklearn",
            exec_time,
            {
                "n_clusters": n_clusters,
                #  "linkage": linkage,
                #  "affinity": affinity,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_spectral_vers2(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
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

        info: SegmentationInfo = self._log_info(
            "spectral_sklearn",
            exec_time,
            {
                "n_clusters": n_clusters,
                #  "n_neighbors": n_neighbors,
                #    "affinity": affinity,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_isolation_forest_vers2(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
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
        contamination = 0.1
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        labels = iso_forest.fit_predict(features)

        # Аномалии = объект
        mask = (labels == -1).reshape(h, w)

        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8) * 255

        info: SegmentationInfo = self._log_info(
            "isolation_forest_sklearn",
            exec_time,
            {
                # "n_estimators": n_estimators,
                "contamination": contamination,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_random_forest_vers2(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
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
            (center_h - obj_size) : (center_h + obj_size),
            (center_w - obj_size) : (center_w + obj_size),
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

        info = self._log_info("random_forest_sklearn", exec_time, kwargs)

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_svm_vers2(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
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
            (center_h - obj_size) : (center_h + obj_size),
            (center_w - obj_size) : (center_w + obj_size),
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
        kernel = "rbf"
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
        info = self._log_info(
            "svm_sklearn",
            exec_time,
            {
                # "C": C,
                "kernel": kernel,
                # "gamma": gamma,
                **kwargs,
            },
        )

        return mask, info

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_pca_segmentation_vers2(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
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
        n_components = 2
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(features)

        # Кластеризуем в новом пространстве
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(transformed)

        # Создаем маску
        mask = self._create_mask_from_labels(labels, (h, w))
        exec_time = time.time() - start_time
        mask = mask.astype(np.uint8) * 255

        info: SegmentationInfo = self._log_info(
            "pca_segmentation_sklearn",
            exec_time,
            {"n_components": n_components, "n_clusters": n_clusters, **kwargs},
        )

        return mask, info

    # ============ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _create_mask_from_labels_vers2(
        self, labels: np.ndarray, shape: Tuple, **kwargs: Any
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

        return cast(np.ndarray, mask)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_floodfill(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация методом заливки (Flood Fill).

        Заполняет связную область, начиная с `seed`, пока разница интенсивности
        между текущим пикселем и семенем не превысит `tolerance`.
        Использует оптимизированную реализацию из `skimage.segmentation.flood`.

        Алгоритм:
        1. Преобразование в grayscale.
        2. Вызов `flood()` с указанным допуском.
        3. Возврат булевой маски, конвертированной в `{0, 255}`.

        Метод особенно эффективен для:
        - Быстрого выделения однородных объектов по точке внутри
        - Предобработки для watershed или активных контуров
        - Интерактивных редакторов и инструментов разметки

        Args:
            img: Входное изображение.
            **kwargs: `seed` (Tuple[int, int]), `tolerance` (float ∈ [0, 1]).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `tolerance` интерпретируется относительно диапазона `[0, 1]`.
            - Автоматически обрабатывает 4- и 8-связность.
            - Быстрее ручного queue-based region growing за счёт C-бэкенда skimage.

        Example:
            ```python
            segmenter = SklearnSegmenter("floodfill", seed=(150, 200), tolerance=0.15)
            mask, _ = segmenter.segment(image)
            ```
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img

            h, w = gray.shape

            start_time: float = time.time()
            seed = self.params.get("seed", (w // 2, h // 2))
            tolerance = self.params.get("tolerance", 0.1)

            # Используем flood из skimage
            mask: MaskArray = (
                segmentation.flood(
                    gray, seed_point=seed[::-1], tolerance=tolerance
                ).astype(np.uint8)
                * 255
            )

            exec_time: float = time.time() - start_time

            info: SegmentationInfo = self._log_info(
                "floodfill_sklearn",
                exec_time,
                {"seed": seed, "tolerance": tolerance, **kwargs},
            )

            return mask, info

        except Exception as e:
            warnings.warn(f"FloodFill failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_active_contour(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация активными контурами (Snakes / Kass-Witkin-Terzopoulos).

        Инициализирует замкнутый контур (окружность) и деформирует его под действием:
        - Внутренних сил (упругость `alpha`, жёсткость `beta`)
        - Внешних сил (притяжение к границам `w_edge`, к линиям `w_line`)
        Минимизирует энергию контура до равновесия.

        Формула энергии:
        ```
        E = ∫ [α|v_s|² + β|v_ss|² + w_edge·E_edge + w_line·E_line] ds
        ```

        Алгоритм:
        1. Создание начального контура в центре изображения.
        2. Гауссово сглаживание изображения для стабилизации градиента.
        3. Итеративное обновление позиций точек контура через `active_contour()`.
        4. Заполнение полигона контура для получения бинарной маски.

        Метод особенно эффективен для:
        - Медицинских изображений с чёткими, но деформируемыми границами
        - Микроскопических снимков клеток и тканей
        - Задач, где важна гладкость и непрерывность контура

        Args:
            img: Входное изображение.
            **kwargs: `alpha`, `beta`, `gamma`, `w_edge`, `w_line`, `max_iter`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `gamma` — шаг времени (learning rate); слишком большой → нестабильность.
            - Требует инициализации контура близко к объекту для сходимости.
            - Возвращает маску внутри финального полигона, а не только линию.

        Example:
            ```python
            segmenter = SklearnSegmenter("active_contour", alpha=0.015, beta=10, max_iter=500)
            mask, _ = segmenter.segment(cell_image)
            ```
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img

            gray_norm = img_as_float(gray)
            h, w = gray_norm.shape
            start_time: float = time.time()

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
            mask_bool = np.zeros((h, w), dtype=bool)
            rr, cc = polygon(snake[:, 1], snake[:, 0], mask_bool.shape)
            mask_bool[rr, cc] = True

            mask: MaskArray = mask_bool.astype(np.uint8) * 255

            exec_time: float = time.time() - start_time
            info: SegmentationInfo = self._log_info(
                "active_contour_sklearn",
                exec_time,
                {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "w_edge": w_edge,
                    "w_line": w_line,
                    "max_iter": max_iter,
                    **kwargs,
                },
            )

            return mask, info

        except Exception as e:
            warnings.warn(f"Active Contour failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_gvf_contour(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация на основе Gradient Vector Flow (GVF).

        Расширяет область захвата активного контура, распространяя информацию
        о градиентах по всему изображению через диффузионный процесс.
        Позволяет контуру «чувствовать» границы даже в областях с нулевым градиентом.

        Уравнение диффузии:
        ```
        ∂v/∂t = μ·∇²v - |∇f|²·(v - ∇f)
        ```
        где `v` — векторное поле GVF, `μ` — коэффициент диффузии.

        Алгоритм:
        1. Вычисление градиентов Собеля `(Gx, Gy)`.
        2. Инициализация поля `v = ∇f`.
        3. Итеративное обновление через Лапласиан и вес границ.
        4. Бинаризация по 70-му процентилю магнитуды GVF.

        Метод особенно эффективен для:
        - Объектов с вогнутыми границами
        - Изображений с прерывистыми или слабыми краями
        - Задач, где стандартный gradient fails

        Args:
            img: Входное изображение.
            **kwargs: `mu` (float), `iterations` (int).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Вычислительно затратный; для больших изображений уменьшайте `iterations`.
            - `mu ∈ [0,1]` контролирует радиус распространения поля.
            - Результат часто требует морфологической постобработки.

        Example:
            ```python
            segmenter = SklearnSegmenter("gvf_contour", mu=0.1, iterations=50)
            mask, _ = segmenter.segment(concave_object_image)
            ```
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img

            gray_norm = img_as_float(gray)
            start_time: float = time.time()

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
            mask: MaskArray = (gvf_mag > threshold).astype(np.uint8) * 255

            exec_time: float = time.time() - start_time

            info: SegmentationInfo = self._log_info(
                "gvf_contour_sklearn",
                exec_time,
                {"mu": mu, "iterations": iterations, **kwargs},
            )

            return mask, info

        except Exception as e:
            warnings.warn(f"GVF Contour failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_morphological_snakes(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Морфологические змеи (Morphological Geodesic Active Contours).

        Эволюция уровня множества через морфологические операции вместо решения PDE.
        Устойчив к топологическим изменениям (разделение/слияние контуров).

        Алгоритм:
        1. Инициализация `init_level_set` (прямоугольник в центре).
        2. Итеративное применение `morphological_geodesic_active_contour`:
        - Сглаживание контура.
        - Движение по градиенту с учётом `balloon` силы.
        3. Возврат финальной бинарной маски.

        Метод особенно эффективен для:
        - Изображений с меняющейся топологией объектов
        - Задач, требующих стабильной сегментации без ручного ре-инициализации
        - Быстрой альтернативы классическим snakes

        Args:
            img: Входное изображение.
            **kwargs: `iterations`, `smoothing`, `threshold`, `balloon`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `balloon > 0` расширяет контур, `< 0` сжимает.
            - Не требует вычисления градиента вручную; использует внутренние фильтры.
            - Очень стабилен к шуму благодаря морфологическому сглаживанию.

        Example:
            ```python
            segmenter = SklearnSegmenter("morphological_snakes", iterations=50, balloon=1)
            mask, _ = segmenter.segment(topology_varying_image)
            ```
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img

            gray_norm = img_as_float(gray)
            h, w = gray_norm.shape
            start_time: float = time.time()

            # Начальный уровень (прямоугольник в центре)
            init_level_set = np.zeros(gray_norm.shape, dtype=np.int8)
            init_level_set[(h // 4) : (3 * h // 4), (w // 4) : (3 * w // 4)] = 1

            # Параметры
            iterations = self.params.get("iterations", 50)
            smoothing = self.params.get("smoothing", 1)
            threshold = self.params.get("threshold", 0.5)
            balloon = self.params.get("balloon", 1)

            # Применяем морфологические змеи
            mask: MaskArray = (
                morphological_geodesic_active_contour(
                    gray_norm,
                    iterations,
                    init_level_set=init_level_set,
                    smoothing=smoothing,
                    threshold=threshold,
                    balloon=balloon,
                ).astype(np.uint8)
                * 255
            )

            exec_time: float = time.time() - start_time
            info: SegmentationInfo = self._log_info(
                "morphological_snakes_sklearn",
                exec_time,
                {
                    "smoothing": smoothing,
                    "threshold": threshold,
                    "iterations": iterations,
                    "balloon": balloon,
                    **kwargs,
                },
            )

            return mask, info

        except Exception as e:
            warnings.warn(f"Morphological Snakes failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_chan_vese(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Модель Чан-Везе — активные контуры без градиентов.

        Энергетическая модель, минимизирующая внутрирегиональную дисперсию.
        Работает даже при отсутствии чётких границ, опираясь на однородность областей.

        Энергетический функционал:
        ```
        E = μ·Length(C) + λ₁·∫_in |I - c₁|² + λ₂·∫_out |I - c₂|²
        ```

        Алгоритм:
        1. Инициализация `level_set` в центре.
        2. Итеративная минимизация энергии через `morphological_chan_vese`.
        3. Возврат маски внутри контура.

        Метод особенно эффективен для:
        - Медицинских снимков с размытыми границами органов
        - Микроскопии с неоднородной текстурой
        - Объектов без явных краёв, но с однородной внутренней областью

        Args:
            img: Входное изображение.
            **kwargs: `mu`, `lambda1`, `lambda2`, `max_iter`, `tol`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `mu` контролирует гладкость; малые значения → следование за шумом.
            - Медленнее gradient-based методов, но устойчивее к отсутствию краёв.
            - `lambda1`/`lambda2` балансируют веса внутренней/внешней областей.

        Example:
            ```python
            segmenter = SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, max_iter=100)
            mask, _ = segmenter.segment(medical_scan)
            ```
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img

            gray_norm = img_as_float(gray)
            h, w = gray_norm.shape
            start_time: float = time.time()

            # Начальный уровень
            init_level_set = np.zeros(gray_norm.shape, dtype=np.int8)
            init_level_set[(h // 4) : (3 * h // 4), (w // 4) : (3 * w // 4)] = 1

            # Параметры
            mu = self.params.get("mu", 0.25)
            lambda1 = self.params.get("lambda1", 1.0)
            lambda2 = self.params.get("lambda2", 1.0)
            tol = self.params.get("tol", 1e-3)
            max_iter = self.params.get("max_iter", 100)

            # Применяем Chan-Vese
            mask: MaskArray = (
                morphological_chan_vese(
                    gray_norm,
                    max_iter,
                    init_level_set=init_level_set,
                    smoothing=1,
                    lambda1=lambda1,
                    lambda2=lambda2,
                ).astype(np.uint8)
                * 255
            )
            exec_time: float = time.time() - start_time

            info: SegmentationInfo = self._log_info(
                "chan_vese_sklearn",
                exec_time,
                {
                    "mu": mu,
                    "lambda1": lambda1,
                    "lambda2": lambda2,
                    "tol": tol,
                    "max_iter": max_iter,
                    **kwargs,
                },
                converged=exec_time < max_iter * 0.1,
            )

            return mask, info

        except Exception as e:
            warnings.warn(f"Chan-Vese failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_watershed(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Watershed (водораздел) сегментация.

        Интерпретирует изображение как топографическую поверхность. «Затопление»
        начинается от маркеров; границы между «водоёмами» становятся контурами объектов.

        Алгоритм:
        1. Создание маркеров: фон (25-й процентиль), объект (75-й процентиль).
        2. Вычисление градиента через Sobel.
        3. Применение `watershed(gradient, markers)`.
        4. Выделение маркера 2 как объекта.

        Метод особенно эффективен для:
        - Разделения слипшихся объектов (клетки, частицы)
        - Задач с чёткой разницей интенсивности объекта и фона
        - Предобработки для кластеризации

        Args:
            img: Входное изображение.
            **kwargs: (параметры не требуются, маркеры генерируются автоматически)

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - Склонен к oversegmentation при плохих маркерах.
            - Градиент Sobel усиливает границы, но может захватывать шум.
            - Для точного контроля маркеров используйте ручную разметку.

        Example:
            ```python
            segmenter = SklearnSegmenter("watershed")
            mask, _ = segmenter.segment(touching_cells_image)
            ```
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img

            gray_norm = img_as_float(gray)
            start_time: float = time.time()

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
            mask: MaskArray = (segmentation == 2).astype(np.uint8) * 255
            exec_time: float = time.time() - start_time

            info: SegmentationInfo = self._log_info(
                "watershed_sklearn", exec_time, kwargs
            )

            return mask, info

        except Exception as e:
            warnings.warn(f"Watershed failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_random_walker(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Сегментация методом Random Walker.

        Решает задачу на графе: каждый пиксель присваивается маркеру,
        до которого случайное блуждание доходит с наибольшей вероятностью.
        Учитывает глобальную структуру изображения.

        Алгоритм:
        1. Создание маркеров: центр (объект), углы (фон).
        2. Построение матрицы Лапласа изображения.
        3. Решение системы уравнений `L·u = b` для вероятностей.
        4. Присвоение меток по максимуму вероятности.

        Метод особенно эффективен для:
        - Зашумлённых изображений с размытыми границами
        - Полу-автоматической сегментации с пользовательскими метками
        - Задач, требующих учёта глобального контекста

        Args:
            img: Входное изображение.
            **kwargs: `beta` (float), `mode` (str).

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `beta` контролирует чувствительность к градиентам; большие → резкие границы.
            - `mode="cg_mg"` — самый быстрый для больших изображений.
            - Требует много памяти для >2000×2000; рассмотрите сэмплирование.

        Example:
            ```python
            segmenter = SklearnSegmenter("random_walker", beta=130, mode="cg_mg")
            mask, _ = segmenter.segment(noisy_microscopy_image)
            ```
        """
        try:
            if len(img.shape) == 3:
                gray = color.rgb2gray(img)
            else:
                gray = img

            gray_norm = img_as_float(gray)
            h, w = gray.shape
            start_time: float = time.time()

            # Создаём маркеры
            markers = np.zeros(gray.shape, dtype=np.uint8)

            # Центральная область - объект (маркер 2)
            markers[(h // 4) : (3 * h // 4), (w // 4) : (3 * w // 4)] = 2

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
            mask: MaskArray = (labels == 2).astype(np.uint8) * 255
            exec_time: float = time.time() - start_time

            info: SegmentationInfo = self._log_info(
                "random_walker_sklearn",
                exec_time,
                {"beta": beta, "mode": mode, **kwargs},
            )

            return mask, info

        except Exception as e:
            warnings.warn(f"Random Walker failed: {e}. Using fallback (Otsu).")
            return self._sklearn_otsu_thresholding(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_quickshift(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Quickshift сегментация.

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
            start_time: float = time.time()

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
            mask: MaskArray = (segments != bg_label).astype(np.uint8) * 255
            exec_time: float = time.time() - start_time

            info: SegmentationInfo = self._log_info(
                "quickshift_sklearn",
                exec_time,
                {
                    "kernel_size": kernel_size,
                    "max_dist": max_dist,
                    "ratio": ratio,
                    **kwargs,
                },
            )

            return mask, info

        except Exception as e:
            warnings.warn(f"Quickshift failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_slic(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """SLIC (Simple Linear Iterative Clustering) — суперпиксельная сегментация.

        Группирует пиксели в компактные регионы на основе цвета (Lab) и координат.
        Итеративно уточняет центры суперпикселей до сходимости.

        Алгоритм:
        1. Инициализация сетки центров на равномерной решётке.
        2. Назначение пикселей ближайшему центру в пространстве `(L,a,b,x,y)`.
        3. Пересчёт центров как средних значений.
        4. Повтор `max_iter` раз.
        5. Определение фона как крупнейшего суперпикселя.

        Метод особенно эффективен для:
        - Предобработки для сложных сегментаторов
        - Изображений с плавными цветовыми переходами
        - Снижения вычислительной нагрузки (работа с суперпикселями)

        Args:
            img: Входное изображение (RGB/Lab).
            **kwargs: `n_segments`, `compactness`, `max_iter`, `enforce_connectivity`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `compactness` балансирует цвет/пространство; малые → правильная сетка.
            - Быстрый и стабильный; стандарт де-факто для суперпикселей.
            - Возвращает бинарную маску, исключая фон.

        Example:
            ```python
            segmenter = SklearnSegmenter("slic", n_segments=200, compactness=10.0)
            mask, _ = segmenter.segment(portrait_image)
            ```
        """
        try:
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            start_time: float = time.time()

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
            mask: MaskArray = (segments != bg_label).astype(np.uint8) * 255
            exec_time: float = time.time() - start_time

            info: SegmentationInfo = self._log_info(
                "slic_sklearn",
                exec_time,
                {
                    "n_segments": n_segments,
                    "compactness": compactness,
                    "max_iter": max_iter,
                    "enforce_connectivity": enforce_connectivity,
                    **kwargs,
                },
            )

            return mask, info

        except Exception as e:
            warnings.warn(f"SLIC failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_felzenszwalb(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Felzenszwalb сегментация — иерархическая на основе MST.

        Строит минимальное остовное дерево в пространстве пикселей.
        Объединяет соседние регионы, если внутреннее различие < межрегионального.

        Алгоритм:
        1. Построение графа смежности пикселей.
        2. Сортировка рёбер по весу (разница цветов).
        3. Итеративное объединение компонент через Union-Find.
        4. Фильтрация по `min_size`.
        5. Выделение объекта (все кроме крупнейшего региона).

        Метод особенно эффективен для:
        - Изображений с объектами разного масштаба
        - Естественных сцен с плавными переходами
        - Быстрой графовой сегментации без параметров кластеризации

        Args:
            img: Входное изображение (RGB).
            **kwargs: `scale`, `sigma`, `min_size`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `scale` контролирует размер регионов; большие → грубее.
            - `sigma` сглаживает перед построением графа.
            - Адаптивен к локальному контрасту; не требует `n_clusters`.

        Example:
            ```python
            segmenter = SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50)
            mask, _ = segmenter.segment(natural_scene)
            ```
        """
        try:
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            start_time: float = time.time()

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
            mask: MaskArray = (segments != bg_label).astype(np.uint8) * 255
            exec_time: float = time.time() - start_time

            info: SegmentationInfo = self._log_info(
                "felzenszwalb_sklearn",
                exec_time,
                {"scale": scale, "sigma": sigma, "min_size": min_size, **kwargs},
                n_segments=len(unique),
            )

            return mask, info

        except Exception as e:
            warnings.warn(f"Felzenszwalb failed: {e}. Using fallback (KMeans).")
            return self._sklearn_kmeans_segmentation(img)

    # ──────────────────────────────────────────────────────────────────────
    def _sklearn_grabcut(
        self,
        img: ImageArray,
        **kwargs: Any,
    ) -> Tuple[MaskArray, SegmentationInfo]:
        """Grabcut эмуляция через Random Forest.

        Имитирует интерактивный GrabCut: пользователь задаёт прямоугольник,
        алгоритм обучает классификатор на цветах внутри/снаружи и предсказывает маску.

        Алгоритм:
        1. Создание маркеров внутри `rect` (объект) и углов (фон).
        2. Извлечение признаков: `(R,G,B, x/W, y/H)`.
        3. Обучение `RandomForestClassifier` на размеченных пикселях.
        4. Предсказание для всего изображения.
        5. Конвертация в `{0, 255}`.

        Метод особенно эффективен для:
        - Быстрой эмуляции GrabCut без OpenCV GMM
        - Изображений с чётким цветовым разделением объекта и фона
        - Задач, где важна интерпретируемость модели

        Args:
            img: Входное изображение (RGB).
            **kwargs: `rect` (Tuple[int,int,int,int]), `n_estimators`.

        Returns:
            Tuple[MaskArray, SegmentationInfo]: Маска и метаданные.

        Note:
            - `rect` должен содержать объект; углы всегда считаются фоном.
            - Быстрее оригинального GrabCut за счёт векторизации RF.
            - Чувствителен к перекрытию цветов объекта и фона.

        Example:
            ```python
            segmenter = SklearnSegmenter("grabcut", rect=(100, 100, 400, 300), n_estimators=50)
            mask, _ = segmenter.segment(object_image)
            ```
        """
        try:
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img

            h, w = img_rgb.shape[:2]
            start_time: float = time.time()

            # Создаём начальную маску с маркерами
            mask: MaskArray = np.zeros((h, w), dtype=np.uint8)

            # Прямоугольник в центре - вероятный передний план
            rect = self.params.get("rect", (w // 4, h // 4, w // 2, h // 2))
            x, y, rw, rh = rect
            mask[y : (y + rh), x : (x + rw)] = 3  # Вероятный передний план

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
            mask = mask_result.astype(np.uint8) * 255
            exec_time: float = time.time() - start_time

            info: SegmentationInfo = self._log_info(
                "grabcut_sklearn",
                exec_time,
                {"rect": rect, "n_estimators": n_estimators, **kwargs},
            )

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
