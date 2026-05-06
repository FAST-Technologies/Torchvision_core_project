# segmenters/OpenCVSegmenter.py

"""
Модуль классических методов сегментации на базе OpenCV.

Предоставляет класс `OpenCVSegmenter`, реализующий 50+ алгоритмов сегментации изображений,
включая пороговые методы, детекторы границ, кластеризацию, активные контуры и интерактивные подходы.

Поддерживаемые категории методов:
1. **Пороговые методы**: глобальный порог, Оцу, адаптивные методы (Niblack, Sauvola, Bernsen, Phansalkar)
2. **Детекторы границ**: Canny, Sobel, Prewitt, Scharr, Roberts, LoG, DoG, Marr-Hildreth, фазовая конгруэнтность
3. **Региональные методы**: рост регионов, split-and-merge, floodfill
4. **Кластеризация**: K-Means, DBSCAN, MeanShift
5. **Активные контуры**: Snakes, GVF, морфологические змеи, модель Чан-Везе
6. **Watershed и графовые методы**: классический watershed, random walker
7. **Суперпиксели**: SLIC, Felzenszwalb, QuickShift
8. **Интерактивные методы**: GrabCut

Все методы возвращают бинарную маску (0/255) типа `np.ndarray` с dtype `uint8`.

Example:
    ```python
    from segmenters.OpenCVSegmenter import OpenCVSegmenter
    import cv2

    # Загрузка изображения
    image = cv2.imread("sample.jpg")

    # Создание сегментера с методом адаптивного порога
    segmenter = OpenCVSegmenter(
        method="adaptive_thresholding",
        block_size=11,
        C=2
    )

    # Выполнение сегментации
    mask = segmenter.segment(image)

    # Или с возвратом визуализации
    overlay, mask = segmenter.segment_with_mask(image, alpha=0.7)

    # Сохранение результата
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("overlay.png", overlay)
    ```

Attributes:
    METHODS_BY_LIBRARY: Глобальный словарь профилей методов по библиотекам.
    ALL_METHODS: Плоский словарь всех доступных методов для быстрого поиска.

Note:
    - Для методов, требующих grayscale, конвертация выполняется автоматически.
    - Параметры в диапазоне [0, 1] автоматически масштабируются в [0, 255] при необходимости.
    - Все методы поддерживают произвольные kwargs для переопределения параметров по умолчанию.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from segmenters.BaseSegmenter import BaseSegmenter

import cv2
import numpy as np
import numpy.typing as npt
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Callable,
)
import warnings
from collections import deque
from scipy import ndimage
from scipy.ndimage import gaussian_filter, laplace
import time

from sklearn.cluster import DBSCAN

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────

ImageArray = npt.NDArray[np.uint8]
"""Тип для входного изображения (RGB или grayscale)."""

GrayImage = npt.NDArray[np.uint8]
"""Тип для grayscale изображения."""

MaskArray = npt.NDArray[np.uint8]
"""Тип для бинарной маски сегментации (0/255)."""

FloatArray = npt.NDArray[np.float32]
"""Тип для массивов с плавающей точкой."""

ParamsDict = Dict[str, Any]
"""Тип для словаря параметров метода."""

SegmentationMethod = Callable[..., MaskArray]
"""Тип для функции сегментации."""


# ──────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ──────────────────────────────────────────────────────────────────────
class OpenCVSegmenter(BaseSegmenter):
    """
    Класс для реализации методов сегментации изображений с использованием OpenCV.

    Поддерживает 50+ алгоритмов, сгруппированных по категориям:
    - Пороговые методы (14 вариантов)
    - Детекторы границ (10 вариантов)
    - Региональные методы (3 варианта)
    - Кластеризация (3 варианта)
    - Активные контуры (4 варианта)
    - Watershed и графовые методы (2 варианта)
    - Суперпиксели (3 варианта)
    - Интерактивные методы (1 вариант)

    Все методы наследуют интерфейс `BaseSegmenter` и возвращают:
    - `segment()`: бинарную маску `MaskArray` (0/255, dtype=uint8)
    - `segment_with_mask()`: кортеж `(визуализация, маска)`

    Attributes:
        method (str): Название текущего метода сегментации.
        params (ParamsDict): Словарь параметров метода (после адаптации).
        raw_params (ParamsDict): Исходные параметры, переданные при инициализации.
        methods (Dict[str, SegmentationMethod]): Словарь зарегистрированных методов.
        info (ParamsDict): Метаданные последнего выполнения (время, параметры).
        _needs_gray (bool): Флаг необходимости конвертации в grayscale.

    Example:
        ```python
        # Базовое использование
        segmenter = OpenCVSegmenter("canny_edge", low=0.1, high=0.3)
        mask = segmenter.segment(image)

        # С параметрами по умолчанию из профиля
        segmenter = OpenCVSegmenter("threshold_sauvola")
        print(segmenter.params)  # {'window_size': 15, 'k': 0.5, 'r': 128}

        # Переопределение параметра
        segmenter = OpenCVSegmenter("threshold_sauvola", window_size=25)
        ```

    Note:
        - Параметры в диапазоне [0, 1] (например, `threshold=0.5`) автоматически
          масштабируются в [0, 255] для методов OpenCV, если это необходимо.
        - Для методов, требующих нечётный `block_size` или `window_size`, значение
          автоматически корректируется (`even → odd`).
        - Методы кластеризации и суперпикселей могут быть медленными на больших
          изображениях (>1000×1000); рекомендуется предварительный ресайз.
    """

    def __init__(self, method: str = "global_thresholding", **kwargs: Any) -> None:
        """
        Инициализация сегментера с указанным методом и параметрами.

        Args:
            method: Название метода сегментации. Должно присутствовать в списке
                   доступных методов (см. `_setup_methods()`).
            **kwargs: Параметры метода. Поддерживаются:
                     - Прямые параметры (например, `threshold=0.5`)
                     - Параметры с конвертацией [0,1] → [0,255] (например, `low=0.1`)
                     - Параметры с переименованием (например, `n_iterations` → `iterations` для GrabCut)

        Raises:
            ValueError: Если `method` не найден в списке доступных методов.

        Example:
            ```python
            # Простая инициализация
            seg = OpenCVSegmenter("otsu_thresholding")

            # С параметрами
            seg = OpenCVSegmenter(
                "adaptive_thresholding",
                block_size=15,
                C=5
            )

            # С конвертацией порога [0,1] → [0,255]
            seg = OpenCVSegmenter("global_thresholding", threshold=0.6)  # → 153
            ```
        """
        super().__init__()
        self.method: str = method
        self.raw_params: ParamsDict = kwargs
        self.params: ParamsDict = self._adapt_params(kwargs.copy())
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

    # ──────────────────────────────────────────────────────────────────────
    def _adapt_params(self, params: ParamsDict) -> ParamsDict:
        """
        Адаптация параметров: конвертация диапазонов и переименование ключей.

        Выполняет три типа преобразований:
        1. **Конвертация интенсивности**: значения [0, 1] → [0, 255] для параметров
           `threshold`, `low`, `high`, `t1`, `t2`, `contrast_threshold`.
        2. **Конвертация смещений**: значения [0, 1] → [0, 255] для параметров
           `C`, `tolerance`, `c` (смещения в адаптивных порогах).
        3. **Переименование ключей**: приведение имён параметров к стандарту OpenCV
           (например, `n_iterations` → `iterations` для GrabCut).

        Args:
            params: Словарь исходных параметров.

        Returns:
            ParamsDict: Словарь адаптированных параметров.

        Example:
            ```python
            # Конвертация порога
            adapted = segmenter._adapt_params({"threshold": 0.5})
            print(adapted["threshold"])  # 127

            # Переименование ключа
            adapted = segmenter._adapt_params({"n_iterations": 10})
            print(adapted["iterations"])  # 10
            ```
        """
        adapted: ParamsDict = params.copy()

        # Конвертация значений (0.0-1.0 -> 0-255) ---
        # Параметры, которые точно являются порогами яркости
        intensity_params: List[str] = [
            "threshold",
            "low",
            "high",
            "t1",
            "t2",
            "contrast_threshold",
        ]
        for key in intensity_params:
            if key in adapted:
                val = adapted[key]
                if isinstance(val, (int, float)) and 0.0 <= val <= 1.0:
                    adapted[key] = int(val * 255)

        # Параметры, зависящие от интенсивности (смещения)
        offset_params: List[str] = [
            "C",
            "tolerance",
            "c",
            #  "k"
        ]
        for key in offset_params:
            if key in adapted:
                val = adapted[key]
                if isinstance(val, (int, float)) and 0.0 <= abs(val) <= 1.0:
                    adapted[key] = int(val * 255)

        mapping: Dict[str, str] = {}
        if self.method == "grabcut":
            mapping = {"n_iterations": "iterations"}
        elif self.method == "dbscan_segmentation":
            mapping = {"epsilon": "eps", "min_points": "min_samples"}
        elif self.method == "kmeans_segmentation":
            mapping = {"n_clusters": "k"}
        elif self.method == "adaptive_thresholding":
            pass

        final_params: ParamsDict = {}
        for key, value in adapted.items():
            new_key: str = mapping.get(key, key)
            final_params[new_key] = value

        return final_params

    # ──────────────────────────────────────────────────────────────────────
    def _setup_methods(self, **kwargs: Any) -> None:
        """
        Регистрация всех доступных методов сегментации в словаре `self.methods`.

        Методы сгруппированы по категориям:
        - Пороговые (14): от простого глобального до сложных адаптивных
        - Граничные (10): от классических операторов до фазовой конгруэнтности
        - Региональные (3): рост регионов, split-and-merge, floodfill
        - Кластеризация (3): K-Means, DBSCAN, MeanShift
        - Активные контуры (4): Snakes, GVF, морфологические змеи, Чан-Везе
        - Watershed (2): классический и random walker
        - Суперпиксели (3): SLIC, Felzenszwalb, QuickShift
        - Интерактивные (1): GrabCut

        Raises:
            ValueError: Если `self.method` не найден в зарегистрированных методах.

        Note:
            Все методы должны соответствовать сигнатуре:
            ```python
            def method_name(self, img: ImageArray, **kwargs) -> MaskArray:
                ...
                return mask
            ```
        """
        self.methods: Dict[str, SegmentationMethod] = {
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
            available: List[str] = list(self.methods.keys())
            raise ValueError(
                f"Неизвестный метод: {self.method}. " f"Доступные методы: {available}"
            )

    # ──────────────────────────────────────────────────────────────────────
    def _log_info(self, method_name: str, exec_time: float, params: ParamsDict) -> None:
        """
        Вспомогательный метод для логирования информации о выполнении.

        Сохраняет метаданные выполнения в атрибут `self.info` и выводит в консоль.

        Args:
            method_name: Название выполненного метода.
            exec_time: Время выполнения в секундах.
            params: Словарь использованных параметров.

        Note:
            Метод вызывается автоматически в конце каждого приватного метода
            сегментации (например, `_opencv_canny_edge`).
        """
        self.info: ParamsDict = {
            "method": method_name,
            "parameters": params,
            "execution_time": exec_time,
        }

    # ──────────────────────────────────────────────────────────────────────
    def segment(  # type: ignore[override]
        self, image: ImageArray, **kwargs: Any
    ) -> MaskArray:
        """
        Основной метод сегментации изображения.

        Выполняет предобработку (конвертацию в grayscale при необходимости) и
        вызывает соответствующий метод из `self.methods`.

        Args:
            image: Входное изображение. Поддерживаются форматы:
                  - Grayscale: `(H, W)`, dtype=uint8
                  - RGB: `(H, W, 3)`, dtype=uint8
                  - BGR: `(H, W, 3)`, dtype=uint8 (автоматически конвертируется)
            **kwargs: Дополнительные параметры для переопределения параметров метода.

        Returns:
            MaskArray: Бинарная маска сегментации формы `(H, W)`, dtype=uint8,
                      значения {0, 255}, где 255 = объект, 0 = фон.

        Raises:
            ValueError: Если метод не найден или входное изображение имеет
                       неподдерживаемый формат.

        Example:
            ```python
            segmenter = OpenCVSegmenter("canny_edge", low=0.1, high=0.3)
            mask = segmenter.segment(image)
            # mask.shape == image.shape[:2], mask.dtype == np.uint8
            ```
        """
        img_array: GrayImage = self.preprocess_image(image, as_gray=self._needs_gray)
        # print(f"Image after OpenCV preprocessing: {image}")

        mask: MaskArray = self.methods[self.method](img_array, **kwargs)
        # print(f"Mask after OpenCV segment: {mask}")
        return mask

    # ──────────────────────────────────────────────────────────────────────
    def segment_with_mask(  # type: ignore[override]
        self, image: ImageArray, alpha: float = 0.9, **kwargs: Any
    ) -> Tuple[ImageArray, MaskArray]:
        """
        Сегментация с возвратом визуализации и бинарной маски.

        Создаёт наложение маски на оригинальное изображение с прозрачностью `alpha`.

        Args:
            image: Входное изображение (см. `segment()`).
            alpha: Коэффициент наложения маски [0, 1]:
                  - 0.0 = только оригинальное изображение
                  - 1.0 = только маска (красным цветом)
                  - 0.9 = по умолчанию (сильный акцент на маске)
            **kwargs: Дополнительные параметры для метода сегментации.

        Returns:
            Tuple[ImageArray, MaskArray]:
                - `overlay`: Визуализация формы `(H, W, 3)`, dtype=uint8, RGB.
                - `mask`: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255}.

        Example:
            ```python
            segmenter = OpenCVSegmenter("otsu_thresholding")
            overlay, mask = segmenter.segment_with_mask(image, alpha=0.6)
            # overlay можно сохранить через cv2.imwrite() или PIL
            ```
        """
        image = self.preprocess_image(image)
        # print(f"Image after OpenCV preprocessing with mask: {image}")
        mask: MaskArray = self.segment(image, **kwargs)

        if mask.dtype != np.uint8:
            mask = (
                (mask * 255).astype(np.uint8)
                if mask.max() <= 1.0
                else mask.astype(np.uint8)
            )

        # Создаем визуализацию
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            overlay = image.copy()
        # Наложение маски (красный цвет для объекта)
        overlay[mask > 127] = [255, 0, 0]
        base_img = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
        )

        result = cv2.addWeighted(overlay, alpha, base_img, 1 - alpha, 0).astype(
            np.uint8
        )

        # print(f"Mask after OpenCV segment_with_mask: {mask}")
        # print(f"Result after OpenCV segment_with_mask: {result}")
        return result, mask

    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    # ============ ПОРОГОВЫЕ МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _opencv_global_thresholding(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Глобальная пороговая сегментация.

        Применяет фиксированный порог яркости ко всему изображению.
        Пиксели ярче порога становятся белыми (объект, 255), остальные — чёрными (фон, 0).

        Формула:
        ```
        mask[x, y] = 255 if img[x, y] > threshold else 0
        ```

        Args:
            img: Входное изображение (grayscale).
            **kwargs: Дополнительные параметры:
                     - `threshold` (float): Порог яркости [0, 255]. По умолчанию 127.

        Returns:
            MaskArray: Бинарная маска (0/255).

        Note:
            - Простой и быстрый метод, но чувствителен к неравномерному освещению.
            - Для адаптивного порога используйте `_opencv_adaptive_thresholding`.

        Example:
            ```python
            mask = segmenter._opencv_global_thresholding(gray, threshold=100)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        # print(f"Gray after OpenCV_thresholding_global: {gray}")

        start_time: float = time.time()

        threshold: float = self.params.get("threshold", 127)
        _, mask_raw = cv2.threshold(gray, threshold, 255.0, cv2.THRESH_BINARY)
        mask: MaskArray = mask_raw.astype(np.uint8)  # type: ignore[assignment]

        exec_time: float = time.time() - start_time

        self._log_info(
            "global_thresholding_opencv",
            exec_time,
            {"threshold": threshold, **kwargs},
        )

        # print(f"Mask after OpenCV_thresholding_global: {mask}")
        print(f"Info after OpenCV_thresholding_global: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_adaptive_thresholding(
        self, img: ImageArray, **kwargs: Any
    ) -> MaskArray:
        """
        Адаптивная пороговая сегментация (Gaussian).

        Вычисляет локальный порог для каждой области изображения на основе
        взвешенной суммы соседних пикселей (гауссово ядро). Эффективна при
        неравномерном освещении.

        Формула порога для пикселя (x, y):
        ```
        T(x, y) = mean(neighbors) - C
        ```

        Args:
            img: Входное изображение (grayscale).
            **kwargs: Дополнительные параметры:
                     - `block_size` (int): Размер окрестности (нечётный, 3–99). По умолчанию 11.
                     - `C` (int): Константа-смещение [-20, 20]. По умолчанию 2.

        Returns:
            MaskArray: Бинарная маска (0/255).

        Note:
            - `block_size` должен быть нечётным; чётные значения автоматически
              корректируются (`block_size += 1`).
            - Меньший `block_size` → более детальные границы, но больше шума.
            - Отрицательный `C` делает порог более строгим (меньше объекта).

        Example:
            ```python
            mask = segmenter._opencv_adaptive_thresholding(
                gray, block_size=15, C=5
            )
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        # print(f"Gray after OpenCV_thresholding_adaptive: {gray}")

        start_time: float = time.time()

        block_size: int = int(self.params.get("block_size", 11))
        C: int = int(self.params.get("C", 2))

        if block_size % 2 == 0:
            block_size += 1

        mask_raw = cv2.adaptiveThreshold(
            gray,
            255.0,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C,
        )
        mask: MaskArray = mask_raw.astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "adaptive_thresholding_opencv",
            exec_time,
            {"block_size": block_size, "C": C, **kwargs},
        )

        # print(f"Mask after OpenCV_thresholding_adaptive: {mask}")
        print(f"Info after OpenCV_thresholding_adaptive: {self._log_info}")
        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_otsu_thresholding(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Автоматическая бинаризация по методу Оцу.

        Находит оптимальный порог, максимизирующий межклассовую дисперсию
        между фоном и объектом. Не требует ручного подбора порога.

        Алгоритм:
        1. Строит гистограмму интенсивностей.
        2. Перебирает все возможные пороги.
        3. Выбирает порог с максимальной межклассовой дисперсией.

        Args:
            img: Входное изображение (grayscale).
            **kwargs: Дополнительные параметры (не используются).

        Returns:
            MaskArray: Бинарная маска (0/255).

        Note:
            - Эффективен для бимодальных гистограмм (чёткое разделение фона/объекта).
            - Может давать плохие результаты для унимодальных или мультимодальных
              гистограмм.
            - Возвращаемый порог доступен как `ret` в `cv2.threshold()`.

        Example:
            ```python
            mask = segmenter._opencv_otsu_thresholding(gray)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        # print(f"Gray after OpenCV_thresholding_otsu: {gray}")
        start_time: float = time.time()
        _, mask_raw = cv2.threshold(
            gray, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask: MaskArray = mask_raw.astype(np.uint8)
        exec_time: float = time.time() - start_time

        self._log_info(
            "otsu_thresholding_opencv",
            exec_time,
            {**kwargs},
        )

        # print(f"Mask after OpenCV_thresholding_otsu: {mask}")
        print(f"Info after OpenCV_thresholding_otsu: {self._log_info}")
        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_threshold_niblack(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Адаптивная пороговая обработка по методу Ниблака.

        Вычисляет локальный порог для каждого пикселя на основе статистик окрестности:
        ```
        T(x, y) = μ(x, y) + k · σ(x, y)
        ```
        где:
        - `μ` — локальное среднее интенсивности в окне `window_size × window_size`
        - `σ` — локальное стандартное отклонение
        - `k` — параметр чувствительности (отрицательный для выделения тёмных объектов)

        Метод эффективен для изображений с неравномерным освещением и умеренным шумом.

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Вычисление локального среднего `μ` через `cv2.boxFilter`.
        3. Вычисление локальной дисперсии: `σ² = E[X²] - (E[X])²`.
        4. Расчёт порога: `threshold = μ + k · σ`.
        5. Бинаризация: `mask = (img > threshold) * 255`.

        Args:
            img: Входное изображение. Поддерживаются форматы:
                - Grayscale: `(H, W)`, dtype=uint8
                - RGB/BGR: `(H, W, 3)`, dtype=uint8 (автоматически конвертируется)
            **kwargs: Дополнительные параметры:
                    - `window_size` (int): Размер окна для локальных статистик (нечётный, 3–99).
                    По умолчанию 15.
                    - `k` (float): Параметр чувствительности [-1.0, 1.0].
                    По умолчанию -0.2 (отрицательный для выделения тёмных объектов на светлом фоне).

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, значения {0, 255},
                    где 255 = объект, 0 = фон.

        Raises:
            ValueError: Если `window_size` чётный (автоматически корректируется).

        Note:
            - Отрицательный `k` лучше подходит для текста/документов (тёмный объект на светлом фоне).
            - Положительный `k` — для светлых объектов на тёмном фоне.
            - Метод чувствителен к шуму; для зашумлённых изображений рассмотрите Sauvola или Phansalkar.
            - Вычисление `σ` через `boxFilter` быстрее, чем через `cv2.blur`, но может давать небольшие
            численные отличия из-за порядка операций.

        Example:
            ```python
            # Базовое использование для скана документа
            segmenter = OpenCVSegmenter("threshold_niblack", window_size=15, k=-0.2)
            mask = segmenter.segment(document_image)

            # Для светлых объектов на тёмном фоне
            segmenter = OpenCVSegmenter("threshold_niblack", window_size=25, k=0.3)
            mask = segmenter.segment(microscopy_image)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        # print(f"Gray after OpenCV_thresholding_niblack: {gray}")

        start_time: float = time.time()

        window_size: int = int(self.params.get("window_size", 15))
        k: float = float(self.params.get("k", -0.2))

        # Вычисление локальных статистик
        # E[X] — локальное среднее
        mean_raw = cv2.boxFilter(
            gray.astype(np.float32), cv2.CV_32F, (window_size, window_size)
        )
        mean: FloatArray = mean_raw.astype(np.uint8)
        # E[X²] — среднее квадратов
        mean_sq_raw = cv2.boxFilter(
            (gray.astype(np.float32) ** 2), cv2.CV_32F, (window_size, window_size)
        )
        mean_sq: FloatArray = mean_sq_raw.astype(np.uint8)

        # σ = sqrt(E[X²] - (E[X])²), с защитой от отрицательных значений из-за численных ошибок
        variance: FloatArray = np.maximum(mean_sq - mean**2, 0)
        std: FloatArray = np.sqrt(variance)

        # Или
        # mean = cv2.blur(gray, (window_size, window_size))
        # std = np.sqrt(cv2.boxFilter(gray.astype(float)**2, -1, (window_size, window_size)) - mean**2)

        # Расчёт порога Ниблака
        threshold: FloatArray = mean + k * std

        # Бинаризация
        mask: MaskArray = (gray.astype(np.float32) > threshold).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time

        self._log_info(
            "niblack_thresholding_opencv",
            exec_time,
            {"window_size": window_size, "k": k, **kwargs},
        )

        # print(f"Mask after OpenCV_thresholding_niblack: {mask}")
        print(f"Info after OpenCV_thresholding_niblack: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_threshold_sauvola(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Адаптивная пороговая обработка по методу Сауволы.

        Улучшенная версия метода Ниблака, оптимизированная для документов и изображений
        с низким контрастом. Порог вычисляется для каждого пикселя на основе локальных
        статистик в окне `window_size × window_size`:

        ```
        T(x, y) = μ(x, y) · [1 + k · (σ(x, y) / R - 1)]
        ```

        где:
        - `μ` — локальное среднее интенсивности
        - `σ` — локальное стандартное отклонение
        - `k` — параметр чувствительности [0, 1] (по умолчанию 0.5)
        - `R` — динамический диапазон интенсивности (обычно 128 для 8-битных изображений)

        Метод особенно эффективен для:
        - Сканированных документов с неравномерным освещением
        - Микроскопических изображений с низким контрастом
        - Текстовых изображений с зашумлённым фоном

        Алгоритм:
        1. Конвертация входного изображения в grayscale при необходимости.
        2. Вычисление локального среднего `μ` через `cv2.boxFilter`.
        3. Вычисление локальной дисперсии: `σ² = E[X²] - (E[X])²`.
        4. Расчёт порога Сауволы по формуле выше.
        5. Бинаризация: `mask = (img > threshold) * 255`.

        Args:
            img: Входное изображение. Поддерживаются форматы:
                - Grayscale: `(H, W)`, dtype=uint8
                - RGB/BGR: `(H, W, 3)`, dtype=uint8 (автоматически конвертируется)
            **kwargs: Дополнительные параметры:
                    - `window_size` (int): Размер окна для локальных статистик (нечётный, 3–99).
                    По умолчанию 15. Меньшие окна → более детальные границы, но больше шума.
                    - `k` (float): Параметр чувствительности [0.0, 1.0].
                    По умолчанию 0.5. Меньшие значения → более строгий порог.
                    - `r` (float): Динамический диапазон [50.0, 255.0].
                    По умолчанию 128.0. Для нормализованных изображений [0,1] используйте 0.5.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, значения {0, 255},
                    где 255 = объект (текст/объект), 0 = фон.

        Raises:
            ValueError: Если `window_size` чётный (автоматически корректируется: `window_size += 1`).

        Note:
            - Метод Сауволы более устойчив к шуму, чем Ниблак, за счёт нормализации по `R`.
            - Для нормализованных изображений [0, 1] установите `r=0.5` вместо `r=128`.
            - Оптимальный `window_size` зависит от размера символов/объектов:
            * Текст: 15–25
            * Микроскопия: 25–45
            * Крупные объекты: 45–99
            - Вычисление `σ` через `boxFilter` быстрее, чем через `cv2.blur`, но может давать
            небольшие численные отличия из-за порядка операций.

        Example:
            ```python
            # Базовое использование для скана документа
            segmenter = OpenCVSegmenter("threshold_sauvola", window_size=15, k=0.5, r=128)
            mask = segmenter.segment(document_image)

            # Для нормализованного изображения [0, 1]
            segmenter = OpenCVSegmenter("threshold_sauvola", window_size=25, k=0.3, r=0.5)
            mask = segmenter.segment(normalized_image)

            # Переопределение параметров при вызове
            mask = segmenter.segment(image, window_size=35, k=0.4)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        # print(f"Gray after OpenCV_thresholding_sauvola: {gray}")

        start_time: float = time.time()

        # Получение параметров с типизацией и значениями по умолчанию
        window_size: int = int(self.params.get("window_size", 15))
        k: float = float(self.params.get("k", 0.5))
        r: float = float(self.params.get("r", 128.0))

        # Коррекция чётного window_size (требуется нечётное для симметричного окна)
        if window_size % 2 == 0:
            window_size += 1

        # Вычисление локальных статистик
        # E[X] — локальное среднее интенсивности
        mean_raw = cv2.boxFilter(
            gray.astype(np.float32), cv2.CV_32F, (window_size, window_size)
        )
        mean: FloatArray = mean_raw.astype(np.uint8)
        # E[X²] — среднее квадратов интенсивности
        mean_sq_raw = cv2.boxFilter(
            (gray.astype(np.float32) ** 2), cv2.CV_32F, (window_size, window_size)
        )
        mean_sq: FloatArray = mean_sq_raw.astype(np.uint8)
        # σ = sqrt(E[X²] - (E[X])²), с защитой от отрицательных значений из-за численных ошибок
        variance: FloatArray = np.maximum(mean_sq - mean**2, 0)
        std: FloatArray = np.sqrt(variance)

        # mean = cv2.blur(gray, (window_size, window_size))
        # std = np.sqrt(cv2.boxFilter(gray.astype(float)**2, -1, (window_size, window_size)) - mean**2)

        # Расчёт порога Сауволы: T = μ · [1 + k · (σ / R - 1)]
        threshold: FloatArray = mean * (1.0 + k * (std / r - 1.0))

        # Бинаризация: пиксели ярче порога = объект (255)
        mask: MaskArray = (gray.astype(np.float32) > threshold).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time

        # Логирование метаданных выполнения
        self._log_info(
            "sauvola_thresholding_opencv",
            exec_time,
            {"window_size": window_size, "k": k, "r": r, **kwargs},
        )

        # print(f"Mask after OpenCV_thresholding_sauvola: {mask}")
        print(f"Info after OpenCV_thresholding_sauvola: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_threshold_bernsen(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Пороговая обработка по методу Бернсена.

        Локальный адаптивный порог на основе контраста в окне. Метод вычисляет порог
        как среднее между локальным минимумом и максимумом интенсивности, но применяет
        его только если локальный контраст превышает заданный порог.

        Формула порога для пикселя (x, y):
        ```
        Если (max - min) >= contrast_threshold:
            T(x, y) = (min + max) / 2
        Иначе:
            T(x, y) = 0  # пиксель считается фоном
        ```

        где:
        - `min`, `max` — локальные минимум и максимум в окне `window_size × window_size`
        - `contrast_threshold` — минимальный контраст для разделения объекта и фона

        Метод особенно эффективен для:
        - Изображений с чёткими границами между объектом и фоном
        - Документов с высоким локальным контрастом
        - Индустриальных изображений с резкими перепадами яркости

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Паддинг изображения для обработки граничных пикселей.
        3. Вычисление локальных минимумов через эрозию.
        4. Вычисление локальных максимумов через дилатацию.
        5. Расчёт локального контраста: `contrast = max - min`.
        6. Применение порога Бернсена только к пикселям с достаточным контрастом.
        7. Бинаризация и возврат маски.

        Args:
            img: Входное изображение (grayscale предпочтительно).
                Поддерживаются: `(H, W)` или `(H, W, 3)`, dtype=uint8.
            **kwargs: Дополнительные параметры:
                    - `window_size` (int): Размер окна для локального анализа (нечётный, 3–99).
                    По умолчанию 15. Меньшие окна → более детальные границы.
                    - `contrast_threshold` (int): Минимальный контраст [0, 255] для разделения.
                    По умолчанию 25. Меньшие значения → больше пикселей считается объектом.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = объект, 0 = фон.

        Raises:
            ValueError: Если `window_size` чётный (автоматически корректируется).

        Note:
            - Метод Бернсена не требует вычисления среднего, только min/max, что делает его
            быстрее, чем Ниблак/Саувола, но более чувствительным к выбросам.
            - Для зашумлённых изображений рекомендуется предварительное сглаживание
            (Гауссово размытие с `sigma=0.5–1.0`).
            - `contrast_threshold` следует подбирать экспериментально:
            * Высококонтрастные изображения: 15–30
            * Низкоконтрастные: 5–15
            * Очень шумные: 30–50
            - Паддинг с `BORDER_REFLECT_101` обеспечивает корректную обработку граничных пикселей
            без артефактов.

        Example:
            ```python
            # Базовое использование для документа с чётким текстом
            segmenter = OpenCVSegmenter("threshold_bernsen", window_size=15, contrast_threshold=25)
            mask = segmenter.segment(document_image)

            # Для изображения с умеренным контрастом
            segmenter = OpenCVSegmenter("threshold_bernsen", window_size=25, contrast_threshold=15)
            mask = segmenter.segment(low_contrast_image)

            # С предварительным сглаживанием для шумных изображений
            blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0.8)
            segmenter = OpenCVSegmenter("threshold_bernsen", contrast_threshold=30)
            mask = segmenter.segment(blurred)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()  # Копия для безопасности

        start_time: float = time.time()

        # Получение параметров с типизацией
        window_size: int = int(self.params.get("window_size", 15))
        contrast_threshold: int = int(self.params.get("contrast_threshold", 25))

        # Коррекция чётного window_size
        if window_size % 2 == 0:
            window_size += 1
        pad: int = window_size // 2

        # Паддинг изображения для корректной обработки границ
        gray_padded: GrayImage = cv2.copyMakeBorder(
            gray,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_REFLECT_101,  # Эквивалент 'reflect' в PyTorch
        ).astype(np.uint8)

        # Вычисление локальных минимумов через эрозию
        min_filter_raw = cv2.erode(
            gray_padded,
            np.ones((window_size, window_size), dtype=np.uint8),
            borderType=cv2.BORDER_REFLECT_101,
        )
        min_filter: GrayImage = min_filter_raw.astype(np.uint8)
        # Вычисление локальных максимумов через дилатацию
        max_filter_raw = cv2.dilate(
            gray_padded,
            np.ones((window_size, window_size), dtype=np.uint8),
            borderType=cv2.BORDER_REFLECT_101,
        )
        max_filter: GrayImage = max_filter_raw.astype(np.uint8)
        # Удаление паддинга
        min_filter = min_filter[pad:-pad, pad:-pad]
        max_filter = max_filter[pad:-pad, pad:-pad]

        # Расчёт локального контраста
        contrast: GrayImage = max_filter - min_filter

        # Расчёт порога Бернсена: среднее между min и max
        threshold: FloatArray = (
            min_filter.astype(np.float32) + max_filter.astype(np.float32)
        ) / 2.0

        # Инициализация пустой маски
        mask: MaskArray = np.zeros_like(gray, dtype=np.uint8)

        # Бинаризация только для пикселей с достаточным контрастом
        high_contrast: npt.NDArray[np.bool_] = contrast >= contrast_threshold
        mask[high_contrast] = (
            gray[high_contrast].astype(np.float32) > threshold[high_contrast]
        ).astype(np.uint8) * 255

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

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_threshold_phansalkar(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Адаптивная пороговая обработка по методу Фансалкара.

        Улучшенная версия метода Ниблака, оптимизированная для изображений с очень низким
        контрастом, таких как медицинские снимки и микрофотографии. Порог вычисляется
        для каждого пикселя на основе локальных статистик в окне `window_size × window_size`:

        ```
        T(x, y) = μ + k·σ·(σ/R) + m·(μ/R - 1)
        ```

        где:
        - `μ` — локальное среднее интенсивности
        - `σ` — локальное стандартное отклонение
        - `k` — параметр чувствительности к дисперсии [0, 1]
        - `m` — параметр смещения [0, 2]
        - `R` — динамический диапазон (128 для 8-битных, 0.5 для нормализованных [0,1])

        Метод особенно эффективен для:
        - Медицинских изображений с низким контрастом (рентген, МРТ)
        - Микрофотографий клеток и тканей
        - Документов с выцветшим текстом или слабым контрастом

        Алгоритм:
        1. Конвертация входного изображения в grayscale при необходимости.
        2. Определение динамического диапазона `R` на основе диапазона интенсивностей.
        3. Вычисление локального среднего `μ` через `cv2.boxFilter`.
        4. Вычисление локальной дисперсии: `σ² = E[X²] - (E[X])²`.
        5. Расчёт порога Фансалкара по формуле выше.
        6. Бинаризация: `mask = (img > threshold) * 255`.

        Args:
            img: Входное изображение. Поддерживаются форматы:
                - Grayscale: `(H, W)`, dtype=uint8
                - RGB/BGR: `(H, W, 3)`, dtype=uint8 (автоматически конвертируется)
            **kwargs: Дополнительные параметры:
                    - `window_size` (int): Размер окна для локальных статистик (нечётный, 3–99).
                    По умолчанию 15. Меньшие окна → более детальные границы.
                    - `k` (float): Параметр чувствительности к дисперсии [0.0, 1.0].
                    По умолчанию 0.25. Меньшие значения → более строгий порог.
                    - `r` (float): Динамический диапазон [0.5, 255.0].
                    По умолчанию 0.5 для нормализованных, 128.0 для 8-битных изображений.
                    - `m` (float): Параметр смещения [0.0, 2.0].
                    По умолчанию 0.5. Корректирует порог на основе среднего.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, значения {0, 255},
                    где 255 = объект, 0 = фон.

        Raises:
            ValueError: Если `window_size` чётный (автоматически корректируется: `window_size += 1`).

        Note:
            - Метод Фансалкара более устойчив к низкому контрасту, чем Саувола, за счёт
            дополнительного члена `m·(μ/R - 1)`, который корректирует порог на основе среднего.
            - Для нормализованных изображений [0, 1] установите `r=0.5`; для 8-битных — `r=128.0`.
            - Оптимальные параметры для разных типов изображений:
            * Медицинские: `k=0.25, m=0.5, window_size=15–25`
            * Микрофотографии: `k=0.3, m=0.4, window_size=25–45`
            * Документы: `k=0.2, m=0.6, window_size=15–20`
            - Вычисление `σ` через `boxFilter` быстрее, чем через `cv2.blur`, но может давать
            небольшие численные отличия из-за порядка операций.

        Example:
            ```python
            # Базовое использование для медицинского изображения
            segmenter = OpenCVSegmenter("threshold_phansalkar", window_size=15, k=0.25, r=128, m=0.5)
            mask = segmenter.segment(medical_image)

            # Для нормализованного изображения [0, 1]
            segmenter = OpenCVSegmenter("threshold_phansalkar", window_size=25, k=0.3, r=0.5, m=0.4)
            mask = segmenter.segment(normalized_microscopy_image)

            # Переопределение параметров при вызове
            mask = segmenter.segment(image, window_size=35, k=0.2)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        # Определение динамического диапазона на основе диапазона интенсивностей
        gray_range: float = float(gray.max() - gray.min())
        is_normalized: bool = gray_range <= 1.0
        R: float = 0.5 if is_normalized else 128.0

        # Получение параметров с типизацией
        window_size: int = int(self.params.get("window_size", 15))
        k: float = float(self.params.get("k", 0.25))
        r: float = float(self.params.get("r", R))
        m: float = float(self.params.get("m", 0.5))

        # Коррекция чётного window_size
        if window_size % 2 == 0:
            window_size += 1

        # Вычисление локальных статистик
        mean_raw: FloatArray = cv2.boxFilter(
            gray.astype(np.float32), cv2.CV_32F, (window_size, window_size)
        ).astype(np.float32)
        mean: FloatArray = mean_raw.astype(np.float32)
        mean_sq_raw = cv2.boxFilter(
            (gray.astype(np.float32) ** 2), cv2.CV_32F, (window_size, window_size)
        ).astype(np.float32)
        mean_sq: FloatArray = mean_sq_raw.astype(np.float32)
        variance: FloatArray = np.maximum(mean_sq - mean**2, 0)
        std: FloatArray = np.sqrt(variance)

        # Расчёт порога Фансалкара: T = μ + k·σ·(σ/R) + m·(μ/R - 1)
        threshold: FloatArray = mean + k * std * (std / r) + m * (mean / r - 1)

        # Бинаризация
        mask: MaskArray = (gray.astype(np.float32) > threshold).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time

        self._log_info(
            "phansalkar_thresholding_opencv",
            exec_time,
            {"window_size": window_size, "k": k, "r": r, "m": m, **kwargs},
        )

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_threshold_kittler_illingworth(
        self, img: ImageArray, **kwargs: Any
    ) -> MaskArray:
        """
        Пороговая обработка по методу Киттлера-Иллингуорта.

        Статистический метод, минимизирующий ошибку классификации на основе гистограммы
        интенсивностей. Предполагает, что гистограмма изображения является смесью двух
        гауссовых распределений (фон и объект) и находит порог, минимизирующий общую
        ошибку классификации.

        Критерий оптимизации:
        ```
        J(t) = w₀·log(σ₀²) + w₁·log(σ₁²) - 2·[w₀·log(w₀) + w₁·log(w₁)]
        ```
        где:
        - `t` — кандидат на порог
        - `w₀, w₁` — вероятности классов (фон/объект)
        - `σ₀², σ₁²` — дисперсии классов

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Построение гистограммы интенсивностей с `num_bins` бинами.
        3. Нормализация гистограммы к вероятностному распределению.
        4. Вычисление кумулятивных сумм для эффективного расчёта статистик.
        5. Перебор всех возможных порогов и вычисление критерия `J(t)`.
        6. Выбор порога с минимальным значением `J(t)`.
        7. Бинаризация изображения найденным порогом.

        Метод особенно эффективен для:
        - Изображений с бимодальной гистограммой (чёткое разделение фона/объекта)
        - Документов с текстом на однородном фоне
        - Медицинских изображений с двумя доминирующими тканями

        Args:
            img: Входное изображение (grayscale предпочтительно).
                Поддерживаются: `(H, W)` или `(H, W, 3)`, dtype=uint8.
            **kwargs: Дополнительные параметры:
                    - `num_bins` (int): Количество бинов гистограммы [32, 512].
                    По умолчанию 256. Меньшие значения → быстрее, но менее точно.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = объект, 0 = фон.

        Note:
            - Метод предполагает бимодальность гистограммы; для унимодальных или
            мультимодальных распределений результат может быть неоптимальным.
            - Вычислительная сложность: O(num_bins), что делает метод быстрым даже
            для больших значений `num_bins`.
            - Для изображений с очень широким динамическим диапазоном (>256 уровней)
            рассмотрите предварительное квантование или увеличение `num_bins`.
            - Метод не учитывает пространственную информацию; для изображений с
            локальными вариациями освещения рассмотрите адаптивные методы.

        Example:
            ```python
            # Базовое использование для документа
            segmenter = OpenCVSegmenter("threshold_kittler_illingworth", num_bins=256)
            mask = segmenter.segment(document_image)

            # Для изображений с узкой гистограммой
            segmenter = OpenCVSegmenter("threshold_kittler_illingworth", num_bins=128)
            mask = segmenter.segment(low_contrast_image)

            # Для высокоточной сегментации
            segmenter = OpenCVSegmenter("threshold_kittler_illingworth", num_bins=512)
            mask = segmenter.segment(high_precision_image)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        # Получение параметров
        num_bins: int = int(self.params.get("num_bins", 256))

        # Построение гистограммы
        hist: npt.NDArray[np.float64]
        _bin_edges: npt.NDArray[np.float64]
        hist, _bin_edges = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))
        hist = hist.astype(np.float64)

        # Нормализация гистограммы к вероятностному распределению
        hist = hist / hist.sum()

        # Кумулятивные суммы для эффективного расчёта
        cum_hist: npt.NDArray[np.float64] = np.cumsum(hist)
        cum_mean: npt.NDArray[np.float64] = np.cumsum(hist * np.arange(num_bins))

        total_mean: float = cum_mean[-1]
        min_error: float = np.inf
        best_threshold: int = 128

        # Поиск оптимального порога
        for t in range(1, num_bins - 1):
            # Пропуск крайних значений с малой вероятностью
            if cum_hist[t] < 1e-6 or (1 - cum_hist[t]) < 1e-6:
                continue

            # Вероятности классов
            w0: float = cum_hist[t]
            w1: float = 1 - w0

            # Средние значения классов
            mu0: float = cum_mean[t] / w0 if w0 > 0 else 0
            mu1: float = (total_mean - cum_mean[t]) / w1 if w1 > 0 else 0

            # Кумулятивная сумма квадратов для дисперсий
            cum_mean_sq: npt.NDArray[np.float64] = np.cumsum(
                hist * np.arange(num_bins) ** 2
            )

            # Дисперсии классов
            sigma0_sq: float = cum_mean_sq[t] / w0 - mu0**2 if w0 > 0 else 0
            sigma1_sq: float = (
                (cum_mean_sq[-1] - cum_mean_sq[t]) / w1 - mu1**2 if w1 > 0 else 0
            )

            # Пропуск вырожденных случаев
            if sigma0_sq <= 1e-6 or sigma1_sq <= 1e-6:
                continue

            # Критерий Киттлера-Иллингуорта
            error: float = (
                w0 * np.log(sigma0_sq)
                + w1 * np.log(sigma1_sq)
                - 2 * (w0 * np.log(w0) + w1 * np.log(w1))
            )

            if error < min_error:
                min_error = error
                best_threshold = t

        # Бинаризация
        mask: MaskArray
        _, mask_raw = cv2.threshold(
            gray, float(best_threshold), 255.0, cv2.THRESH_BINARY
        )
        mask = mask_raw.astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "kittler_illingworth_thresholding_opencv",
            exec_time,
            {"num_bins": num_bins, "optimal_threshold": best_threshold, **kwargs},
        )

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_threshold_entropy_kapur(
        self, img: ImageArray, **kwargs: Any
    ) -> MaskArray:
        """
        Пороговая обработка на основе максимизации энтропии Капура.

        Статистический метод, находящий порог, который максимизирует сумму энтропий
        фона и объекта. Предполагает, что оптимальный порог разделяет гистограмму
        на две части с максимальной информационной неопределённостью.

        Формула энтропии:
        ```
        H(t) = H₀(t) + H₁(t)
        где:
        - H₀(t) = -Σ_{i=0}^{t} (p_i / P₀) · log(p_i / P₀)  # энтропия фона
        - H₁(t) = -Σ_{i=t+1}^{L-1} (p_i / P₁) · log(p_i / P₁)  # энтропия объекта
        - P₀ = Σ_{i=0}^{t} p_i, P₁ = Σ_{i=t+1}^{L-1} p_i
        ```

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Построение нормализованной гистограммы с `num_bins` бинами.
        3. Вычисление кумулятивной гистограммы и кумулятивной энтропии.
        4. Перебор всех возможных порогов и вычисление суммарной энтропии.
        5. Выбор порога с максимальной суммарной энтропией.
        6. Бинаризация изображения найденным порогом.

        Метод особенно эффективен для:
        - Изображений с чётким разделением фона и объекта в гистограмме
        - Документов с текстом на однородном фоне
        - Медицинских изображений с двумя доминирующими тканями

        Args:
            img: Входное изображение (grayscale предпочтительно).
                Поддерживаются: `(H, W)` или `(H, W, 3)`, dtype=uint8.
            **kwargs: Дополнительные параметры:
                    - `num_bins` (int): Количество бинов гистограммы [32, 512].
                    По умолчанию 256. Меньшие значения → быстрее, но менее точно.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = объект, 0 = фон.

        Note:
            - Метод предполагает, что гистограмма имеет хотя бы два выраженных пика;
            для унимодальных распределений результат может быть неоптимальным.
            - Вычислительная сложность: O(num_bins), что делает метод быстрым.
            - Для изображений с широким динамическим диапазоном (>256 уровней)
            рассмотрите увеличение `num_bins` или предварительное квантование.
            - Метод не учитывает пространственную информацию; для изображений с
            локальными вариациями освещения рассмотрите адаптивные методы.

        Example:
            ```python
            # Базовое использование для документа
            segmenter = OpenCVSegmenter("threshold_entropy_kapur", num_bins=256)
            mask = segmenter.segment(document_image)

            # Для изображений с узкой гистограммой
            segmenter = OpenCVSegmenter("threshold_entropy_kapur", num_bins=128)
            mask = segmenter.segment(low_contrast_image)

            # Для высокоточной сегментации
            segmenter = OpenCVSegmenter("threshold_entropy_kapur", num_bins=512)
            mask = segmenter.segment(high_precision_image)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        # Получение параметров
        num_bins: int = int(self.params.get("num_bins", 256))

        # Построение гистограммы с защитой от log(0)
        hist: npt.NDArray[np.float64]
        _bin_edges: npt.NDArray[np.float64]
        hist, _bin_edges = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))
        hist = hist.astype(np.float64) + 1e-10  # Избегаем log(0)
        hist = hist / hist.sum()  # Нормализация к вероятностному распределению

        # Кумулятивная гистограмма и энтропия
        cum_hist: npt.NDArray[np.float64] = np.cumsum(hist)
        cum_entropy: npt.NDArray[np.float64] = np.cumsum(-hist * np.log(hist))

        max_entropy: float = -np.inf
        best_threshold: int = 128

        # Поиск порога с максимальной суммарной энтропией
        for t in range(1, num_bins - 1):
            # Пропуск крайних значений с малой вероятностью
            if cum_hist[t] < 1e-6 or (1 - cum_hist[t]) < 1e-6:
                continue

            # Энтропия фона: H₀ = H_cum[t] / P₀ + log(P₀)
            h0: float = cum_entropy[t] / cum_hist[t] + np.log(cum_hist[t])
            # Энтропия объекта: H₁ = (H_total - H_cum[t]) / P₁ + log(P₁)
            h1: float = (cum_entropy[-1] - cum_entropy[t]) / (1 - cum_hist[t]) + np.log(
                1 - cum_hist[t]
            )

            total_entropy: float = h0 + h1
            if total_entropy > max_entropy:
                max_entropy = total_entropy
                best_threshold = t

        # Бинаризация
        mask: MaskArray
        _, mask_raw = cv2.threshold(
            gray, float(best_threshold), 255.0, cv2.THRESH_BINARY
        )
        mask = mask_raw.astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "entropy_kapur_thresholding_opencv",
            exec_time,
            {"num_bins": num_bins, "optimal_threshold": best_threshold, **kwargs},
        )

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_threshold_triangle(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Пороговая обработка треугольным методом.

        Геометрический метод для бимодальных гистограмм, находящий порог как точку
        максимального перпендикулярного расстояния от линии, соединяющей пик гистограммы
        и конец диапазона интенсивностей.

        Алгоритм:
        1. Построение гистограммы интенсивностей.
        2. Нахождение пика гистограммы (наиболее частое значение).
        3. Построение линии от пика до конца диапазона (минимальное значение).
        4. Для каждого значения интенсивности вычисление расстояния до этой линии.
        5. Выбор значения с максимальным расстоянием как порога.
        6. Бинаризация изображения найденным порогом.

        Формула расстояния от точки (t, hist[t]) до линии:
        ```
        dist = |A·t + B·hist[t] + C| / sqrt(A² + B²)
        где линия задана уравнением: A·x + B·y + C = 0
        ```

        Метод особенно эффективен для:
        - Изображений с выраженной асимметрией гистограммы
        - Документов с текстом на светлом фоне
        - Медицинских изображений с доминирующей тканью

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                    - `num_bins` (int): Количество бинов гистограммы [32, 512].
                    По умолчанию 256.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = объект, 0 = фон.

        Note:
            - Метод работает лучше всего для гистограмм с одним выраженным пиком
            и длинным "хвостом" (например, текст на странице).
            - Для симметричных бимодальных гистограмм рассмотрите метод Оцу.
            - Вычислительная сложность: O(num_bins), метод очень быстрый.
            - Чувствителен к шуму в гистограмме; для зашумлённых изображений
            рассмотрите предварительное сглаживание гистограммы.

        Example:
            ```python
            # Базовое использование для скана документа
            segmenter = OpenCVSegmenter("threshold_triangle", num_bins=256)
            mask = segmenter.segment(document_image)

            # Для изображений с шумной гистограммой
            segmenter = OpenCVSegmenter("threshold_triangle", num_bins=128)
            mask = segmenter.segment(noisy_image)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        # Получение параметров
        num_bins: int = int(self.params.get("num_bins", 256))

        # Построение гистограммы
        hist: npt.NDArray[np.float64]
        _bin_edges: npt.NDArray[np.float64]
        hist, _bin_edges = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))

        # Нахождение пика гистограммы
        peak_idx: int = int(np.argmax(hist))

        # Координаты конца линии (конец диапазона)
        y_peak: float = float(hist[peak_idx])
        y_end: float = float(hist[-1])

        # Уравнение линии: вычисление углового коэффициента
        denominator: float = float(num_bins - 1 - peak_idx)
        m: float = (y_end - y_peak) / denominator if denominator != 0 else 0.0

        # Поиск точки максимального расстояния
        max_dist: float = 0.0
        best_threshold: int = peak_idx

        for t in range(int(peak_idx) + 1, int(num_bins)):
            # Значение линии в точке t
            y_line: float = y_peak + m * (t - peak_idx)
            # Перпендикулярное расстояние от точки до линии
            dist: float = abs(hist[t] - y_line) / np.sqrt(1 + m**2)

            if dist > max_dist:
                max_dist = float(dist)
                best_threshold = int(t)

        # Бинаризация
        mask: MaskArray
        _, mask_raw = cv2.threshold(
            gray, float(best_threshold), 255.0, cv2.THRESH_BINARY
        )
        mask = mask_raw.astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "triangle_thresholding_opencv",
            exec_time,
            {"num_bins": num_bins, "optimal_threshold": best_threshold, **kwargs},
        )

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_threshold_multi_otsu(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Многоуровневая пороговая обработка по методу Оцу.

        Расширение классического метода Оцу для разделения изображения на несколько
        классов (не только фон/объект). Находит пороги, максимизирующие межклассовую
        дисперсию между всеми классами.

        Для двух порогов (три класса) критерий оптимизации:
        ```
        σ_B² = w₀·(μ₀ - μ_T)² + w₁·(μ₁ - μ_T)² + w₂·(μ₂ - μ_T)²
        ```
        где:
        - `wᵢ` — вероятность класса i
        - `μᵢ` — среднее значение класса i
        - `μ_T` — общее среднее значение

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Построение гистограммы интенсивностей.
        3. Для `n_thresholds == 1`: применение классического Оцу.
        4. Для `n_thresholds == 2`: полный перебор пар порогов с вычислением σ_B².
        5. Для `n_thresholds > 2`: рекурсивное применение Оцу (упрощённо).
        6. Бинаризация: объект = самый яркий класс (для совместимости с бинарной сегментацией).

        Метод особенно эффективен для:
        - Изображений с несколькими однородными областями разной яркости
        - Медицинских изображений с несколькими типами тканей
        - Спутниковых снимков с различными типами земной поверхности

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                    - `n_thresholds` (int): Количество порогов [1, 5].
                    По умолчанию 2 (разделение на 3 класса).
                    - `num_bins` (int): Количество бинов гистограммы [32, 512].
                    По умолчанию 256.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = самый яркий класс (предполагаемый объект), 0 = остальные классы.

        Note:
            - Для `n_thresholds > 2` используется упрощённый рекурсивный подход,
            который может быть менее точным, чем полный перебор.
            - Вычислительная сложность для двух порогов: O(num_bins²), что может
            быть медленно для больших `num_bins`.
            - Метод возвращает бинарную маску, выделяя самый яркий класс как объект;
            для многоклассовой сегментации рассмотрите возвращать метки классов.
            - Для изображений с плавными переходами между классами рассмотрите
            методы кластеризации (K-Means, MeanShift).

        Example:
            ```python
            # Разделение на 3 класса (2 порога)
            segmenter = OpenCVSegmenter("threshold_multi_otsu", n_thresholds=2)
            mask = segmenter.segment(medical_image)

            # Разделение на 4 класса (3 порога)
            segmenter = OpenCVSegmenter("threshold_multi_otsu", n_thresholds=3)
            mask = segmenter.segment(satellite_image)

            # Классический Оцу (1 порог)
            segmenter = OpenCVSegmenter("threshold_multi_otsu", n_thresholds=1)
            mask = segmenter.segment(document_image)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        # Получение параметров
        n_thresholds: int = int(self.params.get("n_thresholds", 2))
        num_bins: int = int(self.params.get("num_bins", 256))

        # Построение гистограммы
        hist: npt.NDArray[np.float64]
        _bin_edges: npt.NDArray[np.float64]
        hist, _bin_edges = np.histogram(gray.ravel(), bins=num_bins, range=(0.0, 256.0))
        hist = hist.astype(np.float64)

        # Случай 1: классический Оцу (один порог)
        if n_thresholds == 1:
            mask: MaskArray
            _, mask_raw = cv2.threshold(
                gray, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            mask = mask_raw.astype(np.uint8)
            return mask

        # Случай 2: два порога (три класса) — полный перебор
        if n_thresholds == 2:
            best_var: float = -np.inf
            best_t1: int = 64
            best_t2: int = 192

            # Кумулятивные суммы для эффективного расчёта
            cum_sum: npt.NDArray[np.float64] = np.cumsum(hist)
            cum_mean: npt.NDArray[np.float64] = np.cumsum(hist * np.arange(num_bins))
            total: float = cum_sum[-1]
            total_mean: float = cum_mean[-1] / total if total > 0 else 0

            # Перебор всех пар порогов
            for t1 in range(1, num_bins - 2):
                for t2 in range(t1 + 1, num_bins - 1):
                    # Класс 0: [0, t1)
                    w0: float = cum_sum[t1] / total if total > 0 else 0
                    m0: float = cum_mean[t1] / cum_sum[t1] if cum_sum[t1] > 0 else 0

                    # Класс 1: [t1, t2)
                    w1: float = (cum_sum[t2] - cum_sum[t1]) / total if total > 0 else 0
                    m1: float = (
                        (cum_mean[t2] - cum_mean[t1]) / (cum_sum[t2] - cum_sum[t1])
                        if (cum_sum[t2] > cum_sum[t1])
                        else 0
                    )

                    # Класс 2: [t2, 256)
                    w2: float = (total - cum_sum[t2]) / total if total > 0 else 0
                    m2: float = (
                        (total_mean * total - cum_mean[t2]) / (total - cum_sum[t2])
                        if (total > cum_sum[t2])
                        else 0
                    )

                    # Межклассовая дисперсия
                    var_between: float = (
                        w0 * (m0 - total_mean) ** 2
                        + w1 * (m1 - total_mean) ** 2
                        + w2 * (m2 - total_mean) ** 2
                    )

                    if var_between > best_var:
                        best_var = var_between
                        best_t1, best_t2 = t1, t2

            # Бинаризация: самый яркий класс = объект
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

        # Случай 3: больше двух порогов — рекурсивный Оцу (упрощённо)
        thresholds: List[float] = []
        current_gray: GrayImage = gray.copy()

        for _ in range(n_thresholds):
            _, thresh = cv2.threshold(
                current_gray, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            thresholds.append(float(thresh))
            current_gray = cv2.threshold(
                current_gray, float(thresh), 255.0, cv2.THRESH_BINARY_INV
            )[1].astype(np.uint8)

        # Бинаризация по последнему порогу
        _, mask_raw = cv2.threshold(
            gray, float(thresholds[-1]), 255.0, cv2.THRESH_BINARY
        )
        mask = mask_raw.astype(np.uint8)

        exec_time = time.time() - start_time
        self._log_info(
            "multi_otsu_thresholding_opencv",
            exec_time,
            {"n_thresholds": n_thresholds, "thresholds": thresholds, **kwargs},
        )

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_threshold_percentile(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Процентильная пороговая обработка.

        Порог выбирается как заданный процентиль распределения интенсивностей гистограммы.
        Метод автоматически адаптируется к контрасту изображения: для тёмных сцен порог
        будет ниже, для светлых — выше.

        Алгоритм:
        1. Конвертация входного изображения в grayscale при необходимости.
        2. Вычисление заданного процентиля распределения интенсивностей через `np.percentile`.
        3. Применение глобального порога через `cv2.threshold` с режимом `THRESH_BINARY`.
        4. Возврат бинарной маски, где пиксели выше порога = объект (255).

        Метод особенно эффективен для:
        - Изображений с известным распределением яркости объекта/фона
        - Задач, где нужно выделить верхний/нижний процент самых ярких/тёмных пикселей
        - Предварительной оценки порога перед более сложными методами

        Args:
            img: Входное изображение. Поддерживаются форматы:
                - Grayscale: `(H, W)`, dtype=uint8
                - RGB/BGR: `(H, W, 3)`, dtype=uint8 (автоматически конвертируется)
            **kwargs: Дополнительные параметры:
                - `percentile` (float): Процентиль для порога [0, 100].
                По умолчанию 90. Меньшие значения → более строгий порог (меньше объекта).

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                где 255 = пиксели с интенсивностью выше процентиля (объект).

        Note:
            - Метод не учитывает пространственную информацию; для изображений с
            локальными вариациями освещения рассмотрите адаптивные пороги (Sauvola, Phansalkar).
            - Процентиль 90 означает, что порог отсекает 10% самых ярких пикселей.
            - Для выделения тёмных объектов на светлом фоне используйте `percentile < 50`
            или инвертируйте результат маски.
            - Вычисление процентиля через `np.percentile` работает с плавающей точкой,
            что обеспечивает точность даже для узких гистограмм.

        Example:
            ```python
            # Выделение самых ярких 10% пикселей (например, блики, источники света)
            segmenter = OpenCVSegmenter("threshold_percentile", percentile=90)
            mask = segmenter.segment(image)

            # Выделение тёмных объектов (нижние 20% интенсивности)
            segmenter = OpenCVSegmenter("threshold_percentile", percentile=20)
            mask = segmenter.segment(dark_objects_image)

            # Переопределение параметра при вызове
            mask = segmenter.segment(image, percentile=95)  # Более строгий порог
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        percentile: float = float(self.params.get("percentile", 90))

        # Вычисляем процентиль
        threshold: float = float(np.percentile(gray.astype(np.float32), percentile))

        # Бинаризация
        mask: MaskArray
        _, mask_raw = cv2.threshold(gray, threshold, 255.0, cv2.THRESH_BINARY)
        mask = mask_raw.astype(np.uint8)

        exec_time: float = time.time() - start_time
        self._log_info(
            "percentile_thresholding_opencv",
            exec_time,
            {"percentile": percentile, "threshold": threshold, **kwargs},
        )

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_threshold_local_contrast(
        self, img: ImageArray, **kwargs: Any
    ) -> MaskArray:
        """
        Пороговая обработка на основе локального контраста.

        Пиксель считается объектом, если его интенсивность значительно отличается
        от локального среднего в окне `window_size × window_size`. Метод адаптируется
        к локальным вариациям освещения и эффективен для текстурных изображений.

        Формула порога для пикселя (x, y):
        ```
        T = μ + k·(max - min) в окне,
        где:
        - μ — локальное среднее интенсивности
        - (max - min) — локальный диапазон контраста
        - k — коэффициент чувствительности (contrast_factor)
        ```

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Вычисление локального среднего через `cv2.boxFilter`.
        3. Расчёт локального контраста как абсолютной разницы от среднего.
        4. Определение глобального порога контраста через процентиль `100·(1 - contrast_factor)`.
        5. Бинаризация: пиксели с контрастом выше порога = объект.

        Метод особенно эффективен для:
        - Текстурных изображений с локальными перепадами яркости
        - Медицинских снимков с неоднородным освещением
        - Задач, где объект отличается от фона по локальной изменчивости, а не по абсолютной яркости

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                - `window_size` (int): Размер окна для локального анализа (нечётный, 3–99).
                По умолчанию 15. Меньшие окна → более детальные границы.
                - `contrast_factor` (float): Коэффициент контраста [0.0, 1.0].
                По умолчанию 0.1. Меньшие значения → более строгий порог (меньше объекта).

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                где 255 = пиксели с локальным контрастом выше порога.

        Note:
            - `window_size` должен быть нечётным; чётные значения автоматически
            корректируются (`window_size += 1`).
            - Метод не требует знания абсолютной яркости объекта, только его
            отличие от локального окружения.
            - Для зашумлённых изображений рекомендуется предварительное сглаживание
            (Гауссово размытие с `sigma=0.5–1.0`).
            - Оптимальный `contrast_factor` зависит от задачи:
            * Высококонтрастные объекты: 0.05–0.15
            * Низкоконтрастные: 0.15–0.3
            * Очень шумные: 0.3–0.5

        Example:
            ```python
            # Базовое использование для текстурного изображения
            segmenter = OpenCVSegmenter("threshold_local_contrast", window_size=15, contrast_factor=0.1)
            mask = segmenter.segment(texture_image)

            # Для изображений с умеренным контрастом
            segmenter = OpenCVSegmenter("threshold_local_contrast", window_size=25, contrast_factor=0.2)
            mask = segmenter.segment(low_contrast_image)

            # С предварительным сглаживанием для шумных изображений
            blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0.8)
            segmenter = OpenCVSegmenter("threshold_local_contrast", contrast_factor=0.15)
            mask = segmenter.segment(blurred)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        window_size: int = int(self.params.get("window_size", 15))
        contrast_factor: float = float(self.params.get("contrast_factor", 0.1))

        if window_size % 2 == 0:
            window_size += 1

        # Локальное среднее
        local_mean: FloatArray = cv2.boxFilter(
            gray.astype(np.float32), cv2.CV_32F, (window_size, window_size)
        ).astype(np.float32)

        # Локальный контраст (разница от среднего)
        local_contrast: FloatArray = np.abs(gray.astype(np.float32) - local_mean)

        # Глобальный порог контраста
        global_contrast_threshold: float = float(
            np.percentile(local_contrast, 100 * (1 - contrast_factor))
        )

        # Бинаризация по контрасту
        mask: MaskArray = (local_contrast > global_contrast_threshold).astype(
            np.uint8
        ) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "local_contrast_thresholding_opencv",
            exec_time,
            {"window_size": window_size, "contrast_factor": contrast_factor, **kwargs},
        )

        return mask

    # ============ МЕТОДЫ НА ОСНОВЕ КРАЕВ ============
    # ──────────────────────────────────────────────────────────────────────
    def _opencv_sobel_edge(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Обнаружение границ оператором Собеля.

        Вычисляет аппроксимацию градиента интенсивности изображения через свёртку с ядрами:
        ```
        G_x = img * [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]

        G_y = img * [[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]]
        ```
        Затем вычисляется магнитуда градиента: `|G| = sqrt(G_x² + G_y²)`,
        которая пороговается для получения бинарной маски границ.

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Вычисление градиентов по осям X и Y через `cv2.Sobel` с `ksize=3`.
        3. Нормализация магнитуды к диапазону [0, 255].
        4. Пороговая бинаризация магнитуды.

        Args:
            img: Входное изображение (см. `_opencv_threshold_niblack`).
            **kwargs: Дополнительные параметры:
                    - `threshold` (float): Порог для бинаризации магнитуды [0, 255].
                    По умолчанию 50. Меньшие значения → больше границ, но больше шума.

        Returns:
            MaskArray: Бинарная маска границ формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Оператор Собеля чувствителен к шуму; для зашумлённых изображений
            предварительно примените Гауссово размытие (`sigma=1.0`).
            - Для более точного вычисления градиента используйте оператор Шара (`_opencv_scharr_edge`).
            - Возвращаемая маска содержит только контуры, не заполненные области.
            - Для получения связных объектов после детекции границ примените морфологическое
            замыкание (`cv2.morphologyEx` с `MORPH_CLOSE`).

        Example:
            ```python
            # Базовое обнаружение границ
            segmenter = OpenCVSegmenter("sobel_edge", threshold=50)
            edges = segmenter.segment(image)

            # С предварительным сглаживанием для шумных изображений
            blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
            segmenter = OpenCVSegmenter("sobel_edge", threshold=30)
            edges = segmenter.segment(blurred)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        # print(f"Gray after OpenCV_sobel_edge: {gray}")

        start_time: float = time.time()

        threshold: float = float(self.params.get("threshold", 50))

        # Вычисление градиентов Собеля с глубиной CV_64F для точности
        sobel_x: npt.NDArray[np.float64] = cv2.Sobel(
            gray, cv2.CV_64F, dx=1, dy=0, ksize=3
        ).astype(np.float64)
        sobel_y: npt.NDArray[np.float64] = cv2.Sobel(
            gray, cv2.CV_64F, dx=0, dy=1, ksize=3
        ).astype(np.float64)

        # Магнитуда градиента
        magnitude: npt.NDArray[np.float64] = np.sqrt(
            sobel_x**2 + sobel_y**2
        ).astype(np.float64)

        # Нормализация к [0, 255] для визуализации
        magnitude_norm: FloatArray = (
            255 * magnitude / (np.max(magnitude) + 1e-8)
        ).astype(np.float32)

        # Или
        # magnitude = cv2.magnitude(sobelx, sobely)
        # sobel_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # _, mask = cv2.threshold(sobel_norm.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
        mask: MaskArray
        _, mask_raw = cv2.threshold(
            magnitude_norm.astype(np.float32), threshold, 255.0, cv2.THRESH_BINARY
        )  # type: ignore[call-overload]
        mask = mask_raw.astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "sobel_edge_opencv",
            exec_time,
            {"threshold": threshold, **kwargs},
        )

        # print(f"Mask after OpenCV_sobel_edge: {mask}")
        print(f"Info after OpenCV_sobel_edge: {self._log_info}")
        return mask.astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_canny_edge(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Обнаружение границ оператором Кэнни.

        Многоэтапный алгоритм обнаружения границ:
        1. Сглаживание Гауссом (подавление шума).
        2. Вычисление градиента (Sobel).
        3. Подавление немаксимумов (тонкие границы).
        4. Двойная пороговая фильтрация (сильные/слабые границы).
        5. Отслеживание связности (гистерезис).

        Оптимальный по критерию: хорошая локализация, низкий уровень ложных срабатываний.

        Args:
            img: Входное изображение (grayscale).
            **kwargs: Дополнительные параметры:
                     - `low` (float): Нижний порог гистерезиса [0, 255]. По умолчанию 50.
                     - `high` (float): Верхний порог гистерезиса [0, 255]. По умолчанию 150.
                     - `sigma` (float): Сигма Гауссова размытия. По умолчанию 1.0.

        Returns:
            MaskArray: Бинарная маска границ (0/255).

        Note:
            - Рекомендуется `high ≈ 2–3 × low`.
            - Большие значения порогов → меньше границ, но выше надёжность.
            - Метод возвращает только контуры, не заполненные области.

        Example:
            ```python
            edges = segmenter._opencv_canny_edge(gray, low=100, high=200)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img
        print(f"Gray after OpenCV_canny_edge: {gray}")
        start_time: float = time.time()

        low: int = int(self.params.get("low", 50))
        high: int = int(self.params.get("high", 150))

        mask: MaskArray = cv2.Canny(gray, low, high).astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "canny_edge_opencv",
            exec_time,
            {"low": low, "high": high, **kwargs},
        )
        print(f"Mask after OpenCV_canny_edge: {mask}")
        print(f"Info after OpenCV_canny_edge: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_prewitt_edge(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Обнаружение границ оператором Превитта.

        Вычисляет аппроксимацию градиента интенсивности через свёртку с ядрами 3×3:
        ```
        G_x = img * [[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]]

        G_y = img * [[-1, -1, -1],
                    [ 0,  0,  0],
                    [ 1,  1,  1]]
        ```
        Затем вычисляется магнитуда градиента: `|G| = sqrt(G_x² + G_y²)`,
        которая пороговается для получения бинарной маски границ.

        По сравнению с оператором Собеля:
        - Превитт использует равные веса [1,1,1], что делает его менее чувствительным к шуму
        - Собель использует веса [1,2,1], что даёт лучшую точность, но больше шума
        - Превитт быстрее вычисляется за счёт более простых ядер

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Вычисление градиентов по осям X и Y через `cv2.filter2D` с ядрами Превитта.
        3. Расчёт магнитуды градиента.
        4. Нормализация магнитуды к диапазону [0, 255].
        5. Пороговая бинаризация.

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                    - `threshold` (float): Порог для бинаризации магнитуды [0, 255].
                    По умолчанию 50. Меньшие значения → больше границ, но больше шума.
                    - `direction` (str): Направление градиента: 'x', 'y' или 'both'.
                    По умолчанию 'both'. Позволяет детектировать только горизонтальные
                    или вертикальные границы.

        Returns:
            MaskArray: Бинарная маска границ формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Оператор Превитта менее точен, чем Собель, но более устойчив к шуму.
            - Для зашумлённых изображений рекомендуется предварительное сглаживание.
            - Возвращаемая маска содержит только контуры, не заполненные области.
            - Для получения связных объектов после детекции границ примените
            морфологическое замыкание (`cv2.morphologyEx` с `MORPH_CLOSE`).
            - Параметр `direction` полезен для детекции границ определённой ориентации:
            * 'x' — вертикальные границы (изменения по горизонтали)
            * 'y' — горизонтальные границы (изменения по вертикали)
            * 'both' — все границы (по умолчанию)

        Example:
            ```python
            # Базовое обнаружение всех границ
            segmenter = OpenCVSegmenter("prewitt_edge", threshold=50)
            edges = segmenter.segment(image)

            # Только вертикальные границы
            segmenter = OpenCVSegmenter("prewitt_edge", threshold=40, direction='x')
            vertical_edges = segmenter.segment(image)

            # С предварительным сглаживанием для шумных изображений
            blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0.5)
            segmenter = OpenCVSegmenter("prewitt_edge", threshold=30)
            edges = segmenter.segment(blurred)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        # Получение параметров
        threshold: float = float(self.params.get("threshold", 50))
        direction: str = str(self.params.get("direction", "both"))

        # Ядра Превитта 3×3
        kernel_x: npt.NDArray[np.float32] = np.array(
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32
        )
        kernel_y: npt.NDArray[np.float32] = np.array(
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32
        )

        # Вычисление градиентов в зависимости от направления
        if direction in ["x", "both"]:
            grad_x_raw = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
            grad_x: FloatArray = grad_x_raw.astype(np.float32)
        else:
            grad_x = np.zeros_like(gray, dtype=np.float32)

        if direction in ["y", "both"]:
            grad_y: FloatArray = cv2.filter2D(gray, cv2.CV_32F, kernel_y).astype(
                np.float32
            )
        else:
            grad_y = np.zeros_like(gray, dtype=np.float32)

        # Магнитуда градиента
        magnitude: FloatArray = np.sqrt(grad_x**2 + grad_y**2)

        # Нормализация к [0, 255] для визуализации и бинаризации
        magnitude_norm: FloatArray = (
            255 * magnitude / (np.max(magnitude) + 1e-8)
        ).astype(np.float32)

        # Пороговая бинаризация
        mask: MaskArray
        _, mask_raw = cv2.threshold(
            magnitude_norm.astype(np.float32), threshold, 255.0, cv2.THRESH_BINARY
        )
        mask = mask_raw.astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "prewitt_edge_opencv",
            exec_time,
            {"threshold": threshold, "direction": direction, **kwargs},
        )

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_scharr_edge(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Обнаружение границ оператором Шара.

        Улучшенная версия оператора Собеля с оптимизированными ядрами 3×3, обеспечивающими
        лучшую точность вычисления градиента при сохранении вычислительной эффективности.
        Ядра Шара минимизируют ошибку аппроксимации производной первого порядка.

        Ядра Шара:
        ```
        G_x = img * [[-3, 0, 3],
                    [-10, 0, 10],
                    [-3, 0, 3]] / 16

        G_y = img * [[-3, -10, -3],
                    [ 0,   0,  0],
                    [ 3,  10,  3]] / 16
        ```

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Вычисление градиентов по осям X и Y через `cv2.Scharr`.
        3. Расчёт магнитуды градиента: `|G| = sqrt(G_x² + G_y²)`.
        4. Нормализация магнитуды к диапазону [0, 255].
        5. Пороговая бинаризация.

        Метод особенно эффективен для:
        - Изображений с чёткими, но тонкими границами
        - Задач, требующих высокой точности локализации границ
        - Предварительной обработки для активных контуров и watershed

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                    - `threshold` (float): Порог для бинаризации магнитуды [0, 255].
                    По умолчанию 50. Меньшие значения → больше границ, но больше шума.
                    - `direction` (str): Направление градиента: 'x', 'y' или 'both'.
                    По умолчанию 'both'. Позволяет детектировать только горизонтальные
                    или вертикальные границы.

        Returns:
            MaskArray: Бинарная маска границ формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Оператор Шара более точен, чем Собель, за счёт оптимизированных весов,
            но вычислительно сопоставим.
            - Для зашумлённых изображений рекомендуется предварительное сглаживание
            (Гауссово размытие с `sigma=0.5–1.0`).
            - Возвращаемая маска содержит только контуры, не заполненные области.
            - Для получения связных объектов после детекции границ примените
            морфологическое замыкание (`cv2.morphologyEx` с `MORPH_CLOSE`).
            - Параметр `direction` полезен для детекции границ определённой ориентации:
            * 'x' — вертикальные границы (изменения по горизонтали)
            * 'y' — горизонтальные границы (изменения по вертикали)
            * 'both' — все границы (по умолчанию)

        Example:
            ```python
            # Базовое обнаружение всех границ
            segmenter = OpenCVSegmenter("scharr_edge", threshold=50)
            edges = segmenter.segment(image)

            # Только вертикальные границы
            segmenter = OpenCVSegmenter("scharr_edge", threshold=40, direction='x')
            vertical_edges = segmenter.segment(image)

            # С предварительным сглаживанием для шумных изображений
            blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0.5)
            segmenter = OpenCVSegmenter("scharr_edge", threshold=30)
            edges = segmenter.segment(blurred)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        # Получение параметров
        threshold: float = float(self.params.get("threshold", 50))
        direction: str = str(self.params.get("direction", "both"))

        # Вычисление градиентов Шара в зависимости от направления
        grad_x: npt.NDArray[np.float64]
        grad_y: npt.NDArray[np.float64]

        if direction in ["x", "both"]:
            grad_x = cv2.Scharr(gray, cv2.CV_64F, dx=1, dy=0).astype(np.float64)
        else:
            grad_x = np.zeros_like(gray, dtype=np.float64)

        if direction in ["y", "both"]:
            grad_y = cv2.Scharr(gray, cv2.CV_64F, dx=0, dy=1).astype(np.float64)
        else:
            grad_y = np.zeros_like(gray, dtype=np.float64)

        # Магнитуда градиента
        magnitude: npt.NDArray[np.float64] = np.sqrt(grad_x**2 + grad_y**2)
        # Нормализация к [0, 255] для визуализации и бинаризации
        magnitude_norm: FloatArray = (
            255 * magnitude / (np.max(magnitude) + 1e-8)
        ).astype(np.float32)

        # Пороговая бинаризация
        mask: MaskArray
        _, mask_raw = cv2.threshold(
            magnitude_norm.astype(np.float32), threshold, 255.0, cv2.THRESH_BINARY
        )
        mask = mask_raw.astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "scharr_edge_opencv",
            exec_time,
            {"threshold": threshold, "direction": direction, **kwargs},
        )

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_roberts_cross_edge(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Обнаружение границ оператором Робертса (Cross).

        Простой и быстрый оператор для обнаружения диагональных границ через
        вычисление разности интенсивностей по диагоналям 2×2:

        ```
        G = sqrt( [I(x+1,y+1) - I(x,y)]² + [I(x+1,y) - I(x,y+1)]² )
        ```

        Ядра Робертса:
        ```
        [[+1,  0],    [[ 0, +1],
        [ 0, -1]]     [-1,  0]]
        ```

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Свёртка с ядрами Робертса для вычисления диагональных градиентов.
        3. Расчёт магнитуды градиента.
        4. Нормализация к [0, 255] и пороговая бинаризация.

        Метод особенно эффективен для:
        - Изображений с чёткими диагональными границами
        - Задач, где важна скорость вычислений (ядро 2×2)
        - Предварительной обработки для детекции углов

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                    - `threshold` (float): Порог для бинаризации магнитуды [0, 255].
                    По умолчанию 50. Меньшие значения → больше границ, но больше шума.

        Returns:
            MaskArray: Бинарная маска границ формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Оператор Робертса очень чувствителен к шуму из-за малого размера ядра;
            для зашумлённых изображений рекомендуется предварительное сглаживание.
            - Метод обнаруживает только диагональные границы; для всех направлений
            рассмотрите операторы Собеля или Шара.
            - Возвращаемая маска содержит только контуры, не заполненные области.
            - Из-за ядра 2×2 результат может быть смещён на 1 пиксель относительно
            истинных границ; для точной локализации рассмотрите субпиксельные методы.

        Example:
            ```python
            # Базовое обнаружение диагональных границ
            segmenter = OpenCVSegmenter("roberts_cross_edge", threshold=50)
            edges = segmenter.segment(image)

            # С предварительным сглаживанием для шумных изображений
            blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0.5)
            segmenter = OpenCVSegmenter("roberts_cross_edge", threshold=30)
            edges = segmenter.segment(blurred)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        # Получение параметров
        threshold: float = float(self.params.get("threshold", 50))

        # Ядра Робертса 2×2
        kernel_x: npt.NDArray[np.float32] = np.array(
            [[1, 0], [0, -1]], dtype=np.float32
        )
        kernel_y: npt.NDArray[np.float32] = np.array(
            [[0, 1], [-1, 0]], dtype=np.float32
        )

        # Вычисление диагональных градиентов
        grad_x: npt.NDArray[np.float32] = cv2.filter2D(
            gray, cv2.CV_32F, kernel_x, borderType=cv2.BORDER_REFLECT
        )
        grad_y: npt.NDArray[np.float32] = cv2.filter2D(
            gray, cv2.CV_32F, kernel_y, borderType=cv2.BORDER_REFLECT
        )

        # Магнитуда градиента
        magnitude: FloatArray = np.sqrt(grad_x**2 + grad_y**2)

        # Нормализация к [0, 255]
        magnitude_norm: npt.NDArray[np.float32] = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.float32)

        # Пороговая бинаризация
        _, mask_raw = cv2.threshold(
            magnitude_norm.astype(np.float32), threshold, 255.0, cv2.THRESH_BINARY
        )
        mask: MaskArray = mask_raw.astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "roberts_cross_edge_opencv",
            exec_time,
            {"threshold": threshold, **kwargs},
        )

        return mask.astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_log_edge(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Обнаружение границ Лапласианом Гауссиана (LoG / Laplacian of Gaussian).

        Комбинация Гауссова размытия (подавление шума) и оператора Лапласа
        (детекция границ через вторые производные). Границы обнаруживаются
        по нулевым пересечениям (zero-crossings) лапласиана.

        Формула:
        ```
        LoG(x, y) = ∇²[G(x, y, σ) * I(x, y)]
        где:
        - ∇² — оператор Лапласа (вторые производные)
        - G — Гауссово ядро с параметром σ
        - * — операция свёртки
        ```

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Гауссово размытие изображения с параметром `sigma`.
        3. Применение оператора Лапласа к размытому изображению.
        4. Векторизованное обнаружение нулевых пересечений (соседние пиксели с противоположными знаками).
        5. Фильтрация слабых пересечений по порогу `threshold`.
        6. Возврат бинарной маски границ.

        Метод особенно эффективен для:
        - Изображений с плавными переходами интенсивности
        - Медицинских изображений с размытыми границами тканей
        - Задач, где важна инвариантность к масштабу (через параметр `sigma`)

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                    - `sigma` (float): Стандартное отклонение Гаусса [0.1, 10.0].
                    По умолчанию 1.0. Больше значения → детекция более крупных границ.
                    - `kernel_size` (int): Размер ядра Лапласиана (нечётный, 3–15).
                    По умолчанию 5. Влияет на точность вычисления вторых производных.
                    - `threshold` (int): Порог для отсечения слабых нулевых пересечений [0, 100].
                    По умолчанию 10. Меньшие значения → больше границ, но больше шума.

        Returns:
            MaskArray: Бинарная маска границ формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Параметр `sigma` контролирует масштаб детектируемых границ:
            * Малые σ (0.5–1.0) → мелкие детали, но больше шума
            * Большие σ (2.0–5.0) → крупные границы, но потеря мелких деталей
            - Нулевые пересечения могут быть разрывными; для получения связных
            границ рассмотрите морфологическое замыкание после детекции.
            - Метод чувствителен к шуму; Гауссово размытие частично подавляет шум,
            но для зашумлённых изображений может потребоваться дополнительная
            предобработка (медианный фильтр, bilateral filter).
            - Векторизованная реализация zero-crossing быстрее циклической,
            но может давать небольшие отличия на границах изображения.

        Example:
            ```python
            # Базовое обнаружение границ
            segmenter = OpenCVSegmenter("log_edge", sigma=1.0, threshold=10)
            edges = segmenter.segment(image)

            # Для детекции крупных границ
            segmenter = OpenCVSegmenter("log_edge", sigma=3.0, threshold=15)
            edges = segmenter.segment(large_objects_image)

            # Для зашумлённых изображений
            segmenter = OpenCVSegmenter("log_edge", sigma=2.0, threshold=20)
            edges = segmenter.segment(noisy_image)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        # Получение параметров
        sigma: float = float(self.params.get("sigma", 1.0))
        kernel_size: int = int(self.params.get("kernel_size", 5))
        threshold: int = int(self.params.get("threshold", 10))

        # Коррекция чётного kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Гауссово размытие
        blurred: FloatArray = gaussian_filter(gray.astype(np.float32), sigma=sigma)

        # Лапласиан
        laplacian: npt.NDArray[np.float64] = laplace(blurred)

        # Векторизованное zero-crossing detection
        magnitude: npt.NDArray[np.float64] = np.abs(laplacian)
        sign: npt.NDArray[np.float64] = np.sign(laplacian)

        # Горизонтальные и вертикальные пересечения
        zc_h: npt.NDArray[np.bool_] = sign[:, :-1] * sign[:, 1:] < 0
        zc_v: npt.NDArray[np.bool_] = sign[:-1, :] * sign[1:, :] < 0

        # Объединение пересечений
        zero_crossing_bool: npt.NDArray[np.bool_] = np.zeros_like(laplacian, dtype=bool)
        zero_crossing_bool[:, :-1] |= zc_h
        zero_crossing_bool[:-1, :] |= zc_v

        # Фильтрация по амплитуде (отсечение слабых пересечений)
        zero_crossing: MaskArray = (
            zero_crossing_bool & (magnitude > threshold)
        ).astype(np.uint8) * 255

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

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_dog_edge(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Обнаружение границ разностью Гауссианов (DoG / Difference of Gaussians).

        Аппроксимация оператора Лапласиана Гауссиана (LoG) через разность двух
        Гауссовых фильтров с разными стандартными отклонениями. Эффективен для
        обнаружения границ разного масштаба и подавления шума.

        Формула:
        ```
        DoG(x, y) = G(x, y, σ₁) * I - G(x, y, σ₂) * I, где σ₂ > σ₁
        ```

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Применение двух Гауссовых фильтров с параметрами `sigma1` и `sigma2`.
        3. Вычисление разности: `dog = g1 - g2`.
        4. Векторизованное обнаружение нулевых пересечений (соседние пиксели с противоположными знаками).
        5. Фильтрация слабых пересечений по порогу `threshold`.
        6. Возврат бинарной маски границ.

        Метод особенно эффективен для:
        - Изображений с границами разного масштаба (мелкие детали + крупные объекты)
        - Медицинских изображений с размытыми переходами тканей
        - Задач, где важна инвариантность к масштабу (через подбор σ₁, σ₂)

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                - `sigma1` (float): Стандартное отклонение первого Гаусса [0.1, 5.0].
                По умолчанию 1.0. Меньшие значения → детекция мелких границ.
                - `sigma2` (float): Стандартное отклонение второго Гаусса [0.5, 10.0].
                По умолчанию 2.0. Должно быть > sigma1.
                - `threshold` (int): Порог для отсечения слабых нулевых пересечений [0, 100].
                По умолчанию 10. Меньшие значения → больше границ, но больше шума.

        Returns:
            MaskArray: Бинарная маска границ формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Соотношение `sigma2 / sigma1` контролирует диапазон обнаруживаемых масштабов:
            * 1.5–2.0 → узкий диапазон, высокая точность
            * 2.0–4.0 → широкий диапазон, универсальность
            - Нулевые пересечения могут быть разрывными; для получения связных
            границ рассмотрите морфологическое замыкание после детекции.
            - Векторизованная реализация zero-crossing быстрее циклической,
            но может давать небольшие отличия на границах изображения.
            - Для зашумлённых изображений увеличьте `sigma1` или примените
            предварительное сглаживание.

        Example:
            ```python
            # Базовое обнаружение границ среднего масштаба
            segmenter = OpenCVSegmenter("dog_edge", sigma1=1.0, sigma2=2.0, threshold=10)
            edges = segmenter.segment(image)

            # Для детекции мелких границ
            segmenter = OpenCVSegmenter("dog_edge", sigma1=0.5, sigma2=1.0, threshold=5)
            fine_edges = segmenter.segment(fine_details_image)

            # Для крупных объектов с подавлением шума
            segmenter = OpenCVSegmenter("dog_edge", sigma1=2.0, sigma2=4.0, threshold=20)
            coarse_edges = segmenter.segment(large_objects_image)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        sigma1: float = float(self.params.get("sigma1", 1.0))
        sigma2: float = float(self.params.get("sigma2", 2.0))
        threshold: float = float(self.params.get("threshold", 10))

        # Применяем два Гауссовых фильтра
        g1: FloatArray = gaussian_filter(gray.astype(np.float32), sigma=sigma1)
        g2: FloatArray = gaussian_filter(gray.astype(np.float32), sigma=sigma2)

        # Разность Гауссианов
        dog: FloatArray = g1 - g2

        # Векторизованное zero-crossing
        magnitude: FloatArray = np.abs(dog)
        sign: npt.NDArray[np.float64] = np.sign(dog)
        zc_h: npt.NDArray[np.bool_] = sign[:, :-1] * sign[:, 1:] < 0
        zc_v: npt.NDArray[np.bool_] = sign[:-1, :] * sign[1:, :] < 0
        zero_crossing_bool: npt.NDArray[np.bool_] = np.zeros_like(dog, dtype=bool)
        zero_crossing_bool[:, :-1] |= zc_h
        zero_crossing_bool[:-1, :] |= zc_v

        zero_crossing: MaskArray = (
            zero_crossing_bool & (magnitude > threshold)
        ).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "dog_edge_opencv",
            exec_time,
            {"sigma1": sigma1, "sigma2": sigma2, "threshold": threshold, **kwargs},
        )

        return zero_crossing

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_marr_hildreth_edge(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Обнаружение границ методом Марра-Хилдрета.

        Классический метод, комбинирующий Гауссово размытие (подавление шума) и
        оператор Лапласа (детекция границ через вторые производные). Границы
        обнаруживаются по нулевым пересечениям (zero-crossings) лапласиана.

        Формула:
        ```
        LoG(x, y) = ∇²[G(x, y, σ) * I(x, y)]
        где:
        - ∇² — оператор Лапласа (вторые производные)
        - G — Гауссово ядро с параметром σ
        - * — операция свёртки
        ```

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Гауссово размытие изображения с параметром `sigma`.
        3. Применение оператора Лапласа через `cv2.Laplacian`.
        4. Векторизованное обнаружение нулевых пересечений (соседние пиксели с противоположными знаками).
        5. Фильтрация слабых пересечений по порогу `threshold`.
        6. Возврат бинарной маски границ.

        Метод особенно эффективен для:
        - Изображений с плавными переходами интенсивности
        - Медицинских изображений с размытыми границами тканей
        - Задач, где важна инвариантность к масштабу (через параметр `sigma`)

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                - `sigma` (float): Стандартное отклонение Гаусса [0.1, 10.0].
                По умолчанию 1.0. Больше значения → детекция более крупных границ.
                - `threshold` (int): Порог для отсечения слабых нулевых пересечений [0, 100].
                По умолчанию 10. Меньшие значения → больше границ, но больше шума.

        Returns:
            MaskArray: Бинарная маска границ формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Параметр `sigma` контролирует масштаб детектируемых границ:
            * Малые σ (0.5–1.0) → мелкие детали, но больше шума
            * Большие σ (2.0–5.0) → крупные границы, но потеря мелких деталей
            - Нулевые пересечения могут быть разрывными; для получения связных
            границ рассмотрите морфологическое замыкание после детекции.
            - Метод чувствителен к шуму; Гауссово размытие частично подавляет шум,
            но для зашумлённых изображений может потребоваться дополнительная
            предобработка (медианный фильтр, bilateral filter).
            - Векторизованная реализация zero-crossing быстрее циклической,
            но может давать небольшие отличия на границах изображения.

        Example:
            ```python
            # Базовое обнаружение границ
            segmenter = OpenCVSegmenter("marr_hildreth_edge", sigma=1.0, threshold=10)
            edges = segmenter.segment(image)

            # Для детекции крупных границ
            segmenter = OpenCVSegmenter("marr_hildreth_edge", sigma=3.0, threshold=15)
            edges = segmenter.segment(large_objects_image)

            # Для зашумлённых изображений
            segmenter = OpenCVSegmenter("marr_hildreth_edge", sigma=2.0, threshold=20)
            edges = segmenter.segment(noisy_image)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        sigma: float = float(self.params.get("sigma", 1.0))
        threshold: float = float(self.params.get("threshold", 10))

        # Лапласиан Гауссиана через OpenCV
        laplacian: npt.NDArray[np.float64] = cv2.Laplacian(
            cv2.GaussianBlur(gray, (0, 0), sigma), cv2.CV_64F
        ).astype(np.float64)
        magnitude: npt.NDArray[np.float64] = np.abs(laplacian)

        # Векторизованное zero-crossing
        sign: npt.NDArray[np.float64] = np.sign(laplacian)
        zc_h: npt.NDArray[np.bool_] = sign[:, :-1] * sign[:, 1:] < 0
        zc_v: npt.NDArray[np.bool_] = sign[:-1, :] * sign[1:, :] < 0
        zero_crossing_bool: npt.NDArray[np.bool_] = np.zeros_like(laplacian, dtype=bool)
        zero_crossing_bool[:, :-1] |= zc_h
        zero_crossing_bool[:-1, :] |= zc_v
        zero_crossing: MaskArray = (
            zero_crossing_bool & (magnitude > threshold)
        ).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "marr_hildreth_edge_opencv",
            exec_time,
            {"sigma": sigma, "threshold": threshold, **kwargs},
        )

        return zero_crossing

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_gradient_magnitude_direction(
        self, img: ImageArray, **kwargs: Any
    ) -> MaskArray:
        """
        Обнаружение границ через магнитуду и направление градиента.

        Вычисляет градиент изображения с помощью операторов Собеля, затем позволяет
        фильтрацию границ не только по силе (магнитуде), но и по ориентации.
        Полезен для выделения границ определённого направления (вертикальных,
        горизонтальных, диагональных).

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Вычисление градиентов по осям X и Y через `cv2.Sobel`.
        3. Расчёт магнитуды: `|G| = sqrt(G_x² + G_y²)` и нормализация к [0, 1].
        4. Расчёт направления: `θ = arctan2(G_y, G_x)` в градусах.
        5. Пороговая бинаризация по магнитуде.
        6. Опциональная фильтрация по диапазону углов `angle_range`.

        Метод особенно эффективен для:
        - Изображений с границами определённой ориентации (текст, решётки, волокна)
        - Задач, где нужно отделить вертикальные/горизонтальные структуры
        - Предварительной обработки для детекции линий (Hough Transform)

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                - `threshold` (float): Порог магнитуды [0, 1] (после нормализации).
                По умолчанию 50/255 ≈ 0.196. Меньшие значения → больше границ.
                - `angle_range` (Optional[Tuple[float, float]]): Диапазон углов
                для фильтрации в градусах [0, 360]. По умолчанию None (все углы).
                Пример: (80, 100) для вертикальных границ.

        Returns:
            MaskArray: Бинарная маска границ формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Параметр `threshold` автоматически конвертируется из [0, 255] в [0, 1],
            если переданное значение > 1.0.
            - Углы измеряются в градусах: 0° = вправо, 90° = вверх, 180° = влево.
            - Фильтрация по `angle_range` учитывает симметрию: угол θ эквивалентен θ+180°.
            - Для выделения только горизонтальных границ используйте `angle_range=(0, 20) | (160, 180)`.
            - Для выделения только вертикальных границ используйте `angle_range=(70, 110)`.
            - Возвращаемая маска содержит только контуры, не заполненные области.


        Example:
            ```python
            # Базовое обнаружение всех границ
            segmenter = OpenCVSegmenter("gradient_magnitude_direction", threshold=0.2)
            edges = segmenter.segment(image)

            # Только вертикальные границы (текст, столбцы)
            segmenter = OpenCVSegmenter(
                "gradient_magnitude_direction",
                threshold=0.15,
                angle_range=(80, 100)
            )
            vertical_edges = segmenter.segment(text_image)

            # Только горизонтальные границы (строки, линии)
            segmenter = OpenCVSegmenter(
                "gradient_magnitude_direction",
                threshold=0.15,
                angle_range=(0, 20)
            )
            horizontal_edges = segmenter.segment(table_image)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        threshold: float = float(self.params.get("threshold", 50))
        if threshold > 1.0:  # Если порог в [0,255]
            threshold = threshold / 255.0
        angle_range: Optional[Tuple[float, float]] = self.params.get(
            "angle_range", None
        )

        # Градиенты Собеля
        grad_x: npt.NDArray[np.float64] = cv2.Sobel(
            gray, cv2.CV_64F, 1, 0, ksize=3
        ).astype(np.float64)
        grad_y: npt.NDArray[np.float64] = cv2.Sobel(
            gray, cv2.CV_64F, 0, 1, ksize=3
        ).astype(np.float64)

        # Магнитуда и направление
        magnitude: npt.NDArray[np.float64] = np.sqrt(grad_x**2 + grad_y**2)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        direction: npt.NDArray[np.float64] = (
            np.arctan2(grad_y, grad_x) * 180 / np.pi
        )  # В градусах

        # Фильтрация по магнитуде
        mask: MaskArray = (magnitude > threshold).astype(np.uint8) * 255

        # Опциональная фильтрация по направлению
        if angle_range is not None:
            angle_mask: npt.NDArray[np.bool_] = (
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

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_phase_congruency_edge(
        self, img: ImageArray, **kwargs: Any
    ) -> MaskArray:
        """
        Обнаружение границ через фазовую конгруэнтность (полная реализация Ковези).

        Инвариантный к изменению контраста и яркости метод, обнаруживающий границы через
        выравнивание фаз Фурье-компонент в пространстве изображений. В отличие от методов,
        основанных на градиенте, фазовая конгруэнтность обнаруживает границы даже при
        плавных переходах интенсивности.

        Алгоритм:
        1. Нормализация изображения к диапазону [0, 1].
        2. FFT изображения для перехода в частотную область.
        3. Построение частотной сетки (радиус и угол для каждого пикселя).
        4. Для каждого масштаба и ориентации:
        - Построение фильтра Log-Gabor в частотной области
        - Умножение с изображением и обратное FFT
        - Вычисление even/odd откликов и амплитуды
        - Оценка шума через медианное абсолютное отклонение (MAD)
        - Накопление энергии и шума
        5. Вычисление локальной энергии и компенсация шума.
        6. Нормализация карты фазовой конгруэнтности к [0, 1].
        7. Пороговая бинаризация.

        Метод особенно эффективен для:
        - Медицинских изображений с плавными переходами тканей
        - Спутниковых снимков с размытыми границами объектов
        - Микрофотографий с низким отношением сигнал/шум

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                    - `nscales` (int): Количество масштабов [1, 8]. По умолчанию 4.
                    Больше масштабов → детекция границ разного размера.
                    - `norientations` (int): Количество ориентаций [1, 12]. По умолчанию 4.
                    Больше ориентаций → лучшая детекция границ любой формы.
                    - `min_wavelength` (int): Минимальная длина волны [1, 10]. По умолчанию 3.
                    - `mult` (float): Мультипликатор длины волны между масштабами [1.0, 5.0].
                    По умолчанию 2.0.
                    - `sigma_onf` (float): Стандартное отклонение в частотной области [0.1, 2.0].
                    По умолчанию 0.55.
                    - `k_noise` (float): Коэффициент шумоподавления [0.5, 5.0]. По умолчанию 2.0.
                    - `threshold` (float): Порог для бинаризации [0.0, 1.0]. По умолчанию 0.3.

        Returns:
            MaskArray: Бинарная маска границ формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Метод вычислительно интенсивный (особенно для больших `nscales` и `norientations`);
            для больших изображений рассмотрите предварительный ресайз.
            - Инвариантность к освещению делает метод идеальным для изображений с
            неравномерным освещением или изменяющимся контрастом.
            - Параметры по умолчанию подходят для большинства задач; тонкая настройка
            может потребоваться для специфичных доменов.
            - Для ускорения вычислений можно использовать FFT на основе GPU (cuFFT).

        Example:
            ```python
            # Базовое использование для медицинского изображения
            segmenter = OpenCVSegmenter("phase_congruency_edge", nscales=4, norientations=4, threshold=0.3)
            edges = segmenter.segment(medical_image)

            # Для детекции мелких границ
            segmenter = OpenCVSegmenter("phase_congruency_edge", nscales=6, min_wavelength=2, threshold=0.25)
            edges = segmenter.segment(fine_details_image)

            # Для ускорения на больших изображениях
            segmenter = OpenCVSegmenter("phase_congruency_edge", nscales=3, norientations=3)
            edges = segmenter.segment(large_satellite_image)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        # Получение параметров
        nscales: int = int(self.params.get("nscales", 4))
        norientations: int = int(self.params.get("norientations", 4))
        min_wavelength: int = int(self.params.get("min_wavelength", 3))
        mult: float = float(self.params.get("mult", 2.0))
        sigma_onf: float = float(self.params.get("sigma_onf", 0.55))
        k_noise: float = float(self.params.get("k_noise", 2.0))
        threshold = min(float(self.params.get("threshold", 0.3)), 0.99)
        epsilon: float = 1e-10

        # Нормализация к [0, 1]
        gray_norm: FloatArray = gray.astype(np.float32)
        if gray_norm.max() > 1.0:
            gray_norm = gray_norm / 255.0

        rows: int = gray_norm.shape[0]
        cols: int = gray_norm.shape[1]

        # FFT изображения
        img_fft: npt.NDArray[np.complex128] = np.fft.fft2(gray_norm)
        img_fft_shifted: npt.NDArray[np.complex128] = np.fft.fftshift(img_fft)

        # Частотная сетка
        y: npt.NDArray[np.float64] = np.fft.fftshift(np.fft.fftfreq(rows))
        x: npt.NDArray[np.float64] = np.fft.fftshift(np.fft.fftfreq(cols))
        X, Y = np.meshgrid(x, y)
        R: FloatArray = np.sqrt(X**2 + Y**2 + epsilon)  # Защита от деления на 0
        Theta: FloatArray = np.arctan2(-Y, X)  # Угол в радианах

        # Аккумуляторы
        sum_even: FloatArray = np.zeros((rows, cols), dtype=np.float32)
        sum_odd: FloatArray = np.zeros((rows, cols), dtype=np.float32)
        sum_amp: FloatArray = np.zeros((rows, cols), dtype=np.float32)
        noise_energy: FloatArray = np.zeros((rows, cols), dtype=np.float32)

        orientations: npt.NDArray[np.float64] = np.linspace(
            0, np.pi, norientations, endpoint=False
        ).astype(np.float64)

        for scale in range(nscales):
            wavelength: float = min_wavelength * (mult**scale)
            fo: float = 1.0 / wavelength

            # Log-Gabor фильтр (радиальная часть)
            # sigma_f: float = sigma_onf * fo
            log_ratio: FloatArray = np.log(R / fo + epsilon) / np.log(
                sigma_onf + epsilon
            )
            log_gabor: FloatArray = np.exp(-0.5 * log_ratio**2)
            log_gabor[0, 0] = 0.0  # DC = 0

            for angle in orientations:
                # Угловая часть (Гауссов разброс)
                angular_spread: float = np.pi / 2 / norientations
                d_theta: FloatArray = np.abs(Theta - angle)
                d_theta = np.minimum(d_theta, 2 * np.pi - d_theta)
                angular: FloatArray = np.exp(-0.5 * (d_theta / angular_spread) ** 2)

                # Полный фильтр в частотной области
                filter_f: FloatArray = log_gabor * angular
                product = img_fft_shifted * filter_f

                # Свёртка в частотной области
                response = np.fft.ifft2(np.fft.ifftshift(product))
                even_resp: FloatArray = np.real(response).astype(np.float32)
                odd_resp: FloatArray = np.imag(response).astype(np.float32)

                # Амплитуда отклика
                amp: FloatArray = np.sqrt(even_resp**2 + odd_resp**2 + epsilon)

                # Оценка шума (MAD) для текущего фильтра
                med = np.median(amp)
                noise_est: float = 2.0 * (med / 0.6745)

                # Накопление
                sum_even += even_resp
                sum_odd += odd_resp
                sum_amp += amp
                noise_energy += noise_est**2

        # Вычисление фазовой конгруэнтности
        local_energy: FloatArray = np.sqrt(sum_even**2 + sum_odd**2 + epsilon)

        # Компенсация шума (Ковези)
        T: np.ndarray = noise_energy * k_noise
        pc_map: FloatArray = np.maximum(local_energy - T, 0) / (sum_amp + epsilon)

        # Ограничение [0, 1]
        # pc_map = cv2.normalize(pc_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        pc_map = np.clip(pc_map, 0, 1)

        # Бинаризация
        # _, mask = cv2.threshold(pc_map, threshold, 255, cv2.THRESH_BINARY)
        # mask = mask.astype(np.uint8)
        mask = (pc_map > threshold).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time

        self._log_info(
            "phase_congruency_edge_opencv",
            exec_time,
            {
                "nscales": nscales,
                "norientations": norientations,
                "min_wavelength": min_wavelength,
                "mult": mult,
                "sigma_onf": sigma_onf,
                "k_noise": k_noise,
                "threshold": threshold,
                **kwargs,
            },
        )

        return mask

    # ============ РЕГИОНАЛЬНЫЕ МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _opencv_region_growing(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Сегментация методом роста регионов (Region Growing).

        Алгоритм итеративно добавляет к региону соседние пиксели, интенсивность которых
        отличается от средней интенсивности региона не более чем на заданный допуск.

        Алгоритм:
        1. Выбор начальной точки (семени) `seed`.
        2. Инициализация очереди пикселей для обработки и маски посещённых пикселей.
        3. Пока очередь не пуста:
        - Извлечь пиксель `(x, y)`.
        - Если пиксель не посещён и `|img[x,y] - mean_region| <= tolerance`:
            * Добавить пиксель к региону (установить `mask[x,y] = 255`).
            * Добавить 8-связных соседей в очередь.
        4. Вернуть бинарную маску региона.

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                    - `seed` (Tuple[int, int] | None): Координаты семени `(x, y)`.
                    Если `None`, используется центр изображения.
                    - `tolerance` (int): Максимально допустимое отклонение интенсивности [0, 255].
                    По умолчанию 25. Меньшие значения → более однородный регион, но меньше площадь.

        Returns:
            MaskArray: Бинарная маска региона формы `(H, W)`, dtype=uint8, {0, 255}.

        Raises:
            ValueError: Если `seed` указан, но выходит за границы изображения.

        Note:
            - Метод чувствителен к выбору семени; неправильный выбор может привести
            к сегментации фона вместо объекта.
            - Для многокластерных изображений рассмотрите K-Means или Watershed.
            - Алгоритм имеет сложность O(N) в худшем случае, но на практике работает
            быстро для компактных регионов.
            - Используйте 4-связность вместо 8-связности, если нужны более "угловатые" границы.

        Example:
            ```python
            # Сегментация объекта в центре изображения
            segmenter = OpenCVSegmenter("region_growing", tolerance=20)
            mask = segmenter.segment(image)

            # С явным указанием семени
            segmenter = OpenCVSegmenter("region_growing", seed=(100, 150), tolerance=15)
            mask = segmenter.segment(image)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        start_time: float = time.time()

        h, w = gray.shape

        # Получение параметров
        seed_raw = self.params.get("seed", None)
        tolerance: int = int(self.params.get("tolerance", 25))

        seed: Tuple[int, int]
        if seed_raw is None or not (0 <= seed_raw[0] < w and 0 <= seed_raw[1] < h):
            seed = (w // 2, h // 2)  # (x, y)
        else:
            seed = seed_raw

        # Инициализация структур данных
        mask: MaskArray = np.zeros((h, w), dtype=np.uint8)
        visited: npt.NDArray[np.bool_] = np.zeros((h, w), dtype=bool)
        queue: deque[Tuple[int, int]] = deque([seed])

        # Начальное значение интенсивности
        start_value: float = float(gray[seed[1], seed[0]])

        while queue:
            x, y = queue.popleft()

            # Проверка границ и посещённости
            if x < 0 or x >= w or y < 0 or y >= h or visited[y, x]:
                continue

            visited[y, x] = True

            # Проверка условия схожести
            if abs(int(gray[y, x]) - int(start_value)) <= tolerance:
                mask[y, x] = 255

                # Добавление 8-связных соседей в очередь
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                            queue.append((nx, ny))

        exec_time: float = time.time() - start_time

        self._log_info(
            "region_growing_opencv",
            exec_time,
            {"seed": seed, "tolerance": tolerance, **kwargs},
        )
        print(f"Info after OpenCV_region_growing: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_split_and_merge(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Рекурсивный алгоритм разделения и слияния регионов (Split and Merge).

        Алгоритм итеративно делит изображение на квадранты до тех пор, пока дисперсия
        интенсивности внутри региона не станет меньше заданного порога. Затем похожие
        соседние регионы объединяются. Возвращается маска второго по величине региона
        (предполагаемый объект).

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Инициализация: всё изображение — один регион.
        3. Фаза разделения (Split):
        - Если дисперсия региона > `threshold` и размер > `min_size`:
            * Разделить регион на 4 квадранта
            * Рекурсивно обработать каждый квадрант
        4. Фаза слияния (Merge):
        - Для каждой пары соседних регионов:
            * Если разница средних < `threshold`: объединить регионы
        5. Выбор второго по величине региона как объекта.
        6. Создание бинарной маски.

        Метод особенно эффективен для:
        - Спутниковых изображений с чёткими границами регионов
        - Индустриальных изображений с однородными областями
        - Изображений с несколькими объектами разного размера

        Args:
            img: Входное изображение (RGB или grayscale).
            **kwargs: Дополнительные параметры:
                    - `threshold` (int): Порог дисперсии для разделения регионов [1, 100].
                    По умолчанию 20. Меньшие значения → больше регионов, более детальная сегментация.
                    - `min_size` (int): Минимальный размер региона в пикселях [10, 500].
                    По умолчанию 50. Меньшие значения → возможность выделения мелких объектов.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = второй по величине регион (объект), 0 = фон.

        Note:
            - Алгоритм имеет экспоненциальную сложность в худшем случае, но на практике
            работает быстро для изображений с чёткими границами регионов.
            - Для изображений с плавными переходами (например, естественные сцены)
            метод может давать фрагментированные результаты.
            - Параметр `threshold` следует подбирать экспериментально:
            * Чёткие границы: 10–30
            * Плавные переходы: 30–60
            * Очень однородные: 60–100
            - Возвращается именно второй по величине регион, так как самый крупный
            обычно является фоном. Для многокластерных изображений рассмотрите
            K-Means или Watershed.

        Example:
            ```python
            # Базовое использование для спутникового снимка
            segmenter = OpenCVSegmenter("split_and_merge", threshold=20, min_size=50)
            mask = segmenter.segment(satellite_image)

            # Для изображения с мелкими объектами
            segmenter = OpenCVSegmenter("split_and_merge", threshold=15, min_size=20)
            mask = segmenter.segment(microscopy_image)

            # Для очень однородных регионов
            segmenter = OpenCVSegmenter("split_and_merge", threshold=40, min_size=100)
            mask = segmenter.segment(industrial_image)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        start_time: float = time.time()

        h, w = gray.shape
        threshold: int = int(self.params.get("threshold", 20))
        min_size: int = int(self.params.get("min_size", 50))

        # Внутренняя функция: статистики региона
        def region_stats(region: List[Tuple[int, int]]) -> Tuple[float, float]:
            """Вычисляет среднее и стандартное отклонение интенсивности в регионе."""
            pixels: List[int] = [int(gray[y, x]) for x, y in region]
            if not pixels:
                return 0.0, 0.0
            mean_val: float = float(np.mean(pixels))
            std_val: float = float(np.std(pixels))
            return mean_val, std_val

        # Внутренняя функция: рекурсивное разделение
        def split(
            region: List[Tuple[int, int]], min_sz: int, thresh: int
        ) -> List[List[Tuple[int, int]]]:
            """Рекурсивно делит регион на квадранты, если дисперсия слишком велика."""
            if len(region) <= min_sz:
                return [region]

            mean_val, std_val = region_stats(region)
            if std_val < thresh:
                return [region]

            # Координаты региона
            x_coords: List[int] = [p[0] for p in region]
            y_coords: List[int] = [p[1] for p in region]

            # Середина региона
            x_mid: int = (min(x_coords) + max(x_coords)) // 2
            y_mid: int = (min(y_coords) + max(y_coords)) // 2

            # Разделение на 4 квадранта
            quadrants: List[List[Tuple[int, int]]] = [
                [(x, y) for x, y in region if x <= x_mid and y <= y_mid],  # верх-лево
                [(x, y) for x, y in region if x > x_mid and y <= y_mid],  # верх-право
                [(x, y) for x, y in region if x <= x_mid and y > y_mid],  # низ-лево
                [(x, y) for x, y in region if x > x_mid and y > y_mid],  # низ-право
            ]

            # Рекурсивная обработка квадрантов
            result: List[List[Tuple[int, int]]] = []
            for quad in quadrants:
                if quad:  # Пропускаем пустые квадранты
                    result.extend(split(quad, min_sz, thresh))
            return result

        # Внутренняя функция: слияние похожих регионов
        def merge(
            regions: List[List[Tuple[int, int]]], thresh: int
        ) -> List[List[Tuple[int, int]]]:
            """Объединяет соседние регионы со схожими средними значениями."""
            merged: List[List[Tuple[int, int]]] = []
            used: List[bool] = [False] * len(regions)

            for i, reg1 in enumerate(regions):
                if used[i]:
                    continue

                current: List[Tuple[int, int]] = reg1.copy()
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

        # Начальный регион — всё изображение
        initial: List[Tuple[int, int]] = [(x, y) for y in range(h) for x in range(w)]

        # Фаза разделения
        regions: List[List[Tuple[int, int]]] = split(initial, min_size, threshold)

        # Фаза слияния
        regions = merge(regions, threshold)

        # Создание маски: второй по величине регион = объект
        if len(regions) > 1:
            sizes: List[int] = [len(r) for r in regions]
            idx: int = int(np.argsort(sizes)[-2])  # Второй по величине
            mask: MaskArray = np.zeros((h, w), dtype=np.uint8)
            for x, y in regions[idx]:
                mask[y, x] = 255

            exec_time: float = time.time() - start_time

            self._log_info(
                "split_and_merge_opencv",
                exec_time,
                {"threshold": threshold, "min_size": min_size, **kwargs},
            )
            print(f"Info after OpenCV_split_and_merge: {self._log_info}")

            return mask

        return np.zeros((h, w), dtype=np.uint8)

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_floodfill(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Сегментация методом заливки (Flood Fill).

        Начиная с заданной точки (семени), алгоритм рекурсивно заполняет все связанные
        пиксели, интенсивность которых отличается от исходной не более чем на заданный
        допуск. Использует 4- или 8-связность для определения соседства.

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Инициализация маски с паддингом для корректной обработки границ.
        3. Применение `cv2.floodFill` с параметрами допуска и флагами.
        4. Извлечение финальной маски без паддинга.
        5. Опциональное заполнение дыр через `ndimage.binary_fill_holes`.

        Метод особенно эффективен для:
        - Интерактивной сегментации с указанием точки внутри объекта
        - Изображений с однородными областями и чёткими границами
        - Медицинских изображений с выделением конкретных анатомических структур

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                    - `seed` (Tuple[int, int] | None): Координаты семени (x, y).
                    Если `None`, используется центр изображения.
                    - `tolerance` (int): Допуск интенсивности [0, 255].
                    По умолчанию 20. Меньшие значения → более строгая заливка.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = залитая область, 0 = фон.

        Note:
            - Качество результата критически зависит от правильного выбора семени;
            семя должно находиться внутри целевого объекта.
            - Параметр `tolerance` следует подбирать экспериментально:
            * Однородные области: 10–30
            * Градиентные области: 30–60
            * Очень неоднородные: 60–100
            - Метод использует 4-связность по умолчанию; для 8-связности измените
            флаг `flags` в вызове `cv2.floodFill`.
            - Для больших изображений метод может быть медленным из-за рекурсивной
            природы; рассмотрите предварительный ресайз при необходимости.

        Example:
            ```python
            # Базовое использование с центром изображения как семенем
            segmenter = OpenCVSegmenter("floodfill", tolerance=20)
            mask = segmenter.segment(image)

            # С явным указанием семени
            segmenter = OpenCVSegmenter("floodfill", seed=(100, 150), tolerance=15)
            mask = segmenter.segment(image)

            # Для градиентных областей
            segmenter = OpenCVSegmenter("floodfill", tolerance=40)
            mask = segmenter.segment(gradient_image)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        start_time: float = time.time()

        h: int = gray.shape[0]
        w: int = gray.shape[1]

        # Получение параметров
        seed_raw = self.params.get("seed", None)
        tolerance: int = int(self.params.get("tolerance", 20))

        # Установка семени
        seed: Tuple[int, int]
        if seed_raw is None or not (0 <= seed_raw[0] < w and 0 <= seed_raw[1] < h):
            seed = (w // 2, h // 2)  # (x, y)
        else:
            seed = seed_raw

        # Инициализация маски с паддингом (требуется для cv2.floodFill)
        mask_padded: MaskArray = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Флаги для floodFill: 4-связность + фиксированный диапазон
        flags: int = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE

        # Применение floodFill
        cv2.floodFill(
            gray.copy(),
            mask_padded,
            seed,
            255,  # Новое значение для залитых пикселей
            (tolerance,) * 3,  # Нижний допуск для каждого канала
            (tolerance,) * 3,  # Верхний допуск для каждого канала
            flags,
        )

        # Извлечение маски без паддинга
        mask_final: MaskArray = mask_padded[1:-1, 1:-1] * 255

        # Опциональное заполнение дыр
        mask_final = ndimage.binary_fill_holes(mask_final > 0).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time

        self._log_info(
            "floodfill_opencv",
            exec_time,
            {"seed": seed, "tolerance": tolerance, **kwargs},
        )
        print(f"Info after OpenCV_floodfill: {self._log_info}")

        return mask_final

    # ============ КЛАСТЕРИЗАЦИЯ ============
    # ──────────────────────────────────────────────────────────────────────
    def _opencv_kmeans_segmentation(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Сегментация методом кластеризации K-Means.

        Группирует пиксели изображения в `k` кластеров в пространстве цветовых признаков
        (для RGB) или интенсивности (для grayscale). Самый крупный кластер считается фоном;
        остальные — объектами.

        Алгоритм:
        1. Конвертация изображения в RGB при необходимости.
        2. Преобразование изображения в массив пикселей формы `(N, 3)`.
        3. Применение `cv2.kmeans` с критерием остановки по точности/итерациям.
        4. Преобразование меток кластеров обратно в форму `(H, W)`.
        5. Определение фона как кластера с максимальным количеством пикселей.
        6. Создание бинарной маски: `255` для не-фоновых кластеров, `0` для фона.

        Args:
            img: Входное изображение. Поддерживаются:
                - Grayscale: `(H, W)` → автоматически конвертируется в 3-канальное.
                - RGB/BGR: `(H, W, 3)`, dtype=uint8.
            **kwargs: Дополнительные параметры:
                    - `k` (int): Количество кластеров [2, 20]. По умолчанию 3.
                    Меньшие значения → более грубая сегментация.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Метод не учитывает пространственную связность; для улучшения результата
            можно добавить координаты пикселей в пространство признаков:
            `features = np.hstack([pixels, coords * weight])`.
            - K-Means чувствителен к инициализации центроидов; для воспроизводимости
            установите `cv2.KMEANS_PP_CENTERS` вместо `RANDOM_CENTERS`.
            - Для больших изображений (>1000×1000) рассмотрите предварительный ресайз
            или использование MiniBatchKMeans из scikit-learn.

        Example:
            ```python
            # Базовая сегментация на 3 кластера
            segmenter = OpenCVSegmenter("kmeans_segmentation", k=3)
            mask = segmenter.segment(color_image)

            # С добавлением координат для пространственной связности
            # (требует модификации метода, здесь для иллюстрации)
            segmenter = OpenCVSegmenter("kmeans_segmentation", k=5)
            mask = segmenter.segment(satellite_image)
            ```
        """
        # Конвертация grayscale в 3-канальное для единообразия обработки
        if len(img.shape) == 2:
            img_rgb_raw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_rgb: ImageArray = img_rgb_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            img_rgb = img

        start_time: float = time.time()

        h, w = img.shape[:2]
        # Преобразование в массив пикселей (N, 3)
        pixels: FloatArray = img_rgb.reshape(-1, 3).astype(np.float32)

        # Получение параметров
        k: int = int(self.params.get("k", 3))

        # Критерий остановки: точность 0.2 или максимум 100 итераций
        criteria: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.2,
        )

        # Инициализация меток (обязательный аргумент для cv2.kmeans)
        best_labels: npt.NDArray[np.int32] = np.zeros(
            (pixels.shape[0],), dtype=np.int32
        )

        # Выполнение K-Means
        compactness: float
        labels_flat: npt.NDArray[np.int32]
        compactness, labels_flat_raw, centers_raw = cv2.kmeans(
            pixels,
            k,
            best_labels,
            criteria,
            10,  # количество попыток инициализации
            cv2.KMEANS_RANDOM_CENTERS,
        )
        labels_flat = labels_flat_raw.astype(np.int32)

        # Преобразование меток обратно в форму (H, W)
        labels: npt.NDArray[np.int32] = labels_flat.reshape(h, w)

        # Определение фона как самого крупного кластера
        unique: npt.NDArray[np.int32]
        counts: npt.NDArray[np.int32]
        unique, counts_raw = np.unique(labels, return_counts=True)
        counts = counts_raw.astype(np.int32)
        bg_label: int = int(unique[np.argmax(counts)])

        # Создание бинарной маски: всё кроме фона = объект
        mask: MaskArray = (labels != bg_label).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time

        self._log_info(
            "kmeans_segmentation_opencv",
            exec_time,
            {"k": k, **kwargs},
        )
        print(f"Info after OpenCV_kmeans: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_dbscan_segmentation(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Сегментация методом кластеризации DBSCAN (Density-Based Spatial Clustering).

        Группирует пиксели на основе плотности в пространстве признаков (цвет + координаты).
        В отличие от K-Means, DBSCAN:
        - Не требует указания количества кластеров
        - Может обнаруживать кластеры произвольной формы
        - Автоматически определяет шумовые пиксели (метка -1)

        Алгоритм:
        1. Подготовка признаков: для RGB — цвет (3 канала), для grayscale — интенсивность (1 канал).
        2. Опциональный ресайз для больших изображений (>80000 пикселей) для ускорения.
        3. Нормализация признаков к диапазону [0, 1].
        4. Применение DBSCAN с параметрами `eps` (радиус окрестности) и `min_samples`.
        5. Определение фона как самого крупного кластера.
        6. Создание бинарной маски: всё кроме шума (-1) и фона = объект.
        7. Восстановление исходного размера при необходимости.

        Метод особенно эффективен для:
        - Микроскопических изображений с объектами произвольной формы
        - Индустриальных изображений с дефектами неправильной формы
        - Изображений с неизвестным количеством объектов

        Args:
            img: Входное изображение (RGB или grayscale).
            **kwargs: Дополнительные параметры:
                    - `eps` (float): Радиус окрестности для поиска соседей [0.01, 1.0].
                    По умолчанию 0.05. Меньшие значения → больше, но меньших кластеров.
                    - `min_samples` (int): Минимальное количество точек в кластере [1, 50].
                    По умолчанию 5. Меньшие значения → больше шума, больше кластеров.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = объект (не-фоновые кластеры), 0 = фон или шум.

        Note:
            - DBSCAN чувствителен к выбору `eps` и `min_samples`; рекомендуется подбирать
            экспериментально на репрезентативной выборке изображений.
            - Для больших изображений (>1000×1000) метод автоматически уменьшает разрешение,
            что может снизить точность детекции мелких объектов.
            - Шумовые пиксели (метка -1) исключаются из маски — это может быть как преимуществом
            (подавление шума), так и недостатком (потеря мелких деталей).
            - Для цветных изображений метод работает в пространстве цветовых признаков;
            для улучшения результатов можно добавить координаты пикселей:
            `features = np.hstack([color_features, coords * weight])`.

        Example:
            ```python
            # Базовое использование для микроскопического изображения
            segmenter = OpenCVSegmenter("dbscan_segmentation", eps=0.05, min_samples=5)
            mask = segmenter.segment(microscopy_image)

            # Для изображений с мелкими объектами
            segmenter = OpenCVSegmenter("dbscan_segmentation", eps=0.02, min_samples=3)
            mask = segmenter.segment(small_objects_image)

            # Для изображений с крупными однородными областями
            segmenter = OpenCVSegmenter("dbscan_segmentation", eps=0.1, min_samples=10)
            mask = segmenter.segment(large_regions_image)
            ```
        """
        start_time: float = time.time()

        h, w = img.shape[:2]

        # Оптимизация: уменьшение разрешения для больших изображений
        scale: float = 1.0
        if h * w > 80000:
            scale = np.sqrt(80000.0 / (h * w))
            small: ImageArray = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            ).astype(np.uint8)
        else:
            small = img

        sh, sw = small.shape[:2]

        # Подготовка признаков: цвет или интенсивность
        if len(small.shape) == 3:
            # RGB: нормализация к [0, 1]
            pixels: FloatArray = small.reshape(-1, 3).astype(np.float32) / 255.0
        else:
            # Grayscale: нормализация к [0, 1]
            pixels = small.reshape(-1, 1).astype(np.float32) / 255.0

        # Получение параметров DBSCAN
        eps: float = float(self.params.get("eps", 0.05))
        min_samples: int = int(self.params.get("min_samples", 5))

        # Применение DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels_flat: npt.NDArray[np.int32] = db.fit_predict(pixels)

        # Преобразование меток обратно в форму (H, W)
        labels_2d: npt.NDArray[np.int32] = labels_flat.reshape(sh, sw)

        # Фильтрация валидных меток (исключение шума -1)
        valid: npt.NDArray[np.int32] = labels_flat[labels_flat != -1]

        # Создание маски
        mask_small: MaskArray
        if len(valid) > 0:
            # Определение фона как самого крупного кластера
            unique: npt.NDArray[np.int32]
            counts: npt.NDArray[np.int32]
            unique, counts_raw = np.unique(valid, return_counts=True)
            counts = counts_raw.astype(np.int32)
            bg_label: int = int(unique[np.argmax(counts)])

            # Маска: всё кроме фона и шума = объект
            mask_small = ((labels_2d != bg_label) & (labels_2d != -1)).astype(
                np.uint8
            ) * 255
        else:
            # Если все пиксели — шум, возвращаем пустую маску
            mask_small = np.zeros((sh, sw), dtype=np.uint8)

        # Восстановление исходного размера при необходимости
        mask: MaskArray
        if scale < 1.0:
            mask = cv2.resize(
                mask_small,
                (w, h),
                interpolation=cv2.INTER_NEAREST,  # Ближайший сосед для сохранения целочисленных меток
            ).astype(np.uint8)
        else:
            mask = mask_small

        exec_time: float = time.time() - start_time

        self._log_info(
            "dbscan_segmentation_opencv",
            exec_time,
            {"eps": eps, "min_samples": min_samples, **kwargs},
        )
        print(f"Info after OpenCV_dbscan: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_meanshift(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Сегментация методом MeanShift.

        Итеративный алгоритм, сдвигающий каждый пиксель к локальному центру масс в
        пространстве признаков (цвет + координаты). Результатом является кластеризация
        пикселей по плотности, где каждый кластер соответствует однородной области.

        Алгоритм:
        1. Применение `cv2.pyrMeanShiftFiltering` для сглаживания в пространстве
        признаков с параметрами пространственного и цветового радиусов.
        2. Конвертация результата в grayscale.
        3. Автоматическая бинаризация через порог Оцу.

        Метод особенно эффективен для:
        - Изображений с плавными цветовыми переходами (портреты, пейзажи)
        - Медицинских изображений с однородными тканями
        - Задач, где важна сохранность границ между областями

        Args:
            img: Входное изображение (RGB предпочтительно для цветовой сегментации).
            **kwargs: Дополнительные параметры:
                    - `spatial_radius` (int): Пространственный радиус поиска [5, 100].
                    По умолчанию 60. Меньшие значения → более детальная сегментация.
                    - `color_radius` (int): Цветовой радиус поиска [10, 200].
                    По умолчанию 60. Меньшие значения → более строгая кластеризация.
                    - `max_level` (int): Максимальный уровень пирамиды [0, 5].
                    По умолчанию 1. Больше уровней → быстрее, но менее точно.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = самый крупный кластер (предполагаемый объект), 0 = фон.

        Note:
            - Метод не требует указания количества кластеров, в отличие от K-Means.
            - Вычислительная сложность: O(N·iterations), где N — количество пикселей;
            для больших изображений рассмотрите предварительный ресайз.
            - Параметры `spatial_radius` и `color_radius` следует подбирать совместно:
            * Мелкие детали: `spatial_radius=20–40, color_radius=30–50`
            * Крупные области: `spatial_radius=60–100, color_radius=60–100`
            - Результат может зависеть от инициализации; для воспроизводимости
            установите фиксированное случайное семя при необходимости.

        Example:
            ```python
            # Базовое использование для цветного изображения
            segmenter = OpenCVSegmenter("meanshift", spatial_radius=60, color_radius=60)
            mask = segmenter.segment(color_image)

            # Для детальной сегментации мелких объектов
            segmenter = OpenCVSegmenter("meanshift", spatial_radius=30, color_radius=40)
            mask = segmenter.segment(fine_details_image)

            # Для ускорения на больших изображениях
            segmenter = OpenCVSegmenter("meanshift", max_level=2)
            mask = segmenter.segment(large_image)
            ```
        """
        start_time: float = time.time()

        # Получение параметров
        spatial_radius: int = int(self.params.get("spatial_radius", 60))
        color_radius: int = int(self.params.get("color_radius", 60))
        max_level: int = int(self.params.get("max_level", 1))

        # Применение MeanShift через пирамидальную фильтрацию
        shifted: ImageArray = cv2.pyrMeanShiftFiltering(
            img, sp=spatial_radius, sr=color_radius, maxLevel=max_level
        ).astype(np.uint8)

        # Конвертация в grayscale для бинаризации
        gray_raw = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]

        # Автоматическая бинаризация через Оцу
        mask: MaskArray
        _, mask_raw = cv2.threshold(
            gray, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask = mask_raw.astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "meanshift_opencv",
            exec_time,
            {
                "spatial_radius": spatial_radius,
                "color_radius": color_radius,
                "max_level": max_level,
                **kwargs,
            },
        )
        print(f"Info after OpenCV_meanshift: {self._log_info}")

        return mask

    # ============ АКТИВНЫЕ КОНТУРЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _opencv_active_contour(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Сегментация активными контурами (Snakes).

        Инициализирует замкнутый контур (обычно окружность в центре изображения) и
        итеративно деформирует его под действием внутренних и внешних сил до равновесия.

        Внутренние силы (упругость, жёсткость):
        - Стремятся сделать контур гладким и непрерывным
        - Параметризованы коэффициентами `alpha` (плавность) и `beta` (жёсткость)

        Внешние силы (притяжение к границам):
        - Притягивают контур к границам изображения (градиентам интенсивности)
        - Параметризованы коэффициентами `w_edge` (вес границ) и `w_line` (вес линий)

        Алгоритм (упрощённая реализация через морфологические операции):
        1. Конвертация в grayscale при необходимости.
        2. Инициализация контура как окружности в центре изображения.
        3. Для каждой итерации:
        - Вычисление границ через детектор Кэнни.
        - Применение морфологического замыкания для "притягивания" контура к границам.
        4. Возврат маски внутри финального контура.

        Метод особенно эффективен для:
        - Медицинских изображений с чёткими границами органов
        - Микроскопических изображений клеток и тканей
        - Изображений с объектами, близкими по форме к инициализированному контуру

        Args:
            img: Входное изображение (RGB или grayscale).
            **kwargs: Дополнительные параметры:
                    - `iterations` (int): Количество итераций деформации контура [10, 2000].
                    По умолчанию 10. Больше итераций → точнее результат, но медленнее.
                    - `alpha` (float): Коэффициент плавности контура [0.0, 1.0].
                    По умолчанию 0.015. Меньшие значения → более гибкий контур.
                    - `beta` (float): Коэффициент жёсткости контура [0.0, 50.0].
                    По умолчанию 10. Меньшие значения → контур легче деформируется.
                    - `w_edge` (float): Вес внешних сил от границ [0.0, 10.0].
                    По умолчанию 1.0. Большие значения → сильнее притяжение к границам.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = область внутри замкнутого контура (объект), 0 = фон.

        Note:
            - Данная реализация использует упрощённый подход через морфологические операции
            вместо полной энергетической оптимизации. Для точной сегментации рассмотрите
            библиотеки `scikit-image` или `OpenCV` с полной реализацией snakes.
            - Качество результата сильно зависит от инициализации контура. Для сложных объектов
            может потребоваться ручная инициализация или использование нескольких контуров.
            - Метод чувствителен к шуму; для зашумлённых изображений рекомендуется
            предварительное сглаживание (Гауссово размытие).
            - Параметр `iterations` следует подбирать экспериментально:
            * Простые объекты: 10–50
            * Сложные объекты: 100–500
            * Очень сложные: 500–2000

        Example:
            ```python
            # Базовое использование для медицинского изображения
            segmenter = OpenCVSegmenter("active_contour", iterations=50)
            mask = segmenter.segment(medical_image)

            # Для объектов сложной формы
            segmenter = OpenCVSegmenter("active_contour", iterations=200, alpha=0.01, beta=5)
            mask = segmenter.segment(complex_object_image)

            # С предварительным сглаживанием для шумных изображений
            blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
            segmenter = OpenCVSegmenter("active_contour", iterations=100)
            mask = segmenter.segment(blurred)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        start_time: float = time.time()

        h, w = gray.shape

        # Инициализация контура как окружности в центре
        center_x: int = w // 2
        center_y: int = h // 2
        radius: int = min(center_x, center_y) // 2

        # Создание начальной маски контура
        mask: MaskArray = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(
            mask, (center_x, center_y), radius, 255, -1
        )  # Заполненная окружность

        # Получение параметров
        iterations: int = int(self.params.get("iterations", 10))
        kernel: npt.NDArray[np.uint8] = np.ones((5, 5), dtype=np.uint8)

        # Итеративная деформация контура
        for _ in range(iterations):
            # Вычисление границ через детектор Кэнни
            edges: MaskArray = cv2.Canny(gray, 100, 200).astype(np.uint8)

            # "Притягивание" контура к границам через морфологические операции
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges)).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "active_contour_opencv",
            exec_time,
            {"iterations": iterations, **kwargs},
        )
        print(f"Info after OpenCV_active_contour: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_gvf_contour(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Сегментация на основе градиентного векторного потока (GVF / Gradient Vector Flow).

        Метод распространяет информацию о градиентах по всему изображению через
        диффузионный процесс, позволяя активному контуру "чувствовать" границы
        даже на расстоянии и в областях со слабым градиентом.

        Уравнение диффузии:
        ```
        ∂v/∂t = μ·∇²v - |∇f|²·(v - ∇f)
        где:
        - v — векторное поле GVF
        - μ — коэффициент диффузии (контролирует распространение)
        - ∇f — градиент исходного изображения
        - ∇² — оператор Лапласа
        ```

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Вычисление градиентов изображения через оператор Собеля.
        3. Инициализация векторного поля GVF градиентами.
        4. Итеративное решение уравнения диффузии для распространения градиентов.
        5. Вычисление магнитуды финального векторного поля.
        6. Пороговая бинаризация и заполнение дыр.

        Метод особенно эффективен для:
        - Медицинских изображений с размытыми или прерывистыми границами
        - Микроскопических изображений клеток с неоднородной текстурой
        - Объектов с вогнутыми границами, которые трудно детектировать обычными методами

        Args:
            img: Входное изображение (RGB или grayscale).
            **kwargs: Дополнительные параметры:
                    - `mu` (float): Коэффициент диффузии [0.0, 1.0].
                    По умолчанию 0.1. Меньшие значения → более локальное распространение.
                    - `iterations` (int): Количество итераций диффузии [10, 500].
                    По умолчанию 50. Больше итераций → более полное распространение.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = область с высокой величиной GVF (предполагаемый объект).

        Note:
            - Метод вычислительно интенсивный (особенно для больших `iterations`);
            для больших изображений рассмотрите предварительный ресайз.
            - Параметр `mu` контролирует баланс между сохранением исходных градиентов
            и их распространением: меньшие значения сохраняют детали, но могут
            не "достать" до удалённых границ.
            - Результат зависит от инициализации; для сложных объектов может
            потребоваться ручная инициализация контура или использование
            нескольких начальных точек.
            - Для зашумлённых изображений рекомендуется предварительное сглаживание.

        Example:
            ```python
            # Базовое использование для медицинского изображения
            segmenter = OpenCVSegmenter("gvf_contour", mu=0.1, iterations=50)
            mask = segmenter.segment(medical_image)

            # Для объектов со сложными вогнутыми границами
            segmenter = OpenCVSegmenter("gvf_contour", mu=0.05, iterations=100)
            mask = segmenter.segment(complex_boundary_image)

            # Для зашумлённых изображений
            blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=1.0)
            segmenter = OpenCVSegmenter("gvf_contour", mu=0.15, iterations=75)
            mask = segmenter.segment(blurred)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        start_time: float = time.time()

        # Получение параметров
        mu: float = float(self.params.get("mu", 0.1))
        iterations: int = int(self.params.get("iterations", 50))

        # Вычисление градиентов Собеля
        grad_x: npt.NDArray[np.float64] = cv2.Sobel(
            gray, cv2.CV_64F, dx=1, dy=0, ksize=3
        ).astype(np.float64)
        grad_y: npt.NDArray[np.float64] = cv2.Sobel(
            gray, cv2.CV_64F, dx=0, dy=1, ksize=3
        ).astype(np.float64)

        # Матрица границ (для весов диффузии)
        edges: MaskArray = cv2.Canny(gray, 100, 200).astype(np.uint8)
        edge_weight: npt.NDArray[np.float64] = edges.astype(np.float64) / 255.0

        # Инициализация векторного поля
        u: npt.NDArray[np.float64] = grad_x.copy()
        v: npt.NDArray[np.float64] = grad_y.copy()

        # Итеративное решение уравнения диффузии
        for _ in range(iterations):
            laplacian_u: npt.NDArray[np.float64] = cv2.Laplacian(u, cv2.CV_64F).astype(
                np.float64
            )
            laplacian_v: npt.NDArray[np.float64] = cv2.Laplacian(v, cv2.CV_64F).astype(
                np.float64
            )

            # Обновление векторного поля
            u = u + mu * laplacian_u - edge_weight * (u - grad_x)
            v = v + mu * laplacian_v - edge_weight * (v - grad_y)

        # Магнитуда финального векторного поля
        gvf_mag: npt.NDArray[np.float64] = np.sqrt(u**2 + v**2).astype(np.float64)
        gvf_mag_norm: FloatArray = (255 * gvf_mag / (np.max(gvf_mag) + 1e-8)).astype(
            np.float32
        )

        # Пороговая бинаризация
        mask: MaskArray
        _, mask_raw = cv2.threshold(
            gvf_mag_norm.astype(np.float32),
            50.0,  # Фиксированный порог для GVF-магнитуды
            255.0,
            cv2.THRESH_BINARY,
        )
        mask = mask_raw.astype(np.uint8)
        # Заполнение дыр для получения связной области
        mask = ndimage.binary_fill_holes(mask > 0).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time

        self._log_info(
            "gvf_contour_opencv",
            exec_time,
            {"mu": mu, "iterations": iterations, **kwargs},
        )

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_morphological_snakes(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Сегментация морфологическими змеями (Morphological Snakes).

        Итеративный метод, который расширяет или сужает начальную бинарную маску
        на основе величины градиента изображения. Области с низким градиентом
        "поглощаются" контуром, с высоким — отбрасываются. Метод не требует
        вычисления энергии или решения дифференциальных уравнений.

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Инициализация маски как окружности в центре изображения.
        3. Для каждой итерации:
        - Вычисление градиента изображения через операторы Собеля.
        - Бинаризация градиента по фиксированному порогу.
        - Обновление маски: `mask = mask AND NOT(edges)`.
        - Морфологическое замыкание и открытие для сглаживания контура.
        4. Возврат финальной бинарной маски.

        Метод особенно эффективен для:
        - Медицинских изображений с чёткими границами органов
        - Микроскопических изображений клеток и тканей
        - Изображений с объектами, близкими по форме к инициализированному контуру

        Args:
            img: Входное изображение (RGB или grayscale).
            **kwargs: Дополнительные параметры:
                - `iterations` (int): Количество итераций деформации контура [10, 200].
                По умолчанию 50. Больше итераций → точнее результат, но медленнее.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                где 255 = область внутри замкнутого контура (объект), 0 = фон.

        Note:
            - Данная реализация использует упрощённый подход через морфологические
            операции вместо полной энергетической оптимизации. Для точной
            сегментации рассмотрите библиотеки `scikit-image` или `OpenCV` с
            полной реализацией snakes.
            - Качество результата сильно зависит от инициализации маски. Для сложных
            объектов может потребоваться ручная инициализация или использование
            нескольких начальных контуров.
            - Метод чувствителен к шуму; для зашумлённых изображений рекомендуется
            предварительное сглаживание (Гауссово размытие).
            - Параметр `iterations` следует подбирать экспериментально:
            * Простые объекты: 10–30
            * Сложные объекты: 50–100
            * Очень сложные: 100–200
            - Фиксированный порог градиента (50) может требовать адаптации под
            конкретное изображение; для универсальности рассмотрите адаптивный порог.

        Example:
            ```python
            # Базовое использование для медицинского изображения
            segmenter = OpenCVSegmenter("morphological_snakes", iterations=50)
            mask = segmenter.segment(medical_image)

            # Для объектов сложной формы
            segmenter = OpenCVSegmenter("morphological_snakes", iterations=100)
            mask = segmenter.segment(complex_object_image)

            # С предварительным сглаживанием для шумных изображений
            blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
            segmenter = OpenCVSegmenter("morphological_snakes", iterations=75)
            mask = segmenter.segment(blurred)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        h: int = gray.shape[0]
        w: int = gray.shape[1]

        # Начальная маска (окружность в центре)
        mask: MaskArray = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 4, 255, -1)

        iterations: int = int(self.params.get("iterations", 50))
        kernel: npt.NDArray[np.uint8] = np.ones((3, 3), dtype=np.uint8)

        for _ in range(iterations):
            # Градиент изображения
            grad_x: npt.NDArray[np.float64] = cv2.Sobel(
                gray, cv2.CV_64F, 1, 0, ksize=3
            ).astype(np.float64)
            grad_y: npt.NDArray[np.float64] = cv2.Sobel(
                gray, cv2.CV_64F, 0, 1, ksize=3
            ).astype(np.float64)
            grad_mag: npt.NDArray[np.float64] = np.sqrt(grad_x**2 + grad_y**2)
            grad_mag_norm: MaskArray = (
                255 * grad_mag / (np.max(grad_mag) + 1e-8)
            ).astype(np.uint8)
            grad_binary: MaskArray
            _, grad_binary_raw = cv2.threshold(
                grad_mag_norm.astype(np.float32), 50.0, 255.0, cv2.THRESH_BINARY
            )
            grad_binary = grad_binary_raw.astype(np.uint8)

            # Расширение/сужение на основе градиента
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(grad_binary)).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(np.uint8)

        exec_time: float = time.time() - start_time

        self._log_info(
            "morphological_snakes_opencv",
            exec_time,
            {"iterations": iterations, **kwargs},
        )
        print(f"Info after OpenCV_morphological_snakes: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_chan_vese(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Модель Чан-Везе — активные контуры без градиентов.

        Энергетическая модель, разделяющая изображение на две области с минимальной
        внутрирегиональной дисперсией. В отличие от классических активных контуров,
        метод Чан-Везе не зависит от градиента и эффективно работает с объектами,
        имеющими размытые или отсутствующие границы.

        Энергетический функционал:
        ```
        E = μ·Length(C) + λ₁·∫_inside(C) |I - c₁|² + λ₂·∫_outside(C) |I - c₂|²
        ```
        где:
        - `C` — контур, разделяющий изображение
        - `c₁, c₂` — средние интенсивности внутри и снаружи контура
        - `μ, λ₁, λ₂` — весовые коэффициенты

        Алгоритм:
        1. Конвертация в grayscale при необходимости.
        2. Инициализация контура как прямоугольника в центре изображения.
        3. Итеративная оптимизация:
        - Вычисление средних интенсивностей внутри и снаружи контура
        - Обновление контура на основе разности с этими средними
        - Сглаживание контура морфологическими операциями
        4. Возврат бинарной маски внутри финального контура.

        Метод особенно эффективен для:
        - Медицинских изображений с размытыми границами органов
        - Микроскопических изображений клеток с неоднородной текстурой
        - Объектов без чётких границ, но с однородной внутренней областью

        Args:
            img: Входное изображение (grayscale предпочтительно).
            **kwargs: Дополнительные параметры:
                    - `mu` (float): Вес длины контура [0.0, 5.0]. По умолчанию 0.25.
                    Меньшие значения → более гибкий контур.
                    - `iterations` (int): Количество итераций оптимизации [10, 1000].
                    По умолчанию 100. Больше итераций → точнее результат.
                    - `lambda1` (float): Вес внутренней области [0.1, 10.0]. По умолчанию 1.0.
                    - `lambda2` (float): Вес внешней области [0.1, 10.0]. По умолчанию 1.0.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = область внутри контура (объект), 0 = фон.

        Note:
            - Метод не требует инициализации контура близко к объекту, но качество
            результата может зависеть от начального положения.
            - Параметр `mu` контролирует гладкость контура: меньшие значения позволяют
            контуру следовать за сложными границами, но могут привести к шуму.
            - Для зашумлённых изображений рекомендуется предварительное сглаживание.
            - Вычислительная сложность: O(iterations·N), где N — количество пикселей;
            для больших изображений рассмотрите предварительный ресайз.

        Example:
            ```python
            # Базовое использование для медицинского изображения
            segmenter = OpenCVSegmenter("chan_vese", mu=0.25, iterations=100)
            mask = segmenter.segment(medical_image)

            # Для объектов со сложными границами
            segmenter = OpenCVSegmenter("chan_vese", mu=0.1, iterations=200)
            mask = segmenter.segment(complex_boundary_image)

            # Для зашумлённых изображений
            blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=1.0)
            segmenter = OpenCVSegmenter("chan_vese", mu=0.3, iterations=150)
            mask = segmenter.segment(blurred)
            ```
        """
        # Конвертация в grayscale при необходимости
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        start_time: float = time.time()

        h: int = gray.shape[0]
        w: int = gray.shape[1]

        # Получение параметров
        iterations: int = int(self.params.get("max_iter", 100))
        mu: float = float(self.params.get("mu", 0.25))

        # Инициализация маски (центральная область)
        mask: MaskArray = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(
            mask,
            (w // 4, h // 4),
            (3 * w // 4, 3 * h // 4),
            255,
            -1,  # Заполненный прямоугольник
        )

        # Итеративная оптимизация
        for _ in range(iterations):
            # Вычисление средних интенсивностей
            inside_mask: npt.NDArray[np.bool_] = mask > 0
            outside_mask: npt.NDArray[np.bool_] = ~inside_mask

            inside_mean: float = (
                float(np.mean(gray[inside_mask].astype(np.float32)))
                if np.any(inside_mask)
                else 0.0
            )
            outside_mean: float = (
                float(np.mean(gray[outside_mask].astype(np.float32)))
                if np.any(outside_mask)
                else 0.0
            )

            # Обновление маски на основе разности со средними
            diff_inside: FloatArray = np.abs(gray.astype(np.float32) - inside_mean)
            diff_outside: FloatArray = np.abs(gray.astype(np.float32) - outside_mean)

            new_mask: MaskArray = np.zeros_like(mask)
            new_mask[diff_inside < diff_outside] = 255

            # Сглаживание морфологическими операциями
            kernel: npt.NDArray[np.uint8] = np.ones((3, 3), dtype=np.uint8)
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel).astype(
                np.uint8
            )
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel).astype(
                np.uint8
            )

            mask = new_mask

        exec_time: float = time.time() - start_time

        self._log_info(
            "chan_vese_opencv",
            exec_time,
            {"mu": mu, "iterations": iterations, **kwargs},
        )
        print(f"Info after OpenCV_chan_vese: {self._log_info}")

        return mask

    # ============ WATERSHED И ГРАФОВЫЕ ============
    # ──────────────────────────────────────────────────────────────────────
    def _opencv_watershed(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Сегментация методом водораздела (Watershed).

        Алгоритм интерпретирует изображение как топографическую поверхность, где
        интенсивность пикселя — это высота. "Затопление" начинается от маркеров
        переднего плана и фона; границы между "водоёмами" становятся границами сегментов.

        Алгоритм:
        1. Конвертация в grayscale и бинаризация через Оцу.
        2. Морфологическое открытие для удаления шума.
        3. Выделение "уверенного фона" через дилатацию.
        4. Выделение "уверенного переднего плана" через преобразование расстояния.
        5. Определение "неизвестной области" как разницы между фоном и передним планом.
        6. Создание маркеров: уникальные метки для переднего плана, 0 для неизвестной области.
        7. Применение `cv2.watershed` для распространения меток.
        8. Создание бинарной маски из всех сегментов, кроме фона (метка 1).

        Args:
            img: Входное изображение (RGB или grayscale).
            **kwargs: Дополнительные параметры (не используются в базовой реализации).

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = все сегментированные объекты, 0 = фон.

        Note:
            - Метод требует чёткого разделения переднего плана и фона на этапе маркеров.
            Для сложных изображений может потребоваться ручная разметка маркеров.
            - Границы между сегментами помечаются значением `-1` в массиве маркеров;
            в данной реализации они включаются в маску объектов.
            - Для разделения слипшихся объектов метод очень эффективен, но чувствителен
            к параметрам морфологических операций (размер ядра, количество итераций).
            - Если изображение уже бинарное, можно пропустить шаги 1-2 и сразу перейти
            к вычислению маркеров.

        Example:
            ```python
            # Сегментация клеток на микроскопическом изображении
            segmenter = OpenCVSegmenter("watershed")
            mask = segmenter.segment(microscopy_image)

            # С предварительной морфологической обработкой для улучшения маркеров
            kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
            segmenter = OpenCVSegmenter("watershed")
            mask = segmenter.segment(opened)
            ```
        """
        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img

        start_time: float = time.time()

        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Бинаризация
        # Шаг 1: Бинаризация через Оцу
        binary: MaskArray
        _, binary_raw = cv2.threshold(
            gray, 0.0, 255.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        binary = binary_raw.astype(np.uint8)

        # Шаг 2: Морфологическое открытие для удаления шума
        kernel: npt.NDArray[np.uint8] = np.ones((3, 3), np.uint8)
        opening_raw = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        opening: MaskArray = opening_raw.astype(np.uint8)

        # Шаг 3: "Уверенный фон" — дилатация бинарного изображения
        sure_bg: MaskArray = cv2.dilate(opening, kernel, iterations=3).astype(np.uint8)

        # Шаг 4: "Уверенный передний план" — преобразование расстояния + порог
        dist_transform: FloatArray = cv2.distanceTransform(
            opening, distanceType=cv2.DIST_L2, maskSize=5
        ).astype(np.uint8)
        sure_fg_raw: npt.NDArray[np.uint8]
        _, sure_fg_raw_raw = cv2.threshold(
            dist_transform,
            0.7 * dist_transform.max(),  # порог 70% от максимума
            255.0,
            0,
        )
        sure_fg_raw = sure_fg_raw_raw.astype(np.uint8)
        sure_fg: MaskArray = sure_fg_raw.astype(np.uint8)

        # Шаг 5: "Неизвестная область"
        unknown_raw = cv2.subtract(sure_bg, sure_fg)
        unknown: MaskArray = unknown_raw.astype(np.uint8)

        # Шаг 6: Создание маркеров
        markers: npt.NDArray[np.int32]
        _, markers_raw = cv2.connectedComponents(sure_fg)
        markers = markers_raw.astype(np.int32)
        markers = markers + 1  # сдвиг, чтобы фон имел метку 1, а не 0
        markers[unknown == 255] = 0  # неизвестная область = 0

        # Шаг 7: Применение Watershed
        if len(img.shape) == 3:
            markers_raw = cv2.watershed(img, markers)
            markers = markers_raw.astype(np.uint8)
        else:
            color_img_raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            color_img: ImageArray = color_img_raw.astype(np.uint8)  # type: ignore[assignment]
            markers_raw = cv2.watershed(color_img, markers)
            markers = markers_raw.astype(np.uint8)

        # Шаг 8: Создание бинарной маски (все сегменты, кроме фона (метка 1))
        mask: MaskArray = (markers > 1).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time

        self._log_info(
            "watershed_opencv",
            exec_time,
            {**kwargs},
        )
        print(f"Info after OpenCV_watershed: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_random_walker(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Сегментация методом Random Walker.

        Алгоритм на основе теории графов: каждый пиксель изображения рассматривается
        как узел графа, а рёбра взвешиваются по схожести интенсивности соседних
        пикселей. Случайное блуждание начинается от маркеров (помеченных пикселей),
        и каждый пиксель присваивается тому маркеру, до которого блуждание доходит
        с наибольшей вероятностью.

        Алгоритм:
        1. Конвертация в grayscale и нормализация к [0, 1].
        2. Автоматическая инициализация маркеров:
        - Фон (метка 1): угловые области изображения
        - Объект (метка 2): центральная область
        3. Применение `skimage.segmentation.random_walker` с параметрами `beta` и `mode`.
        4. Создание бинарной маски: пиксели с меткой 2 = объект.
        5. Fallback на Watershed при ошибке выполнения Random Walker.

        Метод особенно эффективен для:
        - Изображений с плавными переходами между объектом и фоном
        - Медицинских изображений с размытыми границами тканей
        - Задач, где важна мягкая, вероятностная сегментация

        Args:
            img: Входное изображение (RGB или grayscale).
            **kwargs: Дополнительные параметры:
                - `beta` (float): Параметр сглаживания графа [10, 500].
                По умолчанию 130. Меньшие значения → более "жёсткие" границы.
                - `mode` (str): Режим решения системы: 'cg_j' (по умолчанию), 'cg', 'bf'.
                'cg_j' — сопряжённые градиенты с предобуславливателем (быстрее).

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                где 255 = пиксели, присвоенные маркеру объекта (метка 2).

        Note:
            - Метод требует установки `scikit-image` для функции `random_walker`.
            - Автоматическая инициализация маркеров подходит для объектов в центре
            изображения; для сложных сцен рассмотрите ручную разметку маркеров.
            - Параметр `beta` контролирует чувствительность к градиентам:
            * Малые β (10–50) → игнорирование слабых границ, крупные регионы
            * Большие β (200–500) → чувствительность к мелким перепадам, фрагментация
            - Режимы решения:
            * 'cg_j' — быстрый, подходит для большинства задач
            * 'cg' — более точный, но медленнее
            * 'bf' — точный, но очень медленный для больших изображений
            - При ошибке выполнения метод автоматически переключается на Watershed
            с теми же маркерами для обеспечения устойчивости.

        Example:
            ```python
            # Базовое использование для изображения с объектом в центре
            segmenter = OpenCVSegmenter("random_walker", beta=130, mode="cg_j")
            mask = segmenter.segment(image)

            # Для изображений с чёткими границами (меньший beta)
            segmenter = OpenCVSegmenter("random_walker", beta=50)
            mask = segmenter.segment(sharp_boundaries_image)

            # Для изображений с плавными переходами (больший beta)
            segmenter = OpenCVSegmenter("random_walker", beta=300)
            mask = segmenter.segment(smooth_transitions_image)
            ```
        """
        from skimage.segmentation import random_walker as sk_random_walker

        if len(img.shape) == 3:
            gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray: GrayImage = gray_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            gray = img.copy()

        start_time: float = time.time()

        h: int = gray.shape[0]
        w: int = gray.shape[1]
        gray_norm: FloatArray = gray.astype(np.float32) / 255.0

        # Создаём маркеры: 1=фон (углы), 2=объект (центр)
        markers: npt.NDArray[np.int32] = np.zeros((h, w), dtype=np.int32)
        markers[(h // 4) : (3 * h // 4), (w // 4) : (3 * w // 4)] = 2
        corner_size: int = min(h, w) // 8
        markers[:corner_size, :corner_size] = 1
        markers[:corner_size, -corner_size:] = 1
        markers[-corner_size:, :corner_size] = 1
        markers[-corner_size:, -corner_size:] = 1

        beta: float = float(self.params.get("beta", 130))
        mode: str = str(self.params.get("mode", "cg_j"))

        try:
            labels: npt.NDArray[np.int32] = sk_random_walker(
                gray_norm, markers, beta=beta, mode=mode
            )
            mask: MaskArray = (labels == 2).astype(np.uint8) * 255
        except Exception as e:
            warnings.warn(f"Random Walker failed: {e}. Falling back to Watershed.")
            # Fallback: используем Watershed с теми же маркерами
            color_img_raw = (
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
            )
            color_img: ImageArray = color_img_raw.astype(np.uint8)  # type: ignore[assignment]
            ws_markers: npt.NDArray[np.int32] = markers.copy().astype(np.int32)
            cv2.watershed(color_img, ws_markers)
            mask = (ws_markers == 2).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "random_walker_opencv",
            exec_time,
            {"beta": beta, "mode": mode, **kwargs},
        )
        print(f"Info after OpenCV_random_walker: {self._log_info}")

        return mask

    # ============ SUPER-PIXEL МЕТОДЫ ============
    # ──────────────────────────────────────────────────────────────────────
    def _opencv_quickshift(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Quickshift сегментация — mode-seeking алгоритм в пространстве (цвет + координаты).

        Быстрый непараметрический метод кластеризации, который группирует пиксели
        на основе плотности в комбинированном пространстве признаков: цвет (RGB) +
        пространственные координаты. В отличие от K-Means, не требует указания
        количества кластеров и автоматически определяет их число.

        Алгоритм:
        1. Конвертация grayscale в 3-канальное при необходимости.
        2. Преобразование цвета BGR → RGB и нормализация к [0, 1].
        3. Применение `skimage.segmentation.quickshift` с параметрами:
        - `kernel_size`: размер ядра для оценки плотности
        - `max_dist`: максимальное расстояние для объединения пикселей
        - `ratio`: баланс между цветом и пространством
        - `sigma`: сглаживание перед кластеризацией
        4. Определение фона как самого крупного сегмента.
        5. Создание бинарной маски: всё кроме фона = объект.

        Метод особенно эффективен для:
        - Изображений с чёткими цветовыми границами (иллюстрации, графика)
        - Медицинских изображений с однородными тканями разного цвета
        - Предварительной сегментации для более сложных методов

        Args:
            img: Входное изображение. Поддерживаются:
                - Grayscale: `(H, W)` → автоматически конвертируется в 3-канальное
                - RGB/BGR: `(H, W, 3)`, dtype=uint8
            **kwargs: Дополнительные параметры:
                - `kernel_size` (int): Размер ядра для оценки плотности [1, 10].
                По умолчанию 3. Меньшие значения → более детальная сегментация.
                - `max_dist` (float): Максимальное расстояние для объединения [1, 20].
                По умолчанию 6. Меньшие значения → больше, но меньших сегментов.
                - `ratio` (float): Баланс цвет/пространство [0.0, 1.0].
                По умолчанию 0.5. Меньшие значения → больше веса цвету.
                - `sigma` (float): Сигма Гауссова сглаживания [0.0, 5.0].
                По умолчанию 0.0 (без сглаживания).

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                где 255 = все сегменты кроме самого крупного (предполагаемый объект).

        Note:
            - Метод требует установки `scikit-image` для функции `quickshift`.
            - Параметр `ratio` контролирует баланс между цветовой и пространственной
            близостью: меньшие значения → сегменты более однородны по цвету,
            но могут быть разрозненными в пространстве.
            - Для цветных изображений метод работает в пространстве RGB; для лучшей
            перцептивной однородности рассмотрите предварительную конвертацию в Lab.
            - Вычислительная сложность: O(N·kernel_size²), что может быть медленно
            для больших изображений (>1000×1000); рассмотрите предварительный ресайз.
            - Самый крупный сегмент считается фоном; для задач с несколькими объектами
            рассмотрите возврат меток сегментов вместо бинарной маски.

        Example:
            ```python
            # Базовое использование для цветного изображения
            segmenter = OpenCVSegmenter("quickshift", kernel_size=3, max_dist=6, ratio=0.5)
            mask = segmenter.segment(color_image)

            # Для детальной сегментации мелких объектов
            segmenter = OpenCVSegmenter("quickshift", kernel_size=2, max_dist=4, ratio=0.3)
            mask = segmenter.segment(fine_details_image)

            # Для крупных однородных областей с сглаживанием
            segmenter = OpenCVSegmenter("quickshift", kernel_size=5, max_dist=10, sigma=1.0)
            mask = segmenter.segment(large_regions_image)
            ```
        """
        from skimage.segmentation import quickshift as sk_quickshift

        if len(img.shape) == 2:
            img_rgb: ImageArray = np.stack([img] * 3, axis=-1)
        else:
            img_rgb_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb_raw.astype(np.uint8)  # type: ignore[assignment]

        start_time: float = time.time()

        kernel_size: int = int(self.params.get("kernel_size", 3))
        max_dist: float = float(self.params.get("max_dist", 6))
        ratio: float = float(self.params.get("ratio", 0.5))
        sigma: float = float(self.params.get("sigma", 0.0))

        img_float: FloatArray = img_rgb.astype(np.float32) / 255.0

        segments: npt.NDArray[np.int32] = sk_quickshift(
            img_float,
            kernel_size=kernel_size,
            max_dist=max_dist,
            ratio=ratio,
            sigma=sigma,
        )

        unique: npt.NDArray[np.int32]
        counts: npt.NDArray[np.int32]
        unique, counts_raw = np.unique(segments, return_counts=True)
        counts = counts_raw.astype(np.int32)
        bg_label: int = int(unique[np.argmax(counts)])

        mask: MaskArray = (segments != bg_label).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "quickshift_opencv",
            exec_time,
            {
                "kernel_size": kernel_size,
                "max_dist": max_dist,
                "ratio": ratio,
                "sigma": sigma,
                **kwargs,
            },
        )
        print(f"Info after OpenCV_quickshift: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_slic(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        SLIC (Simple Linear Iterative Clustering) — суперпиксельная сегментация.

        Группирует пиксели в компактные, однородные регионы (суперпиксели) на основе
        пространственной и цветовой близости в пространстве признаков (Lab + координаты).

        Алгоритм:
        1. Конвертация изображения в цветовое пространство Lab (если нужно).
        2. Инициализация центров суперпикселей на равномерной сетке.
        3. Для каждой итерации:
        - Назначение пикселей ближайшему центру в пространстве признаков.
        - Пересчёт центров как средних значений назначенных пикселей.
        4. Пост-обработка: обеспечение связности суперпикселей.
        5. Определение фона как самого крупного суперпикселя.
        6. Создание бинарной маски: всё кроме фона = объект.

        Метод особенно эффективен для:
        - Изображений с плавными цветовыми переходами (портреты, пейзажи)
        - Медицинских изображений с однородными тканями
        - Предварительной обработки для более сложных методов сегментации

        Args:
            img: Входное изображение (RGB предпочтительно для цветовой сегментации).
            **kwargs: Дополнительные параметры:
                    - `region_size` (int): Приблизительный размер суперпикселя в пикселях [5, 100].
                    По умолчанию 20. Меньшие значения → больше суперпикселей, более детальная сегментация.
                    - `ruler` (float): Компактность суперпикселей [0.1, 50.0].
                    По умолчанию 10.0. Меньшие значения → более компактные (квадратные) суперпиксели.
                    - `num_iterations` (int): Количество итераций кластеризации [1, 50].
                    По умолчанию 10. Больше итераций → более точная кластеризация.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                    где 255 = все суперпиксели кроме самого крупного (предполагаемый объект).

        Note:
            - Метод требует наличия `opencv-contrib-python` для доступа к `cv2.ximgproc`.
            При отсутствии модуля используется fallback на K-Means.
            - Параметр `region_size` следует подбирать в зависимости от размера объектов:
            * Мелкие объекты: 5–15
            * Средние объекты: 20–40
            * Крупные объекты: 50–100
            - Параметр `ruler` контролирует баланс между цветовой и пространственной близостью:
            * Меньшие значения → более компактные суперпиксели (важна геометрия)
            * Большие значения → более однородные по цвету суперпиксели (важен цвет)
            - Для цветных изображений метод работает в пространстве Lab, что обеспечивает
            лучшую перцептивную однородность по сравнению с RGB.

        Example:
            ```python
            # Базовое использование для цветного изображения
            segmenter = OpenCVSegmenter("slic", region_size=20, ruler=10.0)
            mask = segmenter.segment(color_image)

            # Для детальной сегментации мелких объектов
            segmenter = OpenCVSegmenter("slic", region_size=10, ruler=5.0)
            mask = segmenter.segment(fine_details_image)

            # Для крупных однородных областей
            segmenter = OpenCVSegmenter("slic", region_size=50, ruler=20.0)
            mask = segmenter.segment(large_regions_image)
            ```
        """
        # Приведение к BGR для cv2.ximgproc
        if len(img.shape) == 2:
            img_bgr_raw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_bgr: ImageArray = img_bgr_raw.astype(np.uint8)  # type: ignore[assignment]
        else:
            img_bgr_raw = (
                img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            )
            img_bgr = img_bgr_raw.astype(np.uint8)  # type: ignore[assignment]

        start_time: float = time.time()

        # Получение параметров
        region_size: int = int(self.params.get("region_size", 20))
        ruler: float = float(self.params.get("ruler", 10.0))
        num_iterations: int = int(self.params.get("num_iterations", 10))

        # Проверка наличия opencv-contrib-python
        if not hasattr(cv2, "ximgproc"):
            warnings.warn(
                "cv2.ximgproc не доступен. Установите opencv-contrib-python "
                "или используйте альтернативный метод.",
                RuntimeWarning,
            )
            # Fallback: возврат пустой маски или использование K-Means
            return (
                np.zeros_like(img)
                if len(img.shape) == 2
                else np.zeros(img.shape[:2], dtype=np.uint8)
            )

        try:
            # Создание объекта SLIC
            slic = cv2.ximgproc.createSuperpixelSLIC(
                img_bgr,
                algorithm=cv2.ximgproc.SLIC,
                region_size=region_size,
                ruler=ruler,
            )
            # Итеративная кластеризация
            slic.iterate(num_iterations)
            # Получение меток суперпикселей
            labels: npt.NDArray[np.int32] = slic.getLabels()
        except AttributeError:
            # Fallback: ximgproc не установлен — используем K-Means как аппроксимацию
            warnings.warn(
                "cv2.ximgproc не доступен. Используем K-Means как аппроксимацию SLIC."
            )
            return self._opencv_kmeans_segmentation(img, **kwargs)

        # Определение фона как самого крупного суперпикселя
        unique: npt.NDArray[np.int32]
        counts: npt.NDArray[np.int32]
        unique, counts_raw = np.unique(labels, return_counts=True)
        counts = counts_raw.astype(np.int32)
        bg_label: int = int(unique[np.argmax(counts)])

        # Создание бинарной маски: всё кроме фона = объект
        mask: MaskArray = (labels != bg_label).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time

        self._log_info(
            "slic_opencv",
            exec_time,
            {
                "region_size": region_size,
                "ruler": ruler,
                "num_iterations": num_iterations,
                **kwargs,
            },
        )
        print(f"Info after OpenCV_slic: {self._log_info}")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_felzenszwalb(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Алгоритм Felzenszwalb — иерархическая сегментация на основе графов.

        Оригинальный алгоритм, строящий сегментацию через минимальное остовное дерево
        (MST) в пространстве пикселей. Начинает с каждого пикселя как отдельного
        региона и итеративно объединяет соседние регионы, если внутреннее различие
        меньше межрегионального с учётом адаптивного порога.

        Алгоритм:
        1. Конвертация grayscale в 3-канальное при необходимости.
        2. Преобразование цвета BGR → RGB и нормализация к [0, 1].
        3. Применение `skimage.segmentation.felzenszwalb` с параметрами:
        - `scale`: порог для объединения регионов (больше → крупнее сегменты)
        - `sigma`: сглаживание перед построением графа
        - `min_size`: минимальный размер региона после пост-обработки
        4. Определение фона как самого крупного сегмента.
        5. Создание бинарной маски: всё кроме фона = объект.

        Метод особенно эффективен для:
        - Изображений с объектами разного масштаба (от мелких деталей до крупных областей)
        - Естественных сцен с плавными цветовыми переходами (пейзажи, портреты)
        - Задач, где важна адаптивность к локальному контрасту

        Args:
            img: Входное изображение. Поддерживаются:
                - Grayscale: `(H, W)` → автоматически конвертируется в 3-канальное
                - RGB/BGR: `(H, W, 3)`, dtype=uint8
            **kwargs: Дополнительные параметры:
                - `scale` (float): Порог объединения регионов [10, 1000].
                По умолчанию 100. Меньшие значения → более детальная сегментация.
                - `sigma` (float): Сигма Гауссова сглаживания [0.0, 5.0].
                По умолчанию 0.8. Подавляет шум перед построением графа.
                - `min_size` (int): Минимальный размер региона [10, 500].
                По умолчанию 50. Меньшие значения → возможность выделения мелких объектов.

        Returns:
            MaskArray: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255},
                где 255 = все сегменты кроме самого крупного (предполагаемый объект).

        Note:
            - Метод требует установки `scikit-image` для функции `felzenszwalb`.
            - Параметр `scale` — ключевой для контроля детализации:
            * Мелкие объекты: 10–50
            * Средние объекты: 50–200
            * Крупные области: 200–1000
            - Параметр `sigma` следует подбирать в зависимости от уровня шума:
            * Чистые изображения: 0.0–0.5
            * Умеренный шум: 0.5–1.5
            * Сильный шум: 1.5–3.0
            - Параметр `min_size` удаляет мелкие "шумовые" сегменты после кластеризации;
            установите 0, если нужно сохранить все регионы.
            - Алгоритм имеет сложность O(N log N), что делает его эффективным для
            изображений среднего размера; для очень больших рассмотрите предварительный ресайз.

        Example:
            ```python
            # Базовое использование для естественной сцены
            segmenter = OpenCVSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50)
            mask = segmenter.segment(natural_scene_image)

            # Для детальной сегментации мелких объектов
            segmenter = OpenCVSegmenter("felzenszwalb", scale=50, sigma=0.5, min_size=20)
            mask = segmenter.segment(fine_details_image)

            # Для крупных однородных областей с подавлением шума
            segmenter = OpenCVSegmenter("felzenszwalb", scale=300, sigma=1.5, min_size=100)
            mask = segmenter.segment(large_regions_image)
            ```
        """
        from skimage.segmentation import felzenszwalb as sk_felzenszwalb

        if len(img.shape) == 2:
            img_rgb: ImageArray = np.stack([img] * 3, axis=-1)
        else:
            img_rgb_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb_raw.astype(np.uint8)  # type: ignore[assignment]

        start_time: float = time.time()

        scale: float = float(self.params.get("scale", 100))
        sigma: float = float(self.params.get("sigma", 0.8))
        min_size: int = int(self.params.get("min_size", 50))

        # Нормализуем к float [0,1] для skimage
        img_float: FloatArray = img_rgb.astype(np.float32) / 255.0

        segments: npt.NDArray[np.int32] = sk_felzenszwalb(
            img_float, scale=scale, sigma=sigma, min_size=min_size
        )

        # Находим самый большой сегмент (фон) и создаём маску
        unique: npt.NDArray[np.int32]
        counts: npt.NDArray[np.int32]
        unique, counts_raw = np.unique(segments, return_counts=True)
        counts = counts_raw.astype(np.int32)
        bg_label: int = int(unique[np.argmax(counts)])

        mask: MaskArray = (segments != bg_label).astype(np.uint8) * 255

        exec_time: float = time.time() - start_time
        self._log_info(
            "felzenszwalb_opencv",
            exec_time,
            {"scale": scale, "sigma": sigma, "min_size": min_size, **kwargs},
        )
        print(f"Info after OpenCV_felzenszwalb: {self._log_info}")

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

    # ──────────────────────────────────────────────────────────────────────
    def _opencv_grabcut(self, img: ImageArray, **kwargs: Any) -> MaskArray:
        """
        Интерактивная сегментация GrabCut.

        Алгоритм на основе графов и гауссовых смесей (GMM):
        1. Инициализация прямоугольником (фон/объект).
        2. Построение моделей цвета для фона и объекта.
        3. Итеративная оптимизация через минимизацию энергии.
        4. Финальная бинаризация.

        Требует указания прямоугольника, содержащего объект.

        Args:
            img: Входное изображение (RGB/BGR).
            **kwargs: Дополнительные параметры:
                     - `rect` (Tuple[int, int, int, int]): Прямоугольник (x, y, w, h).
                       По умолчанию: центральная область 50%×50%.
                     - `num_iterations` (int): Количество итераций оптимизации.
                       По умолчанию 5.

        Returns:
            MaskArray: Бинарная маска переднего плана (0/255).

        Note:
            - Метод чувствителен к качеству инициализации (прямоугольника).
            - Больше итераций → точнее результат, но медленнее.
            - После выполнения рекомендуется морфологическая пост-обработка.

        Example:
            ```python
            # Прямоугольник: (x=100, y=100, w=200, h=200)
            mask = segmenter._opencv_grabcut(
                image, rect=(100, 100, 200, 200), num_iterations=10
            )
            ```
        """
        start_time: float = time.time()

        rect: Optional[Tuple[int, int, int, int]] = self.params.get("rect", None)
        iter_count: int = int(self.params.get("num_iterations", 10))

        # Инициализация
        mask_init: MaskArray = np.zeros(img.shape[:2], dtype=np.uint8)
        bgd_model: FloatArray = np.zeros((1, 65), dtype=np.float64)  # type: ignore[var-annotated]
        fgd_model: FloatArray = np.zeros((1, 65), dtype=np.float64)  # type: ignore[var-annotated]

        # Если прямоугольник не задан, используем центральную часть
        if rect is None:
            h, w = img.shape[:2]
            rect = (int(w * 0.25), int(h * 0.25), int(w * 0.5), int(h * 0.5))

        # Выполнение GrabCut
        mask_grabcut, bgd_model, fgd_model = cv2.grabCut(  # type: ignore[assignment]
            img,
            mask_init,
            rect,
            bgd_model,
            fgd_model,
            iter_count,
            cv2.GC_INIT_WITH_RECT,
        )

        # Финальная маска
        mask_final: MaskArray = np.where(
            (mask_grabcut == cv2.GC_FGD) | (mask_grabcut == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Пост-обработка
        kernel: npt.NDArray[np.uint8] = np.ones((3, 3), np.uint8)
        mask_final = cv2.morphologyEx(
            mask_final, cv2.MORPH_CLOSE, kernel, iterations=2
        )  # type: ignore[assignment]
        mask_final = cv2.morphologyEx(
            mask_final, cv2.MORPH_OPEN, kernel, iterations=2
        )  # type: ignore[assignment]

        exec_time: float = time.time() - start_time

        self._log_info(
            "grabcut_opencv",
            exec_time,
            {"iterations": iter_count, "rect": rect, **kwargs},
        )
        print(f"Info after OpenCV_grabcut: {self._log_info}")

        return mask_final.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    "OpenCVSegmenter",
    "ImageArray",
    "MaskArray",
    "ParamsDict",
]
"""
Публичный API модуля.

Экспортируемые символы:
- `OpenCVSegmenter`: Основной класс сегментера.
- `ImageArray`, `MaskArray`: Type aliases для массивов.
- `ParamsDict`: Type alias для словаря параметров.
"""
