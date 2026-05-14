# segmenters/BaseSegmenter.py

"""Абстрактный базовый класс для всех методов сегментации изображений.

Модуль определяет унифицированный интерфейс (`BaseSegmenter`) для реализации
алгоритмов сегментации, обеспечивая совместимость между классическими методами
(пороговые, граничные, кластеризация) и нейросетевыми архитектурами.

Ключевые компоненты:
- 🎨 Типизация: Протоколы и TypeAlias для изображений, масок, метрик.
  • ImageInput: Union[str, np.ndarray, PIL.Image, torch.Tensor]
  • BinaryMask: np.ndarray формы (H, W), dtype uint8, значения {0, 255}
  • ProbabilityMask: np.ndarray формы (H, W), dtype float32, значения [0, 1]
  • MetricsDict: Dict[str, float] для результатов оценки качества

- 🔧 Абстрактные методы (требуют реализации в наследниках):
  • segment(image, **kwargs) → BinaryMask: Основная сегментация
  • segment_with_mask(image, **kwargs) → (BinaryMask, Optional[ProbabilityMask]): 
    Сегментация с возвратом вероятностной маски

- 🛠️ Готовые утилиты (доступны всем наследникам):
  • preprocess_image(): Конвертация входов → np.ndarray с опциями grayscale/resize/normalize
  • visualize(): Наложение маски на изображение с альфа-блендингом
  • evaluate_metrics(): Расчёт метрик через SegmentationMetrics
  • segment_and_evaluate(): Комбинированный вызов "сегментация + оценка"
  • _ensure_binary_mask(): Приведение масок к формату {0, 255}
  • get_info(): Мета-информация о сегментере

- 🔄 Гибкий вызов: Перегрузка __call__ позволяет использовать экземпляр как функцию:
  • seg(image) → BinaryMask
  • seg(image, return_mask=True) → (BinaryMask, ProbabilityMask | None)

Особенности реализации:
- 📦 Поддержка 4 форматов входа: путь к файлу, PIL.Image, np.ndarray, torch.Tensor
- 🎨 Конвертация цветовых пространств: RGB ↔ BGR ↔ GRAY через OpenCV/PIL
- 📐 Ресайз с адаптивной интерполяцией: INTER_AREA для уменьшения, INTER_LINEAR для увеличения
- 🎚️ Нормализация: опциональное приведение [0,255] → [0,1] для нейросетей
- 🛡️ Валидация: проверка размеров изображения и маски перед визуализацией/оценкой
- 🔍 Логирование: информативные сообщения об ошибках загрузки и обработки

Workflow для создания нового сегментера:
1. Наследовать класс: `class MySegmenter(BaseSegmenter):`
2. Реализовать абстрактные методы: `segment()` и `segment_with_mask()`
3. (Опционально) Переопределить `preprocess_image()` для специфичной предобработки
4. Использовать готовые утилиты: `visualize()`, `evaluate_metrics()`, `_ensure_binary_mask()`

Примечание:
- Все методы сегментации должны возвращать маску в формате `BinaryMask`: 
  форма `(H, W)`, dtype `uint8`, значения `{0, 255}` (0=фон, 255=объект).
- Для вероятностных выходов используйте тип `ProbabilityMask` и возвращайте `None`, 
  если метод не поддерживает вывод уверенности.
- Метрики рассчитываются через делегирование `SegmentationMetrics.calculate_all_metrics()` — 
  убедитесь в наличии этого модуля в проекте.
- Для массового тестирования используйте обёртки: `BatchClassicTester`, 
  `SegmentationTester`, `TorchImplementationValidator`.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

import torch
import cv2
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    TypeVar,
    Optional,
    Literal,
    Protocol,
    runtime_checkable,
    overload,
)
from typing_extensions import TypeAlias
from metrics.SegmentationMetrics import SegmentationMetrics

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

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
@runtime_checkable
class SegmentationMetricsProtocol(Protocol):
    """Протокол для класса метрик сегментации.

    Гарантирует наличие статического метода `calculate_all_metrics`.
    """

    @staticmethod
    def calculate_all_metrics(
        pred_mask: BinaryMask, gt_mask: BinaryMask, threshold: float
    ) -> MetricsDict:
        """Рассчитывает все метрики сегментации.

        Args:
            pred_mask: Предсказанная бинарная маска.
            gt_mask: Ground truth бинарная маска.
            threshold: Порог бинаризации.

        Returns:
            MetricsDict: Словарь с метриками.
        """
        ...


T = TypeVar("T", bound="BaseSegmenter")


# ──────────────────────────────────────────────────────────────────────
class BaseSegmenter(ABC):
    """Абстрактный базовый класс для всех методов сегментации.

    Определяет единый интерфейс для:
    - Сегментации (`segment()`, `segment_with_mask()`).
    - Предобработки изображений (`preprocess_image()`).
    - Визуализации результатов (`visualize()`).
    - Оценки качества (`evaluate_metrics()`, `segment_and_evaluate()`).

    Поддерживаемые форматы входных данных:
    - Строка: путь к файлу или URL.
    - `PIL.Image`: любое изображение.
    - `np.ndarray`: массив формы `(H, W)` или `(H, W, C)`.
    - `torch.Tensor`: тензор формы `(C, H, W)` или `(H, W, C)`.

    Attributes:
        name (str): Имя класса-наследника.
        metrics_calculator (SegmentationMetricsProtocol): Экземпляр для расчёта метрик.

    Example:
        ```python
        class MySegmenter(BaseSegmenter):
            def segment(self, image, **kwargs):
                # Реализация сегментации
                return binary_mask

            def segment_with_mask(self, image, **kwargs):
                # Реализация с возвратом маски
                return binary_mask, prob_mask
        ```
    """

    def __init__(self) -> None:
        """Инициализация базового сегментатора."""
        self.name: str = self.__class__.__name__
        self.metrics_calculator: SegmentationMetricsProtocol = SegmentationMetrics

    # ──────────────────────────────────────────────────────────────────────
    @abstractmethod
    def segment(self, image: ImageInput, **kwargs: Any) -> BinaryMask:
        """Основной метод сегментации.

        Args:
            image: Входное изображение в любом поддерживаемом формате.
            *args: Дополнительные позиционные аргументы.
            **kwargs: Дополнительные именованные аргументы.

        Returns:
            BinaryMask: Бинарная маска формы `(H, W)`, dtype `uint8`, значения {0, 255}.

        Raises:
            ValueError: Если не удалось загрузить или обработать изображение.
            TypeError: Если передан неподдерживаемый тип изображения.
        """
        pass

    # ──────────────────────────────────────────────────────────────────────
    @abstractmethod
    def segment_with_mask(
        self, image: ImageInput, **kwargs: Any
    ) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        """Сегментация с возвратом бинарной и вероятностной масок.

        Вероятностная маска может быть `None`, если метод не поддерживает
        вывод вероятностей.

        Args:
            image: Входное изображение.
            **kwargs: Дополнительные параметры.

        Returns:
            Tuple[BinaryMask, Optional[ProbabilityMask]]:
            - Бинарная маска: значения {0, 255}.
            - Вероятностная маска: значения [0, 1] или `None`.
        """
        pass

    # ──────────────────────────────────────────────────────────────────────
    def preprocess_image(
        self,
        image: ImageInput,
        as_gray: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
    ) -> NumpyImage:
        """Предобработка изображения к единому формату `np.ndarray`.

        Поддерживает:
        - Загрузку из файла/URL.
        - Конвертацию форматов (PIL → numpy, torch → numpy).
        - Конвертацию цветовых пространств (RGB/GRAY).
        - Ресайз к `target_size`.
        - Нормализацию значений в [0, 1].

        Args:
            image: Входное изображение.
            as_gray: Если `True`, конвертирует в оттенки серого.
            target_size: Целевой размер `(ширина, высота)`. Если `None`, размер не меняется.
            normalize: Если `True`, нормализует значения в [0, 1] (dtype float32).

        Returns:
            NumpyImage: Предобработанное изображение.
            - Если `normalize=False`: dtype `uint8`, диапазон [0, 255].
            - Если `normalize=True`: dtype `float32`, диапазон [0, 1].

        Raises:
            ValueError: Если не удалось загрузить изображение по пути/URL.
            TypeError: Если тип входных данных не поддерживается.
        """
        result: NumpyImage
        if isinstance(image, str):
            # Загрузка из файла
            if as_gray:
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Не удалось загрузить изображение: {image}")
                result = img  # Уже в GRAY
            else:
                img = cv2.imread(image)
                if img is None:
                    raise ValueError(f"Не удалось загрузить изображение: {image}")
                result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB
        elif isinstance(image, Image.Image):
            # PIL Image
            if as_gray:
                result = np.array(image.convert("L"), dtype=np.uint8)  # 'L' = grayscale
            else:
                result = np.array(image.convert("RGB"), dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            # NumPy array
            result = image.copy()
            if as_gray and len(image.shape) == 3:
                # Конвертируем RGB/BGR в GRAY
                if image.shape[2] == 3:
                    result = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif isinstance(image, torch.Tensor):
            # PyTorch tensor
            if image.dim() == 3:
                # Изменение порядка каналов из (C, H, W) в (H, W, C)
                if image.shape[0] in (1, 3):
                    img_np = image.permute(1, 2, 0).cpu().numpy()
                else:
                    img_np = image.cpu().numpy()
            else:
                img_np = image.cpu().numpy()

            # Приведение к uint8
            if img_np.dtype in (np.float32, np.float64):
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)

            result = img_np
            if as_gray and len(result.shape) == 3 and result.shape[2] == 3:
                result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        else:
            raise TypeError(
                f"Неподдерживаемый тип изображения: {type(image)}. "
                f"Ожидается один из: {ImageInput.__args__}"
            )
        if target_size is not None:
            interpolation: int = (
                cv2.INTER_AREA
                if result.shape[0] * result.shape[1] > target_size[0] * target_size[1]
                else cv2.INTER_LINEAR
            )
            result = cv2.resize(result, target_size, interpolation=interpolation)

        if normalize:
            if result.dtype != np.float32:
                result = result.astype(np.float32) / 255.0

        return result

    # ──────────────────────────────────────────────────────────────────────
    def visualize(
        self,
        image: NumpyImage,
        mask: BinaryMask,
        alpha: float = 0.5,
        overlay_color: OverlayColor = (255, 0, 0),
        return_numpy: bool = False,
    ) -> Union[PILImage, NumpyImage]:
        """Визуализация результата сегментации: наложение маски на оригинал.

        Алгоритм:
        1. Создаёт цветную маску (`overlay_color` для объекта, чёрный для фона).
        2. Блендит оригинал и цветную маску с коэффициентом `alpha`.

        Args:
            image: Исходное изображение в формате `(H, W, 3)`, RGB, dtype `uint8`.
            mask: Бинарная маска формы `(H, W)`, значения {0, 255}.
            alpha: Прозрачность наложения (0.0 = только фото, 1.0 = только маска).
            overlay_color: Цвет объекта в формате `(R, G, B)`.
            return_numpy: Если `True`, возвращает `np.ndarray`, иначе `PIL.Image`.

        Returns:
            Union[PILImage, NumpyImage]: Изображение с наложенной маской.

        Raises:
            ValueError: Если размеры изображения и маски не совпадают.
        """
        # Проверка размеров
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Размеры изображения {image.shape[:2]} и маски {mask.shape[:2]} не совпадают"
            )
        colored_mask: NumpyImage = np.zeros_like(image)
        colored_mask[mask > 0] = overlay_color

        # Наложение маски на изображение
        result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return result if return_numpy else Image.fromarray(result)

    # ──────────────────────────────────────────────────────────────────────
    def evaluate_metrics(
        self, pred_mask: BinaryMask, gt_mask: BinaryMask, threshold: float = 0.5
    ) -> MetricsDict:
        """Оценка качества сегментации с помощью различных метрик.

        Делегирует расчёт `SegmentationMetrics.calculate_all_metrics`.

        Args:
            pred_mask: Предсказанная маска.
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации (если маски не бинарные).

        Returns:
            MetricsDict: Словарь с метриками:
            - `iou`, `dice`, `f1_score`, `precision`, `recall`, `pixel_accuracy`, ...
        """
        # Приведение масок к бинарному формат
        pred_binary: BinaryMask = self._ensure_binary_mask(pred_mask, threshold)
        gt_binary: BinaryMask = self._ensure_binary_mask(gt_mask, threshold)

        return self.metrics_calculator.calculate_all_metrics(
            pred_binary, gt_binary, threshold
        )

    # ──────────────────────────────────────────────────────────────────────
    def segment_and_evaluate(
        self,
        image: ImageInput,
        gt_mask: BinaryMask,
        threshold: float = 0.5,
        **segment_kwargs: Any,
    ) -> Tuple[MetricsDict, BinaryMask]:
        """Выполняет сегментацию и сразу оценивает результат.

        Удобно для быстрого тестирования метода на одном изображении.

        Args:
            image: Входное изображение.
            gt_mask: Ground truth маска.
            threshold: Порог для бинаризации.
            **segment_kwargs: Дополнительные аргументы для `segment()`.

        Returns:
            Tuple[MetricsDict, BinaryMask]:
            - Метрики качества.
            - Предсказанная бинарная маска.
        """
        pred_mask: BinaryMask = self.segment(image, **segment_kwargs)
        pred_binary: BinaryMask = self._ensure_binary_mask(pred_mask, threshold)
        gt_binary: BinaryMask = self._ensure_binary_mask(gt_mask, threshold)
        metrics: MetricsDict = self.evaluate_metrics(pred_binary, gt_binary, threshold)
        return metrics, pred_binary

    # ──────────────────────────────────────────────────────────────────────
    # OVERLOAD ДЛЯ __call__
    # ──────────────────────────────────────────────────────────────────────
    @overload
    def __call__(
        self, image: ImageInput, return_mask: Literal[False] = False, **kwargs: Any
    ) -> BinaryMask: ...

    # ──────────────────────────────────────────────────────────────────────
    @overload
    def __call__(
        self, image: ImageInput, return_mask: Literal[True], **kwargs: Any
    ) -> Tuple[BinaryMask, Optional[ProbabilityMask]]: ...

    # ──────────────────────────────────────────────────────────────────────
    def __call__(
        self, image: ImageInput, return_mask: bool = False, **kwargs: Any
    ) -> Union[BinaryMask, Tuple[BinaryMask, Optional[ProbabilityMask]]]:
        """Вызов метода сегментации.

        Args:
            image: Входное изображение.
            return_mask: Возвращать ли дополнительную информацию о маске.
            **kwargs: Дополнительные аргументы для метода сегментации.

        Returns:
            - Если `return_mask=False`: `BinaryMask`.
            - Если `return_mask=True`: `Tuple[BinaryMask, Optional[ProbabilityMask]]`.
        """
        if return_mask:
            return self.segment_with_mask(image, **kwargs)
        return self.segment(image, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    def _ensure_binary_mask(
        self, mask: Union[BinaryMask, ProbabilityMask], threshold: float = 0.5
    ) -> BinaryMask:
        """Приведение маски к бинарному формату {0, 255}.

        Обрабатывает:
        - `uint8` с диапазоном [0, 1] → умножение на 255.
        - `uint8` с диапазоном [0, 255] → пороговая бинаризация.
        - `float32/64` с диапазоном [0, 1] → пороговая бинаризация.
        - `float32/64` с произвольным диапазоном → нормализация + бинаризация.

        Args:
            mask: Входная маска.
            threshold: Порог бинаризации (для вероятностных масок).

        Returns:
            BinaryMask: Бинарная маска формы `(H, W)`, dtype `uint8`, значения {0, 255}.
        """
        if mask.dtype == np.uint8:
            if mask.max() == 1:
                return (mask * 255).astype(np.uint8)
            elif mask.max() <= 255:
                return np.where(mask > threshold * 255, 255, 0).astype(np.uint8)
        elif mask.dtype in (np.float32, np.float64):
            if mask.max() <= 1.0:
                return np.where(mask > threshold, 255, 0).astype(np.uint8)
            else:
                normalized = mask / mask.max()
                return np.where(normalized > threshold, 255, 0).astype(np.uint8)
        return mask.astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────
    def get_info(self) -> Dict[str, Any]:
        """Возвращает мета-информацию о сегментаторе.

        Returns:
            Dict[str, Any]:
            ```python
            {
                "name": str,           # Имя экземпляра
                "class": str,          # Имя класса
                "module": str,         # Модуль класса
            }
            ```
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
        }
