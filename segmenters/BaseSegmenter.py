# segmenters/BaseSegmenter.py

# Импорт основных библиотек
import torch
import cv2
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
from typing import (
    List,
    Union,
    Tuple,
    Dict,
    Set,
    Any,
    TypeVar,
    Optional,
    Literal,
    Protocol,
    runtime_checkable,
    overload,
    TYPE_CHECKING,
)
from typing_extensions import TypeAlias
from metrics.SegmentationMetrics import SegmentationMetrics

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


@runtime_checkable
class SegmentationMetricsProtocol(Protocol):
    """Протокол для класса метрик сегментации"""

    @staticmethod
    def calculate_all_metrics(
        pred_mask: BinaryMask, gt_mask: BinaryMask, threshold: float
    ) -> MetricsDict: ...


T = TypeVar("T", bound="BaseSegmenter")


class BaseSegmenter(ABC):
    """Базовый класс для всех методов сегментации"""

    def __init__(self) -> None:
        self.name: str = self.__class__.__name__
        self.metrics_calculator: SegmentationMetricsProtocol = SegmentationMetrics
        self.metrics_calculator: SegmentationMetricsProtocol = SegmentationMetrics

    @abstractmethod
    def segment(self, image: ImageInput, *args: Any, **kwargs: Any) -> BinaryMask:
        """
        Основной метод сегментации

        Args:
            image: Входное изображение в любом поддерживаемом формате
            *args: Дополнительные позиционные аргументы
            **kwargs: Дополнительные именованные аргументы

        Returns:
            Бинарная маска сегментации (uint8, значения 0 или 255)

        Raises:
            ValueError: Если не удалось загрузить или обработать изображение
            TypeError: Если передан неподдерживаемый тип изображения
        """
        pass

    @abstractmethod
    def segment_with_mask(
        self, image: ImageInput, *args: Any, **kwargs: Any
    ) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        """
        Сегментация с возвратом маски и вероятностей (если доступно)

        Args:
            image: Входное изображение в любом поддерживаемом формате
            *args: Дополнительные позиционные аргументы
            **kwargs: Дополнительные именованные аргументы

        Returns:
            Кортеж (бинарная маска, вероятностная маска)
        """
        pass

    def preprocess_image(
        self,
        image: ImageInput,
        as_gray: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
    ) -> NumpyImage:
        """
        Предобработка изображения

        Args:
            image: Входное изображение
            as_gray: Конвертировать в оттенки серого
            target_size: Целевой размер (ширина, высота)
            normalize: Нормализовать значения пикселей в [0, 1]

        Returns:
            Предобработанное изображение в формате numpy array

        Raises:
            ValueError: Если не удалось загрузить изображение
            TypeError: Если передан неподдерживаемый тип изображения
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
            result = cv2.resize(
                result,
                target_size,
                interpolation=(
                    cv2.INTER_AREA
                    if result.shape[0] * result.shape[1]
                    > target_size[0] * target_size[1]
                    else cv2.INTER_LINEAR
                ),
            )

        if normalize:
            if result.dtype != np.float32:
                result = result.astype(np.float32) / 255.0

        return result

    def visualize(
        self,
        image: NumpyImage,
        mask: BinaryMask,
        alpha: float = 0.5,
        overlay_color: OverlayColor = (255, 0, 0),
        return_numpy: bool = False,
    ) -> Union[PILImage, NumpyImage]:
        """
        Визуализация результата сегментации

        Args:
            image: Исходное изображение (RGB)
            mask: Бинарная маска сегментации
            alpha: Прозрачность наложения (0-1)
            overlay_color: Цвет наложения (RGB)
            return_numpy: Вернуть результат как numpy array вместо PIL Image

        Returns:
            Изображение с визуализацией сегментации
        """
        # Проверка размеров
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Размеры изображения {image.shape[:2]} и маски {mask.shape[:2]} не совпадают"
            )
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = overlay_color

        # Наложение маски на изображение
        result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return result if return_numpy else Image.fromarray(result)

    def evaluate_metrics(
        self, pred_mask: BinaryMask, gt_mask: BinaryMask, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Оценка качества сегментации с помощью различных метрик

        Args:
            pred_mask: Предсказанная маска
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации

        Returns:
            Словарь с метриками сегментации
        """
        # Приведение масок к общему формату
        pred_binary: BinaryMask = self._ensure_binary_mask(pred_mask, threshold)
        gt_binary: BinaryMask = self._ensure_binary_mask(gt_mask, threshold)

        return self.metrics_calculator.calculate_all_metrics(
            pred_binary, gt_binary, threshold
        )

    def segment_and_evaluate(
        self,
        image: ImageInput,
        gt_mask: BinaryMask,
        threshold: float = 0.5,
        **segment_kwargs: Any,
    ) -> Tuple[MetricsDict, BinaryMask]:
        """
        Выполняет сегментацию и сразу оценивает результат

        Args:
            image: Входное изображение
            gt_mask: Ground truth маска
            threshold: Порог для бинаризации
            **segment_kwargs: Дополнительные аргументы для метода segment

        Returns:
            Кортеж (метрики, предсказанная маска)
        """
        pred_mask: BinaryMask = self.segment(image, **segment_kwargs)
        pred_binary = self._ensure_binary_mask(pred_mask, threshold)
        gt_binary = self._ensure_binary_mask(gt_mask, threshold)
        metrics: MetricsDict = self.evaluate_metrics(pred_binary, gt_binary, threshold)
        return metrics, pred_binary

    @overload
    def __call__(
        self, image: ImageInput, return_mask: Literal[False] = False, **kwargs: Any
    ) -> BinaryMask: ...

    @overload
    def __call__(
        self, image: ImageInput, return_mask: Literal[True], **kwargs: Any
    ) -> Tuple[BinaryMask, Optional[ProbabilityMask]]: ...

    def __call__(
        self, image: ImageInput, return_mask: bool = False, **kwargs: Any
    ) -> Union[BinaryMask, Tuple[BinaryMask, Optional[ProbabilityMask]]]:
        """
        Вызов метода сегментации

        Args:
            image: Входное изображение
            return_mask: Возвращать ли дополнительную информацию о маске
            **kwargs: Дополнительные аргументы для метода сегментации

        Returns:
            Если return_mask=False: бинарная маска
            Если return_mask=True: кортеж (бинарная маска, вероятностная маска)
        """
        if return_mask:
            return self.segment_with_mask(image, **kwargs)
        return self.segment(image, **kwargs)

    def _ensure_binary_mask(
        self, mask: Union[BinaryMask, ProbabilityMask], threshold: float = 0.5
    ) -> BinaryMask:
        """
        Приведение маски к бинарному формату

        Args:
            mask: Входная маска
            threshold: Порог бинаризации

        Returns:
            Бинарная маска (uint8, 0 или 255)
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

    def get_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о сегментаторе

        Returns:
            Словарь с информацией о классе
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
        }
