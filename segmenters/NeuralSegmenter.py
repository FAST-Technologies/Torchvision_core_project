# segmenters/NeuralSegmenter.py

"""Универсальный сегментатор с поддержкой множественных нейронных архитектур.

Поддерживает 18+ моделей из 4 источников:
1. **HuggingFace Transformers** (5): SegFormer, Mask2Former, OneFormer, DPT, UPerNet
2. **Segmentation Models PyTorch** (5): U-Net, FPN, PSPNet, DeepLabV3+, SegNet
3. **Torchvision** (3): DeepLabV3+, FCN, Mask R-CNN
4. **Instance Segmentation** (4): SAM, MobileSAM, SAM2, YOLOv8-seg

Ключевые особенности:
- ✅ Единый интерфейс: `.segment()`, `.segment_with_mask()`, `.segment_image()` для всех архитектур
- ✅ Авто-загрузка весов: из HF Hub, локальных чекпоинтов или YAML-конфига
- ✅ Поддержка палитр: ADE20K, COCO, Cityscapes, CheXpert, ISIC, binary
- ✅ Гибкий ввод: str (path/URL), PIL.Image, np.ndarray, torch.Tensor
- ✅ Делегирование инференса: через `utils.strategies.segment_image_unified`
- ✅ Метрики на лету: IoU, Dice, mIoU при передаче `gt_mask`

Типичный workflow:
```python
from segmenters.NeuralSegmenter import NeuralSegmenter

# 1. Загрузка предобученной модели
segmenter = NeuralSegmenter(
    model_type="segformer",
    model_name="nvidia/segformer-b5-finetuned-ade-640-640",
    device="cuda"
)

# 2. Базовая сегментация
mask = segmenter.segment("image.jpg")  # BinaryMask {0, 255}

# 3. Визуализация с палитрой
overlay = segmenter.segment_image("image.jpg", alpha=0.7)
overlay.save("result.png")

# 4. Детальный анализ с метриками
seg_map, info = segmenter.predict_segmentation_map(
    "image.jpg",
    gt_mask=ground_truth,
    class_names=NeuralSegmenter.get_ade_class_names(),
    verbose=True
)
print(f"IoU: {info['iou']:.3f}, mIoU: {info['miou']:.3f}")

# 5. Инстанс-сегментация (SAM с промптом)
sam = NeuralSegmenter(model_type="sam", model_name="mobile_sam.pt")
mask, info = sam.predict_segmentation_map(
    "object.jpg",
    point_coords=[[250, 300]],
    point_labels=[1]
)
```

Attributes:
    model_type_str (str): Строковый идентификатор типа модели.
    model_type (ModelType): Enum-тип модели.
    model_name (str): Имя модели в HF Hub или локальный путь.
    device (torch.device): Устройство для вычислений.
    num_classes (int): Количество выходных классов.
    palette (List[List[int]]): Цветовая палитра для визуализации.
    model (nn.Module): Загруженная модель в режиме `.eval()`.
    processor (Optional[Any]): HF-процессор для препроцессинга (или None).

Note:
    - Для HF-моделей `processor` обязателен для корректного препроцессинга.
    - Для SMP/torchvision `processor=None`; вход — сырой тензор `[B, 3, H, W]`.
    - Бинарная маска: всё, кроме класса 0 (фон), считается объектом (255).
    - При загрузке чекпоинта классификатор автоматически заменяется под `num_classes`.
    - Для SAM/YOLOv8 выходные маски могут требовать постобработки (NMS, фильтрация по score).
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

from segmenters.BaseSegmenter import BaseSegmenter, ImageInput, BinaryMask
from segmenters.NeuralModelFactory import NeuralModelFactory, ModelType, ModelTuple
from utils.strategies import segment_image_unified as infer_unified
from utils.palettes import (
    ade_palette,
    get_ade_class_names,
    get_coco_class_names,
    coco_palette,
    get_cityscapes_extended_class_names,
    cityscapes_extended_palette,
    get_cityscapes_class_names,
    cityscapes_palette,
)

from typing import List, Union, Tuple, Dict, Any, Optional, Literal, cast
import time
import requests
from io import BytesIO

from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import torch

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
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
TRANSFORMERS_AVAILABLE: bool = True
num_classes: int = 150

ValidModelType = Literal[
    "segformer",
    "mask2former",
    "oneformer",
    "dpt",
    "upernet",
    "deeplab_tv",
    "fcn_tv",
    "maskrcnn_tv",
    "unet_smp",
    "mit_smp",
    "fpn_mit",
    "psp_mit",
    "deeplab_smp",
    "segnet",
    "segnet_custom",
    "sam",
    "mobile_sam",
    "sam2",
    "yolov8",
    "yolov8n_seg",
    "yolov8s_seg",
    "yolov8m_seg",
]

# Алиасы для типов изображений
ImagePath = str
NumpyImage = np.ndarray
PILImage = Image.Image
TorchImage = torch.Tensor
DeviceStr = Union[torch.device, str]
ClassNamesDict = Optional[Dict[int, str]]
PaletteType = Optional[List[List[int]]]


# ──────────────────────────────────────────────────────────────────────
class NeuralSegmenter(BaseSegmenter):
    """Универсальный сегментатор с поддержкой множественных нейронных архитектур.

    Поддерживаемые модели:
    - HuggingFace Transformers: SegFormer, Mask2Former, OneFormer, DPT, UPerNet
    - Segmentation Models PyTorch: U-Net, FPN, PSPNet (с разными encoder'ами)
    - Torchvision: DeepLabV3+, FCN, Mask R-CNN
    - Instance segmentation: SAM, YOLOv8

    Особенности:
    - Автоматическая загрузка предобученных весов из HF Hub или локальных чекпоинтов.
    - Единый интерфейс `.segment()` / `.segment_with_mask()` для всех архитектур.
    - Поддержка различных палитр (ADE20K, COCO, Cityscapes) для визуализации.
    - Делегирование инференса стратегии из `utils.strategies.segment_image_unified`.

    Attributes:
        model_type_str (str): Строковый идентификатор типа модели.
        model_type (ModelType): Enum-тип модели.
        model_name (str): Имя модели в HF Hub или локальный путь.
        device (torch.device): Устройство для вычислений.
        num_classes (int): Количество выходных классов.
        palette (List[List[int]]): Цветовая палитра для визуализации.
        model (nn.Module): Загруженная модель в режиме `.eval()`.
        processor (Optional[Any]): HF-процессор для препроцессинга (или None).

    Example:
        ```python
        # Загрузка SegFormer B5
        segmenter = NeuralSegmenter(
            model_type="segformer",
            model_name="nvidia/segformer-b5-finetuned-ade-640-640",
            device="cuda",
        )
        mask = segmenter.segment("test.jpg")  # BinaryMask
        overlay = segmenter.segment_image("test.jpg", alpha=0.7)  # PIL.Image
        ```
    """

    def __init__(
        self,
        model_type: str = "segformer",
        model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        variant: Optional[str] = None,
        device: Optional[str] = None,
        local_path: Optional[str] = None,
        num_classes: int = num_classes,
        palette: PaletteType = None,
        checkpoint_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Инициализация нейросетевого сегментатора.

        Args:
            model_type: Тип модели (строка или значение ModelType enum).
            model_name: Имя модели в HuggingFace Hub (для HF-моделей).
            variant: Вариант модели (например, "b5" для SegFormer, "fcn_resnet50" для FCN).
            device: Устройство для вычислений ("cuda" или "cpu"). Если `None`, авто-определение.
            local_path: Локальный путь к модели (альтернатива model_name для HF).
            num_classes: Количество выходных классов (по умолчанию 150 для ADE20K).
            palette: Цветовая палитра для визуализации. Если `None`, используется дефолтная (ADE20K).
            checkpoint_path: Путь к чекпоинту .pth для SMP/torchvision-моделей.
            **kwargs: Дополнительные параметры для `NeuralModelFactory.create_model()`.

        Raises:
            ImportError: Если `transformers` не установлен для HF-моделей.
            ValueError: Если `model_type` не поддерживается фабрикой.
        """
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        self.model_type_str: str = model_type
        self.model_type: ModelType = ModelType(model_type)
        self.model_name: str = model_name
        self.local_path: Optional[str] = local_path
        self.variant: Optional[str] = variant
        self.device: torch.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.params: Dict[str, Any] = kwargs
        self.num_classes: int = int(num_classes)

        start_time: float = time.perf_counter()
        cp_path: str = (
            checkpoint_path if checkpoint_path is not None else "checkpoint.pth"
        )
        model_tuple: ModelTuple
        if variant is not None:
            # Загрузка из YAML-конфига
            model_tuple = NeuralModelFactory.create_model_from_config(
                model_type=model_type,
                variant=variant,
                device=cast(Literal["cuda", "cpu"], self.device),
                checkpoint_path=cp_path,
                **kwargs,
            )
        else:
            # Прямая загрузка
            model_tuple = NeuralModelFactory.create_model(
                model_type=self.model_type,
                model_name=model_name,
                local_path=local_path,
                checkpoint_path=cp_path,
                device=cast(Literal["cuda", "cpu"], self.device),
                num_classes=num_classes,
                **kwargs,
            )
        self.model, self.processor, self.model_type_str = model_tuple
        load_time: float = time.perf_counter() - start_time
        print(f"Модель загружена за {load_time:.4f} секунд")
        self.palette: Optional[List[List[int]]] = (
            palette if palette else self._get_default_palette()
        )

        # ──────────────────────────────────────────────────────────────
        # Логирование
        # ──────────────────────────────────────────────────────────────
        print("✅ Нейросетевая модель загружена!")
        print(f"   Тип: {self.model_type_str}")
        print(
            f"   Source: {self.local_path if self.local_path else (self.model_name if self.model_name else f'config:{variant}')}"
        )
        print(f"   Устройство: {self.device}")
        print(f"   Количество классов: {self.num_classes}")

        if hasattr(self.model, "config"):
            config = self.model.config
            if hasattr(config, "id2label") and isinstance(config.id2label, dict):
                id2label: Dict[Union[int, str], str] = config.id2label  # type: ignore[assignment]
                print(f"   Количество классов: {len(id2label)}")
                print("Текущие имена классов:")
                for class_id, class_name in id2label.items():
                    print(f"{class_id}: {class_name}")

    # ──────────────────────────────────────────────────────────────────────
    def _get_default_palette(self) -> List[List[int]]:
        """Возвращает палитру ADE20K по умолчанию.

        Returns:
            List[List[int]]: Список `[R, G, B]` для каждого класса (150 классов).
        """
        return ade_palette()

    # ──────────────────────────────────────────────────────────────────────
    # СТАТИЧЕСКИЕ МЕТОДЫ: ИМЕНА КЛАССОВ И ПАЛИТРЫ
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_ade_class_names() -> Dict[int, str]:
        """Возвращает словарь имён классов для датасета ADE20K.

        Returns:
            Dict[int, str]: `{class_id: class_name}` для 150 классов.
        """
        # ADE20K Class Names (0-indexed, 150 classes)
        # Source: http://sceneparsing.csail.mit.edu/
        return get_ade_class_names()

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def ade_palette() -> List[List[int]]:
        """Возвращает палитру ADE20K для визуализации.

        Returns:
            List[List[int]]: Список `[R, G, B]` для каждого класса.
        """
        return ade_palette()

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_coco_class_names() -> Dict[int, str]:
        """Возвращает словарь имён классов для датасета COCO.

        Returns:
            Dict[int, str]: `{class_id: class_name}` для 80 классов.
        """
        # COCO Class Names (0-indexed, 80 classes)
        # Source: https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
        return get_coco_class_names()

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def coco_palette() -> List[List[int]]:
        """Возвращает палитру COCO для визуализации.

        Returns:
            List[List[int]]: Список `[R, G, B]` для каждого класса.
        """
        return coco_palette()

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_cityscapes_extended_class_names() -> Dict[int, str]:
        """Возвращает расширенный словарь имён классов для Cityscapes (34 класса).

        Returns:
            Dict[int, str]: `{class_id: class_name}` для 34 классов.
        """
        # Cityscapes Extended (34 classes - includes "grouped" categories)
        return get_cityscapes_extended_class_names()

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def cityscapes_extended_palette() -> List[List[int]]:
        """Возвращает расширенную палитру Cityscapes для визуализации.

        Returns:
            List[List[int]]: Список `[R, G, B]` для каждого класса.
        """
        return cityscapes_extended_palette()

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_cityscapes_class_names() -> Dict[int, str]:
        """Возвращает стандартный словарь имён классов для Cityscapes (19 классов).

        Returns:
            Dict[int, str]: `{class_id: class_name}` для 19 классов.
        """
        # Cityscapes Class Names (0-indexed, 19 classes for semantic segmentation)
        # Source: https://www.cityscapes-dataset.com/
        return get_cityscapes_class_names()

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def cityscapes_palette() -> List[List[int]]:
        """Возвращает стандартную палитру Cityscapes для визуализации.

        Returns:
            List[List[int]]: Список `[R, G, B]` для каждого класса.
        """
        return cityscapes_palette()

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_chexpert_observation_class_names() -> Dict[int, str]:
        """Возвращает словарь имён классов для CheXpert (14 наблюдений).

        Returns:
            Dict[int, str]: `{class_id: class_name}` для 14 классов.
        """
        # CheXpert Observation Classes (14 labels for classification)
        # Source: https://stanfordmlgroup.github.io/competitions/chexpert/
        chexpert_observation_names: Dict[int, str] = {
            0: "No Finding",
            1: "Enlarged Cardiomediastinum",
            2: "Cardiomegaly",
            3: "Lung Opacity",
            4: "Lung Lesion",
            5: "Edema",
            6: "Consolidation",
            7: "Pneumonia",
            8: "Atelectasis",
            9: "Pneumothorax",
            10: "Pleural Effusion",
            11: "Pleural Other",
            12: "Fracture",
            13: "Support Devices",
        }

        # Для сегментации лёгких (если есть маски):
        chest_segmentation_class_names: Dict[int, str] = {
            0: "background",  # Non-lung area
            1: "lung",  # Lung field (left + right)
        }

        # Проверка
        print(f"✅ CheXpert observations: {len(chexpert_observation_names)} classes")
        print(
            f"✅ Chest segmentation: {len(chest_segmentation_class_names)} classes (binary)"
        )
        return chexpert_observation_names

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def chexpert_observation_palette() -> List[List[int]]:
        """Возвращает палитру для визуализации классов CheXpert.

        Returns:
            List[List[int]]: Список `[R, G, B]` для каждого класса.
        """
        return [
            [120, 120, 120],
            [180, 120, 120],
            [6, 230, 230],
            [80, 50, 50],
            [4, 200, 3],
            [120, 120, 80],
            [140, 140, 140],
            [204, 5, 255],
            [230, 230, 230],
            [4, 250, 7],
            [224, 5, 255],
            [235, 255, 7],
            [150, 5, 61],
            [120, 120, 70],
        ]

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_isic_class_names() -> Dict[int, str]:
        """Возвращает словарь имён классов для ISIC 2018 (бинарная сегментация).

        Returns:
            Dict[int, str]: `{0: "background", 1: "lesion"}`.
        """
        # ISIC 2018 Class Names (Binary: skin lesion segmentation)
        # Source: https://challenge.isic-archive.com/
        isic_class_names: Dict[int, str] = {
            0: "background",  # Healthy skin / non-lesion area
            1: "lesion",  # Skin lesion (melanoma, nevus, etc.)
        }

        # Проверка
        print(f"✅ ISIC classes loaded: {len(isic_class_names)} classes (binary)")
        print(f"   Classes: {list(isic_class_names.values())}")
        return isic_class_names

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def binary_palette() -> List[List[int]]:
        """Возвращает бинарную палитру для визуализации (фон/объект).

        Returns:
            List[List[int]]: `[[120, 120, 120], [180, 120, 120]]`.
        """
        return [[120, 120, 120], [180, 120, 120]]

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _resize_mask_to_original(
        mask: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Утилита для ресайза маски к оригинальному размеру изображения.

        Использует ближайшего соседа (`order=0`) для сохранения целочисленных меток.

        Args:
            mask: Маска формы `(H, W)`.
            target_size: Целевой размер `(высота, ширина)`.

        Returns:
            np.ndarray: Ресайзнутая маска формы `target_size`.
        """
        if mask.shape != target_size:
            sh, sw = target_size[0] / mask.shape[0], target_size[1] / mask.shape[1]
            result: np.ndarray = zoom(mask, (sh, sw), order=0)
            return result
        return mask

    # ──────────────────────────────────────────────────────────────────────
    def load_image(self, input_image: ImageInput) -> Image.Image:
        """Загружает изображение из различных источников в формат `PIL.Image` (RGB).

        Поддерживаемые форматы:
        - Строка: путь к файлу или URL.
        - `PIL.Image`: конвертируется в RGB.
        - `np.ndarray`: конвертируется в PIL (поддержка 2D/3D/4D).
        - `torch.Tensor`: конвертируется через `.cpu().numpy()` → PIL.

        Args:
            input_image: Входное изображение в любом поддерживаемом формате.

        Returns:
            PIL.Image: Изображение в режиме "RGB".

        Raises:
            ValueError: Если не удалось загрузить изображение по пути/URL.
            TypeError: Если тип входных данных не поддерживается.
        """
        img: Image.Image
        if isinstance(input_image, str):
            if input_image.startswith(("http://", "https://")):
                resp = requests.get(input_image, timeout=30)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                img = Image.open(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            img = input_image.convert("RGB")
        elif isinstance(input_image, np.ndarray):
            if len(input_image.shape) == 2:
                img = Image.fromarray(input_image).convert("RGB")
            elif len(input_image.shape) == 3:
                # RGB или BGR
                if input_image.shape[2] == 3:
                    img = Image.fromarray(input_image)
                elif input_image.shape[2] == 4:
                    # RGBA
                    img = Image.fromarray(input_image).convert("RGB")
                else:
                    raise ValueError(
                        f"Неподдерживаемое количество каналов: {input_image.shape[2]}"
                    )
        elif isinstance(input_image, torch.Tensor):
            # PyTorch tensor → numpy → PIL
            t = input_image
            if t.dim() == 4:
                t = t.squeeze(0)
            np_img = (
                t.permute(1, 2, 0).cpu().numpy() if t.dim() == 3 else t.cpu().numpy()
            )
            if np_img.max() <= 1.0:
                np_img = (np_img * 255).astype(np.uint8)
            else:
                np_img = np_img.astype(np.uint8)
            img = Image.fromarray(np_img).convert("RGB")
        else:
            raise ValueError(
                f"Unsupported input type: {type(input_image)}. "
                "Provide a file path, URL, PIL.Image, np.ndarray, or torch.Tensor."
            )
        return img

    # ──────────────────────────────────────────────────────────────────────
    def predict_segmentation_map(
        self,
        input_image: ImageInput,
        verbose: bool = True,
        class_names: ClassNamesDict = None,
        gt_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Предсказывает карту семантической сегментации с опциональной вербозностью и метриками.

        Инференс полностью делегируется функции `utils.strategies.segment_image_unified`.

        Args:
            input_image: Входное изображение в любом поддерживаемом формате.
            verbose: Если `True`, выводит детали инференса в консоль.
            class_names: Словарь `{класс: имя}` для отображения имён классов.
            gt_mask: Ground truth маска для расчёта метрик (опционально).

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
            - `seg_map`: Семантическая маска `[H, W]`, dtype `uint8`.
            - `result_info`: Словарь с метаданными (метрики, время, классы, ...).
        """
        model_type_valid = cast(ValidModelType, self.model_type_str)  # type: ignore[reportInvalidTypeForm]
        class_names_fixed: Optional[Dict[int, str]] = (
            {
                int(k) if isinstance(k, str) and k.isdigit() else k: v  # type: ignore
                for k, v in class_names.items()
            }
            if class_names is not None
            else None
        )
        _, result_info = infer_unified(
            model=self.model,
            processor=self.processor,
            image_input=input_image,
            model_type=model_type_valid,
            alpha=0.5,
            palette=self.palette,
            device=str(self.device),
            verbose=verbose,
            num_classes=self.num_classes,
            class_names=class_names_fixed,
            gt_mask=gt_mask,
        )
        # Возвращаем маску + инфо
        return result_info["mask"], result_info

    # ──────────────────────────────────────────────────────────────────────
    def segment_image_unified(
        self,
        input_image: Union[str, Image.Image],
        alpha: float = 0.9,
        verbose: bool = True,
        class_names: ClassNamesDict = None,
        gt_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Универсальная функция сегментации для любой архитектуры.

        Возвращает наложение маски на оригинальное изображение + метаданные.

        Args:
            input_image: Путь к изображению или `PIL.Image`.
            alpha: Прозрачность наложения (0.0 = только фото, 1.0 = только маска).
            verbose: Если `True`, выводит детали инференса.
            class_names: Словарь имён классов.
            gt_mask: Ground truth для метрик.

        Returns:
            Tuple[PIL.Image, Dict[str, Any]]: (overlay, result_info).
        """
        model_type_valid = cast(ValidModelType, self.model_type_str)  # type: ignore[reportInvalidTypeForm]
        return infer_unified(
            model=self.model,
            processor=self.processor,
            image_input=input_image,
            model_type=model_type_valid,
            alpha=alpha,
            palette=self.palette,
            device=str(self.device),
            verbose=verbose,
            num_classes=self.num_classes,
            class_names=class_names,
            gt_mask=gt_mask,
        )

    # ──────────────────────────────────────────────────────────────────────
    def prepare_mask_for_overlay(
        self, mask_input: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Конвертирует маску в 2D numpy array для создания overlay.

        Обрабатывает:
        - `PIL.Image` (режимы "L" или "RGB").
        - `np.ndarray` с лишними измерениями (например, `(H, W, 1)`).
        - RGB-маски (использует первый канал).

        Args:
            mask_input: Входная маска в любом формате.

        Returns:
            np.ndarray: 2D массив формы `(H, W)`, dtype `uint8`.

        Raises:
            ValueError: Если маска не может быть приведена к 2D.
        """
        # Конвертируем PIL → numpy если нужно
        if isinstance(mask_input, Image.Image):
            mask = np.array(mask_input)
        else:
            mask = np.array(mask_input)

        if mask.ndim == 3:
            if mask.shape[2] == 1:
                # (H, W, 1) → (H, W)
                mask = mask.squeeze(2)
            elif mask.shape[2] == 3:
                # RGB изображение → нужно конвертировать в классы
                # Для Cityscapes: используем первый канал или конвертируем через палитру
                print("⚠️  RGB mask detected, using first channel")
                mask = mask[:, :, 0]
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")
        elif mask.ndim > 3:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D after processing, got {mask.ndim}D")

        return mask

    # ──────────────────────────────────────────────────────────────────────
    def segment_image(self, image: ImageInput, alpha: float = 0.9) -> Image.Image:
        """Выполняет семантическую сегментацию и возвращает наложение маски на оригинал.

        Алгоритм:
        1. Загружает изображение через `load_image()`.
        2. Предсказывает семантическую карту через `predict_segmentation_map()`.
        3. Назначает цвета из палитры для каждого класса.
        4. Блендит оригинал и цветную маску с коэффициентом `alpha`.

        Args:
            image: Входное изображение в любом поддерживаемом формате.
            alpha: Коэффициент блендинга (0.0 = только фото, 1.0 = только маска).

        Returns:
            PIL.Image: Изображение с наложенной цветной маской, режим "RGB".
        """
        img: Image.Image = self.load_image(image)  # type: ignore[arg-type]

        # Получаем карту сегментации
        seg_map, _ = self.predict_segmentation_map(image, verbose=False)  # type: ignore[arg-type]

        # Create color mask
        palette_array: np.ndarray = np.array(self.palette, dtype=np.uint8)
        h, w = seg_map.shape
        color_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette_array[: seg_map.max() + 1]):
            color_mask[seg_map == label] = color

        # Blend original and mask
        orig_arr: np.ndarray = np.array(img.convert("RGB"))
        overlay: np.ndarray = (orig_arr * (1 - alpha) + color_mask * alpha).astype(
            np.uint8
        )
        return Image.fromarray(overlay)

    # ──────────────────────────────────────────────────────────────────────
    def segment(self, image: ImageInput, **kwargs: Any) -> BinaryMask:  # type: ignore[override]
        """Основной метод сегментации.

        Возвращает бинарную маску, где объект = 255, фон = 0.
        Для многоклассовой сегментации: всё, кроме класса 0, считается объектом.

        Args:
            image: Входное изображение в любом поддерживаемом формате.
            **kwargs: Дополнительные параметры (игнорируются, для совместимости).

        Returns:
            BinaryMask: Бинарная маска формы `(H, W)`, dtype `uint8`, значения {0, 255}.
        """
        # Получаем карту сегментации через стратегию инференса
        seg_map, _ = self.predict_segmentation_map(image, verbose=False)
        # Бинаризация: всё кроме фона (класс 0) = объект
        mask: np.ndarray = (seg_map > 0).astype(np.uint8) * 255
        return mask

    # ──────────────────────────────────────────────────────────────────────
    def segment_with_mask(  # type: ignore[override]
        self, image: ImageInput, **kwargs: Any
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Сегментация с возвратом визуализации и бинарной маски.

        Алгоритм:
        1. Предсказывает семантическую карту.
        2. Создаёт бинарную маску (объект = 255, фон = 0).
        3. Создаёт цветной overlay с использованием палитры.
        4. Блендит оригинал и overlay.

        Args:
            image: Входное изображение.
            **kwargs: Дополнительные параметры (например, `alpha` для блендинга).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - `result`: Визуализация (оригинал + цветная маска), форма `(H, W, 3)`, dtype `uint8`.
            - `mask`: Бинарная маска, форма `(H, W)`, dtype `uint8`, значения {0, 255}.
        """
        start_time: float = time.perf_counter()
        alpha: float = kwargs.get("alpha", 0.9)

        # Получаем карту сегментации
        seg_map, _ = self.predict_segmentation_map(image, verbose=False)  # type: ignore[arg-type]
        unique_classes: np.ndarray = np.unique(seg_map)
        print("Предугаданные классы:", unique_classes)

        # Проверяем количество пикселей для каждого класса
        for cls in unique_classes:
            count = (seg_map == cls).sum()
            print(f"Class {cls}: {count} pixels")

        # Загружаем оригинал
        img_pil: Image.Image = self.load_image(image)  # type: ignore[arg-type]
        img_np: np.ndarray = np.array(img_pil.convert("RGB"))

        # Создаем бинарную маску (все, что не фон = 0)
        mask: np.ndarray = (seg_map > 0).astype(np.uint8) * 255

        # Создаем overlay: цветная сегментация поверх оригинала
        palette_array: np.ndarray = np.array(self.palette, dtype=np.uint8)
        h, w = seg_map.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for label in np.unique(seg_map):
            if label < len(palette_array):
                color_mask[seg_map == label] = palette_array[label]

        result: np.ndarray = (img_np * (1 - alpha) + color_mask * alpha).astype(
            np.uint8
        )

        print(f"Neural segmentation completed in {time.time() - start_time:.2f}s")
        return result, mask

    # ──────────────────────────────────────────────────────────────────────
    def detailed_segmentation(
        self, input_image: Union[str, Image.Image]
    ) -> Dict[str, Any]:
        """Детальная сегментация с возвратом всех промежуточных результатов.

        Возвращает:
        - Оригинальное изображение.
        - Семантическую карту (индексы классов).
        - Цветную сегментацию (RGB).
        - Overlay (оригинал + сегментация).
        - Распределение классов по пикселям.

        Args:
            input_image: Путь к изображению или `PIL.Image`.

        Returns:
            Dict[str, Any]:
            ```python
            {
                "original": PIL.Image,
                "segmentation_map": np.ndarray,  # [H, W], int
                "color_seg": np.ndarray,         # [H, W, 3], uint8
                "overlay": np.ndarray,           # [H, W, 3], uint8
                "class_distribution": Dict[str, Dict],  # {name: {id, count, pct}}
                "total_classes": int,
            }
            ```
        """
        img: Image.Image = self.load_image(input_image)

        # Получаем карту сегментации
        seg_map, _ = self.predict_segmentation_map(input_image, verbose=False)

        # Создаем цветную сегментацию
        palette_array: np.ndarray = np.array(self.palette, dtype=np.uint8)
        color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette_array):
            color_seg[seg_map == label] = color

        # Конвертируем из BGR в RGB
        color_seg_new: np.ndarray = color_seg[..., ::-1]

        # Создаем наложение
        orig_arr: np.ndarray = np.array(img)
        overlay: np.ndarray = (orig_arr * 0.2 + color_seg_new * 0.8).astype(np.uint8)

        # Анализ распределения классов
        unique_classes: np.ndarray
        counts: np.ndarray
        unique_classes, counts = np.unique(seg_map, return_counts=True)
        class_distribution: Dict = {}
        total_pixels: int = seg_map.size
        for cls, count in zip(unique_classes, counts):
            class_name: str = "Class_unknown"
            if hasattr(self.model, "config"):
                config = self.model.config
                if hasattr(config, "id2label") and isinstance(config.id2label, dict):
                    id2label: Dict[Union[int, str], str] = config.id2label  # type: ignore[assignment]
                    class_name = id2label.get(cls, f"Class_{cls}")  # type: ignore[arg-type]
            percentage: float = (count / total_pixels) * 100
            class_distribution[class_name] = {
                "class_id": int(cls),
                "pixel_count": int(count),
                "percentage": float(percentage),
            }
        return {
            "original": img,
            "segmentation_map": seg_map,
            "color_seg": color_seg_new,
            "overlay": overlay,
            "class_distribution": class_distribution,
            "total_classes": len(unique_classes),
        }

    # ──────────────────────────────────────────────────────────────────────
    def get_class_info(self) -> Dict[str, Any]:
        """Возвращает информацию о классах модели.

        Проверяет в порядке:
        1. `model.config.id2label` (HF-модели).
        2. Последний `torch.nn.Conv2d` в архитектуре (SMP/torchvision).
        3. Fallback на `self.num_classes`.

        Returns:
            Dict[str, Any]:
            ```python
            {
                "num_classes": int,
                "id2label": Dict[int, str],
                "label2id": Dict[str, int],
            }
            ```
        """
        if not hasattr(self, "model"):
            return {"error": "Model not initialized"}

        # HuggingFace модели
        if hasattr(self.model, "config") and isinstance(self.model, torch.nn.Module):
            config = self.model.config
            if hasattr(config, "num_labels"):
                num_labels_val = config.num_labels
                if isinstance(num_labels_val, torch.Tensor):
                    # 🔧 FIX: Явное игнорирование operator-ошибки для torch.Tensor.item()
                    num_classes_val: int = int(num_labels_val.item())  # type: ignore[operator]
                else:
                    # Для int/float/numpy-scalar
                    num_classes_val = int(num_labels_val)  # type: ignore[arg-type]
                return {
                    "num_classes": num_classes_val,
                    "id2label": getattr(config, "id2label", {}),
                    "label2id": getattr(config, "label2id", {}),
                }

        # SMP / Torchvision модели: ищем последний Conv2d
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return {
                    "num_classes": int(module.out_channels),
                    "id2label": {},
                    "label2id": {},
                }

        return {"num_classes": self.num_classes, "id2label": {}, "label2id": {}}
