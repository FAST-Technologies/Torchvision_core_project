# segmenters/NeuralModelFactory.py

# Импорт основных библиотек
from typing import (
    Tuple,
    Dict,
    Any,
    Optional,
    Union,
    List,
    Literal,
)
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import os
import yaml

import segmentation_models_pytorch as smp
import torchvision.models.segmentation as tv_seg
import torchvision.models.detection as tv_det

try:
    from transformers import (
        SegformerImageProcessor,
        SegformerForSemanticSegmentation,
        Mask2FormerImageProcessor,
        Mask2FormerForUniversalSegmentation,
        MaskFormerImageProcessor,
        MaskFormerForInstanceSegmentation,
        OneFormerProcessor,
        OneFormerForUniversalSegmentation,
        DPTImageProcessor,
        DPTForSemanticSegmentation,
        AutoImageProcessor,
        AutoModelForSemanticSegmentation,
    )
    from huggingface_hub import list_repo_files

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Warning: transformers not installed")

try:
    from ultralytics import SAM, YOLO

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ Warning: ultralytics not installed")

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
num_classes: int = 150
DeviceStr = Literal["cuda", "cpu"]
ModelTuple = Tuple[nn.Module, Optional[Any], str]  # (model, processor, model_type_str)
ProcessorLike = Optional[Union[Any, None]]  # HF processor или None для SMP/torchvision


class ModelType(Enum):
    """
    Перечисление поддерживаемых типов моделей сегментации.

    Категории:
    - HuggingFace Transformers: SEGFORMER, MASK2FORMER, ONEFORMER, DPT, UPERNET, MASKFORMER
    - Segmentation Models PyTorch: UNET_SMP, FPN_SMP, PSPNET_SMP
    - Torchvision: DEEPLAB_TV, FCN_TV, MASKRCNN_TV
    - Instance Segmentation: SAM, YOLOV8
    - Custom: SEGNET
    """

    SEGFORMER = "segformer"
    MASK2FORMER = "mask2former"
    ONEFORMER = "oneformer"
    DEEPLAB_TV = "deeplab_tv"
    UNET_SMP = "unet_smp"
    FPN_SMP = "fpn_smp"
    PSPNET_SMP = "pspnet_smp"
    PSP_SMP = "psp_smp"
    FCN_TV = "fcn_tv"
    SAM = "sam"
    DPT = "dpt"
    UPERNET = "upernet"
    SEGNET = "segnet"
    MASKRCNN_TV = "maskrcnn_tv"
    MASKFORMER = "maskformer"
    YOLOV8 = "yolov8"


class NeuralModelFactory:
    """
    Фабрика для создания и загрузки нейронных моделей сегментации.

    Поддерживает:
    - Загрузку предобученных моделей из HuggingFace Hub (SegFormer, Mask2Former, ...).
    - Создание SMP-моделей с разными encoder'ами (ResNet, MiT, EfficientNet).
    - Загрузку torchvision-моделей с заменой классификатора под NUM_CLASSES.
    - Интеграцию с instance segmentation (SAM, YOLOv8).
    - Конфигурацию через YAML-файл для централизованного управления параметрами.

    Основные методы:
    - `create_model()`: Универсальный конструктор по ModelType enum.
    - `create_model_from_config()`: Конструктор с параметрами из YAML.
    - `load_segformer_variant()`: Загрузка конкретной версии SegFormer (b0–b5).
    - `load_all_pretrained_cnn()`: Пакетная загрузка CNN-моделей для бенчмарка.

    Attributes:
        _model_registry (Dict[ModelType, Dict]): Реестр конфигураций моделей.
        _config_path (Path): Путь к YAML-конфигу.
        _config (Optional[Dict]): Кэшированная конфигурация.

    Example:
        ```python
        # Загрузка SegFormer B5
        model, processor, model_type = NeuralModelFactory.create_model(
            ModelType.SEGFORMER,
            model_name="nvidia/segformer-b5-finetuned-ade-640-640",
            device="cuda",
        )

        # Создание U-Net с чекпоинтом
        model, _, _ = NeuralModelFactory.create_model(
            ModelType.UNET_SMP,
            encoder_name="resnet34",
            checkpoint_path="models/unet_best.pth",
            device="cuda",
        )
        ```
    """

    _model_registry: Dict[ModelType, Dict[str, Any]] = {}
    _config_path: Path = Path("configs/neural_models.yaml")
    _config: Optional[Dict[str, Any]] = None

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Ленивая загрузка конфигурации из YAML-файла.

        При передаче нового пути сбрасывает кеш и загружает заново.
        Если файл не найден, возвращает конфигурацию по умолчанию.

        Args:
            config_path: Опциональный путь к конфигу (переопределяет `_config_path`).

        Returns:
            Dict[str, Any]: Словарь с конфигурацией моделей и обучения.
        """
        if config_path is not None:
            new_path = Path(config_path)
            if cls._config is None or new_path != cls._config_path:
                cls._config = None
                cls._config_path = new_path
        if cls._config is None:
            path = cls._config_path
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    cls._config = yaml.safe_load(f)
                print(f"✅ Config loaded from {path}")
            else:
                print(f"⚠️ Config not found: {path}, using defaults")
                cls._config = cls._get_default_config()
        return cls._config

    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """
        Возвращает конфигурацию по умолчанию, если YAML-файл не найден.

        Returns:
            Dict[str, Any]: Дефолтные параметры для моделей и обучения.
        """
        return {
            "models": {
                "segformer": {
                    "variants": {
                        "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
                        "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
                        "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
                        "b3": "nvidia/segformer-b3-finetuned-ade-640-640",
                        "b4": "nvidia/segformer-b4-finetuned-ade-640-640",
                        "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
                    },
                    "default": "b5",
                },
                "mask2former": {
                    "variants": {
                        "swin_base": "facebook/mask2former-swin-base-ade-semantic",
                        "swin_large": "facebook/mask2former-swin-large-ade-semantic",
                    },
                    "default": "swin_base",
                },
                "unet": {
                    "encoders": ["resnet34", "resnet50", "efficientnet-b0", "mit_b5"],
                    "default": "resnet34",
                },
            },
            "training": {
                "ade20k": {
                    "image_size": [512, 512],
                    "batch_size": 4,
                    "epochs": 50,
                    "lr": 1.0e-4,
                    "weight_decay": 1.0e-4,
                    "early_stop_patience": 5,
                }
            },
            "metrics": {"threshold": 0.5, "include_hausdorff": True},
        }

    @classmethod
    def get_model_name(cls, model_type: str, variant: Optional[str] = None) -> str:
        """
        Получает полное имя модели из конфигурации.

        Args:
            model_type: Ключ модели в конфиге (например, "segformer").
            variant: Вариант модели (например, "b5"). Если `None`, берётся дефолтный.

        Returns:
            str: Полное имя модели (например, "nvidia/segformer-b5-finetuned-ade-640-640").
        """
        config = cls.load_config()
        model_config = config["models"].get(model_type, {})

        if variant is None:
            variant = model_config.get("default")

        variants = model_config.get("variants", {})
        return variants.get(variant, variant)  # type: ignore[return-value]

    @classmethod
    def get_training_config(cls, dataset_name: str = "ade20k") -> Dict[str, Any]:
        """
        Получает конфигурацию обучения для указанного датасета.

        Args:
            dataset_name: Имя датасета (по умолчанию "ade20k").

        Returns:
            Dict[str, Any]: Параметры обучения (image_size, batch_size, lr, ...).
        """
        config = cls.load_config()
        return config["training"].get(dataset_name, config["training"]["ade20k"])

    @classmethod
    def get_metrics_config(cls) -> Dict[str, Any]:
        """
        Получает конфигурацию метрик.

        Returns:
            Dict[str, Any]: Параметры метрик (threshold, include_hausdorff).
        """
        config = cls.load_config()
        return config["metrics"]

    @classmethod
    def create_model_from_config(
        cls,
        model_type: str,
        variant: Optional[str] = None,
        device: DeviceStr = "cuda",
        checkpoint_path: str = "model_path",
        **kwargs: Any,
    ) -> ModelTuple:
        """
        Создаёт модель с параметрами из YAML-конфигурации.

        Args:
            model_type: Тип модели (ключ из конфига: "segformer", "unet", ...).
            variant: Вариант модели (например, "b5" для SegFormer).
            device: Устройство для вычислений ("cuda" или "cpu").
            checkpoint_path: Путь к чекпоинту для загрузки весов (опционально).
            **kwargs: Дополнительные параметры (encoder_name, num_classes, ...).

        Returns:
            Tuple[nn.Module, Optional[Any], str]: (model, processor, model_type_str).
        """
        config = cls.load_config()

        # Для SMP моделей — получаем encoder из конфига
        if model_type == "unet":
            encoders = config["models"]["unet"].get("encoders", ["resnet34"])
            encoder_name = kwargs.get(
                "encoder_name", encoders[0] if variant is None else variant
            )
            return cls.create_model(
                ModelType.UNET_SMP,
                device=device,
                encoder_name=encoder_name,
                checkpoint_path=checkpoint_path,
                **kwargs,
            )

        # Для HF моделей — используем model_name из конфига
        model_name = cls.get_model_name(model_type, variant)

        return cls.create_model(
            getattr(ModelType, model_type.upper()),
            model_name=model_name,
            device=device,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

    @classmethod
    def register_model(cls, model_type: ModelType, config: Dict[str, Any]) -> None:
        """
        Регистрирует конфигурацию новой модели в реестре фабрики.

        Args:
            model_type: Enum-тип модели.
            config: Словарь с параметрами для создания модели.
        """
        cls._model_registry[model_type] = config

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """
        Возвращает список поддерживаемых типов моделей.

        Returns:
            List[str]: Список значений `.value` из ModelType enum.
        """
        return [model_type.value for model_type in ModelType]

    @classmethod
    def create_model(
        cls,
        model_type: ModelType,
        model_name: Optional[str] = None,
        local_path: Optional[str] = None,
        checkpoint_path: Optional[str] = "model_path.pth",
        device: DeviceStr = "cuda",
        num_classes: int = 150,
        **kwargs: Any,
    ) -> ModelTuple:
        """
        Универсальный конструктор модели и процессора для инференса.

        Поддерживает оба подхода:
        1. Прямой: передача `model_name`/`local_path` для HF-моделей.
        2. Через конфиг: параметры подгружаются из YAML.

        Args:
            model_type: Тип модели (ModelType enum).
            model_name: Имя модели в HuggingFace Hub (для HF-моделей).
            local_path: Локальный путь к модели (альтернатива model_name).
            checkpoint_path: Путь к чекпоинту .pth для SMP/torchvision-моделей.
            device: Устройство для вычислений.
            num_classes: Количество выходных классов (по умолчанию 150 для ADE20K).
            **kwargs: Дополнительные параметры (encoder_name, variant, ...).

        Returns:
            Tuple[nn.Module, Optional[Any], str]:
            - model: Загруженная модель в режиме `.eval()`.
            - processor: HF-процессор для препроцессинга (или None для SMP/torchvision).
            - model_type_str: Строковый идентификатор типа модели.

        Raises:
            ValueError: Если model_type не поддерживается.
            ImportError: Если требуется библиотека, которая не установлена.
        """
        if model_type == ModelType.SEGFORMER:
            return cls._load_segformer(model_name, local_path, device)
        elif model_type == ModelType.MASK2FORMER:
            return cls._load_mask2former(model_name, device)
        elif model_type == ModelType.MASKFORMER:
            return cls._load_maskformer(model_name, device)
        elif model_type == ModelType.ONEFORMER:
            return cls._load_oneformer(model_name, device)
        elif model_type == ModelType.DPT:
            return cls._load_dpt(model_name, device)
        elif model_type == ModelType.UPERNET:
            return cls._load_upernet(model_name, device)
        elif model_type == ModelType.DEEPLAB_TV:
            return cls._load_deeplab_tv(device, num_classes, checkpoint_path)
        elif model_type == ModelType.UNET_SMP:
            return cls._load_unet_smp(device, num_classes, checkpoint_path, **kwargs)
        elif model_type == ModelType.FPN_SMP:
            return cls._load_fpn_smp(device, num_classes, checkpoint_path, **kwargs)
        elif model_type == ModelType.PSPNET_SMP:
            return cls._load_psp_smp(device, num_classes, checkpoint_path, **kwargs)
        elif model_type == ModelType.FCN_TV:
            return cls._load_fcn_tv(device, num_classes, checkpoint_path, **kwargs)
        elif model_type == ModelType.SEGNET:
            return cls._load_segnet(device, num_classes, checkpoint_path, **kwargs)
        elif model_type == ModelType.SAM:
            return cls._load_sam(model_name, device)
        elif model_type == ModelType.MASKRCNN_TV:
            return cls._load_maskrcnn_tv(device, **kwargs)
        elif model_type == ModelType.YOLOV8:
            return cls._load_yolov8(model_name, device)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

    # ──────────────────────────────────────────────────────────────────────
    # SEGFORMER
    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def _load_segformer(
        cls,
        model_name: Optional[str] = None,
        local_path: Optional[str] = None,
        device: DeviceStr = "cuda",
        **kwargs: Any,
    ) -> ModelTuple:
        """
        Загружает модель SegFormer из HuggingFace Hub или локального пути.

        Args:
            model_name: Имя модели в HF Hub (например, "nvidia/segformer-b5-finetuned-ade-640-640").
            local_path: Локальный путь к сохранённой модели.
            device: Устройство для вычислений.
            **kwargs: Дополнительные параметры (игнорируются).

        Returns:
            ModelTuple: (model, processor, "segformer").

        Raises:
            ImportError: Если `transformers` не установлен.
            ValueError: Если не указан ни model_name, ни local_path.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        if model_name is None and local_path is None:
            raise ValueError("Укажите model_name или local_path для SegFormer")

        source = local_path if local_path else model_name

        processor = SegformerImageProcessor.from_pretrained(source)  # type: ignore[arg-type]
        model = SegformerForSemanticSegmentation.from_pretrained(source).to(device).eval()  # type: ignore[arg-type]

        return model, processor, "segformer"

    @classmethod
    def load_segformer_variant(
        cls, variant: str = "b2", device: DeviceStr = "cuda"
    ) -> ModelTuple:
        """
        Загружает конкретную версию SegFormer для сравнения.

        Поддерживаемые варианты: b0, b1, b2, b3, b4, b5.

        Args:
            variant: Версия модели ("b0"–"b5").
            device: Устройство для вычислений.

        Returns:
            ModelTuple: (model, processor, f"segformer_{variant}").

        Raises:
            ValueError: Если указана неизвестная версия.
        """
        config = cls.load_config()
        variants = config["models"]["segformer"]["variants"]

        if variant not in variants:
            raise ValueError(
                f"Unknown SegFormer variant: {variant}. Available: {list(variants.keys())}"
            )

        model_name = variants[variant]
        processor = SegformerImageProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
        model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()  # type: ignore[arg-type]
        return model, processor, f"segformer_{variant}"

    @classmethod
    def print_segformer_params(cls, path: str, device: DeviceStr = "cuda") -> None:
        """
        Выводит параметры загруженной модели SegFormer для отладки.

        Args:
            path: Путь к модели (локальный или в HF Hub).
            device: Устройство для загрузки.
        """
        processor = SegformerImageProcessor.from_pretrained(path)  # type: ignore[arg-type]
        model = SegformerForSemanticSegmentation.from_pretrained(path).to(device)  # type: ignore[arg-type]
        print(processor)
        print(model)
        print("✅ Модель успешно загружена!")
        print(f"   Путь: {path}")
        print(f"   Устройство: {device}")
        print(model.config)

    @classmethod
    def print_segformer_variant_params(
        cls, variant: str = "b2", device: DeviceStr = "cuda"
    ) -> None:
        """
        Выводит параметры конкретной версии SegFormer.

        Args:
            variant: Версия модели ("b0"–"b5").
            device: Устройство для загрузки.
        """
        variants: Dict[str, str] = {
            "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
            "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
            "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
            "b3": "nvidia/segformer-b3-finetuned-ade-640-640",
            "b4": "nvidia/segformer-b4-finetuned-ade-640-640",
            "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
        }

        if variant not in variants:
            raise ValueError(f"Unknown SegFormer variant: {variant}")
        model_name = variants[variant]
        processor = SegformerImageProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
        model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)  # type: ignore[arg-type]
        print(processor)
        print(model)
        print(f"✅ SegFormer-{variant} загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ──────────────────────────────────────────────────────────────────────
    # MASK2FORMER / MASKFORMER / ONEFORMER / DPT / UPERNET
    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def _load_mask2former(
        cls, model_name: Optional[str], device: DeviceStr = "cuda"
    ) -> ModelTuple:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        processor = Mask2FormerImageProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device).eval()  # type: ignore[arg-type]
        return model, processor, "mask2former"

    @classmethod
    def print_mask2former_params(
        cls,
        name: str = "facebook/mask2former-swin-base-ade-semantic",
        device: DeviceStr = "cuda",
    ) -> None:
        """Вывод параметров Mask2Former"""
        processor = Mask2FormerImageProcessor.from_pretrained(name)  # type: ignore[arg-type]
        model = Mask2FormerForUniversalSegmentation.from_pretrained(name).to(device)  # type: ignore[arg-type]
        print(processor)
        print(model)
        print("✅ Mask2Former загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== MASKFORMER ==========
    @classmethod
    def _load_maskformer(
        cls, model_name: Optional[str], device: str = "cuda"
    ) -> Tuple[Any, Any, str]:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        processor = MaskFormerImageProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
        model = MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(device).eval()  # type: ignore[arg-type]
        return model, processor, "maskformer"

    @classmethod
    def print_maskformer_params(
        cls,
        model_name: str = "facebook/maskformer-resnet50-ade20k-full",
        device: DeviceStr = "cuda",
    ) -> None:
        """Вывод параметров MaskFormer"""
        processor = MaskFormerImageProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
        model = MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(device)  # type: ignore[arg-type]
        print(processor)
        print(model)
        print("✅ MaskFormer загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== ONEFORMER ==========
    @classmethod
    def _load_oneformer(
        cls, model_name: Optional[str], device: DeviceStr = "cuda"
    ) -> ModelTuple:
        if model_name is None:
            raise ValueError("model_name обязателен для OneFormer")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        files = list_repo_files(model_name)
        print(
            "✅ safetensors found!"
            if any(f.endswith(".safetensors") for f in files)
            else "❌ Только pickle (.bin)"
        )
        processor = OneFormerProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
        model = OneFormerForUniversalSegmentation.from_pretrained(model_name).to(device).eval()  # type: ignore[arg-type]
        return model, processor, "oneformer"

    @classmethod
    def print_oneformer_params(
        cls,
        name: str = "shi-labs/oneformer_ade20k_swin_large",
        device: DeviceStr = "cuda",
    ) -> None:
        """Вывод параметров OneFormer"""
        files = list_repo_files(name)
        print(
            "✅ safetensors found!"
            if any(f.endswith(".safetensors") for f in files)
            else "❌ Только pickle (.bin)"
        )
        processor = OneFormerProcessor.from_pretrained(name)  # type: ignore[arg-type]
        model = OneFormerForUniversalSegmentation.from_pretrained(name).to(device)  # type: ignore[arg-type]
        print(processor)
        print(model)
        print("✅ OneFormer загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== DPT ==========
    @classmethod
    def _load_dpt(
        cls, model_name: Optional[str], device: DeviceStr = "cuda"
    ) -> ModelTuple:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        processor = DPTImageProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
        model = DPTForSemanticSegmentation.from_pretrained(model_name).to(device).eval()  # type: ignore[arg-type]
        return model, processor, "dpt"

    @classmethod
    def print_dpt_params(
        cls, model_name: str = "Intel/dpt-large-ade", device: DeviceStr = "cuda"
    ) -> None:
        """Вывод параметров DPT"""
        processor = DPTImageProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
        model = DPTForSemanticSegmentation.from_pretrained(model_name).to(device)  # type: ignore[arg-type]
        print(processor)
        print(model)
        print("✅ DPT загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== UPERNET ==========
    @classmethod
    def _load_upernet(
        cls, model_name: Optional[str], device: DeviceStr = "cuda"
    ) -> ModelTuple:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        processor = AutoImageProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
        model = (
            AutoModelForSemanticSegmentation.from_pretrained(model_name)  # type: ignore[arg-type]
            .to(device)
            .eval()
        )
        return model, processor, "upernet"

    @classmethod
    def print_upernet_params(
        cls,
        model_name: str = "openmmlab/upernet-convnext-small",
        device: DeviceStr = "cuda",
    ) -> None:
        """Вывод параметров UPerNet"""
        processor = AutoImageProcessor.from_pretrained(model_name)  # type: ignore[arg-type]
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)  # type: ignore[arg-type]
        print(processor)
        print(model)
        print("✅ UPerNet загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== DEEPLAB_TV ==========
    # ──────────────────────────────────────────────────────────────────────
    # TORCHVISION & SMP МОДЕЛИ
    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def _load_deeplab_tv(
        cls,
        device: DeviceStr,
        num_classes: int,
        checkpoint_path: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelTuple:
        num_classes = int(num_classes)
        safe_path = checkpoint_path or "model_path.pth"
        if checkpoint_path and os.path.exists(safe_path):
            model = tv_seg.deeplabv3_resnet101(weights=None)
            model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            model_keys = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("aux_classifier")
            }
            model.load_state_dict(model_keys, strict=False)
            print(f"✅ Loaded DeepLabV3+ from checkpoint: {checkpoint_path}")
        else:
            model = tv_seg.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")
            model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
            print("⚠️ Checkpoint not found, using COCO weights")

        model = model.to(device).eval()
        return model, None, "deeplab_tv"

    @classmethod
    def print_deeplab_params(cls, device: str = "cuda") -> None:
        """Вывод параметров DeepLab"""
        model = tv_seg.deeplabv3_resnet101(weights="DEFAULT")  # Было pretrained=True
        print(model)
        print("✅ DeepLab загружена!")
        print(f"   Устройство: {device}")

    @classmethod
    def _load_unet_smp(
        cls,
        device: DeviceStr,
        num_classes: int,
        checkpoint_path: Optional[str] = None,
        encoder_name: str = "resnet34",
        **kwargs: Any,
    ) -> ModelTuple:
        num_classes = int(num_classes)
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

        safe_path = checkpoint_path or "model_path.pth"
        if checkpoint_path and os.path.exists(safe_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded U-Net from checkpoint: {checkpoint_path}")
        else:
            print("⚠️ Checkpoint not found, using ImageNet encoder only")
        model = model.to(device).eval()
        return model, None, "unet_smp"

    @classmethod
    def print_unet_params(
        cls,
        encoder_name: str = "resnet34",
        num_classes: int = num_classes,
        checkpoint_path: str = "unet_smp.pth",
        device: str = "cuda",
    ) -> None:
        """Вывод параметров U-Net"""
        num_classes = int(num_classes)
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded U-Net from checkpoint: {checkpoint_path}")
        model = model.to(device).eval()
        print(model)
        print("✅ U-Net загружена!")
        print(f"   Устройство: {device}")

    # ========== FPN_SMP ==========
    @classmethod
    def _load_fpn_smp(
        cls,
        device: DeviceStr,
        num_classes: int,
        checkpoint_path: Optional[str] = None,
        encoder_name: str = "mit_b5",
        **kwargs: Any,
    ) -> ModelTuple:
        num_classes = int(num_classes)
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

        safe_path = checkpoint_path or "model_path.pth"
        if checkpoint_path and os.path.exists(safe_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded FPN from checkpoint: {checkpoint_path}")
        else:
            print("⚠️ Checkpoint not found, using ImageNet encoder only")

        model = model.to(device).eval()
        model.output_stride = 32  # type: ignore[attr-defined]
        model.target_size = (512, 512)  # type: ignore[attr-defined]
        if "mit" in encoder_name:
            model_type_str = "fpn_mit"
        elif "efficientnet" in encoder_name:
            model_type_str = "fpn_effnet"
        else:
            model_type_str = "fpn_smp"
        return model, None, model_type_str

    @classmethod
    def print_fpn_params(
        cls,
        encoder_name: str = "mit_b5",
        num_classes: int = num_classes,
        checkpoint_path: str = "fpn_smp.pth",
        device: str = "cuda",
    ) -> None:
        """Вывод параметров FPN"""
        num_classes = int(num_classes)
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded FPN from checkpoint: {checkpoint_path}")

        model = model.to(device).eval()
        model.output_stride = 32
        model.target_size = (512, 512)
        print(model)
        print("✅ FPN загружена!")
        print(f"   Устройство: {device}")

    # ========== PSP_SMP ==========
    @classmethod
    def _load_psp_smp(
        cls,
        device: DeviceStr,
        num_classes: int,
        checkpoint_path: Optional[str] = None,
        encoder_name: str = "mit_b5",
        **kwargs: Any,
    ) -> ModelTuple:
        psp_size = 2048 if "mit" in encoder_name else 512
        num_classes = int(num_classes)
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
            psp_size=psp_size,
        )

        safe_path = checkpoint_path or "model_path.pth"
        if checkpoint_path and os.path.exists(safe_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded PSPNet from checkpoint: {checkpoint_path}")
        else:
            print("⚠️ Checkpoint not found, using ImageNet encoder only")

        model = model.to(device).eval()
        if "mit" in encoder_name:
            model_type_str = "psp_mit"
        elif "efficientnet" in encoder_name:
            model_type_str = "psp_effnet"
        elif "resnet" in encoder_name:
            model_type_str = "psp_resnet"
        else:
            model_type_str = "psp_mit"
        model.output_stride = 8  # type: ignore[attr-defined]
        model.target_size = (512, 512)  # type: ignore[attr-defined]
        return model, None, model_type_str

    @classmethod
    def print_psp_params(
        cls,
        encoder_name: str = "mit_b5",
        num_classes: int = num_classes,
        checkpoint_path: str = "psp_smp.pth",
        device: str = "cuda",
    ) -> None:
        """Вывод параметров PSPNet"""
        num_classes = int(num_classes)
        psp_size = 2048 if "mit" in encoder_name else 512
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
            psp_size=psp_size,
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded PSPNet from checkpoint: {checkpoint_path}")

        model = model.to(device).eval()
        model.output_stride = 8
        model.target_size = (512, 512)
        print(model)
        print("✅ PSPNet загружена!")
        print(f"   Устройство: {device}")

    # ========== FCN_TV ==========
    @classmethod
    def _load_fcn_tv(
        cls,
        device: DeviceStr,
        num_classes: int,
        checkpoint_path: Optional[str] = None,
        variant: str = "fcn_resnet50",
        **kwargs: Any,
    ) -> ModelTuple:
        num_classes = int(num_classes)
        variants = {
            "fcn_resnet50": tv_seg.fcn_resnet50,
            "fcn_resnet101": tv_seg.fcn_resnet101,
        }

        if variant not in variants:
            raise ValueError(f"Unknown FCN variant: {variant}")

        model = variants[variant](weights=None)
        old_classifier = model.classifier[4]
        model.classifier[4] = torch.nn.Conv2d(
            old_classifier.in_channels, num_classes, kernel_size=1
        )
        torch.nn.init.normal_(model.classifier[4].weight, 0, 0.01)
        torch.nn.init.constant_(model.classifier[4].bias, 0)

        safe_path = checkpoint_path or "model_path.pth"
        if checkpoint_path and os.path.exists(safe_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Loaded FCN from checkpoint: {checkpoint_path}")
        else:
            print("⚠️ Checkpoint not found, using ImageNet backbone only")
        model = model.to(device).eval()
        model.output_stride = 1
        model.target_size = (512, 512)
        return model, None, "fcn_tv"

    @classmethod
    def print_fcn_params(
        cls, variant: str = "fcn_resnet50", device: str = "cuda"
    ) -> None:
        """Вывод параметров FCN"""
        variants = {
            "fcn_resnet50": tv_seg.fcn_resnet50,
            "fcn_resnet101": tv_seg.fcn_resnet101,
        }
        if variant not in variants:
            raise ValueError(f"Unknown FCN variant: {variant}")
        model = variants[variant](weights="DEFAULT")
        print(model)
        print("✅ FCN загружена!")
        print(f"   Устройство: {device}")

    # ========== SEGNET ==========
    @classmethod
    def _load_segnet(
        cls,
        device: DeviceStr,
        num_classes: int,
        checkpoint_path: Optional[str] = None,
        encoder_name: str = "resnet34",
        **kwargs: Any,
    ) -> ModelTuple:
        num_classes = int(num_classes)
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

        safe_path = checkpoint_path or "model_path.pth"
        if checkpoint_path and os.path.exists(safe_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded SegNet from checkpoint: {checkpoint_path}")
        else:
            print("⚠️ Checkpoint not found, using ImageNet encoder only")
        model = model.to(device).eval()
        return model, None, "segnet"

    @classmethod
    def print_segnet_params(
        cls, encoder_name: str = "resnet34", device: str = "cuda"
    ) -> None:
        """Вывод параметров SegNet"""
        try:
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
                activation=None,
            ).to(device)
            print(model)
            print("✅ SegNet загружена!")
            print(f"   Устройство: {device}")
        except Exception as e:
            print(f"⚠️ SegNet not available in SMP: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # INSTANCE SEGMENTATION: SAM / YOLOv8 / Mask R-CNN
    # ──────────────────────────────────────────────────────────────────────
    # ========== SAM ==========
    @classmethod
    def _load_sam(
        cls, model_name: Optional[str], device: DeviceStr = "cuda"
    ) -> ModelTuple:
        if model_name is None:
            raise ValueError("model_name обязателен для OneFormer")
        if not SAM_AVAILABLE:
            raise ImportError("ultralytics library required for SAM")

        if os.path.exists(model_name):
            print(
                f"   📁 Found: {model_name} ({os.path.getsize(model_name) / 1024**2:.3f} MB)"
            )
        elif not model_name.startswith("sam2"):
            print(f"   ⚠️ Warning: {model_name} not found in current directory")

        model = SAM(model_name)  # type: ignore[call-arg]

        if "sam2" in model_name.lower():
            model_type = "sam2"
        else:
            model_type = "sam"
        return model, None, model_type

    @classmethod
    def print_sam_params(
        cls, model_name: str = "mobile_sam.pt", device: str = "cuda"
    ) -> None:
        """Вывод параметров SAM"""
        if os.path.exists(model_name):
            print(
                f"   📁 Found: {model_name} ({os.path.getsize(model_name) / 1024**2:.3f} MB)"
            )
        elif not model_name.startswith("sam2"):
            print(f"   ⚠️ Warning: {model_name} not found in current directory")

        model = SAM(model_name)
        print(model)
        print("✅ SAM загружена!")
        print(f"   Устройство: {device}")

    # ========== MASK R-CNN ==========
    @classmethod
    def _load_maskrcnn_tv(
        cls, device: DeviceStr, variant: str = "maskrcnn_resnet50_fpn", **kwargs: Any
    ) -> ModelTuple:
        variants = {
            "maskrcnn_resnet50_fpn": tv_det.maskrcnn_resnet50_fpn,
            "maskrcnn_resnet50_fpn_v2": tv_det.maskrcnn_resnet50_fpn_v2,
        }

        if variant not in variants:
            raise ValueError(f"Unknown Mask R-CNN variant: {variant}")
        model = variants[variant](weights="DEFAULT")  # Было weights="COCO_V1"
        model = model.to(device).eval()
        score_thresh = kwargs.get("score_thresh", 0.5)
        model.score_thresh = score_thresh
        return model, None, "maskrcnn_tv"

    @classmethod
    def print_mask_rcnn_params(
        cls, variant: str = "maskrcnn_resnet50_fpn", device: str = "cuda"
    ) -> None:
        """Вывод параметров Mask R-CNN"""
        variants = {
            "maskrcnn_resnet50_fpn": tv_det.maskrcnn_resnet50_fpn,
            "maskrcnn_resnet50_fpn_v2": tv_det.maskrcnn_resnet50_fpn_v2,
        }
        if variant not in variants:
            raise ValueError(f"Unknown Mask R-CNN variant: {variant}")
        model = variants[variant](
            weights=None
        )  # Было weights="DEFAULT", pretrained=False
        model = model.to(device)
        print(model)
        print("✅ Mask R-CNN загружена!")
        print(f"   Устройство: {device}")

    @classmethod
    def _load_yolov8(
        cls, model_name: Optional[str], device: DeviceStr = "cuda"
    ) -> ModelTuple:
        """Загрузка YOLOv8 для сегментации"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics library required for YOLOv8. Install with: pip install ultralytics"
            )
        if model_name is None:
            model_name = "yolov8n-seg.pt"
        model = YOLO(model_name)  # type: ignore[call-arg]
        print(model)
        print("✅ YOLO загружена!")
        print(f"   Устройство: {device}")
        return model, None, "yolov8"

    # ──────────────────────────────────────────────────────────────────────
    # УНИВЕРСАЛЬНЫЙ ЗАГРУЗЧИК SMP
    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def load_smp_model(
        cls,
        architecture: Literal["unet", "fpn", "pspnet", "deeplabv3+"],
        encoder_name: str,
        model_type: Optional[str] = None,
        key: Optional[str] = None,
        device: DeviceStr = "cuda",
        num_classes: int = num_classes,
        checkpoint_path: Optional[str] = None,
    ) -> ModelTuple:
        """
        Универсальный загрузчик для SMP-моделей.

        Args:
            architecture: Архитектура модели ('unet', 'fpn', 'pspnet', 'deeplabv3+').
            encoder_name: Название encoder ('resnet34', 'mit_b5', 'efficientnet-b0', ...).
            model_type: Тип модели для dispatch (опционально).
            key: Уникальный ключ для доступа к результатам (опционально).
            checkpoint_path: Путь к чекпоинту (опционально).

        Returns:
            ModelTuple: (model, None, model_type_str).
        """
        architectures: Dict[str, Any] = {
            "unet": smp.Unet,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
            "deeplabv3+": smp.DeepLabV3Plus,
        }

        if architecture not in architectures:
            raise ValueError(f"Unknown architecture: {architecture}")

        ModelClass = architectures[architecture]
        extra_kwargs: Dict[str, Any] = {}
        if architecture == "pspnet":
            if "mit" in encoder_name:
                extra_kwargs["psp_size"] = 2048
            elif "efficientnet" in encoder_name:
                extra_kwargs["psp_size"] = 1792 if "b7" in encoder_name else 1280
            else:
                extra_kwargs["psp_size"] = 2048
        model = ModelClass(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
            **extra_kwargs,
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(
                f"✅ Loaded {architecture.upper()} from checkpoint: {checkpoint_path}"
            )
        model = model.to(device).eval()
        if key is None:
            key = f"{architecture}_{encoder_name.replace('-', '_')}"
        if model_type is None:
            model_type = architecture
        return model, None, model_type

    # ──────────────────────────────────────────────────────────────────────
    # ПАКЕТНАЯ ЗАГРУЗКА ДЛЯ БЕНЧМАРКА
    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def load_all_pretrained_cnn(
        cls,
        checkpoint_dir: str = "./checkpoints",
        device: DeviceStr = "cuda",
        num_classes: int = num_classes,
    ) -> Dict[str, ModelTuple]:
        """
        Загружает все CNN-модели с предобученными весами для бенчмарка.

        Returns:
            Dict[str, ModelTuple]: Словарь `{model_key: (model, processor, model_type)}`.
        """
        models_dict: Dict[str, ModelTuple] = {}
        print("\n" + "=" * 60)
        print("📦 Loading all pre-trained CNN models for benchmark")
        print("=" * 60)

        # FPN + MiT-B5
        fpn_checkpoint = os.path.join(checkpoint_dir, "fpn_mit_b5_best.pth")
        fpn_model, _, fpn_type = cls._load_fpn_smp(
            device, num_classes, fpn_checkpoint, encoder_name="mit_b5"
        )
        models_dict["fpn_mit_b5_pretrained"] = (fpn_model, None, fpn_type)

        # PSPNet + MiT-B5
        psp_checkpoint = os.path.join(checkpoint_dir, "psp_mit_b5_best.pth")
        psp_model, _, psp_type = cls._load_psp_smp(
            device, num_classes, psp_checkpoint, encoder_name="mit_b5"
        )
        models_dict["psp_mit_b5_pretrained"] = (psp_model, None, psp_type)

        # FCN ResNet-50
        fcn_model, _, fcn_type = cls._load_fcn_tv(
            device, num_classes, variant="fcn_resnet50"
        )
        models_dict["fcn_resnet50_pretrained"] = (fcn_model, None, fcn_type)

        # SegNet (U-Net proxy)
        segnet_model, _, segnet_type = cls._load_segnet(
            device, num_classes, encoder_name="resnet34"
        )
        models_dict["segnet_resnet34_pretrained"] = (segnet_model, None, segnet_type)

        # Mask R-CNN (COCO)
        mrcnn_model, _, mrcnn_type = cls._load_maskrcnn_tv(
            device, variant="maskrcnn_resnet50_fpn"
        )
        models_dict["maskrcnn_pretrained"] = (mrcnn_model, None, mrcnn_type)

        print("\n✅ All pre-trained CNN models loaded!")
        print(f"   Total models: {len(models_dict)}")
        return models_dict
