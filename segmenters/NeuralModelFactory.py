# segmenters/NeuralModelFactory.py

# Импорт основных библиотек
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
from enum import Enum
import torch
import os
import segmentation_models_pytorch as smp
import torchvision.models.segmentation as tv_seg
import torchvision.models.detection as tv_det
from huggingface_hub import list_repo_files

import yaml
from pathlib import Path

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

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Warning: transformers not installed")

try:
    from ultralytics import SAM

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ Warning: ultralytics not installed")


class ModelType(Enum):
    SEGFORMER = "segformer"
    MASK2FORMER = "mask2former"
    ONEFORMER = "oneformer"
    DEEPLAB_TV = "deeplab_tv"
    UNET_SMP = "unet_smp"
    FPN_SMP = "fpn_smp"
    PSPNET_SMP = "pspnet_smp"
    FCN_TV = "fcn_tv"
    SAM = "sam"
    DPT = "dpt"
    UPERNET = "upernet"
    SEGNET = "segnet"
    MASKRCNN_TV = "maskrcnn_tv"
    MASKFORMER = "maskformer"


num_classes: int = 150


class NeuralModelFactory:
    """Фабрика для создания и загрузки нейронных моделей сегментации"""

    _model_registry: Dict[ModelType, Dict[str, Any]] = {}
    _config_path = Path("configs/neural_models.yaml")
    _config = None

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Ленивая загрузка конфига"""
        if cls._config is None:
            path = Path(config_path) if config_path else cls._config_path
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
        """Конфиг по умолчанию если файл не найден"""
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
        """Получение имени модели из конфига"""
        config = cls.load_config()
        model_config = config["models"].get(model_type, {})

        if variant is None:
            variant = model_config.get("default")

        variants = model_config.get("variants", {})
        return variants.get(variant, variant)

    @classmethod
    def get_training_config(cls, dataset_name: str = "ade20k") -> Dict[str, Any]:
        """Получение конфигурации обучения"""
        config = cls.load_config()
        return config["training"].get(dataset_name, config["training"]["ade20k"])

    @classmethod
    def get_metrics_config(cls) -> Dict[str, Any]:
        """Получение конфигурации метрик"""
        config = cls.load_config()
        return config["metrics"]

    @classmethod
    def create_model_from_config(
        cls,
        model_type: str,
        variant: Optional[str] = None,
        device: str = "cuda",
        checkpoint_path: str = "model_path",
        **kwargs,
    ) -> Tuple[Any, Any, str]:
        """
        Создание модели с параметрами из YAML конфига.

        Args:
            model_type: Тип модели ("segformer", "unet", etc.)
            variant: Вариант модели ("b5", "resnet34", etc.)
            device: Устройство
            checkpoint_path: Путь к чекпоинту (опционально)
            **kwargs: Дополнительные параметры

        Returns:
            tuple: (model, processor, model_type_str)
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
        """Регистрация конфигурации модели"""
        cls._model_registry[model_type] = config

    @classmethod
    def get_supported_models(cls) -> list:
        """Возвращает список поддерживаемых моделей"""
        return [model_type.value for model_type in ModelType]

    @classmethod
    def create_model(
        cls,
        model_type: ModelType,
        model_name: Optional[str] = None,
        local_path: Optional[str] = None,
        checkpoint_path: str = "model_path",
        device: str = "cuda",
        num_classes: int = num_classes,
        **kwargs,
    ) -> Tuple[Any, Any, str]:
        """
        Создаёт модель и процессор для инференса.
        ПОДДЕРЖИВАЕТ оба подхода: прямой и через конфиг

        Args:
            model_type: Тип модели (ModelType enum)
            model_name: Имя модели (для HF)
            local_path: Локальный путь к модели
            checkpoint_path: Путь к чекпоинту (.pth)
            device: Устройство
            num_classes: Количество классов
            **kwargs: Дополнительные параметры (encoder_name, variant, etc.)

        Returns:
            tuple: (model, processor, model_type_str)
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
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

    # ========== SEGFORMER ==========
    @classmethod
    def _load_segformer(
        cls, model_name: str = None, local_path: Optional[str] = None, device: str = "cuda"
    ) -> Tuple[Any, Any, str]:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        processor = SegformerImageProcessor.from_pretrained(
            local_path if local_path else model_name
        )
        model = (
            SegformerForSemanticSegmentation.from_pretrained(
                local_path if local_path else model_name
            )
            .to(device)
            .eval()
        )

        return model, processor, "segformer"

    @classmethod
    def load_segformer_variant(
        cls, variant: str = "b2", device: str = "cuda"
    ) -> Tuple[Any, Any, str]:
        """
        Загрузка разных версий SegFormer для сравнения.

        Args:
            variant: Версия модели ("b0", "b1", "b2", "b3", "b4", "b5")

        Returns:
            self: Для цепочки вызовов

        Raises:
            ValueError: Если указана неизвестная версия
        """
        config = cls.load_config()
        variants = config["models"]["segformer"]["variants"]

        if variant not in variants:
            raise ValueError(
                f"Unknown SegFormer variant: {variant}. Available: {list(variants.keys())}"
            )

        model_name = variants[variant]
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = (
            SegformerForSemanticSegmentation.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        return model, processor, f"segformer_{variant}"

    @classmethod
    def print_segformer_params(cls, path: str, device: str = "cuda") -> None:
        """Вывод параметров SegFormer"""
        processor = SegformerImageProcessor.from_pretrained(path)
        model = SegformerForSemanticSegmentation.from_pretrained(path).to(device)
        print(processor)
        print(model)
        print(f"✅ Модель успешно загружена!")
        print(f"   Путь: {path}")
        print(f"   Устройство: {device}")
        print(model.config)

    @classmethod
    def print_segformer_variant_params(
        cls, variant: str = "b2", device: str = "cuda"
    ) -> None:
        """Вывод параметров версии SegFormer"""
        variants = {
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
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
        print(processor)
        print(model)
        print(f"✅ SegFormer-{variant} загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== MASK2FORMER ==========
    @classmethod
    def _load_mask2former(
        cls, model_name: str, device: str = "cuda"
    ) -> Tuple[Any, Any, str]:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        return model, processor, "mask2former"

    @classmethod
    def print_mask2former_params(
        cls,
        name: str = "facebook/mask2former-swin-base-ade-semantic",
        device: str = "cuda",
    ) -> None:
        """Вывод параметров Mask2Former"""
        processor = Mask2FormerImageProcessor.from_pretrained(name)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(name).to(device)
        print(processor)
        print(model)
        print(f"✅ Mask2Former загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== MASKFORMER ==========
    @classmethod
    def _load_maskformer(
        cls, model_name: str, device: str = "cuda"
    ) -> Tuple[Any, Any, str]:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        processor = MaskFormerImageProcessor.from_pretrained(model_name)
        model = (
            MaskFormerForInstanceSegmentation.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        return model, processor, "maskformer"

    @classmethod
    def print_maskformer_params(
        cls,
        model_name: str = "facebook/maskformer-resnet50-ade20k-full",
        device: str = "cuda",
    ) -> None:
        """Вывод параметров MaskFormer"""
        processor = MaskFormerImageProcessor.from_pretrained(model_name)
        model = MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(device)
        print(processor)
        print(model)
        print(f"✅ MaskFormer загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== ONEFORMER ==========
    @classmethod
    def _load_oneformer(
        cls, model_name: str, device: str = "cuda"
    ) -> Tuple[Any, Any, str]:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        files = list_repo_files(model_name)
        print(
            "✅ safetensors found!"
            if any(f.endswith(".safetensors") for f in files)
            else "❌ Только pickle (.bin)"
        )
        processor = OneFormerProcessor.from_pretrained(model_name)
        model = (
            OneFormerForUniversalSegmentation.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        return model, processor, "oneformer"

    @classmethod
    def print_oneformer_params(
        cls, name: str = "shi-labs/oneformer_ade20k_swin_large", device: str = "cuda"
    ) -> None:
        """Вывод параметров OneFormer"""
        files = list_repo_files(name)
        print(
            "✅ safetensors found!"
            if any(f.endswith(".safetensors") for f in files)
            else "❌ Только pickle (.bin)"
        )
        processor = OneFormerProcessor.from_pretrained(name)
        model = OneFormerForUniversalSegmentation.from_pretrained(name).to(device)
        print(processor)
        print(model)
        print(f"✅ OneFormer загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== DPT ==========
    @classmethod
    def _load_dpt(cls, model_name: str, device: str = "cuda") -> Tuple[Any, Any, str]:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
        return model, processor, "dpt"

    @classmethod
    def print_dpt_params(
        cls, model_name: str = "Intel/dpt-large-ade", device: str = "cuda"
    ) -> None:
        """Вывод параметров DPT"""
        processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForSemanticSegmentation.from_pretrained(model_name).to(device)
        print(processor)
        print(model)
        print(f"✅ DPT загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== UPERNET ==========
    @classmethod
    def _load_upernet(
        cls, model_name: str, device: str = "cuda"
    ) -> Tuple[Any, Any, str]:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = (
            AutoModelForSemanticSegmentation.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        return model, processor, "upernet"

    @classmethod
    def print_upernet_params(
        cls, model_name: str = "openmmlab/upernet-convnext-small", device: str = "cuda"
    ) -> None:
        """Вывод параметров UPerNet"""
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)
        print(processor)
        print(model)
        print(f"✅ UPerNet загружена!")
        print(f"   Устройство: {device}")
        print(model.config)

    # ========== DEEPLAB_TV ==========
    @classmethod
    def _load_deeplab_tv(
        cls, device: str, num_classes: int, checkpoint_path: str = "model_path"
    ) -> Tuple[Any, Any, str]:
        num_classes = int(num_classes)
        if checkpoint_path and os.path.exists(checkpoint_path):
            model = tv_seg.deeplabv3_resnet101(weights=None)
            model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model_keys = {
                k: v
                for k, v in checkpoint.items()
                if not k.startswith("aux_classifier")
            }
            model.load_state_dict(model_keys, strict=False)
            print(f"✅ Loaded DeepLabV3+ from checkpoint: {checkpoint_path}")
        else:
            model = tv_seg.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")
            model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
            print(f"⚠️ Checkpoint not found, using COCO weights")

        model = model.to(device).eval()
        return model, None, "deeplab_tv"

    @classmethod
    def print_deeplab_params(cls, device: str = "cuda") -> None:
        """Вывод параметров DeepLab"""
        model = tv_seg.deeplabv3_resnet101(pretrained=True)
        print(model)
        print(f"✅ DeepLab загружена!")
        print(f"   Устройство: {device}")

    @classmethod
    def _load_unet_smp(
        cls,
        device: str,
        num_classes: int,
        checkpoint_path: str = "unet_smp.pth",
        encoder_name: str = "resnet34",
        **kwargs,
    ) -> Tuple[Any, Any, str]:
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
        else:
            print(f"⚠️ Checkpoint not found, using ImageNet encoder only")
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
        print(f"✅ U-Net загружена!")
        print(f"   Устройство: {device}")

    # ========== FPN_SMP ==========
    @classmethod
    def _load_fpn_smp(
        cls,
        device: str,
        num_classes: int,
        checkpoint_path: str = "fpn_smp.pth",
        encoder_name: str = "mit_b5",
        **kwargs,
    ) -> Tuple[Any, Any, str]:
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
        else:
            print(f"⚠️ Checkpoint not found, using ImageNet encoder only")

        model = model.to(device).eval()
        model.output_stride = 32
        model.target_size = (512, 512)
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
        print(f"✅ FPN загружена!")
        print(f"   Устройство: {device}")

    # ========== PSP_SMP ==========
    @classmethod
    def _load_psp_smp(
        cls,
        device: str,
        num_classes: int,
        checkpoint_path: str = "psp_smp.pth",
        encoder_name: str = "mit_b5",
        **kwargs,
    ) -> Tuple[Any, Any, str]:
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

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded PSPNet from checkpoint: {checkpoint_path}")
        else:
            print(f"⚠️ Checkpoint not found, using ImageNet encoder only")

        model = model.to(device).eval()
        if "mit" in encoder_name:
            model_type_str = "psp_mit"
        elif "efficientnet" in encoder_name:
            model_type_str = "psp_effnet"
        elif "resnet" in encoder_name:
            model_type_str = "psp_resnet"
        else:
            model_type_str = "psp_mit"
        model.output_stride = 8
        model.target_size = (512, 512)
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
        print(f"✅ PSPNet загружена!")
        print(f"   Устройство: {device}")

    # ========== FCN_TV ==========
    @classmethod
    def _load_fcn_tv(
        cls,
        device: str,
        num_classes: int,
        checkpoint_path: str = "fcn_resnet50.pth",
        variant: str = "fcn_resnet50",
        **kwargs,
    ) -> Tuple[Any, Any, str]:
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

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Loaded FCN from checkpoint: {checkpoint_path}")
        else:
            print(f"⚠️ Checkpoint not found, using ImageNet backbone only")
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
        print(f"✅ FCN загружена!")
        print(f"   Устройство: {device}")

    # ========== SEGNET ==========
    @classmethod
    def _load_segnet(
        cls,
        device: str,
        num_classes: int,
        checkpoint_path: str = "segnet.pth",
        encoder_name: str = "resnet34",
        **kwargs,
    ) -> Tuple[Any, Any, str]:
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
            print(f"✅ Loaded SegNet from checkpoint: {checkpoint_path}")
        else:
            print(f"⚠️ Checkpoint not found, using ImageNet encoder only")
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
            print(f"✅ SegNet загружена!")
            print(f"   Устройство: {device}")
        except Exception as e:
            print(f"⚠️ SegNet not available in SMP: {e}")

    # ========== SAM ==========
    @classmethod
    def _load_sam(cls, model_name: str, device: str = "cuda") -> Tuple[Any, Any, str]:
        if not SAM_AVAILABLE:
            raise ImportError("ultralytics library required for SAM")

        if os.path.exists(model_name):
            print(
                f"   📁 Found: {model_name} ({os.path.getsize(model_name) / 1024**2:.3f} MB)"
            )
        elif not model_name.startswith("sam2"):
            print(f"   ⚠️ Warning: {model_name} not found in current directory")

        model = SAM(model_name)

        if "sam2" in model_name.lower():
            model_key = "sam2"
            model_type = "sam2"
        else:
            model_key = "sam"
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
        print(f"✅ SAM загружена!")
        print(f"   Устройство: {device}")

    # ========== MASK R-CNN ==========
    @classmethod
    def _load_maskrcnn_tv(
        cls, device: str, variant: str = "maskrcnn_resnet50_fpn", **kwargs
    ) -> Tuple[Any, Any, str]:
        variants = {
            "maskrcnn_resnet50_fpn": tv_det.maskrcnn_resnet50_fpn,
            "maskrcnn_resnet50_fpn_v2": tv_det.maskrcnn_resnet50_fpn_v2,
        }

        if variant not in variants:
            raise ValueError(f"Unknown Mask R-CNN variant: {variant}")
        model = variants[variant](weights="COCO_V1")
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
        model = variants[variant](weights="DEFAULT", pretrained=False)
        model = model.to(device)
        print(model)
        print(f"✅ Mask R-CNN загружена!")
        print(f"   Устройство: {device}")

    # ========== УНИВЕРСАЛЬНЫЙ ЗАГРУЗЧИК SMP ==========
    @classmethod
    def load_smp_model(
        cls,
        architecture: str,
        encoder_name: str,
        model_type: Optional[str] = None,
        key: Optional[str] = None,
        device: str = "cuda",
        num_classes: int = num_classes,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[Any, Any, str]:
        """
        Универсальный загрузчик для SMP-моделей.

        Args:
            architecture: Архитектура модели ('unet', 'fpn', 'pspnet', 'deeplabv3+')
            encoder_name: Название encoder ('resnet34', 'mit_b5', 'efficientnet-b0', etc.)
            model_type: Тип модели для dispatch
            key: Уникальный ключ для доступа к результатам
            checkpoint_path: Путь к чекпоинту
        """
        num_classes = int(num_classes)
        architectures = {
            "unet": smp.Unet,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
            "deeplabv3+": smp.DeepLabV3Plus,
        }

        if architecture not in architectures:
            raise ValueError(f"Unknown architecture: {architecture}")

        ModelClass = architectures[architecture]
        extra_kwargs = {}
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

    # ========== ЗАГРУЗКА ВСЕХ CNN МОДЕЛЕЙ ==========
    @classmethod
    def load_all_pretrained_cnn(
        cls,
        checkpoint_dir: str = "./checkpoints",
        device: str = "cuda",
        num_classes: int = num_classes,
    ) -> Dict[str, Any]:
        """
        Загружает все CNN-модели с предобученными весами для бенчмарка.

        Returns:
            Dict[str, Tuple]: Словарь {model_key: (model, processor, model_type)}
        """
        num_classes = int(num_classes)
        models_dict = {}
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
