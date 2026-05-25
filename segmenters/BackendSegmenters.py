# segmenters/BackendSegmenters.py

"""Модуль высокопроизводительных сегментеров на базе ONNX Runtime и TensorRT.

Предназначен для ускоренного инференса предварительно экспортированных моделей
сегментации с поддержкой аппаратного ускорения на GPU.

Классы:
- 🟦 ONNXSegmenter: Инференс через ONNX Runtime с поддержкой CUDA/CPU провайдеров.
  • Автоматический выбор Execution Provider (CUDAExecutionProvider → CPUExecutionProvider)
  • Графические оптимизации: ORT_ENABLE_ALL для максимального ускорения
  • Гибкая конфигурация входной формы: input_shape=(B, C, H, W)
  • Устойчивость к ошибкам: возврат пустой маски при сбое инференса

- 🟩 TRTSegmenter: Инференс через TensorRT (torch_tensorrt) для максимальной производительности.
  • Загрузка моделей из .trt файлов или готовых torch.nn.Module
  • Автоматическая конвертация: numpy → torch.Tensor → TRT inference → numpy
  • Только CUDA: TensorRT требует наличия GPU с поддержкой CUDA
  • Оптимизированный конвейер: нормализация [0,1], batch-обработка, пост-процессинг

Особенности реализации:
- 🔄 Единый интерфейс: оба класса наследуют `BaseSegmenter` и реализуют `segment()` / `segment_with_mask()`.
- 🎚️ Автоматическая предобработка: конвертация в 3 канала, нормализация к [0, 1], добавление batch-измерения.
- 🛡️ Устойчивость к ошибкам: при исключении в `segment()` возвращается пустая маска `(H, W)` вместо `None`.
- 📦 Поддержка форматов: вход — `np.ndarray`, `PIL.Image`, `str` (путь), `torch.Tensor`; выход — `np.ndarray` uint8 {0, 255}.
- ⚡ Оптимизации: `torch.no_grad()` для TRT, `SessionOptions` для ONNX, кэширование сессий/моделей.
- 🔍 Логирование: информативные сообщения об инициализации, формах тензоров и ошибках инференса.

Ограничения:
- Вероятностные маски не поддерживаются: `segment_with_mask()` возвращает `(binary_mask, None)`.
- TRTSegmenter требует наличия CUDA-совместимого GPU и установленных `torch-tensorrt`, `tensorrt`.
- ONNX-модели должны быть экспортированы с фиксированной или динамической входной формой, совместимой с `input_shape`.

Workflow:
1. Экспортировать обученную модель в ONNX (.onnx) или TensorRT (.trt) формат.
2. Инициализировать сегментер: `ONNXSegmenter("method", "model.onnx", device="cuda")`.
3. Выполнить инференс: `mask = segmenter.segment(image)`.
4. (Опционально) Использовать в бенчмарках через `CpuCudaBenchmark` или `SegmentationBenchmark`.

Примечание:
- Для классических методов (пороги, границы) используйте `OpenCVSegmenter` / `SklearnSegmenter`.
- Для нейросетевых моделей в PyTorch — `TorchSegmenter` / `TorchSegmenter2`.
- Данный модуль предназначен для **деплоя и бенчмаркинга** уже обученных моделей.
- Функция `load_trt_model()` импортируется из `utils.backend_exporter_new` — убедитесь в её наличии.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

from segmenters.BaseSegmenter import (
    BaseSegmenter,
    ImageInput,
    BinaryMask,
    ProbabilityMask,
)
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
from PIL import Image as PILImage
from scipy.ndimage import zoom
import cv2

from typing import Optional, Set, Dict, Any, Union, Literal, Tuple, List, Sequence, cast, TYPE_CHECKING, TypeAlias
import logging
from PIL import Image

if TYPE_CHECKING:
    from torch_tensorrt import Module as TRTModule

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
ONNXSession: TypeAlias = ort.InferenceSession
"""Тип сессии ONNX Runtime, dtype=ort.InferenceSession."""

TRTModel: TypeAlias = Union[torch.nn.Module, "TRTModule"]
"""Тип модели TensorRT: PyTorch Module или скомпилированный TRT модуль, dtype=Union[torch.nn.Module, "TRTModule"]."""

PreprocessedTensor: TypeAlias = npt.NDArray[np.float32]
"""Тип предобработанного тензора: (1, 3, H, W), float32, [0, 1], dtype=npt.NDArray[np.float32]."""

RawOutput: TypeAlias = npt.NDArray[Any]
"""Тип сырого вывода модели: может быть любой размерности и dtype, dtype=npt.NDArray[Any]."""

OnnxProvider: TypeAlias = Union[str, Tuple[str, Dict[str, Any]]]
"""Тип провайдера ONNX Runtime: либо строка, либо (имя, опции), dtype=Union[str, Tuple[str, Dict[str, Any]]]."""

DeviceType: TypeAlias = Literal["cuda", "cpu"]
"""Тип текущего устройства, dtype=Literal["cuda", "cpu"]."""

NormalizationType: TypeAlias = Literal["imagenet", "none", "custom"]
"""Тип нормализации (imagenet/none/custom), dtype=Literal["imagenet", "none", "custom"]."""

InputShape: TypeAlias = Tuple[int, int, int, int]
"""Размер исходного изображения, dtype=Tuple[int, int, int, int]."""

PrecisionType: TypeAlias = Literal["fp32", "fp16", "bf16"]
"""Тип для указания точности вычислений, dtype=Literal["fp32", "fp16", "bf16"]."""


def _is_neural_by_key(model_key: str) -> bool:
    """Авто-определение: нейросеть или классический метод по ключу."""
    neural_keywords: Set[str] = {
        "unet",
        "fpn",
        "psp",
        "deeplab",
        "fcn",
        "segnet",
        "segformer",
        "mask2former",
        "oneformer",
        "sam",
    }
    return any(kw in model_key.lower() for kw in neural_keywords)


class ONNXSegmenter(BaseSegmenter):
    """Сегментер нейронной модели через ONNX Runtime + CUDAExecutionProvider.

    Работает для всех архитектур (U-Net, FPN, PSPNet, SegNet, DeepLab, FCN).
    Возвращает многоклассовую маску (H, W) с индексами классов 0..N-1,
    а не бинарную {0, 255} как ONNXSegmenter для классических методов.

    Используется когда TRT engine не удалось собрать через torch_tensorrt.
    """

    def __init__(
        self,
        model_key: str,
        onnx_path: str,
        num_classes: int = 150,
        device: DeviceType = "cuda",
        input_shape: InputShape = (1, 3, 512, 512),
        normalization: NormalizationType = "imagenet",
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        is_neural: Optional[bool] = None,
        use_tensorrt_ep: bool = False,
        trt_options: Optional[Dict[str, Any]] = None,
        trt_cache_path: Optional[str] = None,
        precision: PrecisionType = "fp32",
        **kwargs: Any,
    ) -> None:
        """Инициализация ONNX сегментера.

        Args:
            model_key: Имя метода для логирования.
            onnx_path: Путь к .onnx файлу модели.
            num_classes: Число классов экспортируемой модели.
            device: Устройство выполнения: "cuda" или "cpu".
            input_shape: Ожидаемая форма входа: (B, C, H, W).
            normalization: Параметр нормализации (по уиолчанию "imagenet").
            mean: Среднее (может быть None).
            std: Среднеквадратичное отклонение (может быть None).
            is_neural: Флаг проверки на нейронную модель.
            use_tensorrt_ep: Использовать TensorRT Execution Provider.
            trt_options: Опции для TensorRT EP.
            trt_cache_path: Путь для кэша TRT engine.
            precision: Точность вычислений для TRT ("fp32", "fp16", "bf16").
            **kwargs: Дополнительные параметры.
        """
        super().__init__()
        self.method: str = model_key
        self.params: Dict[str, Any] = kwargs
        self.num_classes: int = num_classes
        self.device: torch.device = torch.device(device)
        self.input_shape: InputShape = input_shape
        self.is_neural: bool = is_neural if is_neural is not None else _is_neural_by_key(model_key)

        self.use_tensorrt_ep: bool = use_tensorrt_ep
        self.trt_options: Optional[Dict[str, Any]] = trt_options
        self.trt_cache_path: Optional[str] = trt_cache_path
        self.precision: PrecisionType = precision

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime-gpu не установлен: pip install onnxruntime-gpu")
        
        # ──────────────────────────────────────────────────────────────
        # Формирование списка провайдеров с поддержкой TRT EP
        # ──────────────────────────────────────────────────────────────
        providers: List[OnnxProvider] = []

        if self.device.type == "cuda" and self.use_tensorrt_ep:
            if "TensorrtExecutionProvider" in ort.get_available_providers():
                # Базовые опции для TRT EP
                trt_ep_opts: Dict[str, Any] = {
                    "device_id": 0,
                    "trt_fp16_enable": (precision in ["fp16", "bf16"]),
                    "trt_int8_enable": False,
                    # "trt_engine_cache_enable": True,
                    # "trt_engine_cache_path": trt_cache_path or f"./trt_cache/{precision}/{model_key}",
                    # "trt_timing_cache_enable": True,
                    # "trt_timing_cache_path": trt_cache_path or f"./trt_cache/{precision}/{model_key}/timing",
                    "trt_engine_cache_enable": False,      # ← Ключевое!
                    "trt_timing_cache_enable": False,      # ← И это!
                    "trt_builder_optimization_level": 5,
                    "trt_max_workspace_size": 1 << 30,  # 1GB
                }
                # Объединяем с пользовательскими опциями
                if trt_options:
                    trt_ep_opts.update({k: v for k, v in trt_options.items() if k not in trt_ep_opts})
                
                providers.append(("TensorrtExecutionProvider", trt_ep_opts))
                
                # CUDA EP как fallback
                providers.append((
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        # "do_copy_in_default_stream": False,  # ← Разрешить non-default stream
                        # "enable_cuda_graph": True,  # ← Опционально: CUDA Graphs для ускорения
                        # "cudnn_conv_use_max_workspace": True,
                        "do_copy_in_default_stream": True,
                    },
                ))
                logger.info(f"✅ {model_key}: активирован TensorrtExecutionProvider ({precision})")
            else:
                logger.warning(f"⚠️ {model_key}: TensorrtExecutionProvider не доступен, используем CUDA EP")
                providers.append((
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        # "do_copy_in_default_stream": False,  # ← Разрешить non-default stream
                        # "enable_cuda_graph": True,  # ← Опционально: CUDA Graphs для ускорения
                        # "cudnn_conv_use_max_workspace": True,
                        "do_copy_in_default_stream": True,
                    },
                ))
        elif self.device.type == "cuda":
            # Стандартный CUDA EP без TRT
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ))
        
        # CPU как последний fallback
        providers.append("CPUExecutionProvider")
        
        # ──────────────────────────────────────────────────────────────
        # Создание сессии
        # ──────────────────────────────────────────────────────────────
        sess_options: ort.SessionOptions = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True

        self.sess_options = sess_options

        self.session: ONNXSession = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name

        print(f"🔍 Available outputs: {[o.name for o in self.session.get_outputs()]}")
        print(f"🔍 Expected output_name: {self.output_name}")

        test_input = np.random.randn(*input_shape).astype(np.float32)
        test_output = self.session.run(None, {self.input_name: test_input})
        logger.info(f"Test run: output shape={test_output[0].shape if test_output else None}, "
            f"has_nan={np.isnan(test_output[0]).any() if test_output else 'N/A'}")
        
        if not self._validate_session():
            logger.warning(f"⚠️ {model_key}: сессия не прошла валидацию, возможны проблемы при инференсе")

        # Проверяем реальный output shape из модели
        out_shape: Tuple[Union[str, int], ...] = self.session.get_outputs()[0].shape
        logger.info(
            f"ONNX '{model_key}': input={self.input_name}, " f"output={self.output_name}, output_shape={out_shape}"
        )
        active: Sequence[str] = self.session.get_providers()
        logger.info(
            f"NeuralONNXSegmenter '{model_key}': providers={active}, " f"output={self.session.get_outputs()[0].shape}"
        )
        if normalization == "imagenet":
            self.mean: Optional[np.ndarray] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std: Optional[np.ndarray] = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        elif normalization == "custom" and mean and std:
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)
        else:
            self.mean = None
            self.std = None

    def _validate_session(self) -> bool:
        """Проверка работоспособности сессии."""
        try:
            test_input = np.random.randn(*self.input_shape).astype(np.float32)
            outputs = self.session.run(None, {self.input_name: test_input})
            if outputs and outputs[0] is not None:
                logger.info(f"✅ Session validation OK: {outputs[0].shape}")
                return True
            else:
                logger.error(f"❌ Session validation failed: empty output")
                return False
        except Exception as e:
            logger.error(f"❌ Session validation error: {e}")
            return False

    def _preprocess_classic(self, image: npt.NDArray[np.uint8]) -> PreprocessedTensor:
        """Конвертирует изображение в формат (1, 3, H, W), float32, [0, 1].

        Args:
            image: Входное изображение (H, W) или (H, W, 3), uint8.

        Returns:
            PreprocessedTensor: Тензор формы (1, 3, H, W), float32, нормализованный к [0, 1].
        """
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim == 3 and image.shape[2] == 3:
            tensor: npt.NDArray[np.float32] = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
            return np.expand_dims(tensor, 0)  # (1,3,H,W)
        # Уже (B,C,H,W)
        return image.astype(np.float32)

    def _preprocess_neural(self, image: npt.NDArray[np.uint8]) -> PreprocessedTensor:
        """Imagenet нормализация: конвертирует изображение в формат (1, 3, H, W), float32, [0, 1].

        Args:
            image: Входное изображение (H, W) или (H, W, 3), uint8.

        Returns:
            PreprocessedTensor: Тензор формы (1, 3, H, W), float32, нормализованный к [0, 1].
        """
        if image.ndim == 3 and image.shape[2] != 3:
            image = image[:, :, :3]
        elif image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Resize к целевому размеру
        h_target, w_target = self.input_shape[2], self.input_shape[3]
        if image.shape[0] != h_target or image.shape[1] != w_target:
            pil: Image.Image = PILImage.fromarray(image.astype(np.uint8))
            pil = pil.resize((w_target, h_target), PILImage.Resampling.BILINEAR)
            image = np.array(pil)

        # ImageNet нормализация
        if self.mean is not None and self.std is not None:
            tensor: npt.NDArray[np.float32] = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        else:
            tensor = image.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(tensor, 0)  # (1, C, H, W)

    def segment(self, image: Union[str, np.ndarray, Image.Image, torch.Tensor], **kwargs: Any) -> BinaryMask:
        """Запускает ONNX инференс.

        Алгоритм:
        1. Предобработка изображения → (1, 3, H, W), float32, [0, 1].
        2. Запуск сессии ONNX.
        3. Пост-обработка вывода: squeeze → бинаризация → uint8.

        Args:
            image: Входное изображение (H, W) или (H, W, 3), uint8.
            **kwargs: Дополнительные параметры (игнорируются).

        Returns:
            BinaryMask: Бинарная маска формы (H, W), uint8, значения {0, 255}.
                       При ошибке возвращает пустую маску того же размера.
        """
        # 1. Приведение входа к numpy
        if isinstance(image, str):
            image_np: np.ndarray = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = image
        orig_h, orig_w = image_np.shape[:2]
        try:
            if self.is_neural:
                tensor: PreprocessedTensor = self._preprocess_neural(image_np)
            else:
                tensor = self._preprocess_classic(image_np)
            outputs: List[RawOutput] = self.session.run([self.output_name], {self.input_name: tensor})
            if not outputs or outputs[0] is None:
                logger.error(f"ONNX '{self.method}' returned None/empty output")
                return np.zeros((orig_h, orig_w), dtype=np.uint8)
            logits: RawOutput = outputs[0]  # (1, C, H, W) или (1, H, W)

            # 3. Интеллектуальная пост-обработка
            # Case A: Нейросеть (Batch, Channels, H, W)
            if logits.ndim == 4 and logits.shape[1] > 1:
                # Многоклассовая: argmax
                mask: np.ndarray = logits[0].argmax(axis=0).astype(np.uint8)

            # Case B: Бинарная нейросеть или классика (Batch, 1, H, W) или (Batch, H, W)
            elif logits.ndim == 4 and logits.shape[1] == 1:
                mask = logits[0, 0]
                if mask.max() <= 1.0:
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask = mask.astype(np.uint8)

            elif logits.ndim == 3:
                # (1, H, W)
                mask = logits[0]
                if mask.max() <= 1.0:
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask = mask.astype(np.uint8)

            # Case C: Уже маска (H, W)
            elif logits.ndim == 2:
                mask = logits
                if mask.max() <= 1.0:
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask = mask.astype(np.uint8)

            else:
                # Fallback
                mask = logits.flatten().reshape((orig_h, orig_w)).astype(np.uint8)

            # Resize к оригинальному размеру
            if mask.shape != (orig_h, orig_w):
                from scipy.ndimage import zoom

                sh, sw = orig_h / mask.shape[0], orig_w / mask.shape[1]
                mask = zoom(mask, (sh, sw), order=0).astype(np.uint8)

            return cast(BinaryMask, mask)

        except Exception as e:
            logger.error(f"ONNX '{self.method}' inference error: {e}")
            return np.zeros((orig_h, orig_w), dtype=np.uint8)

    def segment_with_mask(
        self, image: ImageInput, alpha: float = 0.9, **kwargs: Any
    ) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        """Сегментация с возвратом визуализации и бинарной маски.

        Создаёт наложение маски на оригинальное изображение с прозрачностью `alpha`.
        Стиль визуализации соответствует другим сегментерам (OpenCV/Sklearn/Torch).

        Args:
            image: Входное изображение.
            alpha: Коэффициент наложения маски [0, 1]:
                - 0.0 = только оригинальное изображение
                - 1.0 = только маска (красным цветом)
                - 0.9 = по умолчанию (сильный акцент на маске)
            **kwargs: Дополнительные параметры:
                - return_probs (bool): Если True, возвращать вероятностную маску.
                - prob_class (int): Для multi-class: индекс класса для вероятностной маски.
                - prob_threshold (float): Порог для бинаризации вероятностей (по умолчанию 0.5).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - `overlay`: Визуализация формы `(H, W, 3)`, dtype=uint8, RGB.
                - `mask`: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Маска накладывается красным цветом `[255, 0, 0]` для пикселей > 127.
            - Grayscale изображения автоматически конвертируются в 3-канальные для наложения.
            - Формула смешивания: `result = overlay * alpha + original * (1 - alpha)`.
        """
        # ──────────────────────────────────────────────────────────────
        # 1. Конвертация входного изображения в numpy
        # ──────────────────────────────────────────────────────────────
        if isinstance(image, str):
            image_np: np.ndarray = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        elif isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
                image_np = np.transpose(image_np, (1, 2, 0))
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
        elif isinstance(image, np.ndarray):
            image_np = image
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            elif image_np.shape[2] != 3:
                image_np = image_np[:, :, :3]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_h, orig_w = image_np.shape[:2]

        try:
            # ──────────────────────────────────────────────────────────────
            # 2. Предобработка (аналогично методу segment)
            # ──────────────────────────────────────────────────────────────
            if self.is_neural:
                tensor_np: PreprocessedTensor = self._preprocess_neural(image_np)
            else:
                tensor_np = self._preprocess_classic(image_np)

            # ──────────────────────────────────────────────────────────────
            # 3. Инференс модели
            # ──────────────────────────────────────────────────────────────
            # outputs: List[RawOutput] = self.session.run([self.output_name], {self.input_name: tensor_np})

            # if not outputs or outputs[0] is None:
            #     logger.error(f"ONNX '{self.method}' returned None output")
            #     return image_np.copy(), np.zeros((orig_h, orig_w), dtype=np.uint8)

            # logits: RawOutput = outputs[0]

            logger.debug(f"=== ONNX Inference Debug ===")
            logger.debug(f"Input: name={self.input_name}, shape={tensor_np.shape}, dtype={tensor_np.dtype}, range=[{tensor_np.min():.3f}, {tensor_np.max():.3f}]")
            logger.debug(f"Output expected: name={self.output_name}, shape={self.session.get_outputs()[0].shape}")
            logger.debug(f"Active providers: {self.session.get_providers()}")
            logger.debug(f"Session options: graph_opt={self.sess_options.graph_optimization_level}")

            if not tensor_np.flags['C_CONTIGUOUS']:
                logger.warning("⚠️ Input tensor not C-contiguous, making copy")
                tensor_np = np.ascontiguousarray(tensor_np)

            outputs: List[RawOutput] = self.session.run([self.output_name], {self.input_name: tensor_np})
            if not outputs or outputs[0] is None:
                logger.error(f"❌ session.run returned empty/None!")
                logger.error(f"Available outputs: {[o.name for o in self.session.get_outputs()]}")
                all_outputs = self.session.run(None, {self.input_name: tensor_np})
                logger.error(f"All outputs count: {len(all_outputs) if all_outputs else 0}")

                logger.info(f"  Input: shape={tensor_np.shape}, dtype={tensor_np.dtype}, range=[{tensor_np.min():.3f}, {tensor_np.max():.3f}]")
                logger.info(f"  Providers: {self.session.get_providers()}")
                
                # 🔧 Fallback: попробовать с явным именем выхода
                try:
                    outputs = self.session.run([self.output_name], {self.input_name: tensor_np})
                    logger.info(f"  Retry with explicit output name: {len(outputs) if outputs else 'None'} outputs")
                    if outputs and outputs[0] is not None:
                        logger.info(f"✅ Fallback succeeded for {self.method}")
                    else:
                        logger.warning(f"⚠️ Fallback also failed for {self.method}")
                except Exception as e:
                    logger.error(f"  Fallback error: {e}")
                return image_np.copy(), np.zeros((orig_h, orig_w), dtype=np.uint8)
            logits: RawOutput = outputs[0]  # (1, C, H, W) или (1, H, W)
            if np.isnan(logits).any() or np.isinf(logits).any():
                logger.warning(f"⚠️ Logits contain NaN/Inf: nan={np.isnan(logits).sum()}, inf={np.isinf(logits).sum()}")
                logits = np.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=0.0)

            logger.info(f"Output: shape={logits.shape}, dtype={logits.dtype}, "
                 f"range=[{logits.min():.3f}, {logits.max():.3f}], has_nan={np.isnan(logits).any()}")


            # ──────────────────────────────────────────────────────────────
            # 4. Интеллектуальная пост-обработка (аналогично segment)
            # ──────────────────────────────────────────────────────────────
            return_probs: bool = kwargs.get("return_probs", False)
            prob_class: int = kwargs.get("prob_class", -1)
            prob_threshold: float = kwargs.get("prob_threshold", 0.5)
            prob_mask: Optional[ProbabilityMask] = None

            if logits.ndim == 4 and logits.shape[1] > 1:
                # Multi-class: argmax
                mask: np.ndarray = logits[0].argmax(axis=0).astype(np.uint8)
                if return_probs:
                    if 0 <= prob_class < logits.shape[1]:
                        probs = logits[0, prob_class]
                    else:
                        probs = np.max(logits[0], axis=0)
                    if probs.max() > 1.0 or probs.min() < 0:
                        exp_vals = np.exp(logits[0] - np.max(logits[0], axis=0, keepdims=True))
                        softmax = exp_vals / (np.sum(exp_vals, axis=0, keepdims=True) + 1e-8)
                        prob_mask = (
                            softmax[prob_class].astype(np.float32)
                            if 0 <= prob_class < logits.shape[1]
                            else np.max(softmax, axis=0).astype(np.float32)
                        )
                    else:
                        prob_mask = probs.astype(np.float32)
            else:
                # Binary
                raw_mask = logits[0, 0] if logits.ndim == 4 else (logits[0] if logits.ndim == 3 else logits)
                if raw_mask.max() <= 1.0:
                    mask = (raw_mask > prob_threshold).astype(np.uint8) * 255
                else:
                    mask = raw_mask.astype(np.uint8)
                if return_probs:
                    if raw_mask.max() <= 1.0 and raw_mask.min() >= 0:
                        prob_mask = raw_mask.astype(np.float32)
                    else:
                        prob_mask = 1.0 / (1.0 + np.exp(-np.clip(raw_mask.astype(np.float32), -50, 50)))

            # ──────────────────────────────────────────────────────────────
            # 5. Ресайз масок к оригинальному размеру изображения
            # ──────────────────────────────────────────────────────────────
            if mask.shape != (orig_h, orig_w):
                sh, sw = orig_h / mask.shape[0], orig_w / mask.shape[1]
                mask = zoom(mask, (sh, sw), order=0).astype(np.uint8)  # nearest для маски

            if prob_mask is not None and prob_mask.shape != (orig_h, orig_w):
                sh, sw = orig_h / prob_mask.shape[0], orig_w / prob_mask.shape[1]
                prob_mask = zoom(prob_mask, (sh, sw), order=1).astype(np.float32)  # linear для вероятностей

            # ──────────────────────────────────────────────────────────────
            # 5. 🔧 СОЗДАНИЕ ОВЕРЛЕЯ (как в OpenCV/Sklearn/Torch сегментерах)
            # ──────────────────────────────────────────────────────────────
            # Конвертируем исходное изображение в RGB если нужно
            if image_np.ndim == 2:
                base_img = np.stack([image_np] * 3, axis=-1)
            else:
                base_img = image_np.copy()

            # Создаём оверлей: копию изображения для наложения маски
            overlay = base_img.copy()

            # 🔧 FIX: Красный цвет для маски [255, 0, 0] — только там, где mask > 127
            mask_bool = mask > 127
            overlay[mask_bool] = [255, 0, 0]  # RGB: красный

            # 🔧 FIX: Правильная формула alpha blending (как в OpenCVSegmenter):
            # result = overlay * alpha + base_img * (1 - alpha)
            # Это даёт: красная маска с прозрачностью, оригинал виден под ней
            result = cv2.addWeighted(
                overlay.astype(np.float32), alpha, base_img.astype(np.float32), 1.0 - alpha, 0
            ).astype(np.uint8)

            return result, mask.astype(np.uint8)

        except Exception as e:
            logger.error(f"TRT '{self.method}' segment_with_mask error: {e}", exc_info=True)
            # Возвращаем пустые маски при ошибке
            return np.zeros((orig_h, orig_w), dtype=np.uint8), None


class TRTSegmenter(BaseSegmenter):
    """Сегментер нейронной модели через TensorRT engine.

    Поддерживает все форматы TRT:
      - Serialized engine (от tensorrt API / trtexec) через TrtEngineWrapper
      - TorchScript (от torch_tensorrt JIT) через torch.jit
      - OnnxCudaFallback (ONNX Runtime + CUDA EP) как fallback

    Загрузка через load_trt_model() из backend_exporter_new автоматически
    определяет формат и возвращает подходящую обёртку.
    """

    def __init__(
        self,
        model_key: str,
        trt_model_or_path: Union[str, TRTModel],
        num_classes: int = 150,
        input_shape: InputShape = (1, 3, 512, 512),
        device: DeviceType = "cuda",
        normalization: NormalizationType = "imagenet",
        normalization_in_graph: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        is_neural: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Инициализация TensorRT сегментера.

        Args:
            model_key: Имя метода для логирования.
            trt_model_or_path: Путь к .trt файлу или уже загруженная TRT-модель.
            num_classes: Число классов экспортируемой модели.
            input_shape: Ожидаемая форма входа: (B, C, H, W).
            device: Устройство выполнения: "cuda" или "cpu".
            normalization: Параметр нормализации (по уиолчанию "imagenet").
            normalization_in_graph: Нужно ли нормализовать в графе.
            mean: Среднее (может быть None).
            std: Среднеквадратичное отклонение (может быть None).
            is_neural: Флаг проверки на нейронную модель.
            **kwargs: Дополнительные параметры.
        """
        super().__init__()
        self.method: str = model_key
        self.params: Dict[str, Any] = kwargs
        self.num_classes: int = num_classes
        self.input_shape: InputShape = input_shape
        self.normalization_in_graph: bool = normalization_in_graph
        self.device: torch.device = torch.device(device)
        self.is_neural: bool = is_neural if is_neural is not None else _is_neural_by_key(model_key)
        if isinstance(trt_model_or_path, str):
            from utils.backend_exporter_new import load_trt_model

            loaded_model: Optional[TRTModel] = load_trt_model(path=trt_model_or_path, device=device)
            if loaded_model is None:
                raise RuntimeError(f"Не удалось загрузить TRT модель: {trt_model_or_path}")
            self.model: TRTModel = loaded_model
        else:
            self.model = trt_model_or_path
        self.model.eval()
        logger.info(f"NeuralTRTSegmenter '{model_key}': {type(self.model).__name__}")
        if normalization == "imagenet":
            self.mean: Optional[np.ndarray] = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std: Optional[np.ndarray] = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        elif normalization == "custom" and mean and std:
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)
        else:
            self.mean = None
            self.std = None

    def _preprocess_torch(self, image: np.ndarray) -> torch.Tensor:
        """Imagenet нормализация → torch.Tensor (1, 3, H, W) на GPU."""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] != 3:
            image = image[:, :, :3]

        h_t, w_t = self.input_shape[2], self.input_shape[3]
        if image.shape[0] != h_t or image.shape[1] != w_t:
            pil: Image.Image = PILImage.fromarray(image.astype(np.uint8))
            image = np.array(pil.resize((w_t, h_t), PILImage.Resampling.BILINEAR))

        if self.normalization_in_graph:
            tensor: npt.NDArray[np.float32] = image.astype(np.float32) / 255.0  # Только масштабирование
        elif self.mean is not None and self.std is not None:
            tensor = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        else:
            tensor = image.astype(np.float32) / 255.0

        tensor_torch: torch.Tensor = torch.from_numpy(np.transpose(tensor, (2, 0, 1))).unsqueeze(0).to(self.device)
        return tensor_torch

    def _preprocess_classic(self, image: npt.NDArray[np.uint8]) -> PreprocessedTensor:
        """Конвертирует изображение в формат (1, 3, H, W), float32, [0, 1].

        Args:
            image: Входное изображение (H, W) или (H, W, 3), uint8.

        Returns:
            PreprocessedTensor: Тензор формы (1, 3, H, W), float32, нормализованный к [0, 1].
        """
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim == 3 and image.shape[2] == 3:
            tensor: npt.NDArray[np.float32] = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
            return np.expand_dims(tensor, 0)  # (1,3,H,W)
        # Уже (B,C,H,W)
        return image.astype(np.float32)

    def _preprocess_neural(self, image: npt.NDArray[np.uint8]) -> PreprocessedTensor:
        """Imagenet нормализация: конвертирует изображение в формат (1, 3, H, W), float32, [0, 1].

        Args:
            image: Входное изображение (H, W) или (H, W, 3), uint8.

        Returns:
            PreprocessedTensor: Тензор формы (1, 3, H, W), float32, нормализованный к [0, 1].
        """
        if image.ndim == 3 and image.shape[2] != 3:
            image = image[:, :, :3]
        elif image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Resize к целевому размеру
        h_target, w_target = self.input_shape[2], self.input_shape[3]
        if image.shape[0] != h_target or image.shape[1] != w_target:
            pil: Image.Image = PILImage.fromarray(image.astype(np.uint8))
            pil = pil.resize((w_target, h_target), PILImage.Resampling.BILINEAR)
            image = np.array(pil)

        # ImageNet нормализация
        if self.mean is not None and self.std is not None:
            tensor: npt.NDArray[np.float32] = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        else:
            tensor = image.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(tensor, 0)  # (1, C, H, W)

    def segment(self, image: ImageInput, **kwargs: Any) -> BinaryMask:
        """Запускает TRT инференс.

        Алгоритм:
        1. Конвертация numpy → torch.Tensor (C, H, W), float32, [0, 1].
        2. Добавление batch-измерения и перенос на устройство.
        3. Инференс модели с torch.no_grad().
        4. Пост-обработка: squeeze → numpy → uint8 [0, 255].

        Args:
            image: Входное изображение (H, W) или (H, W, 3), uint8.
            **kwargs: Дополнительные параметры (игнорируются).

        Returns:
            BinaryMask: Бинарная маска формы (H, W), uint8, значения {0, 255}.
                       При ошибке возвращает пустую маску того же размера.
        """
        # 1. Приведение входа
        if isinstance(image, str):
            image_np: np.ndarray = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = image
        orig_h, orig_w = image_np.shape[:2]
        try:
            if self.is_neural:
                tensor_np = self._preprocess_neural(image_np)
            else:
                tensor_np = self._preprocess_classic(image_np)

            tensor = torch.from_numpy(tensor_np).to(self.device) if isinstance(tensor_np, np.ndarray) else tensor_np

            # 3. Инференс
            use_stream: bool = kwargs.get("use_cuda_stream", True)

            if use_stream and self.device.type == "cuda":
                # Создаём или используем кэшированный stream
                if not hasattr(self, "_inference_stream"):
                    self._inference_stream: torch.cuda.Stream = torch.cuda.Stream(device=self.device)

                with torch.cuda.stream(self._inference_stream):
                    with torch.no_grad():
                        out = self.model(tensor)
                    # Синхронизация только этого stream перед копированием на CPU
                    self._inference_stream.synchronize()
            else:
                # Fallback: выполнение в дефолтном stream
                with torch.no_grad():
                    out = self.model(tensor)

            if isinstance(out, torch.Tensor):
                out_np: np.ndarray = out.cpu().float().numpy()
            elif isinstance(out, np.ndarray):
                out_np = out.astype(np.float32)
            elif isinstance(out, (list, tuple)):
                # Если модель возвращает несколько выходов, берём первый
                first = out[0]
                if isinstance(first, torch.Tensor):
                    out_np = first.cpu().float().numpy()
                else:
                    out_np = np.asarray(first, dtype=np.float32)
            else:
                # Fallback: пытаемся конвертировать в массив
                out_np = np.asarray(out, dtype=np.float32)

            # 🔧 FIX: Явное приведение типов для linter
            out_np = cast(np.ndarray, out_np)

            # 4. Интеллектуальная пост-обработка (аналогично ONNX)
            if out_np.ndim == 4 and out_np.shape[1] > 1:
                # Multi-class
                # Case A: Нейросеть (Batch, Channels, H, W)
                mask: np.ndarray = out_np[0].argmax(axis=0).astype(np.uint8)
            # Case B: Бинарная нейросеть или классика (Batch, 1, H, W) или (Batch, H, W)
            elif out_np.ndim == 4 and out_np.shape[1] == 1:
                mask = out_np[0, 0]
                if mask.max() <= 1.0:
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask = mask.astype(np.uint8)
            elif out_np.ndim == 3:
                # (1, H, W)
                mask = out_np[0]
                if mask.max() <= 1.0:
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask = mask.astype(np.uint8)

            # Case C: Уже маска (H, W)
            elif out_np.ndim == 2:
                mask = out_np
                if mask.max() <= 1.0:
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask = mask.astype(np.uint8)

            else:
                # Fallback
                mask = out_np.flatten().reshape((orig_h, orig_w)).astype(np.uint8)

            # 5. Ресайз
            if mask.shape != (orig_h, orig_w):
                sh, sw = orig_h / mask.shape[0], orig_w / mask.shape[1]
                mask = zoom(mask, (sh, sw), order=0).astype(np.uint8)
            return cast(BinaryMask, mask)

        except Exception as e:
            logger.error(f"TRT '{self.method}' inference error: {e}")
            return np.zeros((orig_h, orig_w), dtype=np.uint8)

    def segment_with_mask(
        self, image: ImageInput, alpha: float = 0.9, **kwargs: Any
    ) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        """Сегментация с возвратом визуализации и бинарной маски.

        Для TRT-модели:
        - Бинарная маска: результат пост-обработки (argmax для multi-class, порог для binary)
        - Вероятностная маска: нормализованные вероятности/логиты модели (опционально)

        Создаёт наложение маски на оригинальное изображение с прозрачностью `alpha`.
        Стиль визуализации соответствует другим сегментерам (OpenCV/Sklearn/Torch).

        Алгоритм пост-обработки:
        1. Multi-class output (C>1): argmax → binary mask, softmax → prob mask
        2. Binary output (C=1): sigmoid/threshold → binary mask, raw logits → prob mask
        3. Ресайз масок к оригинальному размеру изображения

        Args:
            image: Входное изображение (путь, PIL, numpy или torch).
            alpha: Коэффициент наложения маски [0, 1]:
              - 0.0 = только оригинальное изображение
              - 1.0 = только маска (красным цветом)
              - 0.9 = по умолчанию (сильный акцент на маске)
            **kwargs: Дополнительные параметры:
                - return_probs (bool): Если True, возвращать вероятностную маску.
                - prob_class (int): Для multi-class: индекс класса для вероятностной маски.
                - prob_threshold (float): Порог для бинаризации вероятностей (по умолчанию 0.5).

        Returns:
            Tuple[BinaryMask, Optional[ProbabilityMask]]:
            - `overlay`: Визуализация формы `(H, W, 3)`, dtype=uint8, RGB.
            - `mask`: Бинарная маска формы `(H, W)`, dtype=uint8, {0, 255}.

        Note:
            - Маска накладывается красным цветом `[255, 0, 0]` для пикселей > 127.
            - Grayscale изображения автоматически конвертируются в 3-канальные для наложения.
            - Формула смешивания: `result = overlay * alpha + original * (1 - alpha)`.
        """
        # ──────────────────────────────────────────────────────────────
        # 1. Конвертация входного изображения в numpy
        # ──────────────────────────────────────────────────────────────
        if isinstance(image, str):
            image_np: np.ndarray = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        elif isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
                image_np = np.transpose(image_np, (1, 2, 0))
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
        elif isinstance(image, np.ndarray):
            image_np = image
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            elif image_np.shape[2] != 3:
                image_np = image_np[:, :, :3]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_h, orig_w = image_np.shape[:2]

        try:
            # ──────────────────────────────────────────────────────────────
            # 2. Предобработка (аналогично методу segment)
            # ──────────────────────────────────────────────────────────────
            if self.is_neural:
                tensor_np = self._preprocess_neural(image_np)
            else:
                tensor_np = self._preprocess_classic(image_np)

            # ──────────────────────────────────────────────────────────────
            # 3. Инференс модели
            # ──────────────────────────────────────────────────────────────
            tensor = torch.from_numpy(tensor_np).to(self.device) if isinstance(tensor_np, np.ndarray) else tensor_np
            use_stream = kwargs.get("use_cuda_stream", True)

            if use_stream and self.device.type == "cuda":
                if not hasattr(self, "_inference_stream"):
                    self._inference_stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(self._inference_stream):
                    with torch.no_grad():
                        out = self.model(tensor)
                    self._inference_stream.synchronize()
            else:
                with torch.no_grad():
                    out = self.model(tensor)

            if isinstance(out, torch.Tensor):
                logits: np.ndarray = out.cpu().float().numpy()
            elif isinstance(out, np.ndarray):
                logits = out.astype(np.float32)
            elif isinstance(out, (list, tuple)):
                # Если модель возвращает несколько выходов, берём первый
                first = out[0]
                if isinstance(first, torch.Tensor):
                    logits = first.cpu().float().numpy()
                else:
                    logits = np.asarray(first, dtype=np.float32)
            else:
                # Fallback: пытаемся конвертировать в массив
                logits = np.asarray(out, dtype=np.float32)

            # 🔧 FIX: Явное приведение типов для linter
            logits = cast(np.ndarray, logits)

            # ──────────────────────────────────────────────────────────────
            # 4. Интеллектуальная пост-обработка (аналогично segment)
            # ──────────────────────────────────────────────────────────────
            return_probs: bool = kwargs.get("return_probs", False)
            prob_class: int = kwargs.get("prob_class", -1)
            prob_threshold: float = kwargs.get("prob_threshold", 0.5)

            prob_mask: Optional[ProbabilityMask] = None

            if logits.ndim == 4 and logits.shape[1] > 1:
                # Multi-class: argmax
                mask: np.ndarray = logits[0].argmax(axis=0).astype(np.uint8)
                if return_probs:
                    if 0 <= prob_class < logits.shape[1]:
                        probs = logits[0, prob_class]
                    else:
                        probs = np.max(logits[0], axis=0)
                    if probs.max() > 1.0 or probs.min() < 0:
                        exp_vals = np.exp(logits[0] - np.max(logits[0], axis=0, keepdims=True))
                        softmax = exp_vals / (np.sum(exp_vals, axis=0, keepdims=True) + 1e-8)
                        prob_mask = (
                            softmax[prob_class].astype(np.float32)
                            if 0 <= prob_class < logits.shape[1]
                            else np.max(softmax, axis=0).astype(np.float32)
                        )
                    else:
                        prob_mask = probs.astype(np.float32)
            else:
                # Binary
                raw_mask = logits[0, 0] if logits.ndim == 4 else (logits[0] if logits.ndim == 3 else logits)
                if raw_mask.max() <= 1.0:
                    mask = (raw_mask > prob_threshold).astype(np.uint8) * 255
                else:
                    mask = raw_mask.astype(np.uint8)
                if return_probs:
                    if raw_mask.max() <= 1.0 and raw_mask.min() >= 0:
                        prob_mask = raw_mask.astype(np.float32)
                    else:
                        prob_mask = 1.0 / (1.0 + np.exp(-np.clip(raw_mask.astype(np.float32), -50, 50)))


            # ──────────────────────────────────────────────────────────────
            # 5. Ресайз масок к оригинальному размеру изображения
            # ──────────────────────────────────────────────────────────────
            if mask.shape != (orig_h, orig_w):
                sh, sw = orig_h / mask.shape[0], orig_w / mask.shape[1]
                mask = zoom(mask, (sh, sw), order=0).astype(np.uint8)  # nearest для маски

            if prob_mask is not None and prob_mask.shape != (orig_h, orig_w):
                sh, sw = orig_h / prob_mask.shape[0], orig_w / prob_mask.shape[1]
                prob_mask = zoom(prob_mask, (sh, sw), order=1).astype(np.float32)  # linear для вероятностей

            # ──────────────────────────────────────────────────────────────
            # 5. 🔧 СОЗДАНИЕ ОВЕРЛЕЯ (как в OpenCV/Sklearn/Torch сегментерах)
            # ──────────────────────────────────────────────────────────────
            # Конвертируем исходное изображение в RGB если нужно
            if image_np.ndim == 2:
                base_img = np.stack([image_np] * 3, axis=-1)
            else:
                base_img = image_np.copy()

            # Создаём оверлей: копию изображения для наложения маски
            overlay = base_img.copy()

            # 🔧 FIX: Красный цвет для маски [255, 0, 0] — только там, где mask > 127
            mask_bool = mask > 127
            overlay[mask_bool] = [255, 0, 0]  # RGB: красный

            # 🔧 FIX: Правильная формула alpha blending (как в OpenCVSegmenter):
            # result = overlay * alpha + base_img * (1 - alpha)
            # Это даёт: красная маска с прозрачностью, оригинал виден под ней
            result = cv2.addWeighted(
                overlay.astype(np.float32), alpha, base_img.astype(np.float32), 1.0 - alpha, 0
            ).astype(np.uint8)

            return result, mask.astype(np.uint8)

        except Exception as e:
            logger.error(f"TRT '{self.method}' segment_with_mask error: {e}", exc_info=True)
            # Возвращаем пустые маски при ошибке
            return np.zeros((orig_h, orig_w), dtype=np.uint8), None


# ──────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    # 🔹 Основные классы сегментеров
    "ONNXSegmenter",
    "TRTSegmenter",
    # 🔹 Типизация для аннотаций (публичные Type Aliases)
    "DeviceType",
    "NormalizationType",
    "InputShape",
    "RawOutput",
    "OnnxProvider",
    "ONNXSession",
    "TRTModel",
    "PreprocessedTensor",
    # 🔹 Импортируемые типы из базового модуля (для удобства)
    "ImageInput",
    "BinaryMask",
    "ProbabilityMask",
]
"""Публичный API модуля BackendSegmenters.

Экспортируемые символы:
- `ONNXSegmenter`: Сегментер на базе ONNX Runtime с поддержкой CUDA/CPU.
- `TRTSegmenter`: Сегментер на базе TensorRT для максимальной производительности на GPU.
- `DeviceType`: Literal["cuda", "cpu"] — тип устройства для инференса.
- `NormalizationType`: Literal["imagenet", "none", "custom"] — стратегия нормализации.
- `InputShape`: Tuple[int, int, int, int] — ожидаемая форма входа (B, C, H, W).
- `RawOutput`: npt.NDArray[Any] — тип сырого вывода модели: может быть любой размерности и dtype.
- `OnnxProvider`: Union[str, Tuple[str, Dict[str, Any]]] — тип провайдера ONNX Runtime: либо строка, либо (имя, опции).
- `ONNXSession`, `TRTModel`: Типы сессий/моделей для продвинутого использования.
- `PreprocessedTensor`: Тип предобработанного тензора для отладки/расширения.
- `ImageInput`, `BinaryMask`, `ProbabilityMask`: Базовые типы из BaseSegmenter.

Используется статическими анализаторами (mypy, pyright), linter'ами и IDE
для автодополнения и проверки типов при импорте:
    from segmenters.BackendSegmenters import ONNXSegmenter, DeviceType
"""
