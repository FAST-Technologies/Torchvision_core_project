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
- Функция `load_trt_model()` импортируется из `utils.backend_exporter` — убедитесь в её наличии.
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

# import torch_tensorrt
from typing import (
    Optional,
    Set,
    Dict,
    Any,
    Union,
    Literal,
    Tuple,
    List,
    Sequence,
    cast,
    TYPE_CHECKING,
)
import logging
from PIL import Image

if TYPE_CHECKING:
    from torch_tensorrt import Module as TRTModule

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler: logging.StreamHandler = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
ONNXSession = ort.InferenceSession
"""Тип сессии ONNX Runtime."""

TRTModel = Union[torch.nn.Module, "TRTModule"]
"""Тип модели TensorRT: PyTorch Module или скомпилированный TRT модуль."""

PreprocessedTensor = npt.NDArray[np.float32]
"""Тип предобработанного тензора: (1, 3, H, W), float32, [0, 1]."""

RawOutput = npt.NDArray[Any]
"""Тип сырого вывода модели: может быть любой размерности и dtype."""

OnnxProvider = Union[str, Tuple[str, Dict[str, Any]]]
"""Тип провайдера ONNX Runtime: либо строка, либо (имя, опции)."""


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
        device: Literal["cuda", "cpu"] = "cuda",
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        normalization: Literal["imagenet", "none", "custom"] = "imagenet",
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        is_neural: Optional[bool] = None,
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
            **kwargs: Дополнительные параметры.
        """
        super().__init__()
        self.method: str = model_key
        self.params: Dict[str, Any] = kwargs
        self.num_classes: int = num_classes
        self.device_str: Literal["cuda", "cpu"] = device
        self.input_shape: Tuple[int, int, int, int] = input_shape
        self.is_neural: bool = is_neural if is_neural is not None else _is_neural_by_key(model_key)
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime-gpu не установлен: pip install onnxruntime-gpu")
        providers: List[OnnxProvider] = (
            [
                (
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
                ),
                "CPUExecutionProvider",
            ]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess_options: ort.SessionOptions = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True

        self.session: ort.InferenceSession = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name

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
        from PIL import Image as PILImage

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
            logits: RawOutput = outputs[0]  # (1, C, H, W) или (1, H, W)
            if not outputs or outputs[0] is None:
                logger.error(f"ONNX '{self.method}' returned None output")
                return np.zeros((orig_h, orig_w), dtype=np.uint8)

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

    def segment_with_mask(self, image: ImageInput, **kwargs: Any) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        """Сегментация с возвратом бинарной и вероятностной масок.

        Для ONNX-модели возвращаем только бинарную маску,
        вероятностная маска не поддерживается.

        Args:
            image: Входное изображение.
            **kwargs: Дополнительные параметры.

        Returns:
            Tuple[BinaryMask, Optional[ProbabilityMask]]:
            - Бинарная маска: значения {0, 255}.
            - Вероятностная маска: None (не поддерживается).
        """
        import numpy as np

        # Конвертация входного изображения в numpy
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

        binary_mask: BinaryMask = self.segment(image_np, **kwargs)
        return binary_mask, None


class TRTSegmenter(BaseSegmenter):
    """Сегментер нейронной модели через TensorRT engine.

    Поддерживает все форматы TRT:
      - Serialized engine (от tensorrt API / trtexec) через TrtEngineWrapper
      - TorchScript (от torch_tensorrt JIT) через torch.jit
      - OnnxCudaFallback (ONNX Runtime + CUDA EP) как fallback

    Загрузка через load_trt_model() из backend_exporter автоматически
    определяет формат и возвращает подходящую обёртку.
    """

    def __init__(
        self,
        model_key: str,
        trt_model_or_path: Union[str, TRTModel],
        num_classes: int = 150,
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        device: Literal["cuda", "cpu"] = "cuda",
        normalization: Literal["imagenet", "none", "custom"] = "imagenet",
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
        self.input_shape: Tuple[int, int, int, int] = input_shape
        self.normalization_in_graph: bool = normalization_in_graph
        self.device: torch.device = torch.device(device)
        self.is_neural: bool = is_neural if is_neural is not None else _is_neural_by_key(model_key)
        if isinstance(trt_model_or_path, str):
            from utils.backend_exporter import load_trt_model

            loaded_model: Optional[TRTModel] = load_trt_model(trt_model_or_path)
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
        from PIL import Image as PILImage

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
                # Используем _preprocess_torch с ImageNet нормализацией
                tensor: torch.Tensor = self._preprocess_torch(image_np)
            else:
                # 2. Безопасная предобработка [0, 1]
                # Для классики это то, что нужно.
                # Для нейросетей: если модель обучалась на ImageNet,
                # лучше встроить нормализацию внутрь ONNX/TRT графа при экспорте.
                if image_np.ndim == 2:
                    image_np = np.stack([image_np] * 3, axis=-1)

                h_t, w_t = self.input_shape[2], self.input_shape[3]
                if image_np.shape[:2] != (h_t, w_t):
                    from PIL import Image as PILImage

                    pil: Image.Image = PILImage.fromarray(image_np.astype(np.uint8))
                    image_np = np.array(pil.resize((w_t, h_t), PILImage.Resampling.BILINEAR))

                # Нормализация
                if getattr(self, "normalization_in_graph", False):
                    tensor_np: npt.NDArray[np.float32] = image_np.astype(np.float32) / 255.0
                elif self.mean is not None and self.std is not None:
                    tensor_np = (image_np.astype(np.float32) / 255.0 - self.mean) / self.std
                else:
                    tensor_np = image_np.astype(np.float32) / 255.0

                tensor_np = np.transpose(tensor_np, (2, 0, 1))  # HWC → CHW
                tensor = torch.from_numpy(tensor_np).unsqueeze(0).to(self.device)

            # 3. Инференс
            # with torch.no_grad():
            #     out = self.model(tensor)
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

            out_np: np.ndarray = out.cpu().float().numpy()

            # 4. Интеллектуальная пост-обработка (аналогично ONNX)
            if out_np.ndim == 4 and out_np.shape[1] > 1:
                # Multi-class
                mask: np.ndarray = out_np[0].argmax(axis=0).astype(np.uint8)
            else:
                # Binary / Mask
                if out_np.ndim == 4:
                    mask = out_np[0, 0]
                elif out_np.ndim == 3:
                    mask = out_np[0]
                else:
                    mask = out_np

                if mask.max() <= 1.0:
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask = mask.astype(np.uint8)

            # 5. Ресайз
            if mask.shape != (orig_h, orig_w):
                from scipy.ndimage import zoom

                sh, sw = orig_h / mask.shape[0], orig_w / mask.shape[1]
                mask = zoom(mask, (sh, sw), order=0).astype(np.uint8)

            return mask.astype(np.uint8)

        except Exception as e:
            logger.error(f"TRT '{self.method}' inference error: {e}")
            return np.zeros((orig_h, orig_w), dtype=np.uint8)

    def segment_with_mask(self, image: ImageInput, **kwargs: Any) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        """Сегментация с возвратом бинарной и вероятностной масок.

        Для TRT-модели возвращаем только бинарную маску,
        вероятностная маска не поддерживается.

        Args:
            image: Входное изображение (путь, PIL, numpy или torch).
            **kwargs: Дополнительные параметры.

        Returns:
            Tuple[BinaryMask, Optional[ProbabilityMask]]:
            - Бинарная маска: значения {0, 255}, форма (H, W).
            - Вероятностная маска: None (не поддерживается для TRT).
        """
        import numpy as np
        from PIL import Image as PILImageModule

        # Конвертация входного изображения в numpy
        if not isinstance(image, np.ndarray):
            if isinstance(image, PILImageModule.Image):
                image_np: npt.NDArray[np.uint8] = np.array(image)
            elif isinstance(image, str):
                image_np = np.array(PILImageModule.open(image).convert("RGB"))
            elif isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
                if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
                    image_np = np.transpose(image_np, (1, 2, 0))
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
        else:
            image_np = image
        binary_mask: BinaryMask = self.segment(image, **kwargs)
        return binary_mask, None
