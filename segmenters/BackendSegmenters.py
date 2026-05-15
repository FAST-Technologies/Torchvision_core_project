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
    Dict,
    Any,
    Union,
    Literal,
    Tuple,
    List,
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
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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


class ONNXSegmenter(BaseSegmenter):
    """Сегментер на базе ONNX Runtime."""

    def __init__(
        self,
        method_name: str,
        onnx_path: str,
        device: Literal["cuda", "cpu"] = "cuda",
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        **kwargs: Any,
    ) -> None:
        """Инициализация ONNX сегментера.

        Args:
            method_name: Имя метода для логирования.
            onnx_path: Путь к .onnx файлу модели.
            device: Устройство выполнения: "cuda" или "cpu".
            input_shape: Ожидаемая форма входа: (B, C, H, W).
            **kwargs: Дополнительные параметры.
        """
        super().__init__()
        self.method: str = method_name
        self.params: Dict[str, Any] = kwargs
        self.device_str: Literal["cuda", "cpu"] = device
        self.input_shape: Tuple[int, int, int, int] = input_shape
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime-gpu не установлен: pip install onnxruntime-gpu")
        providers: List[str] = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        )
        sess_options: ort.SessionOptions = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session: ort.InferenceSession = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name

        # Проверяем реальный output shape из модели
        out_shape: Tuple[Union[str, int], ...] = self.session.get_outputs()[0].shape
        logger.info(
            f"ONNX '{method_name}': input={self.input_name}, " f"output={self.output_name}, output_shape={out_shape}"
        )

    def _preprocess(self, image: npt.NDArray[np.uint8]) -> PreprocessedTensor:
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
        if isinstance(image, str):
            image_np = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = image
        try:
            tensor: PreprocessedTensor = self._preprocess(image_np)
            outputs: List[RawOutput] = self.session.run([self.output_name], {self.input_name: tensor})
            if not outputs or outputs[0] is None:
                logger.error(f"ONNX '{self.method}' returned None output")
                return np.zeros(image_np.shape[:2], dtype=np.uint8)

            mask: RawOutput = outputs[0]
            # Flatten до (H,W)
            while mask.ndim > 2:
                mask = mask.squeeze(0)

            # Нормализуем: если float [0,1] → uint8 [0,255]
            if mask.dtype in (np.float32, np.float64):
                if mask.max() <= 1.0 and mask.min() >= 0.0:
                    # Вероятности → бинаризация
                    mask = (mask > 0.5).astype(np.uint8) * 255
                else:
                    # Уже в диапазоне [0, 255]
                    mask = mask.astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

            return cast(BinaryMask, mask)

        except Exception as e:
            logger.error(f"ONNX '{self.method}' inference error: {e}")
            h = image_np.shape[0] if image_np.ndim >= 1 else self.input_shape[2]
            w = image_np.shape[1] if image_np.ndim >= 2 else self.input_shape[3]
            return np.zeros((h, w), dtype=np.uint8)

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
            image_np = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = image  # Уже np.ndarray

        # ✅ Передаём image_np, а не image
        binary_mask: BinaryMask = self.segment(image_np, **kwargs)
        return binary_mask, None


class TRTSegmenter(BaseSegmenter):
    """Сегментер на базе TensorRT (через torch_tensorrt)."""

    def __init__(
        self,
        method_name: str,
        trt_model_or_path: Union[str, TRTModel],
        device: Literal["cuda", "cpu"] = "cuda",
        **kwargs: Any,
    ) -> None:
        """Инициализация TensorRT сегментера.

        Args:
            method_name: Имя метода для логирования.
            trt_model_or_path: Путь к .trt файлу или уже загруженная TRT-модель.
            device: Устройство выполнения (только "cuda" поддерживается).
            **kwargs: Дополнительные параметры.
        """
        super().__init__()
        self.method: str = method_name
        self.params: Dict[str, Any] = kwargs
        self.device: torch.device = torch.device(device)
        if isinstance(trt_model_or_path, str):
            from utils.backend_exporter import load_trt_model

            loaded_model: Optional[TRTModel] = load_trt_model(trt_model_or_path)
            if loaded_model is None:
                raise RuntimeError(f"Не удалось загрузить TRT модель: {trt_model_or_path}")
            self.model: TRTModel = loaded_model
        else:
            self.model = trt_model_or_path
        self.model.eval()

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
        if isinstance(image, str):
            image_np = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if image_np.ndim == 3 and image_np.shape[0] in (1, 3):
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = image
        try:
            # Конвертация в 3 канала если нужно
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)

            # Нормализация и конвертация в torch
            tensor: torch.Tensor = (
                torch.from_numpy(image_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                out: torch.Tensor = self.model(tensor)

            # Flatten до (H,W)
            mask_tensor: torch.Tensor = out.squeeze()
            if mask_tensor.dim() > 2:
                mask_tensor = mask_tensor.squeeze(0)

            # Конвертация в uint8 [0, 255]
            mask: BinaryMask = (mask_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            return mask

        except Exception as e:
            logger.error(f"TRT '{self.method}' inference error: {e}")
            h, w = image_np.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

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
