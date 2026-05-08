# segmenters/BackendSegmenters.py

"""
Сегментеры на базе ONNX Runtime и TensorRT.

Исправления:
  1. ONNXSegmenter.segment — добавлена обработка ошибок, не возвращает None
  2. ONNXSegmenter — поддержка input_shape параметра
  3. TRTSegmenter — загрузка через load_trt_model (поддержка обоих форматов)
  4. Оба класса — segment_with_mask реализован корректно
"""

from segmenters.BaseSegmenter import (
    BaseSegmenter,
    ImagePath,
    NumpyImage,
    PILImage,
    TorchImage,
    ImageInput,
    Mask,
    BinaryMask,
    ProbabilityMask,
)
import numpy as np
import onnxruntime as ort
import torch
import torch_tensorrt
from typing import Optional, Dict, Any, Union, Literal, Tuple
import logging

logger = logging.getLogger(__name__)


class ONNXSegmenter(BaseSegmenter):
    """Сегментер на базе ONNX Runtime"""

    def __init__(
        self,
        method_name: str,
        onnx_path: str,
        device: str = "cuda",
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        **kwargs,
    ):
        super().__init__()
        self.method = method_name
        self.params = kwargs
        self.device_str = device
        self.input_shape = input_shape
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime-gpu не установлен: pip install onnxruntime-gpu"
            )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        # self.session = ort.InferenceSession(onnx_path, providers=providers)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Проверяем реальный output shape из модели
        out_shape = self.session.get_outputs()[0].shape
        logger.info(
            f"ONNX '{method_name}': input={self.input_name}, "
            f"output={self.output_name}, output_shape={out_shape}"
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Конвертирует изображение в (1,3,H,W) float32 [0,1]."""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim == 3 and image.shape[2] == 3:
            tensor = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
            return np.expand_dims(tensor, 0)  # (1,3,H,W)
        # Уже (B,C,H,W)
        return image.astype(np.float32)

    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Запускает ONNX инференс.

        Returns:
            np.ndarray: бинарная маска (H,W) uint8, значения {0,255}.
            При ошибке возвращает пустую маску того же размера.
        """
        try:
            tensor = self._preprocess(image)
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            if not outputs or outputs[0] is None:
                logger.error(f"ONNX '{self.method}' returned None output")
                return np.zeros(image.shape[:2], dtype=np.uint8)

            mask = outputs[0]
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

            return mask

        except Exception as e:
            logger.error(f"ONNX '{self.method}' inference error: {e}")
            h = image.shape[0] if image.ndim >= 1 else self.input_shape[2]
            w = image.shape[1] if image.ndim >= 2 else self.input_shape[3]
            return np.zeros((h, w), dtype=np.uint8)

    def segment_with_mask(
        self, image: ImageInput, **kwargs: Any
    ) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        """
        Сегментация с возвратом бинарной и вероятностной масок.

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
        from PIL import Image as PILImageModule

        if not isinstance(image, np.ndarray):
            if isinstance(image, PILImageModule.Image):
                image = np.array(image)
        binary_mask = self.segment(image, **kwargs)
        return binary_mask, None


class TRTSegmenter(BaseSegmenter):
    """Сегментер на базе TensorRT (через torch_tensorrt)"""

    def __init__(
        self,
        method_name: str,
        trt_model_or_path,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.method = method_name
        self.params = kwargs
        self.device = device
        if isinstance(trt_model_or_path, str):
            from utils.backend_exporter import load_trt_model

            self.model = load_trt_model(trt_model_or_path)
            if self.model is None:
                raise RuntimeError(
                    f"Не удалось загрузить TRT модель: {trt_model_or_path}"
                )
        else:
            self.model = trt_model_or_path
        self.model.eval()

    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Запускает TRT инференс.

        Returns:
            np.ndarray: бинарная маска (H,W) uint8, значения {0,255}.
        """
        try:
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)

            tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                out = self.model(tensor)

            # Flatten до (H,W)
            mask_tensor = out.squeeze()
            if mask_tensor.dim() > 2:
                mask_tensor = mask_tensor.squeeze(0)

            mask = (mask_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            return mask

        except Exception as e:
            logger.error(f"TRT '{self.method}' inference error: {e}")
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

    def segment_with_mask(
        self, image: ImageInput, **kwargs: Any
    ) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        if not isinstance(image, np.ndarray):
            from PIL import Image as PILImageModule

            if isinstance(image, PILImageModule.Image):
                image = np.array(image)
        binary_mask = self.segment(image, **kwargs)
        return binary_mask, None
