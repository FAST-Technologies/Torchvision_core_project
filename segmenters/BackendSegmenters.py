# segmenters/BackendSegmenters.py

"""
Сегментеры на базе ONNX Runtime и TensorRT.

Исправления:
  1. ONNXSegmenter.segment — добавлена обработка ошибок, не возвращает None
  2. ONNXSegmenter — поддержка input_shape параметра
  3. TRTSegmenter — загрузка через load_trt_model (поддержка обоих форматов)
  4. Оба класса — segment_with_mask реализован корректно
"""

from segmenters.BaseSegmenter import BaseSegmenter, ImagePath, NumpyImage, PILImage, TorchImage, ImageInput, Mask, BinaryMask, ProbabilityMask
import numpy as np
import onnxruntime as ort
import torch
import torch_tensorrt
from typing import Optional, Dict, Any, Union, Literal, Tuple
import logging
import cv2

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
            raise ImportError("onnxruntime-gpu не установлен: pip install onnxruntime-gpu")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        # self.session = ort.InferenceSession(onnx_path, providers=providers)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Проверяем реальный output shape из модели
        out_shape = self.session.get_outputs()[0].shape
        logger.info(f"ONNX '{method_name}': input={self.input_name}, "
                    f"output={self.output_name}, output_shape={out_shape}")
        
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # 🔥 Ресайз к фиксированному размеру
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        if image.ndim == 3 and image.shape[2] == 3:
            tensor = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
            return np.expand_dims(tensor, 0)
        return image.astype(np.float32)

    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        try:
            logger.info(f"ONNX '{self.method}': входной тензор формы {image.shape}")
            tensor = self._preprocess(image)  # Ресайзит к self.input_shape
            logger.info(f"ONNX '{self.method}': после preprocess форма {tensor.shape}")
            
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            
            if (not outputs or 
                outputs[0] is None or 
                (hasattr(outputs[0], 'size') and outputs[0].size == 0) or
                (hasattr(outputs[0], 'shape') and len(outputs[0].shape) < 2)):
                
                logger.error(f"ONNX '{self.method}': invalid/empty output")
                h, w = image.shape[:2]
                return np.zeros((h, w), dtype=np.uint8)
            
            mask = outputs[0]
            logger.info(f"ONNX '{self.method}': output shape={mask.shape}, dtype={mask.dtype}, size={mask.size}")
            
            # === НОРМАЛИЗАЦИЯ ФОРМЫ ВЫХОДА ===
            while mask.ndim > 2:
                if mask.shape[0] == 1:
                    mask = np.squeeze(mask, axis=0)
                elif mask.ndim == 4 and mask.shape[1] == 1:
                    mask = np.squeeze(mask, axis=1)
                elif mask.ndim == 3 and mask.shape[-1] in [1, 3]:
                    mask = mask[..., 0]
                else:
                    # Пытаемся взять первый канал или усреднить
                    mask = np.mean(mask, axis=-1) if mask.shape[-1] > 1 else np.squeeze(mask)

            # 🔥 Гарантируем 2D перед ресайзом
            if mask.ndim != 2:
                logger.warning(f"ONNX '{self.method}': unexpected shape {mask.shape}, forcing squeeze")
                mask = np.squeeze(mask)
                if mask.ndim != 2:
                    mask = mask.reshape(image.shape[:2])  # Last resort
            
            # 🔥 КЛЮЧЕВОЕ: Ресайз к оригинальному размеру изображения
            target_h, target_w = image.shape[:2]
            if mask.shape != (target_h, target_w):
                logger.warning(f"ONNX '{self.method}': resize {mask.shape} -> {(target_h, target_w)}")
                mask = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # Нормализация значений
            if mask.dtype in (np.float32, np.float64):
                if mask.max() <= 1.0 + 1e-6:
                    mask = (mask * 255).clip(0, 255).astype(np.uint8)
                else:
                    mask = mask.clip(0, 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            
            return mask

        except Exception as e:
            logger.error(f"ONNX '{self.method}' inference error: {e}", exc_info=True)
            h, w = image.shape[:2] if image.ndim >= 2 else self.input_shape[2:4]
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
        try:
            import numpy as np
            from PIL import Image as PILImageModule
            if not isinstance(image, np.ndarray):
                if isinstance(image, PILImageModule.Image):
                    image = np.array(image)
            binary_mask = self.segment(image, **kwargs)
            return binary_mask, None
        except Exception as e:
            logger.error(f"{self.__class__.__name__} segment_with_mask error: {e}", exc_info=True)
            h = image.shape[0] if isinstance(image, np.ndarray) and image.ndim >= 1 else self.input_shape[2]
            w = image.shape[1] if isinstance(image, np.ndarray) and image.ndim >= 2 else self.input_shape[3]
            return np.zeros((h, w), dtype=np.uint8), None


class TRTSegmenter(BaseSegmenter):
    """Сегментер на базе TensorRT (через torch_tensorrt)"""
    def __init__(
        self,
        method_name: str,
        trt_model_or_path,
        device: str = "cuda",
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        **kwargs,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.method = method_name
        self.params = kwargs
        self.device = device
        if isinstance(trt_model_or_path, str):
            from utils.backend_exporter import load_trt_model
            self.model = load_trt_model(trt_model_or_path)
            if self.model is None:
                raise RuntimeError(f"Не удалось загрузить TRT модель: {trt_model_or_path}")
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
            logger.info(f"TRT '{self.method}': входной тензор формы {image.shape}")
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            
            # 🔥 АВТО-РЕСАЙЗ к размеру, на котором скомпилирован engine
            target_h, target_w = self.input_shape[2], self.input_shape[3]
            if image.shape[:2] != (target_h, target_w):
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(self.device)
            )
            logger.info(f"TRT '{self.method}': после preprocess форма {tensor.shape}")

            with torch.no_grad():
                out = self.model(tensor)

            logger.info(f"TRT '{self.method}': output type={type(out)}, len={len(out) if out else None}")

            # Flatten до (H,W)
            out_dim = int(out.dim())  # или out.ndim, если доступно
            if out_dim >= 3:
                mask_tensor = out.view(-1, *out.shape[-2:])[-1]
            else:
                mask_tensor = out

            mask = (mask_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            return mask

        except Exception as e:
            logger.error(f"TRT '{self.method}' inference error: {e}")
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)


    # def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
    #     orig_h, orig_w = image.shape[:2]
    #     try:
    #         logger.info(f"TRT '{self.method}': входной тензор формы {image.shape}")
            
    #         if image.ndim == 2:
    #             image = np.stack([image] * 3, axis=-1)
            
    #         target_h, target_w = self.input_shape[2], self.input_shape[3]
    #         if image.shape[:2] != (target_h, target_w):
    #             image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    #         tensor = (
    #             torch.from_numpy(image)
    #             .permute(2, 0, 1)
    #             .float()
    #             .div(255.0)
    #             .unsqueeze(0)
    #             .to(self.device)
    #         )
    #         logger.info(f"TRT '{self.method}': после preprocess форма {tensor.shape}")

    #         with torch.no_grad():
    #             out = self.model(tensor)

    #         if out is None:
    #             logger.error(f"TRT '{self.method}' returned None output")
    #             return np.zeros(image.shape[:2], dtype=np.uint8)
            
    #         out_shape = tuple(out.shape) if hasattr(out, 'shape') else None
    #         logger.info(f"TRT '{self.method}': output type={type(out)}, shape={out_shape}")

    #         # === НОРМАЛИЗАЦИЯ ФОРМЫ ===
    #         if out.dim() >= 3:
    #             if out.shape[1] > 1:
    #                 mask_tensor = out[:, -1, :, :]
    #             else:
    #                 mask_tensor = out.squeeze(1)
    #             if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
    #                 mask_tensor = mask_tensor.squeeze(0)
    #         elif out.dim() == 2:
    #             mask_tensor = out
    #         else:
    #             logger.warning(f"TRT '{self.method}': unexpected output dim {out.dim()}")
    #             if out.numel() > 0:
    #                 mask_tensor = out.view(-1, *out.shape[-2:]) if len(out.shape) >= 2 else out.view(target_h, target_w)
    #             else:
    #                 return np.zeros((target_h, target_w), dtype=np.uint8)

    #         if mask_tensor.numel() == 0:
    #             logger.warning(f"TRT '{self.method}': empty mask tensor")
    #             return np.zeros((target_h, target_w), dtype=np.uint8)

    #         mask_np = mask_tensor.cpu().numpy()
            
    #         if mask_np.dtype in (np.float32, np.float64) and mask_np.max() <= 1.0 + 1e-6:
    #             mask = (mask_np * 255).clip(0, 255).astype(np.uint8)
    #         else:
    #             mask = mask_np.clip(0, 255).astype(np.uint8)
            
    #         # 🔥 Ресайз к оригинальному размеру
    #         if mask.shape != (orig_h, orig_w):
    #             logger.info(f"TRT '{self.method}': финальный ресайз {mask.shape} -> {(orig_h, orig_w)}")
    #             mask = cv2.resize(mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    #             if mask.dtype != np.uint8:
    #                 mask = mask.astype(np.uint8)
            
    #         return mask

    #     except Exception as e:
    #         logger.error(f"TRT '{self.method}' inference error: {e}", exc_info=True)
    #         h, w = image.shape[:2] if image.ndim >= 2 else (512, 512)
    #         return np.zeros((h, w), dtype=np.uint8)
        
    def segment_with_mask(
        self, image: ImageInput, **kwargs: Any
    ) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        try:
            if not isinstance(image, np.ndarray):
                from PIL import Image as PILImageModule
                if isinstance(image, PILImageModule.Image):
                    image = np.array(image)
            binary_mask = self.segment(image, **kwargs)
            return binary_mask, None
        except Exception as e:
            logger.error(f"TRT '{self.method}' segment_with_mask error: {e}", exc_info=True)
            h = image.shape[0] if isinstance(image, np.ndarray) and image.ndim >= 1 else self.input_shape[2]
            w = image.shape[1] if isinstance(image, np.ndarray) and image.ndim >= 2 else self.input_shape[3]
            return np.zeros((h, w), dtype=np.uint8), None