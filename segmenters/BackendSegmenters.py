# segmenters/BackendSegmenters.py
from segmenters.BaseSegmenter import BaseSegmenter, ImagePath, NumpyImage, PILImage, TorchImage, ImageInput, Mask, BinaryMask, ProbabilityMask
import numpy as np
import onnxruntime as ort
import torch
import torch_tensorrt
from typing import Optional, Dict, Any, Union, Literal, Tuple


class ONNXSegmenter(BaseSegmenter):
    """Сегментер на базе ONNX Runtime"""
    def __init__(self, method_name: str, onnx_path: str, device: str = "cuda", **kwargs):
        super().__init__()
        self.method = method_name
        self.params = kwargs
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Ожидается формат (B, C, H, W), float32, нормализованный [0,1] или [0,255]
        # Приводим к формату ONNX-модели
        if image.ndim == 3:
            tensor = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
            tensor = np.expand_dims(tensor, 0)
        else:
            tensor = image.astype(np.float32)
        
        outputs = self.session.run(None, {self.input_name: tensor})
        # Предполагаем, что выход - (1, 1, H, W) или (H, W)
        mask = outputs[0].squeeze()
        return (mask * 255).astype(np.uint8)
    
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
        binary_mask = self.segment(image, **kwargs)
        return binary_mask, None


class TRTSegmenter(BaseSegmenter):
    """Сегментер на базе TensorRT (через torch_tensorrt)"""
    def __init__(self, method_name: str, trt_model: torch.nn.Module, device: str = "cuda", **kwargs):
        super().__init__()
        self.method = method_name
        self.params = kwargs
        self.device = device
        self.model = trt_model
        self.model.eval()

    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if image.ndim == 3:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0)
        else:
            tensor = torch.from_numpy(image).float()
            
        with torch.no_grad():
            out = self.model(tensor.to(self.device))
        return (out.squeeze().cpu().numpy() * 255).astype(np.uint8)