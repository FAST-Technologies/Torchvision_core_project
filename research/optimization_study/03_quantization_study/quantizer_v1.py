# research/optimization_study/03_quantization_study/quantizer.py

# 🔬 Исследование 3: Квантование (Quantization)
# Цель: Ускорение через INT8 квантование

"""
Основной класс для квантования методов сегментации.
"""
import torch
from torch.ao.quantization import (
    QuantStub,
    DeQuantStub,
    prepare,
    convert,
    QConfig,
    default_qconfig,
)
from typing import Dict, List, Tuple, Any
import numpy as np
import time
from segmenters.TorchSegmenter import TorchSegmenter


class QuantizedSegmenter:
    """Квантованная версия TorchSegmenter"""

    def __init__(self, original_segmenter: TorchSegmenter):
        self.original = original_segmenter
        self.quantized_methods: Dict[str, Any] = {}

    def prepare_for_quantization(self, method_name: str):
        """Подготовка метода к квантованию"""

        # Обёртка метода с QuantStub/DeQuantStub
        original_func = self.original.methods[method_name]

        class QuantizableWrapper(torch.nn.Module):
            def __init__(self, func):
                super().__init__()
                self.quant = QuantStub()
                self.dequant = DeQuantStub()
                self.func = func

            def forward(self, x):
                x = self.quant(x)
                # Здесь вызов оригинальной логики
                # Внимание: не все операции поддерживают квантование!
                x = self.func(x)
                return self.dequant(x)

        wrapper = QuantizableWrapper(original_func)
        wrapper.qconfig = QConfig(
            activation=default_qconfig.activation,
            weight=default_qconfig.weight,
        )

        return prepare(wrapper, inplace=False)

    def calibrate(
        self,
        method_name: str,
        calibration_data: List[np.ndarray],
        num_calibration_steps: int = 100,
    ):
        """Калибровка квантованной модели"""

        prepared = self.prepare_for_quantization(method_name)
        prepared.eval()

        with torch.no_grad():
            for i, img in enumerate(calibration_data[:num_calibration_steps]):
                input_tensor = self.original.preprocess_image(img)
                _ = prepared(input_tensor)

        # Конвертация в квантованную версию
        quantized = convert(prepared, inplace=False)
        self.quantized_methods[method_name] = quantized
        return quantized

    def benchmark_quantized(
        self, method_name: str, image: np.ndarray, n_runs: int = 50
    ) -> Dict[str, float]:
        """Сравнение квантованной и оригинальной версий"""

        if method_name not in self.quantized_methods:
            raise ValueError(f"Method {method_name} not quantized")

        input_tensor = self.original.preprocess_image(image)

        # Оригинальная версия
        times_orig = []
        with torch.inference_mode():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = self.original.methods[method_name](input_tensor)
                torch.cuda.synchronize()
                times_orig.append((time.perf_counter() - start) * 1000)

        # Квантованная версия
        times_quant = []
        quant_func = self.quantized_methods[method_name]
        with torch.inference_mode():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = quant_func(input_tensor)
                torch.cuda.synchronize()
                times_quant.append((time.perf_counter() - start) * 1000)

        return {
            "original_mean": np.mean(times_orig),
            "quantized_mean": np.mean(times_quant),
            "speedup": np.mean(times_orig) / np.mean(times_quant),
            "accuracy_loss": self._compute_accuracy_loss(
                method_name, image, quant_func
            ),
        }

    def _compute_accuracy_loss(self, method_name, image, quant_func):
        """Оценка потери точности из-за квантования"""
        input_tensor = self.original.preprocess_image(image)

        orig_mask = self.original.methods[method_name](input_tensor)
        quant_mask = quant_func(input_tensor)

        # Приведение к одинаковому формату
        if isinstance(orig_mask, torch.Tensor):
            orig_mask = orig_mask.cpu().numpy()
        if isinstance(quant_mask, torch.Tensor):
            quant_mask = quant_mask.cpu().numpy()

        return {
            "pixel_agreement": np.mean(orig_mask == quant_mask),
            "mse": np.mean((orig_mask - quant_mask) ** 2),
        }
