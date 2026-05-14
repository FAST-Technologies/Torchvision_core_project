"""
torch2trt конвертер (опционально).

⚠️ Требует сборки из исходников:
    pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt

Если модуль не доступен — класс не импортируется.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
import time
import warnings

# Проверка доступности при импорте
try:
    from torch2trt import torch2trt
    import tensorrt as trt

    TORCH2TRT_AVAILABLE = True
except ImportError:
    TORCH2TRT_AVAILABLE = False
    torch2trt = None
    trt = None


class Torch2TRTOptimizer:
    """
    Опциональная оптимизация через torch2trt.

    ⚠️ Этот модуль импортируется только если torch2trt доступен.
    Используйте BackendRegistry для безопасного доступа.
    """

    def __init__(
        self,
        segmenter: Any,
        image_shape: Tuple[int, int, int] = (3, 512, 512),
    ):
        if not TORCH2TRT_AVAILABLE:
            raise ImportError(
                "torch2trt not available. "
                "Install from source: "
                "pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("torch2trt requires CUDA")

        self.segmenter = segmenter
        self.image_shape = image_shape
        self.engines: Dict[str, Any] = {}

    def convert_method_to_trt(
        self,
        method_name: str,
        fp16_mode: bool = True,
        int8_mode: bool = False,
        max_workspace_size: int = 1 << 30,  # 1GB
        log_level: str = "WARNING",
    ) -> Any:
        """
        Конвертация метода через torch2trt.

        Args:
            method_name: Название метода
            fp16_mode: Использовать FP16 точность
            int8_mode: Использовать INT8 (требует калибровки)
            max_workspace_size: Макс. память для оптимизации
            log_level: Уровень логирования TensorRT

        Returns:
            Callable: Обёртка над TRT engine
        """
        if method_name not in self.segmenter.method_map:
            raise ValueError(
                f"Method '{method_name}' not found. " f"Available: {list(self.segmenter.method_map.keys())}"
            )

        original_func = self.segmenter.method_map[method_name]
        dummy_input = torch.randn(1, *self.image_shape).cuda()

        # Конвертация
        trt_model = torch2trt(
            original_func,
            [dummy_input],
            fp16_mode=fp16_mode,
            int8_mode=int8_mode,
            max_workspace_size=max_workspace_size,
            log_level=getattr(trt.Logger, log_level, trt.Logger.WARNING),
        )

        # Обёртка для совместимости
        def trt_wrapper(*args, **kwargs):
            # Конвертация входа
            if len(args) > 0:
                inp = args[0]
                if isinstance(inp, np.ndarray):
                    tensor = torch.from_numpy(inp).cuda().float()
                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(0).unsqueeze(0)
                    elif tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                elif isinstance(inp, torch.Tensor):
                    tensor = inp.cuda() if inp.device.type != "cuda" else inp
                else:
                    tensor = inp
            else:
                tensor = dummy_input

            # Инференс
            output = trt_model(tensor)
            return output

        self.engines[method_name] = trt_wrapper
        return trt_wrapper

    def benchmark_trt_vs_torch(self, method_name: str, n_runs: int = 100, **convert_kwargs) -> Dict[str, float]:
        """Бенчмарк torch2trt vs оригинал."""
        if method_name not in self.engines:
            trt_func = self.convert_method_to_trt(method_name, **convert_kwargs)
        else:
            trt_func = self.engines[method_name]

        original_func = self.segmenter.method_map[method_name]
        dummy_input = torch.randn(1, *self.image_shape).cuda()

        def measure(func, inp, n):
            times = []
            with torch.inference_mode():
                for _ in range(n):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = func(inp)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
            return np.mean(times), np.std(times)

        orig_mean, orig_std = measure(original_func, dummy_input, n_runs)
        trt_mean, trt_std = measure(trt_func, dummy_input, n_runs)

        speedup = orig_mean / trt_mean if trt_mean > 0 else float("inf")

        return {
            "original_ms": orig_mean,
            "original_std": orig_std,
            "trt_ms": trt_mean,
            "trt_std": trt_std,
            "speedup": speedup,
            "backend": "torch2trt",
        }
