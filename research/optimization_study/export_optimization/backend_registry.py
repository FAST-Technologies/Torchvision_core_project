"""
Реестр доступных бэкендов экспорта с авто-детектом.
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import warnings

from .utils import (
    get_available_providers,
    get_available_tensorrt_backends,
)


@dataclass
class BackendInfo:
    """Информация о бэкенде экспорта."""

    name: str
    available: bool
    priority: int  # Чем меньше, тем выше приоритет
    description: str
    requirements: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        if not self.available and self.requirements:
            self.notes = f"❌ Требует: {', '.join(self.requirements)}"


class BackendRegistry:
    """
    Реестр бэкендов экспорта с авто-детектом доступности.

    Пример:
        registry = BackendRegistry()
        best = registry.get_best_backend()  # 'onnx' или 'torch_tensorrt'
        converter = registry.get_converter(best, segmenter)
    """

    def __init__(self):
        self._backends: Dict[str, BackendInfo] = {}
        self._detect_backends()

    def _detect_backends(self):
        """Авто-детект доступных бэкендов."""
        trt_info = get_available_tensorrt_backends()
        ort_providers = get_available_providers()

        # ONNX Runtime — всегда доступен
        self._backends["onnx"] = BackendInfo(
            name="onnx",
            available=True,
            priority=1,
            description="ONNX Runtime (CPU/CUDA/TensorRT EP)",
            requirements=[],
            notes=f"Доступные EP: {', '.join(ort_providers)}",
        )

        # torch-tensorrt (официальный NVIDIA)
        self._backends["torch_tensorrt"] = BackendInfo(
            name="torch_tensorrt",
            available=trt_info.get("torch_tensorrt", False),
            priority=2,
            description="torch-tensorrt (официальный NVIDIA бэкенд)",
            requirements=["tensorrt>=10.0", "torch-tensorrt>=2.0"],
            notes=(
                f"TensorRT version: {trt_info.get('tensorrt_version', 'N/A')}"
                if trt_info.get("torch_tensorrt")
                else ""
            ),
        )

        # torch2trt (опционально, из исходников)
        self._backends["torch2trt"] = BackendInfo(
            name="torch2trt",
            available=trt_info.get("torch2trt", False),
            priority=3,
            description="torch2trt (NVIDIA-AI-IOT, требует сборки)",
            requirements=["tensorrt", "build-essential", "python-dev"],
            notes=(
                "⚠️ Может требовать компиляции из исходников"
                if not trt_info.get("torch2trt")
                else ""
            ),
        )

    def list_backends(self, available_only: bool = False) -> List[str]:
        """Список бэкендов, отсортированный по приоритету."""
        backends = [
            b for b in self._backends.values() if not available_only or b.available
        ]
        return [b.name for b in sorted(backends, key=lambda x: x.priority)]

    def get_backend_info(self, name: str) -> Optional[BackendInfo]:
        """Получить информацию о бэкенде."""
        return self._backends.get(name)

    def get_best_backend(self) -> str:
        """Вернуть лучший доступный бэкенд."""
        available = [b for b in self._backends.values() if b.available]
        if not available:
            raise RuntimeError("No export backends available!")
        return min(available, key=lambda x: x.priority).name

    def get_converter(
        self, backend_name: str, segmenter, image_shape: tuple = (3, 512, 512)
    ):
        """
        Получить конвертер для указанного бэкенда.

        Raises:
            ImportError: Если бэкенд не доступен
            ValueError: Если бэкенд не известен
        """
        backend = self._backends.get(backend_name)
        if not backend:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Available: {list(self._backends.keys())}"
            )

        if not backend.available:
            raise ImportError(
                f"Backend '{backend_name}' not available. " f"{backend.notes}"
            )

        # Импорт конвертеров с защитой
        if backend_name == "onnx":
            from .onnx_converter import ONNXOptimizer

            return ONNXOptimizer(segmenter, image_shape)

        elif backend_name == "torch_tensorrt":
            from .torch_tensorrt_converter import TorchTRTOptimizer

            return TorchTRTOptimizer(segmenter, image_shape)

        elif backend_name == "torch2trt":
            from .torch2trt_converter import Torch2TRTOptimizer

            return Torch2TRTOptimizer(segmenter, image_shape)

        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    def print_status(self):
        """Вывести статус всех бэкендов в консоль."""
        print("\n🔧 Export Backends Status:")
        print("-" * 50)
        for name in self.list_backends():
            info = self._backends[name]
            status = "✅" if info.available else "❌"
            print(f"{status} {name:20s} — {info.description}")
            if info.notes:
                print(f"   └─ {info.notes}")
        print("-" * 50)
        best = self.get_best_backend()
        print(f"🎯 Recommended: {best}\n")


# Глобальный экземпляр
_registry = None


def get_registry() -> BackendRegistry:
    """Получить глобальный реестр бэкендов."""
    global _registry
    if _registry is None:
        _registry = BackendRegistry()
    return _registry
