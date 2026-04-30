# scripts/check_backend_registry.py
"""
Скрипт для проверки статуса бэкендов оптимизации.

Импортирует registry из `research.optimization_study.export_optimization.backend_registry`
и выводит статус доступных бэкендов:
- ✅ onnx — ONNX Runtime (CPU/CUDA/TensorRT EP)
- ✅ torch_tensorrt — torch-tensorrt (официальный NVIDIA бэкенд)
- ❌ torch2trt — torch2trt (NVIDIA-AI-IOT, требует сборки)
- 🎯 Recommended: onnx

Пример использования:
    ```bash
    python scripts/check_backend_registry.py
    ```
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from typing import Any, Optional
from research.optimization_study.export_optimization.backend_registry import (
    get_registry,
)

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
RegistryLike = Any  # Тип зависит от реализации `get_registry()`


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Основной входной пункт скрипта.

    Получает registry через `get_registry()` и вызывает `print_status()`.
    """
    registry: Optional[RegistryLike] = get_registry()
    if registry is None:
        print("⚠️  Registry is None — backend not available")
        return
    registry.print_status()


if __name__ == "__main__":
    main()
