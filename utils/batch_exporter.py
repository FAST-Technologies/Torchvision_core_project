# utils/batch_exporter.py

"""Модуль массового экспорта классических методов сегментации.

Предоставляет инструменты для пакетного экспорта методов сегментации из библиотеки
PyTorch в оптимизированные форматы для продакшн-развёртывания: ONNX и TensorRT.

Основные возможности:
- 🔄 Массовый экспорт пороговых и граничных методов сегментации
- 🎯 Поддержка множественных точностей: fp32, fp16, bf16
- 📦 Экспорт в ONNX (кроссплатформенный формат)
- ⚡ Экспорт в TensorRT (оптимизация для NVIDIA GPU)
- 📊 Автоматическая генерация отчётов о статусе экспорта
- ♻️ Поддержка инкрементального экспорта (пропуск существующих файлов)

Поддерживаемые категории методов:
1. Пороговые методы (THRESHOLD_METHODS):
   - global_thresholding, otsu_thresholding, adaptive_thresholding
   - threshold_niblack, threshold_sauvola, threshold_bernsen
   - threshold_phansalkar, threshold_percentile, threshold_kittler_illingworth
   - threshold_entropy_kapur, threshold_triangle, threshold_multi_otsu
   - threshold_local_contrast

2. Граничные методы (EDGE_METHODS):
   - sobel_edge, canny_edge, prewitt_edge, scharr_edge, laplacian_edge
   - roberts_cross_edge, log_edge, dog_edge, marr_hildreth_edge
   - gradient_magnitude_direction, phase_congruency_edge

Архитектура экспорта:
```
┌─────────────────────────────────────┐
│  TorchSegmenter2 (исходный метод)   │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────┐      ┌─────────────┐
│  ONNX   │      │  TensorRT   │
│ (CPU/GPU)│      │ (GPU only) │
└─────────┘      └─────────────┘
```

Структура выходных файлов:
```
exported_models/
├── onnx/
│   ├── fp32/
│   │   ├── global_thresholding.onnx
│   │   ├── otsu_thresholding.onnx
│   │   └── ...
│   ├── fp16/
│   │   └── ...
│   └── bf16/
│       └── ...
└── tensorrt/
    ├── fp32/
    │   ├── global_thresholding.trt
    │   └── ...
    ├── fp16/
    │   └── ...
    └── bf16/
        └── ...
```

Примеры использования:

1. Базовый экспорт всех методов:
```python
from utils.batch_exporter import export_all_classical_methods

results = export_all_classical_methods(
    output_base_dir="./exported_models",
    export_onnx=True,
    export_trt=torch.cuda.is_available()
)
```

2. Экспорт с выбором точностей и методов:
```python
results = export_all_classical_methods(
    precisions=["fp32", "bf16"],
    methods=["otsu_thresholding", "canny_edge", "sobel_edge"],
    input_shape=(1, 3, 1024, 1024),  # Full HD
    force_reexport=True
)
```

3. Анализ результатов:
```python
for method, backends in results.items():
    for backend, status in backends.items():
        if status == "✅ OK":
            print(f"{method} → {backend}: успешно")
```

Требования:
- PyTorch >= 2.0 (для torch.export и Dynamo)
- onnx >= 1.14 (для экспорта в ONNX)
- tensorrt >= 8.6 (опционально, для экспорта в TensorRT)
- CUDA >= 11.8 (для GPU-экспорта и TensorRT)

Примечания:
- Для экспорта в TensorRT требуется установленный NVIDIA TensorRT и совместимая видеокарта.
- Методы с `torch.compile` не поддерживаются для экспорта — используйте `use_compile=False`.
- Входной shape должен соответствовать ожидаемому формату сегментера: (B, C, H, W).
- Экспорт может занимать значительное время при большом количестве методов и точностей.

See Also:
- `utils.backend_exporter`: Низкоуровневые функции экспорта в ONNX/TRT
- `segmenters.NewTorchSegmenter.TorchSegmenter2`: Исходный класс сегментеров
- `segmenters.BackendSegmenters`: Классы-обёртки для загрузки ONNX/TRT моделей

Author: Vladimir Yamshchikov
Version: 1.0.0
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

from typing import List, Dict, Any, Optional, Tuple
import torch
import os
import time
from segmenters.NewTorchSegmenter import TorchSegmenter2

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ──────────────────────────────────────────────────────────────────────────────
# Списки методов для экспорта
THRESHOLD_METHODS: List[str] = [
    "global_thresholding",
    "otsu_thresholding",
    "adaptive_thresholding",
    "threshold_niblack",
    "threshold_sauvola",
    "threshold_bernsen",
    "threshold_phansalkar",
    "threshold_percentile",
    "threshold_kittler_illingworth",
    "threshold_entropy_kapur",
    "threshold_triangle",
    "threshold_multi_otsu",
    "threshold_local_contrast",
]

# ──────────────────────────────────────────────────────────────────────────────
EDGE_METHODS: List[str] = [
    "sobel_edge",
    "canny_edge",
    "prewitt_edge",
    "scharr_edge",
    "laplacian_edge",
    "roberts_cross_edge",
    "log_edge",
    "dog_edge",
    "marr_hildreth_edge",
    "gradient_magnitude_direction",
    "phase_congruency_edge",
]


# ──────────────────────────────────────────────────────────────────────────────
def export_all_classical_methods(
    output_base_dir: str = "./exported_models",
    precisions: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    force_reexport: bool = False,
    export_onnx: bool = True,
    export_trt: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Массовый экспорт классических методов сегментации.

    Args:
        output_base_dir: Базовая директория для экспорта
        precisions: Список точностей ['fp32', 'fp16', 'bf16']
        methods: Список методов (по умолчанию: все threshold + edge)
        input_shape: Shape входного тензора (B, C, H, W)
        force_reexport: Пересоздавать ли существующие файлы
        export_onnx: Экспортировать ли в ONNX
        export_trt: Экспортировать ли в TensorRT (требует CUDA)

    Returns:
        Dict с результатами: {method_name: {backend: status}}
    """
    if precisions is None:
        precisions = ["fp32"]
        if torch.cuda.is_available():
            precisions.append("fp16")
            if torch.cuda.get_device_capability(0)[0] >= 8:
                precisions.append("bf16")

    if methods is None:
        methods = THRESHOLD_METHODS + EDGE_METHODS

    os.makedirs(output_base_dir, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    from utils.backend_exporter import (
        export_method_to_onnx_safe,
        export_method_to_trt_jit,
    )

    for method_name in methods:
        print(f"\n🔄 Экспорт метода: {method_name}")
        results[method_name] = {}

        for precision in precisions:
            print(f"  ├─ Точность: {precision}")

            # Инициализация сегментера с параметрами для экспорта
            segmenter = TorchSegmenter2(
                method=method_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision=precision,
                use_compile=False,  # 🔥 Важно: отключаем torch.compile для экспорта
                debug_mode=False,
            )

            # ───────── ONNX экспорт ─────────
            if export_onnx:
                onnx_dir = os.path.join(output_base_dir, "onnx", precision)
                os.makedirs(onnx_dir, exist_ok=True)
                onnx_path = os.path.join(onnx_dir, f"{method_name}.onnx")

                if force_reexport and os.path.exists(onnx_path):
                    os.remove(onnx_path)

                if not os.path.exists(onnx_path):
                    try:
                        export_method_to_onnx_safe(
                            segmenter=segmenter,
                            method_name=method_name,
                            output_path=onnx_path,
                            opset_version=17,
                            precision=precision,
                            input_shape=input_shape,
                        )
                        results[method_name][f"onnx_{precision}"] = "✅ OK"
                        print(f"  │  └─ ONNX: {onnx_path}")
                    except Exception as e:
                        results[method_name][f"onnx_{precision}"] = f"❌ {e}"
                        print(f"  │  └─ ONNX error: {e}")
                else:
                    results[method_name][f"onnx_{precision}"] = "⏭️ Exists"

            # ───────── TensorRT экспорт ─────────
            if export_trt and torch.cuda.is_available():
                trt_dir = os.path.join(output_base_dir, "tensorrt", precision)
                os.makedirs(trt_dir, exist_ok=True)
                trt_path = os.path.join(trt_dir, f"{method_name}.trt")

                if force_reexport and os.path.exists(trt_path):
                    os.remove(trt_path)

                if not os.path.exists(trt_path):
                    try:
                        export_method_to_trt_jit(
                            segmenter=segmenter,
                            method_name=method_name,
                            output_path=trt_path,
                            precision=precision,
                            input_shape=input_shape,
                            min_shape=(1, 3, 256, 256),
                            max_shape=(1, 3, 1024, 1024),
                        )
                        results[method_name][f"trt_{precision}"] = "✅ OK"
                        print(f"  │  └─ TRT: {trt_path}")
                    except Exception as e:
                        results[method_name][f"trt_{precision}"] = f"❌ {e}"
                        print(f"  │  └─ TRT error: {e}")
                else:
                    results[method_name][f"trt_{precision}"] = "⏭️ Exists"

            print("\n⏳ Пауза 15 секунд перед запуском бенчмарка...")
            print("   (нажмите Ctrl+C для отмены, если нужно)")
            try:
                time.sleep(15)  # 🔥 Задержка 15 секунд
            except KeyboardInterrupt:
                print("\n⚠️  Бенчмарк пропущен по запросу пользователя")

    # Сводный отчёт
    _print_export_summary(results)
    return results


# ──────────────────────────────────────────────────────────────────────────────
def _print_export_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Печатает сводку по экспорту."""
    print("\n" + "=" * 70)
    print("📊 СВОДКА ПО ЭКСПОРТУ")
    print("=" * 70)

    for method, backends in results.items():
        statuses = [v for v in backends.values()]
        ok_count = sum(1 for s in statuses if s == "✅ OK")
        print(f"{method:30s}: {ok_count}/{len(statuses)} успешных")
        for backend, status in backends.items():
            print(f"  ├─ {backend:15s}: {status}")
