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
- `utils.backend_exporter_new`: Низкоуровневые функции экспорта в ONNX/TRT
- `segmenters.NewTorchSegmenter.TorchSegmenter2`: Исходный класс сегментеров
- `segmenters.BackendSegmenters`: Классы-обёртки для загрузки ONNX/TRT моделей

Author: Vladimir Yamshchikov
Version: 1.0.0
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций

from typing import List, Dict, Any, Optional, Tuple, Literal, TypeAlias
import torch
import os
import time
from segmenters.NewTorchSegmenter import TorchSegmenter2

import logging

from utils.backend_exporter_new import (
    export_method_to_onnx_safe,
    export_onnx_to_trt_via_api,
    export_onnx_to_trt_via_trtexec,
    export_onnx_to_trt_via_onnx_tensorrt,
    _export_via_torch_tensorrt_jit,
    OnnxTrtFallbackSegmenter,
    TRT_PRESETS,
    _build_trt_provider_options,
    TRT_PRESET_PRODUCTION,
)

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
TRTStrategyType: TypeAlias = Literal["api", "trtexec", "onnx_tensorrt", "jit", "auto"]
"""Стратегия экспорта для классического метода сегментации в формат TRT, dtype=Literal["api", "trtexec", "onnx_tensorrt", "jit", "auto"]."""

PrecisionType: TypeAlias = Literal["fp32", "fp16", "bf16"]
"""Возможные точности, dtype=Literal["fp32", "fp16", "bf16"]."""

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
"""Список пороговых методов сегментации, dtype=List[str]."""

# ──────────────────────────────────────────────────────────────────────────────
EDGE_METHODS: List[str] = [
    # "sobel_edge",
    # "canny_edge",
    # "prewitt_edge",
    # "scharr_edge",
    # "laplacian_edge",
    # "roberts_cross_edge",
    # "log_edge",
    # "dog_edge",
    # "marr_hildreth_edge",
    # "gradient_magnitude_direction",
    # "phase_congruency_edge",
]
"""Список граничных методов сегментации, dtype=List[str]."""


# ──────────────────────────────────────────────────────────────────────────────
def export_all_classical_methods(
    output_base_dir: str = "./exported_models",
    precisions: Optional[List[PrecisionType]] = None,
    methods: Optional[List[str]] = None,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    force_reexport: bool = False,
    export_onnx: bool = True,
    export_trt: bool = True,
    trt_strategy: TRTStrategyType = "auto",
    enable_trt_ep_fallback: bool = True,  # Гарантированный fallback через TRT EP
    trt_ep_preset: str = "production",  # Пресет конфигурации: "production" | "debug" | "int8" | "dynamic"
    trt_ep_custom_options: Optional[Dict[str, Any]] = None,  # Пользовательские опции (перезаписывают пресет)
    trt_ep_cache_path: Optional[str] = None,  # Путь для кэша TRT EP
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
        trt_strategy: Стратегия экспорта в TRT:
            - "auto": пробует все стратегии по порядку (рекомендуется)
            - "api": tensorrt Python API (наиболее надёжный)
            - "trtexec": subprocess вызов trtexec
            - "onnx_tensorrt": через onnx-tensorrt parser
            - "jit": legacy torch_tensorrt JIT
        enable_trt_ep_fallback: Если True, создаёт ONNXSegmenter с TensorRT EP
                               как гарантированный рабочий вариант, даже если
                               нативный TRT экспорт не удался.
        trt_ep_preset: Название пресета конфигурации для TensorRT EP:
                      - "production": баланс скорость/точность (по умолчанию)
                      - "debug": FP32 + детальное логирование
                      - "int8": квантование (требует calibration.table)
                      - "dynamic": для моделей с dynamic_axes
        trt_ep_custom_options: Словарь опций, которые перезаписывают значения пресета.
        trt_ep_cache_path: Путь для кэширования TRT engines (по умолчанию './trt_engines_onnxrt')

    Returns:
        Dict с результатами: {method_name: {backend: status}}
        Добавляются ключи:
        - `onnxrt_trt_{precision}`: статус создания fallback с TRT EP
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

    for method_name in methods:
        print(f"\n🔄 Экспорт метода: {method_name}")
        results[method_name] = {}

        for precision in precisions:
            print(f"  ├─ Точность: {precision}")

            if precision == "bf16":
                cap = torch.cuda.get_device_capability(0)
                if cap[0] < 8:  # Ampere+
                    print(f"    ⚠️  bf16 требует GPU Ampere+, пропускаем для {method_name}")
                    results[method_name][f"trt_{precision}"] = "⚠️ BF16 unsupported"
                    continue

            # Инициализация сегментера с параметрами для экспорта
            segmenter: TorchSegmenter2 = TorchSegmenter2(
                method=method_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision=precision,
                use_compile=False,  # Важно: отключаем torch.compile для экспорта
                debug_mode=False,
            )

            # ───────── ONNX экспорт ─────────
            if export_onnx:
                onnx_dir: str = os.path.join(output_base_dir, "onnx", precision)
                os.makedirs(onnx_dir, exist_ok=True)
                onnx_path: str = os.path.join(onnx_dir, f"{method_name}.onnx")

                if force_reexport and os.path.exists(onnx_path):
                    os.remove(onnx_path)

                if not os.path.exists(onnx_path):
                    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                    try:
                        result: bool = export_method_to_onnx_safe(
                            segmenter=segmenter,
                            method_name=method_name,
                            output_path=onnx_path,
                            opset_version=17,
                            precision=precision,
                            input_shape=input_shape,
                        )
                        if result is True and os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0:
                            results[method_name][f"onnx_{precision}"] = "✅ OK"
                            print(f"  │  └─ ONNX: {onnx_path}")
                        else:
                            msg = "❌ Export failed" if result is False else f"❌ File not created (result={result})"
                            results[method_name][f"onnx_{precision}"] = msg
                            print(f"  │  └─ ONNX: {msg}")
                            print("\n⏳ Пауза 15 секунд перед запуском бенчмарка...")
                            print("   (нажмите Ctrl+C для отмены, если нужно)")
                            try:
                                time.sleep(15)  # 🔥 Задержка 15 секунд
                            except KeyboardInterrupt:
                                print("\n⚠️  Бенчмарк пропущен по запросу пользователя")
                    except Exception as e:
                        import traceback

                        results[method_name][f"onnx_{precision}"] = f"❌ {type(e).__name__}: {e}"
                        print(f"  │  └─ ONNX error: {e}")
                        print(f"  │  └─ Traceback:\n{traceback.format_exc()}")
                        print("\n⏳ Пауза 15 секунд перед запуском бенчмарка...")
                        print("   (нажмите Ctrl+C для отмены, если нужно)")
                        try:
                            time.sleep(15)  # 🔥 Задержка 15 секунд
                        except KeyboardInterrupt:
                            print("\n⚠️  Бенчмарк пропущен по запросу пользователя")
                else:
                    results[method_name][f"onnx_{precision}"] = "⏭️ Exists"

            if not os.path.exists(onnx_path):
                results[method_name][f"trt_{precision}"] = "⚠️ ONNX missing"
                print(f"  │  └─ TRT skipped: ONNX not found")
                print("\n⏳ Пауза 15 секунд перед запуском бенчмарка...")
                print("   (нажмите Ctrl+C для отмены, если нужно)")
                try:
                    time.sleep(15)  # 🔥 Задержка 15 секунд
                except KeyboardInterrupt:
                    print("\n⚠️  Бенчмарк пропущен по запросу пользователя")
                continue

            # ───────── TensorRT экспорт ─────────
            if export_trt and torch.cuda.is_available():
                trt_dir: str = os.path.join(output_base_dir, "tensorrt", precision)
                os.makedirs(trt_dir, exist_ok=True)
                trt_path: str = os.path.join(trt_dir, f"{method_name}.trt")

                if force_reexport and os.path.exists(trt_path):
                    os.remove(trt_path)

                if not os.path.exists(trt_path):
                    try:
                        success: bool = _export_trt_with_strategy(
                            method_name=method_name,
                            onnx_path=onnx_path,
                            trt_path=trt_path,
                            precision=precision,
                            input_shape=input_shape,
                            strategy=trt_strategy,
                        )
                        if success:
                            results[method_name][f"trt_{precision}"] = "✅ OK"
                            print(f"  │  └─ TRT: {trt_path}")
                        else:
                            results[method_name][f"trt_{precision}"] = "❌ Failed"
                            print("\n⏳ Пауза 15 секунд перед запуском бенчмарка...")
                            print("   (нажмите Ctrl+C для отмены, если нужно)")
                            try:
                                time.sleep(15)  # 🔥 Задержка 15 секунд
                            except KeyboardInterrupt:
                                print("\n⚠️  Бенчмарк пропущен по запросу пользователя")
                    except Exception as e:
                        results[method_name][f"trt_{precision}"] = f"❌ {e}"
                        print(f"  │  └─ TRT error: {e}")
                else:
                    results[method_name][f"trt_{precision}"] = "⏭️ Exists"

            print("\n⏳ Пауза 15 секунд перед запуском бенчмарка...")
            print("   (нажмите Ctrl+C для отмены, если нужно)")
            try:
                time.sleep(15)  # Задержка 15 секунд
            except KeyboardInterrupt:
                print("\n⚠️  Бенчмарк пропущен по запросу пользователя")

            if enable_trt_ep_fallback and torch.cuda.is_available():
                # Этот блок НЕ создаёт новый файл, а гарантирует, что метод
                # можно будет запустить через ONNX Runtime с TRT EP
                trt_ep_status_key = f"onnxrt_trt_{precision}"

                try:
                    # Проверяем, что onnxruntime с поддержкой TRT EP доступен
                    import onnxruntime as ort

                    available_providers = ort.get_available_providers()

                    if "TensorrtExecutionProvider" in available_providers:
                        # Проверяем, что tensorrt установлен (нужен для инициализации опций)
                        import tensorrt  # noqa: F401

                        # Собираем опции: пресет + кастомные переопределения
                        base_opts = TRT_PRESETS.get(trt_ep_preset, TRT_PRESET_PRODUCTION).copy()
                        if trt_ep_custom_options:
                            base_opts.update(trt_ep_custom_options)

                        cache_path = trt_ep_cache_path or f"./trt_engines_onnxrt/{precision}"

                        # Тестовая инициализация сессии для валидации конфигурации
                        test_opts = _build_trt_provider_options(
                            device_id=0, trt_options=base_opts, cache_path=cache_path
                        )

                        # Создаём тестовую сессию (не сохраняем, только проверяем)
                        sess_opts = ort.SessionOptions()
                        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        sess_opts.log_severity_level = 3  # ERROR

                        providers = [
                            ("TensorrtExecutionProvider", test_opts),
                            ("CUDAExecutionProvider", {"device_id": 0}),
                        ]

                        # Пробуем создать сессию — это триггерит сборку TRT engine в кэш
                        test_session = ort.InferenceSession(
                            onnx_path,
                            sess_options=sess_opts,
                            providers=providers,
                        )

                        # Проверяем, что TRT EP действительно активен
                        active = test_session.get_providers()
                        if "TensorrtExecutionProvider" in active:
                            results[method_name][trt_ep_status_key] = "✅ TRT EP Ready"
                            print(f"  │  └─ ONNX+TRT EP: готов (кэш: {cache_path})")
                        else:
                            results[method_name][trt_ep_status_key] = "⚠️ TRT EP not active"
                            print(f"  │  └─ ONNX+TRT EP: ⚠️ fallback to CUDA EP")

                        # Закрываем тестовую сессию
                        del test_session

                    else:
                        results[method_name][trt_ep_status_key] = "⚠️ TRT EP not available"
                        print(f"  │  └─ ONNX+TRT EP: ⚠️ провайдер не найден в ONNX Runtime")

                except ImportError as e:
                    results[method_name][trt_ep_status_key] = f"⚠️ Import error: {e}"
                    print(f"  │  └─ ONNX+TRT EP: ⚠️ {e}")
                except Exception as e:
                    results[method_name][trt_ep_status_key] = f"⚠️ Init error: {e}"
                    print(f"  │  └─ ONNX+TRT EP: ⚠️ {e}")

    # Сводный отчёт
    _print_export_summary(results)
    return results


# ──────────────────────────────────────────────────────────────────────────────
def _export_trt_with_strategy(
    method_name: str,
    onnx_path: str,
    trt_path: str,
    precision: PrecisionType,
    input_shape: Tuple[int, int, int, int],
    strategy: TRTStrategyType = "auto",
) -> bool:
    """Вспомогательная функция для экспорта ONNX → TRT с выбором стратегии."""
    strategies_order: List[str]
    if strategy == "auto":
        strategies_order = ["api", "trtexec", "onnx_tensorrt", "jit"]
    else:
        strategies_order = [strategy]

    for strat in strategies_order:
        print(f"    🔹 Пробуем стратегию: {strat}")
        try:
            if strat == "api":
                from utils.backend_exporter_new import export_onnx_to_trt_via_api

                return export_onnx_to_trt_via_api(
                    onnx_path=onnx_path,
                    trt_path=trt_path,
                    precision=precision,
                    input_shape=input_shape,
                )
            elif strat == "trtexec":
                from utils.backend_exporter_new import export_onnx_to_trt_via_trtexec

                return export_onnx_to_trt_via_trtexec(
                    onnx_path=onnx_path,
                    trt_path=trt_path,
                    precision=precision,
                    input_shape=input_shape,
                )
            elif strat == "onnx_tensorrt":
                from utils.backend_exporter_new import export_onnx_to_trt_via_onnx_tensorrt

                return export_onnx_to_trt_via_onnx_tensorrt(
                    onnx_path=onnx_path,
                    trt_path=trt_path,
                    precision=precision,
                    input_shape=input_shape,
                )
            elif strat == "jit":
                from utils.backend_exporter_new import _export_via_torch_tensorrt_jit

                # Для JIT нужна сама модель, а не ONNX — пропускаем для классических методов
                print(f"    ⚠️  JIT стратегия требует исходную модель, пропускаем")
                continue
        except Exception as e:
            print(f"    ❌ Все {len(strategies_order)} стратегии не удались для {method_name}/{precision}")
            print("    💡 Рассмотрите: 1) упростить метод, 2) использовать ONNX Runtime fallback")
            continue

    print(f"    ⚠️  Все стратегии не удались для {method_name}")
    return False


# ──────────────────────────────────────────────────────────────────────────────
def _print_export_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Печатает сводку по экспорту."""
    print("\n" + "=" * 70)
    print("📊 СВОДКА ПО ЭКСПОРТУ")
    print("=" * 70)

    for method, backends in results.items():
        ok_count = sum(1 for v in backends.values() if v == "✅ OK" or "Ready" in v)
        total = len(backends)
        status_icon = "✅" if ok_count == total else ("⚠️" if ok_count > 0 else "❌")

        print(f"\n{status_icon} {method:35s} [{ok_count}/{total}]")
        for backend, status in sorted(backends.items()):
            icon = "✅" if status == "✅ OK" or "Ready" in status else ("⚠️" if "⚠️" in status else "❌")
            print(f"     {icon} {backend:25s}: {status}")
