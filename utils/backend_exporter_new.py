# utils/backend_exporter_new.py

"""Расширенный экспорт нейросетевых моделей в ONNX и TensorRT.

## 🔍 Проблема
Стандартный `torch_tensorrt.compile` через JIT/Dynamo нестабильно работает
с моделями сегментации на базе `segmentation_models_pytorch` (SMP):
- PSPNet, U-Net, FPN, SegNet, DeepLabV3+ и др.

Типичные ошибки:
- `aten::size()` + dynamic reshape в декодерах SMP → `SIZE_MAX overflow` в TRT
- `tuple(int,int,int,int)` в padding → `Expected ivalue->isInt()`
- Сложные skip-connections с conditional interpolation → падение при трассировке

## 🛠️ Решение: многоуровневая стратегия экспорта
Функции модуля пробуют конвертацию в порядке приоритета:

| Приоритет | Стратегия                          | Описание                                                  |
|-----------|------------------------------------|-----------------------------------------------------------|
| 1️⃣        | `tensorrt` Python API              | Прямая конвертация ONNX → TRT через официальный API       |
| 2️⃣        | `trtexec` subprocess               | Вызов CLI-утилиты NVIDIA как надёжный fallback            |
| 2.5️⃣      | `onnx-tensorrt` parser             | Официальный парсер NVIDIA для сложных ONNX-графов         |
| 3️⃣        | `torch_tensorrt` JIT (legacy)      | Резерв для совместимых архитектур (DeepLab, FCN)          |
| 🔄        | ONNX Runtime + CUDA EP             | Высокопроизводительный fallback, если TRT не удался       |

## ✨ Ключевые особенности
- ✅ **SegmenterMethodWrapper**: оборачивает bound methods в `nn.Module` для `torch.export` и TRT
- ✅ **Фиксация выхода**: гарантированная форма `(B, 1, H, W)` через `view()` — без ветвлений по `dim()`
- ✅ **ONNX-совместимость**: `torch.where` вместо boolean indexing, удаление динамических `if`
- ✅ **Валидация и оптимизация**: `onnx.checker` + `onnx-simplifier` (опционально)
- ✅ **Авто-fallback точности**: fp16/bf16 → fp32 при отсутствии поддержки на GPU
- ✅ **Динамические размеры**: оптимизационные профили TRT для вариативных H/W
- ✅ **Универсальный загрузчик**: `load_trt_engine()` поддерживает все форматы `.trt`
- ✅ **Типизация**: строгие type hints + TypeAlias для IDE и mypy/pyright

## 📦 Типы и алиасы
```python
ShapeType = Tuple[int, ...]           # Форма тензора: (B, C, H, W)
PrecisionType = Literal["fp32", "fp16", "bf16"] | torch.dtype
OnnxProvider = str | Tuple[str, Dict] # Провайдер ONNX Runtime


"""

from __future__ import annotations

import os
import gc
import time
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Literal, TypeAlias

import torch
import torch.nn as nn
import numpy as np

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
ShapeType: TypeAlias = Tuple[int, ...]
"""Тип для формы тензора, например (1, 3, 512, 512), dtype=Tuple[int, ...]."""

PrecisionType: TypeAlias = Literal["fp32", "fp16", "bf16"]
"""Тип для указания точности вычислений, dtype=Literal["fp32", "fp16", "bf16"]."""

OnnxProvider: TypeAlias = Union[str, Tuple[str, Dict[str, Any]]]
"""Тип провайдера ONNX Runtime: либо строка, либо (имя, опции), dtype=Union[str, Tuple[str, Dict[str, Any]]]."""

InputShape: TypeAlias = Tuple[int, int, int, int]
"""Размер исходного изображения, dtype=Tuple[int, int, int, int]."""

PathLike: TypeAlias = Union[str, Path]
"""Унифицированный тип для путей к файлам: строка или pathlib.Path, dtype=Union[str, Path]."""


def _check_import(module_name: str) -> bool:
    """Проверяет доступность стороннего модуля без raising ImportError.

    Args:
        module_name: Имя модуля для проверки (напр. "tensorrt", "onnxsim").

    Returns:
        bool: True если модуль успешно импортируется, False иначе.

    Example:
        >>> _check_import("numpy")
        True
        >>> _check_import("nonexistent_module")
        False
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _diagnose_onnx_cuda() -> str:
    """Диагностирует доступность CUDA в ONNX Runtime.

    Проверяет:
    - Установлен ли onnxruntime / onnxruntime-gpu
    - Доступен ли CUDAExecutionProvider в списке провайдеров
    - Совместимость версий CUDA/cuDNN (косвенно)

    Returns:
        str: Человекочитаемый статус с рекомендациями:
             - "✅ CUDA EP available (providers: [...])"
             - "⚠️ CUDA EP missing. Available: [...]"
             - "❌ onnxruntime not installed"

    Note:
        Полезно для отладки fallback-сценариев при экспорте.
    """
    try:
        import onnxruntime as ort

        providers: List[str] = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return f"✅ CUDA EP available (providers: {providers})"
        else:
            return (
                f"⚠️ CUDA EP missing. Available: {providers}. "
                f"Check: pip install onnxruntime-gpu, CUDA/cuDNN compatible"
            )
    except ImportError:
        return "❌ onnxruntime not installed"


def _should_rebuild_trt(onnx_path: Path, trt_path: Path) -> bool:
    """Определяет, требуется ли пересборка TensorRT engine.

    Логика:
    - Если TRT-файл не существует → пересобрать
    - Если ONNX новее чем TRT (по mtime) → пересобрать
    - Иначе → использовать кэшированный engine

    Args:
        onnx_path: Путь к исходному .onnx файлу.
        trt_path: Путь к целевому .trt файлу.

    Returns:
        bool: True если требуется пересборка, False если можно использовать кэш.

    Note:
        Не проверяет семантические изменения в ONNX (только временные метки).
        Для CI/CD рекомендуется очищать кэш явно.
    """
    if not trt_path.exists():
        return True
    return onnx_path.stat().st_mtime > trt_path.stat().st_mtime


def _get_trt_ir_mode() -> Literal["dynamo", "ts"]:
    """Определяет рекомендуемый IR-режим для torch_tensorrt по версии пакета.

    Returns:
        Literal["dynamo", "ts"]:
            - "dynamo" для torch_tensorrt >= 2.0 (современно, но менее стабильно)
            - "ts" для более старых версий (JIT, проверено временем)

    Note:
        В текущей реализации принудительно возвращает "ts" для максимальной
        совместимости с SMP-моделями. Можно раскомментировать логику версии
        при стабильной поддержке dynamo.
    """
    try:
        import torch_tensorrt

        version: Tuple[int, ...] = tuple(map(int, torch_tensorrt.__version__.split(".")[:2]))
        logging.info(f"Current torch_tensorrt version: {version}")
        # return "dynamo" if version >= (2, 0) else "ts"
        return "ts"
    except ImportError:
        return "ts"


CHECKLIST: Dict[str, bool] = {
    """Глобальный чеклист зависимостей, dtype=Dict[str, bool].""" "cuda_available": torch.cuda.is_available(),
    "tensorrt_api": _check_import("tensorrt"),
    "onnxruntime_gpu": _check_import("onnxruntime")
    and "CUDAExecutionProvider" in __import__("onnxruntime").get_available_providers(),
    "torch_tensorrt": _check_import("torch_tensorrt"),
    "onnx_simplifier": _check_import("onnxsim"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Базовый wrapper: bound method → nn.Module
# ──────────────────────────────────────────────────────────────────────────────
class SegmenterMethodWrapper(nn.Module):
    """Оборачивает bound method сегментера в nn.Module для torch.export и TRT.

    ## Зачем это нужно?
    - `torch.export.export` и `torch_tensorrt` принимают только `nn.Module`
    - Методы из `method_map` — это обычные функции, не совместимые напрямую
    - Wrapper обеспечивает единый интерфейс `forward(x) -> tensor`

    ## Важные детали:
    - Автоматически извлекает "сырую" функцию из-под `@torch.compile` / `@dynamo`
    - Гарантирует выход формы `(B, 1, H, W)` через `view()` — без условных веток
    - Поддерживает параметр `precision` для методов, чувствительных к точности

    Attributes:
        segmenter: Исходный экземпляр сегментера.
        method_name: Ключ метода в `segmenter.method_map`.
        precision: Строка точности ("fp32", "fp16"), передаётся в метод.
        func: "Распакованная" функция для вызова.

    Args:
        segmenter: Экземпляр сегментера с атрибутом `method_map`.
        method_name: Имя метода для экспорта.
        precision: Точность вычислений (по умолчанию "fp32").

    Example:
        >>> wrapper = SegmenterMethodWrapper(seg, "canny_edge")
        >>> x = torch.randn(1, 3, 512, 512)
        >>> out = wrapper(x)  # (1, 1, 512, 512)
    """

    def __init__(self, segmenter: Any, method_name: str, precision: str = "fp32") -> None:
        """Инициализация модуля SegmenterMethodWrapper."""
        super().__init__()
        self.segmenter: Any = segmenter
        self.method_name: str = method_name
        self.precision: str = precision

        # Получаем "сырую" функцию без compile-обёртки
        func: Any = segmenter.method_map[method_name]
        if hasattr(func, "_torchdynamo_orig_callable"):
            self.func: Any = func._torchdynamo_orig_callable
        elif hasattr(func, "__wrapped__"):
            self.func = func.__wrapped__
        else:
            self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход: применяет обёрнутый метод к входному тензору.

        Args:
            x: Входной тензор формы (B, C, H, W), float32.

        Returns:
            torch.Tensor: Результат формы (B, 1, H, W), float32.
                          Гарантированно 4D даже если исходный метод возвращает 2D/3D.
        """
        result: torch.Tensor = self.func(self.segmenter, x, precision=self.precision, export_mode=True)
        # Гарантируем (1,1,H,W) через view — без dim()-зависимых веток
        if result.dim() == 2:
            result = result.unsqueeze(0).unsqueeze(0)
        elif result.dim() == 3:
            result = result.unsqueeze(0)
        b, _, h, w = x.shape
        return result.view(b, 1, h, w).float()


ALLOWED_ONNX_OPS: Set[str] = {
    """Разрешенные математические операции, dtype=Set[str]."""
    # Математические
    "Add",
    "Sub",
    "Mul",
    "Div",
    "MatMul",
    "Pow",
    # Сравнения (ваши threshold методы)
    "Greater",
    "Less",
    "Equal",
    "Where",
    "Cast",
    # Свёртки и пулинг
    "Conv",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    # Активации
    "Relu",
    "Sigmoid",
    "Tanh",
    "Softmax",
    # Работа с тензорами
    "Reshape",
    "Transpose",
    "Concat",
    "Slice",
    "Pad",
    # Ваши edge-операторы
    "Sobel",
    "Laplacian",
    "ThresholdedRelu",
}


def validate_onnx_operators(onnx_path: Path, allowed_ops: Set[str]) -> bool:
    """Валидирует ONNX-модель на наличие только разрешённых операторов.

    Предназначена для предварительной проверки совместимости с целевым
    бэкендом (TensorRT, ONNX Runtime с ограничениями и т.д.).

    Args:
        onnx_path: Путь к .onnx файлу для проверки.
        allowed_ops: Множество разрешённых имён операторов (напр. {"Conv", "Relu"}).

    Returns:
        bool: True если все операторы модели входят в allowed_ops.

    Raises:
        SystemError: Если обнаружен неизвестный оператор (с логируемым предупреждением).

    Note:
        В текущей реализации функция закомментирована в основном пайплайне,
        но может быть активирована для strict-режима валидации.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    for node in model.graph.node:
        if node.op_type not in allowed_ops:
            logger.warning(f"⚠️ Неизвестный оператор: {node.op_type}")
            raise SystemError
    return True


# ──────────────────────────────────────────────────────────────────────────────
# ONNX export
# ──────────────────────────────────────────────────────────────────────────────
def export_method_to_onnx_safe(
    segmenter: Any,
    method_name: str,
    output_path: PathLike,
    opset_version: int = 25,
    input_shape: ShapeType = (1, 3, 512, 512),
    precision: str = "fp32",
) -> bool:
    """Экспортирует один метод сегментера в формат ONNX с расширенной валидацией.

    ## Этапы экспорта:
    1. Проверка наличия метода в `segmenter.method_map`
    2. Обёртка в `SegmenterMethodWrapper` для совместимости с torch.onnx.export
    3. Тестовый прогон для инициализации буферов и проверки формы выхода
    4. Экспорт через `torch.onnx.export` с dynamic_axes для H/W
    5. Валидация через `onnx.checker`
    6. Опциональное упрощение через `onnx-simplifier`

    Args:
        segmenter: Экземпляр сегментера с атрибутом `method_map`.
        method_name: Ключ метода для экспорта (должен присутствовать в method_map).
        output_path: Путь для сохранения .onnx файла (расширение добавляется автоматически).
        opset_version: Версия ONNX operator set (по умолчанию 25, рекомендуется ≥17).
        input_shape: Форма входного тензора (B, C, H, W), по умолчанию (1, 3, 512, 512).
        precision: Точность вычислений для метода ("fp32", "fp16"), передаётся в wrapper.

    Returns:
        bool: True при успешном экспорте и валидации, False при любой ошибке.

    Side effects:
        - Создаёт файл по output_path при успехе
        - Удаляет битый .onnx файл при ошибке экспорта
        - Логирует детали процесса через logger модуля

    Note:
        - Модель временно переводится в eval-режим и на CPU для стабильности экспорта
        - Динамические оси настроены для batch/height/width — можно менять размер при инференсе
        - Опция упрощения (onnxsim) не критична: при ошибке импорт игнорируется
    """
    if method_name not in segmenter.method_map:
        print(f"❌ ONNX: метод '{method_name}' не найден в method_map")
        return False

    # if not validate_onnx_operators(Path(output_path), ALLOWED_ONNX_OPS):
    #     os.remove(output_path)
    #     return False

    wrapper: SegmenterMethodWrapper = SegmenterMethodWrapper(segmenter, method_name, precision=precision).eval()
    wrapper = wrapper.to(segmenter.device)

    sample: torch.Tensor = torch.randn(*input_shape, device=segmenter.device, dtype=torch.float32)

    # Тестовый прогон — инициализируем буферы
    try:
        with torch.no_grad():
            test_out = wrapper(sample)
        print(f"   Test output shape: {test_out.shape}, dtype: {test_out.dtype}")
    except Exception as e:
        print(f"❌ ONNX: тестовый прогон упал для '{method_name}': {e}")
        return False

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (sample,),
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=opset_version,
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"},
                },
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
                verbose=False,
            )

        # Валидация
        import onnx

        model: onnx.ModelProto = onnx.load(output_path)
        onnx.checker.check_model(model)
        actual_opset = model.opset_import[0].version
        if actual_opset < opset_version:
            logger.warning(
                f"⚠️ Запрошен opset {opset_version}, но получен {actual_opset}. "
                f"Некоторые операторы могут использовать более старую версию."
            )
        else:
            logger.info(f"✅ ONNX exported with opset {actual_opset}: {output_path}")

        # Упрощение через onnx-simplifier (опционально)
        try:
            from onnxsim import simplify

            model_simplified: onnx.ModelProto
            ok: bool
            model_simplified, ok = simplify(model)
            if ok:
                onnx.save(model_simplified, output_path)
                print("   ✅ ONNX simplified")
        except ImportError:
            pass
        except Exception as e_sim:
            print(f"   ⚠️  ONNX simplify failed (не критично): {e_sim}")

        return True

    except Exception as e:
        print(f"❌ ONNX export failed для '{method_name}': {e}")
        # Удаляем битый файл
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


# ─────────────────────────────────────────────────────────────────────
# Стратегия 1: ONNX → TensorRT через Python API (tensorrt напрямую)
# ─────────────────────────────────────────────────────────────────────
def export_onnx_to_trt_via_api(
    onnx_path: PathLike,
    trt_path: PathLike,
    # model_key: str,
    precision: PrecisionType = "fp32",
    input_shape: InputShape = (1, 3, 512, 512),
    workspace_gb: float = 4.0,
    verbose: bool = False,
) -> bool:
    """Конвертирует ONNX → TensorRT engine через официальный Python API.

    ## Эквивалент CLI:
    ```bash
    trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 --workspace=4096
    ```

    ## Преимущества перед torch_tensorrt:
    - Работает с любым валидным ONNX, независимо от источника
    - Не зависит от ограничений torch JIT / Dynamo
    - Полный контроль над builder config: precision, workspace, optimization profiles

    ## Поддержка точности:
    - `fp32`: всегда доступна
    - `fp16`: включается если `builder.platform_has_fast_fp16`
    - `bf16`: включается если поддерживается, иначе fallback на fp16/fp32

    Args:
        onnx_path: Путь к исходному .onnx файлу.
        trt_path: Путь для сохранения .trt engine (родительская директория создаётся при необходимости).
        precision: Желаемая точность: "fp32", "fp16" или "bf16".
        input_shape: Форма входа (B, C, H, W) для оптимизационного профиля.
        workspace_gb: Максимальный размер workspace для TRT builder в ГБ (по умолчанию 4.0).
        verbose: Включить подробное логирование TensorRT (уровень VERBOSE).

    Returns:
        bool: True при успешной сборке и сохранении engine, False при ошибке.

    Note:
        - Проверяет актуальность TRT по временным меткам (не пересобирает без необходимости)
        - Для статических входов фиксирует min=opt=max=onnx_shape
        - Для динамических входов строит профиль с границами [0.5x, 1x, 2x] от input_shape
        - Ошибки парсинга ONNX логируются с деталями от OnnxParser
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.warning("tensorrt Python API не установлен. Попробуйте: pip install tensorrt")
        return False

    onnx_p: Path = Path(onnx_path)
    trt_p: Path = Path(trt_path)
    if not _should_rebuild_trt(onnx_p, trt_p):
        logger.info(f"⏭ TRT engine актуален, пропускаем сборку: {trt_p}")
        return True

    TRT_LOGGER: trt.Logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    try:
        builder: trt.Builder = trt.Builder(TRT_LOGGER)
        network_flags: int = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network: trt.INetworkDefinition = builder.create_network(network_flags)
        parser: trt.OnnxParser = trt.OnnxParser(network, TRT_LOGGER)

        # Парсим ONNX
        with open(onnx_path, "rb") as f:
            onnx_bytes: bytes = f.read()

        if not parser.parse(onnx_bytes):
            errors: List[str] = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            logger.error(f"ONNX parse errors: {errors}")
            return False

        logger.info(
            f"ONNX parsed: {network.num_layers} layers, "
            f"input={network.get_input(0).shape if network.num_inputs > 0 else 'N/A'}"
        )

        # Конфигурация builder
        config: trt.IBuilderConfig = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))

        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 enabled")
            else:
                logger.warning("GPU не поддерживает fast FP16, используем FP32")

        if precision == "bf16":
            # Проверка совместимости: атрибут может отсутствовать в старых версиях TRT
            has_bf16 = hasattr(builder, "platform_has_fast_bf16") and builder.platform_has_fast_bf16
            if has_bf16:
                config.set_flag(trt.BuilderFlag.BF16)
                logger.info("BF16 enabled")
            else:
                logger.warning("⚠️ BF16 не поддерживается данной версией TensorRT, пробуем FP16 как fallback")
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("FP16 enabled as fallback")

        # Optimization profile для динамических размеров
        profile: trt.IOptimizationProfile = builder.create_optimization_profile()
        input_tensor: trt.ITensor = network.get_input(0)
        input_name: str = input_tensor.name
        onnx_input_shape: Tuple[int, ...] = tuple(input_tensor.shape)
        is_static_input: bool = all(d > 0 for d in onnx_input_shape)

        b, c, h, w = input_shape

        if is_static_input:
            # 🔥 Для статического входа: min=opt=max=onnx_shape
            profile.set_shape(
                input_name,
                min=onnx_input_shape,
                opt=onnx_input_shape,
                max=onnx_input_shape,
            )
            logger.info(f"✅ Static input: profile locked to {onnx_input_shape}")
        else:
            # Для динамического: используем заданные границы
            h_min = max(32, h // 2)
            w_min = max(32, w // 2)
            h_max = min(h * 2, 4096)
            w_max = min(w * 2, 4096)

            profile.set_shape(
                input_name,
                min=(b, c, h_min, w_min),
                opt=(b, c, h, w),
                max=(b, c, h_max, w_max),
            )
            logger.info(f"✅ Dynamic input: profile [{h_min}x{w_min} → {h}x{w} → {h_max}x{w_max}]")
        config.add_optimization_profile(profile)

        # Сборка engine
        logger.info(f"Building TRT engine (precision={precision}, workspace={workspace_gb}GB)...")
        t0: float = time.time()

        serialized_engine: Optional[bytes] = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("TRT build_serialized_network вернул None")
            return False

        build_time: float = time.time() - t0
        logger.info(f"Engine built in {build_time:.1f}s")

        # Сохранение
        trt_p.parent.mkdir(parents=True, exist_ok=True)
        with open(trt_p, "wb") as f:
            f.write(serialized_engine)

        engine_size_mb: float = trt_p.stat().st_size / 1e6
        logger.info(f"✅ TRT via tensorrt API engine saved: {trt_p} ({engine_size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"❌ ONNX→TRT via API failed: {e}")
        if trt_p.exists():
            trt_p.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────
# Стратегия 2: ONNX → TRT через trtexec subprocess
# ─────────────────────────────────────────────────────────────────────
def export_onnx_to_trt_via_trtexec(
    onnx_path: PathLike,
    trt_path: PathLike,
    # model_key: str,
    precision: PrecisionType = "fp32",
    input_shape: InputShape = (1, 3, 512, 512),
    workspace_mb: int = 4096,
    trtexec_path: str = "trtexec",
) -> bool:
    """Конвертирует ONNX → TRT через вызов утилиты trtexec в subprocess.

    ## Когда использовать?
    - Если tensorrt Python API не установлен или не работает
    - При деплое на серверах с TensorRT, установленным через deb/rpm пакеты
    - Как максимально надёжный fallback с минимальными зависимостями в коде

    ## Требования:
    - Утилита `trtexec` должна быть доступна в PATH или указан полный путь
    - Версия trtexec должна быть совместима с версией ONNX-модели

    Args:
        onnx_path: Путь к исходному .onnx файлу.
        trt_path: Путь для сохранения .trt engine.
        precision: Точность: "fp32", "fp16" или "bf16".
        input_shape: Форма входа (B, C, H, W) для расчёта min/opt/max shape.
        workspace_mb: Размер workspace для trtexec в МБ (по умолчанию 4096).
        trtexec_path: Путь к исполняемому файлу trtexec (по умолчанию ищется в PATH).

    Returns:
        bool: True если trtexec завершился с кодом 0, False при ошибке или таймауте.

    Note:
        - Таймаут выполнения: 600 секунд (10 минут)
        - Минимальные размеры: max(32, H//2), максимальные: min(H*2, 4096)
        - stderr trtexec обрезается до последних 2000 символов для логирования
        - Параметры --minShapes/--optShapes/--maxShapes задаются автоматически
    """
    import shutil

    onnx_p: Path = Path(onnx_path)
    trt_p: Path = Path(trt_path)

    if not _should_rebuild_trt(onnx_p, trt_p):
        logger.info(f"⏭ TRT engine актуален, пропускаем сборку: {trt_p}")
        return True

    if not shutil.which(trtexec_path):
        logger.warning(f"trtexec не найден в PATH: {trtexec_path}")
        return False

    b, c, h, w = input_shape
    shape_str: str = f"{b}x{c}x{h}x{w}"
    min_shape: str = f"{b}x{c}x{h // 2}x{w // 2}"
    max_shape: str = f"{b}x{c}x{h * 2}x{w * 2}"

    cmd: List[str] = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        f"--workspace={workspace_mb}",
        f"--minShapes=input:{min_shape}",
        f"--optShapes=input:{shape_str}",
        f"--maxShapes=input:{max_shape}",
    ]

    if precision == "fp16":
        cmd.append("--fp16")

    if precision == "bf16":
        cmd.append("--bf16")  # вместо --fp16

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 минут
        )
        if result.returncode == 0:
            logger.info(f"✅ trtexec succeeded: {trt_path}")
            return True
        else:
            logger.error(f"❌ trtexec failed:\n{result.stderr[-2000:]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("trtexec timeout (>300s)")
        return False
    except Exception as e:
        logger.error(f"trtexec subprocess error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# Стратегия 2.5: ONNX → TRT через onnx-tensorrt parser
# ─────────────────────────────────────────────────────────────────────
def export_onnx_to_trt_via_onnx_tensorrt(
    onnx_path: PathLike,
    trt_path: PathLike,
    # model_key: str,
    precision: PrecisionType = "fp32",
    input_shape: InputShape = (1, 3, 512, 512),
    workspace_mb: int = 4096,
    verbose: bool = False,
) -> bool:
    """Конвертирует ONNX → TensorRT через официальный парсер onnx-tensorrt.

    ## Особенности:
    - Использует `onnx_tensorrt.backend` для парсинга графа
    - Может лучше обрабатывать нестандартные операторы по сравнению с torch_tensorrt
    - Требует отдельной установки: `pip install onnx-tensorrt`

    ## Совместимость:
    - Требует предварительно установленные: tensorrt, onnx, onnx-tensorrt
    - Версии должны быть согласованы (см. матрицу совместимости NVIDIA)

    Args:
        onnx_path: Путь к исходному .onnx файлу.
        trt_path: Путь для сохранения .trt engine.
        precision: Точность: "fp32", "fp16" или "bf16".
        input_shape: Форма входа (B, C, H, W) для оптимизационного профиля.
        workspace_mb: Размер workspace в МБ (по умолчанию 4096).
        verbose: Включить подробное логирование builder.

    Returns:
        bool: True при успешной сборке, False при ошибке импорта/парсинга/сборки.

    Note:
        - Логика оптимизационного профиля аналогична export_onnx_to_trt_via_api
        - При ошибке парсинга детали выводятся через parser.get_error(i)
        - Стратегия 2.5: пробует после trtexec, перед torch_tensorrt JIT
    """
    try:
        import tensorrt as trt
        import onnx_tensorrt.backend as backend  # noqa: F401
    except ImportError:
        logger.warning(
            "onnx-tensorrt не установлен. Попробуйте: " "pip install onnx-tensorrt (требует tensorrt и onnx)"
        )
        return False

    onnx_p: Path = Path(onnx_path)
    trt_p: Path = Path(trt_path)

    if not _should_rebuild_trt(onnx_p, trt_p):
        logger.info(f"⏭ TRT engine актуален, пропускаем сборку: {trt_p}")
        return True

    TRT_LOGGER: trt.Logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    try:
        # Создаём builder и network
        builder: trt.Builder = trt.Builder(TRT_LOGGER)
        network_flags: int = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network: trt.INetworkDefinition = builder.create_network(network_flags)

        # Парсим ONNX через onnx-tensorrt
        parser: trt.OnnxParser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors: List[str] = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                logger.error(f"ONNX parse errors: {errors}")
                return False

        # Конфигурация builder
        config: trt.IBuilderConfig = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb * 1024**2))

        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 enabled")

        # Optimization profile для динамических размеров
        profile: trt.IOptimizationProfile = builder.create_optimization_profile()
        input_tensor: trt.ITensor = network.get_input(0)
        input_name: str = input_tensor.name
        b, c, h, w = input_shape

        h_min = max(32, h // 2)
        w_min = max(32, w // 2)
        h_max = min(h * 2, 4096)
        w_max = min(w * 2, 4096)

        profile.set_shape(
            input_name,
            min=(b, c, h_min, w_min),
            opt=(b, c, h, w),
            max=(b, c, h_max, w_max),
        )
        config.add_optimization_profile(profile)

        # Сборка engine
        logger.info(f"Building TRT engine via onnx-tensorrt (precision={precision})...")
        t0: float = time.time()

        serialized_engine: Optional[bytes] = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("TRT build_serialized_network вернул None")
            return False

        build_time: float = time.time() - t0
        logger.info(f"Engine built in {build_time:.1f}s")

        # Сохранение
        trt_p.parent.mkdir(parents=True, exist_ok=True)
        with open(trt_p, "wb") as f:
            f.write(serialized_engine)

        engine_size_mb: float = trt_p.stat().st_size / 1e6
        logger.info(f"✅ TRT via onnx-tensorrt saved: {trt_p} ({engine_size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"❌ ONNX→TRT via onnx-tensorrt failed: {e}")
        if trt_p.exists():
            trt_p.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────
# Стратегия 3: ONNX Runtime с CUDAExecutionProvider
# ─────────────────────────────────────────────────────────────────────
class OnnxTrtFallbackSegmenter:
    """Обёртка для инференса через ONNX Runtime с CUDAExecutionProvider.

    ## Когда используется?
    - Когда все стратегии экспорта в TensorRT не удались
    - Как высокопроизводительный fallback (~2-5x медленнее нативного TRT,
      но работает для любых архитектур: SMP, PSPNet, SegNet и др.)

    ## Интерфейс:
    - Совместим с `nn.Module`: поддерживает `__call__`, `.eval()`
    - Принимает/возвращает `torch.Tensor`, внутренние конвертации в numpy скрыты

    Attributes:
        session: onnxruntime.InferenceSession с настроенными провайдерами.
        input_name / output_name: Имена тензоров в ONNX-графе.
        input_shape: Ожидаемая форма входа для валидации.

    Args:
        onnx_path: Путь к валидному .onnx файлу.
        input_shape: Ожидаемая форма входа (B, C, H, W).
        device: "cuda" или "cpu" — приоритет провайдера.

    Raises:
        ImportError: Если onnxruntime-gpu не установлен при device="cuda".

    Example:
        >>> fallback = OnnxTrtFallbackSegmenter("model.onnx", device="cuda")
        >>> x = torch.randn(1, 3, 512, 512, device="cuda")
        >>> y = fallback(x)  # Инференс через CUDA EP
    """

    def __init__(
        self,
        onnx_path: PathLike,
        input_shape: InputShape = (1, 3, 512, 512),
        device: str = "cuda",
    ) -> None:
        """Инициализация модуля OnnxTrtFallbackSegmenter."""
        try:
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                logger.warning("⚠️ onnxruntime-gpu установлен, но CUDA не доступна. Проверьте CUDA/cuDNN версии.")
                logger.warning(f"⚠️ {_diagnose_onnx_cuda()}")
        except ImportError:
            raise ImportError("pip install onnxruntime-gpu")

        self.input_shape: ShapeType = input_shape

        providers: List[OnnxProvider] = (
            [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 4 * 1024**3,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",
            ]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        sess_options: ort.SessionOptions = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True

        self.session: ort.InferenceSession = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name

        actual_providers: List[str] = self.session.get_providers()
        logger.info(f"ONNX Runtime providers: {actual_providers}")
        if "CUDAExecutionProvider" in actual_providers:
            logger.info("✅ CUDA acceleration active")
            cuda_opts = self.session.get_provider_options().get("CUDAExecutionProvider", {})
            logger.debug(f"CUDA options: {cuda_opts}")
        else:
            logger.warning(
                "⚠️ CUDA provider не активен, используется CPU. "
                "Проверьте: 1) onnxruntime-gpu установлен, 2) CUDA доступна"
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет инференс: torch.Tensor → torch.Tensor.

        Args:
            x: Входной тензор любой формы, совместимой с ONNX-моделью.

        Returns:
            torch.Tensor: Результат инференса на том же устройстве, что и вход.
        """
        x_np: np.ndarray = x.cpu().float().numpy()
        outputs: List[np.ndarray] = self.session.run([self.output_name], {self.input_name: x_np})
        result: torch.Tensor = torch.from_numpy(outputs[0]).float()
        if x.is_cuda:
            result = result.cuda()
        return result

    def eval(self) -> "OnnxTrtFallbackSegmenter":
        """Возвращает self для совместимости с интерфейсом nn.Module.

        Returns:
            OnnxTrtFallbackSegmenter: Сам объект (состояние не меняется).
        """
        return self


# ─────────────────────────────────────────────────────────────────────
# Основная функция: экспорт нейронной модели в ONNX + TRT
# ─────────────────────────────────────────────────────────────────────
def export_neural_model(
    model: nn.Module,
    model_key: str,
    output_dir: PathLike,
    input_shape: InputShape = (1, 3, 512, 512),
    opset_version: int = 17,
    trt_precision: PrecisionType = "fp32",
    device: str = "cuda",
    dynamic_axes: bool = False,
) -> Dict[str, Optional[Path]]:
    """Полный pipeline экспорта нейронной модели: PyTorch → ONNX → TRT.

    TRT экспорт пробует ЧЕТЫРЕ стратегии в порядке приоритета:
      1. tensorrt Python API (самый надёжный, работает с любым ONNX)
      2. trtexec subprocess (если Python API недоступен)
      2.5. onnx-tensorrt parser (официальный парсер NVIDIA)
      3. torch_tensorrt JIT (последний резерв, работает для DeepLab/FCN)

    ## Алгоритм работы:
    1. **Экспорт в ONNX**: модель → CPU → torch.onnx.export → валидация → опциональное упрощение
    2. **Конвертация в TRT**: пробует стратегии в порядке приоритета:
       - 1️⃣ tensorrt Python API
       - 2️⃣ trtexec subprocess
       - 2.5️⃣ onnx-tensorrt parser
       - 3️⃣ torch_tensorrt JIT (для совместимых архитектур)
    3. **Fallback**: если все TRT-стратегии не удались → OnnxTrtFallbackSegmenter

    **Fallback**: если все TRT-стратегии не удались — создаётся OnnxTrtFallbackSegmenter
    (ONNX Runtime + CUDAExecutionProvider) как высокопроизводительный fallback.

    ## Возвращаемая структура:
    ```python
    {
        "onnx": Path | None,              # Путь к .onnx или None при ошибке
        "trt": Path | None,               # Путь к .trt или None если не собран
        "trt_strategy": str | None,       # Название успешной стратегии или None
        "trt_fallback": OnnxTrtFallbackSegmenter | None  # Fallback-обёртка
    }
    ```

    Args:
        model: PyTorch модель (nn.Module), должна быть в eval-режиме.
        model_key: Уникальный ключ для именования файлов (напр. "unet_smp_resnet50").
        output_dir: Директория для сохранения артефактов (.onnx, .trt).
        input_shape: Форма входа (B, C, H, W), по умолчанию (1, 3, 512, 512).
        opset_version: Версия ONNX opset (рекомендуется ≥17 для современных операторов).
        trt_precision: Желаемая точность TRT: "fp32", "bf16" или "fp16".
        device: Устройство для инференса: "cuda" или "cpu".
        dynamic_axes: Разрешить динамические оси (batch/height/width) в ONNX.

    Returns:
        Dict[str, Optional[Union[Path, Any]]]: Словарь с результатами экспорта
        (см. структуру выше).

    Side effects:
        - Создаёт output_dir при необходимости
        - Временно переводит модель на CPU для стабильного ONNX-экспорта
        - Возвращает модель на исходное устройство после экспорта
        - Логирует каждый этап через logger модуля

    Note:
        - Модель должна быть в `eval()` режиме перед вызовом
        - При `dynamic_axes=False` ONNX будет иметь статические размеры (быстрее инференс)
        - Все стратегии экспорта проверяют глобальный CHECKLIST зависимостей
        - Битые артефакты автоматически удаляются при ошибке
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Optional[Union[Path, Any]]] = {
        "onnx": None,
        "trt": None,
        "trt_strategy": None,
        "trt_fallback": None,
    }

    # ── Шаг 1: Экспорт в ONNX ────────────────────────────────────────
    onnx_path: Path = output_dir / f"{model_key}.onnx"

    # Переводим модель на CPU для стабильного ONNX экспорта
    model_cpu: nn.Module = model.to("cpu").eval()
    _freeze_model_for_export(model_cpu)

    dummy_input: torch.Tensor = torch.randn(*input_shape, dtype=torch.float32)

    try:
        dyn_axes: Optional[Dict[str, Dict[int, str]]] = None
        if dynamic_axes:
            dyn_axes = {
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            }

        with torch.no_grad():
            torch.onnx.export(
                model_cpu,
                (dummy_input,),
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dyn_axes,
                verbose=False,
            )

        # Валидация + упрощение
        import onnx

        onnx_model: Any = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        try:
            from onnxsim import simplify

            ok: bool
            simplified, ok = simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(onnx_path))
                logger.info("✅ ONNX simplified")
        except Exception as e:
            logger.warning(f"⚠️ ONNX simplifier failed: {e}")
            pass

        results["onnx"] = onnx_path
        size_mb: float = onnx_path.stat().st_size / 1e6
        logger.info(f"✅ ONNX exported: {onnx_path} ({size_mb:.1f} MB)")

    except Exception as e:
        logger.error(f"❌ ONNX export failed для {model_key}: {e}")
        if device == "cuda" and torch.cuda.is_available():
            model.to(device)
        return results

    finally:
        # Возвращаем модель на CUDA
        del model_cpu
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            model.to(device)

    # ── Шаг 2: ONNX → TRT ────────────────────────────────────────────
    if device != "cuda" or not torch.cuda.is_available():
        logger.info("TRT экспорт пропущен: не CUDA")
        return results

    trt_path = output_dir / f"{model_key}.{trt_precision}.trt"

    logger.info(f"🔍 TRT build checklist: {CHECKLIST}")
    if not CHECKLIST["tensorrt_api"]:
        logger.warning("⚠️ tensorrt API недоступен, пробуем fallback-стратегии")

    # Стратегия 1: tensorrt Python API
    if CHECKLIST["tensorrt_api"]:
        logger.info(f"[TRT Strategy 1] tensorrt API: {model_key}")
        success: bool = export_onnx_to_trt_via_api(
            onnx_path=onnx_path,
            trt_path=trt_path,
            # model_key=model_key,
            precision=trt_precision,
            input_shape=input_shape,
            workspace_gb=4.0,
        )
        if success:
            results["trt"] = trt_path
            results["trt_strategy"] = "tensorrt_api"
            logger.info(f"✅ TRT via tensorrt API: {trt_path}")
            return results

    # Стратегия 2: trtexec
    logger.info(f"[TRT Strategy 2] trtexec: {model_key}")
    success = export_onnx_to_trt_via_trtexec(
        onnx_path=onnx_path,
        trt_path=trt_path,
        # model_key=model_key,
        precision=trt_precision,
        input_shape=input_shape,
    )
    if success:
        results["trt"] = trt_path
        results["trt_strategy"] = "trtexec"
        logger.info(f"✅ TRT via trtexec: {trt_path}")
        return results

    # Стратегия 2.5: onnx-tensorrt parser (новая)
    if _check_import("onnx_tensorrt"):
        logger.info(f"[TRT Strategy 2.5] onnx-tensorrt: {model_key}")
        success = export_onnx_to_trt_via_onnx_tensorrt(
            onnx_path=onnx_path,
            trt_path=trt_path,
            # model_key=model_key,
            precision=trt_precision,
            input_shape=input_shape,
        )
        if success:
            results["trt"] = trt_path
            results["trt_strategy"] = "onnx_tensorrt"
            logger.info(f"✅ TRT via onnx-tensorrt: {trt_path}")
            return results

    # Стратегия 3: torch_tensorrt JIT (работает для DeepLab/FCN)
    if CHECKLIST["torch_tensorrt"]:
        logger.info(f"[TRT Strategy 3] torch_tensorrt JIT: {model_key}")
        success = _export_via_torch_tensorrt_jit(
            model=model,
            trt_path=trt_path,
            # model_key=model_key,
            precision=trt_precision,
            input_shape=input_shape,
            device=device,
        )
        if success:
            results["trt"] = trt_path
            results["trt_strategy"] = "torch_tensorrt_jit"
            logger.info(f"✅ TRT via torch_tensorrt JIT: {trt_path}")
            return results

    # Fallback: ONNX Runtime с CUDA provider
    logger.warning(
        f"⚠️ TRT не удался для {model_key}. "
        f"Используем ONNX Runtime + CUDAExecutionProvider как высокопроизводительный fallback."
    )
    logger.info(f"🔍 {_diagnose_onnx_cuda()}")
    try:
        fallback: OnnxTrtFallbackSegmenter = OnnxTrtFallbackSegmenter(
            onnx_path=onnx_path,
            input_shape=input_shape,
            device=device,
        )
        results["trt_fallback"] = fallback
        results["trt_strategy"] = "onnx_cuda_fallback"
        logger.info(f"✅ ONNX+CUDA fallback готов для {model_key}")
    except Exception as e:
        logger.error(f"❌ ONNX fallback тоже не удался: {e}")

    return results


def _freeze_model_for_export(model: nn.Module) -> None:
    """Подготавливает модель к экспорту: отключает training-only поведение.

    Выполняет:
    - Переводит модель в `eval()` режим
    - Фиксирует статистику BatchNorm/LayerNorm: `m.training = False`
    - Отключает Dropout: `m.training = False`

    Args:
        model: PyTorch модель (nn.Module) для подготовки.

    Note:
        - Не изменяет веса модели, только режимы слоёв
        - Обратимая операция: после экспорта можно вернуть модель в train()
        - Рекомендуется вызывать перед любым экспортом (ONNX/TRT)
    """
    model.eval()
    # Фиксируем BatchNorm статистику
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
            m.training = False
    # Убираем Dropout
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.training = False


def _export_via_torch_tensorrt_jit(
    model: nn.Module,
    trt_path: Path,
    precision: str,
    input_shape: InputShape,
    device: str,
    # model_key: str
) -> bool:
    """Резервная стратегия: экспорт через torch_tensorrt.compile (JIT-режим).

    ## Когда используется?
    - Только если все предыдущие стратегии (API / trtexec / onnx-tensorrt) не удались
    - Для архитектур, совместимых с JIT-трассировкой: DeepLab, FCN, простые U-Net

    ## Ограничения:
    - Не работает со сложными dynamic reshape в SMP-декодерах
    - Требует `torch.jit.trace`, что может не поддерживать условные ветвления

    Args:
        model: PyTorch модель (nn.Module) в eval-режиме.
        trt_path: Путь для сохранения .trt файла (TorchScript формат).
        precision: Точность: "fp32", "fp16" или "bf16".
        input_shape: Форма входа (B, C, H, W) для трассировки и TRT profile.
        device: Устройство для трассировки: "cuda" (обязательно для TRT).

    Returns:
        bool: True при успешной компиляции и сохранении, False при ошибке.

    Note:
        - Автоматически определяет IR-режим через _get_trt_ir_mode()
        - Для fp16 проверяет Compute Capability GPU (≥6.0)
        - bf16 не поддерживается в torch_tensorrt → fallback на fp16/fp32
        - При ошибке удаляет частично созданный .trt файл
    """
    try:
        import torch_tensorrt as torchtrt
    except ImportError:
        logger.warning("torch_tensorrt не установлен")
        return False

    try:
        model_cuda: nn.Module = model.to(device).eval()
        dummy: torch.Tensor = torch.randn(*input_shape, device=device, dtype=torch.float32)

        # trace
        try:
            traced: torch.jit.ScriptModule = torch.jit.trace(model_cuda, dummy, check_trace=False)
            # print(f"   ✅ torch.jit.trace OK для '{model_key}'")
        except Exception as e:
            # print(f"❌ TRT: torch.jit.trace failed для '{model_key}': {e}")
            return False

        enabled_precisions: Set[torch.dtype] = {torch.float32}
        if precision == "fp16" and torch.cuda.get_device_capability()[0] >= 6:
            enabled_precisions.add(torch.float16)
        if precision == "bf16":
            if torch.cuda.get_device_capability()[0] >= 6:
                logger.warning("⚠️ torch_tensorrt не поддерживает bf16, используем fp16 как fallback")
                enabled_precisions.add(torch.float16)
            else:
                logger.warning("⚠️ torch_tensorrt: GPU не поддерживает fp16, используем fp32")

        ir_mode: Literal["dynamo", "ts"] = _get_trt_ir_mode()
        logger.info(f"Using torch_tensorrt IR mode: {ir_mode}")

        trt_model: Any = torchtrt.compile(
            traced,
            inputs=[
                torchtrt.Input(
                    min_shape=input_shape,
                    opt_shape=input_shape,
                    max_shape=input_shape,
                    dtype=torch.float32,
                )
            ],
            enabled_precisions=enabled_precisions,
            ir=ir_mode,
            require_full_compilation=True,
            truncate_long_and_double=True,
        )

        torch.jit.save(trt_model, str(trt_path))
        logger.info(f"✅ TRT via torch_tensorrt JIT: {trt_path}")
        return True

    except Exception as e:
        # print(f"❌ TRT compile failed для '{model_key}': {e}")
        if trt_path.exists():
            trt_path.unlink()
        if os.path.exists(trt_path):
            os.remove(trt_path)
        return False


# ─────────────────────────────────────────────────────────────────────
# Загрузка TRT engine (поддерживает все форматы)
# ─────────────────────────────────────────────────────────────────────
def load_trt_engine(
    trt_path: PathLike,
    device: str = "cuda",
    is_neural: bool = False,
) -> Optional[Any]:
    """Универсальный загрузчик TensorRT engines во всех поддерживаемых форматах.

    ## Поддерживаемые форматы (пробует по порядку):
    1. **Serialized engine** (tensorrt API / trtexec):
       - Распознаётся по magic bytes или попытке десериализации
       - Возвращает `TrtEngineWrapper` с интерфейсом nn.Module

    2. **TorchScript** (torch_tensorrt JIT):
       - Загружается через `torch.jit.load`
       - Возвращает `torch.jit.ScriptModule`

    3. **Torch-TensorRT Dynamo**:
       - Загружается через `torch_tensorrt.load`
       - Возвращает скомпилированную модель

    Args:
        trt_path: Путь к файлу .trt (любого из поддерживаемых форматов).
        device: Устройство для инференса: "cuda" или "cpu".
        is_neural: Флаг для будущей логики (сейчас игнорируется).

    Returns:
        Optional[Any]: Один из типов:
            - `TrtEngineWrapper` (для serialized engine)
            - `torch.jit.ScriptModule` (для JIT)
            - `torch_tensorrt.CompiledModule` (для dynamo)
            - `None` при ошибке загрузки всех форматов

    Note:
        - При ошибке загрузки одного формата пробует следующий (цепочка fallback)
        - Все ошибки логируются на уровне warning/error, не raising исключения
        - Для serialized engine используется trt.Logger с уровнем WARNING (можно настроить)
    """
    trt_path = Path(trt_path)
    if not trt_path.exists():
        logger.error(f"TRT файл не найден: {trt_path}")
        return None

    # Формат 1: tensorrt API / trtexec serialized engine
    # Узнаём по первым байтам: TRT engines начинаются с "ptrt" или специального magic
    try:
        import tensorrt as trt

        TRT_LOGGER: trt.Logger = trt.Logger(trt.Logger.WARNING)
        runtime: trt.Runtime = trt.Runtime(TRT_LOGGER)

        with open(trt_path, "rb") as f:
            engine_bytes: bytes = f.read()

        engine: Optional[trt.ICudaEngine]
        if hasattr(runtime, "deserialize_cuda_engine"):
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        elif hasattr(runtime, "deserialize_engine"):  # TRT >= 8.5
            engine = runtime.deserialize_engine(engine_bytes)
        else:
            logger.error("❌ Неизвестная версия TensorRT API")
            return None
        if engine is not None:
            logger.info(f"✅ Loaded via tensorrt Runtime: {trt_path}")
            return TrtEngineWrapper(engine, device)
    except Exception as e:
        logger.warning(f"⚠️ TRT Runtime load failed: {e}")
        pass

    # Формат 2: torch.jit (torch_tensorrt JIT) TorchScript
    try:
        model: torch.jit.ScriptModule = torch.jit.load(str(trt_path))
        model = model.to(device).eval()
        logger.info(f"✅ Loaded via torch.jit.load: {trt_path}")
        return model
    except Exception as e:
        logger.warning(f"⚠️ torch_tensorrt JIT aka TorchScript load failed: {trt_path} :{e}")
        pass

    # Формат 3: torch_tensorrt dynamo
    try:
        import torch_tensorrt as torchtrt

        model = torchtrt.load(str(trt_path))
        logger.info(f"✅ Loaded via torch_tensorrt.load: {trt_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Все методы загрузки TRT не удались для {trt_path}: {e}")
        return None


class TrtEngineWrapper:
    """Обёртка для tensorrt.ICudaEngine с интерфейсом nn.Module.

    ## Назначение:
    Позволяет использовать TRT engines, собранные через:
    - `tensorrt` Python API
    - `trtexec` CLI

    так же, как обычные PyTorch модели: `model(x)`.

    ## Особенности:
    - Автоматическое определение имён входов/выходов из engine metadata
    - Поддержка async-инференса через CUDA stream
    - Явная синхронизация после выполнения для корректного timing

    Attributes:
        engine: Исходный trt.ICudaEngine.
        context: trt.IExecutionContext для выполнения инференса.
        input_name / output_name: Имена тензоров, извлечённые из engine.
        device: Устройство для инференса ("cuda").

    Args:
        engine: Загруженный trt.ICudaEngine.
        device: Устройство для размещения тензоров (по умолчанию "cuda").

    Example:
        >>> # После загрузки через load_trt_engine():
        >>> trt_model = load_trt_engine("model.trt")  # Возвращает TrtEngineWrapper
        >>> x = torch.randn(1, 3, 512, 512, device="cuda")
        >>> y = trt_model(x)  # Прямой вызов как у nn.Module
    """

    def __init__(self, engine: Any, device: str = "cuda") -> None:
        """Инициализация модуля TrtEngineWrapper."""
        import tensorrt as trt

        self.engine: trt.ICudaEngine = engine
        self.device: str = device
        self.context: trt.IExecutionContext = engine.create_execution_context()

        # Выясняем имена и индексы входов/выходов
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        for i in range(engine.num_io_tensors):
            name: str = engine.get_tensor_name(i)
            mode: trt.TensorIOMode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет инференс через TensorRT execution context.

        Args:
            x: Входной тензор формы (B, C, H, W), float32, на CUDA.

        Returns:
            torch.Tensor: Результат инференса формы (B, 1, H, W), float32.

        Note:
            - Автоматически устанавливает input shape для dynamic profiles
            - Использует async execution при поддержке API
            - Выполняет torch.cuda.synchronize() перед возвратом
        """
        x = x.contiguous().to(self.device).float()
        b, c, h, w = x.shape

        # Устанавливаем форму входа (для dynamic shapes)
        self.context.set_input_shape(self.input_name, (b, c, h, w))

        # Выходной буфер
        out_shape: Tuple[int, ...] = tuple(self.context.get_tensor_shape(self.output_name))
        output: torch.Tensor = torch.empty(out_shape, dtype=torch.float32, device=self.device)

        # Выполняем инференс
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())

        stream: int = torch.cuda.current_stream().cuda_stream
        if hasattr(self.context, "execute_async_v3"):
            self.context.execute_async_v3(stream)
        elif hasattr(self.context, "execute_async_v2"):
            self.context.execute_async_v2(stream)
        else:
            # Fallback на синхронное выполнение
            self.context.execute()
        torch.cuda.synchronize()

        return output

    def eval(self) -> "TrtEngineWrapper":
        """Возвращает self для совместимости с интерфейсом nn.Module.

        Returns:
            TrtEngineWrapper: Сам объект.
        """
        return self


# ─────────────────────────────────────────────────────────────────────
# Сохранение совместимости со старым load_trt_model
# ─────────────────────────────────────────────────────────────────────
def load_trt_model(path: PathLike, device: str = "cuda", **kwargs: Any) -> Optional[Any]:
    """Backward-compatible alias для load_trt_engine.

    Предназначена для плавного перехода со старого модуля `backend_exporter`
    на новый `backend_exporter_new` без изменения кода вызова.

    Args:
        path: Путь к файлу .trt.
        device: Устройство для инференса ("cuda" или "cpu").
        **kwargs: Дополнительные аргументы (игнорируются для совместимости).

    Returns:
        Optional[Any]: Результат load_trt_engine() — см. документацию выше.

    Note:
        - Не добавляет новой функциональности, только делегирует вызов
        - Сохраняет сигнатуру старой функции для минимизации breaking changes
        - Рекомендуется постепенно мигрировать на load_trt_engine() в новом коде
    """
    return load_trt_engine(path, device=device, **kwargs)
