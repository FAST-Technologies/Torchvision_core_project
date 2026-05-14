# utils/backend_exporter.py

"""Экспорт методов TorchSegmenter2 в ONNX и TensorRT.

Поддерживаемые форматы:
1. **ONNX**: кросс-платформенный формат для инференса через ONNX Runtime, TVM, OpenVINO.
2. **TensorRT (Dynamo)**: современный pipeline через torch.export + torch_tensorrt.dynamo.
3. **TensorRT (JIT)**: legacy pipeline через torch.jit.trace + torch_tensorrt.compile.

Ключевые особенности:
- ✅ SegmenterMethodWrapper: оборачивает bound methods в nn.Module для torch.export/TRT
- ✅ Фиксация формы выхода: гарантированный (1,1,H,W) через view() — без dim()-веток
- ✅ ONNX-совместимость: torch.where вместо boolean indexing, удаление динамических if
- ✅ Валидация и упрощение: onnx.checker + onnx-simplifier (опционально)
- ✅ Авто-fallback: fp16 → fp32 если не поддерживается, динамические размеры для TRT
- ✅ Обработка ошибок: очистка битых файлов, подробное логирование

Типичный workflow:
```python
from utils.backend_exporter import (
    export_method_to_onnx_safe,
    export_method_to_trt_dynamo,
    load_trt_model
)
from segmenters.NewTorchSegmenter import TorchSegmenter2
import torch

# 1. Инициализация сегментера
segmenter = TorchSegmenter2(method="canny_edge")

# 2. Экспорт в ONNX
export_method_to_onnx_safe(
    segmenter=segmenter,
    method_name="canny_edge",
    output_path="./exports/canny.onnx",
    opset_version=25
)

# 3. Экспорт в TensorRT (Dynamo, fp16)
export_method_to_trt_dynamo(
    segmenter=segmenter,
    method_name="canny_edge",
    output_path="./exports/canny.trt",
    precision="fp16"
)

# 4. Загрузка и инференс TRT-модели
trt_model = load_trt_model("./exports/canny.trt")
with torch.no_grad():
    input_tensor = torch.randn(1, 3, 512, 512, device="cuda")
    output = trt_model(input_tensor)  # (1, 1, 512, 512)
    mask = (output > 0.5).byte() * 255
```

Note:
- Для `torch.export` и `torch_tensorrt.dynamo` требуется PyTorch ≥ 2.0.
- TensorRT требует установленный NVIDIA TensorRT и CUDA-capable GPU.
- ONNX opset ≥17 рекомендуется для поддержки современных операторов.
- При экспорте в fp16 автоматически проверяется Compute Capability GPU; при несоответствии — fallback на fp32.
- Все функции возвращают `bool` для удобства пакетной обработки; детали ошибок логируются.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Sequence,
    Mapping,
    Collection,
    Literal,
    TYPE_CHECKING,
)

import torch
import torch.nn as nn

if TYPE_CHECKING:
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

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
ShapeType = Tuple[int, ...]
"""Тип для формы тензора, например (1, 3, 512, 512)."""

PrecisionType = Union[Literal["fp32", "fp16", "bf16"], torch.dtype]
"""Тип для указания точности вычислений."""


# ──────────────────────────────────────────────────────────────────────────────
# Базовый wrapper: bound method → nn.Module
# ──────────────────────────────────────────────────────────────────────────────
class SegmenterMethodWrapper(nn.Module):
    """Оборачивает bound method сегментера в nn.Module для torch.export и TRT.

    ВАЖНО: func должна быть оригинальной (не скомпилированной) функцией.
    torch.export.export не принимает torch.compile-обёрнутые функции.
    """

    def __init__(
        self, segmenter: Any, method_name: str, precision: str = "fp32"
    ) -> None:
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
        """Прямой ход для SegmenterMethodWrapper."""
        result: torch.Tensor = self.func(
            self.segmenter, x, precision=self.precision, export_mode=True
        )
        # Гарантируем (1,1,H,W) через view — без dim()-зависимых веток
        if result.dim() == 2:
            result = result.unsqueeze(0).unsqueeze(0)
        elif result.dim() == 3:
            result = result.unsqueeze(0)
        b, _, h, w = x.shape
        return result.view(b, 1, h, w).float()


# ──────────────────────────────────────────────────────────────────────────────
# ONNX export
# ──────────────────────────────────────────────────────────────────────────────
def export_method_to_onnx_safe(
    segmenter,
    method_name: str,
    output_path: Union[str, Path],
    opset_version: int = 25,
    input_shape: ShapeType = (1, 3, 512, 512),
    precision: str = "fp32",
) -> bool:
    """Экспортирует один метод сегментера в ONNX.

    Args:
        segmenter: Экземпляр сегментера с attribute `method_map`.
        method_name: Ключ метода в `method_map`.
        output_path: Путь для сохранения .onnx файла.
        opset_version: Версия ONNX opset (по умолчанию 17).
        input_shape: Форма входного тензора (B, C, H, W).
        precision: Точность вычислений для метода.

    Returns:
        bool: True при успешном экспорте.
    """
    if method_name not in segmenter.method_map:
        print(f"❌ ONNX: метод '{method_name}' не найден в method_map")
        return False

    wrapper: SegmenterMethodWrapper = SegmenterMethodWrapper(
        segmenter, method_name, precision=precision
    ).eval()
    wrapper = wrapper.to(segmenter.device)

    sample: torch.Tensor = torch.randn(
        *input_shape, device=segmenter.device, dtype=torch.float32
    )

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
                print(f"   ✅ ONNX simplified")
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


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT export через torch.export + torch_tensorrt.dynamo
# ──────────────────────────────────────────────────────────────────────────────
def export_method_to_trt_dynamo(
    segmenter: Any,
    method_name: str,
    output_path: Union[str, Path],
    precision: str = "fp32",
    input_shape: ShapeType = (1, 3, 512, 512),
) -> bool:
    """Экспорт метода в TensorRT через torch.export + dynamo.

    Args:
        segmenter: Экземпляр сегментера.
        method_name: Ключ метода в method_map.
        output_path: Путь для сохранения .trt файла.
        precision: Точность вычислений ("fp32" или "fp16").
        input_shape: Форма входного тензора.

    Returns:
        bool: True при успешном экспорте.
    """
    try:
        import torch_tensorrt as torchtrt
    except ImportError:
        print("❌ TRT: torch_tensorrt не установлен: pip install torch-tensorrt")
        return False

    device_obj: torch.device = (
        torch.device(segmenter.device)
        if isinstance(segmenter.device, str)
        else segmenter.device
    )

    if method_name not in segmenter.method_map:
        print(f"❌ TRT: метод '{method_name}' не найден")
        return False

    # FIX 1: оборачиваем в nn.Module — torch.export требует Module или callable
    wrapper: SegmenterMethodWrapper = SegmenterMethodWrapper(
        segmenter, method_name, precision=precision
    ).eval()
    wrapper = wrapper.to(segmenter.device)

    # Определяем dtype для TRT
    target_precision: str = precision
    if precision == "fp16" and torch.cuda.is_available():
        cap: Tuple[int, int] = torch.cuda.get_device_capability()
        if cap[0] < 6:
            print(
                f"⚠️  fp16 не поддерживается на compute capability {cap[0]}.{cap[1]}, переключаемся на fp32"
            )
            target_precision = "fp32"

    target_dtype: torch.dtype = (
        torch.float16 if target_precision == "fp16" else torch.float32
    )
    sample: torch.Tensor = torch.randn(
        *input_shape, device=segmenter.device, dtype=torch.float32
    )

    # Тестовый прогон
    try:
        with torch.no_grad():
            out: torch.Tensor = wrapper(sample)
        print(
            f"Wrapper output: shape={out.shape}, min={out.min()}, max={out.max()}, unique={out.unique()}"
        )
    except Exception as e:
        print(f"❌ TRT: тестовый прогон упал для '{method_name}': {e}")
        return False

    # Шаг 1: torch.export
    try:
        exported: torch.export.ExportedProgram = torch.export.export(
            wrapper,
            (sample,),
            strict=False,  # Разрешить динамические операции
        )
        print(f"   ✅ torch.export OK для '{method_name}'")
    except Exception as e:
        print(f"❌ TRT: torch.export failed для '{method_name}': {e}")
        return False

    # Шаг 2: TensorRT dynamo compile
    try:
        print(f"   ✅ trying tensorRT dynamo  for '{method_name}'")
        # Динамические размеры для реальных изображений
        h, w = input_shape[2], input_shape[3]
        trt_input: torchtrt.Input = torchtrt.Input(
            min_shape=(1, 3, h // 2, w // 2),
            opt_shape=input_shape,
            max_shape=(1, 3, h * 2, w * 2),
            dtype=torch.float32,  # Всегда fp32 на входе, TRT конвертирует внутри
        )

        enabled_precisions: Set[torch.dtype] = {torch.float32}
        if target_dtype == torch.float16:
            enabled_precisions.add(torch.float16)

        trt_model: Any = torchtrt.dynamo.compile(
            exported,
            inputs=[trt_input],
            device=device_obj,
            enabled_precisions=enabled_precisions,
            min_block_size=1,
            truncate_long_and_double=True,
            debug=False,
            use_python_runtime=True,
        )

        # FIX 2: torch_tensorrt.save для dynamo-compiled моделей
        # torch.jit.save не работает с ExportedProgram-based TRT моделями
        torchtrt.save(trt_model, output_path, inputs=[sample])
        print(f"✅ TensorRT engine сохранён: {output_path}")
        return True

    except Exception as e:
        print(f"❌ TRT compile failed для '{method_name}': {e}")
        if os.path.exists(output_path):
            os.remove(output_path)

        # Fallback: пробуем fp32 если была fp16
        if precision == "fp16":
            print(f"💡 Повтор с fp32...")
            return export_method_to_trt_dynamo(
                segmenter,
                method_name,
                output_path,
                precision="fp32",
                input_shape=input_shape,
            )
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка TRT модели
# ──────────────────────────────────────────────────────────────────────────────
def load_trt_model(
    path: Union[str, Path],
    sample_input: Optional[torch.Tensor] = None,
) -> Optional[Any]:
    """Загружает TRT модель. Поддерживает оба формата: torch.jit и torch_tensorrt.

    Args:
        path: Путь к файлу .trt.
        sample_input: Опциональный пример входа для валидации.

    Returns:
        Optional[Any]: Загруженная модель или None при ошибке.
    """
    try:
        import torch_tensorrt as torchtrt

        model: Any = torchtrt.load(path)
        print(f"✅ TRT loaded via torch_tensorrt.load: {path}")
        return model
    except Exception:
        pass

    try:
        model_jit: torch.jit.ScriptModule = torch.jit.load(path)
        print(f"✅ TRT loaded via torch.jit.load: {path}")
        return model_jit
    except Exception as e:
        print(f"❌ TRT load failed: {path}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
def export_method_to_trt_jit(
    segmenter: Any,
    method_name: str,
    output_path: Union[str, Path],
    precision: str = "fp32",
    input_shape: ShapeType = (1, 3, 512, 512),
    min_shape: Optional[ShapeType] = None,
    max_shape: Optional[ShapeType] = None,
) -> bool:
    """Экспорт метода в TensorRT через torch.jit.trace + torch_tensorrt.compile.

    Args:
        segmenter: Экземпляр сегментера.
        method_name: Ключ метода в method_map.
        output_path: Путь для сохранения .trt файла.
        precision: Точность вычислений.
        input_shape: Основная форма входа.
        min_shape: Минимальная форма для динамических размеров.
        max_shape: Максимальная форма для динамических размеров.

    Returns:
        bool: True при успешном экспорте.
    """
    try:
        import torch_tensorrt as torchtrt
    except ImportError:
        print("❌ TRT: torch_tensorrt не установлен")
        return False

    if method_name not in segmenter.method_map:
        print(f"❌ TRT: метод '{method_name}' не найден")
        return False

    # Wrapper с фиксированной формой выхода
    class TRTWrapper(torch.nn.Module):
        def __init__(
            self,
            seg: Any,
            method_name: str,
            precision: str,
            fixed_shape: ShapeType,
        ) -> None:
            super().__init__()
            self.seg: Any = seg
            self.method_name: str = method_name
            self.precision: str = precision
            self.fixed_shape: ShapeType = fixed_shape
            f: Any = seg.method_map[method_name]
            if hasattr(f, "_torchdynamo_orig_callable"):
                f = f._torchdynamo_orig_callable
            elif hasattr(f, "__wrapped__"):
                f = f.__wrapped__
            self.func: Any = f

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result: torch.Tensor = self.func(
                self.seg, x, precision=self.precision, export_mode=True
            )
            # Фиксируем выход к (1, 1, H, W) из fixed_shape
            _, _, target_h, target_w = self.fixed_shape
            if result.dim() == 2:
                result = result.unsqueeze(0).unsqueeze(0)
            elif result.dim() == 3:
                result = result.unsqueeze(0)
            return result.view(1, 1, target_h, target_w).float()

    wrapper: TRTWrapper = TRTWrapper(
        segmenter, method_name, precision, input_shape
    ).eval()
    wrapper = wrapper.to(segmenter.device)

    sample: torch.Tensor = torch.randn(
        *input_shape, device=segmenter.device, dtype=torch.float32
    )

    try:
        # Trace с фиксированным входом
        traced: torch.jit.ScriptModule = torch.jit.trace(
            wrapper, sample, check_trace=False
        )
        print(f"   ✅ torch.jit.trace OK для '{method_name}'")
    except Exception as e:
        print(f"❌ TRT: torch.jit.trace failed для '{method_name}': {e}")
        return False

    try:
        enabled_precisions: Set[torch.dtype] = {torch.float32}
        if precision == "fp16" and torch.cuda.is_available():
            cap: Tuple[int, int] = torch.cuda.get_device_capability()
            if cap[0] >= 6:
                enabled_precisions.add(torch.float16)

        min_shape = min_shape or input_shape
        max_shape = max_shape or input_shape

        # КЛЮЧЕВОЕ: ir="torchscript" + require_full_compilation=True
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
            ir="torchscript",  # Явно указываем IR
            require_full_compilation=True,  # Отключаем partial compilation
            truncate_long_and_double=True,
            debug=False,
        )

        torch.jit.save(trt_model, output_path)
        print(f"✅ TensorRT engine сохранён (JIT): {output_path}")
        return True

    except Exception as e:
        print(f"❌ TRT compile failed для '{method_name}': {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        if precision == "fp16":
            print(f"💡 Повтор с fp32...")
            return export_method_to_trt_jit(
                segmenter,
                method_name,
                output_path,
                precision="fp32",
                input_shape=input_shape,
            )
        return False
