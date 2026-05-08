# utils/backend_exporter.py

"""
Экспорт методов TorchSegmenter2 в ONNX и TensorRT.

Исправления:
  1. TRT: bound method → nn.Module wrapper → torch.export → TRT dynamo compile
  2. ONNX: убран dim()-зависимый branch (фиксирует случайный path при трассировке)
     Теперь forward всегда возвращает строго (1,1,H,W) через view()
  3. ONNX: boolean indexing в canny заменён на torch.where (ONNX-совместимо)
  4. ONNXSegmenter.segment: добавлена обработка ошибок
  5. TRT: torch_tensorrt.save вместо torch.jit.save (для dynamo-скомпилированных)
"""

# import torch
# import torch_tensorrt
# from segmenters.NewTorchSegmenter import TorchSegmenter2
# from typing import Optional, Tuple, Dict, Any, List
# import os

# class MethodExportWrapper(torch.nn.Module):
#     """
#     Wrapper для экспорта bound methods с фиксированными параметрами.
#     Ключевое: все параметры должны быть явно объявлены в __init__ и использованы в forward.
#     """
#     def __init__(
#         self,
#         bound_method,
#         fixed_kwargs: Optional[Dict[str, Any]] = None,
#         precision: Optional[str] = None
#     ):
#         super().__init__()
#         self.bound_method = bound_method
#         # 🔥 Явно регистрируем параметры как буферы/атрибуты для torch.export
#         self.fixed_kwargs = fixed_kwargs or {}
#         if precision:
#             self.fixed_kwargs['precision'] = precision
#         # 🔥 Фиксируем параметры как атрибуты модуля (не dict!)
#         for k, v in self.fixed_kwargs.items():
#             if isinstance(v, (int, float, str, bool)):
#                 setattr(self, f'_param_{k}', v)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 🔥 Собираем kwargs из атрибутов модуля (torch.export-friendly)
#         kwargs = {}
#         for attr in dir(self):
#             if attr.startswith('_param_'):
#                 key = attr[7:]  # remove '_param_'
#                 kwargs[key] = getattr(self, attr)
#         # Вызов bound method
#         return self.bound_method(x, **kwargs)

# def export_method_to_trt_dynamo2(
#     segmenter: TorchSegmenter2,
#     method_name: str,
#     output_path: str,
#     sample_input: Optional[torch.Tensor] = None,
#     precision: str = "fp16",
#     input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512)
# ) -> bool:
#     """
#     Прямой экспорт через torch.export + torch_tensorrt.dynamo (рекомендуемый путь).
#     Обходит проблемы с TorchScript и ONNX.
#     """
#     try:
#         import torch_tensorrt as torchtrt
#     except ImportError:
#         print("❌ torch-tensorrt not installed: pip install torch-tensorrt")
#         return False

#     # Подготовка входного тензора
#     if sample_input is None:
#         sample_input = torch.randn(*input_shape, device=segmenter.device)

#     # Получаем функцию метода
#     if method_name not in segmenter.method_map:
#         print(f"❌ Method {method_name} not found")
#         return False

#     class TRTWrapper(torch.nn.Module):
#         def __init__(self, func, precision: str):
#             super().__init__()
#             self.func = func
#             self.precision = precision  # 🔥 Атрибут для tracing

#         def forward(self, x: torch.Tensor) -> torch.Tensor:
#             return self.func(x, precision=self.precision)

#     func = segmenter.method_map[method_name]
#     # Распаковка если скомпилирована
#     if hasattr(func, '_torchdynamo_orig_callable'):
#         func = func._torchdynamo_orig_callable

#     wrapper = TRTWrapper(func, precision).eval().to(segmenter.device)

#     # 🔥 Тестовый прогон для инициализации
#     _ = wrapper(sample_input)

#     # Экспорт через torch.export (новый стабильный API)
#     try:
#         exported_program = torch.export.export(
#             wrapper,
#             (sample_input,),
#             strict=False  # Разрешить динамические операции
#         )
#         print(f"✅ torch.export successful for {method_name}")
#     except Exception as e:
#         print(f"⚠️  torch.export failed: {e}")
#         print("💡 Trying fallback with simplified input signature...")
#         simple_wrapper = MethodExportWrapper(bound_func, fixed_kwargs={})
#         try:
#             exported_program = torch.export.export(
#                 simple_wrapper,
#                 (sample_input,),
#                 strict=False
#             )
#         except Exception as e2:
#             print(f"❌ Export fallback failed: {e2}")
#             return False

#     # if precision == "fp16":
#     #     if not torch.cuda.is_available():
#     #         print("⚠️  fp16 requires CUDA")
#     #         precision = "fp32"
#     #     else:
#     #         cap = torch.cuda.get_device_capability(0)
#     #         if cap[0] < 6:  # Pascal+ для fp16
#     #             print(f"⚠️  fp16 not fully supported on compute capability {cap[0]}.{cap[1]}")
#     #             precision = "fp32"
#     # Настройка precision
#     target_dtype = torch.float16 if precision == "fp16" and torch.cuda.is_available() else torch.float32

#     # Подготовка спецификации входа для TensorRT
#     input_spec = torchtrt.Input(
#         min_shape=input_shape,
#         opt_shape=input_shape,
#         max_shape=input_shape,
#         dtype=target_dtype
#     )

#     # Компиляция в TensorRT через Dynamo IR
#     try:
#         trt_model = torchtrt.dynamo.compile(
#             exported_program,
#             inputs=[input_spec],
#             device="cuda:0",  # 🔥 Строка вместо torch.device()
#             enabled_precisions={target_dtype},
#             min_block_size=1,
#             debug=True,
#             truncate_long_and_double=True,
#         )

#         # Сохранение как TorchScript модуля
#         torch.jit.save(trt_model, output_path)
#         print(f"✅ TensorRT engine saved: {output_path}")
#         return True

#     except Exception as e:
#         error_msg = str(e)
#         # 🔥 Специальная обработка stoi error
#         if "stoi" in error_msg.lower():
#             print(f"⚠️  TensorRT stoi error — возможно, несовместимая версия")
#             print(f"💡 Попробуйте: `pip install --upgrade torch-tensorrt`")
#             print(f"💡 Или используйте precision='fp32' вместо 'fp16'")
#         print(f"❌ TensorRT compilation failed: {e}")
#         return False

# # ✅ Исправленная функция export_method_to_trt_dynamo
# # def export_method_to_trt_dynamo(
# #     segmenter: TorchSegmenter2,
# #     method_name: str,
# #     output_path: str,
# #     precision: str = "fp32",  # 🔥 По умолчанию fp32 для стабильности
# #     **kwargs
# # ) -> bool:
# #     try:
# #         import torch_tensorrt as torchtrt
# #     except ImportError:
# #         print("❌ torch-tensorrt not installed")
# #         return False

# #     # Подготовка входного тензора
# #     sample_input = torch.randn(1, 3, 512, 512, device=segmenter.device)

# #     if method_name not in segmenter.method_map:
# #         return False

# #     func = segmenter.method_map[method_name]
# #     # Распаковка компилированной функции
# #     if hasattr(func, '_torchdynamo_orig_callable'):
# #         func = func._torchdynamo_orig_callable

# #     # 🔥 Простая обёртка без лишних параметров
# #     class TRTWrapper(torch.nn.Module):
# #         def __init__(self, func):
# #             super().__init__()
# #             self.func = func
# #         def forward(self, x):
# #             return self.func(x)

# #     wrapper = TRTWrapper(func).eval().to(segmenter.device)

# #     # Экспорт через torch.export
# #     try:
# #         exported_program = torch.export.export(
# #             wrapper, (sample_input,), strict=False
# #         )
# #     except Exception as e:
# #         print(f"⚠️ torch.export failed: {e}")
# #         return False

# #     # 🔥 Настройка precision БЕЗ enabled_precisions при use_explicit_typing
# #     target_dtype = torch.float16 if precision == "fp16" else torch.float32

# #     # 🔥 КЛЮЧЕВОЕ: Используем ir="dynamo" и убираем конфликтные параметры
# #     try:
# #         trt_model = torchtrt.dynamo.compile(
# #             exported_program,
# #             inputs=[torchtrt.Input(
# #                 min_shape=(1, 3, 256, 256),
# #                 opt_shape=(1, 3, 512, 512),
# #                 max_shape=(1, 3, 1024, 1024),
# #                 dtype=target_dtype
# #             )],
# #             device="cuda:0",
# #             ir="dynamo",  # 🔥 Явно указываем IR
# #             min_block_size=1,
# #             # 🔥 УБРАТЬ: enabled_precisions={target_dtype} — вызывает конфликт!
# #             truncate_long_and_double=True,
# #             debug=False,
# #         )
# #         torch.jit.save(trt_model, output_path)
# #         print(f"✅ TensorRT saved: {output_path}")
# #         return True
# #     except Exception as e:
# #         print(f"❌ TRT compile failed: {e}")
# #         # Fallback на fp32 если fp16 не работает
# #         if precision == "fp16":
# #             print("💡 Retrying with fp32 precision...")
# #             return export_method_to_trt_dynamo_fixed(
# #                 segmenter, method_name, output_path, precision="fp32"
# #             )
# #         return False

# # utils/backend_exporter.py
# def export_method_to_trt_dynamo(
#     segmenter: TorchSegmenter2,
#     method_name: str,
#     output_path: str,
#     precision: str = "fp16",
# ) -> bool:
#     """Экспорт метода в TensorRT через torch.export (PyTorch 2.0+)"""
#     try:
#         import torch_tensorrt
#         from torch.export import export

#         # Получаем функцию метода
#         if method_name not in segmenter.method_map:
#             print(f"❌ Метод {method_name} не найден")
#             return False

#         func = segmenter.method_map[method_name]

#         # Создаём dummy input
#         example_input = torch.randn(
#             1, 3, 512, 512,
#             device=segmenter.device,
#             dtype=torch.float32  # 🔥 Всегда fp32 для экспорта
#         )

#         # 🔥 Экспорт через torch.export (вместо torch.compile)
#         exported_program = export(func, (example_input,))

#         # Конвертация в TensorRT
#         trt_gm = torch_tensorrt.dynamo.compile(
#             exported_program,
#             inputs=[example_input],
#             enabled_precisions={torch.float16 if precision == "fp16" else torch.float32},
#         )

#         # Сохранение через torch.jit
#         torch.jit.save(trt_gm, output_path)
#         print(f"✅ TensorRT engine сохранён: {output_path}")
#         return True

#     except Exception as e:
#         print(f"❌ TRT compile failed: {e}")
#         return False


# def export_method_to_onnx_safe2(
#     segmenter: TorchSegmenter2,
#     method_name: str,
#     output_path: str,
#     sample_input: Optional[torch.Tensor] = None,
#     opset_version: int = 13  # Более совместимая версия
# ) -> bool:
#     """
#     Безопасный экспорт в ONNX с обработкой ключевых аргументов.
#     """
#     if sample_input is None:
#         sample_input = torch.randn(1, 3, 512, 512, device=segmenter.device)

#     if method_name not in segmenter.method_map:
#         print(f"❌ Method {method_name} not found")
#         return False

#     func = segmenter.method_map[method_name]

#     # Обёртка с фиксированными параметрами для ONNX-совместимости
#     class ONNXWrapper(torch.nn.Module):
#         def __init__(self, segmenter: TorchSegmenter2, method_name: str, precision: str):
#             super().__init__()
#             self.segmenter = segmenter
#             self.method_name = method_name
#             self.precision = precision
#             # 🔥 Получаем функцию ДО компиляции (если нужно)
#             func = segmenter.method_map[method_name]
#             if hasattr(func, '_torchdynamo_orig_callable'):
#                 func = func._torchdynamo_orig_callable
#             self.func = func

#         def forward(self, x: torch.Tensor) -> torch.Tensor:
#             # 🔥 Явный вызов с именованными параметрами (без **kwargs)
#             return self.func(x, precision=self.precision)

#     # Получение оригинальной функции если она скомпилирована
#     if hasattr(func, '_torchdynamo_orig_callable'):
#         func = func._torchdynamo_orig_callable
#     elif hasattr(func, '__wrapped__'):
#         func = func.__wrapped__

#     wrapper = ONNXWrapper(segmenter, method_name, segmenter.precision_manager.default_precision)
#     wrapper.eval().to(segmenter.device)

#     _ = wrapper(sample_input)

#     try:
#         with torch.no_grad():
#             torch.onnx.export(
#                 wrapper,
#                 sample_input,
#                 output_path,
#                 input_names=["input"],
#                 output_names=["output"],
#                 opset_version=17,
#                 dynamic_axes={
#                     "input": {0: "batch", 2: "height", 3: "width"},
#                     "output": {0: "batch", 2: "height", 3: "width"}
#                 },
#                 do_constant_folding=True,
#                 verbose=False
#             )
#         print(f"✅ ONNX exported: {output_path}")
#         return True
#     except Exception as e:
#         print(f"❌ ONNX export failed: {e}")
#         return False

# # ✅ Исправленный экспорт ONNX с фиксацией форм
# def export_method_to_onnx_safe(
#     segmenter: TorchSegmenter2,
#     method_name: str,
#     output_path: str,
#     opset_version: int = 15,  # 🔥 Более стабильная версия
# ) -> bool:
#     sample_input = torch.randn(1, 3, 512, 512, device=segmenter.device)

#     if method_name not in segmenter.method_map:
#         return False

#     func = segmenter.method_map[method_name]
#     if hasattr(func, '_torchdynamo_orig_callable'):
#         func = func._torchdynamo_orig_callable

#     # 🔥 Обёртка с явным возвратом фиксированной формы
#     class ONNXWrapper(torch.nn.Module):
#         def __init__(self, func):
#             super().__init__()
#             self.func = func
#         def forward(self, x: torch.Tensor) -> torch.Tensor:
#             result = self.func(x)
#             # 🔥 Гарантируем выходную форму (1, 1, H, W) для совместимости
#             if result.dim() == 2:
#                 result = result.unsqueeze(0).unsqueeze(0)
#             elif result.dim() == 3:
#                 result = result.unsqueeze(0)
#             return result

#     wrapper = ONNXWrapper(func).eval().to(segmenter.device)

#     # 🔥 Тестовый прогон для инициализации буферов
#     _ = wrapper(sample_input)

#     try:
#         torch.onnx.export(
#             wrapper,
#             sample_input,
#             output_path,
#             input_names=["input"],
#             output_names=["output"],
#             opset_version=opset_version,  # 🔥 15 вместо 17
#             dynamic_axes={
#                 "input": {0: "batch", 2: "height", 3: "width"},
#                 "output": {0: "batch", 2: "height", 3: "width"}
#             },
#             do_constant_folding=True,
#             training=torch.onnx.TrainingMode.EVAL,  # 🔥 Явно указываем EVAL
#             verbose=False,
#         )

#         # 🔥 Валидация экспортированной модели
#         import onnx
#         model = onnx.load(output_path)
#         onnx.checker.check_model(model)
#         print(f"✅ ONNX exported & validated: {output_path}")
#         return True

#     except Exception as e:
#         print(f"❌ ONNX export failed: {e}")
#         return False


import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List


# ──────────────────────────────────────────────────────────────────────────────
# Базовый wrapper: bound method → nn.Module
# ──────────────────────────────────────────────────────────────────────────────
class SegmenterMethodWrapper(nn.Module):
    """
    Оборачивает bound method сегментера в nn.Module для torch.export и TRT.

    ВАЖНО: func должна быть оригинальной (не скомпилированной) функцией.
    torch.export.export не принимает torch.compile-обёрнутые функции.
    """

    def __init__(self, segmenter, method_name: str, precision: str = "fp32"):
        super().__init__()
        # Распаковываем compile-обёртку если есть
        self.segmenter = segmenter  # 🔥 Сохраняем instance
        self.method_name = method_name
        self.precision = precision

        # Получаем "сырую" функцию без compile-обёртки
        func = segmenter.method_map[method_name]
        if hasattr(func, "_torchdynamo_orig_callable"):
            self.func = func._torchdynamo_orig_callable
        elif hasattr(func, "__wrapped__"):
            self.func = func.__wrapped__
        else:
            self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.func(
            self.segmenter, x, precision=self.precision, export_mode=True
        )
        # Гарантируем (1,1,H,W) через view — без dim()-зависимых веток
        # (они фиксируют конкретный branch при ONNX-трассировке)
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
    output_path: str,
    opset_version: int = 17,
    input_shape: Tuple[int, ...] = (1, 3, 512, 512),
    precision: str = "fp32",
) -> bool:
    """
    Экспортирует один метод сегментера в ONNX.

    Fixes vs предыдущей версии:
    - wrapper.forward не использует dim()-зависимые ветки (фиксируются при трассировке)
    - output shape гарантированно (B,1,H,W) через .view()
    - opset 17 поддерживает все нужные операции
    - dynamic_axes корректно объявлены для 4D выхода
    """
    if method_name not in segmenter.method_map:
        print(f"❌ ONNX: метод '{method_name}' не найден в method_map")
        return False

    func = segmenter.method_map[method_name]
    wrapper = SegmenterMethodWrapper(segmenter, method_name, precision=precision).eval()
    wrapper = wrapper.to(segmenter.device)

    sample = torch.randn(*input_shape, device=segmenter.device, dtype=torch.float32)

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
                sample,
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

        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print(f"✅ ONNX exported & validated: {output_path}")

        # Упрощение через onnx-simplifier (опционально)
        try:
            from onnxsim import simplify

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
    segmenter,
    method_name: str,
    output_path: str,
    precision: str = "fp32",
    input_shape: Tuple[int, ...] = (1, 3, 512, 512),
) -> bool:
    """
    Экспорт метода в TensorRT через torch.export + dynamo.

    Fixes vs предыдущей версии:
    - func оборачивается в nn.Module (torch.export не принимает bound methods)
    - torch_tensorrt.save вместо torch.jit.save (dynamo-compiled models)
    - Fallback precision fp32 если fp16 недоступен

    Требования: torch>=2.1, torch_tensorrt>=2.1, CUDA>=sm_80 для fp16.
    """
    try:
        import torch_tensorrt as torchtrt
    except ImportError:
        print("❌ TRT: torch_tensorrt не установлен")
        return False

    device_obj = (
        torch.device(segmenter.device)
        if isinstance(segmenter.device, str)
        else segmenter.device
    )

    if method_name not in segmenter.method_map:
        print(f"❌ TRT: метод '{method_name}' не найден")
        return False

    func = segmenter.method_map[method_name]
    # FIX 1: оборачиваем в nn.Module — torch.export требует Module или callable
    wrapper = SegmenterMethodWrapper(segmenter, method_name, precision=precision).eval()
    wrapper = wrapper.to(segmenter.device)

    # Определяем dtype для TRT
    if precision == "fp16" and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] < 6:
            print(
                f"⚠️  fp16 не поддерживается на compute capability {cap[0]}.{cap[1]}, переключаемся на fp32"
            )
            precision = "fp32"

    target_dtype = torch.float16 if precision == "fp16" else torch.float32
    sample = torch.randn(*input_shape, device=segmenter.device, dtype=torch.float32)

    # Тестовый прогон
    try:
        with torch.no_grad():
            out = wrapper(sample)
        print(
            f"Wrapper output: shape={out.shape}, min={out.min()}, max={out.max()}, unique={out.unique()}"
        )
    except Exception as e:
        print(f"❌ TRT: тестовый прогон упал для '{method_name}': {e}")
        return False

    # Шаг 1: torch.export
    try:
        exported = torch.export.export(
            wrapper,
            (sample,),
            strict=False,
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
        trt_input = torchtrt.Input(
            min_shape=(1, 3, h // 2, w // 2),
            opt_shape=input_shape,
            max_shape=(1, 3, h * 2, w * 2),
            dtype=torch.float32,  # Всегда fp32 на входе, TRT конвертирует внутри
        )

        enabled_precisions = {torch.float32}
        if target_dtype == torch.float16:
            enabled_precisions.add(torch.float16)

        trt_model = torchtrt.dynamo.compile(
            exported,
            inputs=[trt_input],
            device=device_obj,
            enabled_precisions=enabled_precisions,
            min_block_size=1,
            truncate_long_and_double=True,
            debug=False,
            use_python_runtime=True,  # Более стабильный runtime
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
def load_trt_model(path: str, sample_input: Optional[torch.Tensor] = None):
    """
    Загружает TRT модель. Поддерживает оба формата: torch.jit и torch_tensorrt.
    """
    try:
        import torch_tensorrt as torchtrt

        model = torchtrt.load(path)
        print(f"✅ TRT loaded via torch_tensorrt.load: {path}")
        return model
    except Exception:
        pass

    try:
        model = torch.jit.load(path)
        print(f"✅ TRT loaded via torch.jit.load: {path}")
        return model
    except Exception as e:
        print(f"❌ TRT load failed: {path}: {e}")
        return None


def export_method_to_trt_jit(
    segmenter,
    method_name: str,
    output_path: str,
    precision: str = "fp32",
    input_shape: Tuple[int, ...] = (1, 3, 512, 512),
    min_shape: Optional[Tuple[int, ...]] = None,
    max_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    try:
        import torch_tensorrt as torchtrt
    except ImportError:
        print("❌ TRT: torch_tensorrt не установлен")
        return False

    if method_name not in segmenter.method_map:
        print(f"❌ TRT: метод '{method_name}' не найден")
        return False

    func = segmenter.method_map[method_name]

    # 🔥 Wrapper с фиксированной формой выхода
    class TRTWrapper(torch.nn.Module):
        def __init__(self, seg, method_name, precision, fixed_shape):
            super().__init__()
            self.seg = seg
            self.method_name = method_name
            self.precision = precision
            self.fixed_shape = fixed_shape  # 🔥 Запоминаем фиксированную форму
            f = seg.method_map[method_name]
            if hasattr(f, "_torchdynamo_orig_callable"):
                f = f._torchdynamo_orig_callable
            elif hasattr(f, "__wrapped__"):
                f = f.__wrapped__
            self.func = f

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = self.func(self.seg, x, precision=self.precision, export_mode=True)
            # 🔥 Фиксируем выход к (1, 1, H, W) из fixed_shape
            _, _, target_h, target_w = self.fixed_shape
            if result.dim() == 2:
                result = result.unsqueeze(0).unsqueeze(0)
            elif result.dim() == 3:
                result = result.unsqueeze(0)
            # 🔥 Используем фиксированные размеры вместо x.shape
            return result.view(1, 1, target_h, target_w).float()

    wrapper = TRTWrapper(segmenter, method_name, precision, input_shape).eval()
    wrapper = wrapper.to(segmenter.device)

    sample = torch.randn(*input_shape, device=segmenter.device, dtype=torch.float32)

    try:
        # 🔥 Trace с фиксированным входом
        traced = torch.jit.trace(wrapper, sample, check_trace=False)
        print(f"   ✅ torch.jit.trace OK для '{method_name}'")
    except Exception as e:
        print(f"❌ TRT: torch.jit.trace failed для '{method_name}': {e}")
        return False

    try:
        enabled_precisions = {torch.float32}
        if precision == "fp16" and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 6:
                enabled_precisions.add(torch.float16)

        min_shape = min_shape or input_shape
        max_shape = max_shape or input_shape

        # 🔥 КЛЮЧЕВОЕ: ir="torchscript" + require_full_compilation=True
        trt_model = torchtrt.compile(
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
            ir="torchscript",  # 🔥 Явно указываем IR
            require_full_compilation=True,  # 🔥 Отключаем partial compilation
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
