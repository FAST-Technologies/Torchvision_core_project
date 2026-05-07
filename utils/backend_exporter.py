# utils/backend_exporter.py
import torch
import torch_tensorrt
from segmenters.NewTorchSegmenter import TorchSegmenter2
from typing import Optional, Tuple, Dict, Any, List
import os

class MethodExportWrapper(torch.nn.Module):
    """
    Wrapper для экспорта bound methods с фиксированными параметрами.
    Ключевое: все параметры должны быть явно объявлены в __init__ и использованы в forward.
    """
    def __init__(
        self, 
        bound_method, 
        fixed_kwargs: Optional[Dict[str, Any]] = None,
        precision: Optional[str] = None
    ):
        super().__init__()
        self.bound_method = bound_method
        # 🔥 Явно регистрируем параметры как буферы/атрибуты для torch.export
        self.fixed_kwargs = fixed_kwargs or {}
        if precision:
            self.fixed_kwargs['precision'] = precision
        # 🔥 Фиксируем параметры как атрибуты модуля (не dict!)
        for k, v in self.fixed_kwargs.items():
            if isinstance(v, (int, float, str, bool)):
                setattr(self, f'_param_{k}', v)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 🔥 Собираем kwargs из атрибутов модуля (torch.export-friendly)
        kwargs = {}
        for attr in dir(self):
            if attr.startswith('_param_'):
                key = attr[7:]  # remove '_param_'
                kwargs[key] = getattr(self, attr)
        # Вызов bound method
        return self.bound_method(x, **kwargs)

def export_method_to_trt_dynamo2(
    segmenter: TorchSegmenter2,
    method_name: str,
    output_path: str,
    sample_input: Optional[torch.Tensor] = None,
    precision: str = "fp16",
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512)
) -> bool:
    """
    Прямой экспорт через torch.export + torch_tensorrt.dynamo (рекомендуемый путь).
    Обходит проблемы с TorchScript и ONNX.
    """
    try:
        import torch_tensorrt as torchtrt
    except ImportError:
        print("❌ torch-tensorrt not installed: pip install torch-tensorrt")
        return False
    
    # Подготовка входного тензора
    if sample_input is None:
        sample_input = torch.randn(*input_shape, device=segmenter.device)
    
    # Получаем функцию метода
    if method_name not in segmenter.method_map:
        print(f"❌ Method {method_name} not found")
        return False
    
    class TRTWrapper(torch.nn.Module):
        def __init__(self, func, precision: str):
            super().__init__()
            self.func = func
            self.precision = precision  # 🔥 Атрибут для tracing
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.func(x, precision=self.precision)
    
    func = segmenter.method_map[method_name]
    # Распаковка если скомпилирована
    if hasattr(func, '_torchdynamo_orig_callable'):
        func = func._torchdynamo_orig_callable
    
    wrapper = TRTWrapper(func, precision).eval().to(segmenter.device)
    
    # 🔥 Тестовый прогон для инициализации
    _ = wrapper(sample_input)
    
    # Экспорт через torch.export (новый стабильный API)
    try:
        exported_program = torch.export.export(
            wrapper,
            (sample_input,),
            strict=False  # Разрешить динамические операции
        )
        print(f"✅ torch.export successful for {method_name}")
    except Exception as e:
        print(f"⚠️  torch.export failed: {e}")
        print("💡 Trying fallback with simplified input signature...")
        simple_wrapper = MethodExportWrapper(bound_func, fixed_kwargs={})
        try:
            exported_program = torch.export.export(
                simple_wrapper,
                (sample_input,),
                strict=False
            )
        except Exception as e2:
            print(f"❌ Export fallback failed: {e2}")
            return False
        
    # if precision == "fp16":
    #     if not torch.cuda.is_available():
    #         print("⚠️  fp16 requires CUDA")
    #         precision = "fp32"
    #     else:
    #         cap = torch.cuda.get_device_capability(0)
    #         if cap[0] < 6:  # Pascal+ для fp16
    #             print(f"⚠️  fp16 not fully supported on compute capability {cap[0]}.{cap[1]}")
    #             precision = "fp32"
    # Настройка precision
    target_dtype = torch.float16 if precision == "fp16" and torch.cuda.is_available() else torch.float32
    
    # Подготовка спецификации входа для TensorRT
    input_spec = torchtrt.Input(
        min_shape=input_shape,
        opt_shape=input_shape,
        max_shape=input_shape,
        dtype=target_dtype
    )
    
    # Компиляция в TensorRT через Dynamo IR
    try:
        trt_model = torchtrt.dynamo.compile(
            exported_program,
            inputs=[input_spec],
            device="cuda:0",  # 🔥 Строка вместо torch.device()
            enabled_precisions={target_dtype},
            min_block_size=1,
            debug=True,
            truncate_long_and_double=True,
        )
        
        # Сохранение как TorchScript модуля
        torch.jit.save(trt_model, output_path)
        print(f"✅ TensorRT engine saved: {output_path}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        # 🔥 Специальная обработка stoi error
        if "stoi" in error_msg.lower():
            print(f"⚠️  TensorRT stoi error — возможно, несовместимая версия")
            print(f"💡 Попробуйте: `pip install --upgrade torch-tensorrt`")
            print(f"💡 Или используйте precision='fp32' вместо 'fp16'")
        print(f"❌ TensorRT compilation failed: {e}")
        return False
    
# ✅ Исправленная функция export_method_to_trt_dynamo
# def export_method_to_trt_dynamo(
#     segmenter: TorchSegmenter2,
#     method_name: str,
#     output_path: str,
#     precision: str = "fp32",  # 🔥 По умолчанию fp32 для стабильности
#     **kwargs
# ) -> bool:
#     try:
#         import torch_tensorrt as torchtrt
#     except ImportError:
#         print("❌ torch-tensorrt not installed")
#         return False
    
#     # Подготовка входного тензора
#     sample_input = torch.randn(1, 3, 512, 512, device=segmenter.device)
    
#     if method_name not in segmenter.method_map:
#         return False
    
#     func = segmenter.method_map[method_name]
#     # Распаковка компилированной функции
#     if hasattr(func, '_torchdynamo_orig_callable'):
#         func = func._torchdynamo_orig_callable
    
#     # 🔥 Простая обёртка без лишних параметров
#     class TRTWrapper(torch.nn.Module):
#         def __init__(self, func):
#             super().__init__()
#             self.func = func
#         def forward(self, x):
#             return self.func(x)
    
#     wrapper = TRTWrapper(func).eval().to(segmenter.device)
    
#     # Экспорт через torch.export
#     try:
#         exported_program = torch.export.export(
#             wrapper, (sample_input,), strict=False
#         )
#     except Exception as e:
#         print(f"⚠️ torch.export failed: {e}")
#         return False
    
#     # 🔥 Настройка precision БЕЗ enabled_precisions при use_explicit_typing
#     target_dtype = torch.float16 if precision == "fp16" else torch.float32
    
#     # 🔥 КЛЮЧЕВОЕ: Используем ir="dynamo" и убираем конфликтные параметры
#     try:
#         trt_model = torchtrt.dynamo.compile(
#             exported_program,
#             inputs=[torchtrt.Input(
#                 min_shape=(1, 3, 256, 256),
#                 opt_shape=(1, 3, 512, 512), 
#                 max_shape=(1, 3, 1024, 1024),
#                 dtype=target_dtype
#             )],
#             device="cuda:0",
#             ir="dynamo",  # 🔥 Явно указываем IR
#             min_block_size=1,
#             # 🔥 УБРАТЬ: enabled_precisions={target_dtype} — вызывает конфликт!
#             truncate_long_and_double=True,
#             debug=False,
#         )
#         torch.jit.save(trt_model, output_path)
#         print(f"✅ TensorRT saved: {output_path}")
#         return True
#     except Exception as e:
#         print(f"❌ TRT compile failed: {e}")
#         # Fallback на fp32 если fp16 не работает
#         if precision == "fp16":
#             print("💡 Retrying with fp32 precision...")
#             return export_method_to_trt_dynamo_fixed(
#                 segmenter, method_name, output_path, precision="fp32"
#             )
#         return False

# utils/backend_exporter.py
def export_method_to_trt_dynamo(
    segmenter: TorchSegmenter2,
    method_name: str,
    output_path: str,
    precision: str = "fp16",
) -> bool:
    """Экспорт метода в TensorRT через torch.export (PyTorch 2.0+)"""
    try:
        import torch_tensorrt
        from torch.export import export
        
        # Получаем функцию метода
        if method_name not in segmenter.method_map:
            print(f"❌ Метод {method_name} не найден")
            return False
        
        func = segmenter.method_map[method_name]
        
        # Создаём dummy input
        example_input = torch.randn(
            1, 3, 512, 512, 
            device=segmenter.device, 
            dtype=torch.float32  # 🔥 Всегда fp32 для экспорта
        )
        
        # 🔥 Экспорт через torch.export (вместо torch.compile)
        exported_program = export(func, (example_input,))
        
        # Конвертация в TensorRT
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=[example_input],
            enabled_precisions={torch.float16 if precision == "fp16" else torch.float32},
        )
        
        # Сохранение через torch.jit
        torch.jit.save(trt_gm, output_path)
        print(f"✅ TensorRT engine сохранён: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ TRT compile failed: {e}")
        return False


def export_method_to_onnx_safe2(
    segmenter: TorchSegmenter2,
    method_name: str,
    output_path: str,
    sample_input: Optional[torch.Tensor] = None,
    opset_version: int = 13  # Более совместимая версия
) -> bool:
    """
    Безопасный экспорт в ONNX с обработкой ключевых аргументов.
    """
    if sample_input is None:
        sample_input = torch.randn(1, 3, 512, 512, device=segmenter.device)
    
    if method_name not in segmenter.method_map:
        print(f"❌ Method {method_name} not found")
        return False
    
    func = segmenter.method_map[method_name]
    
    # Обёртка с фиксированными параметрами для ONNX-совместимости
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, segmenter: TorchSegmenter2, method_name: str, precision: str):
            super().__init__()
            self.segmenter = segmenter
            self.method_name = method_name
            self.precision = precision
            # 🔥 Получаем функцию ДО компиляции (если нужно)
            func = segmenter.method_map[method_name]
            if hasattr(func, '_torchdynamo_orig_callable'):
                func = func._torchdynamo_orig_callable
            self.func = func
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 🔥 Явный вызов с именованными параметрами (без **kwargs)
            return self.func(x, precision=self.precision)
    
    # Получение оригинальной функции если она скомпилирована
    if hasattr(func, '_torchdynamo_orig_callable'):
        func = func._torchdynamo_orig_callable
    elif hasattr(func, '__wrapped__'):
        func = func.__wrapped__
    
    wrapper = ONNXWrapper(segmenter, method_name, segmenter.precision_manager.default_precision)
    wrapper.eval().to(segmenter.device)

    _ = wrapper(sample_input)
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                sample_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"}
                },
                do_constant_folding=True,
                verbose=False
            )
        print(f"✅ ONNX exported: {output_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False
    
# ✅ Исправленный экспорт ONNX с фиксацией форм
def export_method_to_onnx_safe(
    segmenter: TorchSegmenter2,
    method_name: str,
    output_path: str,
    opset_version: int = 15,  # 🔥 Более стабильная версия
) -> bool:
    sample_input = torch.randn(1, 3, 512, 512, device=segmenter.device)
    
    if method_name not in segmenter.method_map:
        return False
    
    func = segmenter.method_map[method_name]
    if hasattr(func, '_torchdynamo_orig_callable'):
        func = func._torchdynamo_orig_callable
    
    # 🔥 Обёртка с явным возвратом фиксированной формы
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = self.func(x)
            # 🔥 Гарантируем выходную форму (1, 1, H, W) для совместимости
            if result.dim() == 2:
                result = result.unsqueeze(0).unsqueeze(0)
            elif result.dim() == 3:
                result = result.unsqueeze(0)
            return result
    
    wrapper = ONNXWrapper(func).eval().to(segmenter.device)
    
    # 🔥 Тестовый прогон для инициализации буферов
    _ = wrapper(sample_input)
    
    try:
        torch.onnx.export(
            wrapper,
            sample_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,  # 🔥 15 вместо 17
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"}
            },
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,  # 🔥 Явно указываем EVAL
            verbose=False,
        )
        
        # 🔥 Валидация экспортированной модели
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print(f"✅ ONNX exported & validated: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False