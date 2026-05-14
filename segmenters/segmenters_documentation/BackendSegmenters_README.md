# ⚡ BackendSegmenters — Высокопроизводительные сегментеры (ONNX / TensorRT)

## 📖 Описание
Модуль `BackendSegmenters.py` предоставляет два класса для **ускоренного инференса** предварительно обученных моделей сегментации:

| Класс | Бэкенд | Устройство | Формат модели | Сценарий использования |
|-------|--------|-----------|---------------|----------------------|
| `ONNXSegmenter` | ONNX Runtime | CUDA / CPU | `.onnx` | Кроссплатформенный деплой, баланс скорости/совместимости |
| `TRTSegmenter` | TensorRT (torch_tensorrt) | CUDA только | `.trt` / `torch.nn.Module` | Максимальная производительность на NVIDIA GPU |

> ⚠️ **Важно:** Данные сегментеры предназначены для **инференса готовых моделей**, а не для обучения. Для обучения и экспериментов используйте `TorchSegmenter` / `TorchSegmenter2`.

## ✨ Ключевые возможности
### 🟦 ONNXSegmenter
- 🔁 **Автоматический выбор провайдера:** `CUDAExecutionProvider` → `CPUExecutionProvider` fallback.
- ⚙️ **Графические оптимизации:** `ORT_ENABLE_ALL` для максимального ускорения выполнения.
- 📐 **Гибкая конфигурация входа:** параметр `input_shape=(B, C, H, W)` для моделей с фиксированной формой.
- 🛡️ **Устойчивость к ошибкам:** возврат пустой маски `(H, W)` при сбое инференса вместо `None` или исключения.
- 🔍 **Информативное логирование:** вывод имён вход/выход тензоров и их форм при инициализации.

### 🟩 TRTSegmenter
- 🚀 **Максимальная производительность:** инференс через скомпилированный TensorRT engine.
- 📦 **Гибкая загрузка:** поддержка пути к `.trt` файлу или готового `torch.nn.Module`.
- 🎚️ **Автоматическая конвертация:** `numpy → torch.Tensor → TRT inference → numpy` с нормализацией [0, 1].
- 🔒 **Безопасный инференс:** использование `torch.no_grad()` для экономии памяти и ускорения.
- 🛡️ **Устойчивость к ошибкам:** аналогично ONNXSegmenter — возврат пустой маски при исключении.

### Общие особенности
- 🔄 **Единый интерфейс:** наследование от `BaseSegmenter`, реализация `segment()` и `segment_with_mask()`.
- 🎨 **Поддержка форматов входа:** `np.ndarray`, `PIL.Image`, `str` (путь к файлу), `torch.Tensor`.
- 📤 **Стандартизированный выход:** бинарная маска `np.ndarray` формы `(H, W)`, dtype `uint8`, значения `{0, 255}`.
- 🧠 **Автоматическая предобработка:** конвертация в 3 канала, нормализация к [0, 1], добавление batch-измерения.
- ⚡ **Оптимизации памяти:** `torch.cuda.empty_cache()` при необходимости, отсутствие утечек при многократных вызовах.

## 🚀 Быстрый старт
### Использование ONNXSegmenter
```python
from segmenters.BackendSegmenters import ONNXSegmenter
import numpy as np
from PIL import Image

# Загрузка изображения
image = np.array(Image.open("test.jpg").convert("RGB"))

# Инициализация сегментера
segmenter = ONNXSegmenter(
    method_name="unet_onnx",
    onnx_path="models/unet_ade20k.onnx",
    device="cuda",  # или "cpu"
    input_shape=(1, 3, 512, 512)  # Ожидаемая форма входа модели
)

# Инференс
mask = segmenter.segment(image)

# Результат: бинарная маска (H, W), uint8, {0, 255}
print(f"Маска: форма={mask.shape}, dtype={mask.dtype}, уникальные значения={np.unique(mask)}")

# segment_with_mask (возвращает только бинарную маску)
binary_mask, prob_mask = segmenter.segment_with_mask(image)
assert prob_mask is None  # Вероятностная маска не поддерживается
```

### Использование TRTSegmenter
```python
from segmenters.BackendSegmenters import TRTSegmenter
import numpy as np

# Вариант 1: Загрузка из .trt файла
segmenter = TRTSegmenter(
    method_name="deeplab_trt",
    trt_model_or_path="models/deeplab_ade20k.trt",
    device="cuda"  # Только CUDA поддерживается
)

# Вариант 2: Использование уже загруженной модели
# from utils.backend_exporter import load_trt_model
# trt_model = load_trt_model("models/deeplab_ade20k.trt")
# segmenter = TRTSegmenter("deeplab_trt", trt_model, device="cuda")

# Инференс
image = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)
mask = segmenter.segment(image)

print(f"TRT маска: форма={mask.shape}, покрытие={(mask > 0).sum() / mask.size * 100:.2f}%")
```

### Интеграция с бенчмарками
```python
from testing.CpuCudaBenchmark import CpuCudaBenchmark

# Подготовка методов для сравнения
methods = {
    "unet_onnx_cuda": ONNXSegmenter("unet_onnx", "models/unet.onnx", device="cuda"),
    "unet_onnx_cpu": ONNXSegmenter("unet_onnx", "models/unet.onnx", device="cpu"),
    "deeplab_trt": TRTSegmenter("deeplab_trt", "models/deeplab.trt", device="cuda"),
}

# Запуск бенчмарка
benchmark = CpuCudaBenchmark(base_output_dir="./results/backend_benchmark")
df_results = benchmark.benchmark_all_methods(
    methods_dict=methods,
    image=image,
    test_name="onnx_vs_trt",
    save_artifacts=True
)

print(df_results[["method", "device", "mean_time", "speedup"]])
```

## ⚙️ Конфигурация
### Параметры `ONNXSegmenter.__init__()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `method_name` | `str` | — | Имя метода для логирования и отчётов |
| `onnx_path` | `str` | — | Путь к файлу модели `.onnx` |
| `device` | `Literal["cuda", "cpu"]` | `"cuda"` | Устройство для инференса |
| `input_shape` | `Tuple[int, int, int, int]` | `(1, 3, 512, 512)` | Ожидаемая форма входа: `(B, C, H, W)` |
| `**kwargs` | `Any` | — | Дополнительные параметры (сохраняются в `self.params`) |

### Параметры `TRTSegmenter.__init__()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `method_name` | `str` | — | Имя метода для логирования |
| `trt_model_or_path` | `Union[str, TRTModel]` | — | Путь к `.trt` файлу **или** готовая `torch.nn.Module` |
| `device` | `Literal["cuda", "cpu"]` | `"cuda"` | Устройство (поддерживается только `"cuda"`) |
| `**kwargs` | `Any` | — | Дополнительные параметры (сохраняются в `self.params`) |

### Требования к входным данным `segment(image)`
| Атрибут | Требование | Пример |
|---------|-----------|--------|
| **Тип** | `np.ndarray`, `PIL.Image`, `str`, `torch.Tensor` | `np.uint8` массив |
| **Форма** | `(H, W)` grayscale **или** `(H, W, 3)` RGB | `(512, 512, 3)` |
| **Диапазон** | `[0, 255]` для uint8 | `0` = чёрный, `255` = белый |
| **Каналы** | 1 или 3; автоматически конвертируется в 3 канала | Grayscale → RGB через `np.stack` |

### Формат выходных данных
```python
# segment() возвращает:
BinaryMask = np.ndarray  # Форма: (H, W), dtype: uint8, значения: {0, 255}

# segment_with_mask() возвращает:
Tuple[BinaryMask, Optional[ProbabilityMask]]
# Второе значение всегда None (вероятностные маски не поддерживаются)
```

## 🔄 Конвейер обработки данных
### ONNXSegmenter: предобработка → инференс → постобработка
```python
# 1. Предобработка (в _preprocess)
def _preprocess(image: np.ndarray) -> PreprocessedTensor:
    # Grayscale → RGB если нужно
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)  # (H,W) → (H,W,3)
    
    # HWC → CHW, uint8 [0,255] → float32 [0,1]
    tensor = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    
    # Добавление batch-измерения: (3,H,W) → (1,3,H,W)
    return np.expand_dims(tensor, 0)

# 2. Инференс
outputs = session.run([output_name], {input_name: tensor})

# 3. Постобработка
mask = outputs[0]
while mask.ndim > 2:  # Удаление лишних измерений
    mask = mask.squeeze(0)
if mask.dtype in (np.float32, np.float64):
    if mask.max() <= 1.0:  # Вероятности → бинаризация
        mask = (mask > 0.5).astype(np.uint8) * 255
    else:  # Уже [0, 255]
        mask = mask.astype(np.uint8)
return mask.astype(np.uint8)
```

### TRTSegmenter: аналогичный конвейер с torch-конвертацией
```python
# 1. Конвертация в torch.Tensor
tensor = (
    torch.from_numpy(image)      # numpy → torch
    .permute(2, 0, 1)            # HWC → CHW
    .float().div(255.0)          # uint8 [0,255] → float32 [0,1]
    .unsqueeze(0)                # (3,H,W) → (1,3,H,W)
    .to(device)                  # Перенос на CUDA
)

# 2. Инференс с torch.no_grad()
with torch.no_grad():
    out = model(tensor)

# 3. Постобработка
mask = (out.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
return mask
```

## 📂 Структура проекта и зависимости
### Требования для ONNXSegmenter
```bash
# Базовые зависимости
pip install onnxruntime  # CPU версия
# ИЛИ для GPU:
pip install onnxruntime-gpu  # Требует совместимой версии CUDA/cuDNN

# Проверка установки
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Ожидаемый вывод: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Требования для TRTSegmenter
```bash
# TensorRT через torch-tensorrt (требует PyTorch и CUDA)
pip install torch-tensorrt  # Следуйте инструктам на https://github.com/pytorch/TensorRT

# Дополнительные системные зависимости:
# - NVIDIA TensorRT (версия, совместимая с torch-tensorrt)
# - CUDA Toolkit (версия, совместимая с PyTorch)
# - cuDNN

# Проверка установки
python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
```

### Общие зависимости
```text
numpy>=1.20
torch>=1.9  # Для TRTSegmenter и конвертации тензоров
Pillow>=8.0  # Для работы с PIL.Image
```

## 🛠️ Обработка ошибок и устойчивость
### Стратегия возврата при сбое
Оба сегментера реализуют одинаковую логику обработки исключений:

```python
try:
    # ... инференс ...
    return mask
except Exception as e:
    logger.error(f"{backend} '{self.method}' inference error: {e}")
    # Возврат пустой маски того же размера, что и входное изображение
    h, w = image.shape[:2]
    return np.zeros((h, w), dtype=np.uint8)
```

**Преимущества:**
- ✅ Пакетное тестирование не прерывается при ошибке в одном методе.
- ✅ Бенчмарки могут агрегировать статистику даже при частичных сбоях.
- ✅ Вызывающий код может проверить результат: `if mask.sum() == 0: # пустая маска`.

### Рекомендации по отладке
1. **Включите логирование:**
   ```python
   import logging
   logging.getLogger("segmenters.BackendSegmenters").setLevel(logging.DEBUG)
   ```

2. **Проверьте формы тензоров:**
   ```python
   # ONNX: при инициализации выводится информация о input/output shape
   # TRT: убедитесь, что модель ожидает вход (1, 3, H, W)

3. **Тестируйте на маленьком изображении:**
   ```python
   small_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
   mask = segmenter.segment(small_image)  # Быстрая проверка конвейера
   ```

## ⚡ Производительность и оптимизации
### Ожидаемое ускорение (относительно PyTorch CPU)
| Бэкенд | Устройство | Относительная скорость | Примечание |
|--------|-----------|----------------------|------------|
| PyTorch | CPU | 1.0× | Базовая реализация |
| ONNX Runtime | CPU | 1.5–3.0× | Оптимизации графа, MKL/DNNL |
| ONNX Runtime | CUDA | 5–15× | Зависит от модели и GPU |
| TensorRT | CUDA | 10–30× | Максимальная оптимизация, fusion kernels |

### Факторы, влияющие на производительность
1. **Размер модели:** Большие модели (ResNet-101, ViT) выигрывают больше от TRT.
2. **Размер входа:** Фиксированный `input_shape` позволяет агрессивнее оптимизировать граф.
3. **Batch size:** Оба сегментера используют `batch_size=1`; для batch-инференса требуется модификация.
4. **Precision:** TensorRT поддерживает FP16/INT8 квантование — требует отдельной экспорта модели.

### Рекомендации для продакшена
- 🔹 **ONNX:** Используйте `input_shape`, соответствующий целевому разрешению, для избежания ресайза.
- 🔹 **TensorRT:** Экспортируйте модель с `fp16_mode=True` для 2–3× ускорения на современных GPU.
- 🔹 **Кэширование:** Сессии ONNX и TRT-модели кэшируются внутри экземпляра — переиспользуйте сегментер для множества изображений.
- 🔹 **Мониторинг:** Логируйте `execution_time` через `time.perf_counter()` для выявления деградации.

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование BackendSegmenters |
|--------|-------------------------------|
| `CpuCudaBenchmark` | Сравнение производительности ONNX/TRT vs PyTorch |
| `SegmentationBenchmark` | Включение ONNX/TRT моделей в сравнение архитектур |
| `SegmentationTester` | Тестирование деплойнутых моделей на отдельных изображениях |
| `utils/backend_exporter` | Экспорт моделей в ONNX/TRT форматы (функция `load_trt_model`) |

## 📄 Лицензия

Проект распространяется под лицензией **MIT**. См. файл [LICENSE](LICENSE) для деталей.

```
MIT License

Copyright (c) 2026 Torchvision_core_project contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---