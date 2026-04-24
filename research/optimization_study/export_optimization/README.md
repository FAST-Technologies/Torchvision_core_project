# 🔧 02_export_optimization — Экспорт методов в оптимизированные форматы
Исследование 02_export_optimization: ONNX + TensorRT + torch-tensorrt

Модуль для экспорта классических методов сегментации из `TorchSegmenter` 
в форматы, оптимизированные для инференса: **ONNX**, **TensorRT**.

## 🎯 Цель

Ускорение инференса классических методов за счёт:
- Компиляции графа вычислений (ONNX/TensorRT)
- Использования оптимизированных ядер (cuDNN, TensorRT)
- Квантования (FP16/INT8)

## 📦 Доступные бэкенды

| Бэкенд | Статус | Требования | Скорость | Точность |
|--------|--------|-----------|----------|----------|
| **ONNX Runtime** | ✅ Стабильный | `onnx`, `onnxruntime-gpu` | 1.2–2.5× | 🔹 Без потерь |
| **torch-tensorrt** | ✅ Официальный | `tensorrt>=10`, `torch-tensorrt` | 2.0–5.0× | 🔸 FP16: мин. потери |
| **torch2trt** | ⚠️ Опционально | Сборка из исходников | 2.0–4.0× | 🔸 Требует валидации |

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
# Основной стек (всегда работает)
pip install onnx onnxruntime-gpu onnx-simplifier

# Опционально: torch-tensorrt (если есть TensorRT 10+)
pip install torch-tensorrt

# Опционально: torch2trt (требует компиляции)
pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt
```

### 2. Проверка доступных бэкендов
```python
from optimization_study.02_export_optimization import get_registry

registry = get_registry()
registry.print_status()
# ✅ onnx — ONNX Runtime (CPU/CUDA/TensorRT EP)
# ✅ torch_tensorrt — torch-tensorrt (официальный NVIDIA бэкенд)
# ❌ torch2trt — torch2trt (NVIDIA-AI-IOT, требует сборки)
# 🎯 Recommended: onnx
```

### 3. Экспорт и бенчмарк (через скрипт)
```bash
# Базовый запуск с авто-выбором бэкенда
python benchmark.py \
  --methods sobel_edge,global_thresholding \
  --image ./test.jpg \
  --output ./results/

# Сравнение всех ONNX провайдеров
python benchmark.py \
  --methods sobel_edge \
  --backend onnx \
  --compare-all-providers \
  --verbose
```

### 4. Программное использование
```python
from segmenters.TorchSegmenter import TorchSegmenter
from optimization_study.02_export_optimization import ONNXOptimizer

# Инициализация
segmenter = TorchSegmenter(method="sobel_edge", device="cuda")
optimizer = ONNXOptimizer(segmenter, image_shape=(3, 512, 512))

# Экспорт
onnx_path = optimizer.export_method_to_onnx(
    "sobel_edge", 
    "sobel_edge.onnx",
    simplify_model=True  # onnx-simplifier
)

# Бенчмарк
results = optimizer.benchmark_onnx_vs_torch(
    "sobel_edge", 
    onnx_path,
    n_runs=100
)

print(f"⚡ Speedup: {results['speedup']:.2f}×")
# ⚡ Speedup: 1.85×
```

## 🔍 Сравнение бэкендов

### ONNX Runtime (рекомендуется)
```python
# ✅ Преимущества:
# - Кроссплатформенность (CPU/CUDA/TensorRT EP)
# - Простая установка
# - Стабильная работа

# ⚠️ Ограничения:
# - Не все операции поддерживаются в TensorRT EP
# - Требуется конвертация входных данных (torch → numpy)

optimizer = ONNXOptimizer(segmenter)
results = optimizer.benchmark_all_providers("method_name", "model.onnx")
# Returns: {'CUDAExecutionProvider': {...}, 'TensorrtExecutionProvider': {...}}
```

### torch-tensorrt
```python
# ✅ Преимущества:
# - Прямая интеграция с PyTorch API
# - Автоматическая оптимизация графа
# - Поддержка точного контроля точности (FP16/FP32)

# ⚠️ Ограничения:
# - Требует TensorRT 10+
# - Не все методы TorchSegmenter совместимы (динамика, условия)

from optimization_study.02_export_optimization import TorchTRTOptimizer

optimizer = TorchTRTOptimizer(segmenter)
compiled = optimizer.convert_method("sobel_edge", precision="fp16")
results = optimizer.benchmark("sobel_edge", compiled)
```

### torch2trt (опционально)
```python
# ⚠️ Только если собран из исходников!
# Используйте через BackendRegistry для безопасного доступа:

from optimization_study.02_export_optimization import get_registry

registry = get_registry()
if registry.get_backend_info("torch2trt").available:
    converter = registry.get_converter("torch2trt", segmenter)
    # ... использование
```

## 📊 Интерпретация результатов

```python
{
  "torch_mean_ms": 12.45,      # Среднее время оригинала (мс)
  "onnx_mean_ms": 6.73,        # Среднее время ONNX (мс)
  "speedup": 1.85,             # Коэффициент ускорения
  "speedup_formatted": "1.85×", # Человекочитаемый формат
  "onnx_provider": "CUDAExecutionProvider",  # Используемый провайдер
}
```

### Что считать хорошим результатом?
- **Speedup < 1.0×** — экспорт не дал выигрыша (возможно, накладные расходы)
- **Speedup 1.0–1.5×** — умеренное ускорение, приемлемо для сложных методов
- **Speedup > 1.5×** — отличный результат, рекомендуется для продакшена

## ⚠️ Частые проблемы

### 1. "No module named 'tensorrt'" при установке torch2trt
```bash
# Решение: используйте torch-tensorrt вместо torch2trt
pip install torch-tensorrt

# Или соберите torch2trt вручную:
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

### 2. ONNX экспорт падает с "Unsupported operator"
```python
# Решение: упростите метод или используйте dynamic_axes
optimizer.export_method_to_onnx(
    "method_name",
    "model.onnx",
    opset_version=17,  # Попробуйте 16 или 18
    simplify_model=False,  # Отключите onnx-simplifier для отладки
)
```

### 3. Нет ускорения на GPU для простых методов
```
Это ожидаемо! Накладные расходы на передачу данных (Host↔Device) 
могут превышать выигрыш от параллелизма для методов с временем <10 мс.

Решение:
- Используйте пакетную обработку (batch > 1)
- Кэшируйте входные тензоры на GPU
- Для простых методов оставайтесь на CPU
```

## 📁 Структура модуля

```
02_export_optimization/
├── __init__.py              # Публичный API
├── README.md                # Этот файл
├── utils.py                 # Вспомогательные функции
├── backend_registry.py      # Авто-детект бэкендов
├── onnx_converter.py        # ONNX экспорт + бенчмарк
├── torch_tensorrt_converter.py  # torch-tensorrt интеграция
├── torch2trt_converter.py   # torch2trt (опционально)
├── benchmark.py             # CLI скрипт бенчмарка
├── requirements.txt         # Зависимости
└── run_example.sh           # Пример запуска
```

## 🔗 Ссылки

- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [torch-tensorrt GitHub](https://github.com/pytorch/TensorRT)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [torch2trt GitHub](https://github.com/NVIDIA-AI-IOT/torch2trt)

---
*Модуль разработан для исследования оптимизации классических методов сегментации. 
Результаты бенчмарка могут варьироваться в зависимости от железа, драйверов и версий библиотек.*
```

---