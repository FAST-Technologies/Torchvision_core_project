# 🔢 03_quantization_study — Исследование квантования моделей
# 🔢 Исследование 3: Квантование моделей (Quantization Study)

Модуль для оценки влияния квантования на:
- ⚡ **Скорость** выполнения классических методов сегментации
- 🎯 **Точность** результатов относительно эталона (FP32)
- 💾 **Размер** модели в памяти

## 🎯 Цель исследования

Определить оптимальный баланс между производительностью, точностью и размером
при использовании различных схем квантования:

| Схема | Тип | Преимущества | Ограничения |
|-------|-----|--------------|-------------|
| **FP32** | `torch.float32` | ✅ Эталонная точность | — |
| **FP16** | `torch.float16` | ⚡ 2× ускорение на GPU, 50% экономия памяти | ⚠️ Возможна потеря точности |
| **INT8 (dynamic)** | `torch.qint8` | ⚡ 3–4× ускорение на CPU, 75% экономия | ⚠️ Не все операции поддерживаются |
| **INT8 (static)** | `torch.qint8` | ⚡ Максимальное ускорение, лучшая точность | 🔹 Требует калибровки, сложная настройка |

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
# Обязательные
pip install numpy torch>=2.0.0

# Для визуализации (опционально)
pip install matplotlib seaborn
```

### 2. Базовое использование
```python
from segmenters.TorchSegmenter import TorchSegmenter
from optimization_study.03_quantization_study import QuantizedSegmenter

# Инициализация
segmenter = TorchSegmenter(method="sobel_edge")
quantizer = QuantizedSegmenter(segmenter)

# Квантование (динамическое, без калибровки)
quantized = quantizer.quantize_method("sobel_edge", scheme="int8_dynamic")

# Бенчмарк
results = quantizer.benchmark_scheme(
    "sobel_edge", 
    scheme="int8_dynamic",
    image=test_image
)

print(f"⚡ Speedup: {results['speedup']:.2f}×")
print(f"🎯 Agreement: {results['pixel_agreement']*100:.2f}%")
```

### 3. CLI-запуск
```bash
# Базовый запуск
python benchmark.py \
  --image ./test.jpg \
  --methods "sobel_edge,otsu_thresholding" \
  --schemes "fp32,fp16,int8_dynamic" \
  --device cpu \
  --output ./results/

# Со статическим квантованием (требует калибровки)
python benchmark.py \
  --image ./test.jpg \
  --calibration-dir ./calibration_images/ \
  --schemes "fp32,int8_static" \
  --calibration-steps 100 \
  --plot
```

## 📊 Интерпретация результатов

### Метрики скорости
```python
{
  "original_mean_ms": 12.45,    # Среднее время оригинала (мс)
  "quantized_mean_ms": 3.21,    # Среднее время квантованной версии (мс)
  "speedup": 3.88,              # Коэффициент ускорения
}
```

### Метрики точности
```python
{
  "pixel_agreement": 0.9985,    # Доля совпадающих пикселей
  "mse": 1.2e-5,                # Среднеквадратичная ошибка
  "max_diff": 0.003,            # Максимальное отклонение
  "relative_error": 0.0012,     # Относительная ошибка
}
```

### Критерии оценки
| Показатель | Отлично | Приемлемо | Критично |
|-----------|---------|-----------|----------|
| **Speedup** | >3.0× | 1.5–3.0× | <1.5× |
| **Agreement** | >99.5% | 95–99.5% | <95% |
| **Size reduction** | >70% | 40–70% | <40% |

## ⚠️ Важные замечания

### 1. Квантование лучше работает на CPU
> ⚠️ **Важно**: Статическое INT8-квантование имеет ограниченную поддержку на GPU.
> 
> - Для **CPU**: все схемы работают стабильно, максимальное ускорение
> - Для **CUDA**: FP16 работает хорошо, INT8 — экспериментально
> 
> Рекомендуется: `--device cpu` для исследования квантования.

### 2. Не все методы поддерживают квантование
```python
# Методы, которые обычно НЕ квантуются:
non_quantizable = [
    "active_contour",    # Итеративные вычисления
    "chan_vese",         # Энергетическая оптимизация
    "random_walker",     # Решение СЛАУ
    "phase_congruency",  # FFT и сложные операции
]

# Методы, которые хорошо квантуются:
quantizable = [
    "global_thresholding",  # Простые сравнения
    "sobel_edge",           # Линейные свёртки
    "otsu_thresholding",    # Гистограммы
]
```

### 3. Калибровка критична для static INT8
```python
# Статическое квантование требует репрезентативных данных:
quantizer.calibrate(
    "sobel_edge",
    calibration_data=calibration_images,  # 50–200 изображений
    scheme="int8_static",
    num_steps=100
)
```

## 🔧 Настройка конфигурации

```python
from optimization_study.03_quantization_study import QuantizationConfig

config = QuantizationConfig(
    schemes=["fp32", "fp16", "int8_dynamic"],  # Схемы для тестирования
    n_runs=100,              # Количество запусков
    calibration_steps=100,   # Шагов калибровки
    per_channel=False,       # Per-tensor vs per-channel
    fuse_modules=True,       # Слияние conv+bn+relu
    target_device="cpu",     # CPU для лучшего INT8
    tolerance=1e-3,          # Допустимая погрешность
)

quantizer = QuantizedSegmenter(segmenter, config=config)
```

## 📁 Структура модуля

```
03_quantization_study/
├── __init__.py              # Публичный API
├── README.md                # Этот файл
├── config.py                # Конфигурация
├── utils.py                 # Вспомогательные функции
├── quantizer.py             # Основной класс квантования
├── benchmark.py             # CLI скрипт бенчмарка
├── report_generator.py      # Генерация отчётов
├── visualization.py         # Визуализация
├── requirements.txt         # Зависимости
└── run_example.sh           # Пример запуска
```

## 🔗 Ссылки

- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)
- [Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [FBGEMM Backend](https://github.com/pytorch/FBGEMM)
- [QNNPACK Backend](https://github.com/pytorch/QNNPACK)

---
*Модуль разработан для исследования оптимизации классических методов сегментации. 
Результаты могут варьироваться в зависимости от аппаратного обеспечения и версий библиотек.*
```