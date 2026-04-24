# 🔬 01_precision_benchmark — Исследование влияния точности вычислений
📊 Исследование 1: Влияние точности вычислений (Precision Study)

Модуль для оценки влияния типов данных (точности вычислений) на:
- ⚡ **Скорость** выполнения классических методов сегментации
- 🎯 **Точность** результатов относительно эталона (FP32)

## 🎯 Цель исследования

Определить оптимальный баланс между производительностью и точностью
при использовании различных прецизионных режимов:

| Режим | Тип данных | Преимущества | Ограничения |
|-------|-----------|--------------|-------------|
| **FP32** | `torch.float32` | ✅ Эталонная точность, полная совместимость | — |
| **FP16** | `torch.float16` | ⚡ Ускорение на современных GPU (2–4×) | ⚠️ Возможна потеря точности |
| **BF16** | `torch.bfloat16` | ⚡ Баланс скорости/точности, стабильнее FP16 | 🔹 Требует Ampere+ GPU |
| **INT8** | `torch.int8` | 🚀 Максимальное ускорение (4–8×) | ⚠️ Требует квантования, возможна значительная потеря точности |

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install pandas matplotlib seaborn
```

### 2. Базовое использование
```python
from segmenters.TorchSegmenter import TorchSegmenter
from optimization_study.01_precision_benchmark import PrecisionBenchmark

# Инициализация
segmenter = TorchSegmenter(method="sobel_edge")
image = load_image("test.jpg")  # Ваша функция загрузки

benchmark = PrecisionBenchmark(segmenter, image)

# Замер для одного метода
result = benchmark.benchmark_method("sobel_edge", "fp16")
print(f"FP16: {result['median_ms']:.3f} ms")

# Полный бенчмарк
df = benchmark.run_full_benchmark(
    methods=["sobel_edge", "otsu_thresholding", "canny_edge"],
    precisions=["fp32", "fp16", "bf16"]
)

# Визуализация
from optimization_study.01_precision_benchmark import PrecisionVisualizer
viz = PrecisionVisualizer()
viz.plot_all(df)
```

### 3. CLI-запуск (опционально)
```bash
# Запуск бенчмарка для всех методов
python -m optimization_study.01_precision_benchmark \
  --image ./test.jpg \
  --methods "sobel_edge,otsu_thresholding" \
  --precisions "fp32,fp16" \
  --output ./results/
```

## 📊 Интерпретация результатов

### Метрики времени
```python
{
  "mean_ms": 12.45,      # Среднее время (мс)
  "median_ms": 12.30,    # Медиана (рекомендуется использовать)
  "std_ms": 0.35,        # Стандартное отклонение
  "p95_ms": 13.10,       # 95-й перцентиль (для оценки "худшего случая")
}
```

### Метрики точности (относительно FP32)
```python
{
  "pixel_agreement": 0.9995,  # Доля совпадающих пикселей
  "mse": 1.2e-6,              # Среднеквадратичная ошибка
  "max_diff": 0.003,          # Максимальное отклонение
}
```

### Критерии оценки
| Показатель | Отлично | Приемлемо | Критично |
|-----------|---------|-----------|----------|
| **Speedup** | >2.0× | 1.2–2.0× | <1.2× |
| **Agreement** | >99.9% | 99.0–99.9% | <99.0% |

## ⚠️ Важные замечания

### 1. Классические методы ≠ нейросети
> ⚠️ **Важно**: Классические алгоритмы сегментации (Sobel, Otsu, Canny) 
> не используют обучение и не оптимизированы под низкую точность.
> 
> - **FP16/BF16** могут дать ускорение только на GPU за счёт более быстрой арифметики
> - **INT8** требует квантования и может значительно исказить результаты
> - Для простых операций накладные расходы конвертации могут превысить выигрыш

### 2. Поддержка устройств
```python
# Проверка доступных режимов
from optimization_study.01_precision_benchmark import get_available_dtypes

print(get_available_dtypes("cuda"))  # ['fp32', 'fp16', 'bf16']
print(get_available_dtypes("cpu"))   # ['fp32', 'fp16'] (BF16 медленный на CPU)
```

### 3. autocast и смешанная точность
- Используется `torch.autocast` для автоматического приведения типов
- На **CPU** autocast может не давать выигрыша (отключается по умолчанию)
- Для **INT8** используется отдельная логика квантования

## 🔧 Настройка конфигурации

```python
from optimization_study.01_precision_benchmark import PrecisionConfig

config = PrecisionConfig(
    precisions=["fp32", "fp16"],      # Тестируемые режимы
    n_runs=100,                        # Количество запусков
    warmup_runs=20,                    # Прогрев
    enable_autocast=True,              # Использовать autocast
    autocast_cpu_enabled=False,        # Отключить autocast на CPU
    tolerance=1e-4,                    # Допустимая погрешность
    output_dir="./my_results/",        # Папка для сохранения
)

benchmark = PrecisionBenchmark(segmenter, image, config=config)
```

## 📁 Структура модуля

```
01_precision_benchmark/
├── __init__.py              # Публичный API
├── README.md                # Этот файл
├── config.py                # Конфигурация
├── utils.py                 # Вспомогательные функции
├── precision_benchmark.py   # Основной класс
├── report_generator.py      # Генерация отчётов
├── visualization.py         # Визуализация
├── requirements.txt         # Зависимости
└── run_example.sh           # Пример запуска
```

## 🔗 Ссылки

- [PyTorch Autocast Docs](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Mixed Precision Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [Brain Float 16 (BF16) Overview](https://cloud.google.com/tpu/docs/bfloat16)

---
*Модуль разработан для исследования оптимизации классических методов сегментации. 
Результаты могут варьироваться в зависимости от аппаратного обеспечения и версий библиотек.*
```

---