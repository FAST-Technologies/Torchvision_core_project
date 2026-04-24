# 🧠 05_memory_optimization — Исследование оптимизации памяти
🧠 Исследование 5: Оптимизация памяти (Memory Optimization)

Модуль для профилирования и оптимизации использования памяти
в классических методах сегментации на базе PyTorch.

## 🎯 Цель исследования

Снижение пикового потребления памяти и предотвращение утечек через:
- **Кэширование** повторяемых тензоров (ядра свёртки, буферы)
- **Пуллы памяти** для фиксированных размеров
- **Inplace-оптимизации** для снижения аллокаций
- **Детектирование утечек** в реальном времени
- **Анализ паттернов** аллокаций

## 📦 Политики управления памятью

| Политика | Описание | Преимущества | Когда использовать |
|----------|----------|--------------|-------------------|
| **LAZY** | Ленивая аллокация | Мин. накладные расходы | Простые методы, редкие вызовы |
| **POOLED** | Пул предвыделенных буферов | Снижение фрагментации | Методы с фиксированными размерами |
| **REUSE** | Повторное использование тензоров | Мин. аллокаций | Итеративные алгоритмы |
| **AGGRESSIVE_GC** | Агрессивный сборщик мусора | Быстрое освобождение | Долгие методы с временными тензорами |
| **PINNED** | Закреплённая память | Быстрый CPU↔GPU | Частая передача данных |

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
# Обязательные
pip install torch>=2.0.0 numpy

# Для визуализации (опционально)
pip install matplotlib seaborn
```

### 2. Базовое использование
```python
from segmenters.TorchSegmenter import TorchSegmenter
from optimization_study.05_memory_optimization import MemoryOptimizer

# Инициализация
segmenter = TorchSegmenter(method="sauvola_thresholding")
optimizer = MemoryOptimizer(segmenter)

# Оптимизация одного метода
report = optimizer.optimize_method("sauvola_thresholding", image_tensor)

print(f"💾 Saved: {report.memory_saved_mb:.2f} MB ({report.reduction_pct:.1f}%)")
print(f"⚡ Speedup: {report.speedup:.2f}×")
```

### 3. CLI-запуск
```bash
# Базовый запуск с pooled-политикой
python benchmark.py \
  --methods "sauvola_thresholding,niblack_thresholding" \
  --policy pooled \
  --device cuda \
  --output ./results/

# С детекцией утечек и визуализацией
python benchmark.py \
  --methods all \
  --policy aggressive_gc \
  --detect-leaks \
  --plot \
  --verbose
```

## 📊 Интерпретация результатов

### Метрики памяти
```python
{
  "baseline_allocated_mb": 45.2,    # Потребление до оптимизации
  "optimized_peak_mb": 32.1,        # Пик после оптимизации
  "memory_saved_mb": 13.1,          # Экономия
  "reduction_pct": 28.9,            # Процент снижения
  "leak_detected": False,           # Есть ли утечка
}
```

### Критерии оценки
| Показатель | Отлично | Приемлемо | Критично |
|-----------|---------|-----------|----------|
| **Reduction** | >30% | 10–30% | <10% |
| **Leak detected** | False | — | True |
| **Cache hit rate** | >80% | 50–80% | <50% |

## ⚠️ Важные замечания

### 1. Inplace-оптимизации требуют осторожности
```python
# Inplace может сломать градиенты или изменить поведение:
# ❌ Небезопасно:
tensor.relu_(inplace=True)  # Может повлиять на другие ссылки

# ✅ Безопасно в inference-режиме:
with torch.inference_mode():
    tensor.relu_(inplace=True)
```

### 2. Кэширование экономит память, но требует управления
```python
# TTL предотвращает "раздувание" кэша:
cache = KernelCache(max_size=50, ttl_seconds=300)

# Статистика помогает настроить параметры:
stats = cache.stats()
print(f"Hit rate: {stats['hit_rate_pct']:.1f}%")
```

### 3. Детектор утечек — эвристический
```python
# Утечка = рост потребления > порог
# Настройте пороги под вашу задачу:
config = MemoryConfig(
    leak_threshold_mb=50.0,      # Абсолютный порог
    relative_threshold=0.1,      # 10% от baseline
)
```

## 🔧 Настройка конфигурации

```python
from optimization_study.05_memory_optimization import MemoryConfig, MemoryPolicy

config = MemoryConfig(
    policy=MemoryPolicy.POOLED,        # Политика
    cache_max_size=100,                # Размер кэша
    cache_ttl_seconds=600.0,           # TTL кэша
    detect_leaks=True,                 # Детектор утечек
    leak_threshold_mb=25.0,            # Порог утечки
    clear_cache_between_runs=True,     # Очистка между запусками
    device="cuda",                     # Устройство
)

optimizer = MemoryOptimizer(segmenter, config=config)
```

## 📁 Структура модуля

```
05_memory_optimization/
├── __init__.py              # Публичный API
├── README.md                # Этот файл
├── config.py                # Конфигурация
├── utils.py                 # Вспомогательные функции
├── caching_strategy.py      # Кэширование ядер и пулы
├── memory_profiler.py       # Профилирование памяти
├── memory_optimizer.py      # Основной класс оптимизации
├── allocation_tracker.py    # Трекер аллокаций и утечек
├── benchmark.py             # CLI скрипт бенчмарка
├── report_generator.py      # Генерация отчётов
├── visualization.py         # Визуализация
├── requirements.txt         # Зависимости
└── run_example.sh           # Пример запуска
```

## 🔗 Ссылки

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [CUDA Memory Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory)
- [Python gc Module](https://docs.python.org/3/library/gc.html)

---
*Модуль разработан для исследования оптимизации классических методов сегментации. 
Результаты могут варьироваться в зависимости от аппаратного обеспечения и версий библиотек.*
```