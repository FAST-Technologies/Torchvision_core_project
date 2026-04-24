# 🔧 06_compilation_study — Исследование компиляционной оптимизации
🔧 Исследование 6: Компиляция и оптимизация графа (Compilation Study)

Модуль для ускорения классических методов сегментации через
компиляцию графа вычислений в PyTorch.

## 🎯 Цель исследования

Уменьшение времени инференса за счёт:
- **torch.jit** — статическая компиляция через AST-анализ или трассировку
- **torch.compile()** — динамическая компиляция с авто-оптимизацией (PyTorch 2.0+)
- **Graph freezing** — фиксация графа для продакшен-инференса
- **Backend selection** — выбор оптимального бэкенда (inductor, cudagraphs)

## 📦 Стратегии компиляции

| Стратегия | Описание | Преимущества | Ограничения |
|-----------|----------|--------------|-------------|
| **JIT_SCRIPT** | `torch.jit.script` | ✅ Поддержка динамического контроля потока | ⚠️ Требует совместимого с JIT кода |
| **JIT_TRACE** | `torch.jit.trace` | ✅ Работает с любым кодом | ⚠️ Не поддерживает динамический контроль потока |
| **TORCH_COMPILE** | `torch.compile()` | 🚀 Автоматическая оптимизация, лучший выигрыш | 🔹 Требует PyTorch >= 2.0 |
| **GRAPH_FREEZE** | `freeze + optimize_for_inference` | ✅ Финальная оптимизация для продакшена | 🔹 Только для инференса, не для обучения |

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
# Обязательные
pip install torch>=2.0.0 numpy

# Опционально: для визуализации
pip install matplotlib seaborn
```

### 2. Базовое использование
```python
from segmenters.TorchSegmenter import TorchSegmenter
from optimization_study.06_compilation_study import GraphOptimizer

# Инициализация
segmenter = TorchSegmenter(method="sobel_edge")
optimizer = GraphOptimizer(segmenter)

# Оптимизация одного метода
report = optimizer.optimize_method("sobel_edge")

print(f"⚡ Speedup: {report.speedup_formatted}")
print(f"🔧 Compile time: {report.compile_time_ms:.1f} ms")
```

### 3. CLI-запуск
```bash
# Базовый запуск с torch.compile
python benchmark.py \
  --methods "sobel_edge,otsu_thresholding" \
  --strategy torch_compile \
  --device cuda \
  --output ./results/

# Сравнение стратегий
python benchmark.py \
  --methods "sobel_edge" \
  --compare-strategies \
  --plot \
  --verbose
```

## 📊 Интерпретация результатов

### Метрики производительности
```python
{
  "original_time_ms": 12.45,    # Среднее время оригинала (мс)
  "compiled_time_ms": 8.21,     # Среднее время после компиляции (мс)
  "compile_time_ms": 150.3,     # Время самой компиляции (мс)
  "speedup": 1.52,              # Коэффициент ускорения
  "strategy": "torch_compile",  # Применённая стратегия
}
```

### Критерии оценки
| Показатель | Отлично | Приемлемо | Критично |
|-----------|---------|-----------|----------|
| **Speedup** | >2.0× | 1.2–2.0× | <1.2× |
| **Compile time** | <100 ms | 100–500 ms | >500 ms |
| **Graph stability** | Stable | — | Dynamic |

## ⚠️ Важные замечания

### 1. torch.compile требует "прогрева"
```python
# Первый запуск включает компиляцию графа
compiled = torch.compile(original_func)
_ = compiled(example_input)  # "Прогрев" — компиляция происходит здесь
# Теперь можно замерять производительность
```

### 2. Не все методы компилируются одинаково хорошо
```python
# Хорошо компилируются:
good = [
    "global_thresholding",  # Простые операции
    "sobel_edge",           # Линейные свёртки
    "otsu_thresholding",    # Векторизованные вычисления
]

# Могут не компилироваться:
problematic = [
    "canny_edge",           # Сложный контроль потока
    "active_contour",       # Итеративные вычисления с условиями
    "random_walker",        # Решение СЛАУ
]
```

### 3. Выбор режима компиляции
```python
# reduce-overhead: для коротких операций (<10 мс)
#   Минимизирует накладные расходы на запуск kernel
compiled = torch.compile(func, mode="reduce-overhead")

# max-autotune: для долгих операций (>50 мс)
#   Тратит больше времени на компиляцию для лучшего инференса
compiled = torch.compile(func, mode="max-autotune")

# default: баланс между временем компиляции и инференса
compiled = torch.compile(func, mode="default")
```

## 🔧 Настройка конфигурации

```python
from optimization_study.06_compilation_study import CompilationConfig, CompilationStrategy

config = CompilationConfig(
    strategy=CompilationStrategy.TORCH_COMPILE,  # Стратегия
    compile_mode="reduce-overhead",              # Режим torch.compile
    backend="inductor",                          # Бэкенд
    fullgraph=True,                              # Компилировать весь граф
    dynamic=False,                               # Статические shapes
    freeze_graph=True,                           # Заморозить для инференса
    n_runs=50,                                   # Запусков для замера
)

optimizer = GraphOptimizer(segmenter, config=config)
```

## 📁 Структура модуля

```
06_compilation_study/
├── __init__.py              # Публичный API
├── README.md                # Этот файл
├── config.py                # Конфигурация
├── utils.py                 # Вспомогательные функции
├── jit_compilation.py       # torch.jit оптимизации
├── torch_compile_benchmark.py  # torch.compile() бенчмаркинг
├── graph_optimizer.py       # Комплексный оптимизатор графа
├── benchmark.py             # CLI скрипт бенчмарка
├── report_generator.py      # Генерация отчётов
├── visualization.py         # Визуализация
├── requirements.txt         # Зависимости
└── run_example.sh           # Пример запуска
```

## 🔗 Ссылки

- [torch.compile Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [JIT Documentation](https://pytorch.org/docs/stable/jit.html)
- [Inductor Backend Guide](https://pytorch.org/docs/stable/torch.compiler_inductor.html)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)

---
*Модуль разработан для исследования оптимизации классических методов сегментации. 
Результаты могут варьироваться в зависимости от аппаратного обеспечения и версий библиотек.*
```