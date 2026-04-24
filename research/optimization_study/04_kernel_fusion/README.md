# ⚡ 04_kernel_fusion — Исследование векторизации и Kernel Fusion
⚡ Исследование 4: Векторизация и Kernel Fusion

Модуль для оптимизации производительности классических методов сегментации
через объединение операций (kernel fusion) и векторизацию вычислений.

## 🎯 Цель исследования

Уменьшение накладных расходов за счёт:
- **Graph fusion**: Объединение операций на уровне графа вычислений (torch.compile)
- **Manual fusion**: Ручное объединение математических операций в один kernel
- **Custom kernels**: Написание специализированных CUDA kernels для критичных участков
- **Vectorization**: Использование SIMD/SIMT инструкций для параллельных вычислений

## 📦 Стратегии fusion

| Стратегия | Описание | Преимущества | Ограничения |
|-----------|----------|--------------|-------------|
| **GRAPH_FUSION** | torch.compile() / FX graph fusion | ✅ Автоматическая, безопасная | ⚠️ Требует PyTorch >= 2.0 |
| **MANUAL_FUSION** | Ручное объединение операций | ✅ Полный контроль, максимальная оптимизация | 🔹 Требует знаний архитектуры |
| **CUSTOM_KERNEL** | Custom CUDA kernel | 🚀 Максимальная производительность | ❌ Сложная разработка и отладка |
| **VECTORIZED** | Векторизованные операции | ⚡ Хороший баланс скорость/сложность | ⚠️ Эффективно только для больших тензоров |

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
# Обязательные
pip install torch>=2.0.0 numpy

# Опционально: для custom kernels
# pip install ninja  # Для компиляции CUDA расширений
```

### 2. Базовое использование
```python
from segmenters.TorchSegmenter import TorchSegmenter
from optimization_study.04_kernel_fusion import FusionOptimizer

# Инициализация
segmenter = TorchSegmenter(method="sauvola_thresholding")
optimizer = FusionOptimizer(segmenter)

# Применение fusion
fused_op = optimizer.fuse_method("sauvola_thresholding")

# Бенчмарк
result = optimizer.benchmark_fusion("sauvola_thresholding", fused_op)
print(f"⚡ Speedup: {result['speedup']:.2f}×")
```

### 3. CLI-запуск
```bash
# Базовый запуск с graph fusion
python benchmark.py \
  --methods "sauvola_thresholding,niblack_thresholding,sobel_edge" \
  --strategy graph \
  --device cuda \
  --output ./results/

# С профилированием графа
python benchmark.py \
  --methods all \
  --strategy manual \
  --profile-graph \
  --plot
```

## 📊 Интерпретация результатов

### Метрики производительности
```python
{
  "original_mean_ms": 12.45,    # Среднее время оригинала (мс)
  "fused_mean_ms": 8.21,        # Среднее время fused версии (мс)
  "speedup": 1.52,              # Коэффициент ускорения
  "strategy": "GRAPH_FUSION",   # Применённая стратегия
}
```

### Критерии оценки
| Показатель | Отлично | Приемлемо | Критично |
|-----------|---------|-----------|----------|
| **Speedup** | >2.0× | 1.2–2.0× | <1.2× |
| **Fusion potential** | >0.7 | 0.4–0.7 | <0.4 |
| **Memory reduction** | >30% | 10–30% | <10% |

## ⚠️ Важные замечания

### 1. Fusion не всегда даёт выигрыш
> ⚠️ **Важно**: Накладные расходы на fusion могут превысить выигрыш для:
> - Маленьких изображений (<256×256)
> - Простых операций с временем <1 мс
> - Методов со сложным контролем потока (Canny, Watershed)
> 
> Всегда проводите бенчмаркинг на целевых данных!

### 2. torch.compile требует "прогрева"
```python
# Первые вызовы включают компиляцию графа
fused_func = torch.compile(original_func)
_ = fused_func(example_input)  # "Прогрев"
# Теперь можно замерять производительность
```

### 3. Custom kernels — только для экспертов
```python
# Требует:
# - Установленного CUDA toolkit
# - Знаний CUDA C++
# - Тщательного тестирования

if config.enable_custom_kernels:
    from .custom_kernels import get_custom_kernel
    fused_func = get_custom_kernel("sauvola", original_func)
```

## 🔧 Настройка конфигурации

```python
from optimization_study.04_kernel_fusion import FusionConfig, FusionStrategy

config = FusionConfig(
    strategy=FusionStrategy.MANUAL_FUSION,  # Стратегия fusion
    compile_mode="max-autotune",            # Режим torch.compile
    enable_custom_kernels=False,            # Включить custom kernels
    profile_graph=True,                     # Профилировать граф
    min_ops_for_fusion=3,                   # Мин. операций для fusion
    min_expected_speedup=1.1,               # Мин. ожидаемый выигрыш
)

optimizer = FusionOptimizer(segmenter, config=config)
```

## 📁 Структура модуля

```
04_kernel_fusion/
├── __init__.py              # Публичный API
├── README.md                # Этот файл
├── config.py                # Конфигурация
├── utils.py                 # Вспомогательные функции
├── fusion_optimizer.py      # Основной класс оптимизации
├── custom_kernels.py        # Custom CUDA kernels (опционально)
├── graph_profiler.py        # Профилирование графа
├── benchmark.py             # CLI скрипт бенчмарка
├── report_generator.py      # Генерация отчётов
├── visualization.py         # Визуализация
├── requirements.txt         # Зависимости
└── run_example.sh           # Пример запуска
```

## 🔗 Ссылки

- [torch.compile Documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [FX Graph Mode Quantization](https://pytorch.org/docs/stable/fx.html)
- [CUDA Custom Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Kernel Fusion Best Practices](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/)

---
*Модуль разработан для исследования оптимизации классических методов сегментации. 
Результаты могут варьироваться в зависимости от аппаратного обеспечения и версий библиотек.*
```
