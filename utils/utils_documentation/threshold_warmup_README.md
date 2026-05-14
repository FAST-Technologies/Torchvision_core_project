# 🔥 ThresholdWarmUp — Специализированный warm-up для пороговых и граничных методов

## 📖 Описание
Модуль `utils/threshold_warmup.py` предоставляет **целевой инструмент прогрева** для классических методов сегментации: пороговых (thresholding) и детекторов границ (edge detection).

> ⚠️ **Важно:** Данный модуль оптимизирован для **классических алгоритмов** на базе гистограмм и свёрток. Для нейросетевых методов используйте `utils.warmup.SegmentationWarmUp`.

## ✨ Ключевые возможности
### 🎯 Специализация по типам методов

| Категория | Ключевые слова | Примеры методов | Особенности прогрева |
|-----------|---------------|----------------|---------------------|
| **Пороговые** (5) | `global_threshold`, `otsu`, `adaptive_threshold`, `niblack`, `sauvola` | Прогрев кэшей гистограмм, свёрточных ядер | Тестирование на разных размерах изображений |
| **Граничные** (4) | `sobel`, `canny`, `laplacian`, `prewitt` | Прогрев градиентных операторов, свёрток | Тестирование на разных паттернах границ |

### 📐 Тестирование на разных размерах изображений
```python
# Конфигурируемые размеры для масштабного анализа
image_sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]

# Автоматический расчёт статистики для каждого размера:
# - mean_ms: среднее время выполнения
# - std_ms: стандартное отклонение (стабильность)
# - n_runs: количество успешных прогонов
```

### 🎨 Тестовые паттерны для граничных методов
| Паттерн | Описание | Сценарий использования |
|---------|----------|----------------------|
| `horizontal` | Горизонтальная белая полоса 10px по центру | Тестирование вертикальных градиентов |
| `vertical` | Вертикальная белая полоса 10px по центру | Тестирование горизонтальных градиентов |
| `diagonal` | Диагональная линия из единичных пикселей | Тестирование угловых градиентов, связности |
| `noise` | Случайный цветной шум (H×W×3) | Устойчивость к шуму, ложные срабатывания |
| `*` (default) | Шахматная доска (черно-белая) | Комплексное тестирование всех направлений |

### 📊 Типизированные результаты через TypedDict
```python
from utils.threshold_warmup import WarmupMetrics, SizeResults, PatternResults

# Метрики для одного теста
metrics: WarmupMetrics = {
    "mean_ms": 12.34,   # Среднее время (мс)
    "std_ms": 1.23,     # Стандартное отклонение (мс)
    "n_runs": 3         # Количество прогонов
}

# Результаты по размерам
size_results: SizeResults = {
    "sizes": {
        "(256, 256)": metrics,
        "(512, 512)": {...}
    }
}

# Результаты по паттернам
pattern_results: PatternResults = {
    "patterns": {
        "horizontal": metrics,
        "vertical": {...}
    }
}
```

## 🚀 Быстрый старт
### Прогрев пороговых методов на разных размерах
```python
from utils.threshold_warmup import ThresholdWarmUp
from segmenters.SklearnSegmenter import SklearnSegmenter

# Подготовка сегментеров
segmenters = {
    "otsu": SklearnSegmenter("otsu_thresholding"),
    "adaptive": SklearnSegmenter("adaptive_thresholding", block_size=11),
    "sauvola": SklearnSegmenter("threshold_sauvola", window_size=15),
}

# Прогрев и сбор статистики
results = ThresholdWarmUp.warmup_threshold_methods(
    segmenters_dict=segmenters,
    image_sizes=[(256, 256), (512, 512), (1024, 1024)],
    n_runs_per_size=3
)

# Анализ результатов
for method, data in results.items():
    print(f"\n{method}:")
    for size_str, metrics in data["sizes"].items():
        print(f"  {size_str}: {metrics['mean_ms']:.2f}ms ± {metrics['std_ms']:.2f}ms")
```

### Прогрев детекторов границ на разных паттернах
```python
from utils.threshold_warmup import ThresholdWarmUp
from segmenters.OpenCVSegmenter import OpenCVSegmenter

# Подготовка граничных методов
edge_segmenters = {
    "sobel": OpenCVSegmenter("sobel_edge"),
    "canny": OpenCVSegmenter("canny_edge", low=0.1, high=0.3),
    "laplacian": OpenCVSegmenter("log_edge"),
}

# Прогрев на паттернах
results = ThresholdWarmUp.warmup_edge_methods(
    segmenters_dict=edge_segmenters,
    edge_patterns=["horizontal", "vertical", "diagonal", "noise"],
    n_runs_per_pattern=3
)

# Анализ устойчивости к паттернам
for method, data in results.items():
    print(f"\n{method}:")
    for pattern, metrics in data["patterns"].items():
        stability = "✅" if metrics["std_ms"] < 2.0 else "⚠️"
        print(f"  {stability} {pattern:12s}: {metrics['mean_ms']:.2f}ms ± {metrics['std_ms']:.2f}ms")
```

### Комбинированный анализ: размер × паттерн
```python
# Для комплексного бенчмарка можно комбинировать оба метода:
threshold_results = ThresholdWarmUp.warmup_threshold_methods(
    segmenters_dict={"otsu": otsu_segmenter},
    image_sizes=[(256, 256), (512, 512)]
)

edge_results = ThresholdWarmUp.warmup_edge_methods(
    segmenters_dict={"canny": canny_segmenter},
    edge_patterns=["horizontal", "noise"]
)

# Объединение результатов для отчёта
combined = {
    "threshold": threshold_results,
    "edge": edge_results
}
```

### Генерация тестовых паттернов вручную
```python
from utils.threshold_warmup import ThresholdWarmUp

# Создание паттерна для отладки
horizontal = ThresholdWarmUp._create_edge_pattern(256, 256, "horizontal")
vertical = ThresholdWarmUp._create_edge_pattern(256, 256, "vertical")
noise = ThresholdWarmUp._create_edge_pattern(256, 256, "noise")  # RGB!

# Сохранение для визуальной проверки
from PIL import Image
Image.fromarray(horizontal).save("test_horizontal.png")
Image.fromarray(noise).save("test_noise.png")
```

## ⚙️ Конфигурация
### Параметры `warmup_threshold_methods()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `segmenters_dict` | `Dict[str, SegmenterLike]` | — | Словарь `{имя: экземпляр}` пороговых сегментеров |
| `image_sizes` | `List[Tuple[int, int]]` | `[(128,128), (256,256), (512,512)]` | Размеры изображений для тестирования |
| `n_runs_per_size` | `int` | `2` | Количество прогонов на каждый размер |

### Параметры `warmup_edge_methods()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `segmenters_dict` | `Dict[str, SegmenterLike]` | — | Словарь `{имя: экземпляр}` граничных сегментеров |
| `edge_patterns` | `List[str]` | `["horizontal", "vertical", "diagonal", "noise"]` | Паттерны для генерации тестовых изображений |
| `n_runs_per_pattern` | `int` | `3` | Количество прогонов на каждый паттерн |

### Параметры `_create_edge_pattern()`
| Параметр | Тип | Описание |
|----------|-----|----------|
| `h` | `int` | Высота изображения |
| `w` | `int` | Ширина изображения |
| `pattern` | `str` | Имя паттерна: `"horizontal"`, `"vertical"`, `"diagonal"`, `"noise"`, или другое (шахматная доска) |

### Возвращаемые типы (TypedDict)
```python
class WarmupMetrics(TypedDict):
    mean_ms: float   # Среднее время выполнения (мс)
    std_ms: float    # Стандартное отклонение (мс)
    n_runs: int      # Количество прогонов

class SizeResults(TypedDict):
    sizes: Dict[str, WarmupMetrics]  # {размер: метрики}

class PatternResults(TypedDict):
    patterns: Dict[str, WarmupMetrics]  # {паттерн: метрики}
```

## 📚 Справочник методов
### 🔹 Основные методы прогрева
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `warmup_threshold_methods()` | `segmenters_dict`, `image_sizes`, `n_runs_per_size` | Прогрев пороговых методов на разных размерах | `Dict[str, SizeResults]` |
| `warmup_edge_methods()` | `segmenters_dict`, `edge_patterns`, `n_runs_per_pattern` | Прогрев граничных методов на разных паттернах | `Dict[str, PatternResults]` |

### 🔹 Вспомогательные методы
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `_create_edge_pattern()` | `h`, `w`, `pattern` | Генерация тестового изображения с указанным паттерном | `np.ndarray`: `(H,W)` или `(H,W,3)` |

### 🔹 Ключевые слова для фильтрации методов
```python
# Пороговые методы (фильтр по имени)
threshold_methods = [
    "global_threshold", "otsu", "adaptive_threshold", 
    "niblack", "sauvola"
]

# Граничные методы (фильтр по имени)
edge_methods = [
    "sobel", "canny", "laplacian", "prewitt"
]
```

## 🔄 Конвейер прогрева: генерация → запуск → статистика
### Логика `warmup_threshold_methods()`
```python
def warmup_threshold_methods(segmenters_dict, image_sizes, n_runs_per_size):
    results = {}
    
    for name, segmenter in segmenters_dict.items():
        # Фильтрация по ключевым словам
        if not any(tm in name.lower() for tm in threshold_methods):
            continue
            
        method_results = {"sizes": {}}
        
        for size in image_sizes:
            # Генерация тестового изображения
            img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            
            # Замеры времени
            times = []
            for _ in range(n_runs_per_size):
                start = time.perf_counter()
                try:
                    segmenter.segment(img)
                    times.append(time.perf_counter() - start)
                except Exception:
                    times.append(float("inf"))  # Не прерываем цикл
            
            # Расчёт статистики
            method_results["sizes"][str(size)] = {
                "mean_ms": np.mean(times) * 1000,
                "std_ms": np.std(times) * 1000,
                "n_runs": len(times)
            }
        
        results[name] = method_results
    
    return results
```

### Генерация паттернов для граничных методов
```python
def _create_edge_pattern(h, w, pattern):
    if pattern == "horizontal":
        # Горизонтальная полоса 10px
        img[(h//2-5):(h//2+5), :] = 255
    elif pattern == "vertical":
        # Вертикальная полоса 10px
        img[:, (w//2-5):(w//2+5)] = 255
    elif pattern == "diagonal":
        # Диагональ из единичных пикселей
        for i in range(min(h,w)-1):
            img[i,i] = img[i,i+1] = img[i+1,i] = 255
    elif pattern == "noise":
        # Цветной шум для тестирования устойчивости
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        # Шахматная доска по умолчанию
        img[::2,::2] = img[1::2,1::2] = 255
    return img
```

## 📊 Интерпретация результатов
### Анализ стабильности по стандартному отклонению
```python
# Стабильный метод (низкий std)
if metrics["std_ms"] < 1.0:
    print("✅ Стабильное время выполнения")
# Умеренная вариативность
elif metrics["std_ms"] < 5.0:
    print("⚠️ Умеренная вариативность — проверить сборку мусора")
# Высокая нестабильность
else:
    print("❌ Высокая нестабильность — возможен первый запуск (JIT/CUDA)")
```

### Масштабируемость по размеру изображения
```python
# Оценка сложности алгоритма
sizes = [256, 512, 1024]
times = [results["otsu"]["sizes"][f"({s}, {s})"]["mean_ms"] for s in sizes]

# Линейная сложность: время растёт пропорционально N пикселей
# O(N): time[512]/time[256] ≈ 4, time[1024]/time[512] ≈ 4
ratio_512_256 = times[1] / times[0]
ratio_1024_512 = times[2] / times[1]

print(f"Масштабируемость: 512/256={ratio_512_256:.2f}×, 1024/512={ratio_1024_512:.2f}×")
```

### Устойчивость к паттернам для граничных методов
```python
# Сравнение времени на разных паттернах
pattern_times = {p: m["mean_ms"] for p, m in results["canny"]["patterns"].items()}

# Ожидаемо: шум > диагональ > горизонталь/вертикаль
sorted_patterns = sorted(pattern_times.items(), key=lambda x: x[1])
print("Устойчивость (быстрее → медленнее):")
for pattern, time_ms in sorted_patterns:
    print(f"  {pattern:12s}: {time_ms:.2f}ms")
```

## ⚡ Производительность и рекомендации
### Ожидаемое время прогрева (на изображении 512×512, 3 прогона)
```
✅ Быстро (<5 мс/прогон):
   - global_thresholding, otsu_thresholding
   - sobel_edge, prewitt_edge

⚠️ Средне (5–20 мс/прогон):
   - adaptive_thresholding, niblack, sauvola
   - canny_edge, log_edge

❌ Медленно (20–100+ мс/прогон):
   - Методы с адаптивным окном >50×50
   - Канни с высоким sigma (>2.0)
```

### Рекомендации по настройке
1. **Начинайте с малого**: `n_runs_per_size=2`, `image_sizes=[(256,256)]` для быстрой проверки.
2. **Используйте репрезентативные размеры**: включайте размер, близкий к целевому в продакшене.
3. **Для граничных методов**: тестируйте на `noise` паттерне для оценки устойчивости к ложным срабатываниям.
4. **Фильтруйте выбросы**: если `std_ms/mean_ms > 0.5`, увеличьте `n_runs` для более надёжной оценки.
5. **Логируйте `execution_info`**: доступ к `segmenter.params["execution_info"]` помогает отладке.

## 🛠️ Обработка ошибок и устойчивость
### Пропуск методов по ключевым словам
```python
# Автоматическая фильтрация: обрабатываются только релевантные методы
is_threshold = any(tm in name.lower() for tm in threshold_methods)
if not is_threshold:
    continue  # Пропускаем нерелевантные методы
```

### Устойчивость к сбоям в прогонах
```python
try:
    segmenter.segment(img)
    times.append(time.perf_counter() - start)
except Exception:
    times.append(float("inf"))  # Записываем сбой, но продолжаем цикл
    # Остальные прогоны выполнятся нормально
```

### Доступ к метаданным выполнения
```python
# После прогрева можно получить информацию о последнем выполнении
# (если сегментер сохраняет её в params)
print(segmenter.params["execution_info"])
# Пример вывода:
# {'method': 'otsu_thresholding', 'parameters': {}, 'execution_time': 0.0123}
```

### Рекомендации по отладке
1. **Включите подробное логирование**:
   ```python
   import logging
   logging.getLogger("utils.threshold_warmup").setLevel(logging.DEBUG)
   ```

2. **Проверьте наличие метода `.segment()`**:
   ```python
   assert hasattr(segmenter, "segment"), "Сегментер должен иметь метод .segment()"
   ```

3. **Тестируйте на минимальном изображении**:
   ```python
   # Быстрая проверка конвейера
   results = ThresholdWarmUp.warmup_threshold_methods(
       segmenters_dict={"test": test_segmenter},
       image_sizes=[(64, 64)],
       n_runs_per_size=1
   )
   ```

4. **Анализируйте соотношение std/mean**:
   ```python
   for method, data in results.items():
       for size_str, metrics in data["sizes"].items():
           cv = metrics["std_ms"] / metrics["mean_ms"] if metrics["mean_ms"] > 0 else 0
           if cv > 0.5:
               print(f"⚠️ {method} @ {size_str}: high CV={cv:.2f}")
   ```

## 🤝 Зависимости
```text
numpy>=1.20          # Массивы, статистики, генерация паттернов
typing-extensions    # TypedDict для типизации результатов
```

### Опциональные зависимости для визуализации
```bash
# Для сохранения тестовых паттернов
pip install Pillow

# Для построения графиков масштабируемости
pip install matplotlib
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование ThresholdWarmUp |
|--------|------------------------------|
| `BatchClassicTester` | Предварительный прогрев пороговых методов перед пакетным тестом |
| `BatchClassicTester2` | Прогрев граничных методов перед валидацией качества |
| `CpuCudaBenchmark` | Оценка масштабируемости по размеру изображения |
| `utils.warmup.SegmentationWarmUp` | Дополнительный специализированный прогрев для классических методов |
| `segmenters.SklearnSegmenter` | Тестирование пороговых и граничных методов из sklearn/scikit-image |
| `segmenters.OpenCVSegmenter` | Тестирование граничных методов из OpenCV |

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