# 📊 Benchmark Loader — Загрузка профилей методов из результатов бенчмарков

## 📖 Описание
Модуль `utils/benchmark_loader.py` предоставляет **утилиты для импорта и агрегации** результатов бенчмарков в структурированные профили методов сегментации.

> ⚠️ **Важно:** Данный модуль является **инфраструктурным** — он не выполняет сегментацию или бенчмарки, а лишь **загружает и нормализует** результаты для дальнейшего анализа. Используйте его в связке с `AutoSegmenter`, `CpuCudaBenchmark` или `BatchClassicTester`.

## ✨ Ключевые возможности
### 🔄 Поддерживаемые форматы входных данных

| Формат | Расширение | Описание | Пример использования |
|--------|-----------|----------|---------------------|
| **CSV** | `.csv` | Таблица с метриками производительности (время, память) | Результаты `CpuCudaBenchmark` |
| **JSON** | `.json` | Словарь с метриками качества (IoU, Dice, Precision) | Результаты `BatchClassicTester2` |

### 📦 Автоматическая агрегация профилей
- **Объединение метрик**: время из CSV + качество из JSON → единый `MethodProfile`.
- **Нормализация единиц**: конвертация секунд в миллисекунды для удобства сравнения.
- **Fallback-значения**: безопасные дефолты при отсутствии данных в источниках.
- **Типизация**: возврат `Dict[str, MethodProfile]` для статической проверки типов.

### 🎯 Интеграция с `MethodProfile` из `AutoSegmenter`
```python
from segmenters.AutoSegmenter import MethodProfile, ImageType

profile: MethodProfile = MethodProfile(
    name="otsu_thresholding",
    library="opencv",
    avg_time_ms=12.34,      # Среднее время выполнения (мс)
    avg_iou=0.85,           # Средний IoU на валидации
    memory_mb=50,           # Потребление памяти (МБ)
    best_for_type=[ImageType.NATURAL, ImageType.DOCUMENT],
    robustness=0.8,         # Устойчивость к вариациям данных [0, 1]
    parameter_sensitivity=0.5  # Чувствительность к параметрам [0, 1]
)
```

## 🚀 Быстрый старт
### Базовое использование: загрузка профилей из бенчмарков
```python
from utils.benchmark_loader import load_profiles_from_benchmark

# Загрузка профилей для OpenCV-методов
profiles = load_profiles_from_benchmark(
    benchmark_csv_path="./results/benchmark_opencv.csv",
    validation_json_path="./results/validation_metrics.json",
    library="opencv"
)

# Доступ к профилям
for name, profile in profiles.items():
    print(f"{name:30s}: {profile.avg_time_ms:.2f}ms, IoU={profile.avg_iou:.3f}")
```

### Фильтрация по библиотеке и метрикам
```python
# Загрузка профилей для разных библиотек
opencv_profiles = load_profiles_from_benchmark(
    "./results/benchmark_opencv.csv",
    "./results/validation_opencv.json",
    library="opencv"
)

sklearn_profiles = load_profiles_from_benchmark(
    "./results/benchmark_sklearn.csv", 
    "./results/validation_sklearn.json",
    library="sklearn"
)

# Сравнение: быстрый метод с высоким IoU
candidates = {
    name: p for name, p in {**opencv_profiles, **sklearn_profiles}.items()
    if p.avg_time_ms < 50 and p.avg_iou > 0.7
}

for name, profile in candidates.items():
    print(f"✅ {name}: {profile.avg_time_ms:.1f}ms, IoU={profile.avg_iou:.3f}")
```

### Экспорт профилей для AutoSegmenter
```python
from segmenters.AutoSegmenter import AutoSegmenter

# Загрузка профилей
profiles = load_profiles_from_benchmark(
    benchmark_csv_path="./results/final_benchmark.csv",
    validation_json_path="./results/final_validation.json",
    library="mixed"
)

# Регистрация в AutoSegmenter для авто-выбора метода
auto_segmenter = AutoSegmenter()
for name, profile in profiles.items():
    auto_segmenter.register_profile(profile)

# Авто-выбор метода под задачу
best_method = auto_segmenter.select_best_method(
    image_type=ImageType.MEDICAL,
    max_time_ms=100,
    min_iou=0.8
)
print(f"Recommended: {best_method}")
```

### Обработка отсутствующих метрик
```python
# Если валидационный JSON не содержит метрику для метода — используется fallback
profiles = load_profiles_from_benchmark(
    benchmark_csv_path="./results/partial_benchmark.csv",
    validation_json_path="./results/partial_validation.json",
    library="custom"
)

# Проверка на наличие метрик
for name, profile in profiles.items():
    if profile.avg_iou == 0.75:  # Default fallback
        print(f"⚠️ {name}: IoU не найден, использовано значение по умолчанию")
```

## ⚙️ Конфигурация
### Параметры `load_profiles_from_benchmark()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `benchmark_csv_path` | `str` | — | Путь к CSV-файлу с результатами бенчмарка (колонки: `Method`, `Mean_Time_s`, ...) |
| `validation_json_path` | `str` | — | Путь к JSON-файлу с метриками качества (`{method_name: {iou: ..., dice: ...}}`) |
| `library` | `str` | `"opencv"` | Название библиотеки для поля `MethodProfile.library` |

### Структура входного CSV-файла
```csv
Method,Mean_Time_s,Std_Time_s,Memory_MB,Device
otsu_thresholding,0.012,0.002,45,CPU
canny_edge,0.045,0.008,52,CPU
kmeans_segmentation,0.234,0.045,128,CPU
...
```

### Структура входного JSON-файла
```json
{
  "otsu_thresholding": {
    "iou": 0.82,
    "dice": 0.90,
    "precision": 0.88,
    "recall": 0.92
  },
  "canny_edge": {
    "iou": 0.65,
    "dice": 0.78,
    "precision": 0.70,
    "recall": 0.85
  }
}
```

### Возвращаемое значение
```python
Dict[str, MethodProfile]: {
    "method_name": MethodProfile(
        name="method_name",
        library="opencv",
        avg_time_ms=12.34,
        avg_iou=0.85,
        memory_mb=50,
        best_for_type=[ImageType.NATURAL],
        robustness=0.8,
        parameter_sensitivity=0.5
    ),
    ...
}
```

## 📚 Справочник функций
### 🔹 Основная функция
| Функция | Параметры | Описание | Возвращает |
|---------|-----------|----------|-----------|
| `load_profiles_from_benchmark()` | `benchmark_csv_path`, `validation_json_path`, `library` | Загрузка и агрегация профилей методов из результатов бенчмарков | `Dict[str, MethodProfile]` |

### 🔹 Внутренняя логика (для расширения)
| Шаг | Описание |
|-----|----------|
| 1. Чтение CSV | `pd.read_csv()` → `DataFrame` с метриками производительности |
| 2. Чтение JSON | `json.load()` → `Dict` с метриками качества |
| 3. Агрегация по методу | Объединение данных по ключу `Method` / `method_name` |
| 4. Создание профиля | Инициализация `MethodProfile` с нормализованными значениями |
| 5. Возврат результата | Словарь `{имя_метода: профиль}` для дальнейшего использования |

## 🔄 Конвейер загрузки: чтение → агрегация → нормализация → возврат
### Логика `load_profiles_from_benchmark()`
```python
def load_profiles_from_benchmark(benchmark_csv_path, validation_json_path, library="opencv"):
    # 1. Чтение источников данных
    df = pd.read_csv(benchmark_csv_path)  # CSV с временем/памятью
    with open(validation_json_path) as f:
        validation_data = json.load(f)     # JSON с метриками качества
    
    profiles = {}
    
    # 2. Агрегация по каждому методу
    for method_name in df["Method"].unique():
        method_data = df[df["Method"] == method_name].iloc[0]
        
        # 3. Извлечение метрик качества с fallback
        val_metrics = validation_data.get(method_name, {})
        iou = val_metrics.get("iou", 0.75)  # Default при отсутствии
        
        # 4. Создание профиля (заглушки для сложных полей)
        profiles[method_name] = MethodProfile(
            name=method_name,
            library=library,
            avg_time_ms=method_data["Mean_Time_s"] * 1000,  # sec → ms
            avg_iou=iou,
            memory_mb=50,  # TODO: добавить замер памяти
            best_for_type=[ImageType.NATURAL],  # TODO: авто-определение
            robustness=0.8,  # TODO: расчёт из std
            parameter_sensitivity=0.5,  # TODO: анализ чувствительности
        )
    
    return profiles
```

### Нормализация единиц и fallback-значения
```python
# Конвертация времени: секунды → миллисекунды
avg_time_ms = method_data["Mean_Time_s"] * 1000

# Fallback для отсутствующих метрик качества
iou = val_metrics.get("iou", 0.75)  # 0.75 — разумный дефолт для сегментации

# Заглушки для полей, требующих дополнительного анализа
best_for_type = [ImageType.NATURAL]  # TODO: анализ по типам изображений
robustness = 0.8  # TODO: 1 - std/mean из повторных прогонов
parameter_sensitivity = 0.5  # TODO: варьирование параметров и оценка влияния
```

## 📊 Интерпретация полей `MethodProfile`
| Поле | Тип | Диапазон | Интерпретация |
|------|-----|----------|---------------|
| `avg_time_ms` | `float` | [0, ∞) | Среднее время выполнения; меньше = быстрее |
| `avg_iou` | `float` | [0.0, 1.0] | Средний IoU на валидации; больше = лучше качество |
| `memory_mb` | `float` | [0, ∞) | Потребление оперативной памяти; меньше = эффективнее |
| `best_for_type` | `List[ImageType]` | — | Типы изображений, для которых метод оптимален |
| `robustness` | `float` | [0.0, 1.0] | Устойчивость к шуму/вариациям; 1.0 = идеальная стабильность |
| `parameter_sensitivity` | `float` | [0.0, 1.0] | Чувствительность к параметрам; 0.0 = метод "из коробки" |

### Пример использования в авто-выборе метода
```python
def select_method(profiles: Dict[str, MethodProfile], 
                  max_time: float, 
                  min_iou: float) -> Optional[str]:
    """Выбор метода по ограничениям времени и качества."""
    candidates = [
        (name, p) for name, p in profiles.items()
        if p.avg_time_ms <= max_time and p.avg_iou >= min_iou
    ]
    if not candidates:
        return None
    # Возвращаем самый быстрый из подходящих
    return min(candidates, key=lambda x: x[1].avg_time_ms)[0]
```

## ⚡ Производительность и оптимизации
### Сложность загрузки
| Операция | Сложность | Примечание |
|----------|-----------|-----------|
| Чтение CSV | O(N) | N = количество строк в бенчмарке |
| Чтение JSON | O(M) | M = количество методов в валидации |
| Агрегация | O(N + M) | Линейный проход по обоим источникам |
| Создание профилей | O(K) | K = уникальных методов |

### Рекомендации по оптимизации
1. **Кэшируйте результаты загрузки**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=4)
   def load_profiles_cached(csv_path, json_path, library):
       return load_profiles_from_benchmark(csv_path, json_path, library)
   ```

2. **Фильтруйте CSV на этапе чтения** для больших файлов:
   ```python
   df = pd.read_csv(path, usecols=["Method", "Mean_Time_s"])
   ```

3. **Используйте `dtype` при чтении** для экономии памяти:
   ```python
   df = pd.read_csv(path, dtype={"Method": "category", "Mean_Time_s": "float32"})
   ```

4. **Параллельная загрузка** при работе с несколькими библиотеками:
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor() as executor:
       futures = [
           executor.submit(load_profiles_from_benchmark, csv, json, lib)
           for csv, json, lib in benchmark_configs
       ]
       all_profiles = {**[f.result() for f in futures]}
   ```

## 🛠️ Обработка ошибок и устойчивость
### Валидация входных файлов
```python
# Проверка существования файлов перед чтением
import os
if not os.path.exists(benchmark_csv_path):
    raise FileNotFoundError(f"CSV not found: {benchmark_csv_path}")
if not os.path.exists(validation_json_path):
    raise FileNotFoundError(f"JSON not found: {validation_json_path}")
```

### Безопасное извлечение метрик с fallback
```python
# Извлечение IoU с дефолтом при отсутствии
val_metrics = validation_data.get(method_name, {})
iou = val_metrics.get("iou", 0.75)  # 0.75 — разумный дефолт

# Обработка отсутствующих колонок в CSV
if "Mean_Time_s" not in method_data:
    logger.warning(f"Missing 'Mean_Time_s' for {method_name}, using 0.0")
    avg_time_ms = 0.0
else:
    avg_time_ms = method_data["Mean_Time_s"] * 1000
```

### Логирование предупреждений
```python
# Предупреждение при использовании fallback-значений
if iou == 0.75 and method_name not in validation_data:
    logger.warning(f"{method_name}: IoU not found in validation, using default 0.75")

# Информирование о заглушках
if profile.memory_mb == 50:  # Default
    logger.debug(f"{method_name}: memory_mb estimated (not measured)")
```

### Рекомендации по отладке
1. **Включите подробное логирование**:
   ```python
   import logging
   logging.getLogger("utils.benchmark_loader").setLevel(logging.DEBUG)
   ```

2. **Проверьте структуру входных файлов**:
   ```python
   import pandas as pd
   df = pd.read_csv("benchmark.csv")
   print(df.columns.tolist())  # Должно включать "Method", "Mean_Time_s"
   
   import json
   with open("validation.json") as f:
       data = json.load(f)
   print(list(data.keys())[:5])  # Примеры имён методов
   ```

3. **Тестируйте на минимальном наборе**:
   ```python
   # Создайте тестовые файлы с 1–2 методами для проверки конвейера
   profiles = load_profiles_from_benchmark(
       "./tests/tiny_benchmark.csv",
       "./tests/tiny_validation.json",
       library="test"
   )
   assert len(profiles) == 2
   ```

4. **Валидируйте выходные профили**:
   ```python
   for name, profile in profiles.items():
       assert profile.avg_time_ms >= 0, f"{name}: negative time"
       assert 0 <= profile.avg_iou <= 1, f"{name}: IoU out of range"
       assert isinstance(profile.best_for_type, list), f"{name}: best_for_type not list"
   ```

## 🤝 Зависимости
```text
pandas>=1.3          # Чтение и обработка CSV-файлов
numpy>=1.20          # Поддержка массивов (транзитивно через pandas)
segmenters.AutoSegmenter  # Класс MethodProfile, ImageType enum
```

### Опциональные зависимости для расширенного функционала
```bash
# Для параллельной загрузки
pip install concurrent.futures  # Встроен в Python 3.2+

# Для кэширования
pip install joblib  # Альтернатива lru_cache для дискового кэша

# Для валидации схем данных
pip install pydantic  # Строгая валидация входных/выходных структур
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование benchmark_loader.py |
|--------|----------------------------------|
| `AutoSegmenter` | Регистрация профилей через `register_profile()` для авто-выбора метода |
| `CpuCudaBenchmark` | Экспорт результатов в CSV для последующей загрузки |
| `BatchClassicTester2` | Экспорт метрик качества в JSON для агрегации |
| `SegmentationTester` | Сравнение профилей разных реализаций одного метода |
| `utils.strategies` | Выбор стратегии инференса на основе `avg_time_ms` и `avg_iou` |

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