# 🔥 SegmentationWarmUp — Утилита прогрева перед бенчмарком сегментации

## 📖 Описание
Модуль `utils/warmup.py` предоставляет **универсальный инструмент** для предварительного прогрева классических и нейросетевых методов сегментации перед запуском бенчмарков.

> ⚠️ **Важно:** Данный модуль не выполняет сегментацию, а лишь **подготавливает окружение** для стабильных измерений производительности. Используйте его в связке с `CpuCudaBenchmark` или `BatchClassicTester`.

## ✨ Ключевые возможности
### 🎨 Поддерживаемые тестовые паттерны

| Паттерн | Описание | Сценарии использования |
|---------|----------|----------------------|
| `gradient` | Линейный градиент по горизонтали/вертикали | Пороговые методы, адаптивная бинаризация |
| `noise` | Случайный цветной шум (uniform) | Детекторы границ, фильтры, фазовая конгруэнтность |
| `checkerboard` | Шахматная доска с настраиваемым размером клетки | Детекция контуров, морфологические операции |
| `circles` | Концентрические круги разного радиуса | Тестирование кривизны границ, активные контуры |

### ⚙️ Автоматическая адаптация под устройство
- **CPU**: Простой замер через `time.perf_counter()`.
- **CUDA**: Специальный warm-up с `torch.cuda.synchronize()` и `torch.cuda.empty_cache()` для инициализации контекста и прогрева JIT-ядер.
- **Универсальный интерфейс**: Авто-определение метода сегментации (`.segment()` или `.segment_with_mask()`).

### 📊 Типизированная статистика выполнения
```python
from utils.warmup import WarmupStats

stats: WarmupStats = {
    "method": "otsu_thresholding",
    "n_runs": 10,
    "median_time_ms": 12.34,
    "mean_time_ms": 12.56,
    "std_time_ms": 0.45,
    "min_time_ms": 11.89,
    "max_time_ms": 13.21
}
```

### 🛡️ Устойчивость к ошибкам
- Сбойный прогон не прерывает весь warm-up — ошибка логируется, время записывается как `inf`.
- Поддержка сегментеров с любым интерфейсом (через `hasattr()` проверку).
- Автоматический откат на сгенерированное изображение, если `real_image` не задан.

## 🚀 Быстрый старт
### Базовое использование: прогрев одного метода
```python
from utils.warmup import SegmentationWarmUp
from segmenters.SklearnSegmenter import SklearnSegmenter

# Инициализация
warmup = SegmentationWarmUp(
    n_warmup_runs=5,
    image_size=(256, 256),
    device="cuda"  # или "cpu"
)

# Создание и прогрев сегментера
segmenter = SklearnSegmenter("otsu_thresholding")
stats = warmup.warmup_segmenter(
    segmenter=segmenter,
    method_name="otsu",
    verbose=True
)

print(f"Mean time: {stats['mean_time_ms']:.2f}ms ± {stats['std_time_ms']:.2f}ms")
```

### Прогрев всех методов для бенчмарка
```python
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.OpenCVSegmenter import OpenCVSegmenter

# Словарь сегментеров
segmenters = {
    "otsu_sklearn": SklearnSegmenter("otsu_thresholding"),
    "canny_opencv": OpenCVSegmenter("canny_edge"),
    "kmeans_sklearn": SklearnSegmenter("kmeans_segmentation", k=3),
    "slic_sklearn": SklearnSegmenter("slic", n_segments=100),
}

# Прогрев всех
warmup = SegmentationWarmUp(n_warmup_runs=3, image_size=(512, 512))
results = warmup.warmup_all_segmenters(segmenters, verbose=True)

# Сводка
print(warmup.get_warmup_summary())
```

### Использование реального изображения для теста
```python
import cv2
import numpy as np

# Загрузка реального изображения
real_img = cv2.imread("test_sample.jpg")
real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)

# Прогрев с реальными данными
stats = warmup.warmup_segmenter(
    segmenter=segmenter,
    method_name="my_method",
    real_image=real_img,
    use_real_image=True,  # Важно!
    verbose=True
)
```

### Генерация тестовых паттернов
```python
warmup = SegmentationWarmUp(image_size=(512, 512))

# Создание разных паттернов
gradient_img = warmup.create_test_image("gradient")      # Для пороговых методов
noise_img = warmup.create_test_image("noise")            # Для граничных
checkerboard_img = warmup.create_test_image("checkerboard")  # Для контуров
circles_img = warmup.create_test_image("circles")        # Для кривизны

# Сохранение для визуальной проверки
import PIL.Image
PIL.Image.fromarray(gradient_img).save("test_gradient.png")
```

## ⚙️ Конфигурация
### Параметры `__init__()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `n_warmup_runs` | `int` | `10` | Количество прогонов для каждого метода |
| `image_size` | `Tuple[int, int]` | `(256, 256)` | Размер тестовых изображений `(высота, ширина)` |
| `device` | `str` | `"cuda"` если доступно, иначе `"cpu"` | Устройство для вычислений |

### Параметры `warmup_segmenter()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `segmenter` | `SegmenterLike` | — | Экземпляр сегментера с методом `.segment()` или `.segment_with_mask()` |
| `method_name` | `str` | — | Имя метода для логирования и ключа в результатах |
| `real_image` | `Optional[np.ndarray]` | `None` | Реальное изображение для тестирования |
| `verbose` | `bool` | `True` | Вывод прогресса в консоль |
| `use_real_image` | `bool` | `False` | Использовать `real_image` вместо сгенерированного |

### Возвращаемое значение `WarmupStats`
```python
class WarmupStats(TypedDict):
    method: str           # Имя метода
    n_runs: int           # Количество успешных прогонов
    median_time_ms: float # Медианное время (мс)
    mean_time_ms: float   # Среднее время (мс)
    std_time_ms: float    # Стандартное отклонение (мс)
    min_time_ms: float    # Минимальное время (мс)
    max_time_ms: float    # Максимальное время (мс)
```

## 📚 Справочник методов
### 🔹 Основные методы
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `create_test_image()` | `pattern: ImagePattern` | Генерация тестового изображения заданного типа | `np.ndarray[H,W,3], uint8` |
| `warmup_segmenter()` | `segmenter`, `method_name`, `real_image`, `verbose`, `use_real_image` | Прогрев одного сегментера с замером времени | `WarmupStats` |
| `warmup_all_segmenters()` | `segmenters_dict`, `image`, `verbose` | Пакетный прогрев всех методов в словаре | `Dict[str, Union[WarmupStats, Dict]]` |
| `get_warmup_summary()` | — | Текстовая сводка результатов | `str` |

### 🔹 Внутренние методы
| Метод | Параметры | Описание |
|-------|-----------|----------|
| `_warmup_cuda()` | `segmenter`, `image`, `verbose` | Специальный warm-up для CUDA: синхронизация, очистка кэша |

## 🔄 Конвейер warm-up: подготовка → прогоны → статистика
### Логика `warmup_segmenter()`
```python
def warmup_segmenter(self, segmenter, method_name, **kwargs):
    # 1. Выбор изображения
    image = real_image if use_real_image and real_image else self.create_test_image("gradient")
    
    # 2. Цикл прогонов с замером времени
    for i in range(self.n_warmup_runs):
        start = time.perf_counter()
        try:
            # Авто-определение интерфейса
            if hasattr(segmenter, "segment_with_mask"):
                result, mask = segmenter.segment_with_mask(image)
            elif hasattr(segmenter, "segment"):
                result = segmenter.segment(image)
            end = time.perf_counter()
            times.append(end - start)
        except Exception as e:
            times.append(float("inf"))  # Не прерываем цикл
    
    # 3. Специальный CUDA warm-up при необходимости
    if "cuda" in str(segmenter.device).lower():
        self._warmup_cuda(segmenter, image)
    
    # 4. Расчёт статистики
    return {
        "method": method_name,
        "mean_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        # ... остальные метрики
    }
```

### CUDA-специфичный warm-up
```python
def _warmup_cuda(self, segmenter, image, verbose):
    # Синхронизация перед прогонами
    torch.cuda.synchronize()
    
    # Дополнительные прогоны без замера
    for _ in range(self.n_warmup_runs):
        try:
            if hasattr(segmenter, "segment_with_mask"):
                segmenter.segment_with_mask(image)
            else:
                segmenter.segment(image)
        except Exception:
            pass
    
    # Синхронизация и очистка после
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
```

### Генерация тестовых паттернов
```python
def create_test_image(self, pattern: ImagePattern):
    if pattern == "gradient":
        # Градиент для пороговых методов
        img[:, :, 0] = np.tile(np.linspace(0, 255, w), (h, 1))  # R
        img[:, :, 1] = np.tile(np.linspace(0, 255, h).reshape(-1, 1), (1, w))  # G
        img[:, :, 2] = (img[:, :, 0] + img[:, :, 1]) // 2  # B
    elif pattern == "checkerboard":
        # Шахматная доска
        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    img[i:i+size, j:j+size] = 255
    # ... остальные паттерны
    return img
```

## 📊 Интерпретация статистики
| Метрика | Описание | Когда важна |
|---------|----------|------------|
| `mean_time_ms` | Среднее время выполнения | Общая оценка производительности |
| `std_time_ms` | Стандартное отклонение | Стабильность: малое значение = предсказуемое время |
| `median_time_ms` | Медиана | Устойчива к выбросам, лучше mean при неравномерных замерах |
| `min_time_ms` | Лучший результат | Теоретический предел производительности |
| `max_time_ms` | Худший результат | Выявление "просадок" (GC, контекст-свитчи) |
| `n_runs` | Количество прогонов | Надёжность оценки: чем больше, тем точнее |

### Пример интерпретации
```python
# Стабильный метод
{"mean": 12.5, "std": 0.3}  # ✅ Предсказуемое время

# Нестабильный метод
{"mean": 15.2, "std": 4.8}  # ⚠️ Возможны просадки, проверить сборку мусора / CUDA sync

# Сбойный метод
{"mean": inf, "n_runs": 1}  # ❌ Метод падает, нужна отладка
```

## ⚡ Производительность и рекомендации
### Время warm-up (на изображении 256×256, 10 прогонов)
```
✅ Быстро (<1 сек):
   - Пороговые методы (global, otsu, adaptive)
   - Простые детекторы границ (sobel, prewitt)

⚠️ Средне (1–5 сек):
   - Кластеризация (kmeans, dbscan)
   - Суперпиксели (slic, felzenszwalb)
   - Watershed, random walker

❌ Медленно (5–30+ сек):
   - Активные контуры (chan_vese, morphological_snakes)
   - Нейросетевые методы с загрузкой весов
   - CUDA-методы с первой инициализацией контекста
```

### Рекомендации по настройке
1. **Начинайте с малого**: `n_warmup_runs=3`, `image_size=(256, 256)` для быстрой проверки.
2. **Используйте реальные данные** перед финальным бенчмарком: `use_real_image=True`.
3. **Для CUDA**: всегда вызывайте warm-up перед замером — первый запуск может быть в 10× медленнее.
4. **Фильтруйте выбросы**: при анализе используйте `median` вместо `mean`, если `std/mean > 0.3`.
5. **Кэшируйте результаты**: сохраняйте `warmup_results` для последующего сравнения.

## 🛠️ Обработка ошибок и устойчивость
### Авто-определение интерфейса сегментера
```python
if hasattr(segmenter, "segment_with_mask"):
    result, mask = segmenter.segment_with_mask(image)
elif hasattr(segmenter, "segment"):
    result = segmenter.segment(image)
else:
    raise AttributeError("Segmenter must have 'segment' or 'segment_with_mask' method")
```

### Обработка сбойных прогонов
```python
try:
    # ... выполнение ...
    times.append(end - start)
except Exception as e:
    print(f"❌ Warm-up failed: {e}")
    times.append(float("inf"))  # Не прерываем цикл
    # Остальные прогоны продолжатся
```

### CUDA-специфичная защита
```python
def _warmup_cuda(self, segmenter, image, verbose):
    torch.cuda.synchronize()  # Ждём завершения предыдущих операций
    # ... прогоны ...
    torch.cuda.synchronize()  # Ждём завершения warm-up
    torch.cuda.empty_cache()  # Освобождаем неиспользуемую память
```

### Рекомендации по отладке
1. **Включите `verbose=True`** для пошагового мониторинга:
   ```python
   stats = warmup.warmup_segmenter(segmenter, "my_method", verbose=True)
   # Вывод: 🔥 Warm-up: my_method (10 runs)
   #        ✅ Run 1: 12.34ms
   #        📊 Mean: 12.56ms ± 0.45ms
   ```

2. **Проверьте устройство сегментера**:
   ```python
   print(f"Segmenter device: {getattr(segmenter, 'device', 'unknown')}")
   # Убедитесь, что device совпадает с warmup.device
   ```

3. **Тестируйте на минимальном изображении**:
   ```python
   warmup = SegmentationWarmUp(image_size=(64, 64), n_warmup_runs=1)
   stats = warmup.warmup_segmenter(segmenter, "test")
   # Быстрая проверка конвейера без долгого ожидания
   ```

4. **Анализируйте `std/mean` для стабильности**:
   ```python
   if stats['std_time_ms'] / stats['mean_time_ms'] > 0.3:
       print(f"⚠️ High variance for {stats['method']}: consider increasing n_warmup_runs")
   ```

## 🤝 Зависимости
```text
numpy>=1.20          # Массивы, статистики, генерация паттернов
torch>=1.9           # CUDA-поддержка, синхронизация (опционально)
typing-extensions    # TypedDict, Literal (для типизации)
```

### Опциональные зависимости
```bash
# Для визуализации сгенерированных паттернов
pip install Pillow

# Для логирования в файл
pip install loguru
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование SegmentationWarmUp |
|--------|---------------------------------|
| `CpuCudaBenchmark` | Предварительный прогрев перед замером времени инференса |
| `BatchClassicTester` | Warm-up всех классических методов перед пакетным тестом |
| `BatchClassicTester2` | Прогрев перед валидацией качества (чтобы время не влияло на метрики) |
| `TorchImplementationValidator` | Стабилизация времени для сравнения PyTorch vs OpenCV |
| `utils.backend_exporter` | Прогрев перед экспортом в ONNX/TRT для точного замера компиляции |

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
