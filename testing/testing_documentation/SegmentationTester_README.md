# 🧪 SegmentationTester — Универсальный тестировщик сегментаторов

## 📖 Описание
Модуль `SegmentationTester.py` предназначен для **гибкого тестирования и сравнения** произвольных методов сегментации, реализующих интерфейс `BaseSegmenter`. Он объединяет возможности замера производительности, расчёта метрик качества и визуализации результатов в едином интерфейсе.

> ⚠️ **Важно:** Данный модуль работает с *отдельными изображениями* и *произвольными сегментерами*. Для массового тестирования на датасетах используйте `BatchClassicTester` / `BatchClassicTester2`; для сравнения нейросетевых архитектур — `SegmentationBenchmark`.

## ✨ Ключевые возможности
- 🔗 **Гибкая регистрация методов:** Любой класс с методом `.segment_with_mask(image) -> (result, mask)` может быть добавлен через `add_method()`.
- 📈 **Полный набор метрик:** Автоматический расчёт IoU, Dice, F1-Score, Precision, Recall, Pixel Accuracy, MAE, Hausdorff Distance при наличии Ground Truth.
- 🎨 **Визуализация:** 
  - Grid-сравнение результатов нескольких методов.
  - Overlay с альфа-смешиванием (оригинал + предсказание).
  - Отдельное отображение бинарных масок.
  - Превью-галерея, отсортированная по скорости.
- ⚡ **Бенчмарк производительности:**
  - Warm-up прогоны для стабилизации (особенно важно для нейросетей на GPU).
  - Многократные замеры времени с расчётом mean/std/min/max.
  - Статистика площади маски и процента покрытия.
- 💾 **Экспорт в 6 форматах:**
  | Формат | Содержимое | Метод сохранения |
  |--------|-----------|-----------------|
  | `PNG/JPG` | Оригиналы, результаты, overlay, маски | Автоматически при `output_dir` |
  | `JSON` | Детальные метрики, статистика, конфигурация | `_save_statistics()`, `_save_results_summary()` |
  | `CSV` | Сводные таблицы для анализа в pandas/Excel | `_save_statistics()`, `_save_benchmark_results()` |
  | `TXT` | Человекочитаемые отчёты с форматированием | `_save_method_info()`, `_save_benchmark_results()` |
  | `HTML` | Интерактивные отчёты (в разработке) | `_create_html_summary_report()` (заглушка) |
  | `XLSX` | Excel-таблицы с автоформатированием | `_save_benchmark_results()` (опционально) |
- 🛡️ **Устойчивость к ошибкам:** Исключения в отдельных методах логируются, но не прерывают пакетное выполнение.

## 🚀 Быстрый старт
### Базовое использование: сравнение двух методов
```python
from testing.SegmentationTester import SegmentationTester
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.TorchSegmenter import TorchSegmenter
import numpy as np
from PIL import Image

# Загрузка изображения
image = np.array(Image.open("test.jpg").convert("RGB"))

# Инициализация тестера
tester = SegmentationTester(
    base_output_dir="./results/tester_demo",
    enable_warmup=True,   # Включить warm-up для нейросетей
    n_warmup_runs=3       # Количество разогревочных прогонов
)

# Регистрация методов
tester.add_method("Otsu_CV2", OpenCVSegmenter("otsu_thresholding"))
tester.add_method("Otsu_Torch", TorchSegmenter("otsu_thresholding"))

# (Опционально) Загрузка Ground Truth для расчёта метрик
# tester.load_ground_truth("gt_mask.png")

# Сравнение методов с визуализацией
results = tester.compare_methods(
    image=image,
    method_names=["Otsu_CV2", "Otsu_Torch"],
    test_name="otsu_comparison",
    save_comparison=True,
    show_plots=True
)

# Доступ к результатам
for method, data in results.items():
    print(f"{method}: время={data['time']:.3f}s, покрытие={data['mask_percentage']:.2f}%")
```

### Бенчмарк с метриками качества
```python
# Запуск бенчмарка с 5 прогонами и GT-маской
df_bench = tester.benchmark_methods(
    image=image,
    n_runs=5,
    ground_truth=np.array(Image.open("gt.png").convert("L")),
    test_name="quality_benchmark",
    save_benchmark=True
)

# Анализ результатов
print(df_bench[["Method", "Mean_Time_s", "IoU", "Dice", "F1_Score"]].sort_values("IoU", ascending=False))
```

### Пакетное тестирование с метриками
```python
# Сравнение с отображением метрик на графике
results_with_metrics = tester.compare_methods_with_metrics(
    image=image,
    ground_truth=gt_mask,
    method_names=["Otsu_CV2", "Adaptive_CV2", "Canny_Torch"],
    test_name="metrics_comparison",
    show_plots=True
)

# Доступ к метрикам конкретного метода
metrics = results_with_metrics["Adaptive_CV2"]["metrics"]
print(f"IoU: {metrics['iou']:.3f}, F1: {metrics['f1_score']:.3f}")
```

## ⚙️ Конфигурация
### Параметры инициализации `SegmentationTester`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `base_output_dir` | `PathLike` | `"./../data/segmentation_results"` | Базовая директория для всех результатов |
| `ground_truth_path` | `Optional[PathLike]` | `None` | Путь к GT-маске (загружается при инициализации) |
| `enable_warmup` | `bool` | `True` | Включить ли warm-up прогоны перед бенчмарком |
| `n_warmup_runs` | `int` | `3` | Количество разогревочных итераций |

### Параметры `compare_methods()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `image` | `ImageInput` | — | Изображение: путь, `np.ndarray` или `PIL.Image` |
| `method_names` | `Optional[List[str]]` | `None` | Список методов для сравнения (`None` = все зарегистрированные) |
| `figsize` | `Tuple[int, int]` | `(20, 15)` | Размер matplotlib-фигуры |
| `save_comparison` | `bool` | `True` | Сохранять ли сводный график сравнения |
| `test_name` | `Optional[str]` | `None` | Префикс имени теста для организации вывода |
| `show_plots` | `bool` | `True` | Показывать ли график через `plt.show()` |

### Параметры `benchmark_methods()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `n_runs` | `int` | `3` | Количество прогонов для замера времени |
| `save_benchmark` | `bool` | `True` | Сохранять ли отчёт бенчмарка |
| `force_warmup` | `bool` | `False` | Выполнить warm-up для всех методов принудительно |
| `ground_truth` | `Optional[MaskArray]` | `None` | GT-маска для расчёта метрик (переопределяет загруженную) |

## 📂 Структура выходных данных
### Для `compare_methods()` / `compare_methods_with_metrics()`
```
{base_output_dir}/{test_name}_YYYYMMDD_HHMMSS/
├── images/
│   ├── original.jpg              # Исходное изображение
│   ├── {method}_result.jpg       # Результат сегментации
│   ├── {method}_overlay.jpg      # Наложение (30% оригинал + 70% результат)
│   └── {method}_bright_overlay.jpg  # Яркий overlay (10%/90%)
├── masks/
│   └── {method}_mask.png         # Бинарная маска (0=чёрный, 255=белый)
├── comparisons/
│   ├── methods_comparison.jpg    # Grid-визуализация всех методов
│   └── methods_comparison_small.jpg  # Уменьшенная версия
├── statistics/
│   ├── statistics.json           # Детальная статистика по методам
│   ├── statistics.csv            # Сводная таблица для pandas
│   ├── test_report.txt           # Человекочитаемый отчёт
│   └── results_summary.json      # Сводка результатов (без больших массивов)
└── {method}/                     # Индивидуальные папки методов
    ├── result.jpg
    ├── mask.png
    ├── overlay.jpg
    ├── metrics.json              # Если есть GT
    ├── metrics.txt               # Текстовая версия метрик
    └── method_info.txt           # Параметры и информация о методе
```

### Для `benchmark_methods()`
```
{base_output_dir}/benchmark_{test_name}_YYYYMMDD_HHMMSS/
├── images/
│   ├── original.jpg
│   ├── {method}_result.jpg
│   └── {method}_overlay.jpg
├── masks/
│   └── {method}_mask.png
├── comparisons/
│   ├── benchmark_time.png        # Бар-чарт времени выполнения
│   ├── benchmark_scatter.png     # Время vs Площадь маски
│   ├── iou_vs_time.png           # IoU vs Время (если есть GT)
│   └── methods_preview.jpg       # Превью-галерея, сортировка по скорости
├── statistics/
│   ├── benchmark_results.csv     # Сводная таблица бенчмарка
│   ├── benchmark_results.xlsx    # Excel-версия (если установлен openpyxl)
│   └── benchmark_report.txt      # Текстовый отчёт с топ-методами
└── metrics/                      # Если есть GT
    ├── metrics_comparison.json
    ├── metrics_comparison.csv
    └── metrics_table.jpg         # Изображение со сводной таблицей метрик
```

## 📊 Метрики качества
При наличии Ground Truth рассчитываются следующие метрики (через `SegmentationMetrics`):

| Метрика | Описание | Диапазон | Интерпретация |
|---------|----------|----------|---------------|
| **IoU** | Intersection over Union | [0.0, 1.0] | Чем выше, тем лучше; основной критерий |
| **Dice** | Dice / Sørensen coefficient | [0.0, 1.0] | Более устойчив к дисбалансу классов |
| **F1-Score** | Гармоническое среднее Precision/Recall | [0.0, 1.0] | Универсальная метрика качества |
| **Precision** | Точность: доля верных объектов среди предсказанных | [0.0, 1.0] | Высокая = мало ложных срабатываний |
| **Recall** | Полнота: доля найденных объектов среди истинных | [0.0, 1.0] | Высокая = мало пропущенных объектов |
| **Pixel Accuracy** | Доля верно классифицированных пикселей | [0.0, 1.0] | Может быть завышена при дисбалансе |
| **MAE** | Mean Absolute Error | [0.0, 1.0] | Средняя абсолютная ошибка (чем ниже, тем лучше) |
| **Hausdorff Distance** | Максимальное расстояние между контурами | [0, ∞) | Чувствительна к выбросам, важна для границ |
| **Area Difference** | \|area_pred − area_gt\| | [0, ∞) | Разница площадей объекта |

## 🎨 Типы визуализаций
### 1. Grid-сравнение (`compare_methods()`)
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Original   │  Method A   │  Method B   │  Method C   │
│  Image      │  (0.45s)    │  (0.32s)    │  (1.21s)    │
│             │  23.4%      │  25.1%      │  22.8%      │
└─────────────┴─────────────┴─────────────┴─────────────┘
```
- Под каждым методом: время выполнения и процент покрытия маски.
- Автоматическая адаптация количества колонок (макс. 4).

### 2. Сравнение с метриками (`compare_methods_with_metrics()`)
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Original   │  Ground     │  Method A   │  Method B   │
│  Image      │  Truth      │             │             │
│             │             │  IoU:0.92   │  IoU:0.87   │
│             │             │  F1:0.94    │  F1:0.90    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```
- Отдельная панель для Ground Truth (если предоставлен).
- Под каждым методом: ключевые метрики (IoU, Dice, F1, Accuracy).

### 3. Бенчмарк-графики
- **`benchmark_time.png`**: Горизонтальный бар-чарт среднего времени с ошибками (std).
- **`benchmark_scatter.png`**: Scatter-plot: время выполнения (ось X) vs процент покрытия (ось Y).
- **`iou_vs_time.png`**: Соотношение точности (IoU) и скорости (только при наличии GT).
- **`methods_preview.jpg`**: Превью-галерея результатов, отсортированных по скорости.

### 4. Таблица метрик (`metrics_table.jpg`)
- Изображение со сводной таблицей: методы × метрики.
- Форматирование: 4 знака после запятой, цветовое выделение заголовков.

## 🔄 Логика warm-up и бенчмарка
```python
# Для каждого метода:
# 1. Проверка необходимости warm-up
if not self.warmup_completed[method_name]:
    self.warmup_utility.warmup_segmenter(...)  # Прогрев кэшей/памяти
    self.warmup_completed[method_name] = True

# 2. Многократные прогоны
times = []
for run in range(n_runs):
    if device == "cuda":
        torch.cuda.synchronize()  # Ждём завершения предыдущих операций
    t0 = time.perf_counter()
    result, mask = segmenter.segment_with_mask(image)
    if device == "cuda":
        torch.cuda.synchronize()  # Ждём завершения текущего ядра
    times.append(time.perf_counter() - t0)

# 3. Агрегация статистики
mean_time = np.mean(times)
std_time = np.std(times)
# Сохранение результата первого прогона для визуализации
```

> ⚠️ *Без `torch.cuda.synchronize()` замеры на GPU могут быть некорректными из-за асинхронного выполнения.*

## 🛠️ Требования к сегментерам
Любой класс, реализующий интерфейс:
```python
class BaseSegmenter:
    def segment_with_mask(self, image: ImageInput) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Args:
            image: Входное изображение (путь, np.ndarray или PIL.Image)
        Returns:
            result: np.ndarray формы (H, W) или (H, W, 3) — результат сегментации
            mask: Optional[np.ndarray] формы (H, W) — бинарная маска (0/255 или 0.0/1.0)
        """
        ...
```

Поддерживаются "из коробки":
- `OpenCVSegmenter`, `SklearnSegmenter`, `TorchSegmenter` из вашего проекта.
- Кастомные классы, наследующие `BaseSegmenter`.
- Нейросетевые модели с адаптером под интерфейс `segment_with_mask()`.

## ⚡ Рекомендации по использованию
- Для бенчмарка нейросетей всегда включайте `enable_warmup=True` и `n_runs ≥ 5` для стабильных замеров.
- При отсутствии GT-маски метрики качества не рассчитываются — используйте `compare_methods()` для быстрого визуального сравнения.
- Для экономии места на диске установите `save_results=False` в `benchmark_methods()`, если нужны только сводные данные.
- При сравнении >10 методов рассмотрите уменьшение `figsize` или увеличение `n_cols` в `compare_methods()` для компактности.

## 🤝 Зависимости
```text
torch, numpy, pandas, matplotlib, pillow, opencv-python
# Дополнительно для Excel-экспорта: openpyxl
# Дополнительно для ресайза масок: scikit-image
```

## 🔗 Сравнение с другими модулями тестирования
| Модуль | Цель | Данные | Методы | Вывод |
|--------|------|--------|--------|-------|
| `BatchClassicTester` | Консистентность реализаций | Только изображения | Классические (23 алгоритма) | Метрики согласия между библиотеками |
| `BatchClassicTester2` | Качество классических методов | Изображения + GT | Классические | Метрики качества (IoU, Dice vs GT) |
| `SegmentationBenchmark` | Сравнение нейросетей | Изображения + опционально GT | Нейросетевые (15+ архитектур) | mIoU, Acc, F1, per-class, time |
| **`SegmentationTester`** | **Универсальное тестирование** | **Изображение + опционально GT** | **Любые (через BaseSegmenter)** | **Время, метрики, визуализация, экспорт** |

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