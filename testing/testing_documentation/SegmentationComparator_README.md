# 🔍 SegmentationComparator — Сравнительный тестировщик сегментаторов

## 📖 Описание
Модуль `SegmentationComparator.py` предназначен для автоматизированного **сравнения** различных реализаций алгоритмов сегментации между собой. Он позволяет оценить, насколько близки результаты разных методов на одном и том же изображении, и визуализировать расхождения.

> ⚠️ **Важно:** Данный модуль сравнивает *методы между собой*, а не предсказания с Ground Truth. Для валидации качества относительно эталонных масок используйте `BatchClassicTester2`.

## ✨ Ключевые возможности
- 🔁 **Три режима сравнения:**
  | Режим | Описание | Когда использовать |
  |-------|----------|------------------|
  | `compare_methods()` | Попарное сравнение двух методов | Быстрая проверка новой реализации против референса |
  | `batch_comparison()` | Множество методов vs один референс | Подбор лучшего метода для задачи |
  | `matrix_comparison()` | Все-со-всеми / Все-против-референса / Уникальные пары | Полный анализ согласованности семейства алгоритмов |

- 📊 **Полный набор метрик:**
  - Качество: `IoU (Jaccard)`, `Dice`, `F1-Score`, `Precision`, `Recall`, `Accuracy`
  - Площади: `area1`, `area2`, `area_difference`, `area_ratio`
  - Граничные: `Hausdorff Distance`
  - Конфузионная матрица: `TP`, `FP`, `FN`, `TN`

- 🎨 **Визуализация 2×4 grid:**
  ```
  [Original]  [Mask A]    [Mask B]    [Difference]
  [Overlay A] [Overlay B] [Combined]  [Metrics Text]
  ```
  - Цветовое кодирование: 🔴 красный = метод 1, 🟢 зелёный = метод 2, 🟡 жёлтый = пересечение.
  - Heatmap разницы: цветовая схема "hot" (чёрный → красный → жёлтый → белый).

- 📤 **Экспорт в 5 форматах:**
  - `PNG`: Визуализации сравнения, маски, оверлеи, heatmaps.
  - `CSV`: Таблицы с метриками для дальнейшего анализа в pandas/Excel.
  - `JSON`: Детальные результаты (при необходимости).
  - `HTML`: Интерактивный отчёт с навигацией, превью и ссылками.
  - `Heatmaps`: Матрицы сравнения для каждой метрики (RdYlGn colormap).

## 🚀 Быстрый старт
### Попарное сравнение двух методов
```python
from testing.SegmentationComparator import SegmentationComparator
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.TorchSegmenter import TorchSegmenter
import numpy as np
from PIL import Image

# Загрузка изображения
image = np.array(Image.open("test.jpg").convert("RGB"))

# Инициализация компаратора
comparator = SegmentationComparator()

# Создание сегментеров
seg_cv2 = OpenCVSegmenter("otsu_thresholding")
seg_torch = TorchSegmenter("otsu_thresholding")

# Запуск сравнения
result = comparator.compare_methods(
    image=image,
    segmenter1=seg_cv2,
    segmenter2=seg_torch,
    name1="OpenCV_Otsu",
    name2="Torch_Otsu",
    save_comparison=True,
    output_path="results/otsu_comparison.png"
)

# Доступ к метрикам
print(f"IoU: {result['metrics']['jaccard']:.3f}")
print(f"Dice: {result['metrics']['dice_coefficient']:.3f}")
print(f"Δt: {abs(result['info1']['execution_time'] - result['info2']['execution_time']):.3f}s")
```

### Пакетное сравнение с референсом
```python
# Конфигурация тестируемых методов
methods = [
    {"name": "Otsu_CV2", "segmenter": OpenCVSegmenter("otsu_thresholding")},
    {"name": "Adaptive_CV2", "segmenter": OpenCVSegmenter("adaptive_thresholding", block_size=11)},
    {"name": "Sauvola_Sklearn", "segmenter": SklearnSegmenter("threshold_sauvola", window_size=15)},
]

# Референсный метод (например, ручная разметка или эталонный алгоритм)
reference = OpenCVSegmenter("canny_edge", low=0.1, high=0.3)

# Запуск пакетного сравнения
df_results = comparator.batch_comparison(
    image=image,
    methods_config=methods,
    reference_segmenter=reference,
    reference_name="Canny_Reference",
    save_results=True,
    output_dir="./results/batch_comparison"
)

# Анализ результатов
print(df_results.sort_values("f1_score", ascending=False)[["method", "f1_score", "jaccard", "test_time"]])
```

### Матричное сравнение "все-со-всеми"
```python
# Запуск матричного сравнения
matrix_result = comparator.matrix_comparison(
    image=image,
    methods_config=methods + [{"name": "Canny_Ref", "segmenter": reference}],
    comparison_type="all_vs_all",  # или "all_vs_ref", "pairwise"
    reference_method="Canny_Ref",  # только для all_vs_ref
    save_results=True,
    output_dir="./results/matrix_comparison"
)

# Доступ к данным
df_comparisons = matrix_result["df_comparisons"]  # попарные метрики
masks = matrix_result["masks"]  # {method_name: mask_array}
times = matrix_result["execution_times"]  # {method_name: time_seconds}

# Пример: найти наиболее согласованную пару по IoU
best_pair = df_comparisons.loc[df_comparisons["jaccard"].idxmax()]
print(f"Лучшая пара: {best_pair['method1']} ↔ {best_pair['method2']}, IoU={best_pair['jaccard']:.3f}")
```

## ⚙️ Конфигурация
### Параметры `compare_methods()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `image` | `np.ndarray` | — | Входное изображение `(H, W)` или `(H, W, 3)`, dtype=uint8 |
| `segmenter1`, `segmenter2` | `SegmenterLike` | — | Объекты с методом `.segment(image) -> np.ndarray` |
| `name1`, `name2` | `Optional[str]` | `None` | Человекочитаемые имена методов (автоопределение из `.method`) |
| `save_comparison` | `bool` | `True` | Сохранять ли визуализацию 2×4 grid |
| `output_path` | `Optional[str]` | `None` | Путь для сохранения изображения сравнения |

### Параметры `batch_comparison()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `methods_config` | `List[Dict]` | — | Список `{"name": str, "segmenter": obj}` |
| `reference_segmenter` | `SegmenterLike` | — | Референсный метод для сравнения |
| `reference_name` | `Optional[str]` | `None` | Имя референса для отчётов |
| `save_results` | `bool` | `True` | Сохранять ли визуализации и CSV |
| `output_dir` | `PathLike` | `"./data/comparison_results"` | Директория для артефактов |

### Параметры `matrix_comparison()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `comparison_type` | `str` | `"all_vs_all"` | Режим: `"all_vs_all"`, `"all_vs_ref"`, `"pairwise"` |
| `reference_method` | `Optional[str]` | `None` | Имя референса для режима `all_vs_ref` |
| `save_results` | `bool` | `True` | Сохранять ли heatmaps, маски, HTML-отчёт |

## 📂 Структура выходных данных
### Для `batch_comparison()`
```
{output_dir}/
├── comparison_{method_name}.jpg   # Визуализация 2×4 для каждого метода
├── comparison_results.csv         # Сводная таблица метрик
└── comparison_summary.jpg         # 2×2 grid со сводными графиками
```

### Для `matrix_comparison()`
```
{output_dir}/comparison_YYYYMMDD_HHMMSS/
├── comparisons.csv                # Все попарные сравнения (длинный формат)
├── summary_vs_ref.csv            # Сводка по сравнению с референсом (если all_vs_ref)
├── {metric}_matrix.png           # Heatmap для каждой метрики (IoU, Dice, F1...)
├── all_masks.png                 # Grid со всеми сгенерированными масками
├── report.html                   # Интерактивный HTML-отчёт
├── masks/                        # Отдельные PNG-маски для каждого метода
│   ├── {method}_mask.png
│   └── ...
└── images/                       # Оригиналы и оверлеи
    ├── original.png
    ├── {method}_overlay.png
    └── ...
```

## 📊 Метрики сравнения
Для каждой пары масок рассчитываются:

| Категория | Метрика | Описание | Диапазон | Интерпретация |
|-----------|---------|----------|----------|---------------|
| **Качество** | `jaccard` / `iou` | Intersection over Union | [0, 1] | Чем выше, тем лучше |
| | `dice_coefficient` | Dice / Sørensen coefficient | [0, 1] | Более устойчив к дисбалансу |
| | `f1_score` | Гармоническое среднее Precision/Recall | [0, 1] | Универсальная метрика |
| | `precision` / `recall` | Точность / Полнота | [0, 1] | Баланс ложных срабатываний/пропусков |
| | `accuracy` | Доля совпадающих пикселей | [0, 1] | Может быть завышена при дисбалансе |
| **Площади** | `area1` / `area2` | Количество пикселей объекта | [0, H×W] | Для анализа масштаба сегментации |
| | `area_difference` | \|area1 − area2\| | [0, ∞) | Чем меньше, тем согласованнее |
| | `area_ratio` | area1 / area2 | [0, ∞) | ~1.0 = согласованные площади |
| **Граничные** | `hausdorff_distance` | Макс. расстояние между контурами | [0, ∞) | Чувствительна к выбросам, важна для границ |
| **Конфузия** | `TP`/`FP`/`FN`/`TN` | Элементы матрицы ошибок | [0, ∞) | Для детального анализа типов ошибок |

## 🎨 Типы визуализаций
### 1. Попарное сравнение (2×4 grid)
```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Original   │  Mask A     │  Mask B     │  Difference │
│  Image      │  (IoU:0.92) │             │  (hot cmap) │
├─────────────┼─────────────┼─────────────┼─────────────┤
│  Overlay A  │  Overlay B  │  Combined   │  Metrics    │
│  (red)      │  (green)    │  (yellow=∩) │  (text)     │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### 2. Сводные графики пакетного сравнения (2×2)
- **Bar-чарт средних метрик**: Accuracy, Precision, Recall, F1, IoU.
- **Время выполнения**: Тестовые методы vs референс.
- **Площади масок**: Сравнение масштаба сегментации.
- **Корреляционная матрица**: Зависимости между метриками.

### 3. Матричные heatmaps
- Одна матрица `N×N` на каждую метрику (IoU, Dice, F1...).
- Цветовая схема `RdYlGn`: 🔴 низкое согласие → 🟡 среднее → 🟢 высокое.
- Аннотации с числовыми значениями (`.2f`).

### 4. HTML-отчёт
- Адаптивный дизайн с карточками метрик.
- Топ-5 методов по F1 (для режима `all_vs_ref`).
- Превью heatmaps и масок с прямыми ссылками.
- Статистика покрытия и времени выполнения.

## 🔄 Режимы матричного сравнения
| Режим | Формула пар | Количество сравнений | Пример использования |
|-------|-------------|---------------------|---------------------|
| `all_vs_all` | `(m1, m2)` для всех `m1, m2 ∈ M` | `N²` | Полный анализ согласованности семейства методов |
| `all_vs_ref` | `(ref, m)` для всех `m ∈ M \ {ref}` | `N−1` | Валидация новых реализаций против эталона |
| `pairwise` | Уникальные комбинации `C(N, 2)` | `N(N−1)/2` | Экономный режим для большого числа методов |

> 💡 *Для `N=10` методов: `all_vs_all` = 100 сравнений, `pairwise` = 45, `all_vs_ref` = 9.*

## 🛠️ Требования к сегментерам
Любой объект, реализующий интерфейс:
```python
class SegmenterLike:
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: np.ndarray формы (H, W) или (H, W, 3), dtype=uint8
        Returns:
            mask: np.ndarray формы (H, W), dtype=uint8/bool, значения {0, 255} или {0, 1}
        """
        ...
```

Поддерживаются "из коробки":
- `OpenCVSegmenter`, `SklearnSegmenter`, `TorchSegmenter` из вашего проекта.
- Кастомные классы с методом `.segment()`.
- Lambda-функции и обёртки (при соблюдении сигнатуры).

## ⚡ Управление памятью и производительность
- Замеры времени выполняются через `time.perf_counter()` с `torch.cuda.synchronize()` для корректных GPU-замеров.
- Исключения в отдельных методах перехватываются и логируются — пакетное выполнение продолжается.
- Для экономии памяти при матричном сравнении >20 методов рассмотрите режим `pairwise`.

## 🤝 Зависимости
```text
torch, numpy, pandas, matplotlib, pillow
# SegmentationMetrics должен быть доступен в проекте
```

## 🔗 Сравнение с другими модулями тестирования
| Модуль | Цель | Сравнение | Данные | Вывод |
|--------|------|-----------|--------|-------|
| `BatchClassicTester` | Консистентность реализаций | Библиотека ↔ Библиотека | Только изображения | Метрики согласия между масками |
| `BatchClassicTester2` | Качество классических методов | Предсказание ↔ Ground Truth | Изображения + GT | Метрики качества (IoU, Dice vs GT) |
| **`SegmentationComparator`** | **Сравнение методов между собой** | **Метод ↔ Метод** | **Только изображения** | **Метрики сходства + визуализация + heatmaps** |
| `SegmentationBenchmark` | Сравнение нейросетей | Архитектура ↔ Архитектура | Изображения + опционально GT | mIoU, Acc, F1, per-class, time |

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