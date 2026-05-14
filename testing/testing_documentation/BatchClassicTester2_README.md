# 🎯 BatchClassicTester2 — Валидация качества классических методов

## 📖 Описание
Модуль `BatchClassicTester2.py` предназначен для массового автоматизированного тестирования **качества** классических алгоритмов сегментации относительно Ground Truth на датасете **ADE20K**.

> ⚠️ **Важно:** Данный тестер проверяет *качество сегментации* относительно эталонных масок, а не согласованность между реализациями. Для проверки консистентности библиотек используйте `BatchClassicTester`.

## ✨ Ключевые возможности
- 🖼️ **Загрузка данных:** Автоматический ресайз изображений и масок, конвертация многоклассовых аннотаций ADE20K в бинарный формат.
- 🔧 **21 метод из 3 бэкендов:** OpenCV, Scikit-learn, PyTorch — единый интерфейс для всех реализаций.
- 📊 **Полный набор метрик:** IoU, Dice, Precision, Recall, F1-Score, Pixel Accuracy, MAE, Hausdorff Distance.
- 🚦 **Статистика ошибок:** Автоматический сбор и агрегация исключений по каждому методу.
- 💾 **Устойчивость к сбоям:** `resume=True`, автосохранение прогресса, обработка сигналов `SIGINT`/`SIGTERM`.
- 📈 **Визуализация:** Бар-чарты, scatter-плоты, тепловые карты метрик в `charts/`.
- 📄 **Экспорт отчётов:** CSV, JSON, Markdown с топ-10 методов и полной сводной таблицей.

## 🚀 Быстрый старт
```python
from testing.BatchClassicTester2 import BatchClassicTester

# Инициализация тестера
tester = BatchClassicTester(
    ade20k_root="./data/ade20k/ADEChallengeData2016",
    output_dir="./results/quality_test",
    split="validation",
    max_images=50,  # Лимит для быстрого теста
    image_size=(512, 512),
    autosave_interval=5,  # Автосохранение каждые 5 изображений
    resume=True  # Восстановление после прерывания
)

# Запуск тестирования
df_results = tester.run_batch_test()

# Сохранение и визуализация результатов
csv_path, json_path, md_path = tester.save_results(df_results)
tester.plot_results(df_results)
```

## ⚙️ Конфигурация
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `ade20k_root` | `str` | `./data/ade20k/...` | Путь к корневой директории датасета ADE20K |
| `output_dir` | `str` | `./data/batch_classic_test` | Директория для всех артефактов и отчётов |
| `split` | `str` | `"validation"` | Выборка: `"training"` или `"validation"` |
| `max_images` | `Optional[int]` | `None` | Лимит изображений (`None` = все доступные) |
| `image_size` | `Tuple[int, int]` | `(512, 512)` | Целевой размер для ресайза изображений и масок |
| `autosave_interval` | `int` | `5` | Интервал автосохранения прогресса (в итерациях) |
| `resume` | `bool` | `True` | Возобновлять тест с места остановки через `.progress.json` |

## 📂 Структура выходных данных
```
{output_dir}/
├── charts/                     # Аналитические графики (PNG)
│   ├── iou_ranking.png
│   ├── speed_vs_accuracy.png
│   └── metrics_heatmap.png
├── batch_test_results.csv      # Сводная таблица метрик
├── batch_test_details.json     # Детальные результаты + конфиг + ошибки
├── batch_test_report.md        # Человекочитаемый Markdown-отчёт
├── .progress.json              # (временный) метаданные прогресса
└── .results_temp.csv           # (временный) промежуточные результаты
```

## 📝 Тестируемые методы (21 алгоритм)
| Бэкенд | Пороговые методы | Граничные методы |
|--------|-----------------|------------------|
| **OpenCV** | `global`, `otsu`, `adaptive`, `niblack`, `sauvola` | `sobel`, `canny` |
| **Scikit-learn** | `global`, `otsu`, `adaptive`, `niblack`, `sauvola` | `sobel`, `canny` |
| **PyTorch** | `global`, `otsu`, `adaptive`, `niblack`, `sauvola` | `sobel`, `canny` |

> 💡 *Все методы инициализируются с дефолтными гиперпараметрами. Для кастомизации отредактируйте метод `_get_classic_methods()`.*

## 🔄 Конвертация масок ADE20K
Датасет ADE20K содержит многоклассовые аннотации. Модуль автоматически преобразует их в бинарный формат:

```python
# Стратегия: самый частый класс = фон (0), всё остальное = объект (255)
unique, counts = np.unique(mask, return_counts=True)
background_class = unique[np.argmax(counts)]
binary_mask = (mask != background_class).astype(np.uint8) * 255
```

> ⚠️ *Для задач, где важна сегментация конкретных классов, рассмотрите модификацию `_multiclass_to_binary()` или используйте специализированные датасеты.*

## 📊 Метрики качества
Для каждого метода рассчитываются:

| Метрика | Описание | Интерпретация |
|---------|----------|---------------|
| **IoU** | Intersection over Union | Чем выше, тем лучше (идеал = 1.0) |
| **Dice** | F1-Score для бинарной сегментации | Более устойчив к дисбалансу классов |
| **Precision / Recall** | Точность и полнота | Баланс между ложными срабатываниями и пропусками |
| **F1-Score** | Гармоническое среднее Precision и Recall | Универсальная метрика качества |
| **Pixel Accuracy** | Доля верно классифицированных пикселей | Может быть завышена при дисбалансе |
| **MAE** | Mean Absolute Error | Средняя абсолютная ошибка (чем ниже, тем лучше) |
| **Hausdorff Distance** | Максимальное расстояние между контурами | Чувствительна к выбросам, важна для границ |

## ⚡ Рекомендации по производительности
- Для быстрой отладки используйте `max_images=10` и `image_size=(256, 256)`.
- При `resume=True` тестер пропускает уже выполненные комбинации, но **не перезаписывает** агрегированные метрики — для полного пересчёта удалите `.progress.json` и `.results_temp.csv`.
- Память автоматически очищается после каждого изображения (`torch.cuda.empty_cache()`, `gc.collect()`).
- Для больших запусков (>100 изображений) увеличьте `autosave_interval` до 10–20, чтобы снизить нагрузку на диск.

## 🛠️ Обработка ошибок
- Исключения в `_run_single_test()` перехватываются, логируются в `self.errors` и **не прерывают** пакетное выполнение.
- Статистика ошибок включается в итоговые отчёты (`error_count`, `error_rate`).
- Для отладки конкретного метода временно установите `max_images=1` и проверьте логи.

## 🤝 Зависимости
```text
torch, numpy, pandas, matplotlib, seaborn, pillow, tqdm, scikit-image
(opencv-python, scikit-learn требуются для соответствующих сегментеров)
```

## 🔗 Сравнение с BatchClassicTester
| Критерий | BatchClassicTester | BatchClassicTester2 |
|----------|-------------------|---------------------|
| **Цель** | Консистентность реализаций | Качество относительно GT |
| **Данные** | Только изображения | Изображения + Ground Truth маски |
| **Сравнение** | Библиотека ↔ Библиотека | Предсказание ↔ Эталон |
| **Метрики** | Согласованность (между масками) | Качество (предсказание vs GT) |
| **Конвертация масок** | Не требуется | `_multiclass_to_binary()` для ADE20K |

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