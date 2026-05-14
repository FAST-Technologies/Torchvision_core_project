# 🤖 AutoSegmenter — Интеллектуальный селектор методов сегментации

## 📖 Описание
Модуль `segmenters/AutoSegmenter.py` предоставляет **универсальный маршрутизатор (facade)**, который автоматически подбирает оптимальный алгоритм сегментации для конкретного изображения. Вместо ручного перебора методов, `AutoSegmenter` анализирует входные данные, учитывает заданную цель (скорость/точность/память) и на основе встроенных бенчмарк-профилей выбирает лучший метод из 3 библиотек: OpenCV, scikit-learn, PyTorch.

> ⚠️ **Важно:** Данный модуль *не реализует алгоритмы сегментации напрямую*. Он выступает интеллектуальным диспетчером, делегируя выполнение в `OpenCVSegmenter`, `TorchSegmenter`, `NewTorchSegmenter` или `SklearnSegmenter`.

## ✨ Ключевые возможности

### 🎯 Целевая оптимизация
| Цель | Приоритет | Формула весов | Сценарий |
|------|-----------|---------------|----------|
| `SPEED` | Скорость | `time=0.7, acc=0.2, mem=0.1` | Обработка видео, real-time пайплайны |
| `ACCURACY` | Точность (IoU) | `time=0.2, acc=0.7, mem=0.1` | Медицинская диагностика, научные задачи |
| `BALANCED` | Баланс | `time=0.33, acc=0.34, mem=0.33` | Универсальные сценарии по умолчанию |
| `LOW_MEMORY` | Экономия памяти | `time=0.2, acc=0.3, mem=0.5` | Мобильные устройства, edge-девайсы |

### 🔍 Автоматический анализ изображения
| Характеристика | Метод вычисления | Влияние на выбор |
|----------------|------------------|------------------|
| `contrast` | `(max-min)/(max+ε)` | Высокий → пороговые методы, низкий → Sauvola/Phansalkar |
| `noise_level` | Локальная дисперсия (3×3) | Высокий → ×`robustness` штраф для чувствительных методов |
| `edge_density` | `cv2.Canny` (50, 150) | Высокий → watershed/region-growing, низкий → clustering |
| `complexity_score` | Нормализованная энтропия | Высокий → adaptive thresholds, низкий → global/Otsu |
| `estimated_type` | Эвристическая классификация | ×1.3 бонус для методов, оптимизированных под тип |

### 📊 Бенчмарк-профили (50+ методов)
Каждый метод в реестре содержит:
- `avg_time_ms`, `avg_iou`, `memory_mb` (усреднено по DIBCO, BSDS500, ISIC, INRIA)
- `best_for_type`: список доменов, где метод показывает лучшие результаты
- `robustness`: устойчивость к шуму `[0,1]`
- `parameter_sensitivity`: сложность подбора гиперпараметров
- `schema`: JSON-схема для динамической генерации UI-конфигураторов

## 🚀 Быстрый старт

### Автоматический режим (рекомендуется)
```python
from segmenters.AutoSegmenter import AutoSegmenter, SegmentationGoal
import cv2

image = cv2.cvtColor(cv2.imread("sample.jpg"), cv2.COLOR_BGR2RGB)

# Инициализация под точность
selector = AutoSegmenter(goal=SegmentationGoal.ACCURACY)

# Авто-выбор + сегментация
mask = selector.segment(image, auto_select=True)

# Авто-выбор + возврат метаданных
mask, meta = selector.segment(image, return_metadata=True)
print(f"Выбран: {meta['method']} ({meta['library']})")
print(f"Уверенность: {meta['confidence']:.2%}")
print(f"Тип изображения: {meta['image_characteristics'].estimated_type}")
```

### Ручной режим + рекомендации
```python
# Получение топ-3 рекомендаций
recs = selector.get_recommendations(image, top_k=3)
for r in recs:
    print(f"{r['rank']}. {r['method']} | IoU~{r['estimated_iou']:.2f} | {r['estimated_time_ms']:.0f}ms")

# Ручной вызов с валидацией
mask, meta = selector.segment(
    image,
    auto_select=False,
    method_name="chan_vese",
    library="torch",
    return_metadata=True
)
```

## ⚙️ Конфигурация

### Параметры инициализации `AutoSegmenter.__init__()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `goal` | `SegmentationGoal` | `BALANCED` | Целевая функция оптимизации выбора |
| `custom_weights` | `Dict[str, float]` | `None` | Множители скоринга: `{method_name: weight}`. `>1.0` повышает приоритет, `<1.0` понижает |
| `benchmark_data_path` | `str` | `None` | Путь к внешнему JSON/Pickle с профилями. При `None` используются встроенные данные |

### Параметры `segment(image, ...)`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `image` | `np.ndarray` | — | Входное изображение: RGB `(H,W,3)` или Grayscale `(H,W)` |
| `auto_select` | `bool` | `True` | Автоматический выбор метода. При `False` требуются `method_name` и `library` |
| `method_name` | `str` | `None` | Имя метода (только при `auto_select=False`) |
| `library` | `str` | `None` | Бэкенд: `"opencv"`, `"sklearn"`, `"torch"`, `"torch_v2"` |
| `return_metadata` | `bool` | `False` | Возвращать кортеж `(mask, metadata_dict)` вместо только маски |

### Формат возвращаемых метаданных
```python
{
    "method": "threshold_sauvola",
    "library": "torch",
    "parameters": {"window_size": 15, "k": 0.5, "r": 128},
    "confidence": 0.84,               # Нормализованная уверенность [0, 1]
    "image_characteristics": {
        "width": 512, "height": 512, "channels": 3,
        "mean_intensity": 142.3, "std_intensity": 45.1,
        "contrast": 0.92, "noise_level": 0.18,
        "edge_density": 0.04, "complexity_score": 0.71,
        "estimated_type": ImageType.DOCUMENT
    }
}
```

## 📚 Справочник методов

| Метод | Возвращает | Описание |
|-------|------------|----------|
| `analyze_image(image)` | `ImageCharacteristics` | Извлечение статистик, оценка типа изображения |
| `select_best_method(image, library)` | `(name, lib, params, confidence)` | Расчёт скоринга, выбор оптимального метода |
| `segment(image, ...)` | `np.ndarray` или `(mask, dict)` | Основной метод: маршрутизация + выполнение |
| `get_recommendations(image, top_k)` | `List[Dict]` | Топ-K методов с ожидаемыми метриками |
| `_calculate_method_score(...)` | `float` | Внутренний расчёт интегрального скора |
| `_get_segmenter_class(method, lib)` | `Type[BaseSegmenter]` | Фабрика классов сегментеров |

## 📊 Метрики и скоринг

### Формула интегрального скора
```python
score = w_time * time_score + w_acc * accuracy_score + w_mem * memory_score
```
где:
- `time_score = 1 - (profile.avg_time_ms / max_time)` (быстрее = лучше)
- `accuracy_score = profile.avg_iou`
- `memory_score = 1 - (profile.memory_mb / max_memory)`

### Модификаторы скора
| Условие | Множитель | Логика |
|---------|-----------|--------|
| `estimated_type ∈ best_for_type` | `×1.3` | Бонус за доменную специализацию |
| `noise_level > 0.3` | `× profile.robustness` | Штраф для неустойчивых методов на зашумлённых данных |
| `method_name ∈ custom_weights` | `× custom_weights[method]` | Ручная корректировка приоритетов |

### Расчёт уверенности (Confidence)
```python
z_score = (best_score - mean(all_scores)) / (std(all_scores) + 1e-6)
confidence = 1 / (1 + exp(-z_score))  # Сигмоида → [0, 1]
```
> 💡 `confidence > 0.75` означает высокую согласованность бенчмарков с характеристиками изображения.

## ⚡ Производительность и оптимизации

### Временная сложность анализа
| Операция | Сложность | Заметки |
|----------|-----------|---------|
| Статистики (mean, std, contrast) | `O(H·W)` | Векторизованные NumPy операции |
| Шум (local variance 3×3) | `O(H·W)` | `cv2.blur` на GPU/CPU |
| Плотность границ (Canny) | `O(H·W)` | Фиксированные пороги (50, 150) |
| Энтропия (histogram) | `O(H·W + 256)` | Быстрая свёртка в 1D |

### Рекомендации по использованию
1. **Production-пайплайны:** `auto_select=True, return_metadata=False` (минимум оверхеда)
2. **Отладка/исследование:** `return_metadata=True` для анализа причин выбора
3. **Кастомизация под датасет:** передайте `custom_weights={"my_fav_method": 1.5}` при инициализации
4. **Внешние бенчмарки:** укажите `benchmark_data_path="benchmarks/medical_2024.json"` для доменной калибровки

## 🛠️ Обработка ошибок и устойчивость

### Стратегия graceful degradation
```python
# При ошибке выполнения сегментера
try:
    mask = segmenter.segment_with_mask(image)
except Exception:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)  # Гарантированный возврат
```

### Валидация входных данных
- ✅ Проверка наличия метода в указанной библиотеке
- ✅ Авто-определение канала (RGB ↔ Grayscale)
- ✅ Fallback на пустую маску при критических ошибках
- ✅ Логирование выбора в консоль (`🤖 Auto-selected: ...`)

### Рекомендации по отладке
```python
import logging
logging.getLogger("segmenters.AutoSegmenter").setLevel(logging.DEBUG)

# Просмотр доступных методов
print(selector.get_available_methods("torch").keys())

# Проверка анализа изображения
chars = selector.analyze_image(image)
print(f"Тип: {chars.estimated_type}, Шум: {chars.noise_level:.3f}")
```

## 🤝 Интеграция с другими модулями проекта

| Модуль | Использование AutoSegmenter |
|--------|-----------------------------|
| `DatasetManager` | Автоматический подбор метода при загрузке семплов |
| `ADE20KDataset` | Валидация качества аугментаций через авто-селектор |
| `BatchClassicTester` | Генерация оптимальных конфигураций для пакетного тестирования |
| `TorchSegmenter` / `NewTorchSegmenter` | Маршрутизация `auto_select=True` → создание экземпляра с нужными `params` |
| `SegmentationMetrics` | Сравнение `confidence` селектора с реальными метриками IoU/Dice |

### Пример интеграции в DataLoader
```python
from torch.utils.data import DataLoader
from datasets.ADE20KDataset import ADE20KDataset
from segmenters.AutoSegmenter import AutoSegmenter

dataset = ADE20KDataset(root_dir="./data/ade20k", split="val")
loader = DataLoader(dataset, batch_size=4, shuffle=False)

selector = AutoSegmenter(goal=SegmentationGoal.BALANCED)

for batch in loader:
    images = batch["image"]  # [B, 3, H, W]
    masks = []
    for img_tensor in images:
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mask, meta = selector.segment(img_np, return_metadata=True)
        masks.append(mask)
    # ... evaluation / inference ...
```

## 📦 Зависимости

### Обязательные
```text
opencv-python>=4.5  # Анализ изображения, Canny, гистограммы
numpy>=1.20         # Математические операции, векторизация
dataclasses, enum   # Структуры данных, перечисления
```

### Опциональные (бэкенды сегментации)
```text
torch>=1.9.0        # Для library="torch" / "torch_v2"
scikit-learn>=1.0.0 # Для library="sklearn"
scipy>=1.7.0        # Для distance transform, cluster validation
```

### Установка
```bash
# Минимальный набор
pip install opencv-python numpy

# Полная установка
pip install torch scikit-learn scipy

# Проверка
python -c "from segmenters.AutoSegmenter import AutoSegmenter; print('✅ OK')"
```

## 📄 Лицензия

Проект распространяется под лицензией **MIT**. См. файл [LICENSE](LICENSE) для деталей.

```
MIT License

Copyright (c) 2026 Segmentation Project contributors

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

> 💡 **Совет:** Для кастомизации под специфичный датасет используйте `custom_weights` или загрузите собственные бенчмарки:
> ```python
> # Повышаем приоритет Sauvola для документов
> selector = AutoSegmenter(custom_weights={"threshold_sauvola": 1.4})
> 
> # Загрузка внешних профилей
> selector = AutoSegmenter(benchmark_data_path="configs/my_benchmarks.json")
> ```