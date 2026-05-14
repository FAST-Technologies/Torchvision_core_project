# 🎨 Palettes — Цветовые палитры и имена классов для датасетов сегментации

## 📖 Описание
Модуль `utils/palettes.py` предоставляет **централизованный реестр** цветовых палитр и человекочитаемых имён классов для популярных датасетов семантической сегментации.

> ⚠️ **Важно:** Данный модуль является **инфраструктурным** — он не выполняет сегментацию, а обеспечивает **консистентную визуализацию** и **интерпретацию результатов** для различных датасетов.

## ✨ Ключевые возможности
### 🗂️ Поддерживаемые датасеты

| Датасет | Классов | Палитра | Сценарии использования |
|---------|---------|---------|----------------------|
| **ADE20K** | 150 | `ade_palette()` | Универсальная семантическая сегментация, сцены помещений/улиц |
| **COCO** | 80 | `coco_palette()` | Детекция объектов, инстанс-сегментация, общие задачи |
| **Cityscapes** | 19 / 34 | `cityscapes_palette()` / `cityscapes_extended_palette()` | Автономное вождение, уличные сцены |
| **CheXpert** | 14 | `chexpert_observation_palette()` | Медицинская классификация рентгеновских снимков |
| **ISIC 2018** | 2 (binary) | `binary_palette()` | Сегментация кожных поражений, медицинская диагностика |

### 🎨 Особенности палитр
- **Уникальные цвета**: Каждый класс получает визуально различимый `[R, G, B]` цвет.
- **Совместимость с OpenCV/PIL**: Формат `[R, G, B]` для прямой интеграции с `cv2.cvtColor()` и `Image.fromarray()`.
- **Детерминированность**: Фиксированные значения для воспроизводимости визуализаций.
- **Расширяемость**: Легко добавить новую палитру через функцию `List[List[int]]`.

### 🏷️ Имена классов
- **Человекочитаемые**: `{0: "wall", 1: "building", ...}` вместо сырых индексов.
- **0-индексация**: Соответствует стандартам большинства фреймворков (PyTorch, TensorFlow).
- **Источники**: Ссылки на официальную документацию датасетов в docstring'ах.

## 🚀 Быстрый старт
### Базовое использование: визуализация маски ADE20K
```python
from utils.palettes import ade_palette, get_ade_class_names
import numpy as np
from PIL import Image

# Предсказанная маска (значения 0–149)
pred_mask = np.random.randint(0, 150, size=(512, 512), dtype=np.uint8)

# Получение палитры и имён классов
palette = ade_palette()  # List[List[int]], 150×[R,G,B]
class_names = get_ade_class_names()  # Dict[int, str]

# Создание цветной маски
color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
for class_id in range(150):
    color_mask[pred_mask == class_id] = palette[class_id]

# Сохранение результата
Image.fromarray(color_mask).save("colored_mask.png")

# Вывод информации о доминирующем классе
from collections import Counter
counts = Counter(pred_mask.flatten())
top_class = counts.most_common(1)[0][0]
print(f"Top class: {class_names[top_class]} ({counts[top_class]} pixels)")
```

### Переключение между датасетами
```python
from utils.palettes import (
    coco_palette, get_coco_class_names,
    cityscapes_palette, get_cityscapes_class_names
)

# Для COCO
coco_pal = coco_palette()  # 80 цветов
coco_names = get_coco_class_names()  # {0: "person", 1: "bicycle", ...}

# Для Cityscapes (стандартные 19 классов)
cs_pal = cityscapes_palette()  # 19 цветов
cs_names = get_cityscapes_class_names()  # {0: "road", 1: "sidewalk", ...}

# Для расширенного Cityscapes (34 класса)
cs_ext_pal = cityscapes_extended_palette()
cs_ext_names = get_cityscapes_extended_class_names()
```

### Бинарная сегментация (медицинские задачи)
```python
from utils.palettes import binary_palette, get_isic_class_names

# ISIC 2018: фон / поражение
palette = binary_palette()  # [[120,120,120], [180,120,120]]
names = get_isic_class_names()  # {0: "background", 1: "lesion"}

# Визуализация: серый фон, розовое поражение
color_mask = np.array(palette)[binary_mask]  # binary_mask ∈ {0, 1}
```

### Интеграция с NeuralSegmenter
```python
from segmenters.NeuralSegmenter import NeuralSegmenter
from utils.palettes import ade_palette, get_ade_class_names

# Автоматическое применение палитры при инференсе
segmenter = NeuralSegmenter(
    model_type="segformer",
    model_name="nvidia/segformer-b5-finetuned-ade-640-640",
    palette=ade_palette()  # Передаётся в segment_image_unified()
)

# Визуализация с правильными цветами
overlay = segmenter.segment_image("scene.jpg")
overlay.save("result.png")

# Анализ с именами классов
_, info = segmenter.predict_segmentation_map(
    "scene.jpg",
    class_names=get_ade_class_names(),
    verbose=True
)
```

## ⚙️ Конфигурация
### Функции получения имён классов
| Функция | Возвращает | Описание |
|---------|-----------|----------|
| `get_ade_class_names()` | `Dict[int, str]` | 150 классов ADE20K (0: "wall" → 149: "flag") |
| `get_coco_class_names()` | `Dict[int, str]` | 80 классов COCO (0: "person" → 79: "toothbrush") |
| `get_cityscapes_class_names()` | `Dict[int, str]` | 19 стандартных классов Cityscapes |
| `get_cityscapes_extended_class_names()` | `Dict[int, str]` | 34 расширенных класса (включая "ego vehicle", "rectification border") |
| `get_chexpert_observation_class_names()` | `Dict[int, str]` | 14 медицинских наблюдений (CheXpert) |
| `get_isic_class_names()` | `Dict[int, str]` | 2 класса: фон / поражение (ISIC 2018) |

### Функции получения палитр
| Функция | Возвращает | Описание |
|---------|-----------|----------|
| `ade_palette()` | `List[List[int]]` | 150×[R,G,B] для ADE20K |
| `coco_palette()` | `List[List[int]]` | 80×[R,G,B] для COCO |
| `cityscapes_palette()` | `List[List[int]]` | 19×[R,G,B] для Cityscapes |
| `cityscapes_extended_palette()` | `List[List[int]]` | 34×[R,G,B] для расширенного Cityscapes |
| `chexpert_observation_palette()` | `List[List[int]]` | 14×[R,G,B] для CheXpert |
| `binary_palette()` | `List[List[int]]` | 2×[R,G,B]: серый фон, розовый объект |

### Формат возвращаемых данных
```python
# Палитра: список списков [R, G, B], значения 0–255
palette: List[List[int]] = [
    [120, 120, 120],  # Класс 0
    [180, 120, 120],  # Класс 1
    # ...
]

# Имена классов: словарь {индекс: имя}
class_names: Dict[int, str] = {
    0: "wall",
    1: "building",
    # ...
}
```

## 📚 Справочник функций
### 🔹 Получение имён классов
| Функция | Параметры | Возвращает | Пример использования |
|---------|-----------|-----------|---------------------|
| `get_ade_class_names()` | — | `Dict[int, str]` | `names = get_ade_class_names(); print(names[0])  # "wall"` |
| `get_coco_class_names()` | — | `Dict[int, str]` | `names = get_coco_class_names(); print(names[12])  # "person"` |
| `get_cityscapes_class_names()` | — | `Dict[int, str]` | `names = get_cityscapes_class_names(); print(names[13])  # "car"` |
| `get_cityscapes_extended_class_names()` | — | `Dict[int, str]` | `names = get_cityscapes_extended_class_names(); print(names[32])  # "ego vehicle"` |
| `get_chexpert_observation_class_names()` | — | `Dict[int, str]` | `names = get_chexpert_observation_class_names(); print(names[5])  # "Edema"` |
| `get_isic_class_names()` | — | `Dict[int, str]` | `names = get_isic_class_names(); print(names[1])  # "lesion"` |

### 🔹 Получение палитр
| Функция | Параметры | Возвращает | Пример использования |
|---------|-----------|-----------|---------------------|
| `ade_palette()` | — | `List[List[int]]` | `pal = ade_palette(); color = pal[42]  # [255, 7, 71]` |
| `coco_palette()` | — | `List[List[int]]` | `pal = coco_palette(); color = pal[0]  # [120, 120, 120]` |
| `cityscapes_palette()` | — | `List[List[int]]` | `pal = cityscapes_palette(); color = pal[10]  # [143, 255, 140]` |
| `cityscapes_extended_palette()` | — | `List[List[int]]` | `pal = cityscapes_extended_palette(); color = pal[33]  # [10, 255, 71]` |
| `chexpert_observation_palette()` | — | `List[List[int]]` | `pal = chexpert_observation_palette(); color = pal[0]  # [120, 120, 120]` |
| `binary_palette()` | — | `List[List[int]]` | `pal = binary_palette(); color = pal[1]  # [180, 120, 120]` |

## 🔄 Конвейер визуализации: маска → палитра → цветное изображение
### Векторизованное применение палитры
```python
def apply_palette(mask: np.ndarray, palette: List[List[int]]) -> np.ndarray:
    """Применение палитры к маске сегментации."""
    # mask: (H, W), значения 0..N-1
    # palette: List[List[int]], длина >= N
    palette_array = np.array(palette, dtype=np.uint8)  # (N, 3)
    
    # Векторизованная индексация: (H, W) → (H, W, 3)
    color_mask = palette_array[mask]
    
    return color_mask  # (H, W, 3), dtype=uint8

# Использование
mask = np.random.randint(0, 150, (512, 512), dtype=np.uint8)
color_mask = apply_palette(mask, ade_palette())
Image.fromarray(color_mask).save("result.png")
```

### Наложение на оригинальное изображение
```python
def create_overlay(original: np.ndarray, mask: np.ndarray, 
                   palette: List[List[int]], alpha: float = 0.5) -> np.ndarray:
    """Создание overlay: оригинал + цветная маска."""
    color_mask = apply_palette(mask, palette)
    
    # Блендинг
    overlay = (original * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    
    return overlay

# Использование
original = np.array(Image.open("scene.jpg"))  # (H, W, 3)
mask = segmenter.segment("scene.jpg")  # (H, W)
overlay = create_overlay(original, mask, ade_palette(), alpha=0.6)
Image.fromarray(overlay).save("overlay.png")
```

### Анализ распределения классов
```python
from collections import Counter
from utils.palettes import get_ade_class_names

def analyze_mask(mask: np.ndarray, class_names: Dict[int, str], 
                 ignore_index: int = 255) -> None:
    """Вывод статистики по классам в маске."""
    # Фильтрация игнорируемых пикселей
    valid = mask != ignore_index
    flat = mask[valid]
    
    # Подсчёт
    counts = Counter(flat)
    total = len(flat)
    
    print(f"Valid pixels: {total:,}")
    print(f"Unique classes: {len(counts)}\n")
    
    # Топ-10 классов
    for class_id, count in counts.most_common(10):
        name = class_names.get(class_id, f"Class_{class_id}")
        pct = 100 * count / total
        print(f"{class_id:3d}: {name:25s} {count:7,} px ({pct:5.2f}%)")

# Использование
mask = segmenter.segment("scene.jpg")
analyze_mask(mask, get_ade_class_names())
```

## 📊 Интерпретация цветов и классов
### ADE20K: примеры соответствий
| Класс | Имя | Цвет [R,G,B] | Визуальный ориентир |
|-------|-----|--------------|-------------------|
| 0 | `wall` | [120, 120, 120] | Серый |
| 2 | `sky` | [6, 230, 230] | Бирюзовый |
| 4 | `tree` | [4, 200, 3] | Зелёный |
| 12 | `person` | [150, 5, 61] | Тёмно-красный |
| 20 | `car` | [0, 102, 200] | Синий |
| 149 | `flag` | [92, 0, 255] | Фиолетовый |

### Cityscapes: стандартные 19 классов
| Класс | Имя | Сценарий |
|-------|-----|----------|
| 0 | `road` | Дорожное полотно |
| 1 | `sidewalk` | Тротуар |
| 13 | `car` | Легковые автомобили |
| 14 | `truck` | Грузовики |
| 17 | `motorcycle` | Мотоциклы |
| 18 | `bicycle` | Велосипеды |

### Медицинские датасеты: бинарная логика
```python
# ISIC 2018
{
    0: "background",  # Здоровая кожа → серый [120, 120, 120]
    1: "lesion"       # Поражение → розовый [180, 120, 120]
}

# CheXpert (классификация)
{
    0: "No Finding",
    5: "Edema",        # Отёк лёгких
    7: "Pneumonia",    # Пневмония
    10: "Pleural Effusion"  # Плевральный выпот
}
```

## ⚡ Производительность и оптимизации
### Векторизация против циклов
```python
# ✅ Векторизованно (быстро):
palette_array = np.array(palette, dtype=np.uint8)
color_mask = palette_array[mask]  # O(1) на пиксель

# ❌ Циклом (медленно):
color_mask = np.zeros((H, W, 3), dtype=np.uint8)
for i in range(H):
    for j in range(W):
        color_mask[i, j] = palette[mask[i, j]]  # O(H×W) Python-итераций
```

### Кэширование палитр
```python
from functools import lru_cache

@lru_cache(maxsize=8)
def get_cached_palette(dataset: str) -> Tuple[List[List[int]], Dict[int, str]]:
    """Кэширование палитр для повторного использования."""
    if dataset == "ade20k":
        return tuple(map(tuple, ade_palette())), get_ade_class_names()
    elif dataset == "coco":
        return tuple(map(tuple, coco_palette())), get_coco_class_names()
    # ... другие датасеты
```

### Память для больших изображений
```python
# Для изображений >4000×4000 рассмотрите chunked-обработку:
def apply_palette_chunked(mask: np.ndarray, palette: List[List[int]], 
                          chunk_size: int = 1024) -> np.ndarray:
    """По-пиксельная обработка с контролем памяти."""
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    palette_array = np.array(palette, dtype=np.uint8)
    
    for i in range(0, H, chunk_size):
        for j in range(0, W, chunk_size):
            chunk = mask[i:i+chunk_size, j:j+chunk_size]
            color_mask[i:i+chunk_size, j:j+chunk_size] = palette_array[chunk]
    
    return color_mask
```

## 🛠️ Обработка ошибок и устойчивость
### Валидация индексов классов
```python
def safe_apply_palette(mask: np.ndarray, palette: List[List[int]], 
                       ignore_index: int = 255) -> np.ndarray:
    """Безопасное применение палитры с обработкой out-of-range индексов."""
    palette_array = np.array(palette, dtype=np.uint8)
    n_classes = len(palette_array)
    
    # Маска валидных индексов
    valid = (mask >= 0) & (mask < n_classes) & (mask != ignore_index)
    
    # Инициализация результата (чёрный для невалидных)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Применение только к валидным пикселям
    color_mask[valid] = palette_array[mask[valid]]
    
    return color_mask
```

### Логирование загрузки классов
```python
# В каждой функции получения имён классов:
logger.info(f"✅ ADE20K classes loaded: {len(ade20k_class_names)} classes")
logger.info(f"   Range: [{min(keys)}..{max(keys)}]")

# При отладке:
import logging
logging.getLogger("utils.palettes").setLevel(logging.DEBUG)
```

### Fallback для неизвестных классов
```python
def get_class_name_safe(class_names: Dict[int, str], class_id: int, 
                        fallback: str = "unknown") -> str:
    """Безопасное получение имени класса."""
    return class_names.get(class_id, f"{fallback}_{class_id}")

# Использование
name = get_class_name_safe(get_ade_class_names(), 999)  # "unknown_999"
```

### Рекомендации по отладке
1. **Проверьте диапазон значений маски**:
   ```python
   print(f"Mask range: [{mask.min()}, {mask.max()}]")
   print(f"Expected: [0, {len(palette)-1}]")
   ```

2. **Убедитесь в совпадении размеров палитры и количества классов**:
   ```python
   assert len(palette) >= mask.max() + 1, "Palette too short!"
   ```

3. **Визуализируйте палитру для проверки**:
   ```python
   import matplotlib.pyplot as plt
   pal = np.array(ade_palette())
   plt.imshow(pal.reshape(1, -1, 3), aspect='auto')
   plt.axis('off')
   plt.title("ADE20K Palette")
   plt.show()
   ```

## 🤝 Зависимости
```text
numpy>=1.20          # Массивы, векторизация
typing-extensions    # Type hints (опционально для Python <3.9)
```

### Опциональные зависимости для визуализации
```bash
# Для отображения палитр
pip install matplotlib

# Для работы с изображениями
pip install Pillow opencv-python
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование palettes.py |
|--------|--------------------------|
| `NeuralSegmenter` | Передача `palette` и `class_names` в `segment_image_unified()` |
| `utils.strategies` | Визуализация overlay через `_create_overlay_standalone()` |
| `utils.utils` | Анализ предсказаний через `analyze_prediction()` с именами классов |
| `BatchClassicTester2` | Сравнение визуализаций классических и нейросетевых методов |
| `SegmentationTester` | Универсальное тестирование с консистентной цветовой схемой |

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
