# 🧱 BaseSegmenter — Абстрактный базовый класс для сегментаторов

## 📖 Описание
Модуль `BaseSegmenter.py` определяет **единый интерфейс** для всех алгоритмов сегментации в проекте. Это фундамент, на котором строятся:
- Классические методы: `OpenCVSegmenter`, `SklearnSegmenter`
- PyTorch-реализации: `TorchSegmenter`, `TorchSegmenter2`
- Высокопроизводительные бэкенды: `ONNXSegmenter`, `TRTSegmenter`

> ⚠️ **Важно:** `BaseSegmenter` — абстрактный класс. Его нельзя инстанцировать напрямую. Используйте конкретные реализации или создайте наследника для собственного алгоритма.

## ✨ Ключевые возможности
### 🎨 Поддержка форматов входных данных
| Тип | Пример | Обработка |
|-----|--------|-----------|
| `str` | `"image.jpg"`, `"https://..."` | Загрузка через `cv2.imread()` с авто-конвертацией BGR→RGB |
| `PIL.Image` | `Image.open("img.png")` | Конвертация в `np.array` с опцией `.convert("RGB")` или `.convert("L")` |
| `np.ndarray` | `np.random.randint(0,255,(512,512,3))` | Копирование + опциональная конвертация цветового пространства |
| `torch.Tensor` | `torch.rand(3,512,512)` | `.permute()`, `.cpu()`, конвертация float→uint8 при необходимости |

### 🔧 Абстрактные методы (требуют реализации)
```python
from segmenters.BaseSegmenter import BaseSegmenter, BinaryMask, ProbabilityMask
from typing import Optional, Tuple, Any

class MyCustomSegmenter(BaseSegmenter):
    def segment(self, image, **kwargs) -> BinaryMask:
        """Основная сегментация: возвращает бинарную маску {0, 255}."""
        # Ваша реализация...
        return binary_mask
    
    def segment_with_mask(self, image, **kwargs) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        """Сегментация с возвратом вероятностной маски (опционально)."""
        binary = self.segment(image, **kwargs)
        # Если метод поддерживает вероятности:
        # prob = ...  # float32 [0, 1]
        # return binary, prob
        return binary, None  # ProbabilityMask не поддерживается
```

### 🛠️ Готовые утилиты для наследников
| Метод | Назначение | Пример использования |
|-------|-----------|---------------------|
| `preprocess_image()` | Конвертация входов → `np.ndarray` | `img = self.preprocess_image(input, as_gray=True, target_size=(256,256))` |
| `visualize()` | Наложение маски на изображение с альфа-блендингом | `overlay = self.visualize(img, mask, alpha=0.6, overlay_color=(0,255,0))` |
| `evaluate_metrics()` | Расчёт метрик через `SegmentationMetrics` | `metrics = self.evaluate_metrics(pred_mask, gt_mask, threshold=0.5)` |
| `segment_and_evaluate()` | Комбинированный вызов "сегментация + оценка" | `metrics, mask = self.segment_and_evaluate(img, gt_mask)` |
| `_ensure_binary_mask()` | Приведение маски к формату `{0, 255}` | `binary = self._ensure_binary_mask(prob_mask, threshold=0.5)` |
| `get_info()` | Мета-информация о сегментере | `info = seg.get_info()  # {"name": "...", "class": "...", "module": "..."}` |

### 🔄 Гибкий вызов через `__call__`
```python
seg = MyCustomSegmenter()

# Простой вызов: только бинарная маска
mask = seg(image)

# Расширенный вызов: бинарная + вероятностная маска
binary, prob = seg(image, return_mask=True)

# С параметрами сегментации
mask = seg(image, threshold=0.7, window_size=21)
```

## 🚀 Быстрый старт
### Создание собственного сегментера
```python
import numpy as np
from segmenters.BaseSegmenter import BaseSegmenter, BinaryMask, ProbabilityMask
from typing import Optional, Tuple, Any

class SimpleThresholdSegmenter(BaseSegmenter):
    """Пример: пороговая сегментация в оттенках серого."""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
    
    def segment(self, image, **kwargs) -> BinaryMask:
        # Предобработка: grayscale + нормализация
        gray = self.preprocess_image(image, as_gray=True, normalize=True)
        # Пороговая бинаризация
        binary = (gray > self.threshold).astype(np.uint8) * 255
        return binary
    
    def segment_with_mask(self, image, **kwargs) -> Tuple[BinaryMask, Optional[ProbabilityMask]]:
        gray = self.preprocess_image(image, as_gray=True, normalize=True)
        # Возвращаем и бинарную, и вероятностную маску
        prob = gray  # [0, 1]
        binary = (prob > self.threshold).astype(np.uint8) * 255
        return binary, prob

# Использование
seg = SimpleThresholdSegmenter(threshold=0.6)
image = "test.jpg"

# Сегментация
mask = seg(image)

# Визуализация
img_np = seg.preprocess_image(image)  # (H,W,3), uint8
overlay = seg.visualize(img_np, mask, alpha=0.5)
overlay.save("result.png")

# Оценка качества (при наличии GT)
gt_mask = np.load("gt.npy")  # (H,W), uint8 {0,255}
metrics = seg.evaluate_metrics(mask, gt_mask)
print(f"IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}")
```

### Интеграция с тестерами проекта
```python
from testing.SegmentationTester import SegmentationTester

# Регистрация кастомного сегментера
tester = SegmentationTester()
tester.add_method("simple_threshold", SimpleThresholdSegmenter(threshold=0.6))

# Запуск сравнения с другими методами
results = tester.compare_methods(
    image="test.jpg",
    method_names=["simple_threshold", "otsu_cv2", "canny_torch"],
    test_name="custom_comparison"
)
```

## ⚙️ Конфигурация
### Параметры `preprocess_image()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `image` | `ImageInput` | — | Входное изображение любого поддерживаемого формата |
| `as_gray` | `bool` | `False` | Конвертировать ли в оттенки серого (1 канал) |
| `target_size` | `Optional[Tuple[int, int]]` | `None` | Целевой размер `(ширина, высота)` для ресайза |
| `normalize` | `bool` | `False` | Нормализовать ли значения в диапазон `[0, 1]` (dtype float32) |

### Параметры `visualize()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `image` | `NumpyImage` | — | Исходное изображение `(H, W, 3)`, RGB, uint8 |
| `mask` | `BinaryMask` | — | Бинарная маска `(H, W)`, uint8, {0, 255} |
| `alpha` | `float` | `0.5` | Прозрачность наложения: 0.0=только фото, 1.0=только маска |
| `overlay_color` | `OverlayColor` | `(255, 0, 0)` | Цвет объекта в формате `(R, G, B)` |
| `return_numpy` | `bool` | `False` | Возвращать ли `np.ndarray` вместо `PIL.Image` |

### Параметры `evaluate_metrics()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `pred_mask` | `BinaryMask` | — | Предсказанная маска |
| `gt_mask` | `BinaryMask` | — | Ground Truth маска |
| `threshold` | `float` | `0.5` | Порог бинаризации (если маски не в формате {0, 255}) |

## 📂 Форматы данных
### Входные изображения (`ImageInput`)
```python
# Union[str, np.ndarray, PIL.Image, torch.Tensor]

# 1. Путь к файлу
image = "data/test.jpg"

# 2. NumPy массив
image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)  # RGB
image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)     # Grayscale

# 3. PIL.Image
from PIL import Image
image = Image.open("test.jpg").convert("RGB")

# 4. PyTorch тензор
import torch
image = torch.rand(3, 512, 512)        # (C, H, W), float [0,1]
image = torch.randint(0, 256, (512, 512, 3), dtype=torch.uint8)  # (H, W, C)
```

### Выходные маски
```python
# BinaryMask: основная выходная форма
BinaryMask = np.ndarray  # Форма: (H, W), dtype: uint8, значения: {0, 255}

# Пример создания:
binary = np.zeros((512, 512), dtype=np.uint8)
binary[roi] = 255  # Объект = 255, фон = 0

# ProbabilityMask: опциональный дополнительный выход
ProbabilityMask = np.ndarray  # Форма: (H, W), dtype: float32, значения: [0.0, 1.0]

# Пример:
prob = model_output.sigmoid()  # [0, 1]
binary = (prob > 0.5).astype(np.uint8) * 255
```

### Метрики (`MetricsDict`)
```python
MetricsDict = Dict[str, float]
# Пример возврата из evaluate_metrics():
{
    "iou": 0.92,              # Intersection over Union
    "dice": 0.95,             # Dice coefficient
    "f1_score": 0.94,         # F1-Score
    "precision": 0.93,        # Precision
    "recall": 0.91,           # Recall
    "pixel_accuracy": 0.98,   # Pixel Accuracy
    "mae": 0.02,              # Mean Absolute Error
    "hausdorff_distance": 3.5,# Hausdorff Distance (пиксели)
    # ... и другие метрики из SegmentationMetrics
}
```

## 🎨 Визуализация: алгоритм наложения
Метод `visualize()` реализует альфа-блендинг:

```python
# 1. Создание цветной маски
colored_mask = np.zeros_like(image)  # (H, W, 3), uint8
colored_mask[mask > 0] = overlay_color  # (R, G, B) для объекта

# 2. Альфа-смешивание через OpenCV
result = cv2.addWeighted(
    image, 1 - alpha,      # Оригинальное изображение (вес: 1-α)
    colored_mask, alpha,   # Цветная маска (вес: α)
    0                      # Гамма-коррекция (не используется)
)
```

**Примеры визуализации:**
| `alpha` | Результат |
|---------|-----------|
| `0.0` | Только оригинальное изображение |
| `0.3` | Лёгкое наложение: объект подсвечен цветом |
| `0.5` | Баланс: видно и фото, и маску (по умолчанию) |
| `0.8` | Доминирует маска: фото как фон |
| `1.0` | Только цветная маска |

## 🔄 Конвейер предобработки изображений
Метод `preprocess_image()` выполняет последовательность преобразований:

```python
def preprocess_image(image, as_gray=False, target_size=None, normalize=False):
    # 1. Загрузка/конвертация в np.ndarray
    if isinstance(image, str):
        img = cv2.imread(image)  # BGR
        result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not as_gray else ...
    elif isinstance(image, PIL.Image):
        result = np.array(image.convert("RGB" if not as_gray else "L"))
    # ... обработка np.ndarray и torch.Tensor ...
    
    # 2. Опциональная конвертация в grayscale
    if as_gray and len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    
    # 3. Опциональный ресайз с адаптивной интерполяцией
    if target_size is not None:
        interpolation = (
            cv2.INTER_AREA  # Для уменьшения: антиалиасинг
            if result.shape[0]*result.shape[1] > target_size[0]*target_size[1]
            else cv2.INTER_LINEAR  # Для увеличения: билинейная
        )
        result = cv2.resize(result, target_size, interpolation=interpolation)
    
    # 4. Опциональная нормализация [0,255] → [0,1]
    if normalize:
        result = result.astype(np.float32) / 255.0
    
    return result
```

## 🛠️ Приведение масок к бинарному формату
Метод `_ensure_binary_mask()` обрабатывает различные форматы входных масок:

```python
def _ensure_binary_mask(mask, threshold=0.5) -> BinaryMask:
    # uint8 [0, 1] → умножение на 255
    if mask.dtype == np.uint8 and mask.max() == 1:
        return (mask * 255).astype(np.uint8)
    
    # uint8 [0, 255] → пороговая бинаризация
    elif mask.dtype == np.uint8 and mask.max() <= 255:
        return np.where(mask > threshold * 255, 255, 0).astype(np.uint8)
    
    # float [0, 1] → пороговая бинаризация
    elif mask.dtype in (np.float32, np.float64) and mask.max() <= 1.0:
        return np.where(mask > threshold, 255, 0).astype(np.uint8)
    
    # float произвольный диапазон → нормализация + бинаризация
    elif mask.dtype in (np.float32, np.float64):
        normalized = mask / mask.max()
        return np.where(normalized > threshold, 255, 0).astype(np.uint8)
    
    # Fallback
    return mask.astype(np.uint8)
```

## ⚡ Рекомендации по реализации наследников
1. **Всегда вызывайте `super().__init__()`** для корректной инициализации `self.name` и `self.metrics_calculator`.
2. **Возвращайте маску в формате `BinaryMask`**: `(H, W)`, `uint8`, `{0, 255}` — это требование всех тестеров проекта.
3. **Используйте `preprocess_image()`** для единообразной обработки входов — это упрощает отладку и тестирование.
4. **Для нейросетей**: переопределите `preprocess_image()` если нужна специфичная нормализация (например, ImageNet stats).
5. **Для методов с вероятностями**: реализуйте `segment_with_mask()` и возвращайте `ProbabilityMask` в диапазоне `[0, 1]`.
6. **Обработайте исключения**: при ошибке загрузки/обработки выбрасывайте `ValueError` или `TypeError` с информативным сообщением.

## 🤝 Зависимости
```text
numpy>=1.20
opencv-python>=4.5  # Для imread, cvtColor, resize, addWeighted
Pillow>=8.0         # Для работы с PIL.Image
torch>=1.9          # Опционально: для поддержки torch.Tensor входов
typing-extensions   # Для TypeAlias, Protocol, runtime_checkable
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование BaseSegmenter |
|--------|----------------------------|
| `OpenCVSegmenter`, `SklearnSegmenter`, `TorchSegmenter` | Наследование + реализация абстрактных методов |
| `BackendSegmenters` (ONNX/TRT) | Наследование + оптимизированный инференс |
| `SegmentationTester` | Регистрация через `add_method()` и тестирование |
| `BatchClassicTester` | Массовое тестирование согласованности реализаций |
| `TorchImplementationValidator` | Валидация PyTorch-реализаций против эталонов |
| `SegmentationMetrics` | Делегирование расчёта метрик через `evaluate_metrics()` |

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