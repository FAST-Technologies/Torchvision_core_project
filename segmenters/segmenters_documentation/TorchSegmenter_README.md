# ⚡ TorchSegmenter — Классические методы сегментации на чистом PyTorch

## 📖 Описание
Модуль `segmenters/TorchSegmenter.py` предоставляет **универсальный интерфейс** для 50+ классических алгоритмов сегментации изображений, реализованных на чистом PyTorch без зависимости от OpenCV, scikit-learn или scikit-image для основной логики.

> ⚠️ **Важно:** Данный модуль реализует *классические (не нейросетевые)* методы. Для глубокого обучения используйте `NewTorchSegmenter` (с поддержкой AMP, torch.compile) или специализированные архитектуры (`DeepLabV3Plus`, `UNet`).

## ✨ Ключевые возможности

### 🗂️ 8 категорий методов (50+ алгоритмов)

| Категория | Методов | Примеры | Сценарии использования |
|-----------|---------|---------|----------------------|
| **Пороговые** (14) | 14 | `global`, `otsu`, `adaptive`, `niblack`, `sauvola`, `phansalkar`, `bernsen`, `kittler_illingworth`, `entropy_kapur`, `triangle`, `multi_otsu`, `percentile`, `local_contrast` | Документы, сканы, изображения с неравномерным освещением |
| **Детекторы границ** (10) | 10 | `canny`, `sobel`, `prewitt`, `scharr`, `roberts`, `log`, `dog`, `marr_hildreth`, `gradient_magnitude`, `phase_congruency` | Выделение контуров, предобработка для watershed, детекция объектов |
| **Региональные** (3) | 3 | `region_growing`, `split_and_merge`, `floodfill` | Сегментация однородных областей, интерактивное выделение |
| **Кластеризация** (3) | 3 | `kmeans`, `dbscan`, `meanshift` | Цветовая сегментация, группировка пикселей по признакам |
| **Активные контуры** (4) | 4 | `active_contour`, `gvf_contour`, `morphological_snakes`, `chan_vese` | Медицинские изображения, объекты с размытыми границами |
| **Watershed/графовые** (2) | 2 | `watershed`, `random_walker` | Разделение слипшихся объектов, сегментация по маркерам |
| **Суперпиксели** (3) | 3 | `slic`, `felzenszwalb`, `quickshift` | Предварительная сегментация, упрощение изображения |
| **Интерактивные** (1) | 1 | `grabcut` | Полуавтоматическое выделение объекта по прямоугольнику |

### 🔄 Единый интерфейс для всех методов
```python
from segmenters.TorchSegmenter import TorchSegmenter

# Инициализация с методом и параметрами
segmenter = TorchSegmenter(
    method="threshold_sauvola",  # Название метода
    device="cuda",               # "cuda" или "cpu" (авто-детекция)
    window_size=15,              # Параметр 1
    k=0.5,                       # Параметр 2
    r=128                        # Параметр 3
)

# Сегментация: возврат бинарной маски {0, 255}
mask = segmenter.segment(image)  # image: np.ndarray, PIL.Image, str, torch.Tensor

# Сегментация + визуализация: возврат (overlay, mask)
overlay, mask = segmenter.segment_with_mask(image, alpha=0.7)

# Оценка качества (при наличии Ground Truth)
# metrics = segmenter.evaluate_metrics(pred_mask=mask, gt_mask=gt_mask)
# print(f"IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}")
```

### 🎚️ Гибкая настройка устройства и отладки
- **Авто-детекция GPU:** `device=None` → автоматический выбор `cuda` при доступности
- **Fallback на внешние библиотеки:** `use_external_libs=True` → использование sklearn/scipy при необходимости
- **Режим отладки:** `debug_mode=True` → вывод промежуточных результатов и метрик выполнения

### 🎨 Поддержка форматов входа
| Тип | Пример | Обработка |
|-----|--------|-----------|
| `str` | `"image.jpg"` | Загрузка через `PIL.Image.open()`, конвертация в RGB |
| `PIL.Image` | `Image.open("img.png")` | Конвертация в `np.array` → `torch.Tensor` |
| `np.ndarray` | `np.random.randint(0,255,(512,512,3))` | Конвертация в `torch.Tensor` с нормализацией |
| `torch.Tensor` | `torch.rand(3,512,512)` | Прямое использование, перенос на `self.device` |

## 🚀 Быстрый старт

### Базовое использование: пороговая сегментация
```python
import cv2
import numpy as np
from segmenters.TorchSegmenter import TorchSegmenter

# Загрузка изображения
image = cv2.imread("document.jpg")  # BGR формат
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Метод Оцу (автоматический порог)
segmenter = TorchSegmenter(method="otsu_thresholding", device="cuda")
mask_otsu = segmenter.segment(image_rgb)

# Адаптивный порог Сауволы (для неравномерного освещения)
segmenter = TorchSegmenter(
    method="threshold_sauvola",
    window_size=15,
    k=0.5,
    r=128,
    device="cuda"
)
mask_adaptive = segmenter.segment(image_rgb)

# Визуализация
overlay, mask = segmenter.segment_with_mask(image_rgb, alpha=0.6)
cv2.imwrite("result.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

### Детекция границ: Canny с настройкой порогов
```python
# Детектор Кэнни с параметрами
segmenter = TorchSegmenter(
    method="canny_edge",
    low=0.1,      # Нижний порог гистерезиса [0,1]
    high=0.3,     # Верхний порог
    sigma=1.0     # Сигма Гауссова размытия
)
edges = segmenter.segment(image_rgb)

# Постобработка: морфологическое замыкание для связных контуров
import torch
kernel = torch.ones(3, 3, device=segmenter.device)
# ... дальнейшая обработка
```

### Активные контуры: Чан-Везе для медицинских изображений
```python
# Сегментация без градиентов (эффективна для размытых границ)
segmenter = TorchSegmenter(
    method="chan_vese",
    mu=0.25,           # Вес длины контура
    lambda1=1.0,       # Вес внутренней области
    lambda2=1.0,       # Вес внешней области
    max_iter=100,
    init_level_set="checkerboard",  # Инициализация: "checkerboard", "disk", "small_disk"
    device="cuda"
)
mask = segmenter.segment(medical_image)
```

### Кластеризация: K-Means для цветовой сегментации
```python
# Сегментация на 3 кластера в цветовом пространстве
segmenter = TorchSegmenter(
    method="kmeans_segmentation",
    k=3,  # Количество кластеров
    device="cuda"
)
mask = segmenter.segment(color_image)  # Автоматическая обработка RGB

# Самый крупный кластер считается фоном, остальные — объектом
print(f"Покрытие объекта: {(mask > 0).sum() / mask.numel() * 100:.2f}%")
```

## ⚙️ Конфигурация

### Параметры инициализации `TorchSegmenter.__init__()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `method` | `str` | `"global_thresholding"` | Название метода (см. таблицы ниже) |
| `device` | `Optional[str]` | `None` | Устройство: `"cuda"`, `"cpu"` или `None` (авто-детекция) |
| `use_external_libs` | `bool` | `True` | Разрешить использование sklearn/scipy при необходимости |
| `**kwargs` | `Any` | — | Параметры конкретного метода сегментации |

### Параметры `segment(image, **kwargs)`
| Параметр | Тип | Описание |
|----------|-----|----------|
| `image` | `ImageInput` | Входное изображение: `str`, `PIL.Image`, `np.ndarray`, `torch.Tensor` |
| `**kwargs` | `Any` | Переопределение параметров метода при вызове |

### Параметры `segment_with_mask(image, alpha=0.9, **kwargs)`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `alpha` | `float` | `0.9` | Прозрачность наложения: 0.0=только фото, 1.0=только маска |
| `**kwargs` | `Any` | — | Дополнительные параметры для метода сегментации |

## 📚 Справочник методов

### 🔹 Пороговые методы (14 алгоритмов)
| Метод | Параметры | Описание | Рекомендации |
|-------|-----------|----------|-------------|
| `global_thresholding` | `threshold=0.5` | Фиксированный порог яркости | Простые сцены с равномерным освещением |
| `otsu_thresholding` | — | Автоматический порог по максимизации межклассовой дисперсии | Бимодальные гистограммы (текст/фон) |
| `adaptive_thresholding` | `block_size=11`, `C=2` | Локальный порог на основе гауссовой окрестности | Неравномерное освещение, документы |
| `threshold_niblack` | `window_size=15`, `k=-0.2` | `T = μ + k·σ` | Текст на светлом фоне, умеренный шум |
| `threshold_sauvola` | `window_size=15`, `k=0.5`, `r=128` | `T = μ·[1 + k·(σ/R - 1)]` | Низкоконтрастные документы, сканы |
| `threshold_bernsen` | `window_size=15`, `contrast_threshold=0.1` | Порог = (min+max)/2 при достаточном контрасте | Высококонтрастные изображения |
| `threshold_phansalkar` | `window_size=15`, `k=0.25`, `r=128`, `m=0.5` | Улучшенный Ниблак для низкого контраста | Медицинские снимки, микрофотографии |
| `threshold_kittler_illingworth` | `num_bins=256` | Минимизация ошибки классификации по гистограмме | Статистически однородные сцены |
| `threshold_entropy_kapur` | `num_bins=256` | Максимизация суммы энтропий фона/объекта | Информационно насыщенные изображения |
| `threshold_triangle` | `num_bins=256` | Геометрический метод для асимметричных гистограмм | Текст на странице, один выраженный пик |
| `threshold_multi_otsu` | `n_thresholds=2` | Многоуровневый Оцу для разделения на >2 классов | Несколько однородных областей |
| `threshold_percentile` | `percentile=90` | Порог как заданный процентиль распределения | Выделение самых ярких/тёмных пикселей |
| `threshold_local_contrast` | `window_size=15`, `contrast_factor=0.1` | Порог на основе локального контраста | Текстурные изображения, неоднородный фон |

### 🔹 Детекторы границ (10 алгоритмов)
| Метод | Параметры | Описание | Рекомендации |
|-------|-----------|----------|-------------|
| `sobel_edge` | `threshold=0.1` | Градиент через свёртку с ядрами Собеля | Базовая детекция, чувствителен к шуму |
| `canny_edge` | `low=0.1`, `high=0.3`, `sigma=1.0` | Многоэтапный детектор с гистерезисом | Оптимальный баланс точности/шума |
| `prewitt_edge` | `threshold=0.1` | Градиент с равными весами [1,1,1] | Менее точен, но устойчивее к шуму, чем Собель |
| `scharr_edge` | `threshold=0.1` | Оптимизированные ядра для точного градиента | Высокая точность локализации границ |
| `roberts_cross_edge` | `threshold=0.1` | Диагональные границы через ядро 2×2 | Быстрая детекция диагональных структур |
| `log_edge` | `sigma=1.0`, `threshold=0.1` | Лапласиан Гауссиана, нулевые пересечения | Плавные переходы, инвариантность к масштабу |
| `dog_edge` | `sigma1=1.0`, `sigma2=2.0`, `threshold=0.1` | Разность Гауссианов, аппроксимация LoG | Границы разного масштаба, подавление шума |
| `marr_hildreth_edge` | `sigma=1.5`, `threshold=0.1` | Классический LoG через `cv2.Laplacian` | Медицинские изображения, размытые границы |
| `gradient_magnitude_direction` | `threshold=0.1` | Магнитуда + фильтрация по направлению | Выделение границ определённой ориентации |
| `phase_congruency_edge` | `nscales=4`, `norientations=4`, `threshold=0.3` | Инвариантная к контрасту детекция через фазы Фурье | Низкоконтрастные медицинские/спутниковые снимки |

### 🔹 Региональные, кластеризация, активные контуры
| Категория | Метод | Параметры | Применение |
|-----------|-------|-----------|-----------|
| **Региональные** | `region_growing` | `seed=None`, `tolerance=0.1` | Однородные области, интерактивная сегментация |
| | `split_and_merge` | `threshold=20`, `min_size=50` | Спутниковые снимки, индустриальные изображения |
| | `floodfill` | `points=None`, `tolerance=0.15` | Заливка от точки, выделение конкретных структур |
| **Кластеризация** | `kmeans_segmentation` | `k=3` | Цветовая сегментация, группировка по признакам |
| | `dbscan_segmentation` | `eps=0.1`, `min_samples=10` | Объекты произвольной формы, автоматическое число кластеров |
| | `meanshift` | `bandwidth=0.5`, `spatial_radius=35` | Плавные цветовые переходы, сохранение границ |
| **Активные контуры** | `active_contour` | `alpha=0.01`, `beta=0.1`, `max_iter=250` | Медицинские изображения, объекты с чёткими границами |
| | `gvf_contour` | `mu=0.2`, `iterations=20` | Размытые/прерывистые границы, вогнутые объекты |
| | `morphological_snakes` | `iterations=100`, `smoothing=1` | Быстрая альтернатива энергетическим моделям |
| | `chan_vese` | `mu=0.25`, `lambda1=1.0`, `max_iter=100` | Объекты без чётких границ, однородная внутренняя область |

### 🔹 Watershed, суперпиксели, интерактивные
| Категория | Метод | Параметры | Применение |
|-----------|-------|-----------|-----------|
| **Watershed** | `watershed` | `connectivity=4`, `gradient_method="sobel"` | Разделение слипшихся объектов, маркерная сегментация |
| | `random_walker` | `beta=130`, `mode="scipy"`, `tol=1e-3` | Плавные переходы, вероятностная сегментация |
| **Суперпиксели** | `quickshift` | `kernel_size=5`, `max_dist=10`, `ratio=1.0` | Быстрая кластеризация в пространстве цвет+координаты |
| | `slic` | `n_segments=100`, `compactness=10.0` | Компактные однородные регионы, предобработка |
| | `felzenszwalb` | `scale=100`, `sigma=0.8`, `min_size=50` | Адаптивная сегментация разного масштаба |
| **Интерактивные** | `grabcut` | `rect=None`, `num_iterations=5` | Полуавтоматическое выделение по прямоугольнику |

## ⚡ Производительность и оптимизации

### Относительная скорость методов (на изображении 512×512, CPU)
```
✅ Быстро (<10 мс):
   - global_thresholding, otsu_thresholding
   - sobel_edge, prewitt_edge, roberts_cross_edge
   - floodfill, watershed

⚠️ Средне (10–100 мс):
   - adaptive_thresholding, niblack, sauvola, bernsen
   - canny_edge, scharr_edge, log_edge, dog_edge
   - region_growing, split_and_merge, kmeans_segmentation
   - slic, felzenszwalb, quickshift

❌ Медленно (100–1000+ мс):
   - phansalkar, kittler_illingworth, entropy_kapur, triangle
   - phase_congruency_edge (FFT + банк фильтров)
   - dbscan_segmentation, meanshift (sklearn fallback)
   - active_contour, gvf_contour, morphological_snakes, chan_vese
   - random_walker (разреженная система)
```

### Рекомендации по оптимизации
1. **Используйте GPU при возможности:**
   ```python
   segmenter = TorchSegmenter(method="chan_vese", device="cuda")
   ```

2. **Выбирайте метод под задачу:**
   - Для быстрой предварительной обработки: `otsu_thresholding`, `sobel_edge`
   - Для точной сегментации: `chan_vese`, `random_walker`
   - Для интерактивной работы: `grabcut`, `floodfill`

3. **Кэшируйте экземпляр сегментера:**
   ```python
   # Создайте один экземпляр и переиспользуйте для множества изображений
   segmenter = TorchSegmenter(method="sauvola", window_size=15)
   for image in image_list:
       mask = segmenter.segment(image)  # Инициализация однократна
   ```

4. **Отключайте визуализацию для бенчмарков:**
   ```python
   # Используйте segment() вместо segment_with_mask() если не нужен overlay
   mask = segmenter.segment(image)  # Быстрее на ~15-30%
   ```

## 🛠️ Обработка ошибок и устойчивость

### Стратегия возврата при сбое
```python
try:
    # ... выполнение метода ...
    return mask
except Exception as e:
    warnings.warn(f"Ошибка в методе {self.method}: {e}")
    traceback.print_exc()
    # Возврат пустой маски того же размера
    h, w = image.shape[:2] if isinstance(image, np.ndarray) else (256, 256)
    return np.zeros((h, w), dtype=np.uint8)
```

**Преимущества:**
- ✅ Пакетное тестирование не прерывается при ошибке в одном методе.
- ✅ Бенчмарки могут агрегировать статистику даже при частичных сбоях.
- ✅ Вызывающий код может проверить результат: `if mask.sum() == 0: # пустая маска`.

### Рекомендации по отладке
1. **Включите логирование:**
   ```python
   import logging
   logging.getLogger("segmenters.TorchSegmenter").setLevel(logging.DEBUG)
   ```

2. **Проверьте входные данные:**
   ```python
   print(f"Image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
   ```

3. **Тестируйте на маленьком изображении:**
   ```python
   small_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
   mask = segmenter.segment(small_image)  # Быстрая проверка конвейера
   ```

4. **Используйте debug_mode для анализа:**
   ```python
   segmenter = TorchSegmenter(method="chan_vese", debug_mode=True)
   # Вывод промежуточных энергий и параметров в консоль
   ```

## 🤝 Интеграция с другими модулями проекта

| Модуль | Использование TorchSegmenter |
|--------|-----------------------------|
| `BatchClassicTester` | Массовое тестирование согласованности реализаций |
| `BatchClassicTester2` | Валидация качества классических методов против GT |
| `TorchImplementationValidator` | Сравнение эталонных реализаций |
| `SegmentationTester` | Универсальное тестирование через `add_method()` |
| `CpuCudaBenchmark` | Бенчмарк производительности (CPU vs GPU) |

### Пример интеграции с PyTorch DataLoader
```python
from torch.utils.data import DataLoader
from datasets.ADE20KDataset import ADE20KDataset
from segmenters.TorchSegmenter import TorchSegmenter

# Подготовка данных
dataset = ADE20KDataset(root_dir="./data/ade20k", split="val", augment=False)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Инициализация сегментера
segmenter = TorchSegmenter(method="otsu_thresholding", device="cuda")

# Инференс на батче
for batch in loader:
    images = batch["image"]  # [B, 3, H, W]
    
    # Пакетная сегментация (требует модификации метода под batch)
    masks = []
    for img in images:
        mask = segmenter.segment(img)  # (H, W)
        masks.append(mask)
    
    # ... оценка качества, сохранение результатов ...
```

## 📦 Зависимости

### Обязательные
```text
torch>=1.9.0           # PyTorch тензоры, свёртки, FFT
torchvision>=0.10.0    # gaussian_blur, transforms
numpy>=1.20.0          # Fallback-реализации, предобработка
Pillow>=9.0.0          # Загрузка изображений
```

### Опциональные (для расширенной функциональности)
```text
scipy>=1.7.0           # ndimage для морфологии, distance_transform для watershed
scikit-learn>=1.0.0    # DBSCAN, MeanShift fallback
scikit-image>=0.19.0   # Felzenszwalb fallback
opencv-python>=4.5.0   # Визуализация, предобработка
```

### Установка
```bash
# Базовая установка
pip install torch torchvision numpy Pillow

# Полная установка (все опциональные зависимости)
pip install scipy scikit-learn scikit-image opencv-python

# Проверка установки
python -c "from segmenters.TorchSegmenter import TorchSegmenter; print('✅ OK')"
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

> 💡 **Совет:** Для максимальной совместимости с `NewTorchSegmenter` используйте одинаковые имена параметров (например, `threshold` вместо `low`/`high` для пороговых методов). Это упростит миграцию кода между версиями.

```python
# Совместимый вызов для обеих версий:
segmenter_old = TorchSegmenter(method="canny_edge", low=0.1, high=0.3)
segmenter_new = NewTorchSegmenter(method="canny_edge", low_threshold=0.1, high_threshold=0.3)
# Обратите внимание на различия в именах параметров!
```