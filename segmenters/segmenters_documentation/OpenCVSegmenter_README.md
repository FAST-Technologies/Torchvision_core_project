# 🧰 OpenCVSegmenter — Классические методы сегментации на базе OpenCV

## 📖 Описание
Модуль `OpenCVSegmenter.py` предоставляет **универсальный интерфейс** для 50+ классических алгоритмов сегментации изображений, реализованных через библиотеку OpenCV и вспомогательные инструменты (SciPy, scikit-learn, scikit-image).

> ⚠️ **Важно:** Данный модуль реализует *классические (не нейросетевые)* методы. Для глубокого обучения используйте `TorchSegmenter`, `TorchSegmenter2` или `NeuralModelFactory`.

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
from segmenters.OpenCVSegmenter import OpenCVSegmenter

# Инициализация с методом и параметрами
segmenter = OpenCVSegmenter(
    method="threshold_sauvola",  # Название метода
    window_size=15,              # Параметр 1
    k=0.5,                       # Параметр 2
    r=128                        # Параметр 3
)

# Сегментация: возврат бинарной маски {0, 255}
mask = segmenter.segment(image)  # image: np.ndarray, PIL.Image, str, torch.Tensor

# Сегментация + визуализация: возврат (overlay, mask)
overlay, mask = segmenter.segment_with_mask(image, alpha=0.7)

# Оценка качества (при наличии Ground Truth)
metrics = segmenter.evaluate_metrics(pred_mask=mask, gt_mask=gt_mask)
print(f"IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}")
```

### 🎚️ Автоматическая адаптация параметров
- **Конвертация диапазонов:** значения `[0, 1]` → `[0, 255]` для порогов (`threshold`, `low`, `high`).
- **Коррекция размеров:** чётные `block_size`/`window_size` → нечётные (требуется для `cv2.adaptiveThreshold`).
- **Переименование ключей:** `n_iterations` → `iterations` для GrabCut, `n_clusters` → `k` для K-Means.

### 🎨 Поддержка форматов входа
| Тип | Пример | Обработка |
|-----|--------|-----------|
| `str` | `"image.jpg"` | Загрузка через `cv2.imread()`, BGR→RGB конвертация |
| `PIL.Image` | `Image.open("img.png")` | Конвертация в `np.array`, опция `.convert("RGB")`/`"L"` |
| `np.ndarray` | `np.random.randint(0,255,(512,512,3))` | Копирование, опциональная grayscale-конвертация |
| `torch.Tensor` | `torch.rand(3,512,512)` | `.permute()`, `.cpu()`, float→uint8 при необходимости |

## 🚀 Быстрый старт
### Базовое использование: пороговая сегментация
```python
import cv2
import numpy as np
from segmenters.OpenCVSegmenter import OpenCVSegmenter

# Загрузка изображения
image = cv2.imread("document.jpg")  # BGR формат
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Метод Оцу (автоматический порог)
segmenter = OpenCVSegmenter(method="otsu_thresholding")
mask_otsu = segmenter.segment(image_rgb)

# Адаптивный порог (для неравномерного освещения)
segmenter = OpenCVSegmenter(
    method="adaptive_thresholding",
    block_size=11,  # Размер окрестности (нечётный!)
    C=2             # Смещение порога
)
mask_adaptive = segmenter.segment(image_rgb)

# Визуализация
overlay, mask = segmenter.segment_with_mask(image_rgb, alpha=0.6)
cv2.imwrite("result.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

### Детекция границ: Canny + постобработка
```python
# Детектор Кэнни с параметрами
segmenter = OpenCVSegmenter(
    method="canny_edge",
    low=0.1,      # Нижний порог гистерезиса [0,1] → авто-конвертация в [0,255]
    high=0.3,     # Верхний порог
    sigma=1.0     # Сигма Гауссова размытия
)
edges = segmenter.segment(image_rgb)

# Постобработка: морфологическое замыкание для связных контуров
kernel = np.ones((3,3), np.uint8)
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
```

### Кластеризация: K-Means для цветовой сегментации
```python
# Сегментация на 3 кластера в цветовом пространстве
segmenter = OpenCVSegmenter(
    method="kmeans_segmentation",
    k=3  # Количество кластеров
)
mask = segmenter.segment(color_image)  # Автоматическая обработка RGB

# Самый крупный кластер считается фоном, остальные — объектом
print(f"Покрытие объекта: {(mask > 0).sum() / mask.size * 100:.2f}%")
```

### Активные контуры: Chan-Vese для медицинских изображений
```python
# Сегментация без градиентов (эффективна для размытых границ)
segmenter = OpenCVSegmenter(
    method="chan_vese",
    mu=0.25,           # Вес длины контура
    max_iter=100,      # Количество итераций
    lambda1=1.0,       # Вес внутренней области
    lambda2=1.0        # Вес внешней области
)
mask = segmenter.segment(medical_image)
```

## ⚙️ Конфигурация
### Параметры инициализации `OpenCVSegmenter.__init__()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `method` | `str` | `"global_thresholding"` | Название метода (см. таблицы ниже) |
| `**kwargs` | `Any` | — | Параметры метода (автоматически адаптируются) |

### Параметры `segment(image, **kwargs)`
| Параметр | Тип | Описание |
|---|---|---|
| `image` | `ImageInput` | Входное изображение: `str`, `PIL.Image`, `np.ndarray`, `torch.Tensor` |
| `**kwargs` | `Any` | Переопределение параметров метода при вызове |

### Параметры `segment_with_mask(image, alpha=0.9, **kwargs)`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `alpha` | `float` | `0.9` | Прозрачность наложения: 0.0=только фото, 1.0=только маска |

## 📚 Справочник методов
### 🔹 Пороговые методы (14 алгоритмов)
| Метод | Параметры | Описание | Рекомендации |
|-------|-----------|----------|-------------|
| `global_thresholding` | `threshold=127` | Фиксированный порог яркости | Простые сцены с равномерным освещением |
| `otsu_thresholding` | — | Автоматический порог по максимизации межклассовой дисперсии | Бимодальные гистограммы (текст/фон) |
| `adaptive_thresholding` | `block_size=11`, `C=2` | Локальный порог на основе гауссовой окрестности | Неравномерное освещение, документы |
| `threshold_niblack` | `window_size=15`, `k=-0.2` | `T = μ + k·σ` | Текст на светлом фоне, умеренный шум |
| `threshold_sauvola` | `window_size=15`, `k=0.5`, `r=128` | `T = μ·[1 + k·(σ/R - 1)]` | Низкоконтрастные документы, сканы |
| `threshold_bernsen` | `window_size=15`, `contrast_threshold=25` | Порог = (min+max)/2 при достаточном контрасте | Высококонтрастные изображения |
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
| `sobel_edge` | `threshold=50` | Градиент через свёртку с ядрами Собеля | Базовая детекция, чувствителен к шуму |
| `canny_edge` | `low=50`, `high=150`, `sigma=1.0` | Многоэтапный детектор с гистерезисом | Оптимальный баланс точности/шума |
| `prewitt_edge` | `threshold=50` | Градиент с равными весами [1,1,1] | Менее точен, но устойчивее к шуму, чем Собель |
| `scharr_edge` | `threshold=50` | Оптимизированные ядра для точного градиента | Высокая точность локализации границ |
| `roberts_cross_edge` | `threshold=50` | Диагональные границы через ядро 2×2 | Быстрая детекция диагональных структур |
| `log_edge` | `sigma=1.0`, `threshold=10` | Лапласиан Гауссиана, нулевые пересечения | Плавные переходы, инвариантность к масштабу |
| `dog_edge` | `sigma1=1.0`, `sigma2=2.0`, `threshold=10` | Разность Гауссианов, аппроксимация LoG | Границы разного масштаба, подавление шума |
| `marr_hildreth_edge` | `sigma=1.5`, `threshold=10` | Классический LoG через `cv2.Laplacian` | Медицинские изображения, размытые границы |
| `gradient_magnitude_direction` | `threshold=0.2`, `angle_range=None` | Магнитуда + фильтрация по направлению | Выделение границ определённой ориентации |
| `phase_congruency_edge` | `nscales=4`, `norientations=4`, `threshold=0.3` | Инвариантная к контрасту детекция через фазы Фурье | Низкоконтрастные медицинские/спутниковые снимки |

### 🔹 Региональные, кластеризация, активные контуры
| Категория | Метод | Параметры | Применение |
|-----------|-------|-----------|-----------|
| **Региональные** | `region_growing` | `seed=None`, `tolerance=25` | Однородные области, интерактивная сегментация |
| | `split_and_merge` | `threshold=20`, `min_size=50` | Спутниковые снимки, индустриальные изображения |
| | `floodfill` | `seed=None`, `tolerance=20` | Заливка от точки, выделение конкретных структур |
| **Кластеризация** | `kmeans_segmentation` | `k=3` | Цветовая сегментация, группировка по признакам |
| | `dbscan_segmentation` | `eps=0.05`, `min_samples=5` | Объекты произвольной формы, автоматическое число кластеров |
| | `meanshift` | `spatial_radius=60`, `color_radius=60` | Плавные цветовые переходы, сохранение границ |
| **Активные контуры** | `active_contour` | `iterations=10`, `alpha=0.015`, `beta=10` | Медицинские изображения, объекты с чёткими границами |
| | `gvf_contour` | `mu=0.1`, `iterations=50` | Размытые/прерывистые границы, вогнутые объекты |
| | `morphological_snakes` | `iterations=50` | Быстрая альтернатива энергетическим моделям |
| | `chan_vese` | `mu=0.25`, `max_iter=100` | Объекты без чётких границ, однородная внутренняя область |

### 🔹 Watershed, суперпиксели, интерактивные
| Категория | Метод | Параметры | Применение |
|-----------|-------|-----------|-----------|
| **Watershed** | `watershed` | — | Разделение слипшихся объектов, маркерная сегментация |
| | `random_walker` | `beta=130`, `mode="cg_j"` | Плавные переходы, вероятностная сегментация |
| **Суперпиксели** | `slic` | `region_size=20`, `ruler=10.0` | Компактные однородные регионы, предобработка |
| | `felzenszwalb` | `scale=100`, `sigma=0.8`, `min_size=50` | Адаптивная сегментация разного масштаба |
| | `quickshift` | `kernel_size=3`, `max_dist=6`, `ratio=0.5` | Быстрая кластеризация в пространстве цвет+координаты |
| **Интерактивные** | `grabcut` | `rect=None`, `num_iterations=10` | Полуавтоматическое выделение по прямоугольнику |

## 🔄 Конвейер обработки: предобработка → сегментация → постобработка
### Автоматическая предобработка (`preprocess_image()`)
```python
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
```

### Адаптация параметров (`_adapt_params()`)
```python
# Конвертация порогов [0,1] → [0,255]
intensity_params = ["threshold", "low", "high", "t1", "t2", "contrast_threshold"]
for key in intensity_params:
    if key in params and 0.0 <= params[key] <= 1.0:
        params[key] = int(params[key] * 255)

# Коррекция чётных размеров окон
if method in ["adaptive_thresholding", "threshold_niblack", ...]:
    if params.get("block_size", 11) % 2 == 0:
        params["block_size"] += 1

# Переименование ключей
if method == "grabcut":
    params["iterations"] = params.pop("n_iterations", 10)
```

### Логирование выполнения (`_log_info()`)
```python
def _log_info(self, method_name: str, exec_time: float, params: Dict[str, Any]) -> None:
    self.info = {
        "method": method_name,
        "parameters": params,
        "execution_time": exec_time,
    }
    logger.info(f"{method_name}: {exec_time:.3f}s, params={params}")
```

## 📊 Метрики качества (через `SegmentationMetrics`)
При наличии Ground Truth маски доступны следующие метрики:

| Метрика | Описание | Диапазон | Интерпретация |
|---------|----------|----------|---------------|
| **IoU** | Intersection over Union | [0.0, 1.0] | Чем выше, тем лучше; основной критерий |
| **Dice** | Dice / Sørensen coefficient | [0.0, 1.0] | Более устойчив к дисбалансу классов |
| **F1-Score** | Гармоническое среднее Precision/Recall | [0.0, 1.0] | Универсальная метрика качества |
| **Precision** | Точность: верные объекты / предсказанные | [0.0, 1.0] | Высокая = мало ложных срабатываний |
| **Recall** | Полнота: найденные объекты / истинные | [0.0, 1.0] | Высокая = мало пропущенных объектов |
| **Pixel Accuracy** | Доля верно классифицированных пикселей | [0.0, 1.0] | Может быть завышена при дисбалансе |
| **MAE** | Mean Absolute Error | [0.0, 1.0] | Средняя абсолютная ошибка (чем ниже, тем лучше) |
| **Hausdorff Distance** | Макс. расстояние между контурами | [0, ∞) | Чувствительна к выбросам, важна для границ |

## ⚡ Производительность и оптимизации
### Относительная скорость методов (на изображении 512×512)
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
   - phase_congruency_edge (особенно с nscales>4)
   - dbscan_segmentation, meanshift
   - active_contour, gvf_contour, morphological_snakes, chan_vese
   - random_walker, grabcut
```

### Рекомендации по оптимизации
1. **Ресайз перед обработкой:** для изображений >1000×1000 уменьшите до 512×512 для медленных методов.
2. **Выбор параметров:** меньшие `window_size`, `nscales`, `iterations` → быстрее, но менее точно.
3. **Кэширование:** переиспользуйте экземпляр сегментера для множества изображений — инициализация однократна.
4. **Отключение визуализации:** используйте `segment()` вместо `segment_with_mask()` если не нужен overlay.

## 🛠️ Обработка ошибок и устойчивость
### Стратегия возврата при сбое
```python
try:
    # ... выполнение метода ...
    return mask
except Exception as e:
    logger.error(f"{method} inference error: {e}")
    h, w = image.shape[:2]
    return np.zeros((h, w), dtype=np.uint8)  # Пустая маска того же размера
```

**Преимущества:**
- ✅ Пакетное тестирование не прерывается при ошибке в одном методе.
- ✅ Бенчмарки могут агрегировать статистику даже при частичных сбоях.
- ✅ Вызывающий код может проверить результат: `if mask.sum() == 0: # пустая маска`.

### Рекомендации по отладке
1. **Включите логирование:**
   ```python
   import logging
   logging.getLogger("segmenters.OpenCVSegmenter").setLevel(logging.DEBUG)
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

## 🤝 Зависимости
```text
opencv-python>=4.5  # Основные функции: imread, threshold, Canny, etc.
numpy>=1.20         # Массивы, статистики, векторизация
scipy>=1.7          # ndimage, gaussian_filter, laplace для LoG/DoG
scikit-learn>=1.0   # DBSCAN для кластеризации
scikit-image>=0.19  # random_walker, felzenszwalb, quickshift (опционально)
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование OpenCVSegmenter |
|--------|------------------------------|
| `BatchClassicTester` | Массовое тестирование согласованности реализаций |
| `BatchClassicTester2` | Валидация качества классических методов против GT |
| `TorchImplementationValidator` | Сравнение PyTorch-реализаций против OpenCV-эталона |
| `SegmentationTester` | Универсальное тестирование через `add_method()` |
| `CpuCudaBenchmark` | Бенчмарк производительности (OpenCV vs PyTorch vs ONNX) |

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