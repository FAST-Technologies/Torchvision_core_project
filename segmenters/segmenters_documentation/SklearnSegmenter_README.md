# 🧠 SklearnSegmenter — Сегментация на базе scikit-learn и scikit-image

## 📖 Описание
Модуль `SklearnSegmenter.py` предоставляет **универсальный интерфейс** для **80+ алгоритмов сегментации**, реализованных через библиотеки `scikit-learn`, `scikit-image` и вспомогательные инструменты (SciPy, NumPy, OpenCV).

> ⚠️ **Важно:** Данный модуль реализует *классические (не нейросетевые)* методы на основе машинного обучения и статистического анализа. Для глубокого обучения используйте `TorchSegmenter`, `TorchSegmenter2` или `NeuralModelFactory`.

## ✨ Ключевые возможности
### 🗂️ 10 категорий методов (80+ алгоритмов)

| Категория | Методов | Примеры | Сценарии использования |
|-----------|---------|---------|----------------------|
| **Пороговые** (13) | 13 | `global`, `otsu`, `adaptive`, `niblack`, `sauvola`, `bernsen`, `phansalkar`, `kittler_illingworth`, `entropy_kapur`, `triangle`, `multi_otsu`, `percentile`, `local_contrast` | Документы, сканы, изображения с неравномерным освещением |
| **Детекторы границ** (10) | 10 | `sobel`, `canny`, `prewitt`, `scharr`, `roberts`, `log`, `dog`, `marr_hildreth`, `gradient_magnitude`, `phase_congruency` | Выделение контуров, предобработка для watershed, детекция объектов |
| **Региональные** (3) | 3 | `region_growing`, `split_and_merge`, `floodfill` | Сегментация однородных областей, интерактивное выделение |
| **Кластеризация** (15+) | 15+ | `kmeans`, `dbscan`, `meanshift`, `gmm`, `optics`, `agglomerative`, `spectral`, `birch`, `mini_batch_kmeans` | Цветовая сегментация, группировка пикселей по признакам |
| **Активные контуры** (4) | 4 | `active_contour`, `gvf_contour`, `morphological_snakes`, `chan_vese` | Медицинские изображения, объекты с размытыми границами |
| **Watershed/графовые** (2) | 2 | `watershed`, `random_walker` | Разделение слипшихся объектов, сегментация по маркерам |
| **Суперпиксели** (3) | 3 | `slic`, `felzenszwalb`, `quickshift` | Предварительная сегментация, упрощение изображения |
| **ML-классификаторы** (10+) | 10+ | `random_forest`, `svm`, `logistic_regression`, `knn`, `mlp`, `naive_bayes`, `lda` | Полу-автоматическая сегментация с обучением на синтетических метках |
| **Обнаружение аномалий** (4) | 4 | `isolation_forest`, `local_outlier_factor`, `one_class_svm`, `elliptic_envelope` | Детекция дефектов, пятен, инородных объектов |
| **Разложение и многообразия** (6) | 6 | `pca`, `nmf`, `tsne`, `isomap`, `spectral_embedding`, `ica` | Снижение размерности, визуализация, предобработка для кластеризации |

### 🔄 Единый интерфейс для всех методов
```python
from segmenters.SklearnSegmenter import SklearnSegmenter

# Инициализация с методом и параметрами
segmenter = SklearnSegmenter(
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
- **Коррекция размеров:** чётные `block_size`/`window_size` → нечётные (требуется для `threshold_local`).
- **Переименование ключей:** `n_iterations` → `iterations`, `n_clusters` → `k` при необходимости.
- **Нормализация признаков:** автоматическое масштабирование через `StandardScaler` для ML-методов.

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
from segmenters.SklearnSegmenter import SklearnSegmenter

# Загрузка изображения
image = cv2.imread("document.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Метод Оцу (автоматический порог)
segmenter = SklearnSegmenter(method="otsu_thresholding")
mask_otsu = segmenter.segment(image_rgb)

# Адаптивный порог Сауволы (для низкого контраста)
segmenter = SklearnSegmenter(
    method="threshold_sauvola",
    window_size=15,  # Размер окрестности (нечётный!)
    k=0.5,           # Параметр контраста
    r=128            # Динамический диапазон
)
mask_sauvola = segmenter.segment(image_rgb)

# Визуализация
overlay, mask = segmenter.segment_with_mask(image_rgb, alpha=0.6)
cv2.imwrite("result.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

### Кластеризация: K-Means с извлечением признаков
```python
# Автоматическое извлечение: цвет + координаты + текстура
segmenter = SklearnSegmenter(
    method="kmeans_segmentation",
    k=3  # Количество кластеров
)
mask = segmenter.segment(color_image)

# Самый крупный кластер считается фоном, остальные — объектом
print(f"Покрытие объекта: {(mask > 0).sum() / mask.size * 100:.2f}%")
```

### ML-классификатор: Random Forest (полу-автоматический)
```python
# Автоматическая генерация меток: центр = объект, углы = фон
segmenter = SklearnSegmenter(
    method="random_forest",
    n_estimators=50  # Количество деревьев
)
mask = segmenter.segment(texture_image)

# Метод устойчив к шуму и не требует ручной разметки
```

### Обнаружение аномалий: Isolation Forest для дефектов
```python
# Объекты, статистически отличающиеся от фона, помечаются как аномалии
segmenter = SklearnSegmenter(
    method="isolation_forest",
    n_estimators=100,
    contamination="auto"  # Автоматическая оценка доли объекта
)
mask = segmenter.segment(industrial_image)
```

### Активные контуры: Chan-Vese для медицинских изображений
```python
# Сегментация без градиентов (эффективна для размытых границ)
segmenter = SklearnSegmenter(
    method="chan_vese",
    mu=0.25,           # Вес длины контура
    lambda1=1.0,       # Вес внутренней области
    lambda2=1.0,       # Вес внешней области
    max_iter=100       # Количество итераций
)
mask = segmenter.segment(medical_image)
```

## ⚙️ Конфигурация
### Параметры инициализации `SklearnSegmenter.__init__()`
| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `method` | `str` | `"global_thresholding"` | Название метода (см. справочник ниже) |
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
### 🔹 Пороговые методы (13 алгоритмов)
| Метод | Параметры | Описание | Рекомендации |
|-------|-----------|----------|-------------|
| `global_thresholding` | `threshold=0.5` | Фиксированный порог яркости | Простые сцены с равномерным освещением |
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
| `sobel_edge` | `threshold=0.1` | Градиент через свёртку с ядрами Собеля | Базовая детекция, чувствителен к шуму |
| `canny_edge` | `sigma=1.0`, `low=0.1`, `high=0.3` | Многоэтапный детектор с гистерезисом | Оптимальный баланс точности/шума |
| `prewitt_edge` | `threshold=0.1` | Градиент с равными весами [1,1,1] | Менее точен, но устойчивее к шуму |
| `scharr_edge` | `threshold=0.1` | Оптимизированные ядра для точного градиента | Высокая точность локализации границ |
| `roberts_cross_edge` | `threshold=0.1` | Диагональные границы через ядро 2×2 | Быстрая детекция диагональных структур |
| `log_edge` | `sigma=1.0`, `threshold=0.01` | Лапласиан Гауссиана, нулевые пересечения | Плавные переходы, инвариантность к масштабу |
| `dog_edge` | `sigma1=1.0`, `sigma2=2.0`, `threshold=0.01` | Разность Гауссианов, аппроксимация LoG | Границы разного масштаба, подавление шума |
| `marr_hildreth_edge` | `sigma=1.5`, `threshold=0.01` | Классический LoG через `cv2.Laplacian` | Медицинские изображения, размытые границы |
| `gradient_magnitude_direction` | `threshold=0.1`, `angle_range=None` | Магнитуда + фильтрация по направлению | Выделение границ определённой ориентации |
| `phase_congruency_edge` | `nscale=4`, `norientations=4`, `threshold=0.5` | Инвариантная к контрасту детекция через фазы Фурье | Низкоконтрастные медицинские/спутниковые снимки |

### 🔹 Кластеризация и ML-методы (выборка)
| Категория | Метод | Параметры | Применение |
|-----------|-------|-----------|-----------|
| **Кластеризация** | `kmeans_segmentation` | `k=3` | Цветовая сегментация, группировка по признакам |
| | `dbscan_segmentation` | `eps=0.1`, `min_samples=10` | Объекты произвольной формы, автоматическое число кластеров |
| | `meanshift` | `bandwidth=0.5` | Плавные цветовые переходы, сохранение границ |
| | `gmm` | `n_components=3`, `covariance_type="full"` | Статистическое моделирование распределений |
| **ML-классификаторы** | `random_forest` | `n_estimators=50` | Полу-автоматическая сегментация с авто-метками |
| | `svm` | `C=1.0`, `kernel="rbf"` | Точная сегментация при чётких границах в пространстве признаков |
| | `knn` | `n_neighbors=5` | Простая и интерпретируемая сегментация |
| **Аномалии** | `isolation_forest` | `n_estimators=100`, `contamination="auto"` | Детекция дефектов, инородных объектов |
| | `local_outlier_factor` | `n_neighbors=20` | Локальное обнаружение выбросов |

> 📋 Полный список из 80+ методов доступен в атрибуте `SklearnSegmenter.methods`.

## 🔄 Конвейер обработки: предобработка → сегментация → постобработка
### Автоматическая предобработка (`preprocess_image()`)
```python
# 1. Загрузка/конвертация в np.ndarray
# 2. Конвертация в grayscale (если требуется методом)
# 3. Нормализация к [0, 1] для совместимости с scikit-image
# 4. Опциональный ресайз с адаптивной интерполяцией
```

### Извлечение признаков для ML-методов (`_extract_features()`)
```python
# Для каждого пикселя извлекаются:
# 1. Цветовые признаки: интенсивность или 3 канала (RGB/Lab)
# 2. Пространственные координаты: нормализованные (x, y)
# 3. Текстура: градиенты Собеля (Gx, Gy, |G|)
# Итого: 8 признаков на пиксель → автоматическое масштабирование StandardScaler
```

### Логирование выполнения (`_log_info()`)
```python
def _log_info(self, method_name: str, exec_time: float, params: Dict[str, Any]) -> SegmentationInfo:
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
   - global_thresholding, otsu_thresholding, percentile
   - sobel_edge, prewitt_edge, roberts_cross_edge
   - floodfill, watershed

⚠️ Средне (10–100 мс):
   - adaptive_thresholding, niblack, sauvola, bernsen
   - canny_edge, scharr_edge, log_edge, dog_edge
   - region_growing, split_and_merge, kmeans_segmentation
   - slic, felzenszwalb, quickshift
   - random_forest, svm, knn (при сэмплировании)

❌ Медленно (100–1000+ мс):
   - phansalkar, kittler_illingworth, entropy_kapur, triangle
   - phase_congruency_edge (особенно с nscale>4)
   - dbscan_segmentation, meanshift, optics, spectral
   - active_contour, gvf_contour, morphological_snakes, chan_vese
   - random_walker, isolation_forest (на больших изображениях)
```

### Рекомендации по оптимизации
1. **Ресайз перед обработкой:** для изображений >1000×1000 уменьшите до 512×512 для медленных методов.
2. **Сэмплирование для ML:** методы `dbscan`, `spectral`, `tsne` автоматически сэмплируют признаки при N > 1000.
3. **Выбор параметров:** меньшие `window_size`, `nscale`, `iterations` → быстрее, но менее точно.
4. **Кэширование:** переиспользуйте экземпляр сегментера для множества изображений — инициализация однократна.
5. **Отключение визуализации:** используйте `segment()` вместо `segment_with_mask()` если не нужен overlay.

## 🛠️ Обработка ошибок и устойчивость
### Стратегия возврата при сбое
```python
try:
    # ... выполнение метода ...
    return mask
except Exception as e:
    warnings.warn(f"Ошибка в методе {self.method}: {e}. Возвращаем пустую маску.", RuntimeWarning)
    h, w = img_processed.shape[:2]
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
   logging.getLogger("segmenters.SklearnSegmenter").setLevel(logging.DEBUG)
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
scikit-learn>=1.0     # KMeans, DBSCAN, SVM, RandomForest, PCA, etc.
scikit-image>=0.19    # filters, segmentation, feature, color
opencv-python>=4.5    # Загрузка изображений, GrabCut, морфология
numpy>=1.20           # Массивы, статистики, векторизация
scipy>=1.7            # ndimage, signal, gaussian_filter
torch>=1.9            # Поддержка torch.Tensor на входе (опционально)
Pillow>=8.0           # Поддержка PIL.Image на входе (опционально)
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование SklearnSegmenter |
|--------|-------------------------------|
| `BatchClassicTester` | Массовое тестирование согласованности реализаций |
| `BatchClassicTester2` | Валидация качества классических методов против GT |
| `TorchImplementationValidator` | Сравнение PyTorch-реализаций против scikit-learn эталона |
| `SegmentationTester` | Универсальное тестирование через `add_method()` |
| `CpuCudaBenchmark` | Бенчмарк производительности (scikit-learn vs PyTorch vs ONNX) |

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