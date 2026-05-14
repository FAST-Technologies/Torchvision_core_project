# ⚡ NewTorchSegmenter — PyTorch-сегментация с поддержкой AMP, torch.compile и низкоточных вычислений

## 📖 Описание
Модуль `segmenters/NewTorchSegmenter.py` предоставляет **универсальный класс для сегментации изображений** на чистом PyTorch с поддержкой 50+ классических методов, автоматического смешанного точности (AMP), компиляции через `torch.compile` и оптимизаций для различных устройств (CPU/GPU).

> ⚠️ **Важно:** Данный модуль реализует *классические (не нейросетевые)* методы сегментации на чистом PyTorch. Для глубокого обучения используйте `TorchSegmenter`, `DeepLabV3Plus` или другие нейросетевые архитектуры проекта.

## ✨ Ключевые возможности

### 🎚️ Управление точностью вычислений (PrecisionManager)
| Точность | Поддержка | Применение | Экономия памяти |
|----------|-----------|------------|----------------|
| `fp32` | ✅ Все устройства | Стандартные вычисления | — |
| `fp16` | ✅ CUDA ≥ 6.0 | Быстрый инференс на GPU | ~2× |
| `bf16` | ✅ CUDA ≥ 8.0 (Ampere+) | Обучение/инференс на новых GPU | ~2× |
| `float8_*` | ✅ CUDA ≥ 9.0 (Hopper+) | Экспериментальный низкобитный инференс | ~4× |
| `int8` | ✅ CPU + CUDA ≥ 6.0 | Квантование весов (динамическое) | ~4× |

### 🗂️ 8 категорий методов (50+ алгоритмов на чистом PyTorch)

| Категория | Методов | Примеры | Особенности реализации |
|-----------|---------|---------|----------------------|
| **Пороговые** (14) | 14 | `global`, `otsu`, `adaptive`, `niblack`, `sauvola`, `kittler_illingworth`, `entropy_kapur`, `triangle` | Векторизованные гистограммы, сепарабельные свёртки |
| **Детекторы границ** (10) | 10 | `sobel`, `canny`, `prewitt`, `scharr`, `log`, `dog`, `phase_congruency` | Кэшированные ядра через `@lru_cache`, NMS через `torch.roll` |
| **Региональные** (3) | 3 | `region_growing`, `split_and_merge`, `floodfill` | Векторизованный BFS на тензорах, без Python-циклов |
| **Кластеризация** (3) | 3 | `kmeans`, `dbscan`, `meanshift` | `torch.cdist` для расстояний, fallback на sklearn при необходимости |
| **Активные контуры** (4) | 4 | `active_contour`, `gvf_contour`, `morphological_snakes`, `chan_vese` | FFT-решение для циклических матриц, регуляризованные Хевисайд/Дирак |
| **Watershed/графовые** (2) | 2 | `watershed`, `random_walker` | Разреженные тензоры, векторизованная приоритетная очередь, scipy fallback |
| **Суперпиксели** (3) | 3 | `quickshift`, `slic`, `felzenszwalb` | Numpy-реализации с downsample + интерполяция меток |
| **Интерактивные** (1) | 1 | `grabcut` | GMM через `torch.distributions`, EM-алгоритм на GPU |

### ⚡ Оптимизации производительности
```python
# 1. torch.compile поддержка (PyTorch ≥ 2.0)
segmenter = TorchSegmenter2(
    method="sobel_edge",
    use_compile=True,
    compile_mode="reduce-overhead",  # или "max-autotune"
    compile_fullgraph=False,  # True для максимального ускорения (может не скомпилироваться)
    compile_dynamic=True  # Поддержка разных размеров изображений
)

# 2. Автоматическая смешанная точность (AMP)
with segmenter.precision_manager.autocast("bf16"):
    mask = segmenter.segment(image)

# 3. Кэширование ядер и результатов
# - Ядра свёртки кэшируются по (dtype, device, size) через @lru_cache
# - Результаты сегментации кэшируются через LRU (настраиваемый размер)

# 4. Fallback на Numba для CPU-тяжёлых операций
# - region_growing, watershed: автоматическое переключение при h*w > 500k
```

### 🔄 Единый интерфейс для всех методов
```python
from segmenters.NewTorchSegmenter import TorchSegmenter2

# Инициализация с методом и параметрами
segmenter = TorchSegmenter2(
    method="chan_vese",           # Название метода
    device="cuda",                # "cuda" или "cpu"
    precision="bf16",             # Точность вычислений
    use_compile=True,             # Включить torch.compile
    mu=0.25,                      # Параметры метода
    max_iter=100
)

# Сегментация: возврат бинарной маски {0, 255}
mask = segmenter.segment(image)  # image: np.ndarray, PIL.Image, str, torch.Tensor

# Сегментация + визуализация: возврат (overlay, mask)
overlay, mask = segmenter.segment_with_mask(image, alpha=0.7, color=(255, 0, 0))

# Профилирование производительности
profile = segmenter.profile_method(image, n_runs=100, return_trace=True)
print(f"Среднее время: {profile['mean_time_ms']:.2f} ms")
```

## 🚀 Быстрый старт

### Базовое использование: пороговая сегментация
```python
import cv2
import torch
from segmenters.NewTorchSegmenter import TorchSegmenter2

# Загрузка изображения
image = cv2.imread("document.jpg")  # BGR формат
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Метод Оцу с bf16 точностью на GPU
segmenter = TorchSegmenter2(
    method="otsu_thresholding",
    device="cuda" if torch.cuda.is_available() else "cpu",
    precision="bf16"
)
mask_otsu = segmenter.segment(image_rgb)

# Адаптивный порог Сауволы с компиляцией
segmenter = TorchSegmenter2(
    method="threshold_sauvola",
    window_size=25,
    k=0.3,
    r=128,
    use_compile=True,
    compile_mode="reduce-overhead"
)
mask_adaptive = segmenter.segment(image_rgb)

# Визуализация
overlay, mask = segmenter.segment_with_mask(image_rgb, alpha=0.6)
cv2.imwrite("result.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

### Детекция границ: Canny с векторизованным NMS
```python
# Детектор Кэнни с параметрами
segmenter = TorchSegmenter2(
    method="canny_edge",
    low_threshold=0.1,    # Нижний порог [0,1]
    high_threshold=0.3,   # Верхний порог
    sigma=1.0,            # Sigma для гауссова сглаживания
    precision="fp16"      # Половинная точность для скорости
)
edges = segmenter.segment(image_rgb)

# Профилирование времени выполнения
profile = segmenter.profile_method(image_rgb, n_runs=50)
print(f"⏱️  Canny: {profile['mean_time_ms']:.2f} ms ± {profile['std_time_ms']:.2f}")
```

### Активные контуры: Чан-Везе для медицинских изображений
```python
# Сегментация без градиентов (эффективна для размытых границ)
segmenter = TorchSegmenter2(
    method="chan_vese",
    mu=0.25,              # Вес длины контура
    lambda1=1.0,          # Вес ошибки внутри региона
    lambda2=1.0,          # Вес ошибки снаружи региона
    max_iter=150,
    init_type="disk",     # Инициализация: "checkerboard", "disk", "small_disk"
    precision="bf16"
)
mask = segmenter.segment(medical_image)

# Мониторинг сходимости через debug_mode
segmenter = TorchSegmenter2(method="chan_vese", debug_mode=True)
mask = segmenter.segment(medical_image)
# Вывод в консоль: [DEBUG] chan_vese: precision_val=bf16, dtype=torch.bfloat16
```

### Кластеризация: K-Means с векторизованными расстояниями
```python
# Сегментация на 4 кластера в пространстве [Lab, x, y]
segmenter = TorchSegmenter2(
    method="kmeans_segmentation",
    k=4,
    init="kmeans++",      # Ускоренная инициализация
    max_iter=30,
    precision="fp16"
)
mask = segmenter.segment(color_image)

# Самый крупный кластер считается фоном
print(f"Покрытие объекта: {(mask > 0).sum() / mask.numel() * 100:.2f}%")
```

## ⚙️ Конфигурация

### Параметры инициализации `TorchSegmenter2.__init__()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `method` | `str` | `"global_thresholding"` | Название метода (см. таблицы ниже) |
| `device` | `Optional[str]` | `None` | Устройство: `"cuda"` или `"cpu"` (авто-детекция при `None`) |
| `use_external_libs` | `bool` | `True` | Разрешить использование sklearn/scipy при необходимости |
| `use_compile` | `bool` | `True` | Включить `torch.compile` для ускорения (PyTorch ≥ 2.0) |
| `compile_mode` | `str` | `"reduce-overhead"` | Режим компиляции: `"default"`, `"reduce-overhead"`, `"max-autotune"` |
| `compile_fullgraph` | `bool` | `False` | Требовать полный граф для компиляции (быстрее, но может не скомпилироваться) |
| `compile_dynamic` | `bool` | `True` | Поддержка динамических размеров входных данных |
| `debug_mode` | `bool` | `True` | Вывод отладочной информации о производительности и трансферах |
| `**kwargs` | `Any` | — | Параметры конкретного метода сегментации |

### Параметры `segment(image, **kwargs)`
| Параметр | Тип | Описание |
|----------|-----|----------|
| `image` | `ImageInput` | Входное изображение: `str`, `PIL.Image`, `np.ndarray`, `torch.Tensor` |
| `export_mode` | `bool` | Если `True`, использовать TRT/ONNX-совместимые операции (без `.item()`, `torch.histc` и т.д.) |
| `**kwargs` | `Any` | Переопределение параметров метода при вызове |

### Параметры `segment_with_mask(image, alpha=0.9, **kwargs)`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `alpha` | `float` | `0.9` | Прозрачность наложения: 0.0=только фото, 1.0=только маска |
| `color` | `Tuple[int,int,int]` | `(255, 0, 0)` | Цвет маски в формате RGB |
| `precision` | `str` | `None` | Точность вычислений для визуализации: `"fp32"`, `"fp16"`, `"bf16"` |

### PrecisionManager: выбор точности
```python
# Автоматический выбор оптимальной точности
precision_manager = PrecisionManager(default_precision="fp32")
optimal = precision_manager.get_optimal_precision(
    device=torch.device("cuda"),
    operation="inference",
    memory_limit=4.0  # ГБ
)
# Возвращает: "bf16" для Ampere+, "fp16" для Turing, "fp32" для старых GPU

# Проверка поддержки типов
if PrecisionManager.can_use_bf16(torch.device("cuda:0")):
    print("✅ BF16 поддерживается")
```

## 📚 Справочник методов

### 🔹 Пороговые методы (14 алгоритмов)
| Метод | Параметры | Описание | Особенности PyTorch-реализации |
|-------|-----------|----------|------------------------------|
| `global_thresholding` | `threshold=0.5` | Фиксированный порог | Векторизованное сравнение: `(gray > thresh_t)` |
| `otsu_thresholding` | `num_bins=256` | Автоматический порог по максимизации межклассовой дисперсии | Векторизованный критерий, `torch.histc` → fallback на `stack+sum` для TRT |
| `adaptive_thresholding` | `block_size=11`, `C=2`, `method="gaussian"` | Локальный порог на основе гауссовой окрестности | Сепарабельная свёртка через `_get_gaussian_kernel_1d` + outer product |
| `threshold_niblack` | `window_size=15`, `k=-0.2` | `T = μ + k·σ` | Локальная статистика через `_local_stats_torch` (conv2d для mean/var) |
| `threshold_sauvola` | `window_size=15`, `k=0.2`, `r=128` | `T = μ·[1 + k·(σ/R - 1)]` | Улучшенная формула для низкого контраста, защита от `log(0)` через `clamp` |
| `threshold_bernsen` | `window_size=15`, `contrast_threshold=0.1` | Порог = (min+max)/2 при достаточном контрасте | Локальный min/max через `F.max_pool2d` и `-F.max_pool2d(-x)` |
| `threshold_phansalkar` | `window_size=15`, `k=0.25`, `r=128`, `m=0.5` | Улучшенный Ниблак для низкого контраста | Эмпирическая формула, полностью векторизована |
| `threshold_kittler_illingworth` | `num_bins=256` | Минимизация ошибки классификации при гауссовом предположении | Векторизованный критерий без Python-циклов, `gather` вместо прямой индексации для TRT |
| `threshold_entropy_kapur` | `num_bins=256` | Максимизация суммы энтропий фона/объекта | Векторизованная энтропия, защита от `log(0)` через `eps` |
| `threshold_triangle` | `num_bins=256` | Геометрический метод для асимметричных гистограмм | Векторизованный расчёт расстояний до линии, `float-умножение` вместо `bool` для TRT |
| `threshold_multi_otsu` | `n_thresholds=2` | Многоуровневый Оцу для разделения на >2 классов | Рекурсивный поиск порогов, экспорт-режим → упрощённый одно-пороговый Оцу |
| `threshold_percentile` | `percentile=90` | Порог как заданный процентиль распределения | `torch.quantile` на GPU, fallback на гистограммную аппроксимацию для TRT |
| `threshold_local_contrast` | `window_size=15`, `contrast_factor=0.1` | Порог на основе локального контраста | Локальное среднее через conv2d, квантиль контраста через `torch.quantile` |

### 🔹 Детекторы границ (10 алгоритмов)
| Метод | Параметры | Описание | Особенности реализации |
|-------|-----------|----------|----------------------|
| `sobel_edge` | `threshold=0.1`, `normalize=True` | Градиент через свёртку с ядрами Собеля | Кэшированные ядра через `@lru_cache`, `_safe_conv2d` для выравнивания dtype |
| `canny_edge` | `low=0.1`, `high=0.3`, `sigma=1.0` | Многоэтапный детектор с гистерезисом | Векторизованный NMS через `torch.roll`, гистерезис через conv2d (2 итерации) |
| `prewitt_edge` | `threshold=0.1` | Градиент с равными весами [1,1,1] | Аналогично Sobel, но с более простыми ядрами |
| `scharr_edge` | `threshold=0.1` | Оптимизированные ядра для точного градиента | Ядра [-3,0,3; -10,0,10; -3,0,3] для лучшей ротационной симметрии |
| `laplacian_edge` | `sigma=1.0`, `threshold=0.1` | Лапласиан после гауссова сглаживания | Предварительный Gaussian blur через `tv_gaussian_blur`, ядро 3×3 или 5×5 |
| `log_edge` | `sigma=1.0`, `threshold=0.1` | Laplacian of Gaussian, нулевые пересечения | Векторизованный zero-crossing через `torch.sign` и сдвиги |
| `dog_edge` | `sigma1=1.0`, `sigma2=2.0`, `threshold=0.1` | Разность Гауссианов, аппроксимация LoG | Сепарабельные гауссовы ядра, эффективнее прямого LoG |
| `marr_hildreth_edge` | `sigma=1.5`, `threshold=0.1` | Улучшенный LoG с проверкой магнитуды у соседей | Лапласиан 5×5, векторизованный zero-crossing с паддингом |
| `gradient_magnitude_direction` | `threshold=0.1` | Магнитуда + подавление немаксимумов по направлению | Квантование направления на 4 сектора, векторизованный NMS через `F.pad` |
| `phase_congruency_edge` | `nscales=4`, `norientations=4`, `threshold=0.3` | Инвариантная к контрасту детекция через фазы Фурье | FFT-банк фильтров Log-Gabor, оценка шума через MAD, экспорт-режим → DoG-аппроксимация |

### 🔹 Региональные, кластеризация, активные контуры
| Категория | Метод | Параметры | Особенности PyTorch-реализации |
|-----------|-------|-----------|------------------------------|
| **Региональные** | `region_growing` | `seed=None`, `tolerance=0.1`, `max_iterations=H*W` | Векторизованный BFS: обработка "волны" пикселей за итерацию, без `deque` |
| | `split_and_merge` | `min_size=50`, `threshold=20.0` | Рекурсивное квадро-деление на GPU, выбор второго по величине региона |
| | `floodfill` | `points=None`, `tolerance=0.15`, `connectivity=4` | Много-точечная заливка с векторизованным обновлением маски |
| **Кластеризация** | `kmeans_segmentation` | `k=3`, `max_iter=50`, `init="kmeans++"` | `torch.cdist` для расстояний, векторизованное обновление центроидов |
| | `dbscan_segmentation` | `eps=0.1`, `min_samples=10`, `downsample=0.5` | Fallback на sklearn при необходимости, автоматический downsample + интерполяция |
| | `meanshift` | `bandwidth=0.5`, `spatial_radius=35`, `color_radius=60` | Построение признаков [x/sr, y/sr, R/cr, G/cr, B/cr], sklearn fallback |
| **Активные контуры** | `active_contour` | `alpha=0.01`, `beta=0.1`, `gamma=0.001`, `max_iter=250` | FFT-решение для циклической матрицы внутренней энергии, `grid_sample` для интерполяции сил |
| | `gvf_contour` | `mu=0.2`, `iterations=20` | Итеративное сглаживание векторного поля через conv2d |
| | `morphological_snakes` | `iterations=100`, `smoothing=1`, `threshold=0.5` | Векторизованные морфологические операции через conv2d с единичным ядром |
| | `chan_vese` | `mu=0.25`, `lambda1=1.0`, `lambda2=1.0`, `dt=0.5`, `eps=1.0` | Регуляризованные Хевисайд/Дирак через `arctan`, векторизованное уравнение эволюции уровня |

### 🔹 Watershed, суперпиксели, интерактивные
| Категория | Метод | Параметры | Особенности реализации |
|-----------|-------|-----------|----------------------|
| **Watershed** | `watershed` | `connectivity=4`, `gradient_method="sobel"`, `use_numba_fallback=True` | Векторизованная приоритетная очередь через `torch.sort` + `searchsorted`, Numba fallback для CPU |
| | `random_walker` | `beta=130`, `mode="scipy"`, `tol=1e-3` | Разреженные тензоры для лапласиана, решатели: `jacobi`/`cg` (PyTorch) / `scipy` (fallback) |
| **Суперпиксели** | `quickshift` | `kernel_size=5`, `max_dist=10`, `ratio=1.0`, `downsample=0.5` | Numpy-реализация с оценкой плотности через выборку, интерполяция меток обратно |
| | `slic` | `n_segments=100`, `compactness=10.0`, `sigma=0.0`, `enforce_connectivity=True` | K-means в пространстве [Lab, x, y], принудительная связность через scipy.ndimage |
| | `felzenszwalb` | `scale=100`, `sigma=0.8`, `min_size=50` | Обёртка над `skimage.segmentation.felzenszwalb`, fallback на k-means |
| **Интерактивные** | `grabcut` | `rect=None`, `num_iterations=5`, `n_components=5` | GMM через `torch.distributions.MultivariateNormal`, EM-алгоритм на GPU |

## ⚡ Производительность и оптимизации

### Относительная скорость методов (на изображении 512×512, RTX 3090, bf16)
```
✅ Быстро (<5 мс):
   - global_thresholding, otsu_thresholding
   - sobel_edge, prewitt_edge, scharr_edge, laplacian_edge
   - kmeans_segmentation (k≤5)

⚠️ Средне (5–50 мс):
   - adaptive_thresholding, niblack, sauvola, bernsen
   - canny_edge (с гистерезисом), log_edge, dog_edge
   - region_growing, split_and_merge, floodfill
   - morphological_snakes, gvf_contour
   - watershed (векторизованный)

❌ Медленно (50–500+ мс):
   - kittler_illingworth, entropy_kapur, triangle (гистограммные методы)
   - phase_congruency_edge (FFT + банк фильтров)
   - active_contour, chan_vese (итеративные)
   - random_walker (разреженная система)
   - quickshift, slic, felzenszwalb (numpy fallback)
```

### Рекомендации по оптимизации
1. **Используйте `torch.compile` для статических графов:**
   ```python
   segmenter = TorchSegmenter2(
       method="sobel_edge",
       use_compile=True,
       compile_fullgraph=True,  # Если метод поддерживает
       compile_dynamic=False    # Если размер изображения фиксирован
   )
   ```

2. **Выбирайте точность под устройство:**
   ```python
   # Авто-выбор через PrecisionManager
   precision = segmenter.precision_manager.get_optimal_precision(
       device=torch.device("cuda"),
       operation="inference",
       memory_limit=8.0
   )
   segmenter = TorchSegmenter2(method="chan_vese", precision=precision)
   ```

3. **Кэшируйте результаты для повторяющихся входов:**
   ```python
   # Включите кэширование через segment_with_cache
   mask = segmenter.segment_with_cache(image, use_cache=True)
   # При повторном вызове с тем же изображением — мгновенный возврат
   ```

4. **Отключайте визуализацию для бенчмарков:**
   ```python
   # Используйте segment() вместо segment_with_mask() если не нужен overlay
   mask = segmenter.segment(image)  # Быстрее на ~15-30%
   ```

5. **Используйте `export_mode=True` для ONNX/TensorRT:**
   ```python
   # Избегает операций, не поддерживаемых в экспортных форматах:
   # - .item(), torch.histc, bool-индексация, aten::Int.Tensor
   mask = segmenter.segment(image, export_mode=True)
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
1. **Включите `debug_mode=True` для авто-диагностики:**
   ```python
   segmenter = TorchSegmenter2(method="chan_vese", debug_mode=True)
   mask = segmenter.segment(image)
   # Вывод:
   # 🔍 [DEBUG MODE] Запуск автоматической диагностики для метода 'chan_vese'...
   # ⏱️  Среднее время выполнения: 45.32 ms
   # 🔄 Трансферы данных: ✅ Лишних трансферов CPU↔GPU не обнаружено
   # 💾 Устройство: cuda:0
   # ⚡ Режим компиляции: ON
   ```

2. **Проверьте выравнивание dtype для conv2d:**
   ```python
   # Метод _safe_conv2d автоматически приводит ядро к dtype входа
   # Но для ручной отладки:
   print(f"gray.dtype: {gray.dtype}, kernel.dtype: {kernel.dtype}")
   assert gray.dtype == kernel.dtype, "Dtype mismatch for conv2d!"
   ```

3. **Тестируйте на маленьком изображении перед запуском на полном датасете:**
   ```python
   small_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
   mask = segmenter.segment(small_image)  # Быстрая проверка конвейера
   assert mask.shape == (128, 128), "Unexpected output shape!"
   ```

4. **Используйте `profile_with_transfer_detection` для выявления лишних трансферов:**
   ```python
   profile = segmenter.profile_with_transfer_detection(image)
   if profile["transfer_warnings"]:
       print("⚠️  Проблемные трансферы:")
       for warn in profile["transfer_warnings"]:
           print(f"   {warn}")
   ```

## 🤝 Интеграция с другими модулями проекта

| Модуль | Использование NewTorchSegmenter |
|--------|--------------------------------|
| `BatchClassicTester` | Массовое тестирование согласованности реализаций (PyTorch vs OpenCV) |
| `TorchImplementationValidator` | Валидация численной точности методов против эталонных реализаций |
| `CpuCudaBenchmark` | Бенчмарк производительности: CPU vs GPU, fp32 vs fp16 vs bf16 |
| `SegmentationTester` | Универсальное тестирование через `add_method()` с поддержкой всех 50+ методов |
| `VisualizationTool` | Визуализация предсказаний с наложением через `segment_with_mask()` |

### Пример интеграции с PyTorch DataLoader
```python
from torch.utils.data import DataLoader
from datasets.ADE20KDataset import ADE20KDataset
from segmenters.NewTorchSegmenter import TorchSegmenter2

# Подготовка данных
dataset = ADE20KDataset(root_dir="./data/ade20k", split="val", augment=False)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Инициализация сегментера
segmenter = TorchSegmenter2(
    method="otsu_thresholding",
    device="cuda",
    precision="bf16",
    use_compile=True
)

# Инференс на батче
for batch in loader:
    images = batch["image"]  # [B, 3, H, W]
    
    # Пакетная сегментация (требует модификации метода под batch)
    masks = []
    for img in images:
        mask = segmenter.segment(img)  # (1, 1, H, W)
        masks.append(mask)
    
    # ... оценка качества, сохранение результатов ...
```

### Экспорт в TorchScript / TensorRT
```python
# Экспорт в TorchScript
success = segmenter.export_to_jit(
    method_name="sobel_edge",
    output_path="./exported",
    example_input=torch.randn(1, 3, 256, 256, device="cuda")
)

# Экспорт в TensorRT (требует torch-tensorrt)
success = segmenter._try_export_to_tensorrt(
    example_input=torch.randn(1, 3, 256, 256, device="cuda"),
    trt_path="./exported/sobel_edge_fp16.trt",
    precision="fp16"
)
```

## 📦 Зависимости

### Обязательные
```text
torch>=2.0.0           # PyTorch с поддержкой torch.compile, AMP, sparse tensors
torchvision>=0.15.0    # gaussian_blur, transforms
numpy>=1.20.0          # Fallback-реализации, предобработка
Pillow>=9.0.0          # Загрузка изображений
```

### Опциональные (для расширенной функциональности)
```text
numba>=0.56.0          # Ускорение region_growing, watershed на CPU
scipy>=1.7.0           # ndimage для морфологии, fallback для watershed
scikit-learn>=1.0.0    # DBSCAN, MeanShift fallback
scikit-image>=0.19.0   # Felzenszwalb fallback
torch-tensorrt>=1.4.0  # Экспорт в TensorRT Engine
```

### Установка
```bash
# Базовая установка
pip install torch torchvision numpy Pillow

# Полная установка (все опциональные зависимости)
pip install numba scipy scikit-learn scikit-image torch-tensorrt

# Проверка установки
python -c "from segmenters.NewTorchSegmenter import TorchSegmenter2; print('✅ OK')"
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

> 💡 **Совет:** Для максимальной производительности на вашем устройстве используйте `precision_manager.get_optimal_precision()` и включайте `use_compile=True` для методов, поддерживающих `fullgraph=True`. Избегайте динамических параметров (например, изменяющегося `window_size`) при компиляции — фиксируйте их заранее.

```python
# Оптимальная конфигурация для инференса на RTX 3090
segmenter = TorchSegmenter2(
    method="chan_vese",
    device="cuda",
    precision="bf16",  # Авто-выбор через get_optimal_precision()
    use_compile=True,
    compile_mode="reduce-overhead",
    compile_fullgraph=False,  # chan_vese содержит условную остановку
    mu=0.25,
    max_iter=100
)
```