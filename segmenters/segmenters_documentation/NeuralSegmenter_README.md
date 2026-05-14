# 🧠🎨 NeuralSegmenter — Универсальный нейросетевой сегментатор

## 📖 Описание
Модуль `NeuralSegmenter.py` предоставляет **единый интерфейс** для инференса современных архитектур нейронной семантической и инстанс-сегментации.

> ⚠️ **Важно:** Данный модуль работает исключительно с **нейросетевыми методами**. Для классических алгоритмов используйте `OpenCVSegmenter` или `SklearnSegmenter`.

## ✨ Ключевые возможности
### 🏗️ Поддерживаемые архитектуры (18+ вариантов)

| Семейство | Модели | Датасеты | Особенности |
|-----------|--------|----------|-------------|
| **HuggingFace Transformers** (5) | `segformer`, `mask2former`, `oneformer`, `dpt`, `upernet` | ADE20K, COCO, Cityscapes | Предобученные веса, авто-процессоры, поддержка safetensors |
| **Segmentation Models PyTorch** (5) | `unet_smp`, `fpn_mit`, `psp_mit`, `deeplab_smp`, `segnet` | Любые (гибкая настройка) | 50+ encoder'ов, предобучение на ImageNet, transfer learning |
| **Torchvision** (3) | `deeplab_tv`, `fcn_tv`, `maskrcnn_tv` | COCO, VOC | Официальные реализации, предобучение на COCO+VOC |
| **Instance Segmentation** (4) | `sam`, `mobile_sam`, `sam2`, `yolov8` | Zero-shot / COCO | Промпт-управляемая сегментация, детекция + маска |

### 🎨 Поддержка палитр и имён классов
| Датасет | Классов | Палитра | Метод получения |
|---------|---------|---------|----------------|
| **ADE20K** | 150 | `ade_palette()` | `get_ade_class_names()` |
| **COCO** | 80 | `coco_palette()` | `get_coco_class_names()` |
| **Cityscapes** | 19 / 34 | `cityscapes_palette()` | `get_cityscapes_class_names()` |
| **CheXpert** | 14 | `chexpert_observation_palette()` | `get_chexpert_observation_class_names()` |
| **ISIC 2018** | 2 (binary) | `binary_palette()` | `get_isic_class_names()` |

### 🔄 Единый интерфейс для всех моделей
```python
from segmenters.NeuralSegmenter import NeuralSegmenter

# Загрузка предобученной модели
segmenter = NeuralSegmenter(
    model_type="segformer",
    model_name="nvidia/segformer-b5-finetuned-ade-640-640",
    device="cuda",
    palette=NeuralSegmenter.ade_palette()  # Опционально
)

# Базовая сегментация: возврат бинарной маски {0, 255}
mask = segmenter.segment("image.jpg")  # BinaryMask: np.ndarray[H, W], dtype=uint8

# Сегментация + визуализация: возврат (overlay, mask)
overlay, mask = segmenter.segment_with_mask("image.jpg", alpha=0.7)

# Детальная информация с метриками (при наличии GT)
seg_map, info = segmenter.predict_segmentation_map(
    "image.jpg",
    gt_mask=ground_truth,
    verbose=True
)
print(f"IoU: {info.get('iou', 'N/A')}, mIoU: {info.get('miou', 'N/A')}")
```

### 🎚️ Поддержка форматов входа
| Тип | Пример | Обработка |
|-----|--------|-----------|
| `str` (путь) | `"image.jpg"` | Загрузка через `PIL.Image.open()`, конвертация в RGB |
| `str` (URL) | `"https://example.com/img.png"` | `requests.get()` → `BytesIO` → `PIL.Image` |
| `PIL.Image` | `Image.open("img.png")` | `.convert("RGB")` |
| `np.ndarray` | `np.random.randint(0,255,(512,512,3))` | Конвертация в PIL, поддержка 2D/3D/4D |
| `torch.Tensor` | `torch.rand(3,512,512)` | `.permute()`, `.cpu()`, float→uint8 при необходимости |

## 🚀 Быстрый старт
### Базовое использование: сегментация на ADE20K
```python
from segmenters.NeuralSegmenter import NeuralSegmenter
import matplotlib.pyplot as plt

# Инициализация
segmenter = NeuralSegmenter(
    model_type="segformer",
    model_name="nvidia/segformer-b5-finetuned-ade-640-640",
    device="cuda"
)

# Сегментация
mask = segmenter.segment("street_scene.jpg")

# Визуализация
overlay = segmenter.segment_image("street_scene.jpg", alpha=0.6)
overlay.save("result_overlay.png")

# Отображение с подписями классов
seg_map, info = segmenter.predict_segmentation_map(
    "street_scene.jpg",
    class_names=NeuralSegmenter.get_ade_class_names(),
    verbose=True
)
```

### Инстанс-сегментация: SAM / YOLOv8
```python
# SAM (zero-shot, промпт-управляемый)
sam_segmenter = NeuralSegmenter(
    model_type="sam",
    model_name="mobile_sam.pt",  # или "sam_vit_b_01ec64.pth"
    device="cuda"
)
# Для SAM требуется промпт (точка/бокс) — передаётся через kwargs
mask, info = sam_segmenter.predict_segmentation_map(
    "object.jpg",
    point_coords=[[250, 300]],  # Координаты промпта
    point_labels=[1]  # 1 = объект, 0 = фон
)

# YOLOv8-seg (детекция + маска)
yolo_segmenter = NeuralSegmenter(
    model_type="yolov8",
    model_name="yolov8n-seg.pt",  # n/s/m/l/x варианты
    device="cuda"
)
mask, info = yolo_segmenter.predict_segmentation_map("crowd.jpg")
print(f"Detected objects: {info.get('n_objects', 0)}")
```

### Сравнение архитектур на одном изображении
```python
models = {
    "SegFormer-B5": ("segformer", "nvidia/segformer-b5-finetuned-ade-640-640"),
    "Mask2Former": ("mask2former", "facebook/mask2former-swin-base-ade-semantic"),
    "DeepLabV3+": ("deeplab_tv", None),
    "U-Net+ResNet34": ("unet_smp", None),
}

results = {}
for name, (mtype, mname) in models.items():
    seg = NeuralSegmenter(model_type=mtype, model_name=mname, device="cuda")
    start = time.time()
    mask = seg.segment("test.jpg")
    results[name] = {
        "mask": mask,
        "time": time.time() - start,
        "coverage": (mask > 0).sum() / mask.size * 100
    }
    print(f"{name:20s}: {results[name]['time']:.3f}s, coverage: {results[name]['coverage']:.1f}%")
```

### Работа с медицинскими датасетами (ISIC, CheXpert)
```python
# ISIC 2018: бинарная сегментация кожных поражений
isic_segmenter = NeuralSegmenter(
    model_type="unet_smp",
    model_name="isic_unet_checkpoint.pth",
    checkpoint_path="./checkpoints/isic_best.pth",
    num_classes=2,
    palette=NeuralSegmenter.binary_palette()
)

mask, info = isic_segmenter.predict_segmentation_map(
    "lesion.jpg",
    class_names=NeuralSegmenter.get_isic_class_names(),
    gt_mask=ground_truth
)
print(f"Dice: {info.get('dice', 'N/A'):.3f}")

# CheXpert: классификация + сегментация лёгких
chexpert_segmenter = NeuralSegmenter(
    model_type="dpt",
    model_name="Intel/dpt-large-ade",  # или специализированная модель
    num_classes=14,
    palette=NeuralSegmenter.chexpert_observation_palette()
)
```

## ⚙️ Конфигурация
### Параметры `__init__()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model_type` | `str` | `"segformer"` | Тип модели: `"segformer"`, `"mask2former"`, `"unet_smp"`, `"yolov8"`, и т.д. |
| `model_name` | `str` | HF model ID | Имя модели в HuggingFace Hub (для HF-моделей) |
| `variant` | `Optional[str]` | `None` | Вариант модели (например, `"b5"` для SegFormer, `"fcn_resnet50"` для FCN) |
| `device` | `Optional[str]` | `None` (авто) | Устройство: `"cuda"` или `"cpu"` |
| `local_path` | `Optional[str]` | `None` | Локальный путь к модели (альтернатива `model_name`) |
| `num_classes` | `int` | `150` | Количество выходных классов (важно для SMP/torchvision) |
| `palette` | `Optional[List[List[int]]]` | `None` (ADE20K) | Цветовая палитра для визуализации |
| `checkpoint_path` | `Optional[str]` | `None` | Путь к `.pth` чекпоинту для SMP/torchvision-моделей |
| `**kwargs` | `Any` | — | Дополнительные параметры для `NeuralModelFactory.create_model()` |

### Возвращаемые значения основных методов
| Метод | Возвращает | Описание |
|-------|-----------|----------|
| `segment()` | `BinaryMask` | Бинарная маска `{0, 255}`, форма `(H, W)`, dtype `uint8` |
| `segment_with_mask()` | `Tuple[np.ndarray, np.ndarray]` | `(overlay[H,W,3], mask[H,W])` для визуализации |
| `segment_image()` | `PIL.Image` | Изображение с наложенной цветной маской (режим "RGB") |
| `predict_segmentation_map()` | `Tuple[np.ndarray, Dict]` | `(seg_map[H,W], info_dict)` с метриками и метаданными |
| `detailed_segmentation()` | `Dict[str, Any]` | Полный набор результатов: оригинал, карта, overlay, распределение классов |

## 📚 Справочник методов
### 🔹 Основные методы сегментации
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `segment()` | `image`, `**kwargs` | Базовая бинарная сегментация | `BinaryMask` |
| `segment_with_mask()` | `image`, `alpha=0.9`, `**kwargs` | Сегментация + визуализация | `(overlay, mask)` |
| `segment_image()` | `image`, `alpha=0.9` | Возврат `PIL.Image` с overlay | `PIL.Image` |
| `predict_segmentation_map()` | `image`, `verbose`, `class_names`, `gt_mask` | Инференс с метаданными и метриками | `(seg_map, info_dict)` |
| `detailed_segmentation()` | `image` | Полный анализ с распределением классов | `Dict[str, Any]` |

### 🔹 Утилиты загрузки и предобработки
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `load_image()` | `input_image` | Загрузка из пути/URL/np.ndarray/torch.Tensor | `PIL.Image` (RGB) |
| `prepare_mask_for_overlay()` | `mask_input` | Конвертация маски к 2D `uint8` для overlay | `np.ndarray[H,W]` |
| `_resize_mask_to_original()` | `mask`, `target_size` | Ресайз маски с nearest-neighbor интерполяцией | `np.ndarray` |

### 🔹 Статические методы: датасеты и палитры
| Метод | Описание | Возвращает |
|-------|----------|-----------|
| `get_ade_class_names()` | Имена 150 классов ADE20K | `Dict[int, str]` |
| `ade_palette()` | Палитра ADE20K (150×[R,G,B]) | `List[List[int]]` |
| `get_coco_class_names()` | Имена 80 классов COCO | `Dict[int, str]` |
| `coco_palette()` | Палитра COCO | `List[List[int]]` |
| `get_cityscapes_class_names()` | 19 классов Cityscapes | `Dict[int, str]` |
| `cityscapes_palette()` | Палитра Cityscapes | `List[List[int]]` |
| `get_isic_class_names()` | Бинарные классы ISIC | `Dict[int, str]` |
| `binary_palette()` | Палитра для бинарной сегментации | `List[List[int]]` |

### 🔹 Мета-информация
| Метод | Описание | Возвращает |
|-------|----------|-----------|
| `get_class_info()` | Информация о классах модели (из config или последнего Conv2d) | `Dict[str, Any]` |

## 🔄 Конвейер инференса: загрузка → препроцессинг → forward → постобработка
### Делегирование стратегии `segment_image_unified`
```python
# Все методы инференса используют единую стратегию из utils.strategies
def predict_segmentation_map(self, input_image, **kwargs):
    return infer_unified(
        model=self.model,
        processor=self.processor,
        image_input=input_image,
        model_type=self.model_type_str,
        device=str(self.device),
        palette=self.palette,
        num_classes=self.num_classes,
        **kwargs  # class_names, gt_mask, verbose, ...
    )
```

### Обработка выхода в зависимости от типа модели
| Тип модели | Выход `model(input)` | Постобработка |
|------------|---------------------|---------------|
| **HF Transformers** | `ModelOutput(logits=[B,C,H,W])` | `logits.argmax(1)` → resize to original |
| **SMP / Torchvision** | `Tensor [B, C, H, W]` или `dict['out']` | `argmax(1)` → resize to original |
| **SAM / YOLOv8** | `Results` объект с масками/боксами | Извлечение масок → бинаризация → resize |

### Блендинг для визуализации
```python
# Создание цветной маски из палитры
palette_array = np.array(self.palette, dtype=np.uint8)
color_mask = np.zeros((H, W, 3), dtype=np.uint8)
for label in np.unique(seg_map):
    if label < len(palette_array):
        color_mask[seg_map == label] = palette_array[label]

# Блендинг оригинала и маски
overlay = (orig_arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)
```

## 📊 Метрики качества (при наличии Ground Truth)
При передаче `gt_mask` в `predict_segmentation_map()` вычисляются:

| Метрика | Описание | Диапазон | Интерпретация |
|---------|----------|----------|---------------|
| **IoU** | Intersection over Union | [0.0, 1.0] | Основной критерий для семантической сегментации |
| **Dice** | Dice / F1 coefficient | [0.0, 1.0] | Более устойчив к дисбалансу классов |
| **mIoU** | Mean IoU по всем классам | [0.0, 1.0] | Стандарт для ADE20K/COCO/Cityscapes |
| **Pixel Accuracy** | Доля верно классифицированных пикселей | [0.0, 1.0] | Может быть завышена при дисбалансе |
| **Precision/Recall** | Точность и полнота детекции | [0.0, 1.0] | Важны для инстанс-сегментации |
| **Hausdorff Distance** | Макс. расстояние между контурами | [0, ∞) | Чувствительна к выбросам, важна для границ |

## ⚡ Производительность и оптимизации
### Относительная скорость инференса (на изображении 512×512, RTX 3090)
```
✅ Быстро (<50 мс):
   - unet_smp + resnet34
   - fcn_tv + resnet50
   - deeplab_tv + resnet101
   - mobile_sam

⚠️ Средне (50–200 мс):
   - segformer-b0/b1/b2
   - fpn_mit, psp_mit
   - maskrcnn_tv (с постобработкой)
   - yolov8n-seg / yolov8s-seg

❌ Медленно (200–1000+ мс):
   - segformer-b3/b4/b5
   - mask2former-swin-large
   - oneformer-swin-large
   - sam_vit_b / sam2 (zero-shot, зависит от промптов)
   - yolov8m/l/x-seg
```

### Рекомендации по оптимизации
1. **Mixed Precision** для инференса:
   ```python
   with torch.autocast(device_type="cuda", dtype=torch.float16):
       output = model(input_tensor)
   ```

2. **torch.compile()** (PyTorch 2.0+) для SMP/torchvision:
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```

3. **Ресайз входа** для больших изображений:
   ```python
   # SegFormer обучен на 640×640 — ресайз ускоряет инференс
   img = Image.open("large.jpg").resize((640, 640))
   ```

4. **Кэширование процессоров** для HF-моделей:
   ```python
   # Процессор можно переиспользовать для батча
   inputs = processor(images=batch, return_tensors="pt").to(device)
   ```

5. **Выгрузка на CPU** для редко используемых моделей:
   ```python
   model.cpu()
   torch.cuda.empty_cache()
   ```

## 🛠️ Обработка ошибок и устойчивость
### Проверка зависимостей при инициализации
```python
if not TRANSFORMERS_AVAILABLE:
    raise ImportError(
        "transformers library is required. Install with: pip install transformers"
    )
```

### Безопасная загрузка модели
```python
try:
    model_tuple = NeuralModelFactory.create_model(
        model_type=self.model_type,
        model_name=self.model_name,
        device=cast(Literal["cuda", "cpu"], self.device),
        checkpoint_path=cp_path,
        **kwargs
    )
    self.model, self.processor, self.model_type_str = model_tuple
except Exception as e:
    logger.error(f"Failed to load model {self.model_name}: {e}")
    raise
```

### Валидация маски перед визуализацией
```python
def prepare_mask_for_overlay(self, mask_input):
    # Конвертация PIL → numpy
    if isinstance(mask_input, Image.Image):
        mask = np.array(mask_input)
    else:
        mask = np.array(mask_input)
    
    # Удаление лишних измерений
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask.squeeze(2)
    elif mask.ndim == 3 and mask.shape[2] == 3:
        # RGB-маска → используем первый канал
        mask = mask[:, :, 0]
    
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D after processing, got {mask.ndim}D")
    return mask
```

### Рекомендации по отладке
1. **Включите verbose для детального логирования**:
   ```python
   seg_map, info = segmenter.predict_segmentation_map(
       "test.jpg",
       verbose=True,  # Вывод времени, классов, метрик
       class_names=NeuralSegmenter.get_ade_class_names()
   )
   ```

2. **Проверьте устройство и режим модели**:
   ```python
   print(f"Device: {next(segmenter.model.parameters()).device}")
   print(f"Training mode: {segmenter.model.training}")  # Должно быть False
   ```

3. **Тестируйте на маленьком изображении**:
   ```python
   small_img = Image.new("RGB", (224, 224), color="red")
   mask = segmenter.segment(small_img)  # Быстрая проверка конвейера
   ```

4. **Анализируйте распределение классов**:
   ```python
   result = segmenter.detailed_segmentation("scene.jpg")
   for name, stats in result["class_distribution"].items():
       print(f"{name:20s}: {stats['pixel_count']:6d} px ({stats['percentage']:.2f}%)")
   ```

## 🤝 Зависимости
```text
torch>=1.9                    # Основные тензорные операции, autograd
torchvision>=0.10             # DeepLab, FCN, Mask R-CNN
segmentation-models-pytorch>=0.2  # U-Net, FPN, PSPNet с 50+ encoder'ами
transformers>=4.25           # SegFormer, Mask2Former, OneFormer (опционально)
huggingface_hub>=0.10        # Загрузка моделей из HF Hub (опционально)
ultralytics>=8.0             # SAM, YOLOv8 (опционально)
Pillow>=8.0                  # Загрузка и обработка изображений
numpy>=1.20                  # Массивы, метрики
scipy>=1.7                   # zoom для ресайза масок
requests>=2.28               # Загрузка изображений по URL
```

### Опциональные зависимости (градуированная установка)
```bash
# Базовый набор (классические + базовые нейросети)
pip install torch torchvision segmentation-models-pytorch Pillow numpy scipy

# + HuggingFace Transformers
pip install transformers huggingface_hub

# + Instance Segmentation
pip install ultralytics

# + Все зависимости
pip install -r requirements_neural.txt
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование NeuralSegmenter |
|--------|------------------------------|
| `NeuralModelFactory` | Создание и загрузка моделей через `create_model()` |
| `ModelTrainer` | Обучение моделей, чекпоинты которых загружаются в `NeuralSegmenter` |
| `SegmentationTester` | Универсальное тестирование: добавление нейросетей через `add_method()` |
| `BatchClassicTester2` | Сравнение классических и нейросетевых методов на одном датасете |
| `CpuCudaBenchmark` | Бенчмарк производительности: сравнение времени инференса |
| `utils.strategies.segment_image_unified` | Делегирование инференса единой стратегии |

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