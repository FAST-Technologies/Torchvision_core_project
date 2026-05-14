# 🔄 Strategies — Универсальные стратегии инференса для нейросетевой сегментации

## 📖 Описание
Модуль `utils/strategies.py` предоставляет **единую точку входа** для выполнения инференса современных архитектур нейронной сегментации через паттерн "Стратегия".

> ⚠️ **Важно:** Данный модуль отвечает за **инференс и постобработку** моделей. Для создания моделей используйте `NeuralModelFactory`, для обучения — `NeuralTrainer`.

## ✨ Ключевые возможности
### 🏗️ Поддерживаемые архитектуры через единый интерфейс

| Семейство | Модели | Стратегия | Особенности |
|-----------|--------|-----------|-------------|
| **HuggingFace Transformers** (6) | `segformer`, `mask2former`, `oneformer`, `dpt`, `upernet`, `maskformer` | `infer_*()` с `post_process_semantic_segmentation` | Авто-ресайз к оригиналу, поддержка `target_sizes` |
| **Torchvision** (3) | `deeplab_tv`, `fcn_tv`, `maskrcnn_tv` | `infer_*_torchvision()` с ImageNet-нормализацией | Паддинг под `output_stride`, конвертация instance→semantic |
| **SMP/Custom** (10+) | `unet_smp`, `fpn_mit`, `psp_mit`, `segnet`, ... | `infer_smp_model_fixed()` с авто-препроцессингом | `smp.encoders.get_preprocessing_fn()`, кроппинг паддинга |
| **Instance Segmentation** (4) | `sam`, `mobile_sam`, `sam2`, `yolov8` | `infer_sam()`, `infer_yolov8()` | Конвертация масок инстансов в семантическую карту |

### 🔄 Единая функция `segment_image_unified()`
```python
from utils.strategies import segment_image_unified

# Инференс для любой модели через единый интерфейс
overlay, info = segment_image_unified(
    model=loaded_model,
    processor=loaded_processor,  # или None для SMP/torchvision
    image_input="image.jpg",     # str, Path, PIL, np.ndarray, torch.Tensor
    model_type="segformer",      # из INFERENCE_STRATEGIES
    device="cuda",
    alpha=0.7,                   # прозрачность наложения
    palette=ade_palette(),       # цветовая палитра
    gt_mask=ground_truth,        # опционально, для метрик
    verbose=True
)

# Доступ к результатам
print(f"Inference time: {info['inference_time_ms']:.2f}ms")
print(f"Unique classes: {info['unique_classes']}")
if info['metrics']:
    print(f"IoU: {info['metrics']['iou']:.3f}")
overlay.save("result.png")
```

### 🎨 Поддержка форматов входа
| Тип | Пример | Обработка |
|-----|--------|-----------|
| `str` (путь) | `"image.jpg"` | `PIL.Image.open()`, конвертация в RGB |
| `str` (URL) | `"https://example.com/img.png"` | `requests.get()` → `BytesIO` → `PIL.Image` |
| `PIL.Image` | `Image.open("img.png")` | `.convert("RGB")` |
| `np.ndarray` | `np.random.randint(0,255,(512,512,3))` | Авто-конвертация: 2D→RGB, 4D→3D, float→uint8 |
| `torch.Tensor` | `torch.rand(3,512,512)` | `.cpu()`, `.permute()`, float→uint8 при необходимости |

### 🎚️ Автоматическая постобработка
- **Ресайз к оригиналу**: Все стратегии возвращают маску в размере исходного изображения через `scipy.ndimage.zoom` с `order=0` (nearest-neighbor).
- **Паддинг под `output_stride`**: Для моделей с downsample (FPN, PSPNet) автоматически добавляется паддинг и обрезается после инференса.
- **Препроцессинг**: Авто-выбор функции нормализации через `smp.encoders.get_preprocessing_fn()` или ImageNet stats.
- **Instance→Semantic**: Для SAM/YOLOv8 маски инстансов объединяются в единую семантическую карту.

## 🚀 Быстрый старт
### Инференс модели SegFormer
```python
from utils.strategies import segment_image_unified
from segmenters.NeuralModelFactory import NeuralModelFactory, ModelType

# Загрузка модели
model, processor, _ = NeuralModelFactory.create_model(
    ModelType.SEGFORMER,
    model_name="nvidia/segformer-b5-finetuned-ade-640-640",
    device="cuda"
)

# Инференс
overlay, info = segment_image_unified(
    model=model,
    processor=processor,
    image_input="street_scene.jpg",
    model_type="segformer",
    device="cuda",
    verbose=True
)
overlay.save("segformer_result.png")
```

### Инференс SMP-модели с авто-препроцессингом
```python
from segmenters.NeuralModelFactory import NeuralModelFactory, ModelType

# Загрузка U-Net с ResNet-34
model, _, _ = NeuralModelFactory.create_model(
    ModelType.UNET_SMP,
    encoder_name="resnet34",
    checkpoint_path="./models/unet_best.pth",
    device="cuda"
)

# Инференс (препроцессинг определяется автоматически)
overlay, info = segment_image_unified(
    model=model,
    processor=None,  # Не нужен для SMP
    image_input="medical_scan.png",
    model_type="unet_smp",
    device="cuda"
)
```

### Инстанс-сегментация через SAM
```python
# Загрузка MobileSAM
model, _, _ = NeuralModelFactory.create_model(
    ModelType.SAM,
    model_name="mobile_sam.pt",
    device="cuda"
)

# Инференс (автоматическая конвертация инстансов в семантику)
overlay, info = segment_image_unified(
    model=model,
    processor=None,
    image_input="object.jpg",
    model_type="mobile_sam",
    device="cuda"
)
# Каждый обнаруженный объект получает уникальный ID в маске
```

### Пакетный инференс с метриками
```python
# Инференс с ground truth для оценки качества
metrics_info = segment_image_unified(
    model=model,
    processor=processor,
    image_input="test.jpg",
    model_type="segformer",
    gt_mask=ground_truth_array,  # np.ndarray[H,W]
    verbose=True
)

# Доступ к метрикам
if metrics_info['metrics']:
    print(f"IoU: {metrics_info['metrics']['iou']:.3f}")
    print(f"Dice: {metrics_info['metrics']['dice']:.3f}")
    print(f"Pixel Acc: {metrics_info['metrics']['pixel_acc']:.3f}")
```

### Кастомная палитра для визуализации
```python
from utils.palettes import coco_palette, cityscapes_palette

# Использование палитры COCO вместо ADE20K
overlay, _ = segment_image_unified(
    model=model,
    processor=processor,
    image_input="coco_image.jpg",
    model_type="mask2former",
    palette=coco_palette(),  # 80 классов COCO
    alpha=0.6
)
```

## ⚙️ Конфигурация
### Параметры `segment_image_unified()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model` | `Any` | — | Загруженная модель (PyTorch module или HF transformers) |
| `processor` | `Any` | `None` | Процессор для препроцессинга (HF) или `None` для SMP/torchvision |
| `image_input` | `ImageInput` | — | Вход: `str`, `Path`, `PIL.Image`, `np.ndarray`, `torch.Tensor` |
| `model_type` | `ModelType` | — | Ключ из `INFERENCE_STRATEGIES` (см. ниже) |
| `alpha` | `float` | `0.5` | Прозрачность наложения маски: 0.0=только фото, 1.0=только маска |
| `palette` | `Optional[...]` | `None` (ADE20K) | Цветовая палитра: список `[[R,G,B], ...]` или callable |
| `device` | `str` | `"cuda"` | Устройство: `"cuda"` или `"cpu"` |
| `verbose` | `bool` | `True` | Вывод деталей инференса в консоль |
| `num_classes` | `int` | `150` | Количество классов (ADE20K) |
| `class_names` | `Optional[Dict]` | `None` | Словарь `{класс: имя}` для логирования |
| `gt_mask` | `Optional[MaskArray]` | `None` | Ground truth для расчёта метрик |

### Реестр стратегий `INFERENCE_STRATEGIES`
```python
INFERENCE_STRATEGIES: Dict[str, Callable] = {
    # HuggingFace Transformers
    "segformer": infer_segformer,
    "mask2former": infer_mask2former,
    "oneformer": infer_oneformer,
    "dpt": infer_dpt,
    "upernet": infer_mask2former,
    
    # Torchvision
    "deeplab_tv": infer_deeplab_torchvision,
    "fcn_tv": infer_fcn_torchvision_fixed,
    "maskrcnn_tv": infer_mask_rcnn,
    
    # SMP с разными output_stride
    "unet_smp": lambda ...: infer_smp_model_fixed(..., output_stride=1),
    "fpn_mit": lambda ...: infer_smp_model_fixed(..., output_stride=32),
    "psp_mit": lambda ...: infer_smp_model_fixed(..., output_stride=8),
    "deeplab_smp": lambda ...: infer_smp_model_fixed(..., output_stride=16),
    
    # Instance Segmentation
    "sam": infer_sam,
    "mobile_sam": infer_sam,
    "yolov8": infer_yolov8,
}
```

### Возвращаемое значение `segment_image_unified()`
```python
Tuple[Image.Image, Dict[str, Any]]:
- overlay: PIL.Image в режиме "RGB" с наложенной цветной маской
- result_info: словарь с метаданными:
    {
        "model": str,                    # тип модели
        "overlay": PIL.Image,            # визуализация
        "mask": np.ndarray[H,W],         # семантическая маска (uint8)
        "inference_time_ms": float,      # время инференса
        "metrics": Dict[str, float],     # IoU, Dice, ... если есть GT
        "image_size": Tuple[int, int],   # (H, W) оригинала
        "output_shape": Tuple[int, int], # (H, W) маски
        "unique_classes": int,           # количество уникальных классов
        "class_stats": List[Tuple],      # топ-10 классов с процентами
    }
```

## 📚 Справочник функций
### 🔹 Стратегии инференса по типам моделей
| Функция | Модель | Особенности | Возвращает |
|---------|--------|-------------|-----------|
| `infer_segformer()` | SegFormer (HF) | `post_process_semantic_segmentation` с `target_sizes` | `(MaskArray, PIL.Image)` |
| `infer_mask2former()` | Mask2Former (HF) | Поддержка мультитаскового инференса | `(MaskArray, PIL.Image)` |
| `infer_oneformer()` | OneFormer (HF) | `task_inputs=["semantic"]` | `(MaskArray, PIL.Image)` |
| `infer_deeplab_torchvision()` | DeepLabV3+ (TV) | ImageNet нормализация, ресайз к `target_size` | `(MaskArray, PIL.Image)` |
| `infer_unet_smp()` | U-Net/SegNet (SMP) | Авто-препроцессинг через `smp.encoders`, паддинг | `(MaskArray, PIL.Image)` |
| `infer_smp_model_fixed()` | FPN/PSPNet (SMP) | Универсальная с `output_stride`, кроппинг | `(MaskArray, PIL.Image)` |
| `infer_sam()` | SAM/MobileSAM | Конвертация инстанс-масок в семантику | `(MaskArray, PIL.Image)` |
| `infer_yolov8()` | YOLOv8-seg | Фильтрация по `confidence`, instance→semantic | `(MaskArray, PIL.Image)` |

### 🔹 Утилиты и вспомогательные функции
| Функция | Описание | Возвращает |
|---------|----------|-----------|
| `segment_image_unified()` | Универсальный инференс с авто-диспетчеризацией | `(PIL.Image, Dict)` |
| `_create_overlay_standalone()` | Создание визуализации: оригинал + цветная маска | `PIL.Image` |
| `_log_inference_details_standalone()` | Логирование и расчёт метрик при наличии GT | `Dict[str, Any]` |
| `_get_num_classes_standalone()` | Безопасное получение количества классов из модели | `Optional[int]` |
| `SegNet` | Простая encoder-decoder архитектура для бенчмарка | `torch.nn.Module` |

## 🔄 Конвейер инференса: загрузка → стратегия → постобработка → визуализация
### Логика `segment_image_unified()`
```python
def segment_image_unified(model, processor, image_input, model_type, **kwargs):
    # 1. Универсальная загрузка изображения
    if isinstance(image_input, (str, Path)):
        image = _load_image_from_path(image_input)
    elif isinstance(image_input, torch.Tensor):
        image = _tensor_to_pil(image_input)
    # ... остальные форматы ...
    
    # 2. Диспетчеризация стратегии
    if model_type not in INFERENCE_STRATEGIES:
        raise ValueError(f"Unknown model_type: {model_type}")
    infer_func = INFERENCE_STRATEGIES[model_type]
    
    # 3. Выполнение инференса (внутри torch.no_grad())
    seg_map, _ = infer_func(model, processor, image, device)
    
    # 4. Постобработка и логирование
    if verbose:
        result_info = _log_inference_details_standalone(...)
    
    # 5. Создание overlay
    overlay = _create_overlay_standalone(image, seg_map, alpha, palette)
    
    return overlay, result_info
```

### Авто-препроцессинг для SMP-моделей
```python
def infer_smp_model_fixed(model, image, device, output_stride=32):
    # Авто-определение encoder_name
    try:
        encoder_name = model.encoder.name
    except:
        encoder_name = "resnet34"
    
    # Получение функции препроцессинга
    preprocess_fn = smp.encoders.get_preprocessing_fn(encoder_name, "imagenet")
    
    # Применение препроцессинга
    image_np = np.array(image)
    input_tensor = preprocess_fn(image_np)
    input_tensor = torch.from_numpy(input_tensor).permute(2,0,1).float()
    
    # Паддинг под output_stride
    if output_stride > 1:
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        pad_h = (output_stride - h % output_stride) % output_stride
        pad_w = (output_stride - w % output_stride) % output_stride
        if pad_h or pad_w:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    
    # Инференс
    with torch.no_grad():
        outputs = model(input_tensor.unsqueeze(0).to(device))
    
    # Постобработка: argmax + кроппинг + ресайз к оригиналу
    pred_mask = outputs.argmax(1).squeeze(0).cpu().numpy()
    if pad_h or pad_w:
        pred_mask = pred_mask[:orig_h, :orig_w]
    if pred_mask.shape != (orig_h, orig_w):
        pred_mask = zoom(pred_mask, (orig_h/pred_mask.shape[0], orig_w/pred_mask.shape[1]), order=0)
    
    return pred_mask, image
```

### Конвертация instance→semantic для SAM/YOLOv8
```python
def infer_sam(model, processor, image, device):
    img_w, img_h = image.size
    results = model(image)  # ultralytics инференс
    
    # Создание семантической карты из инстанс-масок
    seg_map = np.zeros((img_h, img_w), dtype=np.uint8)
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # [N, H, W]
        for i, mask in enumerate(masks, start=1):
            mask_bin = (mask > 0.5).astype(np.uint8)
            # Ресайз к оригиналу
            if mask_bin.shape != (img_h, img_w):
                mask_pil = Image.fromarray(mask_bin)
                mask_bin = np.array(mask_pil.resize((img_w, img_h), Image.NEAREST))
            # Назначение уникального ID каждому инстансу
            seg_map[(seg_map == 0) & (mask_bin > 0)] = i
    
    return seg_map, image
```

## 📊 Метрики качества (при наличии Ground Truth)
При передаче `gt_mask` автоматически вычисляются метрики через `utils.utils.compute_metrics()`:

| Метрика | Описание | Диапазон | Интерпретация |
|---------|----------|----------|---------------|
| **IoU** | Intersection over Union | [0.0, 1.0] | Основной критерий для семантической сегментации |
| **Dice** | Dice / F1 coefficient | [0.0, 1.0] | Более устойчив к дисбалансу классов |
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
```

### Рекомендации по оптимизации
1. **Используйте `target_size` для torchvision/SMP**: Ресайз входа к размеру обучения ускоряет инференс и улучшает качество.
2. **Кэшируйте препроцессинг**: Для пакетного инференса создайте `preprocess_fn` один раз и переиспользуйте.
3. **Минимизируйте `alpha` при отладке**: Создание overlay — дополнительная операция; используйте `alpha=0` если нужна только маска.
4. **Отключайте `verbose` в продакшене**: Логирование и анализ классов добавляют накладные расходы.
5. **Используйте `torch.compile()`** (PyTorch 2.0+) для SMP/torchvision-моделей:
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```

## 🛠️ Обработка ошибок и устойчивость
### Валидация входных данных
```python
# Проверка поддерживаемых форматов
if isinstance(image_input, (str, Path)):
    # Загрузка из пути или URL
elif isinstance(image_input, (Image.Image, np.ndarray, torch.Tensor)):
    # Конвертация
else:
    raise ValueError(f"Unsupported input type: {type(image_input)}")
```

### Безопасное определение количества классов
```python
def _get_num_classes_standalone(model, model_type, fallback=150):
    try:
        # HF transformers
        if hasattr(model, "config") and hasattr(model.config, "id2label"):
            return len(model.config.id2label)
        # Torchvision
        if model_type in ["deeplab_tv", "fcn_tv"]:
            return model.classifier[-1].out_channels
        # SMP: поиск последнего Conv2d
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module.out_channels
        return fallback
    except Exception:
        return fallback
```

### Обработка ошибок при расчёте метрик
```python
if gt_mask is not None:
    try:
        metrics = compute_metrics(seg_map, gt_np, num_classes=num_classes)
        print(f"✅ Metrics computed: IoU={metrics.get('iou', 0):.4f}")
    except Exception as e:
        print(f"⚠️ Metrics computation failed: {e}")
        metrics = {}  # Не прерываем выполнение
```

### Рекомендации по отладке
1. **Включите `verbose=True`** для пошагового мониторинга:
   ```python
   overlay, info = segment_image_unified(..., verbose=True)
   # Вывод: форма маски, уникальные классы, топ-5 по пикселям, метрики
   ```

2. **Проверьте `model_type` в реестре**:
   ```python
   from utils.strategies import INFERENCE_STRATEGIES
   print(f"Available strategies: {list(INFERENCE_STRATEGIES.keys())}")
   ```

3. **Тестируйте на маленьком изображении**:
   ```python
   small_img = Image.new("RGB", (224, 224), color="red")
   overlay, info = segment_image_unified(..., image_input=small_img)
   # Быстрая проверка конвейера без долгого ожидания
   ```

4. **Анализируйте `inference_time_ms`** для оптимизации:
   ```python
   if info['inference_time_ms'] > 500:
       print(f"⚠️ Slow inference: {info['inference_time_ms']:.0f}ms — consider resizing input")
   ```

## 🤝 Зависимости
```text
torch>=2.0                    # Основные тензорные операции, torch.no_grad()
torchvision>=0.10             # Transforms, предобученные модели (опционально)
segmentation-models-pytorch>=0.2  # SMP: препроцессинг, модели
transformers>=4.25           # HF-модели и процессоры (опционально)
ultralytics>=8.0             # SAM, YOLOv8 (опционально)
Pillow>=8.0                  # Загрузка и обработка изображений
numpy>=1.20                  # Массивы, метрики, ресайз
scipy>=1.7                   # zoom для ресайза масок
requests>=2.28               # Загрузка изображений по URL
```

### Опциональные зависимости для расширенного функционала
```bash
# Для логирования в файл
pip install loguru

# Для визуализации метрик
pip install matplotlib seaborn

# Для экспорта отчётов в Markdown
pip install tabulate
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование strategies.py |
|--------|----------------------------|
| `NeuralSegmenter` | Делегирует инференс через `segment_image_unified()` |
| `NeuralModelFactory` | Создаёт модели, которые передаются в стратегии |
| `SegmentationTester` | Универсальное тестирование: вызов `segment_image_unified()` для всех моделей |
| `CpuCudaBenchmark` | Замер времени инференса через `info['inference_time_ms']` |
| `utils.utils` | Вызов `compute_metrics()`, `analyze_prediction()` внутри `_log_inference_details_standalone()` |

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