# 🏭 NeuralModelFactory — Фабрика нейронных моделей сегментации

## 📖 Описание
Модуль `NeuralModelFactory.py` предоставляет **единую точку входа** для создания, загрузки и управления современными архитектурами нейронной сегментации из различных источников:

> ⚠️ **Важно:** Данный модуль работает исключительно с **нейросетевыми методами**. Для классических алгоритмов используйте `OpenCVSegmenter` или `SklearnSegmenter`.

## ✨ Ключевые возможности
### 🏗️ Поддерживаемые семейства моделей (15+ архитектур)

| Семейство | Модели | Источники | Особенности |
|-----------|--------|-----------|-------------|
| **HuggingFace Transformers** (6) | `segformer`, `mask2former`, `oneformer`, `dpt`, `upernet`, `maskformer` | `transformers`, `huggingface_hub` | Предобученные веса, авто-процессоры, поддержка safetensors |
| **Segmentation Models PyTorch** (3) | `unet_smp`, `fpn_smp`, `pspnet_smp` | `segmentation-models-pytorch` | 50+ encoder'ов (ResNet, MiT, EfficientNet), предобучение на ImageNet |
| **Torchvision** (3) | `deeplab_tv`, `fcn_tv`, `maskrcnn_tv` | `torchvision` | Официальные реализации, предобучение на COCO+VOC |
| **Instance Segmentation** (2) | `sam`, `yolov8` | `ultralytics` | Zero-shot сегментация, детекция + маска |
| **Custom** (1) | `segnet` | proxy через `smp.Unet` | Эмуляция SegNet или кастомная реализация |

### ⚙️ Гибкая система конфигурации
- **YAML-конфиги**: централизованное управление параметрами моделей и обучения через `configs/neural_models.yaml`.
- **Enum-based типизация**: `ModelType` enum гарантирует типобезопасность при выборе архитектуры.
- **Ленивая загрузка**: конфигурация кэшируется, повторные вызовы не читают файл.
- **Fallback на дефолты**: при отсутствии конфига используются встроенные значения.

### 🔄 Единый интерфейс для всех моделей
```python
from segmenters.NeuralModelFactory import NeuralModelFactory, ModelType

# Загрузка SegFormer B5 из HuggingFace
model, processor, model_type = NeuralModelFactory.create_model(
    ModelType.SEGFORMER,
    model_name="nvidia/segformer-b5-finetuned-ade-640-640",
    device="cuda"
)

# Создание U-Net с чекпоинтом
model, _, model_type = NeuralModelFactory.create_model(
    ModelType.UNET_SMP,
    encoder_name="resnet34",
    checkpoint_path="./models/unet_best.pth",
    device="cuda"
)

# Пакетная загрузка для бенчмарка
models = NeuralModelFactory.load_all_pretrained_cnn(
    checkpoint_dir="./checkpoints",
    device="cuda"
)
for name, (model, proc, mtype) in models.items():
    print(f"{name:30s} → {mtype}")
```

### 🎚️ Автоматическая адаптация под тип модели
| Тип модели | Процессор | Загрузка весов | Выходной формат |
|------------|-----------|----------------|-----------------|
| **HF Transformers** | `AutoImageProcessor` | `from_pretrained()` | `logits` → `argmax(1)` |
| **SMP** | `None` (raw tensor) | `torch.load()` + `load_state_dict()` | `[B, C, H, W]` |
| **Torchvision** | `None` | `weights="COCO"` или чекпоинт | `dict['out']` или `Tensor` |
| **SAM/YOLOv8** | Встроенный в `ultralytics` | `.pt` файл | Список масок/боксов |

### 📦 Поддержка чекпоинтов и transfer learning
```python
# Загрузка с заменой классификатора под NUM_CLASSES
model, _, _ = NeuralModelFactory.create_model(
    ModelType.DEEPLAB_TV,
    num_classes=150,  # ADE20K
    checkpoint_path="./models/deeplab_ade20k.pth"
)

# Чекпоинт с ключом "model_state_dict" (стандарт Trainer)
checkpoint = torch.load(path, map_location="cuda")
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)  # Прямой state_dict
```

## 🚀 Быстрый старт
### Базовое использование: загрузка предобученной модели
```python
from segmenters.NeuralModelFactory import NeuralModelFactory, ModelType
import torch

# Загрузка SegFormer для ADE20K
model, processor, _ = NeuralModelFactory.create_model(
    ModelType.SEGFORMER,
    model_name="nvidia/segformer-b5-finetuned-ade-640-640",
    device="cuda"
)

# Инференс (псевдокод, зависит от процессора)
inputs = processor(images=image, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)
    pred_mask = outputs.logits.argmax(1)  # [B, H, W]
```

### Создание SMP-модели с чекпоинтом
```python
# U-Net с ResNet-34 encoder
model, _, _ = NeuralModelFactory.create_model(
    ModelType.UNET_SMP,
    encoder_name="resnet34",
    checkpoint_path="./models/unet_resnet34.pth",
    num_classes=150,
    device="cuda"
)

# Инференс
with torch.no_grad():
    output = model(image_tensor)  # [B, 3, H, W] → [B, 150, H, W]
    pred_mask = output.argmax(1)   # [B, H, W]
```

### Пакетная загрузка для бенчмарка
```python
# Загрузка всех CNN-моделей для сравнения
models = NeuralModelFactory.load_all_pretrained_cnn(
    checkpoint_dir="./checkpoints",
    device="cuda",
    num_classes=150
)

# Сравнение инференса
for name, (model, _, mtype) in models.items():
    start = time.time()
    with torch.no_grad():
        _ = model(test_tensor)
    print(f"{name:30s}: {(time.time()-start)*1000:.1f} ms")
```

### Работа с конфигурацией через YAML
```yaml
# configs/neural_models.yaml
models:
  segformer:
    variants:
      b0: "nvidia/segformer-b0-finetuned-ade-512-512"
      b5: "nvidia/segformer-b5-finetuned-ade-640-640"
    default: "b5"
  unet:
    encoders: ["resnet34", "resnet50", "efficientnet-b0", "mit_b5"]
    default: "resnet34"
training:
  ade20k:
    image_size: [512, 512]
    batch_size: 4
    lr: 1.0e-4
```

```python
# Использование конфига
model, _, _ = NeuralModelFactory.create_model_from_config(
    model_type="segformer",
    variant="b5",
    device="cuda"
)
# Автоматически подставит: "nvidia/segformer-b5-finetuned-ade-640-640"
```

## ⚙️ Конфигурация
### Enum `ModelType` — поддерживаемые архитектуры
```python
class ModelType(Enum):
    # HuggingFace
    SEGFORMER = "segformer"
    MASK2FORMER = "mask2former"
    ONEFORMER = "oneformer"
    DPT = "dpt"
    UPERNET = "upernet"
    MASKFORMER = "maskformer"
    
    # SMP
    UNET_SMP = "unet_smp"
    FPN_SMP = "fpn_smp"
    PSPNET_SMP = "pspnet_smp"
    
    # Torchvision
    DEEPLAB_TV = "deeplab_tv"
    FCN_TV = "fcn_tv"
    MASKRCNN_TV = "maskrcnn_tv"
    
    # Instance
    SAM = "sam"
    YOLOV8 = "yolov8"
    
    # Custom
    SEGNET = "segnet"
```

### Параметры `create_model()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `model_type` | `ModelType` | — | Тип архитектуры (обязательно) |
| `model_name` | `Optional[str]` | `None` | Имя модели в HF Hub (для Transformers) |
| `local_path` | `Optional[str]` | `None` | Локальный путь к модели (альтернатива `model_name`) |
| `checkpoint_path` | `Optional[str]` | `"model_path.pth"` | Путь к `.pth` чекпоинту (для SMP/torchvision) |
| `device` | `DeviceStr` | `"cuda"` | Устройство: `"cuda"` или `"cpu"` |
| `num_classes` | `int` | `150` | Количество выходных классов (ADE20K) |
| `encoder_name` | `str` | `"resnet34"` | Название encoder'а (для SMP-моделей) |
| `**kwargs` | `Any` | — | Дополнительные параметры (variant, psp_size, ...) |

### Возвращаемое значение `ModelTuple`
```python
ModelTuple = Tuple[nn.Module, Optional[Any], str]
# (model, processor, model_type_str)
```

| Элемент | Тип | Описание |
|---------|-----|----------|
| `model` | `nn.Module` | Загруженная модель в режиме `.eval()` |
| `processor` | `Optional[Any]` | HF-процессор для препроцессинга (или `None`) |
| `model_type_str` | `str` | Строковый идентификатор типа модели |

## 📚 Справочник методов
### 🔹 Основные конструкторы
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `create_model()` | `model_type`, `model_name`, `checkpoint_path`, `device`, `**kwargs` | Универсальный конструктор по `ModelType` enum | `ModelTuple` |
| `create_model_from_config()` | `model_type`, `variant`, `device`, `**kwargs` | Конструктор с параметрами из YAML-конфига | `ModelTuple` |
| `load_segformer_variant()` | `variant`, `device` | Загрузка конкретной версии SegFormer (b0–b5) | `ModelTuple` |
| `load_smp_model()` | `architecture`, `encoder_name`, `checkpoint_path`, `**kwargs` | Универсальный загрузчик для SMP-архитектур | `ModelTuple` |

### 🔹 Пакетные операции и утилиты
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `load_all_pretrained_cnn()` | `checkpoint_dir`, `device`, `num_classes` | Пакетная загрузка CNN-моделей для бенчмарка | `Dict[str, ModelTuple]` |
| `get_supported_models()` | — | Список всех поддерживаемых типов моделей | `List[str]` |
| `register_model()` | `model_type`, `config` | Регистрация новой модели в реестре фабрики | `None` |

### 🔹 Работа с конфигурацией
| Метод | Параметры | Описание | Возвращает |
|-------|-----------|----------|-----------|
| `load_config()` | `config_path` | Ленивая загрузка YAML-конфига с кэшированием | `Dict[str, Any]` |
| `get_model_name()` | `model_type`, `variant` | Получение полного имени модели из конфига | `str` |
| `get_training_config()` | `dataset_name` | Параметры обучения для датасета | `Dict[str, Any]` |
| `get_metrics_config()` | — | Конфигурация метрик (threshold, hausdorff) | `Dict[str, Any]` |

### 🔹 Отладка и инспекция
| Метод | Параметры | Описание |
|-------|-----------|----------|
| `print_*_params()` | `model_name`, `device` | Вывод параметров модели для отладки (есть для каждого типа) |

## 🔄 Конвейер загрузки модели
### Логика `create_model()` — dispatch по типу
```python
@classmethod
def create_model(cls, model_type: ModelType, ...) -> ModelTuple:
    if model_type == ModelType.SEGFORMER:
        return cls._load_segformer(model_name, local_path, device)
    elif model_type == ModelType.UNET_SMP:
        return cls._load_unet_smp(device, num_classes, checkpoint_path, **kwargs)
    elif model_type == ModelType.DEEPLAB_TV:
        return cls._load_deeplab_tv(device, num_classes, checkpoint_path)
    # ... и так далее для всех 15+ типов
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
```

### Загрузка чекпоинта с обработкой форматов
```python
safe_path = checkpoint_path or "model_path.pth"
if checkpoint_path and os.path.exists(safe_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Поддержка двух форматов чекпоинтов
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Прямой state_dict
    
    print(f"✅ Loaded from checkpoint: {checkpoint_path}")
else:
    print("⚠️ Checkpoint not found, using pretrained weights only")
```

### Замена классификатора под NUM_CLASSES (для torchvision)
```python
# DeepLabV3+ пример
model = tv_seg.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")
old_out_ch = model.classifier[4].out_channels  # 21 (COCO)
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # 150 (ADE20K)
nn.init.normal_(model.classifier[4].weight, 0, 0.01)
nn.init.constant_(model.classifier[4].bias, 0)
```

## 📊 Инференс и вывод
### Форматы выхода по типу модели
| Тип модели | Выход `model(input)` | Постобработка для маски |
|------------|---------------------|------------------------|
| **HF Transformers** | `ModelOutput(logits=[B,C,H,W])` | `outputs.logits.argmax(1)` |
| **SMP** | `Tensor [B, C, H, W]` | `output.argmax(1)` |
| **Torchvision DeepLab/FCN** | `dict {'out': [B,C,H,W], 'aux': ...}` | `output['out'].argmax(1)` |
| **Torchvision Mask R-CNN** | `List[Dict]` с боксами и масками | `pred['masks'].squeeze(1)` |
| **SAM/YOLOv8** | `Results` объект ultralytics | `result.masks.xy` или `.data` |

### Пример инференса для разных типов
```python
# HF Transformers (SegFormer)
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    pred = outputs.logits.argmax(1)  # [B, H, W]

# SMP / Torchvision
with torch.no_grad():
    output = model(image_tensor)  # [B, C, H, W]
    if isinstance(output, dict):
        output = output['out']
    pred = output.argmax(1)  # [B, H, W]

# SAM (ultralytics)
results = model.predict(source=image, device=device)
masks = results[0].masks.xy  # Список полигонов [N, 4]
```

## ⚡ Производительность и оптимизации
### Относительная скорость инференса (на изображении 512×512, RTX 3090)
```
✅ Быстро (<50 мс):
   - unet_smp + resnet34
   - fcn_tv + resnet50
   - deeplab_tv + resnet101 (без aux)

⚠️ Средне (50–200 мс):
   - fpn_smp + mit_b5
   - pspnet_smp + mit_b5
   - segformer-b0/b1/b2
   - maskrcnn_tv (с постобработкой)

❌ Медленно (200–1000+ мс):
   - segformer-b3/b4/b5
   - mask2former-swin-large
   - oneformer-swin-large
   - sam (zero-shot, зависит от количества промптов)
   - yolov8-seg (зависит от количества объектов)
```

### Рекомендации по оптимизации
1. **Используйте `torch.compile()`** (PyTorch 2.0+) для SMP/torchvision-моделей:
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```

2. **Mixed Precision** для инференса:
   ```python
   with torch.autocast(device_type="cuda", dtype=torch.float16):
       output = model(image_tensor)
   ```

3. **Кэширование процессоров** для HF-моделей:
   ```python
   # Процессор можно переиспользовать для батча изображений
   processor = SegformerImageProcessor.from_pretrained(model_name)
   inputs = processor(images=batch_images, return_tensors="pt")
   ```

4. **Выгрузка на CPU** для редко используемых моделей:
   ```python
   model.cpu()  # Освобождает VRAM
   torch.cuda.empty_cache()
   ```

5. **Батчинг инференса**: обрабатывайте несколько изображений за один forward-pass для SMP/torchvision.

## 🛠️ Обработка ошибок и устойчивость
### Проверка зависимостей при импорте
```python
try:
    from transformers import SegformerForSemanticSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Warning: transformers not installed")

# В методах загрузки:
if not TRANSFORMERS_AVAILABLE:
    raise ImportError("transformers library required for SegFormer")
```

### Безопасная загрузка чекпоинтов
```python
safe_path = checkpoint_path or "model_path.pth"
if checkpoint_path and os.path.exists(safe_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Поддержка двух форматов
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)  # strict=False для частичной загрузки
        print(f"✅ Loaded from {checkpoint_path}")
    except Exception as e:
        print(f"⚠️ Failed to load checkpoint: {e}. Using pretrained weights.")
```

### Валидация параметров модели
```python
# Проверка variant для SegFormer
variants = config["models"]["segformer"]["variants"]
if variant not in variants:
    raise ValueError(
        f"Unknown SegFormer variant: {variant}. Available: {list(variants.keys())}"
    )

# Проверка encoder для SMP
valid_encoders = smp.encoders.encoders.keys()
if encoder_name not in valid_encoders:
    raise ValueError(f"Unknown encoder: {encoder_name}. Available: {list(valid_encoders)[:10]}...")
```

### Рекомендации по отладке
1. **Используйте `print_*_params()` методы** для инспекции модели:
   ```python
   NeuralModelFactory.print_segformer_params(
       "nvidia/segformer-b5-finetuned-ade-640-640",
       device="cuda"
   )
   ```

2. **Проверьте устройство и режим модели**:
   ```python
   print(f"Device: {next(model.parameters()).device}")
   print(f"Training mode: {model.training}")  # Должно быть False для инференса
   ```

3. **Тестируйте на маленьком тензоре**:
   ```python
   test_input = torch.randn(1, 3, 224, 224).to(device)
   with torch.no_grad():
       output = model(test_input)
   print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else output['out'].shape}")
   ```

## 🤝 Зависимости
```text
torch>=1.9                    # Основные тензорные операции
torchvision>=0.10             # DeepLab, FCN, Mask R-CNN
segmentation-models-pytorch>=0.2  # U-Net, FPN, PSPNet с 50+ encoder'ами
transformers>=4.25           # SegFormer, Mask2Former, OneFormer (опционально)
huggingface_hub>=0.10        # Загрузка моделей из HF Hub (опционально)
ultralytics>=8.0             # SAM, YOLOv8 (опционально)
pyyaml>=6.0                  # Конфигурация через YAML
numpy>=1.20                  # Массивы, метрики
```

### Опциональные зависимости (градуированная установка)
```bash
# Только классические + базовые нейросети
pip install torch torchvision segmentation-models-pytorch pyyaml

# + HuggingFace Transformers
pip install transformers huggingface_hub

# + Instance Segmentation
pip install ultralytics
```

## 🔗 Интеграция с другими модулями проекта
| Модуль | Использование NeuralModelFactory |
|--------|---------------------------------|
| `ModelTrainer` | Создание моделей для обучения через `create_model()` |
| `SegmentationTester` | Универсальное тестирование: добавление нейросетей через `add_method()` |
| `CpuCudaBenchmark` | Бенчмарк производительности: загрузка моделей через `load_all_pretrained_cnn()` |
| `TorchImplementationValidator` | Валидация: сравнение предсказаний с эталонными реализациями |
| `BatchClassicTester2` | Сравнение классических и нейросетевых методов на одном датасете |

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