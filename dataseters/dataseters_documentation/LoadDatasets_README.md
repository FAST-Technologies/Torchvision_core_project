# 📦 LoadDatasets — Менеджер загрузки и подготовки датасетов для сегментации

## 📖 Описание
Модуль `datasets/load_datasets.py` предоставляет **универсальный менеджер** для загрузки, валидации и подготовки датасетов для задач семантической, инстанс- и медицинской сегментации.

> ⚠️ **Важно:** Данный модуль отвечает за *инфраструктуру данных* (загрузка, организация, предобработка). Для непосредственной сегментации используйте модули `OpenCVSegmenter`, `TorchSegmenter` или `NeuralModelFactory`.

## ✨ Ключевые возможности

### 🗂️ Поддержка датасетов и источников

| Категория | Примеры | Форматы | Источник |
|-----------|---------|---------|----------|
| **Семантическая сегментация** | ADE20K, Cityscapes | JPG/PNG + PNG-маски | ZIP, HuggingFace |
| **Instance/Panoptic** | COCO | JSON-аннотации + изображения | HuggingFace |
| **Медицинская бинарная** | ISIC 2018, CheXpert | DICOM, NIfTI, PNG | HuggingFace, Direct |
| **Пользовательские** | Любые кастомные датасеты | NPZ, HDF5, Parquet | ZIP/TAR, Direct URL |

### 🔄 Полный workflow загрузки
```
1. Регистрация конфигураций (код или YAML)
   ↓
2. Скачивание с валидацией контрольных сумм (SHA256)
   ↓
3. Распаковка и реорганизация к стандартной структуре
   ↓
4. Пост-обработка (Parquet→файлы, медицинская нормализация)
   ↓
5. Создание индексного файла для быстрого доступа
```

### 🎯 Стандартная структура после загрузки
```
data/
└── {dataset_name}/
    ├── images/
    │   ├── training/   # *.jpg, *.png
    │   ├── validation/
    │   └── testing/
    ├── annotations/
    │   ├── training/   # *.png (маски)
    │   ├── validation/
    │   └── testing/
    ├── index.json      # Мета-индекс для быстрого доступа
    └── configs/        # Опционально: кастомные конфиги
```

## 🚀 Быстрый старт

### Базовое использование
```python
from datasets.load_datasets import DatasetManager

# Инициализация менеджера
manager = DatasetManager(base_dir="./data", verbose=True)

# Загрузка датасета (с проверкой целостности)
path = manager.download("ade20k", force=False)

# Загрузка одного примера
img, mask = manager.load_sample("ade20k", split="val", idx=0)

# Создание PyTorch Dataset для обучения
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = manager.create_pytorch_dataset("ade20k", split="train", transform=transform)
```

### Работа с медицинскими датасетами
```python
# ISIC 2018: сегментация кожных поражений
manager = DatasetManager(base_dir="./medical_data")
isic_path = manager.download("isic2018")

# Загрузка с медицинской пост-обработкой (CLAHE, нормализация)
img, mask = manager.load_sample("isic2018", split="train", idx=42)

# Доступ к метаданным конфигурации
config = manager.get_config("isic2018")
print(f"Модальность: {config.modality}")  # Dermoscopy
print(f"Pixel spacing: {config.pixel_spacing}")  # (0.025, 0.025) мм
```

### Загрузка из HuggingFace Hub
```python
# Автоматическая загрузка через datasets library
manager = DatasetManager()
coco_path = manager.download("coco")  # source_type=HF

# Или прямая загрузка тестового изображения
img = manager.load_test_image_from_hf(
    repo_id="Chris1/cityscapes",
    split="validation",
    filename=None  # None = первый доступный файл
)
```

## ⚙️ Конфигурация

### Регистрация датасета через код
```python
from datasets.load_datasets import DatasetManager, DatasetConfig, DatasetType, SourceType

manager = DatasetManager(base_dir="./my_data")

# Регистрация кастомного датасета
manager._registry["my_dataset"] = DatasetConfig(
    name="my_dataset",
    dataset_type=DatasetType.SEMANTIC,
    source_type=SourceType.ZIP,
    source_url="https://example.com/my_dataset.zip",
    root_dir="./my_data",
    num_classes=10,
    checksum="sha256_hash_here",  # Опционально
    splits={"train": "train", "val": "val"},
    metadata={"license": "MIT", "homepage": "https://example.com"}
)
```

### Регистрация через YAML (`configs/datasets.yaml`)
```yaml
my_medical_dataset:
  dataset_type: medical_binary
  source_type: hf
  source_url: "username/my-medical-ds"
  root_dir: "./medical_data"
  num_classes: 2
  modality: "MRI"
  pixel_spacing: [0.5, 0.5]
  intensity_normalization: "zscore"
  anatomical_regions: ["brain", "tumor"]
  metadata:
    license: "CC-BY-4.0"
    task: "Brain tumor segmentation"
```

### Параметры инициализации `DatasetManager`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `base_dir` | `PathLike` | `"./data"` | Базовая директория для всех датасетов |
| `verbose` | `bool` | `True` | Вывод подробных логов с цветным форматированием |

### Параметры `DatasetConfig`
| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `name` | `str` | ✅ | Уникальный идентификатор датасета |
| `dataset_type` | `DatasetType` | ✅ | Тип задачи (SEMANTIC, INSTANCE, MEDICAL_BINARY, ...) |
| `source_type` | `SourceType` | ✅ | Источник: HF, ZIP, TAR, DIRECT |
| `source_url` | `str` | ✅ | URL или ID репозитория |
| `root_dir` | `str` | ✅ | Базовая директория сохранения |
| `num_classes` | `int` | ❌ (150) | Количество классов сегментации |
| `checksum` | `Optional[str]` | ❌ | SHA256 для валидации загрузки |
| `splits` | `Dict[str,str]` | ❌ | Маппинг `{split: directory}` |
| `preprocessing` | `Dict` | ❌ | Параметры предобработки |

### Дополнительные поля `MedicalConfig`
| Параметр | Тип | Описание |
|----------|-----|----------|
| `modality` | `Literal[...]` | Тип модальности: "X-Ray", "CT", "MRI", "Ultrasound", "Dermoscopy" |
| `pixel_spacing` | `Tuple[float,float]` | Физический размер пикселя в мм |
| `intensity_normalization` | `Literal[...]` | Метод: "minmax", "zscore", "histogram" |
| `anatomical_regions` | `List[str]` | Список анатомических областей |
| `privacy_compliant` | `bool` | Соответствие требованиям приватности |

## 📚 Справочник методов `DatasetManager`

### 🔹 Загрузка и валидация
| Метод | Возвращает | Описание |
|-------|------------|----------|
| `download(dataset_name, force=False)` | `Path` | Скачивает и валидирует датасет, возвращает путь |
| `_validate_dataset(config)` | `bool` | Проверяет целостность структуры и файлов |
| `_compute_sha256(filepath)` | `str` | Вычисляет SHA256 хеш для валидации |
| `_check_disk_space(path, required_gb)` | `bool` | Проверяет наличие свободного места |

### 🔹 Работа с данными
| Метод | Возвращает | Описание |
|-------|------------|----------|
| `load_sample(dataset_name, split, idx)` | `Tuple[Image, Optional[Image]]` | Загружает пару (изображение, маска) по индексу |
| `create_pytorch_dataset(dataset_name, split, transform)` | `torch.utils.data.Dataset` | Создаёт PyTorch Dataset с поддержкой аугментаций |
| `get_config(dataset_name)` | `DatasetConfig` | Возвращает конфигурацию датасета |

### 🔹 Внутренние методы (расширение)
| Метод | Описание |
|-------|----------|
| `_download_huggingface(config)` | Загрузка через `huggingface_hub` с fallback-логикой |
| `_download_zip(config)` | Потоковая загрузка ZIP с прогрессом и валидацией |
| `_reorganize_ade_structure(source, target)` | Приведение структуры к стандарту `images/annotations` |
| `_convert_parquet_to_files(config)` | Конвертация Parquet → файловая структура изображений |
| `_medical_postprocess(config)` | Применение CLAHE, нормализация медицинских масок |
| `_create_index(config)` | Генерация `index.json` для быстрого доступа к метаданным |

## 🏥 Утилиты для медицинских датасетов (`MedicalDatasetUtils`)

```python
from datasets.load_datasets import MedicalDatasetUtils
from pathlib import Path

# Загрузка DICOM серии в 3D numpy массив
volume = MedicalDatasetUtils.load_dicom_series(Path("ct_scan/"))

# Загрузка NIfTI с метаданными
data, meta = MedicalDatasetUtils.load_nifti(Path("brain.nii.gz"))
print(f"Shape: {meta['shape']}, Affine: {meta['affine']}")

# CT windowing для визуализации
windowed = MedicalDatasetUtils.window_ct(
    image=data, 
    window_center=40,   # HU для мягких тканей
    window_width=400
)

# Сохранение в формате, готовом для обучения
MedicalDatasetUtils.save_for_training(
    image=windowed,
    mask=segmentation_mask,
    output_dir=Path("processed/"),
    prefix="patient_001",
    config=medical_config
)
```

## 🖥️ CLI интерфейс

```bash
# Просмотр доступных датасетов
python datasets/load_datasets.py --list

# Загрузка конкретного датасета
python datasets/load_datasets.py ade20k --base-dir ./my_data

# Загрузка всех медицинских датасетов
python datasets/load_datasets.py --medical-only

# Принудительная перезагрузка
python datasets/load_datasets.py cityscapes --force

# Загрузка и сохранение тестового примера
python datasets/load_datasets.py --sample "isic2018:train:5"
```

### Аргументы CLI
| Аргумент | Описание |
|----------|----------|
| `datasets` | Список имён датасетов для загрузки |
| `--all` | Загрузить все зарегистрированные датасеты |
| `--medical-only` | Загрузить только медицинские датасеты |
| `--base-dir` | Базовая директория (по умолчанию: `./data`) |
| `--force` | Перезагрузить даже при наличии валидной копии |
| `--list` | Показать список доступных датасетов |
| `--sample` | Загрузить пример: `dataset:split:idx` |

## 📊 Метрики и логирование

### Формат логов
```
[14:32:15] ℹ️ 📦 ЗАГРУЗКА ДАТАСЕТА: ADE20K (SEMANTIC)...
[14:32:15] ℹ️ Тип: SEMANTIC
[14:32:15] ℹ️ Источник: ZIP
[14:32:15] ℹ️ Классы: 150
[14:32:15] ℹ️ Целевая директория: ./data/ade20k
[14:32:45] ✅ 📥 Скачивание завершено! Размер: 2.14 GB
[14:33:20] ✅ 📄 Создан индекс: ./data/ade20k/index.json
[14:33:20] ✅ 🎯 ADE20K загружен успешно!
```

### Уровни логирования
| Уровень | Цвет | Использование |
|---------|------|---------------|
| `info` | 🔵 | Стандартные сообщения о ходе выполнения |
| `success` | 🟢 | Успешное завершение операций |
| `warning` | 🟡 | Предупреждения (несоответствие структуры, пропущенные файлы) |
| `error` | 🔴 | Критические ошибки (недостаточно места, невалидный checksum) |

## ⚡ Производительность и оптимизации

### Стратегии ускорения загрузки
1. **Кэширование:** Повторные вызовы `download()` проверяют валидность перед скачиванием.
2. **Потоковая загрузка:** Большие архивы загружаются чанками с прогресс-баром.
3. **Индексация:** `index.json` позволяет быстро получать метаданные без сканирования ФС.
4. **Symlinks:** Опция `use_symlinks=True` экономит место при реорганизации (только Unix).

### Рекомендации для больших датасетов
```python
# 1. Проверка места перед загрузкой
manager._check_disk_space(Path("./data"), required_gb=50.0)

# 2. Отключение подробных логов для пакетной загрузки
manager = DatasetManager(verbose=False)

# 3. Использование force=False для избежания повторных загрузок
path = manager.download("coco", force=False)  # Быстрый возврат если уже есть
```

## 🛠️ Обработка ошибок и устойчивость

### Стратегия fallback для HuggingFace
```
1. Попытка загрузки через `snapshot_download()` (быстро, атомарно)
   ↓ (при ошибке 401/404)
2. Fallback: пофайловая загрузка через `hf_hub_download()`
   ↓ (при ошибке API)
3. Fallback: прямой HTTP-запрос с прогрессом
```

### Валидация целостности
```python
# Проверка контрольной суммы при загрузке
if config.checksum:
    actual = manager._compute_sha256(downloaded_file)
    if actual != config.checksum:
        raise ValueError(f"Checksum mismatch! Expected {config.checksum[:16]}..., got {actual[:16]}...")

# Валидация структуры после распаковки
if not manager._validate_dataset(config):
    logger.warning("⚠️ Датасет загружен, но структура не соответствует ожиданиям")
```

### Рекомендации по отладке
1. **Включите DEBUG-логирование:**
   ```python
   import logging
   logging.getLogger("datasets.load_datasets").setLevel(logging.DEBUG)
   ```

2. **Проверьте конфигурацию:**
   ```python
   config = manager.get_config("my_dataset")
   print(f"Source: {config.source_type}, URL: {config.source_url}")
   ```

3. **Тестируйте на маленьком датасете:**
   ```python
   # Быстрая проверка конвейера на ISIC (небольшой медицинский датасет)
   path = manager.download("isic2018")
   img, mask = manager.load_sample("isic2018", "train", 0)
   ```

## 🤝 Интеграция с другими модулями проекта

| Модуль | Использование LoadDatasets |
|--------|---------------------------|
| `BatchClassicTester` | Получение путей к датасетам для массового тестирования |
| `TorchSegmenter` | Создание `torch.utils.data.Dataset` через `create_pytorch_dataset()` |
| `MedicalSegmentationPipeline` | Загрузка медицинских датасетов с пост-обработкой |
| `BenchmarkRunner` | Автоматическая подготовка данных перед бенчмарком |
| `VisualizationTool` | Загрузка примеров через `load_sample()` для визуализации |

### Пример интеграции с PyTorch
```python
from torch.utils.data import DataLoader
from datasets.load_datasets import DatasetManager

manager = DatasetManager(base_dir="./data")
manager.download("cityscapes")  # Гарантируем наличие данных

# Создание Dataset с аугментациями
from torchvision import transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

dataset = manager.create_pytorch_dataset(
    "cityscapes", 
    split="train", 
    transform=train_transform
)

# DataLoader для обучения
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
for batch in loader:
    images, masks = batch["image"], batch["mask"]
    # ... training loop
```

## 📦 Зависимости

### Обязательные
```text
numpy>=1.20          # Массивы, статистики, векторизация
Pillow>=9.0          # Работа с изображениями (JPG, PNG)
requests>=2.25       # HTTP-запросы для загрузки
tqdm>=4.60           # Прогресс-бары
torch>=1.9           # PyTorch Dataset интеграция
```

### Опциональные (для расширенной функциональности)
```text
PyYAML>=6.0          # Загрузка конфигураций из YAML
huggingface_hub>=0.10  # Загрузка из HuggingFace Hub
datasets>=2.0        # Native поддержка HF datasets
pandas>=1.3          # Конвертация Parquet → файлы
pydicom>=2.0         # Работа с DICOM (медицинские датасеты)
nibabel>=3.2         # Работа с NIfTI (3D медицинские данные)
opencv-python>=4.5   # CLAHE и медицинская пост-обработка
```

### Установка
```bash
# Базовая установка
pip install numpy Pillow requests tqdm torch

# Полная установка (все опциональные зависимости)
pip install -e ".[medical,hf]"

# Или вручную:
pip install PyYAML huggingface_hub datasets pandas pydicom nibabel opencv-python
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

> 💡 **Совет:** Для добавления нового датасета создайте запись в `configs/datasets.yaml` или зарегистрируйте через `DatasetConfig` в коде. Убедитесь, что структура после распаковки соответствует ожидаемому формату `images/{split}/*` и `annotations/{split}/*`.