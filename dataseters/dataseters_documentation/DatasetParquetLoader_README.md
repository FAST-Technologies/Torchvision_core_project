# 🔄 convert_coco_parquet — Конвертер COCO-panoptic Parquet → файловая структура

## 📖 Описание
Скрипт `scripts/convert_coco_parquet.py` предназначен для **конвертации датасета COCO-panoptic** из формата Parquet (HuggingFace datasets) в стандартную файловую структуру изображений и масок.

> ⚠️ **Важно:** Данный скрипт является *утилитой предобработки данных*, а не модулем сегментации. Он подготавливает данные для последующего использования с `DatasetManager`, `ADE20KDataset` или другими загрузчиками проекта.

## ✨ Ключевые возможности

### 🗂️ Конвертация форматов
| Исходный формат | Целевой формат | Описание |
|----------------|----------------|----------|
| Parquet (HF datasets) | JPG + PNG файлы | Извлечение бинарных данных изображений и масок |
| `dict{"bytes": ...}` | PIL.Image | Декодирование через `BytesIO` |
| Встроенные метаданные | Отдельные файлы | Разделение на `images/` и `annotations/` |

### 🎯 Стандартная структура после конвертации
```
coco-panoptic-val2017/
├── images/
│   └── test/
│       ├── coco-panoptic-val2017_0000001.jpg  # RGB изображение
│       ├── coco-panoptic-val2017_0000002.jpg
│       └── ...
└── annotations/
    └── test/
        ├── coco-panoptic-val2017_0000001.png  # Grayscale маска (режим "L")
        ├── coco-panoptic-val2017_0000002.png
        └── ...
```

### 🔧 Особенности обработки
- **Изображения:** Сохраняются в режиме `RGB` с качеством `95` (JPG)
- **Маски:** Сохраняются в режиме `L` (grayscale, 8-bit) для совместимости с сегментацией
- **Именование:** Единый шаблон `coco-panoptic-val2017_{idx:06d}1.{ext}` для синхронизации пар
- **Логирование:** Прогресс-бар с интервалом `BATCH_LOG_INTERVAL`

## 🚀 Быстрый старт

### Базовое использование
```bash
# 1. Убедитесь, что зависимости установлены
pip install pandas pillow pyarrow  # pyarrow для чтения Parquet

# 2. Отредактируйте пути в скрипте (или передайте через аргументы)
#    В секции КОНСТАНТЫ укажите актуальные пути:
#    PARQUET_PATH = "path/to/your/file.parquet"
#    OUTPUT_DIR_PATH = "path/to/output/images"
#    MASK_DIR_PATH = "path/to/output/masks"

# 3. Запустите конвертацию
python scripts/convert_coco_parquet.py
```

### Пример вывода в консоль
```
2024-01-15 14:32:10 - __main__ - INFO - 📊 Processing 5000 rows from train-00001-of-00002-94aa8570497c415e.parquet...
2024-01-15 14:32:10 - __main__ - INFO - Processed 0/5000
2024-01-15 14:32:45 - __main__ - INFO - Processed 100/5000
2024-01-15 14:33:20 - __main__ - INFO - Processed 200/5000
...
2024-01-15 14:45:00 - __main__ - INFO - ✅ Done!
```

### Проверка результата
```bash
# Проверка структуры
tree coco-panoptic-val2017/ -L 3

# Проверка файлов
ls -lh coco-panoptic-val2017/images/test/ | head
ls -lh coco-panoptic-val2017/annotations/test/ | head

# Визуальная проверка пары
python -c "
from PIL import Image
img = Image.open('coco-panoptic-val2017/images/test/coco-panoptic-val2017_0000001.jpg')
mask = Image.open('coco-panoptic-val2017/annotations/test/coco-panoptic-val2017_0000001.png')
print(f'Image: {img.size}, mode={img.mode}')
print(f'Mask:  {mask.size}, mode={mask.mode}')
"
```

## ⚙️ Конфигурация

### Константы скрипта (редактировать в коде)
| Константа | Тип | По умолчанию | Описание |
|-----------|-----|--------------|----------|
| `PARQUET_PATH` | `PathLike` | `"coco-panoptic-val2017/data/train-00001-of-00002-94aa8570497c415e.parquet"` | Путь к исходному Parquet-файлу |
| `OUTPUT_DIR_PATH` | `PathLike` | `"coco-panoptic-val2017/images/test"` | Директория для сохранения изображений |
| `MASK_DIR_PATH` | `PathLike` | `"coco-panoptic-val2017/annotations/test"` | Директория для сохранения масок |
| `BATCH_LOG_INTERVAL` | `int` | `100` | Интервал логирования прогресса (в строках) |

### Параметры функции `decode_image_from_dict()`
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `data` | `Any` | — | Словарь с ключом `"bytes"` или `None` |
| `mode` | `str` | `"RGB"` | Режим конвертации PIL: `"RGB"`, `"L"`, `"RGBA"` |

### Формат входных данных (Parquet schema)
Ожидаемая структура строки Parquet:
```python
{
    "image": {  # Ключ для изображения
        "bytes": b"...\x89PNG\r\n\x1a\n..."  # или JPEG bytes
    },
    "label": {  # Ключ для паноптической маски
        "bytes": b"...\x89PNG\r\n\x1a\n..."  # Grayscale PNG bytes
    },
    # ... другие метаданные (игнорируются)
}
```

## 📚 Справочник функций

### 🔹 `decode_image_from_dict(data, mode="RGB")`
```python
def decode_image_from_dict(
    data: Any, 
    mode: str = "RGB"
) -> Optional[Image.Image]:
    """Декодирует изображение из словаря с ключом "bytes".
    
    Args:
        data: Данные изображения (ожидается `dict` с ключом `"bytes"`).
        mode: Режим PIL для конвертации ("RGB", "L", ...).
    
    Returns:
        Optional[PIL.Image]: Декодированное изображение или `None` при ошибке.
    
    Example:
        >>> row = {"image": {"bytes": b"\x89PNG..."}}
        >>> img = decode_image_from_dict(row["image"], mode="RGB")
        >>> print(img.mode)  # "RGB"
    """
```

**Логика работы:**
```
1. Проверка: isinstance(data, dict) and "bytes" in data
   ↓
2. Извлечение: data["bytes"] → bytes объект
   ↓
3. Декодирование: BytesIO(bytes) → PIL.Image.open()
   ↓
4. Конвертация: .convert(mode) → возврат Image или None
```

### 🔹 `main()` — точка входа
```python
def main() -> None:
    """Основной входной пункт скрипта.
    
    Логика:
    1. Чтение Parquet-файла через pandas
    2. Создание выходных директорий
    3. Итерация по строкам:
       - Декодирование изображения → сохранение как JPG
       - Декодирование маски → сохранение как PNG (grayscale)
       - Логирование прогресса
    4. Завершение с сообщением об успехе
    """
```

**Поток данных:**
```
📄 Parquet файл
   ↓
📊 pandas.DataFrame (iterrows)
   ↓
🖼️  decode_image_from_dict(row["image"], "RGB") → save as .jpg
🎭 decode_image_from_dict(row["label"], "L")     → save as .png
   ↓
📁 filesystem: images/test/ + annotations/test/
```

## ⚡ Производительность и оптимизации

### Оценочное время конвертации
| Размер датасета | Примерное время | Память |
|----------------|-----------------|--------|
| 100 изображений | ~10 секунд | ~200 MB |
| 1,000 изображений | ~2 минуты | ~500 MB |
| 5,000 изображений (COCO val) | ~10 минут | ~1.5 GB |
| 118,000 изображений (COCO train) | ~4 часа | ~3 GB |

### Рекомендации по ускорению
```python
# 1. Используйте многопроцессорность для больших датасетов
#    (требует модификации скрипта с multiprocessing.Pool)

# 2. Увеличьте BATCH_LOG_INTERVAL для снижения накладных расходов
BATCH_LOG_INTERVAL = 500  # Вместо 100

# 3. Сохраняйте на SSD, а не HDD
OUTPUT_DIR_PATH = "/mnt/ssd/coco/images"  # Быстрее запись

# 4. Отключите логирование для максимальной скорости
logger.setLevel(logging.ERROR)  # Только ошибки
```

### Оптимизация памяти при чтении Parquet
```python
# Для очень больших файлов используйте chunked reading:
for chunk in pd.read_parquet(pq_file, chunksize=1000):
    for idx, row in chunk.iterrows():
        # ... обработка ...
    # Освобождение памяти после каждого чанка
    del chunk
```

## 🛠️ Обработка ошибок и устойчивость

### Стратегия "мягкого" отказа
```python
# Если изображение не декодируется — просто пропускаем строку
img: Optional[Image.Image] = decode_image_from_dict(img_data, mode="RGB")
if img is not None:
    img.save(img_path, quality=95)
# else: silently skip (логировать можно при отладке)
```

**Преимущества:**
- ✅ Конвертация не прерывается из-за одного битого файла
- ✅ Статистика успеха/ошибок может быть добавлена постфактум
- ✅ Подходит для пакетной обработки с минимальным вмешательством

### Рекомендации по отладке
```python
# 1. Включите подробное логирование
logger.setLevel(logging.DEBUG)

# 2. Добавьте логирование пропущенных файлов
#    В decode_image_from_dict():
if not (isinstance(data, dict) and "bytes" in data):
    logger.debug(f"⚠️ Invalid data format: {type(data)}")
    return None

# 3. Проверьте исходный Parquet файл
python -c "
import pandas as pd
df = pd.read_parquet('your_file.parquet')
print(df.columns.tolist())
print(df.iloc[0]['image'].keys())  # Проверка структуры
"
```

### Валидация результата
```bash
# Скрипт для проверки соответствия пар изображений и масок
python -c "
from pathlib import Path
img_dir = Path('coco-panoptic-val2017/images/test')
mask_dir = Path('coco-panoptic-val2017/annotations/test')

img_files = {f.stem for f in img_dir.glob('*.jpg')}
mask_files = {f.stem for f in mask_dir.glob('*.png')}

print(f'Images: {len(img_files)}, Masks: {len(mask_files)}')
print(f'Missing masks: {len(img_files - mask_files)}')
print(f'Missing images: {len(mask_files - img_files)}')
"
```

## 🤝 Интеграция с другими модулями проекта

| Модуль | Использование convert_coco_parquet |
|--------|-----------------------------------|
| `DatasetManager` | Загрузка конвертированных файлов через `download("coco")` |
| `ADE20KDataset` | Аналогичная структура позволяет переиспользовать код загрузки |
| `BatchClassicTester` | Тестирование классических методов на конвертированных данных |
| `TorchSegmenter` | Обучение моделей на подготовленных изображениях и масках |
| `SegmentationMetrics` | Оценка качества предсказаний против конвертированных масок |

### Пример пайплайна: конвертация → обучение
```python
# 1. Конвертация данных (запуск скрипта)
# $ python scripts/convert_coco_parquet.py

# 2. Использование в коде обучения
from datasets.load_datasets import DatasetManager
from torch.utils.data import DataLoader

manager = DatasetManager(base_dir="./data")

# Регистрация конфигурации для уже конвертированных данных
from datasets.load_datasets import DatasetConfig, DatasetType, SourceType
manager._registry["coco_converted"] = DatasetConfig(
    name="coco_converted",
    dataset_type=DatasetType.INSTANCE,
    source_type=SourceType.DIRECT,  # Уже на диске
    source_url="./coco-panoptic-val2017",
    root_dir="./data",
    num_classes=80,  # COCO: 80 классов объектов
    image_ext=".jpg",
    mask_ext=".png",
    splits={"val": "test"}  # Маппинг сплитов
)

# Загрузка без скачивания (файлы уже на месте)
dataset = manager.create_pytorch_dataset("coco_converted", split="val")
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# Обучение / инференс
for batch in loader:
    images, masks = batch["image"], batch["mask"]
    # ... ваш код ...
```

### Адаптация под другие Parquet-датасеты
```python
# Для конвертации другого датасета скопируйте скрипт и измените:
# 1. Пути в КОНСТАНТЫ
# 2. Ключи в row.get(): "image" → "your_image_key", "label" → "your_mask_key"
# 3. Шаблон именования файлов при необходимости

# Пример для ISIC-подобного датасета:
PARQUET_PATH = "isic-data/train.parquet"
OUTPUT_DIR_PATH = "isic-data/images/train"
MASK_DIR_PATH = "isic-data/masks/train"

# В цикле обработки:
img_data = row.get("input_image")   # Вместо "image"
mask_data = row.get("segmentation") # Вместо "label"
```

## 📦 Зависимости

### Обязательные
```text
pandas>=1.3.0          # Чтение Parquet-файлов
Pillow>=9.0.0          # Декодирование и сохранение изображений
pyarrow>=5.0.0         # Бэкенд для pandas.read_parquet()
```

### Опциональные (для отладки)
```text
tqdm>=4.60.0           # Прогресс-бар (можно добавить при желании)
```

### Установка
```bash
# Минимальный набор
pip install pandas pillow pyarrow

# Или через requirements проекта
pip install -r requirements-data.txt

# Проверка установки
python -c "import pandas, PIL, pyarrow; print('✅ Dependencies OK')"
```

## 🔧 Расширение и модификация

### Добавление прогресс-бара с tqdm
```python
# Замените цикл for на:
from tqdm import tqdm

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
    # ... существующая логика ...
```

### Поддержка многопроцессорной обработки
```python
# Пример с multiprocessing (требует рефакторинга):
from multiprocessing import Pool

def process_row(args: tuple) -> None:
    idx, row, output_dir, mask_dir = args
    # ... логика обработки одной строки ...

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        args_list = [
            (idx, row, output_dir, mask_dir) 
            for idx, row in df.iterrows()
        ]
        pool.map(process_row, args_list)
```

### Добавление валидации контрольных сумм
```python
# Для проверки целостности после сохранения:
import hashlib

def compute_md5(filepath: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# После img.save():
# logger.debug(f"Saved {img_path.name} (MD5: {compute_md5(img_path)[:8]}...)")
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

> 💡 **Совет:** Перед запуском на полном датасете протестируйте скрипт на небольшом подмножестве (первые 10–20 строк Parquet-файла), чтобы убедиться в корректности путей и формата данных.

```python
# Быстрый тест на 10 строках
df_test = pd.read_parquet(PARQUET_PATH).head(10)
for idx, row in df_test.iterrows():
    # ... ваша логика ...
    print(f"✓ Processed test row {idx}")
```