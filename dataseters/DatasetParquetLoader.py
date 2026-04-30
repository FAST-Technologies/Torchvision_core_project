# scripts/convert_coco_parquet.py
"""
Скрипт для конвертации Parquet-файлов COCO-panoptic в файловую структуру.

Извлекает изображения и маски из Parquet-файла и сохраняет их как:
- Изображения: `coco-panoptic-val2017_{idx:06d}1.jpg` (RGB)
- Маски: `coco-panoptic-val2017_{idx:06d}1.png` (grayscale, mode "L")

Пример использования:
    ```bash
    python scripts/convert_coco_parquet.py
    ```
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ & TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
from typing import Dict, Any, Optional, Union
from pathlib import Path
from io import BytesIO

import pandas as pd
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ──────────────────────────────────────────────────────────────────────
PathLike = Union[str, Path]
ParquetRow = Dict[str, Any]
ImageDataDict = Dict[str, bytes]

# ──────────────────────────────────────────────────────────────────────
# КОНСТАНТЫ
# ──────────────────────────────────────────────────────────────────────
PARQUET_PATH: PathLike = (
    "coco-panoptic-val2017/data/train-00001-of-00002-94aa8570497c415e.parquet"
)
OUTPUT_DIR_PATH: PathLike = "coco-panoptic-val2017/images/test"
MASK_DIR_PATH: PathLike = "coco-panoptic-val2017/annotations/test"
BATCH_LOG_INTERVAL: int = 100


# ──────────────────────────────────────────────────────────────────────
# УТИЛИТЫ
# ──────────────────────────────────────────────────────────────────────
def decode_image_from_dict(data: Any, mode: str = "RGB") -> Optional[Image.Image]:
    """
    Декодирует изображение из словаря с ключом "bytes".

    Args:
        data: Данные изображения (ожидается `dict` с ключом `"bytes"`).
        mode: Режим PIL для конвертации ("RGB", "L", ...).

    Returns:
        Optional[PIL.Image]: Декодированное изображение или `None` при ошибке.
    """
    if isinstance(data, dict) and "bytes" in data and isinstance(data["bytes"], bytes):
        return Image.open(BytesIO(data["bytes"])).convert(mode)
    return None


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Основной входной пункт скрипта.

    Читает Parquet-файл, извлекает изображения и маски, сохраняет в файловую структуру.
    """
    pq_file: Path = Path(PARQUET_PATH)
    df: pd.DataFrame = pd.read_parquet(pq_file)

    output_dir: Path = Path(OUTPUT_DIR_PATH)
    mask_dir: Path = Path(MASK_DIR_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    total_rows: int = len(df)
    print(f"📊 Processing {total_rows} rows from {pq_file.name}...")

    for idx, row in df.iterrows():
        # ── Изображение ──────────────────────────────────────────────
        img_data: Any = row.get("image")
        img: Optional[Image.Image] = decode_image_from_dict(img_data, mode="RGB")
        if img is not None:
            img_path: Path = output_dir / f"coco-panoptic-val2017_{idx:06d}1.jpg"
            img.save(img_path, quality=95)

        # ── Маска ───────────────────────────────────────────────────
        mask_data: Any = row.get("label")
        if mask_data is not None:
            mask: Optional[Image.Image] = decode_image_from_dict(mask_data, mode="L")
            if mask is not None:
                mask_path: Path = mask_dir / f"coco-panoptic-val2017_{idx:06d}1.png"
                mask.save(mask_path)

        # ── Прогресс ────────────────────────────────────────────────
        if idx % BATCH_LOG_INTERVAL == 0:
            print(f"Processed {idx}/{total_rows}")

    print("✅ Done!")


if __name__ == "__main__":
    main()
