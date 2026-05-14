# utils/palettes.py

"""Цветовые палитры и имена классов для датасетов семантической сегментации.

Поддерживаемые датасеты:
1. **ADE20K** (150 классов): универсальная сегментация сцен
2. **COCO** (80 классов): детекция объектов, инстанс-сегментация
3. **Cityscapes** (19/34 класса): автономное вождение, уличные сцены
4. **CheXpert** (14 классов): медицинская классификация рентгеновских снимков
5. **ISIC 2018** (2 класса): бинарная сегментация кожных поражений

Ключевые особенности:
- ✅ Уникальные цвета: каждый класс получает визуально различимый [R,G,B]
- ✅ Совместимость: формат [R,G,B] для OpenCV/PIL, 0-индексация
- ✅ Детерминированность: фиксированные значения для воспроизводимости
- ✅ Расширяемость: легко добавить новую палитру через List[List[int]]
- ✅ Логирование: информирование о загрузке классов и диапазоне индексов

Типичный workflow:
```python
from utils.palettes import ade_palette, get_ade_class_names
import numpy as np
from PIL import Image

# 1. Получение палитры и имён
palette = ade_palette()  # 150×[R,G,B]
class_names = get_ade_class_names()  # {0: "wall", 1: "building", ...}

# 2. Визуализация маски
mask = np.random.randint(0, 150, (512, 512), dtype=np.uint8)
palette_array = np.array(palette, dtype=np.uint8)
color_mask = palette_array[mask]  # Векторизованное применение
Image.fromarray(color_mask).save("result.png")

# 3. Анализ распределения классов
from collections import Counter
counts = Counter(mask.flatten())
for class_id, count in counts.most_common(5):
    name = class_names.get(class_id, f"Class_{class_id}")
    print(f"{name}: {count} pixels")
```

Note:
- Все палитры возвращают список списков `[R, G, B]` с целочисленными значениями 0–255.
- Имена классов используют 0-индексацию, соответствующую стандартам PyTorch/TensorFlow.
- Для бинарной сегментации используйте `binary_palette()` с масками значений {0, 1}.
- При визуализации убедитесь, что `len(palette) >= mask.max() + 1` для избежания IndexError.
- Логирование включено на уровне INFO; для отладки установите DEBUG через `logging.getLogger("utils.palettes").setLevel(logging.DEBUG)`.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций
from typing import (
    List,
    Dict,
)

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ──────────────────────────────────────────────────────────────────────
def get_ade_class_names() -> Dict[int, str]:
    """Получение ade20k_class_names."""
    # ADE20K Class Names (0-indexed, 150 classes)
    # Source: http://sceneparsing.csail.mit.edu/
    ade20k_class_names: Dict[int, str] = {
        0: "wall",
        1: "building",
        2: "sky",
        3: "floor",
        4: "tree",
        5: "ceiling",
        6: "road",
        7: "bed",
        8: "windowpane",
        9: "grass",
        10: "cabinet",
        11: "sidewalk",
        12: "person",
        13: "earth",
        14: "door",
        15: "table",
        16: "mountain",
        17: "plant",
        18: "curtain",
        19: "chair",
        20: "car",
        21: "water",
        22: "painting",
        23: "sofa",
        24: "shelf",
        25: "house",
        26: "sea",
        27: "mirror",
        28: "rug",
        29: "field",
        30: "armchair",
        31: "seat",
        32: "fence",
        33: "desk",
        34: "rock",
        35: "wardrobe",
        36: "lamp",
        37: "bathtub",
        38: "railing",
        39: "cushion",
        40: "base",
        41: "box",
        42: "column",
        43: "signboard",
        44: "chest of drawers",
        45: "counter",
        46: "sand",
        47: "sink",
        48: "skyscraper",
        49: "fireplace",
        50: "refrigerator",
        51: "grandstand",
        52: "path",
        53: "stairs",
        54: "runway",
        55: "case",
        56: "pool table",
        57: "pillow",
        58: "screen door",
        59: "stairway",
        60: "river",
        61: "bridge",
        62: "bookcase",
        63: "blind",
        64: "coffee table",
        65: "toilet",
        66: "flower",
        67: "book",
        68: "hill",
        69: "bench",
        70: "countertop",
        71: "stove",
        72: "palm",
        73: "kitchen island",
        74: "computer",
        75: "swivel chair",
        76: "boat",
        77: "bar",
        78: "arcade machine",
        79: "hovel",
        80: "bus",
        81: "towel",
        82: "light",
        83: "truck",
        84: "tower",
        85: "chandelier",
        86: "awning",
        87: "streetlight",
        88: "booth",
        89: "television receiver",
        90: "airplane",
        91: "dirt track",
        92: "apparel",
        93: "pole",
        94: "land",
        95: "bannister",
        96: "escalator",
        97: "ottoman",
        98: "bottle",
        99: "buffet",
        100: "poster",
        101: "stage",
        102: "van",
        103: "ship",
        104: "fountain",
        105: "conveyer belt",
        106: "canopy",
        107: "washer",
        108: "plaything",
        109: "swimming pool",
        110: "stool",
        111: "barrel",
        112: "basket",
        113: "waterfall",
        114: "tent",
        115: "bag",
        116: "minibike",
        117: "cradle",
        118: "oven",
        119: "ball",
        120: "food",
        121: "step",
        122: "tank",
        123: "trade name",
        124: "microwave",
        125: "pot",
        126: "animal",
        127: "bicycle",
        128: "lake",
        129: "dishwasher",
        130: "screen",
        131: "blanket",
        132: "sculpture",
        133: "hood",
        134: "sconce",
        135: "vase",
        136: "traffic light",
        137: "tray",
        138: "ashcan",
        139: "fan",
        140: "pier",
        141: "crt screen",
        142: "plate",
        143: "monitor",
        144: "bulletin board",
        145: "shower",
        146: "radiator",
        147: "glass",
        148: "clock",
        149: "flag",
    }
    logger.info(f"✅ ADE20K classes loaded: {len(ade20k_class_names)} classes")
    logger.info(
        f"   Range: [{min(ade20k_class_names.keys())}..{max(ade20k_class_names.keys())}]"
    )
    return ade20k_class_names


# ──────────────────────────────────────────────────────────────────────
def ade_palette() -> List[List[int]]:
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]


# ──────────────────────────────────────────────────────────────────────
def get_coco_class_names() -> Dict[int, str]:
    """Получение COCO_class_names."""
    # COCO Class Names (0-indexed, 80 classes)
    # Source: https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
    COCO_class_names: Dict[int, str] = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }
    logger.info(f"✅ COCO classes loaded: {len(COCO_class_names)} classes")
    logger.info(
        f"   Range: [{min(COCO_class_names.keys())}..{max(COCO_class_names.keys())}]"
    )
    return COCO_class_names


# ──────────────────────────────────────────────────────────────────────
def coco_palette() -> List[List[int]]:
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
    ]


# ──────────────────────────────────────────────────────────────────────
def get_cityscapes_extended_class_names() -> Dict[int, str]:
    """Получение cityscapes_extended_class_names."""
    # Cityscapes Extended (34 classes - includes "grouped" categories)
    cityscapes_extended_class_names: Dict[int, str] = {
        0: "road",
        1: "sidewalk",
        2: "parking",
        3: "rail track",
        4: "building",
        5: "wall",
        6: "fence",
        7: "guard rail",
        8: "bridge",
        9: "tunnel",
        10: "pole",
        11: "polegroup",
        12: "traffic light",
        13: "traffic sign",
        14: "vegetation",
        15: "terrain",
        16: "sky",
        17: "person",
        18: "rider",
        19: "car",
        20: "truck",
        21: "bus",
        22: "caravan",
        23: "trailer",
        24: "train",
        25: "motorcycle",
        26: "bicycle",
        27: "license plate",
        28: "ground",
        29: "static",
        30: "dynamic",
        31: "unlabeled",
        32: "ego vehicle",
        33: "rectification border",
    }
    logger.info(
        f"✅ Cityscapes Extended classes loaded: {len(cityscapes_extended_class_names)} classes"
    )
    logger.info(
        f"   Range: [{min(cityscapes_extended_class_names.keys())}..{max(cityscapes_extended_class_names.keys())}]"
    )
    return cityscapes_extended_class_names


# ──────────────────────────────────────────────────────────────────────
def cityscapes_extended_palette() -> List[List[int]]:
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
    ]


# ──────────────────────────────────────────────────────────────────────
def get_cityscapes_class_names() -> Dict[int, str]:
    """Получение cityscapes_class_names."""
    # Cityscapes Class Names (0-indexed, 19 classes for semantic segmentation)
    # Source: https://www.cityscapes-dataset.com/
    cityscapes_class_names: Dict[int, str] = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic light",
        7: "traffic sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle",
    }
    logger.info(f"✅ Cityscapes classes loaded: {len(cityscapes_class_names)} classes")
    logger.info(
        f"   Range: [{min(cityscapes_class_names.keys())}..{max(cityscapes_class_names.keys())}]"
    )
    return cityscapes_class_names


# ──────────────────────────────────────────────────────────────────────
def cityscapes_palette() -> List[List[int]]:
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
    ]


# ──────────────────────────────────────────────────────────────────────
def get_chexpert_observation_class_names() -> Dict[int, str]:
    """Получение chexpert_observation_names."""
    # CheXpert Observation Classes (14 labels for classification)
    # Source: https://stanfordmlgroup.github.io/competitions/chexpert/
    chexpert_observation_names: Dict[int, str] = {
        0: "No Finding",
        1: "Enlarged Cardiomediastinum",
        2: "Cardiomegaly",
        3: "Lung Opacity",
        4: "Lung Lesion",
        5: "Edema",
        6: "Consolidation",
        7: "Pneumonia",
        8: "Atelectasis",
        9: "Pneumothorax",
        10: "Pleural Effusion",
        11: "Pleural Other",
        12: "Fracture",
        13: "Support Devices",
    }

    # Для сегментации лёгких (если есть маски):
    chest_segmentation_class_names: Dict[int, str] = {
        0: "background",  # Non-lung area
        1: "lung",  # Lung field (left + right)
    }

    logger.info(f"✅ CheXpert observations: {len(chexpert_observation_names)} classes")
    logger.info(
        f"✅ Chest segmentation: {len(chest_segmentation_class_names)} classes (binary)"
    )
    return chexpert_observation_names


# ──────────────────────────────────────────────────────────────────────
def chexpert_observation_palette() -> List[List[int]]:
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
    ]


# ──────────────────────────────────────────────────────────────────────
def get_isic_class_names() -> Dict[int, str]:
    """Получение isic_class_names."""
    # ISIC 2018 Class Names (Binary: skin lesion segmentation)
    # Source: https://challenge.isic-archive.com/
    isic_class_names: Dict[int, str] = {
        0: "background",  # Healthy skin / non-lesion area
        1: "lesion",  # Skin lesion (melanoma, nevus, etc.)
    }

    logger.info(f"✅ ISIC classes loaded: {len(isic_class_names)} classes (binary)")
    logger.info(f"   Classes: {list(isic_class_names.values())}")
    return isic_class_names


# ──────────────────────────────────────────────────────────────────────
def binary_palette() -> List[List[int]]:
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120]]
