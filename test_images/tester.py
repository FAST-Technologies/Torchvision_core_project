# main.py

"""
Класс для вывода информации по тестируемым изображениям.
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from typing import Dict
from PIL import Image

image_paths: Dict[str, str] = {
    "war_frame_1": "2340_frame.jpg",
    "war_frame_2": "3330_frame.jpg",
    "war_frame_3": "4130_frame.jpg",
    "war_frame_4": "4480_frame.jpg",
    "building": "test_gt_image.jpg",
    "animals": "animals.jpg",
    "test_image_architecture": "test_image_architecture.jpg",
    "test_image_countryside": "test_image_countryside.jpg",
    "test_image_mountain": "test_image_mountain.jpg",
    "test_image_traffic": "test_image_traffic.jpg",
    "test_image_trucks": "test_image_trucks.jpg",
    "test_image_nature": "test_image_nature.jpg",
}

for name, path in image_paths.items():
    try:
        img: Image.Image = Image.open(path)
        print(f"✅ {name}: {img.size}, info: {img.info}")

    except Exception as e:
        print(f"❌ Ошибка загрузки {name}: {e}")
