# debug_palette.py
from utils.palettes import ade_palette, get_ade_class_names
import matplotlib.pyplot as plt
import numpy as np

# Загрузи палитру и имена классов
palette = ade_palette()
class_names = get_ade_class_names()

print("🎨 Проверка палитры ADE20K")
print("=" * 80)

# Выведи первые 20 классов с их цветами
print("\nТоп-20 классов ADE20K:")
print(f"{'Index':<6} {'Class Name':<25} {'RGB':<20} {'HEX'}")
print("-" * 80)

for idx in range(150):
    if idx in class_names:
        name = class_names[idx]
        rgb = palette[idx]
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        print(f"{idx:<6} {name:<25} {str(rgb):<20} {hex_color}")

# 🔍 Проверка критических классов
critical_classes = {
    0: "wall",
    1: "building",
    2: "sky",
    3: "floor",
    4: "tree",
    6: "road",
    9: "grass",
}

print("\n🔍 Критические классы для твоего изображения:")
print(f"{'Index':<6} {'Class Name':<25} {'RGB':<20} {'HEX'}")
print("-" * 80)

for idx, expected_name in critical_classes.items():
    actual_name = class_names.get(idx, "N/A")
    rgb = palette[idx]
    hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    match = "✓" if actual_name == expected_name else "✗"
    print(f"{match} {idx:<6} {actual_name:<25} {str(rgb):<20} {hex_color}")

# 📊 Визуализация палитры
fig, ax = plt.subplots(figsize=(15, 3))
ax.set_xlim(0, len(palette))
ax.set_ylim(0, 1)
ax.axis("off")

# Отобразим первые 50 классов
for idx in range(min(150, len(palette))):
    rgb = palette[idx]
    rgb_normalized = [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255]
    ax.add_patch(plt.Rectangle((idx, 0), 1, 1, color=rgb_normalized))
    ax.text(
        idx + 0.5, 0.5, str(idx), ha="center", va="center", fontsize=6, color="black" if sum(rgb) > 384 else "white"
    )

ax.set_title("ADE20K Color Palette (first 50 classes)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("debug_palette_visualization.png", dpi=300, bbox_inches="tight")
print("\n✅ Визуализация палитры сохранена: debug_palette_visualization.png")
plt.show()
