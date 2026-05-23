from utils.palettes import ade_palette, get_ade_class_names

palette = ade_palette()
class_names = get_ade_class_names()

print("🔍 Проверка соответствия палитры и имён:")
for idx in range(10):
    name = class_names.get(idx, "N/A")
    color = palette[idx]
    print(f"   {idx}: {name:20s} → {color}")
