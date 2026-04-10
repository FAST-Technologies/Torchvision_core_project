import torch

torch.cuda.empty_cache()
idx = 0  # номер GPU (0, 1, 2...)
print(f"GPU: {torch.cuda.get_device_name(idx)}")
print(
    f"Всего VRAM: {torch.cuda.get_device_properties(idx).total_memory / 1024**3:.2f} GB"
)
print(f"Занято тензорами: {torch.cuda.memory_allocated(idx) / 1024**3:.2f} GB")
print(f"Зарезервировано кэшем: {torch.cuda.memory_reserved(idx) / 1024**3:.2f} GB")
