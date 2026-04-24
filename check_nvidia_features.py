#!/usr/bin/env python3
import sys
import torch
import subprocess

import torch


def print_gpu_mem():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"🔋 GPU: {alloc:.2f} GB / {reserved:.2f} GB reserved")


# Вызывайте после каждого epoch или batch
print_gpu_mem()


def run(cmd):
    try:
        return (
            subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            .decode()
            .strip()
        )
    except:
        return "❌ Ошибка"


print("🔍 NVIDIA Feature Check\n" + "=" * 40)

# 1. Базовый драйвер
print(
    "1. nvidia-smi:", run("nvidia-smi --query-gpu=name --format=csv,noheader") or "❌"
)

# 2. CUDA Compiler
print("2. nvcc:", run("nvcc --version | grep release") or "❌ Не установлен")

# 3. PyTorch CUDA
if torch.cuda.is_available():
    print("3. PyTorch CUDA: ✅")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
else:
    print("3. PyTorch CUDA: ❌")

# 4. NVIDIA Management Library (через pynvml)
try:
    import pynvml

    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    print(f"4. pynvml (NVML): ✅ {count} GPU(s)")
    pynvml.nvmlShutdown()
except ImportError:
    print("4. pynvml: ⚠️ Не установлен. pip install pynvml")
except Exception as e:
    print(f"4. pynvml: ❌ {e}")

# 5. cuDNN (проверка через PyTorch)
try:
    print(f"5. cuDNN: ✅ v{torch.backends.cudnn.version()}")
except:
    print("5. cuDNN: ❌ Не активна")

# 6. TensorRT (если установлен)
try:
    import tensorrt

    print(f"6. TensorRT: ✅ v{tensorrt.__version__}")
except ImportError:
    print("6. TensorRT: ⚠️ Не установлен")
except Exception as e:
    print(f"6. TensorRT: ❌ {e}")

# 7. NVDEC/NVENC (аппаратное кодирование/декодирование)
print(
    "7. NVDEC/NVENC:",
    run("nvidia-smi --query-gpu=decoder_stat,encoder_stat --format=csv,noheader")
    or "❌",
)

print("\n" + "=" * 40)
print("💡 Если PyTorch CUDA работает, обучение моделей будет работать штатно.")
print("   nvidia-smi нужен только для мониторинга и отладки драйвера.")
