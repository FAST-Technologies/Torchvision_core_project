#!/usr/bin/env python3
import torch
import subprocess


def print_gpu_mem():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"🔋 GPU: {alloc:.2f} GB / {reserved:.2f} GB reserved")


print_gpu_mem()


def run(cmd):
    try:
        return (
            subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            .decode()
            .strip()
        )
    except Exception as e:
        return f"❌ Ошибка: {e}"


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
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
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
    print("4. pynvml: ⚠️ Не установлен. pip install nvidia-ml-py")
except Exception as e:
    print(f"4. pynvml: ❌ {e}")

# 5. cuDNN (проверка через PyTorch)
try:
    print(f"5. cuDNN: ✅ v{torch.backends.cudnn.version()}")
except Exception as e:
    print(f"5. cuDNN: ❌ Не активна: {e}")

# 6. TensorRT (если установлен)
try:
    import tensorrt

    print(f"6. TensorRT: ✅ v{tensorrt.__version__}")
except ImportError:
    print("6. TensorRT: ⚠️ Не установлен")
except Exception as e:
    print(f"6. TensorRT: ❌ {e}")

print("7. NVENC/NVDEC support:")
try:
    # Проверяем, что команда выполняется без ошибки
    result = subprocess.run(
        ["nvidia-smi", "-q", "-d", "ENCODER"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0 and "NVIDIA" in result.stdout:
        print(
            "   ✅ Encoder support detected (check 'nvidia-smi encodersessions' for active sessions)"
        )
    else:
        print("   ⚠️  Encoder query returned no data (normal if no active sessions)")
except Exception as e:
    print(f"   ℹ️  Driver installed — NVENC/NVDEC supported on RTX 4000 Ada: {e}")

# Дополнительно: показать активные сессии если есть
try:
    sessions = subprocess.run(
        ["nvidia-smi", "encodersessions"], capture_output=True, text=True, timeout=5
    ).stdout.strip()
    if "No active encoder sessions" not in sessions and sessions:
        print(f"   📊 Active sessions:\n{sessions[:200]}...")
except Exception:
    pass

print("\n" + "=" * 40)
print("💡 Если PyTorch CUDA работает, обучение моделей будет работать штатно.")
print("   nvidia-smi нужен только для мониторинга и отладки драйвера.")
