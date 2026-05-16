import torch
from typing import Dict, Any


def check_bf16_readiness() -> Dict[str, Any]:
    """Проверка готовности системы к bf16 тестированию."""
    result = {
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": False,
        "pytorch_version": torch.__version__,
        "recommendations": [],
    }

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        result["gpu"] = torch.cuda.get_device_name(0)
        result["compute_capability"] = f"{cap[0]}.{cap[1]}"
        result["bf16_supported"] = cap[0] >= 8

        if not result["bf16_supported"]:
            result["recommendations"].append("GPU не поддерживает bf16. Используйте fp16 или fp32.")

    if torch.__version__ < "1.9.0":
        result["recommendations"].append("Обновите PyTorch до >=1.9.0 для стабильной поддержки bf16")

    return result


# Использование:
status = check_bf16_readiness()
print(f"BF16 ready: {status['bf16_supported']}")
if status["recommendations"]:
    for rec in status["recommendations"]:
        print(f"⚠️ {rec}")
