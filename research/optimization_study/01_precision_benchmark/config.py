"""
Конфигурация исследования точности вычислений.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class PrecisionConfig:
    """Конфигурация параметров бенчмарка точности."""

    # Доступные типы данных для тестирования
    precisions: List[str] = field(default_factory=lambda: ["fp32", "fp16", "bf16"])

    # Параметры бенчмарка
    n_runs: int = 50  # Количество запусков для замера
    warmup_runs: int = 10  # Прогревочные запуски
    sync_cuda: bool = True  # Синхронизация CUDA для точных замеров

    # Параметры точности
    reference_precision: str = "fp32"  # Эталон для сравнения
    tolerance: float = 1e-4  # Допустимая погрешность для "идентичных" результатов

    # Настройки autocast
    enable_autocast: bool = True
    autocast_cpu_enabled: bool = False  # autocast на CPU может не давать выигрыша

    # Фильтрация методов
    exclude_methods: List[str] = field(default_factory=list)
    include_methods: Optional[List[str]] = None  # Если None — все методы

    # Вывод
    verbose: bool = True
    log_level: str = "INFO"  # DEBUG, INFO, WARNING

    # Пути для сохранения
    output_dir: str = "./results/precision_benchmark"
    save_raw_data: bool = True
    save_plots: bool = True

    def __post_init__(self):
        """Валидация конфигурации."""
        valid_precisions = {"fp32", "fp16", "bf16", "int8"}
        for p in self.precisions:
            if p not in valid_precisions:
                raise ValueError(
                    f"Invalid precision: {p}. " f"Available: {valid_precisions}"
                )

        if self.reference_precision not in self.precisions:
            raise ValueError(
                f"Reference precision '{self.reference_precision}' "
                f"not in precisions list"
            )


# Конфигурация по умолчанию
DEFAULT_CONFIG = PrecisionConfig()

# Сопоставление строк → torch.dtype
PRECISION_TO_DTYPE: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,  # Требует специальной обработки
}


# Поддержка dtypes на разных устройствах
def is_dtype_supported(dtype: torch.dtype, device: str) -> bool:
    """Проверяет поддержку типа данных на устройстве."""
    if device == "cpu":
        # CPU поддерживает fp32, fp16 (медленно), bf16 (медленно)
        return dtype in [torch.float32, torch.float16, torch.bfloat16]

    if not torch.cuda.is_available():
        return dtype == torch.float32

    # Проверка поддержки на GPU
    if dtype == torch.bfloat16:
        # BF16 требует Ampere+ (compute capability >= 8.0)
        return torch.cuda.get_device_capability()[0] >= 8

    if dtype == torch.float16:
        # FP16 требует CUDA GPU с поддержкой
        return torch.cuda.is_available()

    return True
