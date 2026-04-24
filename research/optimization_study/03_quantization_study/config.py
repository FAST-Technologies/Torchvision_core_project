"""
Конфигурация исследования квантования.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import torch

# Доступные схемы квантования
QUANTIZATION_SCHEMES = Literal[
    "fp32",  # Эталон (без квантования)
    "fp16",  # Половинная точность
    "int8_dynamic",  # Динамическое INT8 (активации квантуются на лету)
    "int8_static",  # Статическое INT8 (требует калибровки)
]

# Поддерживаемые типы данных
SUPPORTED_DTYPES: Dict[QUANTIZATION_SCHEMES, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "int8_dynamic": torch.qint8,
    "int8_static": torch.qint8,
}


@dataclass
class QuantizationConfig:
    """Конфигурация параметров квантования."""

    # Схемы квантования для тестирования
    schemes: List[QUANTIZATION_SCHEMES] = field(
        default_factory=lambda: ["fp32", "fp16", "int8_dynamic"]
    )

    # Параметры калибровки (для static INT8)
    calibration_steps: int = 100  # Количество изображений для калибровки
    calibration_batch_size: int = 1  # Размер батча при калибровке

    # Параметры бенчмарка
    n_runs: int = 50  # Количество запусков для замера
    warmup_runs: int = 10  # Прогревочные запуски
    sync_cuda: bool = True  # Синхронизация CUDA для точных замеров

    # Параметры точности
    reference_scheme: str = "fp32"  # Эталон для сравнения
    tolerance: float = 1e-3  # Допустимая погрешность для "идентичных" результатов

    # Настройки квантования
    per_channel: bool = False  # Per-channel vs per-tensor квантование
    reduce_range: bool = False  # Уменьшение диапазона для совместимости
    fuse_modules: bool = True  # Автоматическое слияние модулей (conv+bn+relu)

    # Фильтрация методов
    exclude_methods: List[str] = field(default_factory=list)
    include_methods: Optional[List[str]] = None  # Если None — все методы

    # Вывод
    verbose: bool = True
    log_level: str = "INFO"

    # Пути для сохранения
    output_dir: str = "./results/quantization_study"
    save_raw_data: bool = True
    save_plots: bool = True

    # Аппаратные настройки
    target_device: str = "cpu"  # Квантование лучше работает на CPU

    def __post_init__(self):
        """Валидация конфигурации."""
        valid_schemes = set(QUANTIZATION_SCHEMES.__args__)
        for s in self.schemes:
            if s not in valid_schemes:
                raise ValueError(f"Invalid scheme: {s}. " f"Available: {valid_schemes}")

        if self.reference_scheme not in self.schemes:
            raise ValueError(
                f"Reference scheme '{self.reference_scheme}' " f"not in schemes list"
            )

        # Предупреждение о поддержке INT8 на GPU
        if "int8_static" in self.schemes and self.target_device == "cuda":
            import warnings

            warnings.warn(
                "⚠️ Static INT8 quantization has limited GPU support. "
                "Consider using 'cpu' for best results."
            )


# Конфигурация по умолчанию
DEFAULT_CONFIG = QuantizationConfig()
