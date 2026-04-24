"""
Конфигурация исследования оптимизации памяти.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum, auto


class MemoryPolicy(Enum):
    """Стратегии управления памятью."""

    LAZY = auto()  # Ленивая аллокация (только при необходимости)
    POOLED = auto()  # Пул предвыделенных буферов
    REUSE = auto()  # Повторное использование тензоров
    AGGRESSIVE_GC = auto()  # Агрессивный сборщик мусора
    PINNED = auto()  # Закреплённая память для CPU↔GPU


@dataclass
class MemoryConfig:
    """Конфигурация параметров оптимизации памяти."""

    # Политика управления памятью
    policy: MemoryPolicy = MemoryPolicy.POOLED

    # Параметры кэширования
    cache_kernels: bool = True
    cache_max_size: int = 100  # Макс. число кэшируемых ядер
    cache_ttl_seconds: float = 300.0  # Время жизни кэша (сек)

    # Параметры профилирования
    profile_allocations: bool = True
    track_peak_memory: bool = True
    detect_leaks: bool = True
    leak_threshold_mb: float = 50.0  # Порог для детекта утечки

    # Параметры бенчмарка
    n_runs: int = 20
    warmup_runs: int = 5
    sync_cuda: bool = True

    # Очистка памяти
    clear_cache_between_runs: bool = True
    empty_cuda_cache: bool = True
    collect_garbage: bool = True

    # Фильтрация методов
    exclude_methods: List[str] = field(default_factory=list)
    include_methods: Optional[List[str]] = None

    # Вывод
    verbose: bool = True
    output_dir: str = "./results/memory_optimization"

    # Аппаратные настройки
    device: str = "cuda"  # "cuda" или "cpu"
    pinned_memory: bool = False  # Использовать pinned memory для CPU↔GPU

    def __post_init__(self):
        """Валидация конфигурации."""
        if self.cache_max_size < 1:
            raise ValueError("cache_max_size must be >= 1")
        if self.leak_threshold_mb < 0:
            raise ValueError("leak_threshold_mb must be >= 0")


# Конфигурация по умолчанию
DEFAULT_CONFIG = MemoryConfig()

# Политики по типам методов
MEMORY_POLICIES: Dict[str, MemoryPolicy] = {
    # Методы с локальной статистикой — хорошо кэшируются ядра
    "niblack_thresholding": MemoryPolicy.POOLED,
    "sauvola_thresholding": MemoryPolicy.POOLED,
    "adaptive_thresholding": MemoryPolicy.POOLED,
    # Градиентные методы — кэшируем ядра свёртки
    "sobel_edge": MemoryPolicy.REUSE,
    "prewitt_edge": MemoryPolicy.REUSE,
    "scharr_edge": MemoryPolicy.REUSE,
    # Итеративные методы — агрессивный GC
    "active_contour": MemoryPolicy.AGGRESSIVE_GC,
    "chan_vese": MemoryPolicy.AGGRESSIVE_GC,
    "morphological_snakes": MemoryPolicy.AGGRESSIVE_GC,
    # По умолчанию
    "default": MemoryPolicy.LAZY,
}
