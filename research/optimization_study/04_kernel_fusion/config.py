"""
Конфигурация исследования kernel fusion.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum, auto


class FusionStrategy(Enum):
    """Стратегии объединения операций."""

    NONE = auto()  # Без fusion
    GRAPH_FUSION = auto()  # torch.compile / FX graph fusion
    MANUAL_FUSION = auto()  # Ручное объединение операций
    CUSTOM_KERNEL = auto()  # Custom CUDA kernel
    VECTORIZED = auto()  # Векторизованные операции


@dataclass
class FusionConfig:
    """Конфигурация параметров fusion."""

    # Стратегия fusion
    strategy: FusionStrategy = FusionStrategy.GRAPH_FUSION

    # Параметры torch.compile
    compile_mode: str = "reduce-overhead"  # или "max-autotune"
    fullgraph: bool = True
    dynamic: bool = False

    # Параметры custom kernels
    enable_custom_kernels: bool = False
    kernel_cache_dir: str = "./cache/kernels"

    # Параметры векторизации
    enable_vectorization: bool = True
    vector_width: int = 8  # Ширина вектора для SIMD

    # Фильтрация методов
    exclude_methods: List[str] = field(default_factory=list)
    include_methods: Optional[List[str]] = None

    # Профилирование
    profile_graph: bool = True
    log_fusion_decisions: bool = True

    # Вывод
    verbose: bool = True
    output_dir: str = "./results/kernel_fusion"

    # Пороги для принятия решений
    min_ops_for_fusion: int = 3  # Мин. операций для применения fusion
    min_expected_speedup: float = 1.1  # Мин. ожидаемый выигрыш

    def __post_init__(self):
        """Валидация конфигурации."""
        if self.vector_width not in [4, 8, 16, 32]:
            raise ValueError(f"Invalid vector_width: {self.vector_width}")


# Конфигурация по умолчанию
DEFAULT_CONFIG = FusionConfig()

# Стратегии fusion по типам операций
FUSION_STRATEGIES: Dict[str, FusionStrategy] = {
    # Пороговые методы — хорошо векторизуются
    "global_thresholding": FusionStrategy.VECTORIZED,
    "otsu_thresholding": FusionStrategy.GRAPH_FUSION,
    "adaptive_thresholding": FusionStrategy.MANUAL_FUSION,
    "niblack_thresholding": FusionStrategy.MANUAL_FUSION,
    "sauvola_thresholding": FusionStrategy.MANUAL_FUSION,
    # Градиентные методы — хорошо fusion'ятся
    "sobel_edge": FusionStrategy.GRAPH_FUSION,
    "prewitt_edge": FusionStrategy.GRAPH_FUSION,
    "scharr_edge": FusionStrategy.GRAPH_FUSION,
    # Сложные методы — требуют custom kernels
    "canny_edge": FusionStrategy.CUSTOM_KERNEL,
    "phase_congruency_edge": FusionStrategy.CUSTOM_KERNEL,
    # По умолчанию
    "default": FusionStrategy.GRAPH_FUSION,
}
