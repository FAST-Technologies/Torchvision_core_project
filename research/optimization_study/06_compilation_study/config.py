"""
Конфигурация исследования компиляционной оптимизации.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum, auto


class CompilationStrategy(Enum):
    """Стратегии компиляции."""

    NONE = auto()  # Без компиляции
    JIT_SCRIPT = auto()  # torch.jit.script
    JIT_TRACE = auto()  # torch.jit.trace
    TORCH_COMPILE = auto()  # torch.compile() (PyTorch 2.0+)
    GRAPH_FREEZE = auto()  # torch.jit.freeze + optimize_for_inference
    CUDAGRAPHS = auto()  # CUDA graphs для минимизации overhead


@dataclass
class CompilationConfig:
    """Конфигурация параметров компиляции."""

    # Стратегия компиляции
    strategy: CompilationStrategy = CompilationStrategy.TORCH_COMPILE

    # Параметры torch.compile
    compile_mode: str = "reduce-overhead"  # или "max-autotune", "default"
    fullgraph: bool = True  # Компилировать весь граф
    dynamic: bool = False  # Динамические shapes
    backend: str = "inductor"  # "inductor", "cudagraphs", "aot_eager"

    # Параметры torch.jit
    jit_optimize: bool = True  # Применять оптимизации после script/trace
    freeze_graph: bool = True  # Заморозить граф для инференса

    # Параметры бенчмарка
    n_runs: int = 50
    warmup_runs: int = 10
    sync_cuda: bool = True

    # Фильтрация методов
    exclude_methods: List[str] = field(default_factory=list)
    include_methods: Optional[List[str]] = None

    # Вывод
    verbose: bool = True
    log_compilation_time: bool = True
    output_dir: str = "./results/compilation_study"

    # Аппаратные настройки
    device: str = "cuda"

    def __post_init__(self):
        """Валидация конфигурации."""
        valid_modes = ["default", "reduce-overhead", "max-autotune"]
        if self.compile_mode not in valid_modes:
            raise ValueError(f"Invalid compile_mode: {self.compile_mode}. " f"Available: {valid_modes}")

        valid_backends = ["inductor", "cudagraphs", "aot_eager", "onnxrt"]
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend: {self.backend}. " f"Available: {valid_backends}")


# Конфигурация по умолчанию
DEFAULT_CONFIG = CompilationConfig()

# Стратегии по типам методов
COMPILATION_STRATEGIES: Dict[str, CompilationStrategy] = {
    # Простые методы — хорошо компилируются
    "global_thresholding": CompilationStrategy.TORCH_COMPILE,
    "otsu_thresholding": CompilationStrategy.TORCH_COMPILE,
    "adaptive_thresholding": CompilationStrategy.TORCH_COMPILE,
    # Градиентные методы — тоже хорошо
    "sobel_edge": CompilationStrategy.TORCH_COMPILE,
    "prewitt_edge": CompilationStrategy.TORCH_COMPILE,
    "scharr_edge": CompilationStrategy.TORCH_COMPILE,
    # Сложные методы — могут требовать JIT trace
    "canny_edge": CompilationStrategy.JIT_TRACE,
    "phase_congruency_edge": CompilationStrategy.JIT_TRACE,
    # Итеративные методы — могут не компилироваться
    "active_contour": CompilationStrategy.NONE,
    "chan_vese": CompilationStrategy.NONE,
    # По умолчанию
    "default": CompilationStrategy.TORCH_COMPILE,
}
