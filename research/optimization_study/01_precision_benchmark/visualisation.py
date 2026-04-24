"""
Визуализация результатов бенчмарка точности.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib/seaborn not installed. Visualization disabled.")


class PrecisionVisualizer:
    """
    Визуализатор результатов бенчмарка точности.

    Доступные типы графиков:
    - bar_time: Время выполнения по прецизионным режимам
    - bar_speedup: Коэффициент ускорения
    - scatter_tradeoff: Trade-off "скорость ↔ точность"
    - heatmap: Тепловая карта результатов
    """

    def __init__(self, output_dir: str = "./plots", style: str = "seaborn-v0_8"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if MATPLOTLIB_AVAILABLE:
            plt.style.use(style)
            sns.set_palette("husl")

    def plot_time_by_precision(
        self,
        df: pd.DataFrame,
        output_file: Optional[str] = None,
        figsize: tuple = (10, 6),
    ) -> Optional[str]:
        """Столбчатая диаграмма времени выполнения по прецизионным режимам."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if output_file is None:
            output_file = "time_by_precision.png"
        output_path = self.output_dir / output_file

        # Группировка данных
        plot_df = df.groupby(["method", "precision"])["median_ms"].mean().unstack()

        plt.figure(figsize=figsize)
        plot_df.plot(kind="bar", figsize=figsize)
        plt.ylabel("Time (ms)")
        plt.xlabel("Method")
        plt.title("Execution Time by Precision")
        plt.legend(title="Precision", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def plot_speedup(
        self,
        df: pd.DataFrame,
        reference_precision: str = "fp32",
        output_file: Optional[str] = None,
        figsize: tuple = (10, 6),
    ) -> Optional[str]:
        """Диаграмма коэффициента ускорения относительно reference."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if "speedup_vs_reference" not in df.columns:
            # Вычисляем speedup если нет в данных
            ref_times = df[df["precision"] == reference_precision].set_index("method")[
                "median_ms"
            ]
            df = df.copy()
            df["speedup_vs_reference"] = df.apply(
                lambda row: (
                    ref_times.get(row["method"], row["median_ms"]) / row["median_ms"]
                    if row["precision"] != reference_precision
                    else 1.0
                ),
                axis=1,
            )

        if output_file is None:
            output_file = "speedup_by_precision.png"
        output_path = self.output_dir / output_file

        plot_df = (
            df.groupby(["method", "precision"])["speedup_vs_reference"].mean().unstack()
        )

        plt.figure(figsize=figsize)
        plot_df.plot(kind="bar", figsize=figsize)
        plt.ylabel("Speedup vs FP32")
        plt.xlabel("Method")
        plt.title("Speedup by Precision (Reference: FP32)")
        plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No speedup")
        plt.legend(title="Precision", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def plot_tradeoff(
        self,
        df: pd.DataFrame,
        output_file: Optional[str] = None,
        figsize: tuple = (8, 8),
    ) -> Optional[str]:
        """Scatter plot: trade-off между скоростью и точностью."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if "pixel_agreement" not in df.columns:
            warnings.warn("No accuracy data available for tradeoff plot")
            return None

        if output_file is None:
            output_file = "speed_accuracy_tradeoff.png"
        output_path = self.output_dir / output_file

        plt.figure(figsize=figsize)

        colors = {"fp32": "blue", "fp16": "green", "bf16": "orange", "int8": "red"}

        for precision in df["precision"].unique():
            subset = df[df["precision"] == precision]
            plt.scatter(
                subset["median_ms"],
                subset["pixel_agreement"],
                label=precision,
                c=colors.get(precision),
                alpha=0.7,
                s=100,
                edgecolors="black",
            )

        plt.xlabel("Time (ms) — Lower is better →")
        plt.ylabel("Pixel Agreement vs FP32 — Higher is better ↑")
        plt.title("Speed vs Accuracy Trade-off")
        plt.legend(title="Precision")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def plot_all(
        self,
        df: pd.DataFrame,
        prefix: Optional[str] = None,
    ) -> Dict[str, str]:
        """Генерация всех доступных графиков."""
        if prefix is None:
            prefix = ""

        return {
            "time": self.plot_time_by_precision(
                df, output_file=f"{prefix}time_by_precision.png"
            ),
            "speedup": self.plot_speedup(
                df, output_file=f"{prefix}speedup_by_precision.png"
            ),
            "tradeoff": self.plot_tradeoff(df, output_file=f"{prefix}tradeoff.png"),
        }
