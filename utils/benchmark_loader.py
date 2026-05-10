# utils/benchmark_loader.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations  # PEP 563: отложенная оценка аннотаций
import pandas as pd
import json
from typing import List, Dict, Any
from segmenters.AutoSegmenter import MethodProfile, ImageType

import logging

# Настройка логгера
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ──────────────────────────────────────────────────────────────────────
def load_profiles_from_benchmark(
    benchmark_csv_path: str, validation_json_path: str, library: str = "opencv"
) -> Dict[str, MethodProfile]:
    """
    Загрузка профилей методов из результатов ваших бенчмарков.
    """
    # Чтение CSV с бенчмарками
    df: pd.DataFrame = pd.read_csv(benchmark_csv_path)

    # Чтение JSON с валидацией
    with open(validation_json_path, "r") as f:
        validation_data = json.load(f)

    profiles: Dict[str, MethodProfile] = {}

    for method_name in df["Method"].unique():
        method_data = df[df["Method"] == method_name].iloc[0]

        # Извлечение метрик из валидации
        val_metrics: Dict[str, Any] = validation_data.get(method_name, {})
        iou: float = val_metrics.get("iou", 0.75)  # Default fallback

        # Определение лучшего типа (можно улучшить)
        best_types: List[ImageType] = [ImageType.NATURAL]  # Заглушка

        profiles[method_name] = MethodProfile(
            name=method_name,
            library=library,
            avg_time_ms=method_data["Mean_Time_s"] * 1000,  # sec -> ms
            avg_iou=iou,
            memory_mb=50,  # Можно добавить замер памяти
            best_for_type=best_types,
            robustness=0.8,  # Можно рассчитать из std
            parameter_sensitivity=0.5,
        )

    return profiles
