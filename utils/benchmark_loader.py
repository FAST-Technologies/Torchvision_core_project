# utils/benchmark_loader.py

import pandas as pd
import json
from typing import Dict, Optional
from segmenters.AutoSegmenter import MethodProfile, ImageType


def load_profiles_from_benchmark(
    benchmark_csv_path: str, validation_json_path: str, library: Optional[str] = None
) -> Dict[str, MethodProfile]:
    """
    Загрузка профилей методов из результатов ваших бенчмарков.
    """
    # Чтение CSV с бенчмарками
    df = pd.read_csv(benchmark_csv_path)

    # Чтение JSON с валидацией
    with open(validation_json_path, "r") as f:
        validation_data = json.load(f)

    profiles = {}

    for method_name in df["Method"].unique():
        method_data = df[df["Method"] == method_name].iloc[0]

        # Извлечение метрик из валидации
        val_metrics = validation_data.get(method_name, {})
        iou = val_metrics.get("iou", 0.75)  # Default fallback

        # Определение лучшего типа (можно улучшить)
        best_types = [ImageType.NATURAL]  # Заглушка

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
