# utils/benchmark_loader.py

"""Загрузка профилей методов из результатов бенчмарков.

Объединяет метрики производительности (из CSV) и качества (из JSON)
в структурированные профили для использования в AutoSegmenter и других
модулях авто-выбора методов.

Ключевые особенности:
- ✅ Автоматическая агрегация: время + качество → единый профиль
- ✅ Нормализация единиц: секунды → миллисекунды для удобства
- ✅ Fallback-значения: безопасные дефолты при отсутствии данных
- ✅ Типизированный возврат: Dict[str, MethodProfile] для статической проверки
- ✅ Гибкая библиотека: поле `library` настраивается при вызове

Типичный workflow:
```python
from utils.benchmark_loader import load_profiles_from_benchmark
from segmenters.AutoSegmenter import AutoSegmenter

# 1. Загрузка профилей
profiles = load_profiles_from_benchmark(
    benchmark_csv_path="./results/benchmark.csv",
    validation_json_path="./results/validation.json",
    library="opencv"
)

# 2. Фильтрация по требованиям
fast_accurate = {
    name: p for name, p in profiles.items()
    if p.avg_time_ms < 50 and p.avg_iou > 0.8
}

# 3. Регистрация в AutoSegmenter
auto = AutoSegmenter()
for profile in fast_accurate.values():
    auto.register_profile(profile)

# 4. Авто-выбор метода под задачу
best = auto.select_best_method(
    image_type=ImageType.MEDICAL,
    max_time_ms=100,
    min_iou=0.85
)
print(f"Recommended: {best}")
```

Args:
    benchmark_csv_path: Путь к CSV-файлу с колонками:
        - "Method": имя метода (ключ для объединения)
        - "Mean_Time_s": среднее время выполнения в секундах
        - (опционально) "Std_Time_s", "Memory_MB", "Device"
    validation_json_path: Путь к JSON-файлу формата:
        ```json
        {
            "method_name": {"iou": 0.85, "dice": 0.92, ...},
            ...
        }
        ```
    library: Название библиотеки для поля `MethodProfile.library`
        (по умолчанию "opencv"; варианты: "sklearn", "torch", "mixed")

Returns:
    Dict[str, MethodProfile]: Словарь профилей, где ключ — имя метода,
    а значение — объект `MethodProfile` с агрегированными метриками.

Note:
    - При отсутствии метрики качества в JSON используется `avg_iou=0.75` (fallback).
    - Поля `best_for_type`, `robustness`, `parameter_sensitivity` пока заполняются
        заглушками; для продакшена рекомендуется реализовать их расчёт.
    - Функция не кэширует результаты; для повторных вызовов используйте `@lru_cache`.
    - Ошибки чтения файлов пробрасываются как `FileNotFoundError`/`json.JSONDecodeError`.

Raises:
    FileNotFoundError: Если один из входных файлов не найден.
    json.JSONDecodeError: Если JSON-файл имеет невалидный формат.
    KeyError: Если в CSV отсутствует колонка "Method" или "Mean_Time_s".
"""

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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ──────────────────────────────────────────────────────────────────────
def load_profiles_from_benchmark(
    benchmark_csv_path: str, validation_json_path: str, library: str = "opencv"
) -> Dict[str, MethodProfile]:
    """Загрузка профилей методов из результатов ваших бенчмарков."""
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
