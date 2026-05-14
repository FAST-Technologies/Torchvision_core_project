# utils/function_tracker.py

"""Модуль декораторов для трассировки вызовов функций и обработки исключений.

Предоставляет инструменты для отладки, мониторинга и логирования выполнения функций
в рамках фреймворка сегментации изображений. Позволяет отслеживать:
- Вход в функцию с аргументами
- Успешный возврат результата
- Исключения с полным стек-трейсом

Особенности:
- Сохранение оригинальной сигнатуры функции через `functools.wraps`
- Типизация через `ParamSpec` и `TypeVar` для полной совместимости с mypy
- Гибкое управление уровнем детализации логов (INFO/DEBUG)
- Безопасное логирование больших аргументов (обрезка, типизация)

Конфигурация:
- Логгер: "function_calls" (уровень по умолчанию: INFO)
- Формат: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
- Вывод: stdout (можно переопределить через logging.config)

Использование:
```python
from utils.function_tracker import track_calls, track_calls_verbose

@track_calls
def process_image(path: str) -> np.ndarray:
    '''Обработка изображения.'''
    ...

@track_calls_verbose
def debug_heavy_function(data: list) -> dict:
    '''Функция с подробным логированием аргументов/результата.'''
    ...

Environment variables:
    TRACK_FUNCTION_CALLS=1 — включить декорирование в main.py
    LOG_LEVEL=DEBUG — показать аргументы функций в логах

Author: Vladimir Yamshchikov
Version: 1.0.0
"""

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import logging
from functools import wraps
from typing import List, Callable, TypeVar, ParamSpec, cast

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & GENERICS
# ──────────────────────────────────────────────────────────────────────
P = ParamSpec("P")  # Параметры функции
R = TypeVar("R")  # Возвращаемое значение
# FuncT = Callable[P, R]  # Тип произвольной функции

# Настройка логгера
logger: logging.Logger = logging.getLogger("function_calls")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ──────────────────────────────────────────────────────────────────────
def track_calls(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор для логирования вызовов функции и обработки исключений.

    Логирует на уровне INFO:
    - Вход в функцию: "🔹 Called: module.func_name"
    - Успешный возврат: "✅ Returned from func_name"
    - Исключения: "❌ Error in func_name: ExceptionType: message" + стек-трейс

    На уровне DEBUG дополнительно логирует:
    - Все позиционные и именованные аргументы (repr)

    Особенности:
    - Сохраняет метаданные функции (__name__, __doc__, __annotations__) через @wraps
    - Полная типизация через ParamSpec/TypeVar — совместимо с mypy strict mode
    - Не влияет на производительность при отключённом логировании (short-circuit)
    - Потокобезопасен (логгер logging по умолчанию использует Lock)

    Args:
        func: Декорируемая функция произвольной сигнатуры.

    Returns:
        Callable[P, R]: Обёртка с идентичной сигнатурой, добавляющая логирование.

    Raises:
        Любое исключение, выброшенное оригинальной функцией, 
        пробрасывается дальше после логирования.

    Example:
        ```python
        >>> @track_calls
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>> add(2, 3)
        # Лог:
        # 🔹 Called: __main__.add
        # ✅ Returned from add
        5

        >>> @track_calls
        ... def divide(a: float, b: float) -> float:
        ...     return a / b
        >>> divide(1, 0)
        # Лог:
        # 🔹 Called: __main__.divide
        # ❌ Error in divide: ZeroDivisionError: division by zero
        # Traceback (most recent call last): ...
        # ZeroDivisionError: division by zero
        ```

    See Also:
        - track_calls_verbose: для подробного логирования аргументов/результата
        - logging.basicConfig: для настройки формата и уровня логгера
        - functools.wraps: основа сохранения метаданных функции

    Note:
        Для продакшена рекомендуется уровень логгера >= WARNING, 
        чтобы избежать накладных расходов на форматирование строк.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        func_name: str = f"{func.__module__}.{func.__qualname__}"
        logger.info(f"🔹 Called: {func_name}")

        # Логируем аргументы только на уровне DEBUG (чтобы не засорять вывод)
        if logger.isEnabledFor(logging.DEBUG):
            args_repr: List[str] = [repr(a) for a in args]
            kwargs_repr: List[str] = [f"{k}={v!r}" for k, v in kwargs.items()]
            logger.debug(f"   Args: ({', '.join(args_repr + kwargs_repr)})")

        try:
            result: R = func(*args, **kwargs)
            logger.info(f"✅ Returned from {func.__qualname__}")
            return result
        except Exception as e:
            logger.error(
                f"❌ Error in {func.__qualname__}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    return cast(Callable[P, R], wrapper)  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────
def track_calls_verbose(func: Callable[P, R]) -> Callable[P, R]:
    """Расширенная версия декоратора: логирует аргументы и результат на уровне INFO.

    ⚠️ Предупреждение: может значительно увеличить объём логов и снизить 
    производительность для функций с большими аргументами или частыми вызовами.

    Логирует на уровне INFO:
    - Вход: "🔹 Called: module.func_name(arg1_repr, arg2_type=type, ...)"
      • Позиционные аргументы: первые 3 через repr(), остальные — "... +N more"
      • Именованные аргументы: только тип значения (f"{k}={type(v).__name__}")
    - Возврат: "✅ Returned from func_name: <result_repr_or_type>"
      • Примитивы (int/float/str/bool): полный repr
      • Сложные типы: "<TypeName>" для компактности

    Отличия от track_calls:
    | Аспект | track_calls | track_calls_verbose |
    |--------|-------------|-------------------|
    | Аргументы | Только DEBUG | INFO (с обрезкой) |
    | Результат | Не логируется | INFO (с типизацией) |
    | Производительность | Высокая | Средняя |
    | Использование | Продакшен | Отладка/тесты |

    Args:
        func: Декорируемая функция произвольной сигнатуры.

    Returns:
        Callable[P, R]: Обёртка с подробным логированием.

    Example:
        ```python
        >>> @track_calls_verbose
        ... def process(data: list, config: dict, threshold: float) -> dict:
        ...     return {"status": "ok"}
        
        >>> process([1,2,3], {"a": 1}, 0.5)
        # Лог:
        # 🔹 Called: __main__.process([1, 2, 3], <dict>, threshold=float)
        # ✅ Returned from process: {'status': 'ok'}
        ```

    Best Practices:
        1. Используйте только для функций с небольшим входом/выходом
        2. Отключайте в продакшене через logging.setLevel(WARNING)
        3. Для больших данных логируйте только метаданные:
           ```python
           if logger.isEnabledFor(logging.INFO):
               logger.info(f"Input: shape={data.shape}, dtype={data.dtype}")
           ```

    See Also:
        - track_calls: лёгкая версия для продакшена
        - logging.Logger.isEnabledFor: проверка уровня перед форматированием
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        func_name: str = f"{func.__module__}.{func.__qualname__}"
        args_repr: List[str] = [
            repr(a) for a in args[:3]
        ]  # Логируем только первые 3 аргумента
        if len(args) > 3:
            args_repr.append(f"... +{len(args) - 3} more")
        kwargs_repr: List[str] = [f"{k}={type(v).__name__}" for k, v in kwargs.items()]

        logger.info(f"🔹 Called: {func_name}({', '.join(args_repr + kwargs_repr)})")

        try:
            result: R = func(*args, **kwargs)
            result_type: str = type(result).__name__
            result_repr: str = (
                repr(result)
                if isinstance(result, (int, float, str, bool))
                else f"<{result_type}>"
            )
            logger.info(f"✅ Returned from {func.__qualname__}: {result_repr}")
            return result
        except Exception as e:
            logger.error(
                f"❌ Error in {func.__qualname__}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    return cast(Callable[P, R], wrapper)  # type: ignore[return-value]
