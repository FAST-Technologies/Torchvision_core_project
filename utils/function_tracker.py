# utils/function_tracker.py

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
    """
    Декоратор для логирования вызовов функции и обработки исключений.

    Логирует:
    - Момент входа в функцию с аргументами (если уровень DEBUG).
    - Момент успешного возврата.
    - Исключения с трассировкой стека.

    Args:
        func: Декорируемая функция.

    Returns:
        Callable[P, R]: Обёртка с логированием, сохраняющая сигнатуру оригинала.

    Example:
        ```python
        @track_calls
        def process_image(path: str) -> np.ndarray:
            ...
        ```
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
    """
    Расширенная версия декоратора: логирует аргументы и результат даже на INFO-уровне.

    ⚠️ Использовать с осторожностью для функций с большими аргументами/результатами.

    Args:
        func: Декорируемая функция.

    Returns:
        Callable[P, R]: Обёртка с подробным логированием.
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
