# utils/function_tracker.py
import logging
from functools import wraps

logger = logging.getLogger("function_calls")


def track_calls(func):
    """Декоратор для логирования вызовов функции"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"🔹 Called: {func.__module__}.{func.__qualname__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ Returned from {func.__qualname__}")
            return result
        except Exception as e:
            logger.error(f"❌ Error in {func.__qualname__}: {e}")
            raise

    return wrapper
