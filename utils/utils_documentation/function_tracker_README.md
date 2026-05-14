# Function Tracker Module

Модуль для трассировки вызовов функций в фреймворке сегментации изображений.

## 🎯 Назначение

- **Отладка**: Отслеживание потока выполнения и аргументов функций
- **Мониторинг**: Логирование успешных завершений и исключений
- **Аудит**: Запись истории вызовов для анализа производительности
- **Безопасность**: Изоляция ошибок с сохранением стек-трейса

## 📦 Установка

Модуль не требует внешних зависимостей — использует стандартную библиотеку `logging` и `functools`.

```bash
# Просто импортируйте в ваш проект
from utils.function_tracker import track_calls
```

## 🚀 Быстрый старт

### Базовое использование

```python
from utils.function_tracker import track_calls

@track_calls
def segment_image(path: str, threshold: float = 0.5) -> np.ndarray:
    """Сегментация изображения."""
    # Ваш код...
    return mask
```

**Вывод в лог:**
```
2024-01-15 10:30:45,123 - function_calls - INFO - 🔹 Called: __main__.segment_image
2024-01-15 10:30:45,456 - function_calls - INFO - ✅ Returned from segment_image
```

### Подробное логирование

```python
from utils.function_tracker import track_calls_verbose

@track_calls_verbose
def complex_pipeline(data: list, config: dict) -> dict:
    """Сложная функция с множеством параметров."""
    return {"result": "ok"}
```

**Вывод:**
```
2024-01-15 10:30:45,123 - function_calls - INFO - 🔹 Called: __main__.complex_pipeline(data=<list>, config=dict)
2024-01-15 10:30:45,789 - function_calls - INFO - ✅ Returned from complex_pipeline: <dict>
```

### Обработка исключений

```python
@track_calls
def risky_operation(x: int) -> float:
    if x == 0:
        raise ValueError("Division by zero")
    return 100 / x

try:
    risky_operation(0)
except ValueError:
    pass  # Исключение залогировано с exc_info=True
```

**Вывод:**
```
2024-01-15 10:30:45,123 - function_calls - INFO - 🔹 Called: __main__.risky_operation
2024-01-15 10:30:45,125 - function_calls - ERROR - ❌ Error in risky_operation: ValueError: Division by zero
Traceback (most recent call last):
  File "...", line ..., in risky_operation
    ...
ValueError: Division by zero
```

## ⚙️ Конфигурация логгера

### Уровень логирования

```python
import logging

# Показать аргументы функций (только для track_calls)
logging.getLogger("function_calls").setLevel(logging.DEBUG)

# Скрыть все логи трекера
logging.getLogger("function_calls").setLevel(logging.WARNING)
```

### Формат вывода

```python
import logging
from utils.function_tracker import logger

# Кастомный формат
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "[%(levelname)s] %(name)s: %(message)s"
))
logger.handlers = [handler]
logger.setLevel(logging.INFO)
```

### Вывод в файл

```python
import logging
from utils.function_tracker import logger

file_handler = logging.FileHandler("function_calls.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))
logger.addHandler(file_handler)
```

## 🔧 Интеграция с фреймворком

### В main.py

```python
# Условное включение через environment variable
if os.getenv("TRACK_FUNCTION_CALLS") == "1":
    from utils.function_tracker import track_calls
else:
    def track_calls(func):
        """Заглушка-декоратор без логирования."""
        return func

@track_calls
def heavy_computation(data: np.ndarray) -> np.ndarray:
    """Функция, которую нужно отслеживать только при отладке."""
    ...
```

### В сегментерах

```python
# segmenters/BaseSegmenter.py
from utils.function_tracker import track_calls

class BaseSegmenter:
    @track_calls
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Базовый метод сегментации с трассировкой."""
        ...
```

## 📊 Сравнение декораторов

| Декоратор | Аргументы | Результат | Уровень | Производительность |
|-----------|-----------|-----------|---------|-------------------|
| `track_calls` | Только DEBUG | Нет | INFO | ⭐⭐⭐⭐⭐ |
| `track_calls_verbose` | Первые 3 + типы | Тип результата | INFO | ⭐⭐⭐ |

### Когда использовать какой:

- **`track_calls`**: Продакшен, мониторинг, базовая отладка
- **`track_calls_verbose`**: Разработка, отладка сложных функций, тестирование

## ⚠️ Предостережения

### 1. Большие аргументы

```python
# ❌ Не используйте track_calls_verbose для больших данных:
@track_calls_verbose
def process_huge_array(data: np.ndarray) -> np.ndarray:  # data может быть 1ГБ+
    ...

# ✅ Используйте track_calls или кастомную логику:
@track_calls
def process_huge_array(data: np.ndarray) -> np.ndarray:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Input shape: {data.shape}, dtype: {data.dtype}")
    ...
```

### 2. Рекурсивные функции

```python
# ⚠️ Может создать огромный лог:
@track_calls
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# ✅ Решение: логировать только верхний уровень
def fibonacci(n: int, _depth: int = 0) -> int:
    if _depth == 0 and logger.isEnabledFor(logging.INFO):
        logger.info(f"🔹 fibonacci({n}) called")
    if n <= 1:
        return n
    result = fibonacci(n-1, _depth+1) + fibonacci(n-2, _depth+1)
    if _depth == 0 and logger.isEnabledFor(logging.INFO):
        logger.info(f"✅ fibonacci({n}) = {result}")
    return result
```

### 3. Многопоточность

Логгер потокобезопасен, но порядок записей может не совпадать с порядком вызовов:

```python
from concurrent.futures import ThreadPoolExecutor

@track_calls
def worker(x: int) -> int:
    return x * 2

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(worker, range(10)))
# Логи могут быть перемешаны — используйте correlation_id при необходимости
```

## 🔍 Расширенное использование

### Кастомный декоратор с метриками

```python
import time
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

def track_with_timing(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор с замером времени выполнения."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info(
                f"⏱️  {func.__qualname__}: {elapsed*1000:.2f}ms"
            )
    return wrapper
```

### Фильтрация по модулю

```python
def track_if_debug(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор, активный только при уровне DEBUG."""
    if logger.level > logging.DEBUG:
        return func  # Не добавляем оверхед
    
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.debug(f"🔹 {func.__qualname__}({args}, {kwargs})")
        return func(*args, **kwargs)
    return wrapper
```

### Контекстный менеджер для группировки логов

```python
from contextlib import contextmanager

@contextmanager
def log_block(name: str, **kwargs):
    """Контекстный менеджер для логирования блока кода."""
    logger.info(f"🔷 START: {name} {kwargs}")
    try:
        yield
    except Exception as e:
        logger.error(f"❌ FAIL: {name} — {e}", exc_info=True)
        raise
    finally:
        logger.info(f"🔷 END: {name}")

# Использование:
with log_block("Preprocessing", image_size=img.shape):
    processed = preprocess(img)
```

## 🧪 Тестирование

```python
import unittest
from unittest.mock import patch
from utils.function_tracker import track_calls

class TestFunctionTracker(unittest.TestCase):
    
    @track_calls
    def _test_func(self, x: int) -> int:
        return x * 2
    
    @patch("utils.function_tracker.logger")
    def test_logs_call(self, mock_logger):
        result = self._test_func(5)
        self.assertEqual(result, 10)
        mock_logger.info.assert_any_call("🔹 Called: ..._test_func")
        mock_logger.info.assert_any_call("✅ Returned from _test_func")
    
    @patch("utils.function_tracker.logger")
    def test_logs_exception(self, mock_logger):
        @track_calls
        def failing():
            raise RuntimeError("test")
        
        with self.assertRaises(RuntimeError):
            failing()
        mock_logger.error.assert_called()
```

## 🗂️ Структура модуля

```
utils/
├── function_tracker.py    # Этот модуль
├── __init__.py           # Экспорт: from .function_tracker import ...
└── ...                   # Другие утилиты
```

## 📄 Лицензия

MIT License — свободно используйте в коммерческих и открытых проектах.

## 🤝 Вклад в развитие

1. Fork репозитория
2. Создайте ветку (`git checkout -b feature/logging-improvements`)
3. Внесите изменения + добавьте тесты
4. Запустите `mypy utils/function_tracker.py` для проверки типов
5. Создайте Pull Request

## 📞 Поддержка

- 🐛 Баги: создайте Issue с тегом `bug`
- 💡 Идеи: создайте Issue с тегом `enhancement`
- ❓ Вопросы: используйте Discussions или тег `question`

---

*Документация актуальна для версии 1.0.0*
```

> 💡 **Pro Tip**: Для продакшена используйте условный импорт в `main.py`:
> ```python
> if os.getenv("TRACK_FUNCTION_CALLS") == "1":
>     from utils.function_tracker import track_calls
> else:
>     def track_calls(f): return f  # Zero-overhead stub
> ```
> Это полностью устраняет накладные расходы декоратора, когда трассировка не нужна.
---