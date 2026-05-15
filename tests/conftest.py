# tests/conftest.py
# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
"""Глобальные фикстуры и конфигурация pytest для тестового фреймворка.

Этот модуль содержит общие фикстуры, которые используются во всех тестах:
- Фикстуры для создания тестовых изображений различных размеров и типов
- Фикстуры для временных файлов (изображения, маски)
- Конфигурация pytest (маркеры, хуки)
- Автоматический пропуск тестов при отсутствии GPU

Пример использования:
    ```python
    def test_segmenter(rgb_image):
        # rgb_image автоматически предоставляется фикстурой
        result = segmenter.segment(rgb_image)
        assert result.shape == rgb_image.shape[:2]
    ```

Attributes:
    None

Note:
    Все фикстуры используют `scope="session"` или `scope="function"`
    в зависимости от необходимости переиспользования.
"""
# ──────────────────────────────────────────────────────────────────────
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import os
import torch
from typing import Any


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Возвращает путь к директории с тестовыми данными.

    Фикстура с областью видимости `session` — создаётся один раз
    для всей сессии тестирования и переиспользуется во всех тестах.

    Returns:
        Path: Объект пути к директории `test_data` относительно
        текущего файла конфигурации.

    Example:
        ```python
        def test_load_data(test_data_dir):
            data_file = test_data_dir / "sample.png"
            assert data_file.exists()
        ```
    """
    return Path(__file__).parent / "test_data"


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def rgb_image() -> np.ndarray:
    """Генерирует случайное тестовое изображение в формате RGB.

    Создаёт изображение размером 256×256 пикселей с тремя цветовыми
    каналами (Red, Green, Blue). Значения пикселей равномерно
    распределены в диапазоне [0, 255] и имеют тип `uint8`.

    Returns:
        np.ndarray: Массив формы (256, 256, 3), dtype=uint8,
        содержащий случайные значения интенсивности.

    Note:
        Фикстура имеет область видимости `function` — новое изображение
        создаётся для каждого теста, что обеспечивает изоляцию тестов.

    Example:
        ```python
        def test_rgb_processing(rgb_image):
            assert rgb_image.shape == (256, 256, 3)
            assert rgb_image.dtype == np.uint8
            assert rgb_image.min() >= 0 and rgb_image.max() <= 255
        ```
    """
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def gray_image() -> np.ndarray:
    """Генерирует случайное тестовое изображение в градациях серого.

    Создаёт одноканальное изображение размером 256×256 пикселей.
    Значения пикселей равномерно распределены в диапазоне [0, 255]
    и имеют тип `uint8`.

    Returns:
        np.ndarray: Массив формы (256, 256), dtype=uint8,
        содержащий случайные значения яркости.

    Note:
        Подходит для тестирования методов, работающих с
        одноканальными изображениями (пороговые методы, градиенты).

    Example:
        ```python
        def test_gray_thresholding(gray_image):
            mask = segmenter.segment(gray_image)
            assert mask.shape == gray_image.shape
        ```
    """
    return np.random.randint(0, 255, (256, 256), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def textured_gray_image() -> np.ndarray:
    """Генерирует тестовое изображение с текстурными областями.

    Создаёт сложное изображение 256×256 в градациях серого, содержащее:
    1. Тёмную текстурированную область (80–120) в верхнем левом квадранте
    2. Светлую текстурированную область (200–240) в нижнем правом квадранте
    3. Горизонтальный градиент (0→255) в правой половине изображения

    Это изображение предназначено для тестирования адаптивных методов
    пороговой обработки, которые должны корректно работать при
    неравномерном освещении и локальных перепадах контраста.

    Returns:
        np.ndarray: Массив формы (256, 256), dtype=uint8,
        содержащий структурированные текстурные паттерны.

    Example:
        ```python
        def test_adaptive_thresholding(textured_gray_image):
            # Адаптивный метод должен корректно сегментировать
            # области с разным локальным освещением
            mask = segmenter.segment(textured_gray_image)
            assert mask.shape == textured_gray_image.shape
        ```
    """
    img: np.ndarray = np.zeros((256, 256), dtype=np.uint8)
    # Тёмная текстура (верхний левый квадрант)
    img[32:96, 32:96] = np.random.randint(80, 120, (64, 64), dtype=np.uint8)
    # Светлая текстура (нижний правый квадрант)
    img[160:224, 160:224] = np.random.randint(200, 240, (64, 64), dtype=np.uint8)
    # Горизонтальный градиент для проверки адаптивности
    img[:, 128:] = np.tile(np.linspace(0, 255, 128), (256, 1))
    return img


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def binary_mask() -> np.ndarray:
    """Генерирует эталонную бинарную маску для тестов.

    Создаёт маску размером 256×256, где:
    - Фон: значение 0 (чёрный)
    - Объект: значение 255 (белый квадрат 128×128 в центре)

    Маска используется для:
    - Проверки точности сегментации (IoU, Dice, Precision/Recall)
    - Визуализации результатов
    - Тестирования методов с ground truth

    Returns:
        np.ndarray: Массив формы (256, 256), dtype=uint8,
        содержащий значения {0, 255}.

    Example:
        ```python
        def test_iou_calculation(binary_mask):
            # Идеальное предсказание должно дать IoU = 1.0
            iou = SegmentationMetrics.calculate_iou(binary_mask, binary_mask)
            assert iou == 1.0
        ```
    """
    mask: np.ndarray = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 255  # Квадрат в центре
    return mask


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def temp_image_file(tmp_path: Path, rgb_image: np.ndarray) -> str:
    """Создаёт временный файл изображения для тестов с файловым вводом.

    Конвертирует предоставленное `rgb_image` в формат JPEG и сохраняет
    во временную директорию, управляемую pytest (`tmp_path`). Файл
    автоматически удаляется после завершения теста.

    Args:
        tmp_path (Path): Фикстура pytest для временных файлов.
        rgb_image (np.ndarray): Исходное изображение для сохранения.

    Returns:
        str: Абсолютный путь к созданному файлу изображения.

    Note:
        - Использует `PIL.Image.save()` для сохранения в формате JPEG
        - Файл автоматически удаляется после завершения теста
        - Подходит для тестирования методов, принимающих путь к файлу

    Example:
        ```python
        def test_segment_from_file(temp_image_file):
            # Сегментер должен уметь загружать изображение по пути
            mask = segmenter.segment(temp_image_file)
            assert mask is not None
        ```
    """
    img: Image.Image = Image.fromarray(rgb_image)
    path: Path = tmp_path / "test_image.jpg"
    img.save(path)
    return str(path)


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def temp_mask_file(tmp_path: Path, binary_mask: np.ndarray) -> str:
    """Создаёт временный файл бинарной маски для тестов.

    Конвертирует предоставленную `binary_mask` в формат PNG и сохраняет
    во временную директорию. Используется для тестирования загрузки
    ground truth масок и сравнения результатов.

    Args:
        tmp_path (Path): Фикстура pytest для временных файлов.
        binary_mask (np.ndarray): Исходная бинарная маска для сохранения.

    Returns:
        str: Абсолютный путь к созданному файлу маски.

    Note:
        - Сохраняет в формате PNG для сохранения бинарных значений без потерь
        - Файл автоматически удаляется после завершения теста

    Example:
        ```python
        def test_load_ground_truth(temp_mask_file):
            gt = load_mask(temp_mask_file)
            assert set(np.unique(gt)).issubset({0, 255})
        ```
    """
    mask: Image.Image = Image.fromarray(binary_mask)
    path: Path = tmp_path / "test_mask.png"
    mask.save(path)
    return str(path)


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def small_image() -> np.ndarray:
    """Генерирует маленькое тестовое изображение для быстрых тестов.

    Создаёт изображение размером 64×64×3 (RGB) для тестов, где
    производительность не критична, но важна скорость выполнения
    (юнит-тесты, проверка логики, отладка).

    Returns:
        np.ndarray: Массив формы (64, 64, 3), dtype=uint8,
        содержащий случайные значения.

    Note:
        - Идеально подходит для тестов, запускаемых десятки/сотни раз
        - Не подходит для тестов производительности или методов,
          чувствительных к размеру изображения

    Example:
        ```python
        def test_method_logic(small_image):
            # Быстрая проверка корректности работы метода
            result = method(small_image)
            assert result.shape == small_image.shape[:2]
        ```
    """
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def large_image() -> np.ndarray:
    """Генерирует большое тестовое изображение для бенчмарков.

    Создаёт изображение размером 512×512×3 (RGB) для тестов:
    - Производительности и масштабирования
    - Потребления памяти
    - Работы с большими данными
    - Триггеров оптимизаций (Numba, torch.compile)

    Returns:
        np.ndarray: Массив формы (512, 512, 3), dtype=uint8,
        содержащий случайные значения.

    Note:
        - Может требовать больше памяти и времени выполнения
        - Рекомендуется использовать с маркером `@pytest.mark.slow`
          для тестов, не запускаемых в CI по умолчанию

    Example:
        ```python
        @pytest.mark.slow
        def test_large_image_performance(large_image):
            start = time.perf_counter()
            result = segmenter.segment(large_image)
            elapsed = time.perf_counter() - start
            assert elapsed < MAX_ALLOWED_TIME
        ```
    """
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────
def pytest_configure(config: Any) -> None:
    """Регистрирует пользовательские маркеры pytest при инициализации.

    Этот хук вызывается pytest при старте и позволяет определить
    кастомные маркеры для фильтрации тестов.

    Зарегистрированные маркеры:
    - `gpu`: Тесты, требующие наличия CUDA-устройства

    Args:
        config: Конфигурационный объект pytest.

    Note:
        Маркеры позволяют запускать только определённые группы тестов:
        ```bash
        # Запустить только тесты для GPU
        pytest -m gpu

        # Пропустить тесты для GPU
        pytest -m "not gpu"
        ```
    """
    config.addinivalue_line("markers", "gpu: requires CUDA hardware")


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def skip_if_no_gpu(request: Any) -> None:
    """Автоматически пропускает тесты с маркером @pytest.mark.gpu при отсутствии CUDA.

    Фикстура с `autouse=True` применяется ко всем тестам автоматически.
    Если тест помечен маркером `@pytest.mark.gpu` и:
    1. В системе нет доступного CUDA-устройства, И
    2. Не задана переменная окружения `FORCE_GPU_TEST=true`

    то тест будет пропущен с соответствующим сообщением.

    Это позволяет:
    - Запускать полный набор тестов на машинах без GPU
    - Принудительно запускать GPU-тесты в CI через переменную окружения

    Args:
        request: Объект запроса pytest, предоставляющий информацию о тесте.

    Raises:
        pytest.skip: Если тест помечен как `gpu`, но условия не выполнены.

    Example:
        ```python
        @pytest.mark.gpu
        def test_cuda_acceleration():
            # Этот тест выполнится только при наличии CUDA
            # или при FORCE_GPU_TEST=true
            assert torch.cuda.is_available()
        ```
    """
    if request.node.get_closest_marker("gpu"):
        # Проверяем: есть ли реальное железо ИЛИ мы форсируем тест в CI
        has_hardware: bool = torch.cuda.is_available()
        force_run: bool = os.getenv("FORCE_GPU_TEST", "false").lower() == "true"
        if not has_hardware and not force_run:
            pytest.skip("CUDA hardware not available (skipping GPU test)")
