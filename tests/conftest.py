# tests/conftest.py

# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import os
import torch


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Директория с тестовыми данными"""
    return Path(__file__).parent / "test_data"


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def rgb_image() -> np.ndarray:
    """RGB изображение 256x256"""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def gray_image() -> np.ndarray:
    """Grayscale изображение 256x256"""
    return np.random.randint(0, 255, (256, 256), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def textured_gray_image() -> np.ndarray:
    """Grayscale изображение с текстурными областями для тестов адаптивных порогов"""
    img: np.ndarray = np.zeros((256, 256), dtype=np.uint8)
    # Тёмная текстура
    img[32:96, 32:96] = np.random.randint(80, 120, (64, 64), dtype=np.uint8)
    # Светлая текстура
    img[160:224, 160:224] = np.random.randint(200, 240, (64, 64), dtype=np.uint8)
    # Градиент для проверки адаптивности
    img[:, 128:] = np.tile(np.linspace(0, 255, 128), (256, 1))
    return img


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def binary_mask() -> np.ndarray:
    """Бинарная маска для тестов"""
    mask: np.ndarray = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 255  # Квадрат в центре
    return mask


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def temp_image_file(tmp_path: Path, rgb_image: np.ndarray) -> str:
    """Временный файл изображения для тестов"""
    img: Image.Image = Image.fromarray(rgb_image)
    path: Path = tmp_path / "test_image.jpg"
    img.save(path)
    return str(path)


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def temp_mask_file(tmp_path: Path, binary_mask: np.ndarray) -> str:
    """Временный файл маски"""
    mask: Image.Image = Image.fromarray(binary_mask)
    path: Path = tmp_path / "test_mask.png"
    mask.save(path)
    return str(path)


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def small_image() -> np.ndarray:
    """Маленькое изображение для быстрых тестов"""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def large_image() -> np.ndarray:
    """Большое изображение для тестов производительности"""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────
def pytest_configure(config) -> None:
    """Регистрация маркеров"""
    config.addinivalue_line("markers", "gpu: requires CUDA hardware")


# ──────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def skip_if_no_gpu(request) -> None:
    """
    Автоматически пропускает тесты с маркером @pytest.mark.gpu,
    если в среде нет CUDA или не задана переменная FORCE_GPU_TEST.
    """
    if request.node.get_closest_marker("gpu"):
        # Проверяем: есть ли реальное железо ИЛИ мы форсируем тест в CI
        has_hardware: bool = torch.cuda.is_available()
        force_run: bool = os.getenv("FORCE_GPU_TEST", "false").lower() == "true"
        if not has_hardware and not force_run:
            pytest.skip("CUDA hardware not available (skipping GPU test)")
