# tests/conftest.py
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import os
import torch


@pytest.fixture(scope="session")
def test_data_dir():
    """Директория с тестовыми данными"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def rgb_image():
    """RGB изображение 256x256"""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def gray_image():
    """Grayscale изображение 256x256"""
    return np.random.randint(0, 255, (256, 256), dtype=np.uint8)


@pytest.fixture
def binary_mask():
    """Бинарная маска для тестов"""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 255  # Квадрат в центре
    return mask


@pytest.fixture
def temp_image_file(tmp_path, rgb_image):
    """Временный файл изображения для тестов"""
    img = Image.fromarray(rgb_image)
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return str(path)


@pytest.fixture
def temp_mask_file(tmp_path, binary_mask):
    """Временный файл маски"""
    mask = Image.fromarray(binary_mask)
    path = tmp_path / "test_mask.png"
    mask.save(path)
    return str(path)


@pytest.fixture
def small_image():
    """Маленькое изображение для быстрых тестов"""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def large_image():
    """Большое изображение для тестов производительности"""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


def pytest_configure(config):
    """Регистрация маркеров"""
    config.addinivalue_line("markers", "gpu: requires CUDA hardware")


@pytest.fixture(autouse=True)
def skip_if_no_gpu(request):
    """
    Автоматически пропускает тесты с маркером @pytest.mark.gpu,
    если в среде нет CUDA или не задана переменная FORCE_GPU_TEST.
    """
    if request.node.get_closest_marker("gpu"):
        # Проверяем: есть ли реальное железо ИЛИ мы форсируем тест в CI
        has_hardware = torch.cuda.is_available()
        force_run = os.getenv("FORCE_GPU_TEST", "false").lower() == "true"
        if not has_hardware and not force_run:
            pytest.skip("CUDA hardware not available (skipping GPU test)")
