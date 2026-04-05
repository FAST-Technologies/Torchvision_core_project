# tests/conftest.py
import os
import pytest
import torch

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