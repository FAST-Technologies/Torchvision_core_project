# tests/test_base_segmenter.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from PIL import Image
from segmenters.BaseSegmenter import BaseSegmenter


class DummySegmenter(BaseSegmenter):
    def segment(self, image, **kwargs):
        # Возвращаем пустую маску того же размера что и вход
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = 256, 256
        return np.zeros((h, w), dtype=np.uint8)

    def segment_with_mask(self, image, **kwargs):
        # Возвращаем кортеж (результат, маска)
        mask = self.segment(image, **kwargs)
        # Для визуализации просто копируем изображение
        if isinstance(image, np.ndarray):
            result = (
                image.copy()
                if len(image.shape) == 3
                else np.stack([image] * 3, axis=-1)
            )
        else:
            result = np.zeros((256, 256, 3), dtype=np.uint8)
        return result, mask


class TestBaseSegmenter:
    def test_import(self):
        from segmenters.BaseSegmenter import BaseSegmenter

        assert BaseSegmenter is not None

    def test_abstract_methods(self):
        """Базовый класс не должен инстанцироваться напрямую"""
        with pytest.raises(TypeError):
            BaseSegmenter()

    def test_preprocess_image_from_path(self, temp_image_file):
        """Тест предобработки из файла"""
        seg = DummySegmenter()
        result = seg.preprocess_image(temp_image_file)
        assert result is not None

    def test_preprocess_image_from_pil(self):
        """Тест предобработки из PIL Image"""
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        seg = DummySegmenter()
        result = seg.preprocess_image(img)
        assert result is not None

    def test_preprocess_image_from_numpy(self):
        """Тест предобработки из numpy array"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        seg = DummySegmenter()
        result = seg.preprocess_image(img)
        assert result is not None

    def test_segment_with_mask_base(self):
        """Тест базового segment_with_mask"""
        seg = DummySegmenter()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result, mask = seg.segment_with_mask(img)
        assert result.shape == img.shape
        assert mask.shape == img.shape[:2]
        assert result.dtype == np.uint8
        assert mask.dtype == np.uint8
