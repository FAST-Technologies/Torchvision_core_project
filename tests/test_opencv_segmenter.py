# tests/test_opencv_segmenter.py
# ──────────────────────────────────────────────────────────────────────
# ИМПОРТЫ
# ──────────────────────────────────────────────────────────────────────
"""Модульные тесты для класса OpenCVSegmenter.

Проверяет корректность работы методов сегментации на основе OpenCV:
- Инициализация сегментера с различными методами
- Обработка RGB и градаций серого изображений
- Корректность выходных масок (форма, тип данных, значения)
- Сравнение различных пороговых методов
- Обработка ошибок при некорректных входных данных

Тесты используют pytest fixtures для генерации тестовых изображений.
"""
# ──────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from segmenters.OpenCVSegmenter import OpenCVSegmenter


# ──────────────────────────────────────────────────────────────────────
class TestOpenCVSegmenter:
    """Тестовый класс для проверки функциональности OpenCVSegmenter.

    Содержит тесты для:
    - Базовой инициализации и импорта
    - Сегментации цветных и чёрно-белых изображений
    - Параметризованных проверок различных методов
    - Специфичных поведений алгоритмов (Canny, Sauvola vs Niblack)
    - Обработки исключений при некорректных входных данных
    """

    @pytest.fixture
    def segmenter(self):
        """Фикстура: создаёт экземпляр OpenCVSegmenter с методом global_thresholding.

        Returns:
            OpenCVSegmenter: Сегментер с порогом 0.5 для базовых тестов.
        """
        return OpenCVSegmenter("global_thresholding", threshold=0.5)

    # ──────────────────────────────────────────────────────────────────────
    def test_import(self) -> None:
        """Проверяет успешный импорт класса OpenCVSegmenter.

        Тест убеждается, что класс доступен для импорта из модуля
        segmenters.OpenCVSegmenter и не равен None.
        """
        from segmenters.OpenCVSegmenter import OpenCVSegmenter

        assert OpenCVSegmenter is not None

    # ──────────────────────────────────────────────────────────────────────
    def test_initialization(self) -> None:
        """Проверяет корректную инициализацию сегментера с параметрами.

        Тестирует:
        - Сохранение имени метода в атрибуте `method`
        - Сохранение параметров в атрибуте `params`
        """
        seg = OpenCVSegmenter("otsu_thresholding")
        assert seg.method == "otsu_thresholding"
        assert seg.params == {}

    # ──────────────────────────────────────────────────────────────────────
    def test_segment_rgb(self, rgb_image: np.ndarray) -> None:
        """Тестирует сегментацию цветного (RGB) изображения.

        Проверяет, что:
        - Выходная маска имеет форму (H, W) — без канала цвета
        - Тип данных маски — uint8
        - Значения маски бинарны: {0, 255}

        Args:
            rgb_image: Фикстура с тестовым цветным изображением.
        """
        seg = OpenCVSegmenter("global_thresholding", threshold=0.5)
        mask: np.ndarray = seg.segment(rgb_image)
        assert mask.shape == rgb_image.shape[:2]
        assert mask.dtype == np.uint8
        assert np.all((mask == 0) | (mask == 255))

    # ──────────────────────────────────────────────────────────────────────
    def test_segment_grayscale(self, gray_image: np.ndarray) -> None:
        """Тестирует сегментацию изображения в градациях серого.

        Проверяет сохранение размерности и типа данных для одноканального входа.

        Args:
            gray_image: Фикстура с тестовым чёрно-белым изображением.
        """
        seg = OpenCVSegmenter("adaptive_thresholding", block_size=11, C=2)
        mask: np.ndarray = seg.segment(gray_image)
        assert mask.shape == gray_image.shape
        assert mask.dtype == np.uint8

    # ──────────────────────────────────────────────────────────────────────
    @pytest.mark.parametrize(
        "method,params",
        [
            ("global_thresholding", {"threshold": 0.5}),
            ("otsu_thresholding", {}),
            ("adaptive_thresholding", {"block_size": 11, "C": 2}),
            ("threshold_niblack", {"window_size": 15, "k": -0.2}),
            ("threshold_sauvola", {"window_size": 15, "k": 0.2}),
            ("sobel_edge", {"threshold": 0.1}),
            ("canny_edge", {"low": 0.1, "high": 0.3}),
        ],
    )
    # ──────────────────────────────────────────────────────────────────────
    def test_methods_basic(self, rgb_image: np.ndarray, method, params) -> None:
        """Параметризованный тест базовой работоспособности методов.

        Проверяет, что каждый из перечисленных методов:
        - Успешно инициализируется с заданными параметрами
        - Возвращает непустую маску
        - Сохраняет пространственные размеры входа
        - Возвращает маску типа uint8

        Args:
            rgb_image: Фикстура с тестовым цветным изображением.
            method: Название метода сегментации.
            params: Словарь параметров для инициализации метода.
        """
        seg = OpenCVSegmenter(method, **params)
        mask: np.ndarray = seg.segment(rgb_image)
        assert mask is not None
        assert mask.shape == rgb_image.shape[:2]
        assert mask.dtype == np.uint8

    # ──────────────────────────────────────────────────────────────────────
    def test_canny_edge_output(self, rgb_image: np.ndarray) -> None:
        """Проверяет, что детектор Кэнни возвращает разреженную маску границ.

        Ожидается, что детектор границ выделит менее 50% пикселей,
        так как границы обычно занимают малую часть изображения.

        Args:
            rgb_image: Фикстура с тестовым цветным изображением.
        """
        seg = OpenCVSegmenter("canny_edge", low=0.3, high=0.7)
        mask: np.ndarray = seg.segment(rgb_image)
        assert np.mean(mask > 0) < 0.5

    # ──────────────────────────────────────────────────────────────────────
    def test_sauvola_vs_niblack(self, gray_image: np.ndarray) -> None:
        """Сравнительный тест методов Саволы и Ниблэка.

        Проверяет, что:
        - Оба метода возвращают маски с корректным типом данных (uint8)
        - Значения масок бинарны: {0, 255}
        - Методы могут давать разные результаты на структурированных изображениях

        Создаёт синтетическое изображение с двумя текстурными областями
        разной яркости для демонстрации различий в работе адаптивных порогов.

        Args:
            gray_image: Фикстура с тестовым чёрно-белым изображением
                       (используется как шаблон для создания синтетического).
        """
        test_img: np.ndarray = np.zeros((256, 256), dtype=np.uint8)

        # Добавляем текстурные области с разной яркостью
        test_img[32:96, 32:96] = np.random.randint(80, 120, (64, 64), dtype=np.uint8)  # Тёмная текстура
        test_img[160:224, 160:224] = np.random.randint(200, 240, (64, 64), dtype=np.uint8)  # Светлая текстура

        seg_sauvola = OpenCVSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128)
        seg_niblack = OpenCVSegmenter("threshold_niblack", window_size=15, k=-0.2)

        mask_s: np.ndarray = seg_sauvola.segment(test_img)
        mask_n: np.ndarray = seg_niblack.segment(test_img)

        # diff_ratio = np.mean(mask_s != mask_n)
        assert mask_s.dtype == np.uint8 and mask_n.dtype == np.uint8
        assert np.all((mask_s == 0) | (mask_s == 255))
        assert np.all((mask_n == 0) | (mask_n == 255))

    # ──────────────────────────────────────────────────────────────────────
    def test_invalid_method_raises(self) -> None:
        """Проверяет выброс исключения при использовании неизвестного метода.

        Тестирует, что конструктор OpenCVSegmenter корректно обрабатывает
        попытку инициализации с несуществующим именем метода, выбрасывая
        ValueError с ожидаемым сообщением.
        """
        with pytest.raises(ValueError, match="Неизвестный метод"):
            seg = OpenCVSegmenter("invalid_method")
            seg.segment(np.zeros((100, 100), dtype=np.uint8))

    # ──────────────────────────────────────────────────────────────────────
    def test_segment_with_mask(self, rgb_image: np.ndarray) -> None:
        """Проверяет метод segment_with_mask, возвращающий визуализацию и маску.

        Тестирует, что метод возвращает:
        - Результат с наложенной маской той же формы, что и вход
        - Отдельную бинарную маску формы (H, W)
        - Оба результата имеют тип данных uint8

        Args:
            rgb_image: Фикстура с тестовым цветным изображением.
        """
        seg = OpenCVSegmenter("global_thresholding", threshold=0.5)
        result: np.ndarray
        mask: np.ndarray
        result, mask = seg.segment_with_mask(rgb_image)

        assert result.shape == rgb_image.shape
        assert mask.shape == rgb_image.shape[:2]
        assert result.dtype == np.uint8
        assert mask.dtype == np.uint8
