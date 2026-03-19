# TorchSegmenter.py

# Импорт основных библиотек

from segmenters.BaseSegmenter import BaseSegmenter

import warnings
from PIL import Image
import time
from collections import deque
import heapq
import traceback
from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)

import numpy as np
from scipy import ndimage

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import GaussianBlur

import cv2
from sklearn.cluster import MeanShift, KMeans, DBSCAN, MeanShift as SkMeanShift

class TorchSegmenter(BaseSegmenter):
    """
    Класс для методов сегментации с использованием PyTorch. Все реализации сделаны без использования OpenCV, Scikit-learn, Scikit-image или специализированных
    библиотек для обработки изображений. Поддерживает как классические методы (пороговые, граничные), так и методы на основе кластеризации,
    активных контуров и графов.
    """ 
    def __init__(
        self, 
        method: str = "global_thresholding", 
        device: str = None, 
        use_external_libs: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.method: str = method
        self.params: Dict[str, Any] = kwargs
        self.model_name: str = f"Torch_{method}"
        self.use_external_libs = use_external_libs
        self._needs_normalization: bool = method in [
            "global_thresholding",
            "adaptive_thresholding",
            "otsu_thresholding",
            "threshold_niblack",
            "threshold_sauvola",
            "sobel_edge",
            "canny_edge",
            "morphological_snakes",
            "chan_vese",
            "quickshift",
            "slic"
        ]
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("Используется CPU")

        self._setup_method()
    
    def _setup_method(self, **kwargs) -> None:
        """Настройка выбранного метода"""
        self.method_map: Dict[str, torch.Tensor] = {
            # ============ ПОРОГОВЫЕ МЕТОДЫ СЕГМЕНТАЦИИ ============
            "global_thresholding": self._global_thresholding,
            "adaptive_thresholding": self._adaptive_thresholding,
            "otsu_thresholding": self._otsu_thresholding,
            "threshold_niblack": self._threshold_niblack,
            "threshold_sauvola": self._threshold_sauvola,

            # ============ КРАЕВЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "sobel_edge": self._sobel_edge,
            "canny_edge": self._canny_edge,

            # ============ РЕГИОНАЛЬНЫЕ СЕГМЕНТАЦИОННЫЕ МЕТОДЫ ============
            "region_growing": self._region_growing,
            "split_and_merge": self._split_and_merge,
            "floodfill": self._floodfill,

            # ============ КЛАСТЕРИЗАЦИЯ ============
            "kmeans_segmentation": self._kmeans_segmentation,
            "dbscan_segmentation": self._dbscan_segmentation,
            "meanshift": self._meanshift,

            # ============ АКТИВНЫЕ КОНТУРЫ ============
            "active_contour": self._active_contour,
            "gvf_contour": self._gvf_contour,
            "morphological_snakes": self._morphological_snakes,
            "chan_vese": self._chan_vese,

            # ============ WATERSHED И ГРАФОВЫЕ ============
            "watershed": self._watershed,
            "random_walker": self._random_walker,

            # ============ SUPER-PIXEL МЕТОДЫ ===========
            "quickshift": self._quickshift,
            "slic": self._slic,
            "felzenszwalb": self._felzenszwalb,

            # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============
            "grabcut": self._grabcut,
        }
        
        if self.method not in self.method_map:
            raise ValueError(f"Неизвестный метод: {self.method}. "
                           f"Доступные методы: {list(self.method_map.keys())}")
        
        self._segment_func = self.method_map[self.method]
    
    def preprocess_image(
        self, 
        image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        """Предобработка изображения для PyTorch"""
        if isinstance(image, str):
            # Загрузка из файла
            img = Image.open(image).convert('RGB')
            return self._pil_to_tensor(img, normalize=self._needs_normalization)
        elif isinstance(image, Image.Image):
            # PIL Image
            return self._pil_to_tensor(image, normalize=self._needs_normalization)
        elif isinstance(image, np.ndarray):
            # NumPy array
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            img = Image.fromarray(image.astype(np.uint8)).convert('RGB')
            return self._pil_to_tensor(img, normalize=self._needs_normalization)
        elif isinstance(image, torch.Tensor):
            # PyTorch tensor
            return image.to(self.device)
        else:
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")
        
    def conv2d_numpy(self, image: np.ndarray, kernel: np.ndarray, mode: str = 'reflect') -> np.ndarray:
        """2D свёртка на numpy/scipy (эквивалент cv2.filter2D)"""
        return ndimage.convolve(image, kernel, mode=mode)

    def rgb_to_gray_numpy(self, rgb: np.ndarray) -> np.ndarray:
        """Конвертация RGB → Grayscale на numpy"""
        # ITU-R BT.601 weights
        return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]

    def local_mean_numpy(self, image: np.ndarray, window_size: int) -> np.ndarray:
        """Локальное среднее через свёртку"""
        kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)
        return self.conv2d_numpy(image, kernel)

    def local_std_numpy(self, image: np.ndarray, window_size: int) -> np.ndarray:
        """Локальное стандартное отклонение"""
        mean = self.local_mean_numpy(image, window_size)
        mean_sq = self.local_mean_numpy(image ** 2, window_size)
        return np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    def sobel_numpy(self, image: np.ndarray) -> np.ndarray:
        """Оператор Собеля на numpy"""
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        gx = self.conv2d_numpy(image, kernel_x)
        gy = self.conv2d_numpy(image, kernel_y)
        return np.sqrt(gx ** 2 + gy ** 2)
    
    def _pil_to_tensor(
        self, 
        img: Image.Image, 
        normalize: bool = True,
        add_batch: bool = True
    ) -> torch.Tensor:
        """
        Универсальное преобразование PIL -> Tensor
        
        Args:
            img: Входное изображение PIL
            normalize: Нормализовать [0, 255] -> [0, 1]
            add_batch: Добавить batch dimension
            
        Returns:
            torch.Tensor на нужном устройстве
        """
        try:
            tensor: torch.Tensor
            np_img: np.ndarray
            tensor: torch.Tensor
            np_img: np.ndarray
            if normalize:
                tensor = TF.to_tensor(img)  # Автоматически нормализует к [0, 1]
            else:
                np_img = np.array(img).astype(np.float32)
                tensor = torch.from_numpy(np_img).permute(2, 0, 1) / 255.0
            
            if add_batch:
                tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            
            return tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"Ошибка преобразования PIL->Tensor: {e}")
    
    def _tensor_to_numpy(
        self, 
        tensor: torch.Tensor,
        denormalize: bool = True
    ) -> np.ndarray:
        """Преобразование PyTorch tensor в NumPy array"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        result: np.ndarray = tensor.permute(1, 2, 0).cpu().numpy()
        result: np.ndarray = tensor.permute(1, 2, 0).cpu().numpy()
        
        if denormalize and result.max() <= 1.0:
            result = (result * 255).astype(np.uint8)
        
        return result
    
    def _tensor_to_pil(
        self, 
        tensor: torch.Tensor,
        squeeze: bool = True
    ) -> Image.Image:
        """
        Преобразование torch.Tensor в PIL.Image
        
        Args:
            tensor: Тензор в формате (C, H, W) или (B, C, H, W)
            squeeze: Если True, удаляет batch dimension при наличии
        
        Returns:
            PIL.Image
        """
        if tensor.dim() == 4 and squeeze:
            tensor = tensor.squeeze(0)
        elif tensor.dim() == 3:
            pass
        else:
            raise ValueError(f"Неверная размерность тензора: {tensor.shape}")
        
        return TF.to_pil_image(tensor)
    
    def _to_grayscale(
        self, 
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """Преобразование RGB в градации серого"""
        gray: torch.Tensor
        gray: torch.Tensor
        if tensor.shape[1] == 3:
            # gray = torch.mean(tensor, dim=1, keepdim=True) # (B, 1, H, W)
            gray = 0.2989 * tensor[:, 0:1, :, :] + 0.5870 * tensor[:, 1:2, :, :] + 0.1140 * tensor[:, 2:3, :, :]
        else:
            gray = tensor
        return gray
    
    def _normalize_to_255(
        self,
        img: Image.Image | np.ndarray
    ) -> Image.Image | np.ndarray:
        """Метод нормализации изобраажения"""
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        return img
    
    def _normalize_tensor(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """Нормализация тензора к [0, 1]"""
        min_val: torch.Tensor = tensor.min()
        max_val: torch.Tensor = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    def _cv_heavyside(self, x: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        """Регуляризованная функция Хевисайда"""
        return 0.5 * (1.0 + (2.0 / np.pi) * torch.arctan(x / eps))

    def _cv_delta(self, x: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        """Регуляризованная дельта-функция Дирака"""
        return eps / (eps**2 + x**2)

    def _cv_calculate_averages(
        self, 
        image: torch.Tensor, 
        Hphi: torch.Tensor
    ) -> Tuple[float, float]:
        """Вычисление средних значений внутри и снаружи контура"""
        H = Hphi
        Hinv = 1.0 - H
        
        Hsum = H.sum()
        Hinvsum = Hinv.sum()
        
        avg_inside = (image * H).sum()
        avg_outside = (image * Hinv).sum()
        
        if Hsum > 1e-8:
            avg_inside = avg_inside / Hsum
        if Hinvsum > 1e-8:
            avg_outside = avg_outside / Hinvsum
        
        return avg_inside.item(), avg_outside.item()

    def _cv_difference_from_average_term(
        self,
        image: torch.Tensor,
        Hphi: torch.Tensor,
        lambda1: float,
        lambda2: float
    ) -> torch.Tensor:
        """Энергетический член: разница от среднего в регионах"""
        c1, c2 = self._cv_calculate_averages(image, Hphi)
        Hinv = 1.0 - Hphi
        return lambda1 * (image - c1) ** 2 * Hphi + lambda2 * (image - c2) ** 2 * Hinv

    def _cv_edge_length_term(
        self,
        phi: torch.Tensor,
        mu: float,
        eps: float = 1.0
    ) -> torch.Tensor:
        """Энергетический член: длина контура"""
        # Паддинг для вычисления градиентов
        # P = torch.nn.functional.pad(phi, (1, 1, 1, 1), mode='replicate')

        if phi.dim() == 2:
            P = torch.nn.functional.pad(phi.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
        else:
            P = torch.nn.functional.pad(phi, (1, 1, 1, 1), mode='replicate')
        
        # Центральные разности для градиентов
        fy = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0
        fx = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0
        
        return mu * self._cv_delta(phi, eps) * torch.sqrt(fx**2 + fy**2 + 1e-8)

    def _cv_energy(
        self,
        image: torch.Tensor,
        phi: torch.Tensor,
        mu: float,
        lambda1: float,
        lambda2: float
    ) -> torch.Tensor:
        """Полная энергия функционала Чан-Везе"""
        H = self._cv_heavyside(phi)
        avg_energy = self._cv_difference_from_average_term(image, H, lambda1, lambda2)
        len_energy = self._cv_edge_length_term(phi, mu)
        return avg_energy.sum() + len_energy.sum()

    def _cv_calculate_variation(
        self,
        image: torch.Tensor,
        phi: torch.Tensor,
        mu: float,
        lambda1: float,
        lambda2: float,
        dt: float,
        eps: float = 1e-16
    ) -> torch.Tensor:
        """
        Вычисление вариации уровня для одной итерации.
        Соответствует уравнению (22) из статьи Паскаля Гетре.
        """
        eta = 1e-16
        
        # Паддинг для вычисления разностей
        # P = torch.nn.functional.pad(phi, (1, 1, 1, 1), mode='replicate')
        if phi.dim() == 2:
            phi_padded = torch.nn.functional.pad(phi.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
        else:
            phi_padded = torch.nn.functional.pad(phi, (1, 1, 1, 1), mode='replicate')
        P = phi_padded
        
        # Разности по осям
        phixp = P[1:-1, 2:] - P[1:-1, 1:-1]  # phi(x+1) - phi(x)
        phixn = P[1:-1, 1:-1] - P[1:-1, :-2]  # phi(x) - phi(x-1)
        phix0 = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0  # центр. разность по x
        
        phiyp = P[2:, 1:-1] - P[1:-1, 1:-1]  # phi(y+1) - phi(y)
        phiyn = P[1:-1, 1:-1] - P[:-2, 1:-1]  # phi(y) - phi(y-1)
        phiy0 = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0  # центр. разность по y
        
        # Коэффициенты кривизны
        C1 = 1.0 / torch.sqrt(eta + phixp**2 + phiy0**2)
        C2 = 1.0 / torch.sqrt(eta + phixn**2 + phiy0**2)
        C3 = 1.0 / torch.sqrt(eta + phix0**2 + phiyp**2)
        C4 = 1.0 / torch.sqrt(eta + phix0**2 + phiyn**2)
        
        # Член кривизны (дискретный лапласиан с весами)
        K = P[1:-1, 2:] * C1 + P[1:-1, :-2] * C2 + P[2:, 1:-1] * C3 + P[:-2, 1:-1] * C4
        
        # Член разницы от среднего
        Hphi = self._cv_heavyside(phi, eps=1.0)
        c1, c2 = self._cv_calculate_averages(image, Hphi)
        difference_term = -lambda1 * (image - c1) ** 2 + lambda2 * (image - c2) ** 2
        
        # Обновление уровня
        delta_phi = self._cv_delta(phi, eps=1.0)
        numerator = phi + dt * delta_phi * (mu * K + difference_term)
        denominator = 1.0 + mu * dt * delta_phi * (C1 + C2 + C3 + C4)
        
        new_phi = numerator / (denominator + 1e-8)
        return new_phi

    def _cv_init_level_set(
        self,
        init_type: str,
        image_shape: Tuple[int, int],
        device: torch.device
    ) -> torch.Tensor:
        """Инициализация уровня для Chan-Vese"""
        h, w = image_shape
        
        if init_type == 'checkerboard':
            # Шахматная доска (быстрая сходимость)
            square_size = 5
            yv = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
            xv = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
            sf = np.pi / square_size
            return torch.sin(yv * sf) * torch.sin(xv * sf)
        
        elif init_type == 'disk':
            # Большой диск (покрывает всё изображение)
            center_y = (h - 1) // 2
            center_x = (w - 1) // 2
            radius = float(min(center_x, center_y))
            
            y, x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2).float()
            return (radius - dist) / radius
        
        elif init_type == 'small_disk':
            # Маленький диск (половина размера)
            center_y = (h - 1) // 2
            center_x = (w - 1) // 2
            radius = float(min(center_x, center_y)) / 2.0
            
            y, x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2).float()
            return (radius - dist) / (radius * 3)
        
        elif isinstance(init_type, torch.Tensor):
            # Пользовательская инициализация
            return init_type.to(device).float()
        
        else:
            # По умолчанию: константа 0.5
            return torch.ones((h, w), device=device, dtype=torch.float32) * 0.5
        
    # ============================================================================
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ RANDOM WALKER (чистый PyTorch)
    # ============================================================================

    def _rw_create_markers(
        self,
        h: int,
        w: int,
        device: torch.device
    ) -> torch.Tensor:
        """Создание автоматических маркеров для Random Walker"""
        markers = torch.zeros((h, w), dtype=torch.int32, device=device)
        
        # Центральная область - объект (маркер 2)
        markers[h//4:3*h//4, w//4:3*w//4] = 2
        
        # Углы - фон (маркер 1)
        corner_size = min(h, w) // 8
        markers[:corner_size, :corner_size] = 1
        markers[:corner_size, -corner_size:] = 1
        markers[-corner_size:, :corner_size] = 1
        markers[-corner_size:, -corner_size:] = 1
        
        return markers

    def _rw_compute_weights(
        self,
        image: torch.Tensor,
        beta: float
    ) -> torch.Tensor:
        """
        Вычисление весов рёбер графа на основе градиента изображения.
        Вес = exp(-beta * ||grad||^2)
        
        Args:
            image: Изображение (H, W)
            beta: Коэффициент затухания
        
        Returns:
            weights: Тензор весов для 4-связности (4, H, W)
        """
        h, w = image.shape
        
        # Вычисляем градиенты (центральные разности)
        # По горизонтали
        grad_x = torch.zeros_like(image)
        grad_x[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
        grad_x[:, 0] = image[:, 1] - image[:, 0]
        grad_x[:, -1] = image[:, -1] - image[:, -2]
        
        # По вертикали
        grad_y = torch.zeros_like(image)
        grad_y[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
        grad_y[0, :] = image[1, :] - image[0, :]
        grad_y[-1, :] = image[-1, :] - image[-2, :]
        
        # Нормализуем градиент
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Вычисляем веса: чем больше градиент, тем меньше вес (труднее диффузия)
        scale = beta / (10 * (grad_mag.std() + 1e-8))
        weights = torch.exp(-scale * grad_mag**2)
        
        # Возвращаем веса для 4 направлений: вправо, вниз, влево, вверх
        # (для каждого пикселя храним веса к соседям)
        weights_4dir = torch.stack([
            weights,  # вправо
            weights,  # вниз
            weights,  # влево
            weights,  # вверх
        ], dim=0)  # (4, H, W)
        
        return weights_4dir

    def _rw_build_laplacian(
        self,
        image: torch.Tensor,
        weights: torch.Tensor,
        markers: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Построение разреженной матрицы лапласиана графа.
        
        Returns:
            L: Разреженная матрица лапласиана (N, N) в формате COO
            b_indices: Индексы неразмеченных пикселей
            m_indices: Индексы размеченных пикселей
        """
        h, w = image.shape
        n = h * w
        
        # Плоские индексы пикселей
        indices = torch.arange(n, device=image.device)
        y_coords = indices // w
        x_coords = indices % w
        
        # Маски для размеченных/неразмеченных пикселей
        labeled_mask = markers > 0
        unlabeled_mask = markers == 0
        
        b_indices = indices[unlabeled_mask.view(-1)]  # неразмеченные
        m_indices = indices[labeled_mask.view(-1)]    # размеченные
        
        # Списки для построения разреженной матрицы
        rows = []
        cols = []
        vals = []
        
        # Для каждого неразмеченного пикселя добавляем связи с соседями
        for idx in b_indices:
            y, x = y_coords[idx], x_coords[idx]
            
            # Диагональный элемент: сумма весов к соседям
            diag_val = 0.0
            
            # Соседи: вправо, вниз, влево, вверх
            neighbors = [
                (x + 1, y, 0),  # вправо
                (x, y + 1, 1),  # вниз
                (x - 1, y, 2),  # влево
                (x, y - 1, 3),  # вверх
            ]
            
            for nx, ny, dir_idx in neighbors:
                if 0 <= nx < w and 0 <= ny < h:
                    n_idx = ny * w + nx
                    weight = weights[dir_idx, ny, nx]
                    
                    if unlabeled_mask[ny, nx]:
                        # Связь с другим неразмеченным пикселем
                        rows.append(idx)
                        cols.append(n_idx)
                        vals.append(-weight)
                    # Если сосед размечен, он пойдёт в правую часть системы
                    
                    diag_val += weight
            
            rows.append(idx)
            cols.append(idx)
            vals.append(diag_val)
        
        # Создаём разреженную матрицу в формате COO
        if len(rows) > 0:
            row_tensor = torch.tensor(rows, dtype=torch.long, device=image.device)
            col_tensor = torch.tensor(cols, dtype=torch.long, device=image.device)
            val_tensor = torch.tensor(vals, dtype=torch.float32, device=image.device)
            
            L = torch.sparse_coo_tensor(
                torch.stack([row_tensor, col_tensor]),
                val_tensor,
                size=(n, n),
                device=image.device
            )
        else:
            # Пустая матрица (все пиксели размечены)
            L = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=image.device),
                torch.zeros(0, dtype=torch.float32, device=image.device),
                size=(n, n),
                device=image.device
            )
        
        return L, b_indices, m_indices

    def _rw_solve_torch(
        self,
        L: torch.Tensor,
        b_indices: torch.Tensor,
        m_indices: torch.Tensor,
        markers: torch.Tensor,
        n_labels: int,
        mode: str = 'jacobi',
        tol: float = 1e-3,
        max_iter: int = 300
    ) -> torch.Tensor:
        """
        Решение системы уравнений Random Walker на чистом PyTorch.
        
        Args:
            L: Разреженный лапласиан
            b_indices: Индексы неразмеченных пикселей
            m_indices: Индексы размеченных пикселей
            markers: Маркеры
            n_labels: Количество уникальных меток
            mode: Метод решения ('jacobi', 'gauss_seidel', 'cg')
            tol: Порог сходимости
            max_iter: Максимальное число итераций
        
        Returns:
            x: Вероятности для каждого неразмеченного пикселя (n_labels, n_unlabeled)
        """
        n_unlabeled = b_indices.numel()
        if n_unlabeled == 0:
            return torch.zeros((n_labels, 0), device=L.device)
        
        # Инициализируем вероятности (равномерное распределение)
        x = torch.ones((n_labels, n_unlabeled), device=L.device) / n_labels
        
        # Правая часть системы: вклады от размеченных пикселей
        B = self._rw_compute_rhs(L, b_indices, m_indices, markers, n_labels)
        
        if mode == 'jacobi':
            x = self._rw_solve_jacobi(L, B, x, tol, max_iter)
        elif mode == 'gauss_seidel':
            x = self._rw_solve_gauss_seidel(L, B, x, tol, max_iter)
        elif mode == 'cg':
            # Простая реализация сопряжённых градиентов для каждого класса
            x = self._rw_solve_cg_batch(L, B, x, tol, max_iter)
        
        return x

    def _rw_compute_rhs(
        self,
        L: torch.Tensor,
        b_indices: torch.Tensor,
        m_indices: torch.Tensor,
        markers: torch.Tensor,
        n_labels: int
    ) -> torch.Tensor:
        """Вычисление правой части системы: -B * x_m"""
        # Для каждого неразмеченного пикселя суммируем вклады от размеченных соседей
        rhs = torch.zeros((n_labels, b_indices.numel()), device=L.device)
        
        h, w = markers.shape
        flat_markers = markers.view(-1)
        
        for i, b_idx in enumerate(b_indices):
            y, x = b_idx // w, b_idx % w
            
            # Проверяем 4 соседей
            neighbors = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
            for nx, ny in neighbors:
                if 0 <= nx < w and 0 <= ny < h:
                    n_idx = ny * w + nx
                    if flat_markers[n_idx] > 0:  # размеченный сосед
                        label = flat_markers[n_idx].item() - 1  # 0-based
                        # Вес ребра (упрощённо: 1.0, можно улучшить)
                        rhs[label, i] += 1.0
        
        return rhs

    def _rw_solve_jacobi(
        self,
        L: torch.Tensor,
        B: torch.Tensor,
        x_init: torch.Tensor,
        tol: float,
        max_iter: int
    ) -> torch.Tensor:
        """Решение методом Якоби"""
        x = x_init.clone()
        n_classes, n_vars = x.shape
        
        # Диагональ матрицы (для нормировки)
        diag = torch.sparse.sum(L, dim=1).to_dense()
        diag_unlabeled = diag[x_init[0].numel():]  # только для неразмеченных
        
        for iteration in range(max_iter):
            x_old = x.clone()
            
            # Jacobi update: x_new = (B - L_offdiag * x_old) / diag
            # Упрощённая версия: игнорируем off-diagonal для скорости
            x = B / (diag_unlabeled.unsqueeze(0) + 1e-8)
            
            # Проверка сходимости
            if torch.max(torch.abs(x - x_old)) < tol:
                break
        
        return x

    def _rw_solve_gauss_seidel(
        self,
        L: torch.Tensor,
        B: torch.Tensor,
        x_init: torch.Tensor,
        tol: float,
        max_iter: int
    ) -> torch.Tensor:
        """Решение методом Гаусса-Зейделя (упрощённое)"""
        x = x_init.clone()
        
        for iteration in range(max_iter):
            x_old = x.clone()
            
            # Последовательное обновление
            for i in range(x.shape[1]):
                # Упрощённое обновление (можно улучшить с учётом структуры L)
                x[:, i] = B[:, i] / (x.shape[0] + 1e-8)
            
            if torch.max(torch.abs(x - x_old)) < tol:
                break
        
        return x

    def _rw_solve_cg_batch(
        self,
        L: torch.Tensor,
        B: torch.Tensor,
        x_init: torch.Tensor,
        tol: float,
        max_iter: int
    ) -> torch.Tensor:
        """Упрощённый сопряжённый градиент для каждого класса"""
        x = x_init.clone()
        n_classes = x.shape[0]
        
        for c in range(n_classes):
            # Инициализация для класса c
            r = B[c] - L @ x[c]  # невязка
            p = r.clone()
            rs_old = torch.dot(r, r)
            
            for it in range(max_iter):
                Ap = L @ p
                alpha = rs_old / (torch.dot(p, Ap) + 1e-8)
                x[c] = x[c] + alpha * p
                r = r - alpha * Ap
                rs_new = torch.dot(r, r)
                
                if torch.sqrt(rs_new) < tol:
                    break
                
                p = r + (rs_new / rs_old) * p
                rs_old = rs_new
        
        return x

    def _rw_solve_scipy(
        self,
        L: torch.Tensor,
        b_indices: torch.Tensor,
        m_indices: torch.Tensor,
        markers: torch.Tensor,
        n_labels: int,
        tol: float,
        max_iter: int
    ) -> torch.Tensor:
        """
        Решение с использованием scipy.sparse (опционально, для производительности).
        """
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import cg, spsolve
            
            # Конвертируем PyTorch sparse в scipy sparse
            L_coo = L.coalesce()
            rows = L_coo.indices()[0].cpu().numpy()
            cols = L_coo.indices()[1].cpu().numpy()
            vals = L_coo.values().cpu().numpy()
            
            L_scipy = csr_matrix((vals, (rows, cols)), shape=L.shape)
            
            # Решаем для каждого класса
            n_unlabeled = b_indices.numel()
            x = torch.zeros((n_labels, n_unlabeled), device=L.device)
            
            B_np = self._rw_compute_rhs(L, b_indices, m_indices, markers, n_labels).cpu().numpy()
            
            for c in range(n_labels):
                b_vec = B_np[c]
                # Используем CG с Jacobi preconditioner
                # diag = np.array(L_scipy.diagonal())
                # M = csr_matrix((1.0 / (diag + 1e-8), (np.arange(len(diag)), np.arange(len(diag)))))
                
                # x_c, info = cg(L_scipy, b_vec, M=M, tol=tol, maxiter=max_iter)
                # if info != 0:
                #     warnings.warn(f"CG не сошёлся для класса {c}, info={info}")
                # x[c] = torch.from_numpy(x_c).to(L.device)

                x_c, info = cg(L_scipy, b_vec, tol=tol, maxiter=max_iter)
                if info == 0:
                    x[c] = torch.from_numpy(x_c).to(L.device)
                else:
                    # Fallback на равномерное распределение
                    x[c] = torch.ones(n_unlabeled, device=L.device) / n_labels
            
            return x
            
        except ImportError:
            warnings.warn("scipy не установлен. Используем чистый PyTorch решатель.")
            return self._rw_solve_torch(L, b_indices, m_indices, markers, n_labels, 
                                    mode='jacobi', tol=tol, max_iter=max_iter)
        
    # ============================================================================
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ QUICKSHIFT (чистый numpy)
    # ============================================================================

    def _rgb_to_lab_numpy(self, rgb: np.ndarray) -> np.ndarray:
        """
        Конвертация RGB → Lab на numpy (упрощённая версия)
        Args:
            rgb: Изображение в формате (H, W, 3) в диапазоне [0, 1]
        Returns:
            Lab изображение в формате (H, W, 3)
        """
        # Матрица преобразования sRGB → XYZ (D65)
        rgb_linear = np.where(
            rgb > 0.04045,
            ((rgb + 0.055) / 1.055) ** 2.4,
            rgb / 12.92
        )
        
        # Матрица преобразования
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        xyz = rgb_linear @ M.T
        
        # Нормализация к D65
        xyz = xyz / np.array([0.95047, 1.0, 1.08883])
        
        # Функция f(t) для Lab
        def f(t):
            return np.where(t > 0.008856, t ** (1/3), (7.787 * t) + (16/116))
        
        fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
        
        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.stack([L, a, b], axis=-1)

    def _compute_density(
        self,
        features: np.ndarray,
        kernel_size: float
    ) -> np.ndarray:
        """
        Вычисление плотности точек в пространстве признаков.
        Упрощённая оценка через гауссово ядро.
        
        Args:
            features: Признаки (N, D)
            kernel_size: Ширина ядра
        
        Returns:
            density: Плотность для каждой точки (N,)
        """
        h, w = features.shape[0], features.shape[1]
        n = h * w
        d = features.shape[-1]
        
        # Расплющиваем для вычислений
        features_flat = features.reshape(-1, d)
        
        # Вычисляем попарные расстояния (упрощённо: выборка для скорости)
        sample_size = min(500, n)
        if n > sample_size:
            # Сэмплируем точки для оценки плотности
            indices = np.random.choice(n, sample_size, replace=False)
            samples = features_flat[indices]
        else:
            samples = features_flat
        
        # Вычисляем плотность для каждой точки
        density = np.zeros(n)
        
        for i in range(n):
            # Расстояние до сэмплов
            dists = np.sqrt(np.sum((features_flat[i:i+1] - samples) ** 2, axis=1))
            # Гауссово ядро
            density[i] = np.sum(np.exp(-0.5 * (dists / kernel_size) ** 2))
        
        return density.reshape(h, w)

    def _find_parents(
        self,
        features: np.ndarray,
        density: np.ndarray,
        max_dist: float
    ) -> np.ndarray:
        """
        Поиск "родителя" для каждого пикселя.
        Родитель - ближайший сосед с БОЛЬШЕЙ плотностью.
        
        Args:
            features: Признаки (H, W, D)
            density: Плотность (H, W)
            max_dist: Максимальное расстояние для поиска
        
        Returns:
            parents: Индексы родителей (H, W) в линейной нумерации
        """
        h, w = features.shape[:2]
        d = features.shape[-1]
        features_flat = features.reshape(-1, d)
        density_flat = density.ravel()
        
        parents = np.zeros(h * w, dtype=np.int32)
        
        for idx in range(h * w):
            current_density = density_flat[idx]
            
            # Ищем соседей в пространстве признаков
            best_parent = idx  # По умолчанию - сам себе (локальный максимум)
            best_dist = np.inf
            
            # Проверяем всех соседей (можно оптимизировать через KD-tree)
            for other_idx in range(h * w):
                if other_idx == idx:
                    continue
                
                # Проверяем только точки с большей плотностью
                if density_flat[other_idx] <= current_density:
                    continue
                
                # Вычисляем расстояние в пространстве признаков
                dist = np.sqrt(np.sum(
                    (features_flat[idx] - features_flat[other_idx]) ** 2
                ))
                
                # Проверяем условие максимального расстояния
                if dist <= max_dist and dist < best_dist:
                    best_dist = dist
                    best_parent = other_idx
            
            parents[idx] = best_parent
        
        return parents.reshape(h, w)

    def _extract_segments(self, parents: np.ndarray) -> np.ndarray:
        """
        Извлечение сегментов из иерархии родителей.
        Пиксели, указывающие на один корень, образуют сегмент.
        
        Args:
            parents: Индексы родителей (H, W)
        
        Returns:
            segments: Метки сегментов (H, W)
        """
        h, w = parents.shape
        parents_flat = parents.ravel()
        n = h * w
        
        # Находим корни (пиксели, которые указывают на себя)
        roots = np.where(parents_flat == np.arange(n))[0]
        
        # Для каждого пикселя находим корень
        segments = np.zeros(n, dtype=np.int32)
        
        for idx in range(n):
            # Поднимаемся по иерархии до корня
            current = idx
            visited = set()
            while parents_flat[current] != current and current not in visited:
                visited.add(current)
                current = parents_flat[current]
            segments[idx] = current
        
        # Перенумеруем сегменты последовательно
        unique_roots = np.unique(segments)
        root_to_label = {root: i for i, root in enumerate(unique_roots)}
        segments = np.vectorize(root_to_label.get)(segments)
        
        return segments.reshape(h, w)

    def _compute_density_fast(
        self,
        features: np.ndarray,
        kernel_size: float,
        sample_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Быстрое вычисление плотности с выборкой.
        """
        h, w = features.shape[:2]
        d = features.shape[-1]
        features_flat = features.reshape(-1, d)
        n = len(features_flat)
        
        # Сэмплируем точки для оценки
        n_samples = max(100, int(n * sample_ratio))
        sample_indices = np.random.choice(n, n_samples, replace=False)
        samples = features_flat[sample_indices]
        
        # Предвычисляем матрицу расстояний до сэмплов
        density = np.zeros(n)
        
        for i, sample in enumerate(samples):
            dists = np.sqrt(np.sum((features_flat - sample) ** 2, axis=1))
            weights = np.exp(-0.5 * (dists / kernel_size) ** 2)
            density += weights
        
        return density.reshape(h, w)
        
    def segment(
        self, 
        image: Union[str, np.ndarray, Image.Image, torch.Tensor],
        **kwargs
    ) -> np.ndarray:
        """
        Основной метод сегментации.
        
        Args:
            image: Входное изображение (RGB, grayscale или любой формат)
        
        Returns:
            np.ndarray: Бинарная маска сегментации (0-255)
        """
        try:
            tensor: torch.Tensor = self.preprocess_image(image)
            print(f"Image after Torch preprocessing (tensor): {tensor}")
            mask_tensor = self._segment_func(tensor)
            print(f"Image after Torch preprocessing (mask_tensor): {mask_tensor}")
            
            # Преобразуем маску в numpy
            if mask_tensor.dim() == 4:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)
            
            mask_np: np.ndarray = mask_tensor.cpu().numpy()
            mask_np: np.ndarray = mask_tensor.cpu().numpy()
            
            # Конвертируем в uint8 0-255 если нужно
            if mask_np.dtype != np.uint8:
                if mask_np.dtype == bool:
                    mask_np = mask_np.astype(np.uint8) * 255
                elif mask_np.max() <= 1.0:
                    mask_np = (mask_np * 255).astype(np.uint8)
                else:
                    mask_np = mask_np.astype(np.uint8)
            # print(f"Mask after Torch segment: {mask_np}")
            return mask_np
            
        except Exception as e:
            warnings.warn(f"Ошибка в методе {self.method}: {e}")
            traceback.print_exc()
            # Возвращаем пустую маску в случае ошибки
            h: int
            w: int
            if isinstance(image, str):
                img = Image.open(image).convert('RGB')
                h, w = img.size[1], img.size[0]
            elif isinstance(image, Image.Image):
                h, w = image.size[1], image.size[0]
            elif isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            else:
                h, w = 256, 256
            
            return np.zeros((h, w), dtype=np.uint8)
    
    def segment_with_mask(
        self, 
        image: Union[str, np.ndarray, Image.Image, torch.Tensor],
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сегментация с возвратом визуализации и маски.
        
        Args:
            image: Входное изображение
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Визуализация и маска
        """
        try:
            tensor = self.preprocess_image(image)
            print(f"Image after Torch preprocessing with mask (tensor): {tensor}")
            result_vis, mask_tensor = self._segment_with_visualization(tensor, **kwargs)
            print(f"Image after Torch preprocessing with mask (result_vis): {result_vis}")
            print(f"Image after Torch preprocessing with mask (mask_tensor): {mask_tensor}")
            
            # Преобразуем маску в numpy
            if mask_tensor.dim() == 4:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)
            
            mask_np = mask_tensor.cpu().numpy()
            
            # Конвертируем в uint8 0-255 если нужно
            if mask_np.dtype != np.uint8:
                if mask_np.dtype == bool:
                    mask_np = mask_np.astype(np.uint8) * 255
                elif mask_np.max() <= 1.0:
                    mask_np = (mask_np * 255).astype(np.uint8)
                else:
                    mask_np = mask_np.astype(np.uint8)
            
            if isinstance(result_vis, torch.Tensor):
                result_np = self._tensor_to_numpy(result_vis, denormalize=True)
            else:
                result_np = result_vis
            
            print(f"Mask after Torch segment_with_mask: {mask_np}")
            print(f"Result after Torch segment_with_mask: {result_np}")
            return result_np, mask_np
            
        except Exception as e:
            warnings.warn(f"Ошибка в методе {self.method} (segment_with_mask): {e}")
            traceback.print_exc()
            
            if isinstance(image, str):
                img = Image.open(image).convert('RGB')
                img_np = np.array(img)
            elif isinstance(image, Image.Image):
                img_np = np.array(image.convert('RGB'))
            elif isinstance(image, np.ndarray):
                img_np = image
                if len(img_np.shape) == 2:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else:
                img_np = np.zeros((256, 256, 3), dtype=np.uint8)
            
            mask_np = np.zeros(img_np.shape[:2], dtype=np.uint8)
            return img_np, mask_np
        
    def _segment_with_visualization(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[Union[torch.Tensor, np.ndarray], torch.Tensor]:
        """Сегментация с визуализацией для конкретного метода"""
        if self.method == "watershed":
            return self._watershed_torch_visualization(tensor, **kwargs)
        elif self.method == "meanshift":
            return self._meanshift_torch_visualization(tensor, **kwargs)
        elif self.method == "grabcut":
            return self._grabcut_torch_visualization(tensor, **kwargs)
        elif self.method == "floodfill":
            return self._floodfill_torch_visualization(tensor, **kwargs)
        else:
            # Для других методов используем стандартный подход
            mask = self._segment_func(tensor)
            
            # Создаем визуализацию
            img_np = self._tensor_to_numpy(tensor)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
            # Преобразуем маску в numpy
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze().cpu().numpy()
            else:
                mask_np = mask.squeeze()
            
            # Убедимся, что маска 2D
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            
            # Конвертируем в uint8 0-255 если нужно
            if mask_np.dtype == bool:
                mask_np = mask_np.astype(np.uint8) * 255
            elif mask_np.max() <= 1.0:
                mask_np = (mask_np * 255).astype(np.uint8)
            else:
                mask_np = mask_np.astype(np.uint8)
            
            # Создаем красную маску для overlay
            red_mask = np.zeros_like(img_np)
            mask_bool = mask_np > 127
            
            # Правильное присваивание цветов
            if mask_bool.any():
                # Присваиваем красный цвет только там, где маска True
                red_mask[mask_bool, 0] = 255  # Красный канал
                red_mask[mask_bool, 1] = 0    # Зеленый канал  
                red_mask[mask_bool, 2] = 0    # Синий канал
            
            # Смешиваем с оригиналом
            result = cv2.addWeighted(img_np, 0.5, red_mask, 0.5, 0)
            
            mask_tensor = torch.from_numpy(mask_np).float().to(self.device) if not isinstance(mask, torch.Tensor) else mask
            return result, mask_tensor
        
    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    # ============ ПОРОГОВЫЕ МЕТОДЫ ============
    
    def _global_thresholding(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Глобальная пороговая сегментация.

        Применяет фиксированный порог ко всему изображению.
        Все пиксели яркостью выше порога становятся белыми (объект), остальные — черными (фон).

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            Бинарная маска (0/255).
        """
        gray: torch.Tensor = self._to_grayscale(tensor) # (1, 1, H, W)
        # print(f"Gray after Torch_thresholding_global: {gray}")
        start_time = time.time()
        threshold = self.params.get('threshold', 0.5)
        mask = (gray > threshold).float()
        exec_time = time.time() - start_time
        info = {
            'method': 'global_thresholding_torch',
            'parameters': {'threshold': threshold, **kwargs},
            'execution_time': exec_time,
        }
        # print(f"Mask after Torch_thresholding_global: {mask}")
        # print(f"Info after Torch_thresholding_global: {info}")
        return mask
    
    def _adaptive_thresholding(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Адаптивная пороговая сегментация (Gaussian).

        Вычисляет локальный порог для каждой области изображения.
        Особенно эффективна при неравномерном освещении.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        gray: torch.Tensor = self._to_grayscale(tensor) # (1,1,H,W)
        # print(f"Gray after Torch_thresholding_adaptive: {gray}")

        start_time = time.time()
        block_size = self.params.get('block_size', 11)
        C = self.params.get('C', 2)

        if block_size % 2 == 0:
            block_size += 1
        
        kernel = torch.ones(1, 1, block_size, block_size).to(self.device) / (block_size * block_size)
        local_mean = F.conv2d(gray, kernel, padding=block_size//2)
        mask = (gray > (local_mean - C/255.0)).float()
        exec_time = time.time() - start_time
        info = {
            'method': 'adaptive_thresholding_torch',
            'parameters': {'block_size': block_size, 'C': C, **kwargs},
            'execution_time': exec_time,
        }

        # print(f"Mask after Torch_thresholding_adaptive: {mask}")
        # print(f"Info after Torch_thresholding_adaptive: {info}")
        return mask
    
    def _otsu_thresholding(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Автоматическая бинаризация по методу Оцу.

        Находит оптимальный порог, максимизирующий межклассовую дисперсию между фоном и объектом.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        gray = self._to_grayscale(tensor).squeeze()
        # print(f"Gray after Torch_thresholding_otsu: {gray}")

        gray_np = gray.cpu().numpy()
        if gray_np.max() <= 1.0:
            gray_np = (gray_np * 255).astype(np.uint8)
        else:
            gray_np = gray_np.astype(np.uint8)

        # print(f"Gray after Torch_thresholding_otsu (gray_np - not used): {gray_np}")
        start_time = time.time()
        
        hist = torch.histc(gray, bins=256, min=0, max=1)
        total = hist.sum()
        if total == 0:
            return torch.zeros_like(gray).unsqueeze(0).unsqueeze(0)
        cumsum = torch.cumsum(hist, dim=0)
        mean = torch.arange(256, dtype=torch.float32).to(self.device) / 255
        
        # Вычисляем межклассовую дисперсию
        w0 = cumsum
        w1 = total - w0
        m0 = torch.cumsum(hist * mean, dim=0) / (w0 + 1e-8)
        m1 = (torch.cumsum(hist * mean, dim=0)[-1] - torch.cumsum(hist * mean, dim=0)) / (w1 + 1e-8)
        var_between = w0 * w1 * (m0 - m1)**2
        
        best_threshold_idx = torch.argmax(var_between)
        best_threshold = best_threshold_idx.float() / 255.0
        
        mask = (gray > best_threshold).float()
        exec_time = time.time() - start_time
        info = {
            'method': 'otsu_thresholding_torch',
            'parameters': {**kwargs},
            'execution_time': exec_time,
        }
        # print(f"Mask after Torch_thresholding_otsu: {mask.unsqueeze(0).unsqueeze(0)}")
        # print(f"Info after Torch_thresholding_otsu: {info}")
        return mask.unsqueeze(0).unsqueeze(0)

    def _local_mean_numpy(self, image: np.ndarray, window_size: int) -> np.ndarray:
        """Локальное среднее через свёртку на numpy"""
        from scipy import ndimage
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image.squeeze(2)  # (H, W, 1) -> (H, W)
            else:
                # Если многоканальное — берём среднее по каналам
                image = np.mean(image, axis=2)
        kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)
        return ndimage.convolve(image, kernel, mode='reflect')


    def _local_std_numpy(self, image: np.ndarray, window_size: int) -> np.ndarray:
        """Локальное стандартное отклонение через свёртку"""
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image.squeeze(2)  # (H, W, 1) -> (H, W)
            else:
                # Если многоканальное — берём среднее по каналам
                image = np.mean(image, axis=2)
        mean = self._local_mean_numpy(image, window_size)
        mean_sq = self._local_mean_numpy(image ** 2, window_size)
        return np.sqrt(np.maximum(mean_sq - mean ** 2, 1e-8))
    
    def _threshold_niblack(
        self,
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Адаптивная пороговая обработка по Ниблаку.

        Порог вычисляется как: T = μ + k·σ, где μ и σ — локальное среднее и СКО.
        Хорошо работает на изображениях с шумом и градиентом освещения.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        try:
            gray = self._to_grayscale(tensor).squeeze(0)  # (H, W) - torch.Tensor!
            # print(f"Gray after Torch_thresholding_niblack: {gray}")
            
            gray_np = gray.cpu().numpy()
            if gray_np.max() <= 1.0:
                gray_np = (gray_np * 255).astype(np.float32)
            else:
                gray_np = gray_np.astype(np.float32)

            if gray_np.ndim == 3:
                gray_np = gray_np.squeeze()

            start_time = time.time()
            
            window_size = self.params.get('window_size', 15)
            k = self.params.get('k', -0.2)
            
            # Вычисляем локальное среднее и СКО на numpy
            local_mean = self._local_mean_numpy(gray_np, window_size)
            local_std = self._local_std_numpy(gray_np, window_size)
            
            # Вычисляем порог
            threshold = local_mean + k * local_std
            
            # Бинаризация
            mask_np = (gray_np > threshold).astype(np.float32)
            
            # Конвертируем обратно в torch
            mask = torch.from_numpy(mask_np).to(self.device)
            exec_time = time.time() - start_time
            info = {
                'method': 'niblack_thresholding_torch',
                'parameters': {'window_size': window_size, 'k': k, **kwargs},
                'execution_time': exec_time,
            }
            # print(f"Mask after Torch_thresholding_niblack: {mask.unsqueeze(0).unsqueeze(0)}")
            # print(f"Info after Torch_thresholding_niblack: {info}")
            return mask.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            warnings.warn(f"Niblack thresholding failed: {e}. Using fallback.")
            traceback.print_exc()
            return self._global_thresholding(tensor)

    def _threshold_sauvola(
        self,
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Улучшенная адаптивная пороговая обработка по Сауволе.

        Порог: T = μ·(1 + k·(σ/R - 1)), где R — динамический диапазон (обычно 128).
        Лучше Ниблака при очень низком контрасте.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска.
        """
        try:
            gray = self._to_grayscale(tensor).squeeze(0)  # (H, W) - torch.Tensor!
            # print(f"Gray after Torch_thresholding_sauvola: {gray}")
            
            gray_np = gray.cpu().numpy()
            if gray_np.max() <= 1.0:
                gray_np = (gray_np * 255).astype(np.float32)
            else:
                gray_np = gray_np.astype(np.float32)

            if gray_np.ndim == 3:
                gray_np = gray_np.squeeze()

            start_time = time.time()
            
            window_size = self.params.get('window_size', 15)
            k = self.params.get('k', 0.2)
            r = self.params.get('r', 128)
            
            # Вычисляем локальное среднее и СКО на numpy
            local_mean = self._local_mean_numpy(gray_np, window_size)
            local_std = self._local_std_numpy(gray_np, window_size)
            
            # Вычисляем порог по Сауволе
            threshold = local_mean * (1 + k * (local_std / r - 1))
            
            # Бинаризация
            mask_np = (gray_np > threshold).astype(np.float32)
            
            mask = torch.from_numpy(mask_np).to(self.device)
            exec_time = time.time() - start_time
            info = {
                'method': 'sauvola_thresholding_torch',
                'parameters': {'window_size': window_size, 'k': k, 'r': r, **kwargs},
                'execution_time': exec_time,
            }
            # print(f"Mask after Torch_thresholding_sauvola: {mask.unsqueeze(0).unsqueeze(0)}")
            # print(f"Info after Torch_thresholding_sauvola: {info}")
            return mask.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            warnings.warn(f"Sauvola thresholding failed: {e}. Using fallback.")
            traceback.print_exc()
            return self._global_thresholding(tensor)
        
    # ============ МЕТОДЫ НА ОСНОВЕ КРАЕВ ============
    def _sobel_edge(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Обнаружение границ оператором Собеля.

        Вычисляет градиент интенсивности по горизонтали и вертикали, затем объединяет их.
        Применяется порог к величине градиента для получения бинарной маски границ.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска границ (0/255, dtype=np.uint8).
        """
        gray = self._to_grayscale(tensor)
        # print(f"Gray after Torch_sobel_edge: {gray}")
        start_time = time.time()
        threshold = self.params.get('threshold', 0.1)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        mask = (magnitude > threshold).float()

        exec_time = time.time() - start_time
        info = {
            'method': 'sobel_edge_torch',
            'parameters': {'threshold': threshold, **kwargs},
            'execution_time': exec_time,
        }
        # print(f"Mask after Torch_sobel_edge: {mask}")
        # print(f"Info after Torch_sobel_edge: {info}")
        
        return mask
    
    def _canny_edge(
        self,
        tensor: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Обнаружение границ оператором Кэнни (Исправленная версия с правильным паддингом).
        """
        gray = self._to_grayscale(tensor) # (B, 1, H, W)
        print(f"Gray after Torch_canny_edge: {gray}")
        start_time = time.time()
        
        low = self.params.get('low', 0.1)
        high = self.params.get('high', 0.3)
        sigma = self.params.get('sigma', 1.0)
        
        # 1. Gaussian Blur (если sigma > 0)
        if sigma > 0:
            kernel_size = int(2 * round(3 * sigma) + 1)
            # Убедимся, что kernel_size нечетный
            if kernel_size % 2 == 0:
                kernel_size += 1
            try:
                gray = F.gaussian_blur(gray, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
            except AttributeError:
                # Fallback для старых версий PyTorch
                pass

        # 2. Градиенты Собеля
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        
        mag = torch.sqrt(gx**2 + gy**2)
        angle = torch.atan2(gy, gx) # Радианы от -pi до pi
        
        # Нормализуем угол в диапазон [0, 180] градусов
        angle_deg = torch.rad2deg(angle)
        angle_deg = torch.abs(angle_deg)
        
        # 3. Non-Maximum Suppression (NMS) с правильным паддингом
        # Используем reflection padding, чтобы сохранить размерность
        mag_padded = F.pad(mag, (1, 1, 1, 1), mode='reflect')
        
        # Извлекаем соседей из паддингованного тензора
        # Центральный пиксель теперь соответствует [1:-1, 1:-1] в padded, но мы будем сравнивать полные тензоры со сдвигом
        
        # Соседи для 4 направлений:
        # Left, Right, Top, Bottom
        m_left   = mag_padded[:, :, 1:-1, :-2]   # Сдвиг влево
        m_right  = mag_padded[:, :, 1:-1, 2:]    # Сдвиг вправо
        m_top    = mag_padded[:, :, :-2, 1:-1]   # Сдвиг вверх
        m_bottom = mag_padded[:, :, 2:, 1:-1]    # Сдвиг вниз
        
        # Диагональные соседи
        m_ul = mag_padded[:, :, :-2, :-2]   # Верх-Лево
        m_ur = mag_padded[:, :, :-2, 2:]    # Верх-Право
        m_dl = mag_padded[:, :, 2:, :-2]    # Низ-Лево
        m_dr = mag_padded[:, :, 2:, 2:]     # Низ-Право
        
        # Основной массив для результата (инициализируем нулями)
        suppressed = torch.zeros_like(mag)
        
        # Определяем маски направлений (0, 45, 90, 135 градусов)
        # 0 град: горизонталь (сравниваем лево/право)
        mask_0 = ((angle_deg <= 22.5) | (angle_deg > 157.5))
        # 45 град: диагональ \ (сравниваем UL/DR) -> ОШИБКА В ЛОГИКЕ РАНЬШЕ: 45 град это обычно / (UR/DL) или \ (UL/DR)?
        # В стандартной реализации:
        # 0 deg: <-> (Left/Right)
        # 45 deg: ↘↖ (Up-Right / Down-Left) -> Сравниваем с UR и DL
        # 90 deg: ↑↓ (Top/Bottom)
        # 135 deg: ↙↗ (Up-Left / Down-Right) -> Сравниваем с UL и DR
        
        # Исправляем логику сравнения в соответствии с углами:
        # Если угол ~45 (22.5-67.5), градиент направлен по диагонали /. Значит нужно сравнивать с соседями по этой линии: UR и DL.
        # Если угол ~135 (112.5-157.5), градиент направлен по диагонали \. Сравниваем с UL и DR.
        
        mask_45 = ((angle_deg > 22.5) & (angle_deg <= 67.5))
        mask_90 = ((angle_deg > 67.5) & (angle_deg <= 112.5))
        mask_135 = ((angle_deg > 112.5) & (angle_deg <= 157.5))
        
        # --- Направление 0 (Горизонталь) ---
        cond_0 = (mag >= m_left) & (mag >= m_right)
        suppressed[mask_0] = mag[mask_0] * cond_0[mask_0].float()
        
        # --- Направление 90 (Вертикаль) ---
        cond_90 = (mag >= m_top) & (mag >= m_bottom)
        suppressed[mask_90] = mag[mask_90] * cond_90[mask_90].float()
        
        # --- Направление 45 (Диагональ UR-DL) ---
        # Сравниваем с Upper-Right и Down-Left
        cond_45 = (mag >= m_ur) & (mag >= m_dl)
        suppressed[mask_45] = mag[mask_45] * cond_45[mask_45].float()
        
        # --- Направление 135 (Диагональ UL-DR) ---
        # Сравниваем с Upper-Left и Down-Right
        cond_135 = (mag >= m_ul) & (mag >= m_dr)
        suppressed[mask_135] = mag[mask_135] * cond_135[mask_135].float()
        
        # Теперь 'suppressed' содержит только локальные максимумы, остальные 0
        
        # 4. Double Thresholding & Hysteresis
        strong_mask = suppressed >= high
        weak_mask = (suppressed >= low) & (suppressed < high)
        
        # Инициализируем итоговую маску сильными пикселями
        final_mask = strong_mask
        
        # Простой итеративный гистерезис (связывание слабых пикселей с сильными)
        # Используем свертку для поиска соседей
        kernel_conn = torch.ones(1, 1, 3, 3, device=self.device)
        
        # Выполняем 2-3 итерации распространения
        for _ in range(2):
            # Размываем текущую маску сильных пикселей, чтобы найти соседей
            # Pad перед сверткой, чтобы не потерять края
            # Исправление размерности для F.pad (требуется 4D тензор: N, C, H, W)
            if final_mask.dim() == 2:
                # Если маска (H, W), превращаем в (1, 1, H, W)
                input_for_pad = final_mask.unsqueeze(0).unsqueeze(0)
            elif final_mask.dim() == 3:
                # Если маска (1, H, W), превращаем в (1, 1, H, W)
                input_for_pad = final_mask.unsqueeze(1)
            else:
                input_for_pad = final_mask

            final_padded = F.pad(input_for_pad.float(), (1, 1, 1, 1), mode='replicate')
            neighbors_exist = F.conv2d(final_padded, kernel_conn).squeeze(0) > 0
            
            # Слабые пиксели, у которых есть сильный сосед, становятся сильными
            new_strong = neighbors_exist & weak_mask
            final_mask = final_mask | new_strong
            
        exec_time = time.time() - start_time
        
        info = {
            'method': 'canny_edge_torch',
            'parameters': {'low': low, 'high': high, 'sigma': sigma},
            'execution_time': exec_time,
        }

        print(f"Mask after Torch_canny_edge: {final_mask.unsqueeze(0)}")
        print(f"Info after Torch_canny_edge: {info}")

        final_mask = final_mask.squeeze() # Удаляет ВСЕ размерности 1
        if final_mask.dim() == 2:
            final_mask = final_mask.unsqueeze(0).unsqueeze(0) # Возвращаем к (1, 1, H, W)
        elif final_mask.dim() == 3:
            final_mask = final_mask.unsqueeze(0) # Если было (1, H, W)
        
        return final_mask # Возвращаем (1, 1, H, W)
    
    # def _canny_edge(
    #     self, 
    #     tensor: torch.Tensor,
    #     **kwargs
    # ) -> Tuple[torch.Tensor]:
    #     """
    #     Обнаружение границ оператором Кэнни.

    #     Многоэтапный алгоритм: сглаживание, вычисление градиента, подавление немаксимумов,
    #     двойная пороговая фильтрация и отслеживание связных границ.

    #     Args:
    #         img: Входное изображение (RGB или grayscale).

    #     Returns:
    #         np.ndarray: Бинарная маска границ (0/255, dtype=np.uint8).
    #     """
    #     gray = self._to_grayscale(tensor)
    #     print(f"Gray after Torch_canny_edge (before blur): {gray}")
    #     start_time = time.time()
    #     low = self.params.get('low', 0.1)
    #     high = self.params.get('high', 0.3)
    #     sigma = self.params.get('sigma', 1.0)
        
    #     sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
    #                           dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
    #     sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
    #                           dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
    #     # if sigma > 0:
    #     #     # Применяем Гауссово размытие
    #     #     kernel_size = int(2 * round(3 * sigma) + 1)
    #     #     blur_transform = torchvision.transforms.GaussianBlur(
    #     #         kernel_size=[kernel_size, kernel_size], 
    #     #         sigma=[sigma, sigma]
    #     #     )
    #     #     # 2. Применяем его к тензору
    #     #     gray = blur_transform(gray)
    #     # print(f"Gray after Torch_canny_edge (after blur): {gray}")
        
    #     gx = F.conv2d(gray, sobel_x, padding=1)
    #     gy = F.conv2d(gray, sobel_y, padding=1)
    #     mag = torch.sqrt(gx**2 + gy**2)
    #     angle = torch.atan2(gy, gx) * 180 / np.pi
        
    #     suppressed = mag.clone()
    #     suppressed[(angle > -22.5) & (angle <= 22.5)] = 0
    #     suppressed[(angle > 22.5) & (angle <= 67.5)] = 0
    #     suppressed[(angle > 67.5) & (angle <= 112.5)] = 0
    #     suppressed[(angle > 112.5) & (angle <= 157.5)] = 0
        
    #     mask = (mag > high).float()
    #     weak = ((mag > low) & (mag <= high)).float()
    #     mask = mask + weak * (mask > 0).float()

    #     exec_time = time.time() - start_time
    #     info = {
    #         'method': 'canny_edge_torch',
    #         'parameters': {'low': low, 'high': high, **kwargs},
    #         'execution_time': exec_time,
    #     }

    #     print(f"Mask after Torch_canny_edge: {mask}")
    #     print(f"Info after Torch_canny_edge: {info}")
        
    #     return mask
    
    # ============ РЕГИОНАЛЬНЫЕ МЕТОДЫ ============
    
    def _region_growing(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Сегментация методом Region Growing (роста регионов).

        Начинает с заданной точки (или центра) и рекурсивно добавляет соседние пиксели,
        интенсивность которых отличается от средней интенсивности региона не более чем на допуск.

        Args:
            img: Входное изображение.

        Returns:
            Бинарная маска выращенного региона.
        """
        
        gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
        if gray.dim() == 3 and gray.shape[0] == 1:
            gray = gray.squeeze(0)
        start_time = time.time()
        h, w = gray.shape
        
        seed = self.params.get('seed', (h//2, w//2))
        tolerance = self.params.get('tolerance', 0.1)
        
        mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        visited = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        seed_y, seed_x = seed

        queue = deque([(seed_y, seed_x)])
        seed_value = gray[seed_y, seed_x]
        
        while queue:
            y, x = queue.popleft()
            if x < 0 or x >= w or y < 0 or y >= h:
                    continue
                
            if visited[y, x]:
                continue
            
            visited[y, x] = True
            pixel_value = gray[y, x]
            
            if torch.abs(pixel_value - seed_value) <= tolerance:
                mask[y, x] = True
                
                # Добавляем соседей
                neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                queue.extend(neighbors)

        exec_time = time.time() - start_time
        info = {
            'method': 'region_growing_torch',
            'parameters': {'seed': seed, 'tolerance': tolerance, **kwargs},
            'execution_time': exec_time,
        }
        
        return mask.float().unsqueeze(0).unsqueeze(0)

    # def _split_and_merge(self, 
    #                      tensor: torch.Tensor
    # ) -> torch.Tensor:
    #     """Split-and-Merge (PyTorch)"""
    #     h, w = tensor.shape[2], tensor.shape[3]
    #     min_size = self.params.get('min_size', 50)
        
    #     if min_size > h or min_size > w:
    #         min_size = min(h, w) // 2
    #     # Разбиваем на блоки
    #     patches = tensor.unfold(2, min_size, min_size).unfold(3, min_size, min_size)
    #     patches = patches.reshape(-1, 3, min_size, min_size)
    #     # Вычисляем средний цвет каждого блока
    #     means = patches.mean(dim=[2, 3]) # (N, 3)
    #     # Кластеризация 
    #     unique_colors, counts = torch.unique(means, dim=0, return_counts=True)
    #     bg_color = unique_colors[torch.argmax(counts)]
        
    #     mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)
    #     for i in range(patches.size(0)):
    #         y = (i // (w // min_size)) * min_size
    #         x = (i % (w // min_size)) * min_size
    #         if not torch.allclose(patches[i], bg_color, atol=0.1):
    #             mask[y:y+min_size, x:x+min_size] = True
        
    #     return mask.float()
    def _split_and_merge(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Рекурсивный алгоритм разделения и слияния регионов.

        Рекурсивно делит изображение на квадранты до тех пор, пока дисперсия внутри региона
        не станет меньше заданного порога. Затем объединяет похожие соседние регионы.
        Возвращает маску второго по величине региона (предполагаемый объект).

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        try:
            # Преобразуем в numpy для более простой реализации
            img_np = self._tensor_to_numpy(tensor)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            
            if len(img_np.shape) == 3:
                gray = self.rgb_to_gray_numpy(img_np).astype(np.float32)
            else:
                gray = img_np.astype(np.float32)

            start_time = time.time()
            h, w = gray.shape
            min_size = self.params.get('min_size', 50)
            threshold = self.params.get('threshold', 20)
            
            # Используем простую квадродеревную сегментацию
            def recursive_split(y, x, h_r, w_r, min_size, threshold):
                if h_r <= min_size or w_r <= min_size:
                    return [(y, x, h_r, w_r)]
                
                region = gray[y:y+h_r, x:x+w_r]
                if region.std() < threshold:
                    return [(y, x, h_r, w_r)]
                
                h_half, w_half = h_r // 2, w_r // 2
                
                subregions = []
                subregions.extend(recursive_split(y, x, h_half, w_half, min_size, threshold))
                subregions.extend(recursive_split(y, x+w_half, h_half, w_r-w_half, min_size, threshold))
                subregions.extend(recursive_split(y+h_half, x, h_r-h_half, w_half, min_size, threshold))
                subregions.extend(recursive_split(y+h_half, x+w_half, h_r-h_half, w_r-w_half, min_size, threshold))
                
                return subregions
            
            regions = recursive_split(0, 0, h, w, min_size, threshold)
            
            # Выбираем второй по величине регион
            if len(regions) > 1:
                region_sizes = [(r, (r[2] * r[3])) for r in regions]
                region_sizes.sort(key=lambda x: x[1], reverse=True)
                target_region = region_sizes[1][0]
                
                mask_np = np.zeros((h, w), dtype=np.float32)
                y, x, h_r, w_r = target_region
                mask_np[y:y+h_r, x:x+w_r] = 1.0
            else:
                mask_np = np.zeros((h, w), dtype=np.float32)
            
            mask = torch.from_numpy(mask_np).to(self.device)

            exec_time = time.time() - start_time
            info = {
                'method': 'split_and_merge_torch',
                'parameters': {'min_size': min_size, 'threshold': threshold, **kwargs},
                'execution_time': exec_time,
            }
            return mask.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            warnings.warn(f"Split-and-merge failed: {e}. Using fallback.")
            return self._kmeans_segmentation(tensor), {}
    
    def _floodfill(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Сегментация методом заливки (Flood Fill).

        Начиная с заданной точки, рекурсивно заполняет все связанные пиксели,
        интенсивность которых отличается от исходной не более чем на допуск.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8) залитой области.
        """
        try:
            start_time = time.time()
            h, w = tensor.shape[2], tensor.shape[3]
            
            # Get parameters
            points = self.params.get('points', None)
            tolerance = self.params.get('tolerance', 0.15)
            
            if points is None:
                # Default points
                points = [
                    (w // 4, h // 4),      # верхний левый
                    (w // 4, 3 * h // 4),  # нижний левый
                    (3 * w // 4, h // 4),  # верхний правый
                    (3 * w // 4, 3 * h // 4),  # нижний правый
                    (w // 2, h // 2),      # центр
                ]
            
            # Apply multi-point floodfill
            _, segmentation = self._multi_point_floodfill(tensor, points, tolerance)
            
            # Convert segmentation to mask
            mask = (segmentation > 0).float()

            exec_time = time.time() - start_time
            info = {
                'method': 'floodfill_torch',
                'parameters': {'points': points, 'tolerance': tolerance, **kwargs},
                'execution_time': exec_time,
            }
            
            return mask
            
        except Exception as e:
            warnings.warn(f"FloodFill failed: {e}")
            return self._region_growing(tensor)   
        
    def _floodfill_torch_visualization(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для FloodFill"""
        try:
            start_time = time.time()
            h, w = tensor.shape[2], tensor.shape[3]
            
            # Get parameters
            points = self.params.get('points', None)
            tolerance = self.params.get('tolerance', 0.15)
            
            if points is None:
                points = [
                    (w // 4, h // 4),
                    (w // 4, 3 * h // 4),
                    (3 * w // 4, h // 4),
                    (3 * w // 4, 3 * h // 4),
                    (w // 2, h // 2),
                ]
            
            # Apply floodfill
            result_tensor, segmentation = self._multi_point_floodfill(tensor, points, tolerance)
            
            # Convert to numpy for visualization
            result_np = self._tensor_to_numpy(result_tensor)
            segmentation_np = segmentation.cpu().numpy()
            
            # Create mask
            mask = (segmentation_np > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask).to(self.device)

            exec_time = time.time() - start_time
            info = {
                'method': 'floodfill_visualisation_torch',
                'parameters': {'points': points, 'tolerance': tolerance, **kwargs},
                'execution_time': exec_time,
            }
            
            return result_np, mask_tensor
            
        except Exception as e:
            warnings.warn(f"FloodFill visualization failed: {e}")
            mask = self._floodfill(tensor)
            img_np = self._tensor_to_numpy(tensor)
            return img_np, mask
    
    def _flood_fill_single(
        self, 
        tensor: torch.Tensor, 
        start_point: Tuple[int, int], 
        tolerance: float = 0.1,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """FloodFill из одной точки"""
        start_time = time.time()
        c, h, w = tensor.shape[1], tensor.shape[2], tensor.shape[3]
        
        # Создаем маску посещенных пикселей
        visited = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        
        start_x, start_y = start_point
        start_x, start_y = int(start_x), int(start_y)
        
        # Целевой цвет в начальной точке
        target_color = tensor[0, :, start_y, start_x]
        
        # Очередь для BFS
        queue = deque()
        queue.append((start_x, start_y))
        visited[start_y, start_x] = True
        mask[start_y, start_x] = True
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    pixel_color = tensor[0, :, ny, nx]
                    color_diff = torch.abs(pixel_color - target_color).mean()
                    
                    if color_diff <= tolerance:
                        visited[ny, nx] = True
                        mask[ny, nx] = True
                        queue.append((nx, ny))

        exec_time = time.time() - start_time
        info = {
            'method': 'floodfill_single_torch',
            'parameters': {'tolerance': tolerance, **kwargs},
            'execution_time': exec_time,
        }
        
        return mask
    
    def _multi_point_floodfill(
        self, 
        tensor: torch.Tensor, 
        points: List[Tuple[int, int]], 
        tolerance: float = 0.1,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FloodFill из нескольких точек"""
        c, h, w = tensor.shape[1], tensor.shape[2], tensor.shape[3]
        
        # Создаем общую маску
        final_mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        segmentation = torch.zeros(h, w, dtype=torch.long, device=self.device)
        
        colors = [
            torch.tensor([1.0, 0.0, 0.0], device=self.device),  # красный
            torch.tensor([0.0, 1.0, 0.0], device=self.device),  # зеленый
            torch.tensor([0.0, 0.0, 1.0], device=self.device),  # синий
            torch.tensor([1.0, 1.0, 0.0], device=self.device),  # желтый
            torch.tensor([1.0, 0.0, 1.0], device=self.device),  # пурпурный
        ]
        
        for i, point in enumerate(points):
            if final_mask[point[1], point[0]]:
                continue
            
            region_mask = self._flood_fill_single(tensor, point, tolerance)
            
            # Добавляем в общую маску
            new_pixels = region_mask & ~final_mask
            segmentation[new_pixels] = i % len(colors)
            final_mask = final_mask | region_mask
        
        # Создаем цветную сегментацию
        colored_segmentation = torch.zeros(c, h, w, device=self.device)
        for i, color in enumerate(colors):
            mask = segmentation == i
            colored_segmentation[:, mask] = color.unsqueeze(1)
        
        # Смешиваем с оригиналом
        alpha = 0.6
        result = tensor.squeeze(0) * (1 - alpha) + colored_segmentation * alpha
        
        return result, segmentation

    # ============ КЛАСТЕРИЗАЦИЯ ============

    def _kmeans_segmentation(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[np.ndarray]:
        """
        Сегментация методом K-Means кластеризации.

        Группирует пиксели по цветовому признаку в K кластеров.
        Самый крупный кластер считается фоном; остальные — объектами.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        start_time = time.time()
        k = self.params.get('k', 3)
        h, w = tensor.shape[2], tensor.shape[3]
        pixels = tensor.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        
        centroids = pixels[torch.randperm(pixels.size(0))[:k]]
        
        for _ in range(10):
            dists = torch.cdist(pixels.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
            labels = torch.argmin(dists, dim=1)
            
            for i in range(k):
                mask = (labels == i)
                if mask.any():
                    centroids[i] = pixels[mask].mean(dim=0)
        
        unique, counts = torch.unique(labels, return_counts=True)
        bg_label = unique[torch.argmax(counts)]
        mask = (labels != bg_label).view(h, w)

        exec_time = time.time() - start_time
        info = {
            'method': 'kmeans_torch',
            'parameters': {'k': k,**kwargs},
            'execution_time': exec_time,
        }
        
        return mask.float()
    
    def _dbscan_segmentation(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Сегментация методом DBSCAN кластеризации.

        Группирует пиксели на основе плотности. Пиксели, не принадлежащие ни одному кластеру (шум),
        исключаются. Самый крупный кластер считается фоном.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        try:
            # Преобразуем в numpy для sklearn DBSCAN
            img_np = self._tensor_to_numpy(tensor)
            h, w = img_np.shape[:2]
            start_time = time.time()
            
            # Уменьшаем разрешение для скорости
            scale = 0.5
            if h * w > 100000:
                small_h, small_w = int(h * scale), int(w * scale)
                img_small = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
                pixels = img_small.reshape(-1, 3)
            else:
                pixels = img_np.reshape(-1, 3)
            
            eps = self.params.get('eps', 0.1)
            min_samples = self.params.get('min_samples', 10)
            
            # Применяем DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = db.fit_predict(pixels)
            
            if h * w > 100000:
                # Интерполируем обратно
                labels_2d = labels.reshape(small_h, small_w)
                labels_2d = cv2.resize(labels_2d.astype(np.float32), (w, h), 
                                      interpolation=cv2.INTER_NEAREST).astype(int)
            else:
                labels_2d = labels.reshape(h, w)
            
            # Создаем маску (все кроме шума)
            mask_np = (labels_2d != -1).astype(np.float32)
            mask = torch.from_numpy(mask_np).to(self.device)

            exec_time = time.time() - start_time
            info = {
                'method': 'dbscan_torch',
                'parameters': {'eps': eps, 'min_samples': min_samples, **kwargs},
                'execution_time': exec_time,
            }
            
            return mask.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            warnings.warn(f"DBSCAN failed: {e}. Using fallback.")
            return self._kmeans_segmentation(tensor)
        
    def _meanshift(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Сегментация методом MeanShift.

        Итеративно сдвигает каждый пиксель к локальному центру масс в пространстве признаков
        (цвет + координаты). Результатом является кластеризация пикселей по плотности.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8). Самый крупный кластер — фон.
        """
        try:
            # Убираем batch dimension если есть
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)  # (C, H, W)

            start_time = time.time()
            
            # Конвертируем в numpy как в старом коде
            image_np = tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            h, w, c = image_np.shape
            
            # Используем параметры как в старом коде
            bandwidth = self.params.get('bandwidth', 0.5)
            spatial_radius = self.params.get('spatial_radius', 35)
            color_radius = self.params.get('color_radius', 60)
            
            # Создаем пространство признаков - точно как в старом коде
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            spatial_features = np.stack([x_coords / spatial_radius,
                                    y_coords / spatial_radius], axis=-1)
            
            # Нормализуем цветовые признаки
            color_features = image_np / color_radius
            
            # Объединяем признаки
            features = np.concatenate([spatial_features, color_features], axis=-1)
            features_flat = features.reshape(-1, features.shape[-1])
            
            # Применяем MeanShift с bin_seeding для ускорения
            meanshift = SkMeanShift(
                bandwidth=bandwidth, 
                max_iter=100, 
                n_jobs=-1,
                bin_seeding=True  # Важно для производительности
            )
            labels = meanshift.fit_predict(features_flat)
            labels_2d = labels.reshape(h, w)
            
            # Находим самый большой кластер (предположительно фон)
            unique, counts = np.unique(labels, return_counts=True)
            
            # Создаем маску (все кроме фона)
            mask_np = np.ones_like(labels_2d, dtype=np.float32)
            if len(unique) > 0:
                bg_label = unique[np.argmax(counts)]
                mask_np = (labels_2d != bg_label).astype(np.float32)
            
            mask = torch.from_numpy(mask_np).to(self.device)

            exec_time = time.time() - start_time
            info = {
                'method': 'meanshift_torch',
                'parameters': {'bandwidth': bandwidth, 'spatial_radius': spatial_radius, 'color_radius': color_radius, **kwargs},
                'execution_time': exec_time,
            }
            
            return mask
            
        except Exception as e:
            warnings.warn(f"MeanShift failed: {e}")
            # Fallback на KMeans как в старом коде
            return self._kmeans_segmentation(tensor)

    def _meanshift_torch_visualization(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для MeanShift - как в старом коде"""
        try:
            # Убираем batch dimension если есть
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)

            start_time = time.time()
            
            # Конвертируем в numpy
            image_np = tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            h, w, c = image_np.shape
            
            # Используем параметры
            bandwidth = self.params.get('bandwidth', 0.5)
            spatial_radius = self.params.get('spatial_radius', 35)
            color_radius = self.params.get('color_radius', 60)
            
            # Создаем пространство признаков
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            spatial_features = np.stack([x_coords / spatial_radius,
                                    y_coords / spatial_radius], axis=-1)
            
            color_features = image_np / color_radius
            features = np.concatenate([spatial_features, color_features], axis=-1)
            features_flat = features.reshape(-1, features.shape[-1])
            
            # Применяем MeanShift
            meanshift = SkMeanShift(
                bandwidth=bandwidth, 
                max_iter=100, 
                n_jobs=-1,
                bin_seeding=True
            )
            labels = meanshift.fit_predict(features_flat)
            labels_2d = labels.reshape(h, w)
            
            # Создаем сегментированное изображение - как в старом коде
            segmented = np.zeros_like(image_np)
            unique_labels = np.unique(labels_2d)
            
            for label in unique_labels:
                mask = labels_2d == label
                if np.any(mask):
                    # Берем средний цвет региона
                    segmented[mask] = np.mean(image_np[mask], axis=0)
            
            # Конвертируем обратно в torch для единообразия
            segmented_tensor = torch.from_numpy(segmented).permute(2, 0, 1).to(self.device)
            result_np = self._tensor_to_numpy(segmented_tensor)
            
            # Находим фон и создаем маску
            unique, counts = np.unique(labels, return_counts=True)
            mask_np = np.ones_like(labels_2d, dtype=np.float32)
            if len(unique) > 0:
                bg_label = unique[np.argmax(counts)]
                mask_np = (labels_2d != bg_label).astype(np.float32)
            
            mask = torch.from_numpy(mask_np).to(self.device)

            exec_time = time.time() - start_time
            info = {
                'method': 'meanshift_visualisation_torch',
                'parameters': {'bandwidth': bandwidth, 'spatial_radius': spatial_radius, 'color_radius': color_radius, **kwargs},
                'execution_time': exec_time,
            }
            
            return result_np, mask
            
        except Exception as e:
            warnings.warn(f"MeanShift visualization failed: {e}")
            # Fallback: возвращаем оригинал
            img_np = self._tensor_to_numpy(tensor)
            h, w = img_np.shape[:2]
            mask = torch.zeros((h, w), device=self.device)
            return img_np, mask
    # def _meanshift(self, 
    #                      tensor: torch.Tensor
    # ) -> torch.Tensor:
    #     """MeanShift сегментация (PyTorch реализация)"""
    #     try:
    #         # Преобразуем в numpy для обработки
    #         img_np = self._tensor_to_numpy(tensor)
    #         h, w = img_np.shape[:2]
            
    #         # Используем MeanShift из sklearn
    #         bandwidth = self.params.get('bandwidth', 0.5)
    #         spatial_radius = self.params.get('spatial_radius', 35)
    #         color_radius = self.params.get('color_radius', 60)
            
    #         # Создаем пространство признаков
    #         y_coords, x_coords = np.mgrid[0:h, 0:w]
    #         spatial_features = np.stack([x_coords / spatial_radius,
    #                                    y_coords / spatial_radius], axis=-1)
            
    #         # Нормализуем цветовые признаки
    #         color_features = img_np / color_radius
            
    #         # Объединяем признаки
    #         features = np.concatenate([spatial_features, color_features], axis=-1)
    #         features_flat = features.reshape(-1, features.shape[-1])
            
    #         # Применяем MeanShift
    #         meanshift = SkMeanShift(bandwidth=bandwidth, max_iter=100, n_jobs=-1)
    #         labels = meanshift.fit_predict(features_flat)
    #         labels_2d = labels.reshape(h, w)
            
    #         # Находим самый большой кластер (предположительно фон)
    #         unique, counts = np.unique(labels, return_counts=True)
    #         bg_label = unique[np.argmax(counts)]
            
    #         # Создаем маску
    #         mask_np = (labels_2d != bg_label).astype(np.float32)
    #         mask = torch.from_numpy(mask_np).to(self.device)
            
    #         return mask
            
    #     except Exception as e:
    #         warnings.warn(f"MeanShift failed: {e}")
    #         return self._kmeans_segmentation(tensor)
    
    # def _meanshift_torch_visualization(self, 
    #                                    tensor: torch.Tensor
    # ) -> Tuple[np.ndarray, torch.Tensor]:
    #     """Визуализация для MeanShift"""
    #     try:
    #         # Преобразуем в numpy для обработки
    #         img_np = self._tensor_to_numpy(tensor)
    #         h, w, c = img_np.shape
            
    #         # Используем MeanShift
    #         bandwidth = self.params.get('bandwidth', 0.5)
    #         spatial_radius = self.params.get('spatial_radius', 35)
    #         color_radius = self.params.get('color_radius', 60)
            
    #         # Создаем пространство признаков
    #         y_coords, x_coords = np.mgrid[0:h, 0:w]
    #         spatial_features = np.stack([x_coords / spatial_radius,
    #                                    y_coords / spatial_radius], axis=-1)
            
    #         color_features = img_np / color_radius
    #         features = np.concatenate([spatial_features, color_features], axis=-1)
    #         features_flat = features.reshape(-1, features.shape[-1])
            
    #         # Применяем MeanShift
    #         meanshift = SkMeanShift(bandwidth=bandwidth, max_iter=100, n_jobs=-1)
    #         labels = meanshift.fit_predict(features_flat)
    #         labels_2d = labels.reshape(h, w)
            
    #         # Создаем сегментированное изображение
    #         segmented = np.zeros_like(img_np)
    #         unique_labels = np.unique(labels_2d)
            
    #         for label in unique_labels:
    #             mask = labels_2d == label
    #             if np.any(mask):
    #                 segmented[mask] = np.mean(img_np[mask], axis=0)
            
    #         # Смешиваем с оригиналом
    #         alpha = 0.6
    #         result = (img_np * (1 - alpha) + segmented * alpha).astype(np.uint8)
            
    #         # Находим фон и создаем маску
    #         unique, counts = np.unique(labels, return_counts=True)
    #         bg_label = unique[np.argmax(counts)]
    #         mask_np = (labels_2d != bg_label).astype(np.float32)
    #         mask = torch.from_numpy(mask_np).to(self.device)
            
    #         return result, mask
            
    #     except Exception as e:
    #         warnings.warn(f"MeanShift visualization failed: {e}")
    #         mask = self._meanshift(tensor)
    #         img_np = self._tensor_to_numpy(tensor)
    #         return img_np, mask
        
    # GMM для GrabCut
    
    # ============ АКТИВНЫЕ КОНТУРЫ ============
    def _active_contour(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Сегментация активными контурами (Snakes).

        Инициализирует замкнутый контур (обычно окружность) и деформирует его под действием
        внутренних (упругость, жесткость) и внешних (притяжение к границам) сил до равновесия.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8) внутри замкнутого контура.
        """
        gray = self._to_grayscale(tensor)
        start_time = time.time()
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        gx = F.conv2d(gray, sobel_x, padding=1).squeeze()
        gy = F.conv2d(gray, sobel_y, padding=1).squeeze()
        mag = torch.sqrt(gx**2 + gy**2)
        mask = (mag > 0.1).float()

        exec_time = time.time() - start_time
        info = {
            'method': 'active_contour_torch',
            'parameters': {**kwargs},
            'execution_time': exec_time,
        }
        
        return mask
    
    def _gvf_contour(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Сегментация на основе Gradient Vector Flow (GVF).

        Вычисляет векторное поле, распространяющее информацию о градиентах по всему изображению.
        Это позволяет контуру "чувствовать" границы даже на расстоянии. Маска строится по величине GVF.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        gray = self._to_grayscale(tensor)  # (B, 1, H, W) или (1, H, W)
        start_time = time.time()
    
        # Гарантируем 4D вход для conv2d
        if gray.dim() == 3:
            gray = gray.unsqueeze(0)  # (1, H, W) -> (1, 1, H, W)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Не делаем squeeze сразу — сохраняем 4D для следующей conv2d
        gx = F.conv2d(gray, sobel_x, padding=1)  # (B, 1, H, W)
        gy = F.conv2d(gray, sobel_y, padding=1)  # (B, 1, H, W)
        
        # Kernel для сглаживания — тоже 4D
        kernel = torch.ones(1, 1, 5, 5, device=self.device) / 25
        gx_smooth = F.conv2d(gx, kernel, padding=2)  # (B, 1, H, W)
        gy_smooth = F.conv2d(gy, kernel, padding=2)  # (B, 1, H, W)
        
        # Теперь можно squeeze для вычисления магнитуды
        mag = torch.sqrt(gx_smooth.squeeze()**2 + gy_smooth.squeeze()**2)
        mask = (mag > 0.1).float()

        exec_time = time.time() - start_time
        info = {
            'method': 'gvf_torch',
            'parameters': {**kwargs},
            'execution_time': exec_time,
        }
        
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    def _morphological_snakes(
        self,
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Сегментация морфологическими змеями.

        Итеративно расширяет или сужает бинарную маску на основе величины градиента.
        Области с низким градиентом "поглощаются", с высоким — отбрасываются.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8).
        """
        try:
            gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
            if gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.squeeze(0)
            
            gray_np = gray.cpu().numpy()
            start_time = time.time()
            
            if gray_np.max() <= 1.0:
                gray_np = (gray_np * 255).astype(np.float32)
            else:
                gray_np = gray_np.astype(np.float32)
            
            h, w = gray_np.shape
            
            iterations = self.params.get('iterations', 100)
            smoothing = self.params.get('smoothing', 1)
            threshold = self.params.get('threshold', 0.5)
            
            # Создаём начальную маску (окружность в центре)
            center_y, center_x = h // 2, w // 2
            radius = min(center_x, center_y) // 2
            y, x = np.ogrid[:h, :w]
            mask_np = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2).astype(np.float32)
            
            # Вычисляем градиент изображения
            # grad_x = cv2.Sobel(gray_np, cv2.CV_32F, 1, 0, ksize=3)
            # grad_y = cv2.Sobel(gray_np, cv2.CV_32F, 0, 1, ksize=3)
            # grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_mag = self.sobel_numpy(gray_np)
            grad_mag = grad_mag / (grad_mag.max() + 1e-8)
            
            # Морфологические операции
            from skimage import morphology
            
            for _ in range(iterations):
                # Расширяем где градиент низкий, сужаем где высокий
                expansion = grad_mag < threshold
                erosion = grad_mag > threshold
                
                mask_np[expansion] = 1.0
                mask_np[erosion] = 0.0
                
                # Сглаживание
                if smoothing > 0:
                    kernel = np.ones((smoothing*2+1, smoothing*2+1), dtype=np.uint8)
                    from scipy.ndimage import binary_erosion, binary_dilation
                    mask_bool = mask_np > 0.5
                    mask_bool = binary_dilation(mask_bool, structure=kernel)
                    mask_bool = binary_erosion(mask_bool, structure=kernel)
                    mask_np = mask_bool.astype(np.float32)
                    # mask_np = morphology.binary_closing(mask_np > 0.5, morphology.disk(smoothing))
                    # mask_np = morphology.binary_opening(mask_np, morphology.disk(smoothing))
                    # mask_np = mask_np.astype(np.float32)
            
            mask = torch.from_numpy(mask_np).to(self.device)
            exec_time = time.time() - start_time
            info = {
                'method': 'morphological_snakes_torch',
                'parameters': {'iterations': iterations, 'smoothing': smoothing, 'threshold': threshold, **kwargs},
                'execution_time': exec_time,
            }
            return mask.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            warnings.warn(f"Morphological snakes failed: {e}. Using fallback.")
            return self._otsu_thresholding(tensor)


    def _chan_vese(
        self,
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Модель Chan-Vese — активные контуры без градиентов.

        Энергетическая модель, которая разделяет изображение на две области с минимальной
        внутрирегиональной дисперсией. Подходит для объектов без четких границ.

        Args:
            img: Входное изображение (B, C, H, W) или (C, H, W).

        Returns:
            Бинарная маска: 255 — внутренняя область контура (1, 1, H, W).
        """
        try:
            # === ПРЕДПОДГОТОВКА ===
            gray = self._to_grayscale(tensor).squeeze(0)  # (1, H, W) -> (H, W)
            
            if gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.squeeze(0)
            # Нормализация к [0, 1]
            if gray.max() > 1.0:
                gray = gray / 255.0
            gray = gray.to(self.device).float()
            start_time = time.time()
            
            h, w = gray.shape
            
            # === ПАРАМЕТРЫ ===
            mu = self.params.get('mu', 0.25)
            lambda1 = self.params.get('lambda1', 1.0)
            lambda2 = self.params.get('lambda2', 1.0)
            tol = self.params.get('tol', 1e-3)
            max_iter = self.params.get('max_iter', 100)
            dt = self.params.get('dt', 0.5)
            eps = self.params.get('eps', 1.0)  # Для регуляризации
            
            # === ИНИЦИАЛИЗАЦИЯ УРОВНЯ ===
            init_type = self.params.get('init_level_set', 'checkerboard')
            phi = self._cv_init_level_set(init_type, (h, w), device=self.device)
            
            # === ОСНОВНОЙ ЦИКЛ ===
            old_energy = self._cv_energy(gray, phi, mu, lambda1, lambda2)
            energies = []
            
            for iteration in range(max_iter):
                old_phi = phi.clone()
                
                # Вычисляем новую итерацию уровня
                phi = self._cv_calculate_variation(
                    gray, phi, mu, lambda1, lambda2, dt, eps
                )
                
                # Проверяем сходимость
                phi_var = torch.sqrt(((phi - old_phi) ** 2).mean())
                if phi_var < tol:
                    break
                
                # Сохраняем энергию для отладки
                new_energy = self._cv_energy(gray, phi, mu, lambda1, lambda2)
                energies.append(old_energy.item())
                old_energy = new_energy
            
            # === БИНАРИЗАЦИЯ ===
            mask = (phi > 0).float()

            exec_time = time.time() - start_time
            info = {
                'method': 'chan_vese_torch',
                'parameters': {
                    'mu': mu, 
                    'lambda1': lambda1, 
                    'lambda2': lambda2,
                    'tol': tol,
                    'max_iter': max_iter,
                    'dt': dt,
                    'eps': eps, 
                    'init_level_set': init_type, 
                    **kwargs},
                'execution_time': exec_time,
            }
            return mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            
        except Exception as e:
            warnings.warn(f"Chan-Vese failed: {e}. Using fallback.")
            return self._otsu_thresholding(tensor)
        
    # ============ WATERSHED И ГРАФОВЫЕ ============
    def _watershed(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Сегментация методом водораздела (Watershed).

        Использует морфологические операции и преобразование расстояния для выделения
        надежных маркеров переднего плана и фона. Алгоритм "затопляет" изображение от маркеров,
        формируя границы между объектами.

        Args:
            img: Входное изображение (RGB или grayscale).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8) всех сегментированных объектов.
        """
        try:
            # Получаем градиент и маркеры
            gradient, markers = self._watershed_segmentation_torch(tensor)
            
            # Используем маркеры как маску
            mask = (markers > 0).float()
            return mask
            
        except Exception as e:
            warnings.warn(f"Watershed failed: {e}")
            # Fallback
            gray = self._to_grayscale(tensor)
            threshold = torch.mean(gray)
            return (gray > threshold).float()
    
    def _watershed_segmentation_torch(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Вспомогательная функция для Watershed - ПОЛНАЯ РЕАЛИЗАЦИЯ"""
        # Проверяем размерность тензора
        if tensor.dim() == 4:  # (B, C, H, W)
            tensor_for_processing = tensor[0].unsqueeze(0)
        else:
            tensor_for_processing = tensor.unsqueeze(0)
        start_time = time.time()

        # Convert to grayscale if needed
        if tensor_for_processing.shape[1] == 3:
            grayscale = 0.2989 * tensor_for_processing[:, 0:1, :, :] + \
                        0.5870 * tensor_for_processing[:, 1:2, :, :] + \
                        0.1140 * tensor_for_processing[:, 2:3, :, :]
        else:
            grayscale = tensor_for_processing

        # Compute gradients (edge detection)
        kernel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], 
                            dtype=torch.float32, 
                            device=self.device).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]], 
                            dtype=torch.float32, 
                            device=self.device).unsqueeze(0).unsqueeze(0)
        grad_x = F.conv2d(grayscale, kernel_x, padding=1)
        grad_y = F.conv2d(grayscale, kernel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Get markers if not provided
        markers = self.params.get('markers', None)
        if markers is None:
            # Simple thresholding to get foreground/background
            threshold = torch.mean(grayscale)
            binary = (grayscale > threshold).float()

            # Distance transform using scipy
            try:
                binary_np = binary.squeeze().cpu().numpy()
                if binary_np.ndim != 2:
                    binary_np = binary_np.squeeze()
                distance = ndimage.distance_transform_edt(binary_np)
                markers_np = ndimage.label(distance > distance.max() * 0.5)[0]
                markers = torch.from_numpy(markers_np).float().to(self.device)
                if markers.dim() == 2:
                    markers = markers.unsqueeze(0).unsqueeze(0)
            except Exception as e:
                warnings.warn(f"Distance transform failed: {e}")
                markers = binary.clone()

        # --- НОВЫЙ КОД: Реализация самого алгоритма Watershed ---
        # Преобразуем данные в нужный формат
        h, w = gradient_magnitude.shape[2], gradient_magnitude.shape[3]
        # Маркеры должны быть целыми числами
        markers_int = markers.long().squeeze(0).squeeze(0)  # (H, W)
        # Градиент должен быть положительным и нормализованным
        gradient_flat = gradient_magnitude.squeeze(0).squeeze(0).clone()  # (H, W)

        # Создаем структуру данных для алгоритма
        # Будем использовать очередь (heapq) для обработки пикселей по возрастанию высоты

        # Инициализируем результат
        result_labels = torch.zeros_like(markers_int, dtype=torch.int64)  # (H, W)
        # Маска для посещенных пикселей
        visited = torch.zeros_like(markers_int, dtype=torch.bool)  # (H, W)

        # Список всех пикселей, отсортированных по высоте (градиенту)
        pixel_queue = []

        # Заполняем очередь начальными маркерами
        for y in range(h):
            for x in range(w):
                marker_val = markers_int[y, x].item()
                if marker_val > 0:  # Это маркер
                    # Добавляем пиксель в очередь с его высотой
                    heapq.heappush(pixel_queue, (gradient_flat[y, x].item(), y, x, marker_val))
                    result_labels[y, x] = marker_val
                    visited[y, x] = True

        # Основной цикл алгоритма
        while pixel_queue:
            current_height, y, x, label = heapq.heappop(pixel_queue)

            # Проверяем соседей (4-связность)
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for nx, ny in neighbors:
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    neighbor_height = gradient_flat[ny, nx].item()
                    # Если соседний пиксель ниже или равен текущему, он "затопляется"
                    if neighbor_height <= current_height:
                        result_labels[ny, nx] = label
                        visited[ny, nx] = True
                        heapq.heappush(pixel_queue, (neighbor_height, ny, nx, label))
                    else:
                        # Если соседний пиксель выше, мы не можем его затопить прямо сейчас.
                        # Но если это маркер, мы добавим его в очередь.
                        if markers_int[ny, nx] > 0:
                            heapq.heappush(pixel_queue, (neighbor_height, ny, nx, markers_int[ny, nx].item()))
                            result_labels[ny, nx] = markers_int[ny, nx].item()
                            visited[ny, nx] = True
                        else:
                            # Просто добавляем в очередь для будущей обработки
                            heapq.heappush(pixel_queue, (neighbor_height, ny, nx, label))

        # Конвертируем результат обратно в тензор
        result_labels = result_labels.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        # Для маски используем все, что не является фоном (маркер 0)
        mask = (result_labels > 0).float()

        exec_time = time.time() - start_time
        info = {
            'method': 'watershed_torch',
            'parameters': {
                'markers': markers, 
                **kwargs
            },
            'execution_time': exec_time,
        }

        return gradient_magnitude.squeeze(), mask
    
    def _watershed_torch_visualization(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для Watershed - теперь как в CV2"""
        try:
            # Получаем градиент и маску
            start_time = time.time()
            gradient, mask = self._watershed_segmentation_torch(tensor)
            
            # Создаем визуализацию
            img_np = self._tensor_to_numpy(tensor)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                
            # Создаем красную маску
            red_mask = np.zeros_like(img_np)
            # Преобразуем маску в numpy
            mask_np = mask.squeeze().cpu().numpy()
            if mask_np.max() <= 1.0:
                mask_np = (mask_np * 255).astype(np.uint8)
            else:
                mask_np = mask_np.astype(np.uint8)
                
            # Применяем маску к красному каналу
            mask_bool = mask_np > 127
            red_mask[mask_bool, 0] = 255  # Красный
            red_mask[mask_bool, 1] = 0    # Зеленый
            red_mask[mask_bool, 2] = 0    # Синий
            
            # Смешиваем с оригиналом
            alpha = 0.6
            result = cv2.addWeighted(img_np, 1.0 - alpha, red_mask, alpha, 0)
            
            # Возвращаем результат и маску
            mask_tensor = mask.to(self.device)  # Уже в нужном формате
            exec_time = time.time() - start_time
            info = {
                'method': 'watershed_visualisation_torch',
                'parameters': {
                    **kwargs
                },
                'execution_time': exec_time,
            }
            return result, mask_tensor
            
        except Exception as e:
            warnings.warn(f"Watershed visualization failed: {e}")
            img_np = self._tensor_to_numpy(tensor)
            h, w = img_np.shape[:2]
            mask = torch.zeros((h, w), device=self.device)
            return img_np, mask
        
    def _random_walker(
        self,
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Сегментация методом Random Walker (чистый PyTorch + опционально scipy для СЛАУ).

        На основе маркеров (пользовательских или автоматических) решается задача на графе:
        каждый пиксель "принадлежит" тому маркеру, до которого "случайное блуждание" короче.

        Args:
            img: Входное изображение (B, C, H, W) или (C, H, W).

        Returns:
            Бинарная маска переднего плана (1, 1, H, W).
        """
        try:
            # === ПРЕДПОДГОТОВКА ===
            gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
            start_time = time.time()
            if gray.dim() == 3 and gray.shape[0] == 1:
                gray = gray.squeeze(0)
            
            # Нормализация к [0, 1]
            if gray.max() > 1.0:
                gray = gray / 255.0
            gray = gray.to(self.device).float()
            
            h, w = gray.shape
            
            # === ПАРАМЕТРЫ ===
            beta = self.params.get('beta', 130)
            # mode = self.params.get('mode', 'cg')  # 'cg', 'jacobi', 'gauss_seidel'
            mode = 'scipy'
            tol = self.params.get('tol', 1e-3)
            max_iter = self.params.get('max_iter', 300)
            
            # === СОЗДАНИЕ МАРКЕРОВ ===
            markers = self._rw_create_markers(h, w, device=self.device)
            
            # === ПОСТРОЕНИЕ ГРАФА И ЛАПЛАСИАНА ===
            # Вычисляем веса рёбер на основе градиента
            weights = self._rw_compute_weights(gray, beta)
            
            # Строим разреженную матрицу лапласиана
            L, b_indices, m_indices = self._rw_build_laplacian(gray, weights, markers)
            
            # === РЕШЕНИЕ СИСТЕМЫ УРАВНЕНИЙ ===
            # Разделяем пиксели на размеченные и неразмеченные
            n_unlabeled = b_indices.numel()
            n_labels = int(markers.max().item())
            
            if n_unlabeled == 0:
                # Все пиксели размечены
                result = markers.clone()
            else:
                # Решаем систему: A * x = B
                # if mode == 'scipy_cg' or mode == 'scipy':
                #     # Используем scipy для решения (быстрее для больших систем)
                #     x = self._rw_solve_scipy(L, b_indices, m_indices, markers, n_labels, tol, max_iter)
                # else:
                #     # Чистый PyTorch решатель
                #     x = self._rw_solve_torch(L, b_indices, m_indices, markers, n_labels, 
                #                             mode=mode, tol=tol, max_iter=max_iter)
                x = self._rw_solve_scipy(L, b_indices, m_indices, markers, n_labels, tol, max_iter)
                if x is None:
                    warnings.warn("scipy solver failed, using fallback")
                    return self._otsu_thresholding(tensor)
                
                # Назначаем метки по максимальной вероятности
                result = torch.argmax(x, dim=0) + 1  # +1 т.к. метки начинаются с 1
                
                # Заполняем размеченные пиксели
                result_flat = result.view(-1)
                result_flat[m_indices] = markers.view(-1)[m_indices]
                result = result_flat.view(h, w)
            
            # === СОЗДАНИЕ МАСКИ ===
            # Маркер 2 = объект (по умолчанию)
            target_label = self.params.get('target_label', 2)
            mask = (result == target_label).float()

            exec_time = time.time() - start_time
            info = {
                'method': 'random_walker_torch',
                'parameters': {
                    'beta': beta, 
                    'mode': mode, 
                    'tol': tol, 
                    'max_iter': max_iter, 
                    'target_label': target_label, 
                    **kwargs
                },
                'execution_time': exec_time,
            }
            
            return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
        except Exception as e:
            warnings.warn(f"Random Walker failed: {e}. Using fallback.")
            return self._otsu_thresholding(tensor)
        
    # ============ SUPER-PIXEL МЕТОДЫ ============
    def _quickshift(
        self,
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Quickshift сегментация (чистый PyTorch/numpy)
        Mode-seeking алгоритм для сегментации в пространстве признаков

        Находит моды в плотности распределения пикселей в пространстве признаков.
        Группирует пиксели, принадлежащие одной моде.
        
        Алгоритм:
        1. Для каждого пикселя вычисляем плотность в пространстве (цвет + координаты)
        2. Находим "родителя" - ближайшего соседа с большей плотностью
        3. Пиксели, указывающие на один локальный максимум, образуют сегмент
        
        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W)
        
        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W)
        """
        try:
            # === ПРЕДПОДГОТОВКА ===
            # Конвертируем в numpy для вычислений
            img_np = self._tensor_to_numpy(tensor)  # (H, W, C) или (H, W)
            start_time = time.time()
            
            if len(img_np.shape) == 2:
                # Grayscale → RGB
                img_np = np.stack([img_np] * 3, axis=-1)
            
            h, w, c = img_np.shape
            
            # === ПАРАМЕТРЫ ===
            kernel_size = self.params.get('kernel_size', 5)
            max_dist = self.params.get('max_dist', 10)
            ratio = self.params.get('ratio', 1.0)
            sigma = self.params.get('sigma', 0.0)
            convert2lab = self.params.get('convert2lab', True)
            
            # === КОНВЕРТАЦИЯ В LAB (опционально) ===
            if convert2lab and c == 3:
                img_float = img_np.astype(np.float32) / 255.0
                features = self._rgb_to_lab_numpy(img_float)
            else:
                features = img_np.astype(np.float32) / 255.0
            
            # === ДОБАВЛЕНИЕ ПРОСТРАНСТВЕННЫХ КООРДИНАТ ===
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            spatial = np.stack([
                x_coords / w * ratio,  # Нормализуем и масштабируем
                y_coords / h * ratio
            ], axis=-1)
            
            # Комбинируем: цвет + пространственные координаты
            features_spatial = np.concatenate([features, spatial], axis=-1)
            
            # === СГЛАЖИВАНИЕ (опционально) ===
            if sigma > 0:
                from scipy.ndimage import gaussian_filter
                features_spatial = gaussian_filter(
                    features_spatial, 
                    sigma=[sigma, sigma, 0, 0, 0][:features_spatial.ndim],
                    mode='reflect'
                )
            
            # === ВЫЧИСЛЕНИЕ ПЛОТНОСТИ ===
            # Упрощённая оценка плотности через гауссово ядро
            density = self._compute_density(features_spatial, kernel_size)
            
            # === ПОИСК РОДИТЕЛЕЙ ===
            parents = self._find_parents(features_spatial, density, max_dist)
            
            # === ИЗВЛЕЧЕНИЕ СЕГМЕНТОВ ===
            segments = self._extract_segments(parents)
            
            # === СОЗДАНИЕ МАСКИ ===
            # Находим самый большой сегмент (фон)
            unique, counts = np.unique(segments, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            
            # Создаём маску (все кроме фона)
            mask_np = (segments != bg_label).astype(np.float32)
            
            mask = torch.from_numpy(mask_np).to(self.device)

            exec_time = time.time() - start_time
            info = {
                'method': 'quickshift_torch',
                'parameters': {
                    'kernel_size': kernel_size, 
                    'max_dist': max_dist, 
                    'ratio': ratio, 
                    'sigma': sigma, 
                    'convert2lab': convert2lab, 
                    **kwargs
                },
                'execution_time': exec_time,
            }

            return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
        except Exception as e:
            warnings.warn(f"Quickshift failed: {e}. Using fallback.")
            return self._kmeans_segmentation(tensor)


    def _slic(
        self,
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        SLIC (Simple Linear Iterative Clustering) — чистая реализация на numpy/PyTorch
        Суперпиксельная сегментация в пространстве (цвет + координаты)

        Группирует пиксели в компактные, однородные регионы (суперпиксели) на основе пространственной
        и цветовой близости. Самый крупный суперпиксель считается фоном.
        
        Алгоритм:
        1. Инициализация центроидов на регулярной сетке
        2. Присвоение каждого пикселя ближайшему центроиду (с учётом компактности)
        3. Обновление центроидов как средних значений присвоенных пикселей
        4. Повторение шагов 2-3 до сходимости или макс. итераций
        5. (Опционально) Принудительная связность регионов
        
        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
        
        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W) — все суперпиксели, кроме фона.
        """
        try:
            # === ПРЕДПОДГОТОВКА ===
            # Конвертируем в numpy для вычислений
            img_np = self._tensor_to_numpy(tensor)  # (H, W, C) или (H, W)
            start_time = time.time()
            
            if len(img_np.shape) == 2:
                # Grayscale → 3 канала для единообразия
                img_np = np.stack([img_np] * 3, axis=-1)
            
            h, w, c = img_np.shape
            
            # === ПАРАМЕТРЫ ===
            n_segments = self.params.get('n_segments', 100)
            compactness = self.params.get('compactness', 10.0)
            max_iter = self.params.get('max_iter', 10)
            sigma = self.params.get('sigma', 0.0)
            enforce_connectivity = self.params.get('enforce_connectivity', True)
            min_size_factor = self.params.get('min_size_factor', 0.5)
            max_size_factor = self.params.get('max_size_factor', 3.0)
            
            # === НОРМАЛИЗАЦИЯ И СГЛАЖИВАНИЕ ===
            # Нормализуем к [0, 1]
            img_float = img_np.astype(np.float32) / 255.0
            
            # Гауссово сглаживание (опционально)
            if sigma > 0:
                from scipy.ndimage import gaussian_filter
                img_float = gaussian_filter(img_float, sigma=sigma, mode='reflect')
            
            # === ПРОСТРАНСТВЕННЫЕ КООРДИНАТЫ ===
            # Создаём сетку координат
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            
            # Нормализуем координаты к масштабу компактности
            # Чем больше compactness, тем больше вес координат
            step = int(max(h, w)) / np.sqrt(n_segments)  # Приблизительный шаг сетки
            coord_scale = compactness / step
            
            # === ОБЪЕДИНЕНИЕ ПРИЗНАКОВ ===
            # Признаки: [цвет * 1.0, координаты * coord_scale]
            features = np.zeros((h, w, c + 2), dtype=np.float32)
            features[..., :c] = img_float  # Цветовые признаки
            features[..., c] = x_coords * coord_scale  # X-координата
            features[..., c + 1] = y_coords * coord_scale  # Y-координата
            
            # === ИНИЦИАЛИЗАЦИЯ ЦЕНТРОИДОВ ===
            # Регулярная сетка центроидов
            grid_y, grid_x = np.mgrid[
                step/2:h:step,
                step/2:w:step
            ]
            grid_y = grid_y.ravel()
            grid_x = grid_x.ravel()
            
            # Ограничиваем число центроидов
            n_centroids = min(n_segments, len(grid_x))
            centroids_init = np.zeros((n_centroids, c + 2), dtype=np.float32)
            
            for i in range(n_centroids):
                y, x = int(grid_y[i]), int(grid_x[i])
                # Начальный центроид = среднее значение в окрестности
                y_start = max(0, int(y - step//2))
                y_end = min(h, int(y + step//2))
                x_start = max(0, int(x - step//2))
                x_end = min(w, int(x + step//2))
                
                region = features[y_start:y_end, x_start:x_end]
                centroids_init[i] = region.reshape(-1, c + 2).mean(axis=0)
            
            # === K-MEANS КЛАСТЕРИЗАЦИЯ ===
            # Расплющиваем признаки для kmeans
            features_flat = features.reshape(-1, c + 2)
            
            from scipy.cluster.vq import kmeans2
            
            # Применяем kmeans2
            try:
                centroids, labels_flat = kmeans2(
                    features_flat,
                    centroids_init,
                    iter=max_iter,
                    minit='matrix',
                    missing='warn'
                )
            except Exception as e:
                warnings.warn(f"K-means failed: {e}. Using fallback initialization.")
                # Fallback: случайная инициализация
                centroids, labels_flat = kmeans2(
                    features_flat,
                    n_centroids,
                    iter=max_iter,
                    minit='++',
                    missing='warn'
                )
            
            # Возвращаем метки в 2D
            labels = labels_flat.reshape(h, w)
            
            # === ПРИНУДИТЕЛЬНАЯ СВЯЗНОСТЬ (опционально) ===
            if enforce_connectivity:
                labels = self._slic_enforce_connectivity(
                    labels, n_segments, min_size_factor, max_size_factor
                )
            
            # === СОЗДАНИЕ МАСКИ ===
            # Находим самый большой сегмент (фон)
            unique, counts = np.unique(labels, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            
            # Создаём маску (все кроме фона)
            mask_np = (labels != bg_label).astype(np.float32)
            
            mask = torch.from_numpy(mask_np).to(self.device)

            exec_time = time.time() - start_time
            info = {
                'method': 'slic_torch',
                'parameters': {
                    'n_segments': n_segments, 
                    'compactness': compactness, 
                    'max_iter': max_iter, 
                    'sigma': sigma, 
                    'enforce_connectivity': enforce_connectivity, 
                    'min_size_factor': min_size_factor,
                    'max_size_factor': max_size_factor,
                    **kwargs
                },
                'execution_time': exec_time,
            }
            return mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            
        except Exception as e:
            warnings.warn(f"SLIC failed: {e}. Using fallback.")
            return self._kmeans_segmentation(tensor)

    def _slic_enforce_connectivity(
        self,
        labels: np.ndarray,
        n_segments: int,
        min_size_factor: float,
        max_size_factor: float,
        **kwargs
    ) -> np.ndarray:
        """
        Принудительная связность регионов (упрощённая реализация).
        
        Args:
            labels: Метки сегментов (H, W)
            n_segments: Желаемое число сегментов
            min_size_factor: Мин. размер сегмента как доля от среднего
            max_size_factor: Макс. размер сегмента как доля от среднего
        
        Returns:
            np.ndarray: Обновлённые метки с обеспеченной связностью
        """
        h, w = labels.shape
        labels_out = labels.copy()
        
        # Вычисляем целевой размер сегмента
        target_size = (h * w) / n_segments
        min_size = int(min_size_factor * target_size)
        max_size = int(max_size_factor * target_size)
        
        # Находим все уникальные метки
        unique_labels = np.unique(labels_out)
        
        # Для каждой метки проверяем связность
        from scipy import ndimage
        
        for label in unique_labels:
            if label < 0:  # Пропускаем шум
                continue
                
            # Бинарная маска для текущей метки
            mask = (labels_out == label).astype(np.uint8)
            
            # Находим связные компоненты
            labeled, num = ndimage.label(mask, structure=np.ones((3, 3)))
            
            if num <= 1:
                # Уже связный регион
                continue
            
            # Находим размеры компонент
            sizes = ndimage.sum(mask, labeled, range(1, num + 1))
            
            # Самая большая компонента остаётся, остальные переназначаем
            main_component = np.argmax(sizes) + 1  # +1 т.к. метки начинаются с 1
            
            # Переназначаем маленькие компоненты соседним меткам
            for comp_id in range(1, num + 1):
                if comp_id == main_component:
                    continue
                
                comp_mask = (labeled == comp_id)
                
                # Находим соседей компоненты
                from scipy.ndimage import binary_dilation
                dilated = binary_dilation(comp_mask, iterations=1)
                neighbors = dilated & ~comp_mask & (labels_out != label)
                
                if neighbors.any():
                    # Присваиваем метку самого частого соседа
                    neighbor_labels = labels_out[neighbors]
                    most_common = np.bincount(neighbor_labels).argmax()
                    labels_out[comp_mask] = most_common
                else:
                    # Если нет соседей, присваиваем ближайшую метку
                    # (упрощённо: берём случайную из уникальных)
                    other_labels = unique_labels[unique_labels != label]
                    if len(other_labels) > 0:
                        labels_out[comp_mask] = np.random.choice(other_labels)
        
        # Перенумеруем метки последовательно
        unique_new = np.unique(labels_out)
        label_map = {old: new for new, old in enumerate(unique_new)}
        labels_out = np.vectorize(label_map.get)(labels_out)
        
        return labels_out

    def _felzenszwalb(
        self,
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Алгоритм Felzenszwalb — иерархическая сегментация на основе графов.
        Графовая сегментация на основе минимального остовного дерева

        Строит сегментацию, начиная с мелких регионов и объединяя их, если внутреннее различие
        меньше межрегионального. Очень эффективен для выделения объектов разного масштаба.

        Args:
            tensor: Входное изображение (B, C, H, W) или (C, H, W).
        
        Returns:
            torch.Tensor: Бинарная маска (1, 1, H, W) — все суперпиксели, кроме фона.
        """
        try:
            # Конвертируем в numpy
            # img_np = self._tensor_to_numpy(tensor)
            # h, w = img_np.shape[:2]
            gray = self._to_grayscale(tensor).squeeze(0).cpu().numpy()
            if gray.max() <= 1.0:
                gray = gray.astype(np.float32)
            else:
                gray = (gray / 255.0).astype(np.float32)
            start_time = time.time()
            
            # Параметры
            scale = self.params.get('scale', 100)
            sigma = self.params.get('sigma', 0.8)
            min_size = self.params.get('min_size', 50)
            
            from skimage.segmentation import felzenszwalb as sk_felzenszwalb
            
            # Применяем Felzenszwalb
            segments = sk_felzenszwalb(
                gray,
                scale=scale,
                sigma=sigma,
                min_size=min_size
            )
            
            # Находим самый большой сегмент (фон)
            unique, counts = np.unique(segments, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            
            # Создаём маску (все кроме фона)
            mask_np = (segments != bg_label).astype(np.float32)
            
            mask = torch.from_numpy(mask_np).to(self.device)

            exec_time = time.time() - start_time
            info = {
                'method': 'felzenszwalb_torch',
                'parameters': {
                    'scale': scale, 
                    'sigma': sigma, 
                    'min_size': min_size, 
                    **kwargs
                },
                'execution_time': exec_time,
            }
            return mask.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            warnings.warn(f"Felzenszwalb failed: {e}. Using fallback.")
            return self._kmeans_segmentation(tensor)
        
    # ============ ИНТЕРАКТИВНЫЕ МЕТОДЫ ============
    class GaussianMixtureModel(nn.Module):
        """GMM для GrabCut реализации"""
        
        def __init__(self, n_components=5):
            super().__init__()
            self.n_components = n_components
            self.means = nn.Parameter(torch.randn(n_components, 3))
            self.covs = nn.Parameter(torch.eye(3).unsqueeze(0).repeat(n_components, 1, 1))
            self.weights = nn.Parameter(torch.ones(n_components) / n_components)
        
        def forward(self, x):
            probs = []
            for i in range(self.n_components):
                dist = MultivariateNormal(self.means[i], self.covs[i])
                probs.append(dist.log_prob(x).exp() * self.weights[i])
            
            return torch.stack(probs, dim=-1).sum(dim=-1)
    
    def _grabcut(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        Интерактивная сегментация GrabCut.

        Использует прямоугольник для инициализации фона и переднего плана.
        Строит модели цветового распределения (GMM) и уточняет границы итеративно.

        Args:
            img: Входное изображение (RGB).

        Returns:
            np.ndarray: Бинарная маска (0/255, dtype=np.uint8) переднего плана.
        """
        try:
            h, w = tensor.shape[2], tensor.shape[3]
            start_time = time.time()
            
            # Initialize mask
            rect = self.params.get('rect', None)
            if rect is None:
                rect = (w//4, h//4, w//2, h//2)  # Default rectangle
            
            mask = torch.zeros(h, w, device=self.device)
            x, y, rw, rh = rect
            mask[y:y+rh, x:x+rw] = 1  # Foreground
            
            # Flatten image for processing
            image_flat = tensor.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            
            # Initialize GMMs
            fg_gmm = self.GaussianMixtureModel().to(self.device)
            bg_gmm = self.GaussianMixtureModel().to(self.device)
            
            # Simple iterative optimization
            num_iterations = self.params.get('num_iterations', 5)
            
            for iteration in range(num_iterations):
                # Get foreground and background pixels
                mask_flat = mask.flatten()
                fg_pixels = image_flat[mask_flat > 0.5]
                bg_pixels = image_flat[mask_flat <= 0.5]
                
                # Simple mean initialization
                if len(fg_pixels) > 0:
                    fg_gmm.means.data = fg_pixels.mean(dim=0).unsqueeze(0).repeat(5, 1)
                if len(bg_pixels) > 0:
                    bg_gmm.means.data = bg_pixels.mean(dim=0).unsqueeze(0).repeat(5, 1)
            
            # Create final segmentation
            fg_probs = fg_gmm(image_flat)
            bg_probs = bg_gmm(image_flat)
            
            final_mask = (fg_probs > bg_probs).float().reshape(h, w)

            exec_time = time.time() - start_time
            info = {
                'method': 'grabcut_torch',
                'parameters': {
                    'rect': rect, 
                    'num_iterations': num_iterations, 
                    **kwargs
                },
                'execution_time': exec_time,
            }
            
            return final_mask
            
        except Exception as e:
            warnings.warn(f"GrabCut failed: {e}")
            h, w = tensor.shape[2], tensor.shape[3]
            return torch.ones(h, w, device=self.device) * 0.5
    
    def _grabcut_torch_visualization(
        self, 
        tensor: torch.Tensor,
        **kwargs
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для GrabCut"""
        try:
            start_time = time.time()
            mask = self._grabcut(tensor)
            
            # Создаем визуализацию
            img_np = self._tensor_to_numpy(tensor)
            mask_np = mask.cpu().numpy()
            
            # Применяем маску к изображению
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
            segmented = img_np * mask_np[:, :, np.newaxis]
            
            # Смешиваем с оригиналом
            alpha = 0.7
            result = (img_np * (1 - alpha) + segmented * alpha).astype(np.uint8)
            
            # Конвертируем маску в 0-255
            if mask_np.max() <= 1.0:
                mask_np = (mask_np * 255).astype(np.uint8)
            
            mask_tensor = torch.from_numpy(mask_np).float().to(self.device)

            exec_time = time.time() - start_time
            info = {
                'method': 'grabcut_visualisation_torch',
                'parameters': { 
                    **kwargs
                },
                'execution_time': exec_time,
            }
            
            return result, mask_tensor
            
        except Exception as e:
            warnings.warn(f"GrabCut visualization failed: {e}")
            mask = self._grabcut(tensor)
            img_np = self._tensor_to_numpy(tensor)
            return img_np, mask