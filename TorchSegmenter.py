# TorchSegmenter.py
from BaseSegmenter import BaseSegmenter
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import functional as TF
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple, Dict, Any, List
from collections import deque
from sklearn.cluster import MeanShift, KMeans, DBSCAN
import warnings
import time
from scipy import ndimage
from sklearn.cluster import MeanShift as SkMeanShift

class TorchSegmenter(BaseSegmenter):
    """Класс для методов сегментации с использованием PyTorch"""
    
    def __init__(self, 
                 method: str = "global_thresholding", 
                 device: str = None, 
                 **kwargs
    ) -> None:
        super().__init__()
        self.method = method
        self.params = kwargs
        self.model_name = f"Torch_{method}"

        self._needs_normalization = method in [
            "global_thresholding",    # Работает с [0, 1]
            "adaptive_thresholding",  # Работает с [0, 1]
            "otsu_thresholding",      # Работает с [0, 1]
            "sobel_edge",             # Работает с [0, 1]
            "canny_edge",             # Работает с [0, 1]
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
    
    def _setup_method(self) -> None:
        """Настройка выбранного метода"""
        method_map = {
            "global_thresholding": self._global_thresholding,
            "adaptive_thresholding": self._adaptive_thresholding,
            "otsu_thresholding": self._otsu_thresholding,
            "region_growing": self._region_growing,
            "split_and_merge": self._split_and_merge,
            "sobel_edge": self._sobel_edge,
            "canny_edge": self._canny_edge,
            "kmeans_segmentation": self._kmeans_segmentation,
            "dbscan_segmentation": self._dbscan_segmentation,
            "active_contour": self._active_contour,
            "gvf_contour": self._gvf_contour,
            "watershed": self._watershed,
            "meanshift": self._meanshift,
            "grabcut": self._grabcut,
            "floodfill": self._floodfill
        }
        
        if self.method not in method_map:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        self._segment_func = method_map[self.method]
    
    def preprocess_image(self, 
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
        elif isinstance(image, torch.Tensor):
            # PyTorch tensor
            return image.to(self.device)
        else:
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")
    
    def _pil_to_tensor(self, 
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
    
    def _tensor_to_numpy(self, 
                         tensor: torch.Tensor,
                         denormalize: bool = True
    ) -> np.ndarray:
        """Преобразование PyTorch tensor в NumPy array"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        result = tensor.permute(1, 2, 0).cpu().numpy()
        
        if denormalize and result.max() <= 1.0:
            result = (result * 255).astype(np.uint8)
        
        return result
    
    def _tensor_to_pil(self, 
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
    
    def _to_grayscale(self, 
                      tensor: torch.Tensor
    ) -> torch.Tensor:
        """Преобразование RGB в градации серого"""
        if tensor.shape[1] == 3:
            # gray = torch.mean(tensor, dim=1, keepdim=True) # (B, 1, H, W)
            gray = 0.2989 * tensor[:, 0:1, :, :] + 0.5870 * tensor[:, 1:2, :, :] + 0.1140 * tensor[:, 2:3, :, :]
        else:
            gray = tensor
        return gray
    
    def _normalize_to_255(self,
                          img: Image.Image | np.ndarray
    ) -> Image.Image | np.ndarray:
        """Метод нормализации изобраажения"""
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        return img
    
    def _normalize_tensor(self,
                          tensor: torch.Tensor
    ) -> torch.Tensor:
        """Нормализация тензора к [0, 1]"""
        min_val: torch.Tensor = tensor.min()
        max_val: torch.Tensor = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)
    
    def segment(self, 
                image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """Сегментация изображения - возвращает маску 0-255"""
        try:
            tensor = self.preprocess_image(image)
            mask_tensor = self._segment_func(tensor)
            
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
            
            return mask_np
            
        except Exception as e:
            warnings.warn(f"Ошибка в методе {self.method}: {e}")
            import traceback
            traceback.print_exc()
            # Возвращаем пустую маску в случае ошибки
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
    
    def segment_with_mask(self, 
                          image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Сегментация с возвратом маски (0-255) и визуализации"""
        try:
            tensor = self.preprocess_image(image)
            result_vis, mask_tensor = self._segment_with_visualization(tensor)
            
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
            
            return result_np, mask_np
            
        except Exception as e:
            warnings.warn(f"Ошибка в методе {self.method} (segment_with_mask): {e}")
            import traceback
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
        
    def _segment_with_visualization(self, 
                            tensor: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, np.ndarray], torch.Tensor]:
        """Сегментация с визуализацией для конкретного метода"""
        if self.method == "watershed":
            return self._watershed_torch_visualization(tensor)
        elif self.method == "meanshift":
            return self._meanshift_torch_visualization(tensor)
        elif self.method == "grabcut":
            return self._grabcut_torch_visualization(tensor)
        elif self.method == "floodfill":
            return self._floodfill_torch_visualization(tensor)
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
    
    def _global_thresholding(self, 
                             tensor: torch.Tensor
    ) -> torch.Tensor:
        """Глобальная пороговая обработка (PyTorch)"""
        gray = self._to_grayscale(tensor) # (1, 1, H, W)
        threshold = self.params.get('threshold', 0.5)
        # if gray.max() > 1.0:
        #     threshold = threshold * 255  # Конвертируем 0.5 → 127.5
        mask = (gray > threshold).float()
        return mask
    
    def _adaptive_thresholding(self, 
                               tensor: torch.Tensor
    ) -> torch.Tensor:
        """Адаптивная пороговая обработка (PyTorch)"""
        gray = self._to_grayscale(tensor) # (1,1,H,W)
        block_size = self.params.get('block_size', 11)
        c = self.params.get('C', 2)

        if block_size % 2 == 0:
            block_size += 1
        
        kernel = torch.ones(1, 1, block_size, block_size).to(self.device) / (block_size * block_size)
        local_mean = F.conv2d(gray, kernel, padding=block_size//2)
        mask = (gray > (local_mean - c/255.0)).float()
        return mask
    
    def _otsu_thresholding(self, 
                           tensor: torch.Tensor
    ) -> torch.Tensor:
        """Метод Оцу (PyTorch)"""
        gray = self._to_grayscale(tensor).squeeze()

        gray_np = gray.cpu().numpy()
        if gray_np.max() <= 1.0:
            gray_np = (gray_np * 255).astype(np.uint8)
        else:
            gray_np = gray_np.astype(np.uint8)
        
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
        return mask.unsqueeze(0).unsqueeze(0)
    
    def _region_growing(self, 
                        tensor: torch.Tensor
    ) -> torch.Tensor:
        """Region Growing (PyTorch)"""
        from collections import deque
        
        gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
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
    def _split_and_merge(self, 
                         tensor: torch.Tensor
    ) -> torch.Tensor:
        """Split-and-Merge (PyTorch) - упрощенная реализация"""
        try:
            # Преобразуем в numpy для более простой реализации
            img_np = self._tensor_to_numpy(tensor)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
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
            
            min_size = self.params.get('min_size', 50)
            threshold = self.params.get('threshold', 20)
            
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
            return mask.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            warnings.warn(f"Split-and-merge failed: {e}. Using fallback.")
            return self._kmeans_segmentation(tensor)
    
    def _sobel_edge(self, 
                    tensor: torch.Tensor
    ) -> torch.Tensor:
        """Оператор Собеля (PyTorch)"""
        gray = self._to_grayscale(tensor)
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
        
        return mask
    
    def _canny_edge(self, 
                    tensor: torch.Tensor
    ) -> torch.Tensor:
        """Оператор Кэнни (PyTorch)"""
        gray = self._to_grayscale(tensor)
        low = self.params.get('low', 0.1)
        high = self.params.get('high', 0.3)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        mag = torch.sqrt(gx**2 + gy**2)
        angle = torch.atan2(gy, gx) * 180 / np.pi
        
        suppressed = mag.clone()
        suppressed[(angle > -22.5) & (angle <= 22.5)] = 0
        suppressed[(angle > 22.5) & (angle <= 67.5)] = 0
        suppressed[(angle > 67.5) & (angle <= 112.5)] = 0
        suppressed[(angle > 112.5) & (angle <= 157.5)] = 0
        
        mask = (mag > high).float()
        weak = ((mag > low) & (mag <= high)).float()
        mask = mask + weak * (mask > 0).float()
        
        return mask
    
    def _kmeans_segmentation(self, 
                             tensor: torch.Tensor
    ) -> torch.Tensor:
        """K-Means кластеризация (PyTorch)"""
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
        
        return mask.float()
    
    def _dbscan_segmentation(self, 
                             tensor: torch.Tensor
    ) -> torch.Tensor:
        """DBSCAN кластеризация (PyTorch)"""
        try:
            # Преобразуем в numpy для sklearn DBSCAN
            img_np = self._tensor_to_numpy(tensor)
            h, w = img_np.shape[:2]
            
            # Уменьшаем разрешение для скорости
            scale = 0.5
            if h * w > 100000:
                small_h, small_w = int(h * scale), int(w * scale)
                img_small = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
                pixels = img_small.reshape(-1, 3)
            else:
                pixels = img_np.reshape(-1, 3)
            
            eps = self.params.get('eps', 10)
            min_samples = self.params.get('min_samples', 100)
            
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
            
            return mask.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            warnings.warn(f"DBSCAN failed: {e}. Using fallback.")
            return self._kmeans_segmentation(tensor)
    
    def _active_contour(self, 
                        tensor: torch.Tensor
    ) -> torch.Tensor:
        """Active Contour (PyTorch)"""
        gray = self._to_grayscale(tensor).squeeze(0)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        gx = F.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_x, padding=1).squeeze()
        gy = F.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_y, padding=1).squeeze()
        mag = torch.sqrt(gx**2 + gy**2)
        mask = (mag > 0.1).float()
        
        return mask
    
    def _gvf_contour(self, 
                     tensor: torch.Tensor
    ) -> torch.Tensor:
        """GVF (PyTorch)"""
        gray = self._to_grayscale(tensor).squeeze(0)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        gx = F.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_x, padding=1).squeeze()
        gy = F.conv2d(gray.unsqueeze(0).unsqueeze(0), sobel_y, padding=1).squeeze()
        
        kernel = torch.ones(1, 1, 5, 5, device=self.device) / 25
        gx_smooth = F.conv2d(gx.unsqueeze(0).unsqueeze(0), kernel, padding=2).squeeze()
        gy_smooth = F.conv2d(gy.unsqueeze(0).unsqueeze(0), kernel, padding=2).squeeze()
        
        mag = torch.sqrt(gx_smooth**2 + gy_smooth**2)
        mask = (mag > 0.1).float()
        
        return mask
    
    def _watershed(self, 
                         tensor: torch.Tensor
    ) -> torch.Tensor:
        """Watershed сегментация (PyTorch реализация)"""
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
    
    def _watershed_segmentation_torch(self, 
                                      tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Вспомогательная функция для Watershed"""
        # Convert to grayscale if needed
        if tensor.shape[1] == 3:
            grayscale = self._to_grayscale(tensor)
        else:
            grayscale = tensor
        
        # Compute gradients (edge detection)
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        grad_x = F.conv2d(grayscale, kernel_x, padding=1)
        grad_y = F.conv2d(grayscale, kernel_y, padding=1)
        
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Get markers if not provided
        markers = self.params.get('markers', None)
        
        if markers is None:
            # Simple thresholding to get foreground/background
            threshold = torch.mean(grayscale)
            binary = (grayscale > threshold).float()
            
            # Distance transform using scipy
            binary_np = binary.squeeze().cpu().numpy()
            distance = ndimage.distance_transform_edt(binary_np)
            markers_np = ndimage.label(distance > distance.max() * 0.5)[0]
            markers = torch.from_numpy(markers_np).float().to(self.device)
        
        return gradient_magnitude.squeeze(), markers
    
    def _watershed_torch_visualization(self, 
                                       tensor: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для Watershed"""
        # Получаем градиент и маркеры
        gradient, markers = self._watershed_segmentation_torch(tensor)
        
        # Создаем цветную визуализацию маркеров
        img_np = self._tensor_to_numpy(tensor)
        markers_np = markers.cpu().numpy()
        
        # Нормализуем маркеры для визуализации
        if markers_np.max() > 0:
            markers_np = (markers_np / markers_np.max() * 255).astype(np.uint8)
        
        # Создаем цветную карту маркеров
        from matplotlib.cm import tab20
        cmap = tab20
        markers_colored = cmap(markers_np % 20)[:, :, :3]
        markers_colored = (markers_colored * 255).astype(np.uint8)
        
        # Смешиваем с оригиналом
        alpha = 0.6
        result = (img_np * (1 - alpha) + markers_colored * alpha).astype(np.uint8)
        
        # Маска - все ненулевые маркеры
        mask = (markers > 0).float()
        
        return result, mask
    
    # def _meanshift(self, 
    #                tensor: torch.Tensor
    # ) -> torch.Tensor:
    #     """MeanShift (PyTorch)"""
    #     from sklearn.cluster import MeanShift
        
    #     h, w = tensor.shape[2], tensor.shape[3]
    #     pixels = tensor.squeeze(0).permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        
    #     bandwidth = self.params.get('bandwidth', 0.5)
    #     meanshift = MeanShift(bandwidth=bandwidth)
    #     labels = meanshift.fit_predict(pixels)
        
    #     mask = torch.from_numpy(labels.reshape(h, w)).float().to(self.device)
    #     return mask
    # def _meanshift(self, 
    #                tensor: torch.Tensor
    # ) -> torch.Tensor:
    #     """MeanShift сегментация - используем проверенную реализацию"""
    #     from collections import deque
        
    #     try:
    #         # Преобразуем в numpy для обработки
    #         img_np = self._tensor_to_numpy(tensor)
    #         if img_np.max() <= 1.0:
    #             img_np = (img_np * 255).astype(np.uint8)
            
    #         h, w = img_np.shape[:2]
            
    #         # Используем проверенную реализацию MeanShift
    #         class MeanShiftTorch:
    #             def __init__(self, bandwidth=0.5, max_iter=300):
    #                 self.bandwidth = bandwidth
    #                 self.max_iter = max_iter
    #                 # self.clustering = SkMeanShift(bandwidth=bandwidth, max_iter=max_iter)
                
    #             def segment_image(self, image_np, spatial_radius=35, color_radius=60):
    #                 h, w, c = image_np.shape
                    
    #                 # Создаем пространство признаков: координаты + цвет
    #                 y_coords, x_coords = np.mgrid[0:h, 0:w]
    #                 spatial_features = np.stack([x_coords / spatial_radius,
    #                                            y_coords / spatial_radius], axis=-1)
                    
    #                 # Нормализуем цветовые признаки
    #                 color_features = image_np / color_radius
                    
    #                 # Объединяем признаки
    #                 features = np.concatenate([spatial_features, color_features], axis=-1)
    #                 features_flat = features.reshape(-1, features.shape[-1])
                    
    #                 # Применяем MeanShift
    #                 try:
    #                     meanshift = MeanShift(bandwidth=self.bandwidth, n_jobs=-1, max_iter=self.max_iter)
    #                     labels = meanshift.fit_predict(features_flat)
    #                     labels_2d = labels.reshape(h, w)
    #                 except:
    #                     # Если не хватает памяти, используем KMeans
    #                     from sklearn.cluster import KMeans
    #                     kmeans = KMeans(n_clusters=5, random_state=42)
    #                     labels = kmeans.fit_predict(features_flat)
    #                     labels_2d = labels.reshape(h, w)
                    
    #                 return labels_2d
            
    #         # Параметры
    #         bandwidth = self.params.get('bandwidth', 0.5)
    #         spatial_radius = self.params.get('spatial_radius', 35)
    #         color_radius = self.params.get('color_radius', 60)
            
    #         # Применяем MeanShift
    #         meanshift = MeanShiftTorch(bandwidth=bandwidth)
    #         labels = meanshift.segment_image(img_np, spatial_radius, color_radius)
            
    #         # Находим самый большой кластер (предположительно фон)
    #         unique, counts = np.unique(labels, return_counts=True)
    #         bg_label = unique[np.argmax(counts)]
            
    #         # Создаем маску
    #         mask_np = (labels != bg_label).astype(np.float32)
    #         mask = torch.from_numpy(mask_np).to(self.device)
            
    #         return mask.unsqueeze(0).unsqueeze(0)
            
    #     except Exception as e:
    #         warnings.warn(f"MeanShift failed: {e}. Using fallback.")
    #         return self._kmeans_segmentation(tensor)
    def _meanshift(self, 
                         tensor: torch.Tensor
    ) -> torch.Tensor:
        """MeanShift сегментация (PyTorch реализация)"""
        try:
            # Преобразуем в numpy для обработки
            img_np = self._tensor_to_numpy(tensor)
            h, w = img_np.shape[:2]
            
            # Используем MeanShift из sklearn
            bandwidth = self.params.get('bandwidth', 0.5)
            spatial_radius = self.params.get('spatial_radius', 35)
            color_radius = self.params.get('color_radius', 60)
            
            # Создаем пространство признаков
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            spatial_features = np.stack([x_coords / spatial_radius,
                                       y_coords / spatial_radius], axis=-1)
            
            # Нормализуем цветовые признаки
            color_features = img_np / color_radius
            
            # Объединяем признаки
            features = np.concatenate([spatial_features, color_features], axis=-1)
            features_flat = features.reshape(-1, features.shape[-1])
            
            # Применяем MeanShift
            meanshift = SkMeanShift(bandwidth=bandwidth, max_iter=100, n_jobs=-1)
            labels = meanshift.fit_predict(features_flat)
            labels_2d = labels.reshape(h, w)
            
            # Находим самый большой кластер (предположительно фон)
            unique, counts = np.unique(labels, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            
            # Создаем маску
            mask_np = (labels_2d != bg_label).astype(np.float32)
            mask = torch.from_numpy(mask_np).to(self.device)
            
            return mask
            
        except Exception as e:
            warnings.warn(f"MeanShift failed: {e}")
            return self._kmeans_segmentation(tensor)
    
    def _meanshift_torch_visualization(self, 
                                       tensor: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для MeanShift"""
        try:
            # Преобразуем в numpy для обработки
            img_np = self._tensor_to_numpy(tensor)
            h, w, c = img_np.shape
            
            # Используем MeanShift
            bandwidth = self.params.get('bandwidth', 0.5)
            spatial_radius = self.params.get('spatial_radius', 35)
            color_radius = self.params.get('color_radius', 60)
            
            # Создаем пространство признаков
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            spatial_features = np.stack([x_coords / spatial_radius,
                                       y_coords / spatial_radius], axis=-1)
            
            color_features = img_np / color_radius
            features = np.concatenate([spatial_features, color_features], axis=-1)
            features_flat = features.reshape(-1, features.shape[-1])
            
            # Применяем MeanShift
            meanshift = SkMeanShift(bandwidth=bandwidth, max_iter=100, n_jobs=-1)
            labels = meanshift.fit_predict(features_flat)
            labels_2d = labels.reshape(h, w)
            
            # Создаем сегментированное изображение
            segmented = np.zeros_like(img_np)
            unique_labels = np.unique(labels_2d)
            
            for label in unique_labels:
                mask = labels_2d == label
                if np.any(mask):
                    segmented[mask] = np.mean(img_np[mask], axis=0)
            
            # Смешиваем с оригиналом
            alpha = 0.6
            result = (img_np * (1 - alpha) + segmented * alpha).astype(np.uint8)
            
            # Находим фон и создаем маску
            unique, counts = np.unique(labels, return_counts=True)
            bg_label = unique[np.argmax(counts)]
            mask_np = (labels_2d != bg_label).astype(np.float32)
            mask = torch.from_numpy(mask_np).to(self.device)
            
            return result, mask
            
        except Exception as e:
            warnings.warn(f"MeanShift visualization failed: {e}")
            mask = self._meanshift(tensor)
            img_np = self._tensor_to_numpy(tensor)
            return img_np, mask
        
    # GMM для GrabCut
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
    
    def _grabcut(self, 
                       tensor: torch.Tensor
    ) -> torch.Tensor:
        """GrabCut сегментация (упрощенная PyTorch реализация)"""
        try:
            h, w = tensor.shape[2], tensor.shape[3]
            
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
            
            return final_mask
            
        except Exception as e:
            warnings.warn(f"GrabCut failed: {e}")
            h, w = tensor.shape[2], tensor.shape[3]
            return torch.ones(h, w, device=self.device) * 0.5
    
    def _grabcut_torch_visualization(self, 
                                     tensor: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для GrabCut"""
        try:
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
            
            return result, mask_tensor
            
        except Exception as e:
            warnings.warn(f"GrabCut visualization failed: {e}")
            mask = self._grabcut(tensor)
            img_np = self._tensor_to_numpy(tensor)
            return img_np, mask
    
    # def _floodfill(self, 
    #                tensor: torch.Tensor
    # ) -> torch.Tensor:
    #     """FloodFill (PyTorch)"""
    #     from collections import deque
        
    #     h, w = tensor.shape[2], tensor.shape[3]
        
    #     seed = self.params.get('seed', (w//2, h//2))
    #     tolerance = self.params.get('tolerance', 0.1)
        
    #     gray = self._to_grayscale(tensor).squeeze(0)
    #     visited = torch.zeros(h, w, dtype=torch.bool, device=self.device)
    #     mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        
    #     start_x, start_y = int(seed[0]), int(seed[1])
    #     target_color = gray[start_y, start_x]
        
    #     queue = deque([(start_x, start_y)])
    #     visited[start_y, start_x] = True
    #     mask[start_y, start_x] = True
        
    #     directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
    #     while queue:
    #         x, y = queue.popleft()
            
    #         for dx, dy in directions:
    #             nx, ny = x + dx, y + dy
                
    #             if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
    #                 pixel_color = gray[ny, nx]
    #                 color_diff = torch.abs(pixel_color - target_color)
                    
    #                 if color_diff <= tolerance:
    #                     visited[ny, nx] = True
    #                     mask[ny, nx] = True
    #                     queue.append((nx, ny))
        
    #     return mask.float()
    def _floodfill(self, 
                   tensor: torch.Tensor
    ) -> torch.Tensor:
        """FloodFill сегментация (PyTorch реализация)"""
        try:
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
            
            return mask
            
        except Exception as e:
            warnings.warn(f"FloodFill failed: {e}")
            return self._region_growing(tensor)
    
    def _floodfill_torch_visualization(self, 
                                       tensor: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Визуализация для FloodFill"""
        try:
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
            
            return result_np, mask_tensor
            
        except Exception as e:
            warnings.warn(f"FloodFill visualization failed: {e}")
            mask = self._floodfill(tensor)
            img_np = self._tensor_to_numpy(tensor)
            return img_np, mask
    
    def _flood_fill_single(self, 
                           tensor: torch.Tensor, 
                           start_point: Tuple[int, int], 
                           tolerance: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FloodFill из одной точки"""
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
        
        return mask
    
    def _multi_point_floodfill(self, 
                               tensor: torch.Tensor, 
                               points: List[Tuple[int, int]], 
                               tolerance: float = 0.1
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