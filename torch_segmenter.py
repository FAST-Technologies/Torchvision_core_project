from base_segmenter import BaseSegmenter
import torch
import torch.nn.functional as F
from torch import nn
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image
from typing import Union, Tuple

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
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
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
            return self._pil_to_tensor(img)
        elif isinstance(image, Image.Image):
            # PIL Image
            return self._pil_to_tensor(image)
        elif isinstance(image, np.ndarray):
            # NumPy array
            img = Image.fromarray(image).convert('RGB')
            return self._pil_to_tensor(img)
        elif isinstance(image, torch.Tensor):
            # PyTorch tensor
            return image.to(self.device)
        else:
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")
    
    def _pil_to_tensor(self, 
                       img: Image.Image
    ) -> torch.Tensor:
        """Преобразование PIL Image в PyTorch tensor"""
        transform = transforms.ToTensor()
        tensor = transform(img).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        return tensor
    
    def _tensor_to_numpy(self, 
                         tensor: torch.Tensor
    ) -> np.ndarray:
        """Преобразование PyTorch tensor в NumPy array"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        return tensor.permute(1, 2, 0).cpu().numpy()
    
    def _to_grayscale(self, 
                      tensor: torch.Tensor
    ) -> torch.Tensor:
        """Преобразование RGB в градации серого"""
        if tensor.shape[1] == 3:
            gray = torch.mean(tensor, dim=1, keepdim=True)
        else:
            gray = tensor
        return gray
    
    def segment(self, 
                image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """Сегментация изображения"""
        tensor = self.preprocess_image(image)
        mask_tensor = self._segment_func(tensor)
        
        # Преобразуем маску в numpy
        if mask_tensor.dim() == 4:
            mask_tensor = mask_tensor.squeeze(0)
        if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.squeeze(0)
        
        mask_np = mask_tensor.cpu().numpy()
        
        # Нормализация к 0-255
        if mask_np.dtype == bool:
            mask_np = mask_np.astype(np.uint8) * 255
        elif mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
        
        return mask_np
    
    def segment_with_mask(self, 
                          image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Сегментация с возвратом маски и обработанного изображения"""
        tensor = self.preprocess_image(image)
        mask_tensor = self._segment_func(tensor)
        
        # Преобразуем маску в numpy
        if mask_tensor.dim() == 4:
            mask_tensor = mask_tensor.squeeze(0)
        if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.squeeze(0)
        
        mask_np = mask_tensor.cpu().numpy()
        
        # Нормализация маски
        if mask_np.dtype == bool:
            mask_np_uint8 = mask_np.astype(np.uint8) * 255
        elif mask_np.max() <= 1.0:
            mask_np_uint8 = (mask_np * 255).astype(np.uint8)
        else:
            mask_np_uint8 = mask_np.astype(np.uint8)
        
        # Преобразуем исходное изображение в numpy для визуализации
        img_np = self._tensor_to_numpy(tensor)
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Создаем визуализацию
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        overlay = img_np.copy()
        mask_bool = mask_np_uint8 > 0
        overlay[mask_bool] = [255, 0, 0]
        result = cv2.addWeighted(img_np, 0.5, overlay, 0.5, 0)
        
        return result, mask_np_uint8
    
    # ============ РЕАЛИЗАЦИИ МЕТОДОВ ============
    
    def _global_thresholding(self, 
                             tensor: torch.Tensor
    ) -> torch.Tensor:
        """Глобальная пороговая обработка (PyTorch)"""
        gray = self._to_grayscale(tensor)
        threshold = self.params.get('threshold', 0.5)
        mask = (gray > threshold).float()
        return mask
    
    def _adaptive_thresholding(self, 
                               tensor: torch.Tensor
    ) -> torch.Tensor:
        """Адаптивная пороговая обработка (PyTorch)"""
        gray = self._to_grayscale(tensor)
        block_size = self.params.get('block_size', 11)
        c = self.params.get('C', 2)
        
        kernel = torch.ones(1, 1, block_size, block_size).to(self.device) / (block_size * block_size)
        local_mean = F.conv2d(gray, kernel, padding=block_size//2)
        mask = (gray > (local_mean - c)).float()
        return mask
    
    def _otsu_thresholding(self, 
                           tensor: torch.Tensor
    ) -> torch.Tensor:
        """Метод Оцу (PyTorch)"""
        gray = self._to_grayscale(tensor).squeeze().flatten()
        
        hist = torch.histc(gray, bins=256, min=0, max=1)
        total = hist.sum()
        cumsum = torch.cumsum(hist, dim=0)
        mean = torch.arange(256, dtype=torch.float32).to(self.device) / 255
        
        w0 = cumsum
        w1 = total - w0
        m0 = torch.cumsum(hist * mean, dim=0) / (w0 + 1e-8)
        m1 = (torch.cumsum(hist * mean, dim=0)[-1] - torch.cumsum(hist * mean, dim=0)) / (w1 + 1e-8)
        var_between = w0 * w1 * (m0 - m1)**2
        
        best_threshold_idx = torch.argmax(var_between)
        threshold = best_threshold_idx.float() / 255.0
        
        mask = (self._to_grayscale(tensor) > threshold).float()
        return mask
    
    def _region_growing(self, 
                        tensor: torch.Tensor
    ) -> torch.Tensor:
        """Region Growing (PyTorch)"""
        from collections import deque
        
        gray = self._to_grayscale(tensor).squeeze(0)  # (H, W)
        h, w = gray.shape
        
        seed = self.params.get('seed', (100, 100))
        tolerance = self.params.get('tolerance', 0.1)
        
        mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        queue = deque([(seed[1], seed[0])])
        seed_value = gray[seed[1], seed[0]]
        
        while queue:
            y, x = queue.popleft()
            if 0 <= y < h and 0 <= x < w and not mask[y, x]:
                if abs(gray[y, x] - seed_value) <= tolerance:
                    mask[y, x] = True
                    for dy, dx in [(1,0), (-1,0), (0,1), (0,-1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx]:
                            queue.append((ny, nx))
        
        return mask.float()
    
    def _split_and_merge(self, 
                         tensor: torch.Tensor
    ) -> torch.Tensor:
        """Split-and-Merge (PyTorch)"""
        h, w = tensor.shape[2], tensor.shape[3]
        min_size = self.params.get('min_size', 50)
        
        if min_size > h or min_size > w:
            min_size = min(h, w) // 2
        
        patches = tensor.unfold(2, min_size, min_size).unfold(3, min_size, min_size)
        patches = patches.reshape(-1, 3, min_size, min_size)
        
        means = patches.mean(dim=[2, 3])
        unique_colors, counts = torch.unique(means, dim=0, return_counts=True)
        bg_color = unique_colors[torch.argmax(counts)]
        
        mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        for i in range(patches.size(0)):
            y = (i // (w // min_size)) * min_size
            x = (i % (w // min_size)) * min_size
            if not torch.allclose(patches[i], bg_color, atol=0.1):
                mask[y:y+min_size, x:x+min_size] = True
        
        return mask.float()
    
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
        from sklearn.cluster import DBSCAN
        
        eps = self.params.get('eps', 10)
        min_samples = self.params.get('min_samples', 100)
        
        h, w = tensor.shape[2], tensor.shape[3]
        pixels = tensor.squeeze(0).permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
        labels = torch.from_numpy(db.labels_).view(h, w).to(self.device)
        
        mask = (labels != -1) & (labels != 0)
        return mask.float()
    
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
        """Watershed (PyTorch)"""
        from scipy import ndimage
        
        gray = self._to_grayscale(tensor).squeeze(0).cpu().numpy()
        
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        grad_x = ndimage.convolve(gray, sobel_x)
        grad_y = ndimage.convolve(gray, sobel_y)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        threshold = np.mean(gray)
        binary = (gray > threshold).astype(np.uint8)
        distance = ndimage.distance_transform_edt(binary)
        markers = ndimage.label(distance > distance.max() * 0.5)[0]
        
        mask = torch.from_numpy(markers > 0).float().to(self.device)
        return mask
    
    def _meanshift(self, 
                   tensor: torch.Tensor
    ) -> torch.Tensor:
        """MeanShift (PyTorch)"""
        from sklearn.cluster import MeanShift
        
        h, w = tensor.shape[2], tensor.shape[3]
        pixels = tensor.squeeze(0).permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        
        bandwidth = self.params.get('bandwidth', 0.5)
        meanshift = MeanShift(bandwidth=bandwidth)
        labels = meanshift.fit_predict(pixels)
        
        mask = torch.from_numpy(labels.reshape(h, w)).float().to(self.device)
        return mask
    
    def _grabcut(self, 
                 tensor: torch.Tensor
    ) -> torch.Tensor:
        """GrabCut (PyTorch)"""
        h, w = tensor.shape[2], tensor.shape[3]
        
        rect = self.params.get('rect', (w//4, h//4, w//2, h//2))
        x, y, rw, rh = rect
        
        mask = torch.zeros(h, w, device=self.device)
        mask[y:y+rh, x:x+rw] = 1
        
        return mask.float()
    
    def _floodfill(self, 
                   tensor: torch.Tensor
    ) -> torch.Tensor:
        """FloodFill (PyTorch)"""
        from collections import deque
        
        h, w = tensor.shape[2], tensor.shape[3]
        
        seed = self.params.get('seed', (w//2, h//2))
        tolerance = self.params.get('tolerance', 0.1)
        
        gray = self._to_grayscale(tensor).squeeze(0)
        visited = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        mask = torch.zeros(h, w, dtype=torch.bool, device=self.device)
        
        start_x, start_y = int(seed[0]), int(seed[1])
        target_color = gray[start_y, start_x]
        
        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        mask[start_y, start_x] = True
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    pixel_color = gray[ny, nx]
                    color_diff = torch.abs(pixel_color - target_color)
                    
                    if color_diff <= tolerance:
                        visited[ny, nx] = True
                        mask[ny, nx] = True
                        queue.append((nx, ny))
        
        return mask.float()