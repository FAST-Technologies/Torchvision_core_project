"""
Трекер аллокаций памяти для детектирования утечек.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import gc
import warnings


@dataclass
class AllocationSnapshot:
    """Снимок состояния аллокаций."""

    timestamp: float
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    tensor_count: int
    gc_counts: Tuple[int, int, int]  # Сборки мусора по поколениям

    @classmethod
    def capture(cls, device: torch.device) -> "AllocationSnapshot":
        """Захватывает текущее состояние."""
        allocated = reserved = peak = 0.0

        if device.type == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1e6
            reserved = torch.cuda.memory_reserved(device) / 1e6
            peak = torch.cuda.max_memory_allocated(device) / 1e6

        # Счётчик тензоров
        tensor_count = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))

        return cls(
            timestamp=time.time(),
            allocated_mb=allocated,
            reserved_mb=reserved,
            peak_mb=peak,
            tensor_count=tensor_count,
            gc_counts=gc.get_count(),
        )


class AllocationTracker:
    """
    Трекер аллокаций для детектирования утечек и анализа паттернов.

    Поддерживает:
    - Периодические снимки состояния
    - Детектирование роста потребления
    - Анализ тензоров-кандидатов на утечку
    """

    def __init__(
        self,
        device: str = "cuda",
        snapshot_interval_sec: float = 1.0,
        max_snapshots: int = 100,
    ):
        self.device = torch.device(device)
        self.snapshot_interval = snapshot_interval_sec
        self.max_snapshots = max_snapshots

        self._snapshots: List[AllocationSnapshot] = []
        self._tracking: bool = False
        self._start_time: Optional[float] = None

        # Трек тензоров для анализа утечек
        self._tensor_registry: Dict[int, Dict[str, Any]] = {}

    def start_tracking(self):
        """Начинает отслеживание аллокаций."""
        self._tracking = True
        self._start_time = time.time()
        self._snapshots.clear()

    def stop_tracking(self) -> List[AllocationSnapshot]:
        """Останавливает отслеживание и возвращает снимки."""
        self._tracking = False
        return self._snapshots

    def capture_snapshot(self) -> AllocationSnapshot:
        """Захватывает и сохраняет снимок."""
        if not self._tracking:
            warnings.warn("Tracking not started. Call start_tracking() first.")

        snapshot = AllocationSnapshot.capture(self.device)
        self._snapshots.append(snapshot)

        # Ограничение размера
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

        return snapshot

    def detect_leak(
        self,
        window_size: int = 10,
        threshold_mb: float = 10.0,
        threshold_pct: float = 0.1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Детектирует утечку по последним снимкам.

        Args:
            window_size: Размер окна анализа
            threshold_mb: Абсолютный порог роста (МБ)
            threshold_pct: Относительный порог роста

        Returns:
            Tuple[bool, Dict]: (есть ли утечка, детали)
        """
        if len(self._snapshots) < window_size:
            return False, {"reason": "insufficient_snapshots"}

        recent = self._snapshots[-window_size:]
        start_alloc = recent[0].allocated_mb
        end_alloc = recent[-1].allocated_mb

        diff_mb = end_alloc - start_alloc
        diff_pct = diff_mb / start_alloc if start_alloc > 0 else 0

        is_leak = diff_mb > threshold_mb or diff_pct > threshold_pct

        return is_leak, {
            "window_size": window_size,
            "start_mb": start_alloc,
            "end_mb": end_alloc,
            "diff_mb": diff_mb,
            "diff_pct": diff_pct * 100,
            "threshold_mb": threshold_mb,
            "threshold_pct": threshold_pct * 100,
        }

    def analyze_tensors(self) -> Dict[str, Any]:
        """
        Анализирует тензоры в памяти для поиска кандидатов на утечку.

        Returns:
            Dict с анализом
        """
        tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_leaf]

        # Группировка по устройству и размеру
        by_device: Dict[str, List[torch.Tensor]] = defaultdict(list)
        by_size: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for t in tensors:
            by_device[str(t.device)].append(t)

            size_mb = (t.element_size() * t.nelement()) / 1024**2
            if size_mb > 10:
                by_size["large"].append(t)
            elif size_mb > 1:
                by_size["medium"].append(t)
            else:
                by_size["small"].append(t)

        return {
            "total_tensors": len(tensors),
            "by_device": {dev: len(ts) for dev, ts in by_device.items()},
            "by_size": {size: len(ts) for size, ts in by_size.items()},
            "largest_tensors": [
                {
                    "shape": t.shape,
                    "dtype": str(t.dtype),
                    "device": str(t.device),
                    "size_mb": (t.element_size() * t.nelement()) / 1024**2,
                }
                for t in sorted(tensors, key=lambda x: x.nelement(), reverse=True)[:10]
            ],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Сводная статистика по трекингу."""
        if not self._snapshots:
            return {}

        allocs = [s.allocated_mb for s in self._snapshots]
        peaks = [s.peak_mb for s in self._snapshots]

        return {
            "duration_sec": time.time() - self._start_time if self._start_time else 0,
            "snapshot_count": len(self._snapshots),
            "alloc_mean_mb": np.mean(allocs),
            "alloc_std_mb": np.std(allocs),
            "alloc_trend": allocs[-1] - allocs[0],
            "peak_max_mb": np.max(peaks),
            "tensor_count_trend": self._snapshots[-1].tensor_count - self._snapshots[0].tensor_count,
        }
