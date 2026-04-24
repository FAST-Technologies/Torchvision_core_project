# research/optimization_study/05_memory_optimization/caching_strategy.py

"""
Стратегии кэширования для оптимизации памяти.
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Callable, Any, List
from functools import lru_cache
from dataclasses import dataclass, field
from collections import OrderedDict
import time
import warnings

from .config import MemoryPolicy
from .utils import estimate_tensor_size, format_memory_size


@dataclass
class CacheEntry:
    """Запись в кэше."""

    tensor: torch.Tensor
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_mb: float = 0.0

    def __post_init__(self):
        self.size_mb = estimate_tensor_size(self.tensor)["mb"]


class KernelCache:
    """
    Кэширование ядер свёртки и других повторяемых операций.

    Поддерживает:
    - LRU (Least Recently Used) eviction policy
    - TTL (Time-To-Live) для записей
    - Статистику использования
    - Автоматическую очистку

    Пример:
        >>> cache = KernelCache(max_size=50)
        >>> kernel = cache.get_or_create(
        ...     "sobel_x", (1, 1, 3, 3), torch.float32,
        ...     creator=lambda s, d, dev: torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=dev).view(s)
        ... )
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 300.0,
        policy: MemoryPolicy = MemoryPolicy.POOLED,
    ):
        """
        Args:
            max_size: Макс. число записей в кэше
            ttl_seconds: Время жизни записи (сек)
            policy: Политика управления кэшем
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.policy = policy

        # LRU кэш: OrderedDict для порядка доступа
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = False  # Для потокобезопасности (опционально)

        # Статистика
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def _make_key(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        **kwargs,
    ) -> str:
        """Генерирует уникальный ключ для кэша."""
        # Сортируем kwargs для консистентности
        extra = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{name}_{shape}_{dtype}_{device}{'_' + extra if extra else ''}"

    def get_or_create_kernel(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        creator: Optional[Callable] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Получить тензор из кэша или создать новый.

        Args:
            name: Имя тензора
            shape: Форма
            dtype: Тип данных
            device: Устройство
            creator: Функция создания (если не в кэше)
            **kwargs: Дополнительные параметры для ключа

        Returns:
            torch.Tensor: Кэшированный или новый тензор
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        key = self._make_key(name, shape, dtype, device, **kwargs)
        now = time.time()

        # Проверка кэша
        if key in self._cache:
            entry = self._cache[key]

            # Проверка TTL
            if now - entry.created_at > self.ttl_seconds:
                # TTL истёк — удаляем
                del self._cache[key]
                self._misses += 1
            else:
                # Hit
                entry.last_accessed = now
                entry.access_count += 1
                # Перемещаем в конец (LRU)
                self._cache.move_to_end(key)
                self._hits += 1
                return (
                    entry.tensor.clone()
                    if self.policy == MemoryPolicy.REUSE
                    else entry.tensor
                )

        # Miss — создаём новый
        self._misses += 1

        if creator:
            tensor = creator(shape, dtype, device, **kwargs)
        else:
            tensor = torch.randn(shape, dtype=dtype, device=device)

        # Добавляем в кэш
        self._add_to_cache(key, tensor)

        return tensor.clone() if self.policy == MemoryPolicy.REUSE else tensor

    def _add_to_cache(self, key: str, tensor: torch.Tensor):
        """Добавляет тензор в кэш с eviction при необходимости."""
        # Очистка просроченных записей
        self._cleanup_expired()

        # Eviction если переполнение
        while len(self._cache) >= self.max_size:
            # Удаляем наименее используемый (первый в OrderedDict)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._evictions += 1

        # Добавляем новую запись
        entry = CacheEntry(
            tensor=tensor,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
        )
        self._cache[key] = entry

    def _cleanup_expired(self):
        """Удаляет просроченные записи."""
        now = time.time()
        expired = [
            key
            for key, entry in self._cache.items()
            if now - entry.created_at > self.ttl_seconds
        ]
        for key in expired:
            del self._cache[key]
            self._evictions += 1

    def clear(self):
        """Полная очистка кэша."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> Dict[str, int]:
        """Статистика кэша"""
        otal_size_mb = sum(e.size_mb for e in self._cache.values())
        hit_rate = (
            self._hits / (self._hits + self._misses)
            if (self._hits + self._misses) > 0
            else 0
        )
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_size_mb": total_size_mb,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate_pct": hit_rate * 100,
            "avg_access_count": (
                np.mean([e.access_count for e in self._cache.values()])
                if self._cache
                else 0
            ),
            "memory_mb": sum(
                t.element_size() * t.nelement() / 1024**2 for t in self._cache.values()
            ),
        }

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"KernelCache(size={stats['size']}/{self.max_size}, "
            f"memory={stats['total_size_mb']:.2f}MB, "
            f"hit_rate={stats['hit_rate_pct']:.1f}%)"
        )


class TensorPool:
    """
    Пул предвыделенных тензоров для снижения фрагментации памяти.

    Идеально для методов с фиксированными размерами буферов.
    """

    def __init__(
        self,
        pool_size: int = 10,
        shape: Tuple[int, ...] = (1, 3, 512, 512),
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pool: List[torch.Tensor] = [
            torch.empty(shape, dtype=dtype, device=device) for _ in range(pool_size)
        ]
        self._available: List[int] = list(range(pool_size))
        self._in_use: Dict[int, str] = {}  # index -> owner name

    def acquire(self, owner: str = "unknown") -> Optional[torch.Tensor]:
        """Получить тензор из пула."""
        if not self._available:
            warnings.warn(f"TensorPool exhausted for {owner}")
            return None

        idx = self._available.pop(0)
        self._in_use[idx] = owner
        return self.pool[idx]

    def release(self, tensor: torch.Tensor, owner: str):
        """Вернуть тензор в пул."""
        for idx, t in enumerate(self.pool):
            if t.data_ptr() == tensor.data_ptr():
                if idx in self._in_use and self._in_use[idx] == owner:
                    del self._in_use[idx]
                    self._available.append(idx)
                    # Опционально: обнуление
                    tensor.zero_()
                break

    def stats(self) -> Dict[str, int]:
        return {
            "pool_size": len(self.pool),
            "available": len(self._available),
            "in_use": len(self._in_use),
        }


# Глобальный кэш для всего модуля
_global_kernel_cache = KernelCache(max_size=50)


def get_global_kernel_cache() -> KernelCache:
    """Получить глобальный экземпляр кэша."""
    return _global_kernel_cache
