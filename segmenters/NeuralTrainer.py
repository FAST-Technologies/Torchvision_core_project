# segmenters/NeuralTrainer.py

# Импорт основных библиотек
import os
import time
import gc
from typing import (
    List,
    Dict,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score

# ──────────────────────────────────────────────────────────────────────
# TYPE ALIASES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────
num_classes: int = 150
DeviceStr = Union[torch.device, str]
LossValue = float
MetricValue = float
HistoryDict = Dict[str, List[float]]
BatchDict = Dict[str, torch.Tensor]


class NeuralTrainer:
    """
    Трейнер для fine-tuning нейронных моделей семантической сегментации.

    Поддерживает:
    - Обучение с учётом `ignore_index` для игнорируемых пикселей (например, 255 в ADE20K).
    - Вспомогательный лосс (`aux_loss`) для моделей с дополнительными выходами (DeepLabV3+, FCN).
    - Градиентный клиппинг для стабильности обучения.
    - CosineAnnealingLR scheduler с автоматическим уменьшением LR.
    - Early stopping по валидационному mIoU.
    - Сохранение лучшего чекпоинта по метрике.

    Особенности:
    - Использует `CrossEntropyLoss` с параметром `ignore_index`.
    - Автоматически фильтрует невалидные значения масок (<0 или >=num_classes).
    - Рассчитывает macro mIoU через `sklearn.metrics.jaccard_score` с учётом присутствующих классов.
    - Ведёт историю метрик: `train_loss`, `val_loss`, `val_miou`.

    Attributes:
        model (nn.Module): Обучаемая модель сегментации.
        train_loader (DataLoader): DataLoader для тренировочного набора.
        val_loader (DataLoader): DataLoader для валидационного набора.
        device (str): Устройство для вычислений ("cuda" или "cpu").
        num_classes (int): Количество выходных классов.
        ignore_index (int): Индекс пикселей для игнорирования в лоссе.
        aux_loss_weight (float): Коэффициент для вспомогательного лосса (если есть).
        verbose_first_batch (bool): Если `True`, выводит отладочную информацию по первому батчу.
        criterion (nn.CrossEntropyLoss): Функция потерь.
        optimizer (torch.optim.AdamW): Оптимизатор.
        scheduler (torch.optim.lr_scheduler.CosineAnnealingLR): Scheduler LR.
        history (HistoryDict): История метрик по эпохам.
        best_miou (float): Лучший достигнутый mIoU на валидации.

    Example:
        ```python
        trainer = NeuralTrainer(
            model=unet_model,
            train_loader=train_dl,
            val_loader=val_dl,
            num_classes=150,
            ignore_index=255,
            aux_loss_weight=0.4,  # для DeepLabV3+
            device="cuda",
        )
        history = trainer.fit(epochs=100, checkpoint_path="models/unet_best.pth")
        print(f"Best mIoU: {trainer.best_miou:.4f}")
        ```
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = num_classes,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ignore_index: int = 255,
        aux_loss_weight: float = 0.4,  # FIX 4: коэф. aux_loss для DeepLab/FCN
        verbose_first_batch: bool = False,
    ) -> None:
        """
        Инициализация трейнера.

        Args:
            model: Модель сегментации (nn.Module) с выходом `[B, C, H, W]` или `dict` с ключами `"out"`/`"aux"`.
            train_loader: DataLoader для тренировочного набора.
            val_loader: DataLoader для валидационного набора.
            num_classes: Количество выходных классов (по умолчанию 150 для ADE20K).
            device: Устройство для вычислений ("cuda" или "cpu").
            lr: Начальный learning rate.
            weight_decay: Коэффициент L2-регуляризации.
            ignore_index: Индекс пикселей для игнорирования в `CrossEntropyLoss`.
            aux_loss_weight: Вес вспомогательного лосса (для моделей с `aux` выходом).
            verbose_first_batch: Если `True`, выводит отладочную информацию по первому батчу.
        """
        self.model: nn.Module = model.to(device)
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.device: str = str(device) if isinstance(device, torch.device) else device
        self.num_classes: int = int(num_classes)
        self.ignore_index: int = ignore_index
        self.aux_loss_weight: float = aux_loss_weight
        self.verbose_first_batch: bool = verbose_first_batch

        # Критерий и оптимизатор
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )
        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler: T_max = общее число шагов за 50 эпох (можно переопределить в fit)
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(train_loader) * 50,
                eta_min=lr * 0.01,
            )
        )

        # История и лучшие метрики
        self.history: HistoryDict = {
            "train_loss": [],
            "val_loss": [],
            "val_miou": [],
        }
        self.best_miou: MetricValue = 0.0

    def train_epoch(self) -> LossValue:
        """
        Выполняет одну эпоху обучения.

        Логика:
        1. Переключает модель в режим `.train()`.
        2. Для каждого батча:
           - Перемещает данные на `device`.
           - Выполняет forward pass.
           - Рассчитывает основной лосс + вспомогательный (если есть `aux` выход).
           - Выполняет backward pass с градиентным клиппингом.
           - Обновляет веса и scheduler.
        3. Возвращает средний тренировочный лосс.

        Отладка:
        - Если `verbose_first_batch=True`, выводит статистику по маскам первого батча:
          диапазон значений, уникальные классы, количество пикселей для ключевых классов.

        Returns:
            float: Средний тренировочный лосс за эпоху.
        """
        self.model.train()
        total_loss: LossValue = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device).long()

            # ──────────────────────────────────────────────────────────────
            # Отладка первого батча (опционально)
            # ──────────────────────────────────────────────────────────────
            if batch_idx == 0:
                print(f"DEBUG: masks shape={masks.shape}, dtype={masks.dtype}")
                print(f"🔍 DEBUG masks (ignore_index={self.ignore_index}):")
                print(f"   Range: [{masks.min()}, {masks.max()}]")
                print(f"   Unique: {torch.unique(masks)[:30]}")
                print(f"   Count of class 0 (wall): {(masks == 0).sum().item()}")
                print(f"   Count of ignore_index (255): {(masks == 255).sum().item()}")
                print(f"   Count of class 149: {(masks == 149).sum().item()}")

            # ──────────────────────────────────────────────────────────────
            # Forward pass
            # ──────────────────────────────────────────────────────────────
            self.optimizer.zero_grad()
            outputs: Union[torch.Tensor, Dict[str, torch.Tensor]] = self.model(
                images
            )  # [B, C, H, W]

            if isinstance(outputs, dict):
                main_out = outputs["out"]
                aux_out = outputs.get("aux", None)
            else:
                main_out = outputs
                aux_out = None

            # ──────────────────────────────────────────────────────────────
            # Валидация масок: замена невалидных значений на ignore_index
            # ──────────────────────────────────────────────────────────────
            invalid = ((masks < 0) | (masks >= self.num_classes)) & (
                masks != self.criterion.ignore_index
            )
            if torch.any(invalid):
                if batch_idx == 0:
                    print(
                        f"⚠️  Found {invalid.sum().item()} invalid mask values, "
                        f"replacing with ignore_index={self.ignore_index}"
                    )
                masks = masks.clone()
                masks[invalid] = self.criterion.ignore_index

            # ──────────────────────────────────────────────────────────────
            # Расчёт лосса
            # ──────────────────────────────────────────────────────────────
            loss: torch.Tensor = self.criterion(main_out, masks)
            if aux_out is not None and self.aux_loss_weight > 0:
                loss = loss + self.aux_loss_weight * self.criterion(aux_out, masks)

            # ──────────────────────────────────────────────────────────────
            # Backward pass + оптимизация
            # ──────────────────────────────────────────────────────────────
            loss.backward()

            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            # Логирование прогресса
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"   Batch {batch_idx + 1}/{len(self.train_loader)} | Loss: {loss.item():.4f}"
                )

        return total_loss / len(self.train_loader)

    def validate(self) -> Tuple[LossValue, MetricValue]:
        """
        Выполняет валидацию и рассчитывает macro mIoU.

        Логика:
        1. Переключает модель в режим `.eval()` и отключает градиенты.
        2. Для каждого валидационного батча:
           - Выполняет forward pass.
           - Рассчитывает лосс.
           - Сохраняет предсказания и таргеты (после `.argmax(1)`).
        3. Рассчитывает macro mIoU через `sklearn.metrics.jaccard_score`:
           - Фильтрует пиксели с `ignore_index`.
           - Использует только присутствующие в валидации классы.
           - `zero_division=0` для защиты от деления на 0.

        Returns:
            Tuple[float, float]:
            - Средний валидационный лосс.
            - Macro mIoU на валидационном наборе.
        """
        self.model.eval()
        total_loss: LossValue = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                outputs: Union[torch.Tensor, Dict[str, torch.Tensor]] = self.model(
                    images
                )
                if isinstance(outputs, dict):
                    outputs = outputs["out"]

                loss: torch.Tensor = self.criterion(outputs, masks.long())
                total_loss += loss.item()

                # Предсказания: класс с максимальным логитом
                preds: torch.Tensor = outputs.argmax(1)  # [B, H, W]
                all_preds.extend(preds.cpu().flatten().tolist())
                all_targets.extend(masks.cpu().flatten().tolist())

        avg_loss: LossValue = total_loss / len(self.val_loader)

        # ──────────────────────────────────────────────────────────────
        # Расчёт mIoU (macro, только присутствующие классы)
        # ──────────────────────────────────────────────────────────────
        filtered: List[Tuple[int, int]] = [
            (p, t) for p, t in zip(all_preds, all_targets) if t != self.ignore_index
        ]
        if filtered:
            f_preds, f_targets = zip(*filtered)
            present_labels: List[int] = list(set(f_targets))
            miou = jaccard_score(
                f_targets,
                f_preds,
                average="macro",
                labels=present_labels,
                zero_division=0,
            )
        else:
            miou = 0.0
        return avg_loss, miou

    def fit(
        self,
        epochs: int = 20,
        checkpoint_path: str = "./models/best_model.pth",
        early_stop_patience: int = 200,
    ) -> HistoryDict:
        """
        Полный цикл обучения с валидацией и early stopping.

        Для каждой эпохи:
        1. Выполняет `train_epoch()`.
        2. Выполняет `validate()`.
        3. Логирует метрики и LR.
        4. Сохраняет лучший чекпоинт по `val_miou`.
        5. Проверяет early stopping: если `val_miou` не улучшается `early_stop_patience` эпох — остановка.

        После обучения:
        - Очищает CUDA-кэш и собирает мусор.
        - Возвращает историю метрик.

        Args:
            epochs: Максимальное количество эпох обучения.
            checkpoint_path: Путь для сохранения лучшего чекпоинта.
            early_stop_patience: Количество эпох без улучшения `val_miou` для остановки.

        Returns:
            HistoryDict: Словарь с историями метрик:
            ```python
            {
                "train_loss": List[float],
                "val_loss": List[float],
                "val_miou": List[float],
            }
            ```

        Note:
            - Чекпоинт сохраняется в формате:
              ```python
              {
                  "epoch": int,
                  "model_state_dict": state_dict,
                  "optimizer_state_dict": state_dict,
                  "miou": float,
                  "ignore_index": int,
              }
              ```
            - Early stopping срабатывает, если `val_miou` не превышает `best_miou * 0.999`
              в течение `early_stop_patience` эпох (защита от микро-флуктуаций).
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        print(
            f"🎯 Starting training for {epochs} epochs "
            f"| ignore_index={self.ignore_index} | aux_loss_w={self.aux_loss_weight}"
        )
        patience_counter: int = 0
        for epoch in range(epochs):
            start_time: float = time.perf_counter()
            train_loss: LossValue = self.train_epoch()
            val_loss: LossValue
            val_miou: MetricValue
            val_loss, val_miou = self.validate()

            epoch_time: float = time.perf_counter() - start_time

            # Обновление истории
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_miou"].append(val_miou)

            # Текущий LR
            lr_now: float = self.optimizer.param_groups[0]["lr"]
            print(
                f"📊 Epoch {epoch + 1}/{epochs} | Time: {epoch_time:.3f}s | LR: {lr_now:.3e}"
            )
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Val mIoU:   {val_miou:.4f}")
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "miou": val_miou,
                        "ignore_index": self.ignore_index,
                    },
                    checkpoint_path,
                )
                print(f"   💾 Saved best model (mIoU: {val_miou:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                # Early stopping: если нет улучшения в течение patience эпох
                if patience_counter >= early_stop_patience:
                    print("   ⏹️  Early stopping")
                    break
            torch.cuda.empty_cache()
            gc.collect()
        print(f"✅ Training complete! Best mIoU: {self.best_miou:.4f}")
        return self.history
