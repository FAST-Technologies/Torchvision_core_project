# segmenters/NeuralTrainer.py

# Импорт основных библиотек
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import (
    List,
    Dict,
    Any,
)
from sklearn.metrics import jaccard_score

import time
import gc

num_classes: int = 150


class NeuralTrainer:
    """Трейнер для fine-tuning нейронных моделей сегментации"""

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
        self.model = model.to(device)
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.device: str = device
        self.num_classes: int = int(num_classes)
        self.ignore_index: int = ignore_index
        self.aux_loss_weight: float = aux_loss_weight
        self.verbose_first_batch: bool = verbose_first_batch
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 50,
            eta_min=lr * 0.01,
        )
        self.history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_miou": [],
        }
        self.best_miou: float = 0.0

    def train_epoch(self) -> float:
        """Одна эпоха обучения"""
        self.model.train()
        total_loss: float = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device).long()

            if batch_idx == 0:
                print(f"DEBUG: masks shape={masks.shape}, dtype={masks.dtype}")
                print(f"🔍 DEBUG masks (ignore_index={self.ignore_index}):")
                print(f"   Range: [{masks.min()}, {masks.max()}]")
                print(f"   Unique: {torch.unique(masks)[:30]}")
                print(f"   Count of class 0 (wall): {(masks == 0).sum().item()}")
                print(f"   Count of ignore_index (255): {(masks == 255).sum().item()}")
                print(f"   Count of class 149: {(masks == 149).sum().item()}")

            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(images)  # [B, C, H, W]

            if isinstance(outputs, dict):
                main_out = outputs["out"]
                aux_out = outputs.get("aux", None)
            else:
                main_out = outputs
                aux_out = None

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
            loss = self.criterion(main_out, masks)
            # if aux_out is not None:
            #     loss += self.aux_loss_weight * self.criterion(aux_out, masks)
            loss.backward()

            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"   Batch {batch_idx + 1}/{len(self.train_loader)} | Loss: {loss.item():.4f}"
                )

        return total_loss / len(self.train_loader)

    def validate(self) -> tuple:
        """Валидация с вычислением mIoU"""
        self.model.eval()
        total_loss: float = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]

                loss = self.criterion(outputs, masks.long())
                total_loss += loss.item()

                # Предсказания
                preds = outputs.argmax(1)  # [B, H, W]
                all_preds.extend(preds.cpu().flatten().tolist())
                all_targets.extend(masks.cpu().flatten().tolist())

        avg_loss = total_loss / len(self.val_loader)
        filtered = [
            (p, t) for p, t in zip(all_preds, all_targets) if t != self.ignore_index
        ]
        if filtered:
            f_preds, f_targets = zip(*filtered)
            present_labels = list(set(f_targets))
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
    ) -> Dict[str, List]:
        """Полный цикл обучения"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        print(
            f"🎯 Starting training for {epochs} epochs "
            f"| ignore_index={self.ignore_index} | aux_loss_w={self.aux_loss_weight}"
        )
        patience_counter = 0
        for epoch in range(epochs):
            start_time: float = time.time()
            train_loss = self.train_epoch()
            val_loss, val_miou = self.validate()
            epoch_time: float = time.time() - start_time
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_miou"].append(val_miou)
            lr_now = self.optimizer.param_groups[0]["lr"]
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
                if patience_counter >= early_stop_patience:
                    print("   ⏹️  Early stopping")
                    break
            torch.cuda.empty_cache()
            gc.collect()
        print(f"✅ Training complete! Best mIoU: {self.best_miou:.4f}")
        return self.history
