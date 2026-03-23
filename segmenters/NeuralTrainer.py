# segmenters/NeuralTrainer.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path

class NeuralTrainer:
    """Трейнер для fine-tuning нейронных моделей сегментации"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader, 
        val_loader: DataLoader,
        num_classes: int = 150,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ignore_index: int = 255
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 50  # 50 эпох
        )
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_miou": []
        }
        self.best_miou = 0
        self.history = {"train_loss": [], "val_loss": [], "val_miou": []}
    
    def train_epoch(self) -> float:
        """Одна эпоха обучения"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            if batch_idx == 0:
                print(f"DEBUG: masks shape={masks.shape}, dtype={masks.dtype}")
                print(f"DEBUG: masks range=[{masks.min()}, {masks.max()}]")
                print(f"DEBUG: unique values={torch.unique(masks)[:20]}")
            
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(images)  # [B, C, H, W]
            
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            if torch.any(masks < 0) or torch.any(masks >= self.num_classes):
                invalid = ((masks < 0) | (masks >= self.num_classes)) & (masks != self.criterion.ignore_index)
                if torch.any(invalid):
                    print(f"⚠️  Invalid mask values: min={masks.min()}, max={masks.max()}")
                    print(f"   Unique invalid: {torch.unique(masks[invalid])}")
                    # Клиппинг на лету (экстренная мера)
                    masks = torch.clamp(masks, 0, self.num_classes - 1)
            loss = self.criterion(outputs, masks.long())
            loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> tuple:
        """Валидация с вычислением mIoU"""
        from sklearn.metrics import jaccard_score
        
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                loss = self.criterion(outputs, masks.long())
                total_loss += loss.item()
                
                # Предсказания
                preds = outputs.argmax(1)  # [B, H, W]
                all_preds.extend(preds.cpu().flatten().tolist())
                all_targets.extend(masks.cpu().flatten().tolist())
        
        avg_loss = total_loss / len(self.val_loader)
        miou = jaccard_score(
            all_targets, 
            all_preds,
            average='weighted',
            labels=range(self.num_classes),
            zero_division=0
        )
        
        return avg_loss, miou
    
    def fit(
        self,
        epochs: int = 20,
        checkpoint_path: str = "best_model.pth",
        early_stop_patience: int = 5
    ) -> Dict[str, List]:
        """Полный цикл обучения"""
        import time
        import gc
        
        print(f"🎯 Starting training for {epochs} epochs...")
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss, val_miou = self.validate()
            
            epoch_time = time.time() - start_time
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_miou"].append(val_miou)
            
            print(f"📊 Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Val mIoU:   {val_miou:.4f}")
            
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'miou': val_miou,
                }, checkpoint_path)
                print(f"   💾 Saved best model (mIoU: {val_miou:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"   ⏹️  Early stopping")
                    break
            
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"✅ Training complete! Best mIoU: {self.best_miou:.4f}")
        return self.history