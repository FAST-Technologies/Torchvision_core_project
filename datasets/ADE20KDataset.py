# datasets/ADE20KDataset.py

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import torch
from torchvision import transforms

class ADE20KDataset(Dataset):
    """Загрузчик датасета ADE20K"""
    
    def __init__(
        self,
        root_dir: str = './data/ade20k',
        split: str = 'training',
        image_size: tuple = (512, 512),
        augment: bool = False,
        subset_fraction: float = None
    ) -> None:
        self.image_size = image_size
        self.augment = augment
        
        base_dir = os.path.join(root_dir, 'ADEChallengeData2016')
        self.images_dir = os.path.join(base_dir, 'images', split)
        self.masks_dir = os.path.join(base_dir, 'annotations', split)

        print(f"📂 Загрузка {split} датасета...")
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks dir not found: {self.masks_dir}")
        
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.endswith('.jpg')
        ])
        print(f"   Найдено {len(self.image_files)} изображений")
        
        self.valid_indices = []
        for i, img_file in enumerate(self.image_files):
            mask_file = img_file.replace('.jpg', '.png')
            if os.path.exists(os.path.join(self.masks_dir, mask_file)):
                self.valid_indices.append(i)
        print(f"   Валидных пар: {len(self.valid_indices)}")
        
        if subset_fraction is not None and subset_fraction < 1.0:
            n = int(len(self.valid_indices) * subset_fraction)
            self.valid_indices = self.valid_indices[:n]
            print(f"   Используем {n} образцов ({subset_fraction*100:.0f}%)")
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_file = self.image_files[real_idx]
        
        img = Image.open(
            os.path.join(self.images_dir, img_file)
        ).convert('RGB')
        
        mask_pil = Image.open(
            os.path.join(self.masks_dir, img_file.replace('.jpg', '.png'))
        ).convert('L')
        
        mask = np.array(mask_pil, dtype=np.int64)
        mask = np.clip(mask, 0, 149)
        
        # Аугментации
        if self.augment and np.random.rand() > 0.5:
            img = transforms.functional.hflip(img)
            mask_pil = transforms.functional.hflip(mask_pil)
            mask = np.array(mask_pil, dtype=np.int64)
            mask = np.clip(mask, 0, 149)
        
        # Resize
        img = transforms.functional.resize(
            img, self.image_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )
        
        mask_pil_resized = transforms.functional.resize(
            mask_pil, self.image_size,
            interpolation=transforms.InterpolationMode.NEAREST
        )
        mask = np.array(mask_pil_resized, dtype=np.int64)
        mask = np.clip(mask, 0, 149)
        
        img = self.img_transform(img)
        mask = torch.from_numpy(mask).long()
        
        return {'image': img, 'mask': mask, 'image_id': img_file}
    
def test_dataloader():
    print("\n" + "="*50)
    print("Тестирование загрузчика ADE20K")
    print("="*50)
    
    try:
        train_dataset = ADE20KDataset(
            root_dir='./data/ade20k',
            split='training',
            image_size=(512, 512),
            augment=False,
            subset_fraction=0.01
        )

        #  Начинаем с num_workers=0 для отладки
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,  # Маленький batch
            shuffle=False,
            num_workers=0,  # 🔥 0 для отладки!
            pin_memory=False
        )

        print(f"✅ DataLoader ready: {len(train_loader)} batches")
        
        print("\n📊 Проверка загрузки данных:")
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image']
            masks = batch['mask']
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"   Images: {images.shape}, dtype={images.dtype}, range=[{images.min():.3f}, {images.max():.3f}]")
            print(f"   Masks: {masks.shape}, dtype={masks.dtype}, unique={torch.unique(masks)[:10].tolist()}")
            
            # Проверка валидности
            assert not torch.isnan(images).any(), "NaN in images!"
            assert masks.min() >= 0 and masks.max() <= 150, f"Mask out of range: [{masks.min()}, {masks.max()}]"
            print("   ✅ Batch valid")
            
            if batch_idx >= 2:
                break
                
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Все тесты пройдены!")
    return True