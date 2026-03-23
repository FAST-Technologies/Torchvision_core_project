# datasets/load_datasets.py

# Импорт основных библиотек

from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)

import os
import zipfile
from tqdm import tqdm
import shutil
import requests

from io import BytesIO
from PIL import Image

from huggingface_hub import hf_hub_download, list_repo_files
from datasets import load_dataset


def download_ade20k(root_dir: str = './../data/ade20k') -> None:
    """
    Скачивание и распаковка ADE20K датасета
    """
    os.makedirs(root_dir, exist_ok=True)
    
    url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    zip_path = os.path.join(root_dir, 'ADEChallengeData2016.zip')
    
    if not os.path.exists(zip_path):
        print(f"📥 Скачивание ADE20K датасета...")
        print(f"   URL: {url}")
        print(f"   Размер: ~3.8 GB")
        print(f"   Это может занять некоторое время...")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Скачивание") as pbar:
                for data in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    f.write(data)
                    pbar.update(len(data))
        
        print("✅ Скачивание завершено!")
    else:
        print(f"✅ Архив уже существует: {zip_path}")
    
    zip_size = os.path.getsize(zip_path) / (1024*1024*1024)
    print(f"   Размер архива: {zip_size:.2f} GB")
    extract_dir = os.path.join(root_dir, 'extracted')
    
    if not os.path.exists(os.path.join(root_dir, 'ADEChallengeData2016')):
        print("📦 Распаковка архива...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for file in tqdm(file_list, desc="Распаковка", unit="files"):
                zip_ref.extract(file, extract_dir)
        
        print("✅ Распаковка завершена!")
        print("\n🔍 Анализ структуры распакованных файлов...")
        found_structure = False
        for root, dirs, files in os.walk(extract_dir):
            if 'images' in dirs and 'annotations' in dirs:
                source_dir = root
                print(f"✅ Найдена структура в: {source_dir}")
                found_structure = True
                break
        
        if found_structure:
            target_dir = os.path.join(root_dir, 'ADEChallengeData2016')
            os.makedirs(target_dir, exist_ok=True)
            print("📋 Организация файлов в правильную структуру...")
            
            source_images = os.path.join(source_dir, 'images')
            target_images = os.path.join(target_dir, 'images')
            if os.path.exists(source_images):
                print(f"   Копирование {source_images} -> {target_images}")
                shutil.copytree(source_images, target_images, dirs_exist_ok=True)
            
            source_annotations = os.path.join(source_dir, 'annotations')
            target_annotations = os.path.join(target_dir, 'annotations')
            if os.path.exists(source_annotations):
                print(f"   Копирование {source_annotations} -> {target_annotations}")
                shutil.copytree(source_annotations, target_annotations, dirs_exist_ok=True)
            
            print(f"✅ Файлы организованы в: {target_dir}")
        else:
            print("❌ Не удалось найти папки images и annotations")
            print("\nСодержимое распакованной папки:")
            for root, dirs, files in os.walk(extract_dir):
                level = root.replace(extract_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for d in dirs[:5]:
                    print(f'{subindent}📁 {d}/')
                for f in files[:5]:
                    print(f'{subindent}📄 {f}')
        print("\n🧹 Очистка временных файлов...")
        shutil.rmtree(extract_dir, ignore_errors=True)
        print("✅ Готово")
    
    else:
        print(f"✅ Датасет уже распакован в {os.path.join(root_dir, 'ADEChallengeData2016')}")
    print("\n📂 Финальная структура датасета:")
    ade_dir = os.path.join(root_dir, 'ADEChallengeData2016')
    if os.path.exists(ade_dir):
        for split in ['training', 'validation']:
            img_dir = os.path.join(ade_dir, 'images', split)
            ann_dir = os.path.join(ade_dir, 'annotations', split)
            if os.path.exists(img_dir):
                n_images = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
                print(f"   {split} images: {n_images} файлов")
            if os.path.exists(ann_dir):
                n_masks = len([f for f in os.listdir(ann_dir) if f.endswith('.png')])
                print(f"   {split} masks: {n_masks} файлов")

def load_test_image_from_hf(
    repo_id: str, 
    filename: str = None, 
    split: str = "validation"
) -> Optional[Image.Image | None]:
    """
    Универсальная загрузка тестового изображения из HuggingFace.
    
    Args:
        repo_id: ID репозитория (например, "cityscapes")
        filename: конкретный файл (если None, берем первый из split)
        split: split датасета
    
    Returns:
        PIL.Image или None
    """
    try:
        if filename:
            path = hf_hub_download(repo_id, filename, repo_type="dataset")
            return Image.open(path).convert("RGB")
        else:
            dataset = load_dataset(repo_id, split=split)
            if 'image' in dataset.features:
                return dataset[0]['image'].convert("RGB")
            else:
                print(f"   ⚠️  No 'image' feature in {repo_id}")
                return None
    except Exception as e:
        print(f"   ❌ Failed to load {repo_id}: {e}")
        return None

print("="*50)
print("ADE20K Dataset Downloader")
print("="*50)
download_ade20k(root_dir='./../data/ade20k')

print("\n" + "="*50)
print("✅ Готово! Теперь можно использовать датасет.")

# ============================================================================
# 2. CITYSCAPES (городские сцены)
# ============================================================================
print("\n Cityscapes Dataset...")
cityscapes_img = load_test_image_from_hf("Chris1/cityscapes", split="train")
cityscapes_img.save("./../data/cityscapes_img.jpg")

# ============================================================================
# 3. COCO (общие объекты)
# ============================================================================
print("\n COCO Dataset...")
coco_img = load_test_image_from_hf("detection-datasets/coco", split="train")
coco_img.save("./../data/coco_img.jpg")

# ============================================================================
# 4. МЕДИЦИНСКИЙ ДАТАСЕТ (ISIC - skin lesion segmentation)
# ============================================================================
print("\n Medical Dataset (ISIC - Skin Lesion)...")
isic_img = load_test_image_from_hf("researchjyotsna/isic2018_10", split="train")
isic_img.save("./../data/isic_img.jpg")

# ============================================================================
# 5. Chest X-Ray
# ============================================================================
print("\n Chest X-Ray Segmentation...")
chest_x_ray = load_test_image_from_hf("danjacobellis/chexpert")
chest_x_ray.save("./../data/chest_x_ray.jpg")

print("\n" + "=" * 70)
print("✅ DATASET DOWNLOAD COMPLETE")
print("=" * 70)
print(f"\n📁 Available test images:")
print(f"   • ADE20K:     cityscapes_img.jpg")
print(f"   • Cityscapes: cityscapes_img.jpg")
print(f"   • COCO:       coco_img.jpg")
print(f"   • Medical:    isic_img.jpg")
print(f"   • Medical:    chest_x_ray.jpg")