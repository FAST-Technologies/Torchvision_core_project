from SegmentationTester import SegmentationTester
from torch_segmenter import TorchSegmenter
from neural_segmenter import NeuralSegmenter
from cv2_sklearn_segmenter import CV2SklearnSegmenter
import pandas as pd
from typing import Union, Dict, Any
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import os

def main() -> Union[SegmentationTester, Dict[str, Any], pd.DataFrame]:
    """Пример использования всех классов"""
    
    # Инициализация тестера
    tester = SegmentationTester()
    
    # ============ ДОБАВЛЕНИЕ МЕТОДОВ ============
    
    print("=" * 60)
    print("ИНИЦИАЛИЗАЦИЯ МЕТОДОВ СЕГМЕНТАЦИИ")
    print("=" * 60)
    
    # 1. Методы CV2/Sklearn
    print("\n1. Загрузка методов CV2/Sklearn...")
    cv2_methods = {
        "Global_Threshold_CV2": CV2SklearnSegmenter("global_thresholding", threshold=127),
        "Adaptive_Threshold_CV2": CV2SklearnSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_CV2": CV2SklearnSegmenter("otsu_thresholding"),
        "Region_Growing_CV2": CV2SklearnSegmenter("region_growing", seed=(100, 100), tolerance=20),
        "Sobel_CV2": CV2SklearnSegmenter("sobel_edge", threshold=50),
        "Canny_CV2": CV2SklearnSegmenter("canny_edge", low=50, high=150),
        "KMeans_CV2": CV2SklearnSegmenter("kmeans_segmentation", k=3),
        "Watershed_CV2": CV2SklearnSegmenter("watershed"),
    }
    
    for name, segmenter in cv2_methods.items():
        tester.add_method(name, segmenter)
        print(f"   ✅ {name}")
    
    # 2. Методы PyTorch
    print("\n2. Загрузка методов PyTorch...")
    torch_methods = {
        "Global_Threshold_Torch": TorchSegmenter("global_thresholding", threshold=0.5),
        "Adaptive_Threshold_Torch": TorchSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_Torch": TorchSegmenter("otsu_thresholding"),
        "Region_Growing_Torch": TorchSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        "Sobel_Torch": TorchSegmenter("sobel_edge", threshold=0.1),
        "Canny_Torch": TorchSegmenter("canny_edge", low=0.1, high=0.3),
        "KMeans_Torch": TorchSegmenter("kmeans_segmentation", k=3),
        "FloodFill_Torch": TorchSegmenter("floodfill", seed=(200, 150), tolerance=0.15),
    }
    
    for name, segmenter in torch_methods.items():
        tester.add_method(name, segmenter)
        print(f"   ✅ {name}")

     # 3. Нейросетевая сегментация
    print("\n3. Загрузка нейросетевых методов...")
    try:
        neural_segmenter = NeuralSegmenter(
            model_name="nvidia/segformer-b5-finetuned-ade-640-640"
        )
        tester.add_method("Neural_SegFormer", neural_segmenter)
        print(f"   ✅ Neural_SegFormer")
    except Exception as e:
        print(f"   ❌ Neural_SegFormer - ошибка: {e}")
        print("   Установите transformers: pip install transformers")
    
    print(f"\nВсего методов загружено: {len(tester.methods)}")
    
    # ============ ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ ============
    
    print("\n" + "=" * 60)
    print("ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ")
    print("=" * 60)
    
    # Загрузка тестового изображения
    try:
        # Загрузка тестового изображения из Hugging Face
        repo_id = "hf-internal-testing/fixtures_ade20k"
        print(f"Загрузка из репозитория: {repo_id}")
        
        image_path = hf_hub_download(
            repo_id=repo_id, 
            filename="ADE_val_00000001.jpg", 
            repo_type="dataset"
        )
        image = Image.open(image_path)
        
        segmentation_map_path = hf_hub_download(
            repo_id=repo_id, 
            filename="ADE_val_00000001.png", 
            repo_type="dataset"
        )
        segmentation_map = Image.open(segmentation_map_path)
        
        print(f"✅ Изображение загружено: {image_path}")
        print(f"   Размер: {image.size}")
        print(f"✅ Ground truth загружен: {segmentation_map_path}")
        
        # Сохраняем локально для тестов
        local_image_path = "test_image.jpg"
        image.save(local_image_path)
        print(f"✅ Изображение сохранено локально: {local_image_path}")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки из Hugging Face: {e}")
        print("Используем тестовое изображение по умолчанию...")
        
        # Альтернативный URL
        img_url = "https://i.pinimg.com/736x/17/e7/fc/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg"
        local_image_path = "test_image_default.jpg"
        
        try:
            response = requests.get(img_url)
            image = Image.open(BytesIO(response.content))
            image.save(local_image_path)
            print(f"✅ Изображение загружено из URL: {img_url}")
            print(f"   Размер: {image.size}")
        except Exception as e2:
            print(f"❌ Ошибка загрузки из URL: {e2}")
            raise
    
    # ============ ДЕТАЛЬНАЯ НЕЙРОСЕТЕВАЯ СЕГМЕНТАЦИЯ ============
    
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНАЯ НЕЙРОСЕТЕВАЯ СЕГМЕНТАЦИЯ")
    print("=" * 60)

    try:
        neural_segmenter = NeuralSegmenter()
        
        # Вариант 1: Детальная визуализация
        print("\nВариант 1: Детальная визуализация...")
        detailed_result = neural_segmenter.visualize_detailed_segmentation(
            local_image_path,
            figsize=(16, 12)
        )
        
        # Вариант 2: Сравнение с ground truth (если есть)
        if 'segmentation_map_path' in locals():
            print("\nВариант 2: Сравнение с Ground Truth...")
            comparison_result = neural_segmenter.compare_with_ground_truth(
                local_image_path,
                segmentation_map_path
            )
        
        # Вариант 3: Простая сегментация (segment_image)
        print("\nВариант 3: Простая сегментация (segment_image)...")
        simple_result = neural_segmenter.segment_image(local_image_path, alpha=0.7)
        
        # Сохраняем результат
        simple_result.save("neural_segmentation_result.jpg")
        print(f"✅ Результат сохранен: neural_segmentation_result.jpg")
        
        # Показываем простой результат
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(simple_result)
        axes[1].set_title("Neural Segmentation (alpha=0.7)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Ошибка нейросетевой сегментации: {e}")
    
    # ============ СРАВНЕНИЕ ВСЕХ МЕТОДОВ ============
    
    print("\n" + "=" * 60)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ СЕГМЕНТАЦИИ")
    print("=" * 60)
    
    # Выбираем подмножество методов для сравнения
    selected_methods = [
        "Global_Threshold_CV2",
        "Otsu_CV2", 
        "Canny_CV2",
        "KMeans_CV2",
        "Watershed_CV2",
        "Global_Threshold_Torch",
        "Canny_Torch",
        "KMeans_Torch",
        "FloodFill_Torch"
    ]
    
    if "Neural_SegFormer" in tester.methods:
        selected_methods.append("Neural_SegFormer")
    
    print(f"Сравниваем {len(selected_methods)} методов...")
    results = tester.compare_methods(
        local_image_path, 
        method_names=selected_methods,
        figsize=(20, 15)
    )
    
    # Визуализация сравнения
    tester.visualize_comparison(results, show_masks=True)
    
    # ============ БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ ============
    
    print("\n" + "=" * 60)
    print("БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    
    df = tester.benchmark_methods(local_image_path, n_runs=2)
    
    # ============ СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ============
    
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    output_dir = "segmentation_comparison_results"
    tester.save_results(results, output_dir)
    
    # Дополнительно: сохраняем сводную таблицу
    summary_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"✅ Сводная таблица сохранена: {summary_path}")

    # ============ ВЫВОД ИНФОРМАЦИИ ============
    
    print("\n" + "=" * 60)
    print("РЕЗЮМЕ")
    print("=" * 60)
    print(f"✓ Загружено методов: {len(tester.methods)}")
    print(f"✓ Протестировано: {len(selected_methods)}")
    print(f"✓ Изображение: {local_image_path}")
    print(f"✓ Результаты сохранены в: {output_dir}")
    print(f"✓ Бенчмарк выполнен: {len(df)} методов")
    print("=" * 60)
    
    return tester, results, df

def test_neural_segmentation_variants():
    """Тестирование различных вариантов нейросетевой сегментации"""
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ВАРИАНТОВ НЕЙРОСЕТЕВОЙ СЕГМЕНТАЦИИ")
    print("=" * 60)
    
    # Загрузка тестового изображения
    img_url = "https://i.pinimg.com/736x/17/e7/fc/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg"
    response = requests.get(img_url)
    test_image = Image.open(BytesIO(response.content))
    
    # Создаем нейросетевой сегментатор
    segmenter = NeuralSegmenter()
    
    # Вариант 1: Различные значения alpha
    print("\n1. Тестирование разных значений alpha:")
    alphas = [0.3, 0.5, 0.7, 1.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, alpha in enumerate(alphas):
        result = segmenter.segment_image(test_image, alpha=alpha)
        axes[i].imshow(result)
        axes[i].set_title(f"alpha = {alpha}")
        axes[i].axis('off')
        
        # Сохраняем
        result.save(f"neural_alpha_{alpha}.jpg")
    
    plt.suptitle("Neural Segmentation with Different Alpha Values", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Вариант 2: Детальный анализ
    print("\n2. Детальный анализ сегментации:")
    detailed_result = segmenter.detailed_segmentation(test_image)
    
    # Выводим информацию о классах
    print(f"Обнаружено классов: {detailed_result['total_classes']}")
    print("\nТоп-5 классов по площади:")
    
    sorted_classes = sorted(detailed_result['class_distribution'].items(), 
                           key=lambda x: x[1]['pixel_count'], 
                           reverse=True)[:5]
    
    for class_name, info in sorted_classes:
        print(f"  {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")
    
    # Вариант 3: segment_with_mask
    print("\n3. Тестирование segment_with_mask:")
    result_np, mask = segmenter.segment_with_mask(test_image, alpha=0.5)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(test_image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(result_np)
    axes[1].set_title("Segmentation Result")
    axes[1].axis('off')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Binary Mask")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Размер маски: {mask.shape}")
    print(f"Площадь маски: {np.sum(mask > 0)} пикселей")
    
    return segmenter, detailed_result

if __name__ == "__main__":
    # Основной тест
    print("ЗАПУСК ОСНОВНОГО ТЕСТА")
    print("=" * 60)
    tester, results, df = main()
    
    # Дополнительный тест нейросетевых вариантов
    print("\n\nЗАПУСК ДОПОЛНИТЕЛЬНОГО ТЕСТА НЕЙРОСЕТЕВЫХ ВАРИАНТОВ")
    print("=" * 60)
    segmenter, detailed_result = test_neural_segmentation_variants()