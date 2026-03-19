# main.py

# Импорт основных библиотек
from segmenters.NeuralSegmenter import NeuralSegmenter
from segmenters.OpenCVSegmenter import OpenCVSegmenter
from segmenters.SklearnSegmenter import SklearnSegmenter
from segmenters.TorchSegmenter import TorchSegmenter
from testing.SegmentationTester import SegmentationTester
from testing.SegmentationComparator import SegmentationComparator
from testing.TorchImplementationValidator import TorchImplementationValidator
from metrics.SegmentationMetrics import SegmentationMetrics
import glob

from typing import (
    List, Union, Tuple, Dict, Any, TypeVar, Optional, 
    Literal, Protocol, runtime_checkable, overload, TYPE_CHECKING
)

import os
import shutil
import datetime
from datetime import datetime
import traceback
import warnings
import time
import requests
from io import BytesIO
from PIL import Image

from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import torch
import gc

def main():
    print("=" * 60)
    print("ОБЪЕДИНЁННЫЙ ФРЕЙМВОРК СЕГМЕНТАЦИИ")
    print("=" * 60)
    
    # ============ ДОБАВЛЕНИЕ МЕТОДОВ ============
    
    print("=" * 60)
    print("ИНИЦИАЛИЗАЦИЯ МЕТОДОВ СЕГМЕНТАЦИИ")
    print("=" * 60)
    tester = SegmentationTester()
    
    print("\n1. Загрузка методов OpenCV...")
    cv2_methods = {
        "Global_Threshold_CV2": OpenCVSegmenter("global_thresholding", threshold=0.5),
        "Adaptive_Threshold_CV2": OpenCVSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_Thresholding_CV2": OpenCVSegmenter("otsu_thresholding"),
        "Niblack_Thresholding_CV2": OpenCVSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "Sauvola_Thresholding_CV2": OpenCVSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
        "Sobel_CV2": OpenCVSegmenter("sobel_edge", threshold=0.1),
        "Canny_CV2": OpenCVSegmenter("canny_edge", low=0.1, high=0.3),
        # "Region_Growing_CV2": OpenCVSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        # "Split_And_Merge_CV2": OpenCVSegmenter("split_and_merge", min_size=50, threshold=0.1),
        # "Floodfill_CV2": OpenCVSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        # "KMeans_CV2": OpenCVSegmenter("kmeans_segmentation", k=3),
        # "DBSCAN_CV2": OpenCVSegmenter("dbscan_segmentation", eps=0.1, min_samples=10),
        # "Meanshift_CV2": OpenCVSegmenter("meanshift", spatial_radius=35, color_radius=60),
        # "Active_Contour_CV2": OpenCVSegmenter("active_contour", iterations=10),
        # "GVF_CV2": OpenCVSegmenter("gvf_contour", mu=0.1, iterations=50),
        # "Morphological_Snakes_CV2": OpenCVSegmenter("morphological_snakes", iterations=100),
        # "Chan_Vese_CV2": OpenCVSegmenter("chan_vese", mu=0.25, max_iter=100),
        # "Watershed_CV2": OpenCVSegmenter("watershed"),
        # "Random_Walker_CV2": OpenCVSegmenter("random_walker"),
        # # "Quickshift_CV2": OpenCVSegmenter("quickshift", bandwidth=0.5),
        # "Slic_CV2": OpenCVSegmenter("slic", region_size=20, ruler=10.0),
        # "Felzenszwalb_CV2": OpenCVSegmenter("felzenszwalb"),
        # "GrabCut_CV2": OpenCVSegmenter("grabcut", num_iterations=10),
    }

    print("\n2. Загрузка методов SKlearn...")
    sklearn_methods = {
        "Global_Threshold_Sklearn": SklearnSegmenter("global_thresholding", threshold=0.5),
        "Adaptive_Threshold_Sklearn": SklearnSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_Thresholding_Sklearn": SklearnSegmenter("otsu_thresholding"),
        "Niblack_Thresholding_Sklearn": SklearnSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "Sauvola_Thresholding_Sklearn": SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
        "Sobel_Sklearn": SklearnSegmenter("sobel_edge", threshold=0.1),
        "Canny_Sklearn": SklearnSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0, use_quantiles=False),
        # "Region_Growing_Sklearn": SklearnSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        # "Split_And_Merge_Sklearn": SklearnSegmenter("split_and_merge", min_size=50, threshold=0.1),
        # "Floodfill_Sklearn": SklearnSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        # "KMeans_Sklearn": SklearnSegmenter("kmeans_segmentation", k=3),
        # "DBSCAN_Sklearn": SklearnSegmenter("dbscan_segmentation", eps=0.1, min_samples=10),
        # "MeanShift_Sklearn": SklearnSegmenter("meanshift", bandwidth=0.5),
        # "Active_Contour_Sklearn": SklearnSegmenter("active_contour", alpha=0.015, beta=10, gamma=0.001, max_iterations=2000, w_edge=1, w_line=0),
        # "GVF_Sklearn": SklearnSegmenter("gvf_contour", mu=0.1, iterations=50),
        # "Morphological_Snakes_Sklearn": SklearnSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
        # "Chan_Vese_Sklearn": SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
        # "Watershed_Sklearn": SklearnSegmenter("watershed"),
        # "Random_Walker_Sklearn": SklearnSegmenter("random_walker", beta=10),
        # # "Quickshift_Sklearn": SklearnSegmenter("quickshift", kernel_size=5, max_dist=10, ratio=1.0),
        # "Slic_Sklearn": SklearnSegmenter("slic", n_segments=100, compactness=10.0),
        # "Felzenszwalb_Sklearn": SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
        # "GrabCut_Sklearn": SklearnSegmenter("grabcut"),
    }
    
    print("\n3. Загрузка методов PyTorch...")
    torch_methods = {
        "Global_Threshold_Torch": TorchSegmenter("global_thresholding", threshold=0.5),
        "Adaptive_Threshold_Torch": TorchSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_Thresholding_Torch": TorchSegmenter("otsu_thresholding"),
        "Niblack_Thresholding_Torch": TorchSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "Sauvola_Thresholding_Torch": TorchSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
        "Sobel_Torch": TorchSegmenter("sobel_edge", threshold=0.1),
        "Canny_Torch": TorchSegmenter("canny_edge", low=0.1, high=0.3),
        # "Region_Growing_Torch": TorchSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        # "Split_And_Merge_Torch": TorchSegmenter("split_and_merge", min_size=50, threshold=20),
        # "Floodfill_Torch": TorchSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        # "KMeans_Torch": TorchSegmenter("kmeans_segmentation", k=3),
        # "DBSCAN_Torch": TorchSegmenter("dbscan_segmentation", eps=0.5, min_samples=5),
        # "MeanShift_Torch": TorchSegmenter("meanshift", bandwidth=0.5, spatial_radius=35, color_radius=60),
        # "Active_Contour_Torch": TorchSegmenter("active_contour"),
        # "GVF_Torch": TorchSegmenter("gvf_contour"),
        # "Morphological_Snakes_Torch": TorchSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
        # "Chan_Vese_Torch": TorchSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100, dt=0.5, eps=1.0),
        # "Watershed_Torch": TorchSegmenter("watershed"),
        # "Random_Walker_Torch": TorchSegmenter("random_walker", beta=130, tol=1e-3, max_iter=300, target_label=2),
        # # "Quickshift_Torch": TorchSegmenter("quickshift", kernel_size=5, max_dist=10, ratio=1.0, sigma=0.0, convert2lab=True),
        # "Slic_Torch": TorchSegmenter("slic", n_segments=100, compactness=10.0, max_iter=10, sigma=0.0, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3.0),
        # "Felzenszwalb_Torch": TorchSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
        # "GrabCut_Torch": TorchSegmenter("grabcut", rect=(100, 100, 200, 200), num_iterations=5),
    }
    
    for name, segmenter in {**cv2_methods, **sklearn_methods, **torch_methods}.items():
        tester.add_method(name, segmenter)
        print(f"   ✅ {name}")

    print("🧹 Очистка памяти CUDA перед загрузкой тяжелой модели...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    print("\n4. Загрузка нейросетевых методов...")
    try:
        neural_segmenter = NeuralSegmenter(
            local_path="/home/yamshchikov/models/segformer-b5-ready"
        )
        tester.add_method("Neural_SegFormer", neural_segmenter)
        print(f"   ✅ Neural_SegFormer")
    except Exception as e:
        print(f"   ❌ Neural_SegFormer - ошибка: {e}")
        print(traceback.format_exc())
    
    print(f"\nВсего методов загружено: {len(tester.methods)}")
    
    # ============ 2. ЗАГРУЗКА ДАННЫХ ============
    print("\n2. Загрузка тестовых изображений...")
    test_images = load_test_images()
    
    # ============ 3. БЕНЧМАРК ============
    print("\n3. Бенчмарк производительности...")
    for img_name, (img_path, img_pil, gt) in test_images.items():
        img_array = np.array(img_pil)
        df_benchmark = tester.benchmark_methods(
            img_array,
            n_runs=3,
            test_name=f"benchmark_{img_name}",
            save_results=True
        )
        print(f"   ✅ Бенчмарк для {img_name} завершён")

    # ============ 4. ВАЛИДАЦИЯ (Torch vs OpenCV/Sklearn) ============
    print("\n4. Валидация реализаций...")
    validator = TorchImplementationValidator(output_dir="./data/validation")

    all_results = {}
    all_results = validator.validate_all_methods(test_images['countryside'][0])
    validator.generate_validation_report(all_results)
    print(f"\n✅ Все результаты сохранены в: {validator.output_dir}")
    
    # ============ 5. МАТРИЧНОЕ СРАВНЕНИЕ ============
    print("\n5. Матричное сравнение методов...")
    comparator = SegmentationComparator()

    all_segmenters = {**cv2_methods, **sklearn_methods, **torch_methods}

    methods_config_list = [
        {
            "name": name,
            "segmenter": segmenter,
            # "type": "custom" 
        }
        for name, segmenter in all_segmenters.items()
    ]

    for img_name, (img_path, img, gt) in test_images.items():
        img_array = np.array(img)
        results = comparator.matrix_comparison(
            image=img_array,
            methods_config=methods_config_list,
            comparison_type="all_vs_all",
            save_results=True,
            output_dir=f"./data/matrix_comparison_{img_name}"
        )
        print(f"   ✅ Матрица сравнения для {img_name}")
        print(f"      - Сравнено пар: {len(results['df_comparisons'])}")
    
    # ============ 6. СРАВНЕНИЕ С GROUND TRUTH (если есть) ============
    print("\n6. Сравнение с Ground Truth...")
    for img_name, (img_path, img, gt) in test_images.items():
        if gt is not None:
            metrics_all = {}
            for name, segmenter in {**cv2_methods, **sklearn_methods, **torch_methods}.items():
                pred_mask = segmenter.segment(img_path)
                metrics = SegmentationMetrics.calculate_all_metrics(pred_mask, gt)
                metrics_all[name] = metrics
            save_metrics_report(metrics_all, f"./data/gt_metrics_{img_name}.json")
            print(f"   ✅ Метрики для {img_name} сохранены")

    # ============ 7. ФИНАЛЬНЫЙ ОТЧЁТ ============
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"✓ Методов протестировано: {len(tester.methods)}")
    print(f"✓ Изображений обработано: {len(test_images)}")
    print(f"✓ Результаты в: ./data/")

    return tester, results, comparator

def load_test_images() -> Dict[str, Tuple[str, Image.Image, Optional[np.ndarray]]]:
    """
    Загружает тестовые изображения. Возвращает словарь:
    {
        'имя_изображения': (путь, PIL.Image, ground_truth_mask или None)
    }
    """
    
    test_images = {}
    
    # Примеры изображений с возможными ground truth
    image_urls = {
        "countryside": "https://i.pinimg.com/736x/17/e7/fc/17e7fc299466b2afd989e709fe7c9815.jpg",
        "nature": "https://i.pinimg.com/736x/f7/5a/f2/f75af26820b50c24600f50f3998eb02f.jpg",
        "architecture": "https://i.pinimg.com/736x/86/f6/07/86f60748d5d9ae4cb9092018d1321648.jpg",
        "trucks": "https://www.shutterstock.com/shutterstock/videos/1106252821/thumb/1.jpg?ip=x480",
        "traffic": "https://images.pond5.com/pov-car-and-truck-traffic-footage-190002081_iconl.jpeg",
        "mountain": "https://i.pinimg.com/736x/17/66/c4/1766c4f667af39f91172ef8eb21ab18a.jpg"
    }
    
    for name, url in image_urls.items():
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            
            local_path = f"test_image_{name}.jpg"
            img.save(local_path)
            
            # Для примера, будем считать что ground truth нет
            # На практике здесь можно было бы загрузить ground truth если он есть
            test_images[name] = (local_path, img, None)
            
            print(f"✅ {name}: {img.size}, ground truth: {'да' if None else 'нет'}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки {name}: {e}")
    
    return test_images

def test_with_ground_truth(
    tester: SegmentationTester, 
    image_path: str,
    ground_truth: np.ndarray,
    test_name: str
) -> Dict[str, Any]:
    """
    Тестирование с доступным ground truth
    """
    print(f"\n🔬 Тестирование с Ground Truth: {test_name}")
    
    # Инициализируем тестер с ground truth
    tester_with_gt = SegmentationTester(
        base_output_dir=f"results_with_gt_{test_name}",
        ground_truth_path=None
    )
    tester_with_gt.ground_truth_mask = ground_truth
    
    # Добавляем методы для сравнения
    add_segmentation_methods(tester_with_gt)
    
    # Выполняем сравнение с метриками
    results = tester_with_gt.compare_methods_with_metrics(
        image=image_path,
        method_names=get_methods_for_comparison(),
        test_name=f"gt_comparison_{test_name}",
        show_plots=True
    )
    return results

def test_without_ground_truth(
    tester: SegmentationTester,
    image_path: str,
    image: Image.Image,
    test_name: str
) -> Dict[str, Any]:
    """
    Тестирование без ground truth
    Используем несколько стратегий сравнения
    """
    print(f"\n🎯 Тестирование без Ground Truth: {test_name}")
    
    # Стратегия 1: Сравнение всех методов между собой
    print("\n1. Матричное сравнение методов между собой...")
    results_matrix = compare_methods_matrix(image_path, test_name)
    
    # Стратегия 2: Использование референсных методов (sklearn/scikit-image)
    print("\n2. Сравнение с референсными реализациями...")
    results_reference = compare_with_reference_implementations(image, test_name)
    
    # Стратегия 3: Оценка внутренней согласованности
    print("\n3. Оценка внутренней согласованности...")
    results_consistency = evaluate_consistency(image_path, test_name)
    
    # Стратегия 4: Сравнение на нескольких изображениях
    print("\n4. Кросс-валидация на нескольких изображениях...")
    results_cross = cross_image_comparison(image_path, test_name)
    
    return {
        'matrix_comparison': results_matrix,
        'reference_comparison': results_reference,
        'consistency_analysis': results_consistency,
        'cross_validation': results_cross
    }

def compare_methods_matrix(image_path: str, test_name: str) -> Dict[str, Any]:
    """
    Матричное сравнение всех методов между собой без ground truth
    """
    # Используем ExtendedSegmentationComparator
    comparator = SegmentationComparator()
    
    # Конфигурация методов для сравнения
    methods_config = [
        {"name": "kmeans", "type": "sklearn", "params": {"n_clusters": 3}},
        {"name": "dbscan", "type": "sklearn", "params": {"eps": 0.5, "min_samples": 5}},
        {"name": "meanshift", "type": "sklearn", "params": {"bandwidth": 0.5}},
        {"name": "felzenszwalb", "type": "skimage", "params": {"scale": 100, "sigma": 0.8}},
        {"name": "slic", "type": "skimage", "params": {"n_segments": 100}},
        {"name": "watershed", "type": "skimage", "params": {}},
        {"name": "threshold_otsu", "type": "skimage", "params": {}},
        {"name": "canny", "type": "skimage", "params": {"sigma": 1.0}},
    ]
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return {}
    
    # Выполняем матричное сравнение
    results = comparator.matrix_comparison(
        image=image,
        methods_config=methods_config,
        comparison_type="all_vs_all",
        output_dir=f"matrix_comparison_{test_name}",
        save_results=True
    )
    
    # Анализируем результаты
    if 'df_comparisons' in results:
        df = results['df_comparisons']
        
        print(f"\n📊 Матричное сравнение завершено:")
        print(f"  - Сравнено пар: {len(df)}")
        print(f"  - Методов: {len(results.get('masks', {}))}")
        
        # Находим наиболее похожие методы
        if not df.empty and 'f1_score' in df.columns:
            # Исключаем сравнение с самим собой
            valid_comparisons = df[df['method1'] != df['method2']]
            
            if not valid_comparisons.empty:
                most_similar = valid_comparisons.nlargest(3, 'f1_score')
                print(f"\n  Наиболее похожие методы (высокий F1):")
                for _, row in most_similar.iterrows():
                    print(f"    {row['method1']} vs {row['method2']}: F1={row['f1_score']:.3f}")
                
                least_similar = valid_comparisons.nsmallest(3, 'f1_score')
                print(f"\n  Наиболее разные методы (низкий F1):")
                for _, row in least_similar.iterrows():
                    print(f"    {row['method1']} vs {row['method2']}: F1={row['f1_score']:.3f}")
    
    return results

def compare_with_reference_implementations(image: Image.Image, test_name: str) -> Dict[str, Any]:
    """
    Сравнение кастомных реализаций с референсными (sklearn/scikit-image)
    """
    
    # Создаем компаратор
    comparator = SegmentationComparator()
    
    # Конвертируем PIL в numpy
    img_np = np.array(image)
    
    # Сравниваем несколько методов
    comparisons = []
    
    # Список сравнений: (кастомный_метод, референсный_метод, параметры)
    comparison_pairs = [
        ("kmeans_segmentation", "kmeans", {"k": 3}),
        ("otsu_thresholding", "threshold_otsu", {}),
        ("watershed", "watershed", {}),
        ("canny_edge", "canny", {"low": 50, "high": 150}),
    ]
    
    for custom_method, ref_method, params in comparison_pairs:
        try:
            # Кастомная реализация
            custom_segmenter = OpenCVSegmenter(custom_method, **params)
            custom_mask, _ = custom_segmenter.segment_with_mask(img_np)
            
            # Референсная реализация
            if ref_method in ["kmeans", "dbscan", "meanshift", "gmm"]:
                ref_mask, ref_info = comparator.segment_with_sklearn(
                    img_np, ref_method, **params)
            else:
                ref_mask, ref_info = comparator.segment_with_skimage(
                    img_np, ref_method, **params)
            
            # Вычисляем метрики
            metrics = comparator.compute_metrics(
                custom_mask, ref_mask, 
                f"Custom_{custom_method}", 
                f"Reference_{ref_method}"
            )
            
            comparisons.append({
                'custom_method': custom_method,
                'reference_method': ref_method,
                'metrics': metrics,
                'custom_mask': custom_mask,
                'reference_mask': ref_mask
            })
            
            print(f"  ✅ {custom_method} vs {ref_method}: "
                  f"F1={metrics.get('f1_score', 0):.3f}, "
                  f"IoU={metrics.get('jaccard', 0):.3f}")
            
        except Exception as e:
            print(f"  ❌ Ошибка сравнения {custom_method} vs {ref_method}: {e}")
    
    # Визуализация результатов
    if comparisons:
        visualize_reference_comparisons(comparisons, img_np, test_name)
    
    return {'comparisons': comparisons}

def visualize_reference_comparisons(comparisons: List[Dict], 
                                  original_image: np.ndarray,
                                  test_name: str):
    """Визуализирует сравнение кастомных и референсных методов"""
    n_comparisons = len(comparisons)
    
    if n_comparisons == 0:
        return
    
    fig, axes = plt.subplots(n_comparisons, 4, figsize=(16, n_comparisons * 4))
    
    if n_comparisons == 1:
        axes = axes.reshape(1, -1)
    
    for i, comparison in enumerate(comparisons):
        custom_mask = comparison['custom_mask']
        ref_mask = comparison['reference_mask']
        metrics = comparison['metrics']
        
        # Оригинал
        if len(original_image.shape) == 2:
            axes[i, 0].imshow(original_image, cmap='gray')
        else:
            axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f"{comparison['custom_method']}\nOriginal")
        axes[i, 0].axis('off')
        
        # Кастомная маска
        axes[i, 1].imshow(custom_mask, cmap='gray')
        axes[i, 1].set_title(f"Custom\n{comparison['custom_method']}")
        axes[i, 1].axis('off')
        
        # Референсная маска
        axes[i, 2].imshow(ref_mask, cmap='gray')
        axes[i, 2].set_title(f"Reference\n{comparison['reference_method']}")
        axes[i, 2].axis('off')
        
        # Разность
        diff = np.abs(custom_mask.astype(float) - ref_mask.astype(float))
        im = axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title(f"Difference\nF1={metrics.get('f1_score', 0):.3f}")
        axes[i, 3].axis('off')
        
        # Цветовая шкала для разности
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Сравнение кастомных и референсных реализаций - {test_name}", 
                 fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = f"reference_comparison_{test_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "comparison_summary.jpg"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Визуализация сохранена в {output_dir}/")

def evaluate_consistency(image_path: str, test_name: str) -> Dict[str, Any]:
    """
    Оценивает внутреннюю согласованность методов без ground truth
    """
    image = cv2.imread(image_path)
    if image is None:
        return {}
    
    methods = {
        "Global_Threshold_CV2": OpenCVSegmenter("global_thresholding", threshold=0.5),
        "Adaptive_Threshold_CV2": OpenCVSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_Thresholding_CV2": OpenCVSegmenter("otsu_thresholding"),
        "Niblack_Thresholding_CV2": OpenCVSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "Sauvola_Thresholding_CV2": OpenCVSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
        "Sobel_CV2": OpenCVSegmenter("sobel_edge", threshold=0.1),
        "Canny_CV2": OpenCVSegmenter("canny_edge", low=0.1, high=0.3),
        "Region_Growing_CV2": OpenCVSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        "Split_And_Merge_CV2": OpenCVSegmenter("split_and_merge", min_size=50, threshold=0.1),
        "Floodfill_CV2": OpenCVSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        "KMeans_CV2": OpenCVSegmenter("kmeans_segmentation", k=3),
        "DBSCAN_CV2": OpenCVSegmenter("dbscan_segmentation", eps=0.1, min_samples=10),
        "Meanshift_CV2": OpenCVSegmenter("meanshift", spatial_radius=35, color_radius=60),
        "Active_Contour_CV2": OpenCVSegmenter("active_contour", iterations=10),
        "GVF_CV2": OpenCVSegmenter("gvf_contour", mu=0.1, iterations=50),
        "Morphological_Snakes_CV2": OpenCVSegmenter("morphological_snakes", iterations=100),
        "Chan_Vese_CV2": OpenCVSegmenter("chan_vese", mu=0.25, max_iter=100),
        "Watershed_CV2": OpenCVSegmenter("watershed"),
        "Random_Walker_CV2": OpenCVSegmenter("random_walker"),
        # "Quickshift_CV2": OpenCVSegmenter("quickshift", bandwidth=0.5),
        "Slic_CV2": OpenCVSegmenter("slic", region_size=20, ruler=10.0),
        "Felzenszwalb_CV2": OpenCVSegmenter("felzenszwalb"),
        "GrabCut_CV2": OpenCVSegmenter("grabcut", num_iterations=10),
    }
    
    masks = {}
    execution_times = {}
    
    for name, segmenter in methods.items():
        try:
            start_time = datetime.now()
            mask, _ = segmenter.segment_with_mask(image)
            exec_time = (datetime.now() - start_time).total_seconds()
            
            masks[name] = mask
            execution_times[name] = exec_time
            
            print(f"  ✅ {name}: {exec_time:.3f}s")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    # Вычисляем метрики согласованности
    consistency_metrics = {}
    
    if len(masks) > 1:
        method_names = list(masks.keys())
        n_methods = len(method_names)
        
        # Матрица попарного согласия
        agreement_matrix = np.zeros((n_methods, n_methods))
        
        for i, m1 in enumerate(method_names):
            for j, m2 in enumerate(method_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    # Простое согласие по пикселям
                    mask1_bin = (masks[m1] > 127).astype(np.uint8).flatten()
                    mask2_bin = (masks[m2] > 127).astype(np.uint8).flatten()
                    
                    agreement = np.mean(mask1_bin == mask2_bin)
                    agreement_matrix[i, j] = agreement
        
        # Вычисляем среднее согласие для каждого метода
        mean_agreement = {}
        for i, m in enumerate(method_names):
            # Исключаем само согласие
            other_indices = [j for j in range(n_methods) if j != i]
            mean_agreement[m] = np.mean(agreement_matrix[i, other_indices])
        
        # Сводная статистика
        consistency_metrics = {
            'agreement_matrix': agreement_matrix,
            'mean_agreement': mean_agreement,
            'method_names': method_names,
            'execution_times': execution_times,
            'overall_mean_agreement': np.mean(list(mean_agreement.values())),
            'overall_std_agreement': np.std(list(mean_agreement.values()))
        }
        
        # Визуализация матрицы согласия
        visualize_consistency_matrix(agreement_matrix, method_names, test_name)
        
        print(f"\n📊 Метрики согласованности:")
        print(f"  Среднее согласие: {consistency_metrics['overall_mean_agreement']:.3f}")
        print(f"  Стандартное отклонение: {consistency_metrics['overall_std_agreement']:.3f}")
        
        print(f"\n  Согласие по методам:")
        for method, agreement in sorted(mean_agreement.items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"    {method}: {agreement:.3f}")
    
    return consistency_metrics

def visualize_consistency_matrix(
    matrix: np.ndarray, 
    method_names: List[str],
    test_name: str
):
    """Визуализирует матрицу согласия методов"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Сокращаем имена для подписей
    short_names = [name[:10] for name in method_names]
    
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(method_names)))
    ax.set_yticks(range(len(method_names)))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_yticklabels(short_names)
    
    # Добавляем значения в ячейки
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                         ha="center", va="center",
                         color="black" if matrix[i, j] < 0.7 else "white",
                         fontsize=9)
    
    ax.set_title("Матрица согласия методов", fontsize=14)
    plt.colorbar(im, ax=ax, label='Согласие')
    plt.tight_layout()
    
    output_dir = f"consistency_analysis_{test_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "agreement_matrix.jpg"),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Матрица согласия сохранена в {output_dir}/")

def cross_image_comparison(image_path: str, test_name: str) -> Dict[str, Any]:
    """
    Кросс-валидация на нескольких изображениях
    """
    
    image_dir = os.path.dirname(image_path) or "."
    all_images = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                 glob.glob(os.path.join(image_dir, "*.png")) + \
                 glob.glob(os.path.join(image_dir, "*.jpeg"))
    
    max_images = 5
    test_images = [image_path] + all_images[:max_images-1] if len(all_images) > 1 else [image_path]
    
    print(f"\n📸 Кросс-валидация на {len(test_images)} изображениях")
    
    # Методы для тестирования
    methods = {
        "Global_Threshold_CV2": OpenCVSegmenter("global_thresholding", threshold=0.5),
        "Adaptive_Threshold_CV2": OpenCVSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_Thresholding_CV2": OpenCVSegmenter("otsu_thresholding"),
        "Niblack_Thresholding_CV2": OpenCVSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "Sauvola_Thresholding_CV2": OpenCVSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
        "Sobel_CV2": OpenCVSegmenter("sobel_edge", threshold=0.1),
        "Canny_CV2": OpenCVSegmenter("canny_edge", low=0.1, high=0.3),
        "Region_Growing_CV2": OpenCVSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        "Split_And_Merge_CV2": OpenCVSegmenter("split_and_merge", min_size=50, threshold=0.1),
        "Floodfill_CV2": OpenCVSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        "KMeans_CV2": OpenCVSegmenter("kmeans_segmentation", k=3),
        "DBSCAN_CV2": OpenCVSegmenter("dbscan_segmentation", eps=0.1, min_samples=10),
        "Meanshift_CV2": OpenCVSegmenter("meanshift", spatial_radius=35, color_radius=60),
        "Active_Contour_CV2": OpenCVSegmenter("active_contour", iterations=10),
        "GVF_CV2": OpenCVSegmenter("gvf_contour", mu=0.1, iterations=50),
        "Morphological_Snakes_CV2": OpenCVSegmenter("morphological_snakes", iterations=100),
        "Chan_Vese_CV2": OpenCVSegmenter("chan_vese", mu=0.25, max_iter=100),
        "Watershed_CV2": OpenCVSegmenter("watershed"),
        "Random_Walker_CV2": OpenCVSegmenter("random_walker"),
        # "Quickshift_CV2": OpenCVSegmenter("quickshift", bandwidth=0.5),
        "Slic_CV2": OpenCVSegmenter("slic", region_size=20, ruler=10.0),
        "Felzenszwalb_CV2": OpenCVSegmenter("felzenszwalb"),
        "GrabCut_CV2": OpenCVSegmenter("grabcut", num_iterations=10),
    }
    
    # Собираем результаты по всем изображениям
    all_results = []
    
    for img_idx, img_path in enumerate(test_images):
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            img_name = os.path.basename(img_path)
            print(f"  Обработка {img_name}...")
            
            img_results = {'image': img_name, 'methods': {}}
            
            for method_name, method_factory in methods.items():
                try:
                    segmenter = method_factory()
                    mask, _ = segmenter.segment_with_mask(image)
                    
                    # Базовые метрики маски
                    mask_binary = mask > 127
                    area = np.sum(mask_binary)
                    coverage = area / mask_binary.size * 100
                    
                    img_results['methods'][method_name] = {
                        'area': int(area),
                        'coverage': float(coverage),
                        'mask_shape': mask.shape
                    }
                    
                except Exception as e:
                    print(f"    ❌ {method_name} на {img_name}: {e}")
            
            all_results.append(img_results)
            
        except Exception as e:
            print(f"  ❌ Ошибка обработки {img_path}: {e}")
    
    # Анализируем стабильность методов
    stability_analysis = {}
    
    for method_name in methods.keys():
        # Собираем покрытия для данного метода на всех изображениях
        coverages = []
        
        for img_result in all_results:
            if method_name in img_result['methods']:
                coverages.append(img_result['methods'][method_name]['coverage'])
        
        if coverages:
            stability_analysis[method_name] = {
                'mean_coverage': np.mean(coverages),
                'std_coverage': np.std(coverages),
                'cv_coverage': np.std(coverages) / np.mean(coverages) * 100 if np.mean(coverages) > 0 else 0,
                'n_images': len(coverages)
            }
    
    # Визуализируем стабильность
    visualize_stability_analysis(stability_analysis, test_name)
    
    print(f"\n📊 Анализ стабильности методов:")
    for method_name, stats in stability_analysis.items():
        print(f"  {method_name}:")
        print(f"    Среднее покрытие: {stats['mean_coverage']:.1f}%")
        print(f"    Стандартное отклонение: {stats['std_coverage']:.1f}%")
        print(f"    Коэффициент вариации: {stats['cv_coverage']:.1f}%")
    
    return {
        'all_results': all_results,
        'stability_analysis': stability_analysis,
        'n_images_processed': len(all_results)
    }

def visualize_stability_analysis(stability_analysis: Dict[str, Any], test_name: str):
    """Визуализирует анализ стабильности методов"""
    if not stability_analysis:
        return
    
    methods = list(stability_analysis.keys())
    mean_coverages = [stats['mean_coverage'] for stats in stability_analysis.values()]
    std_coverages = [stats['std_coverage'] for stats in stability_analysis.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: Среднее покрытие с ошибками
    x_pos = np.arange(len(methods))
    ax1.bar(x_pos, mean_coverages, yerr=std_coverages, 
           capsize=5, alpha=0.7, color='skyblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45)
    ax1.set_ylabel('Покрытие (%)')
    ax1.set_title('Среднее покрытие маски ± стандартное отклонение')
    ax1.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, (mean, std) in enumerate(zip(mean_coverages, std_coverages)):
        ax1.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # График 2: Коэффициент вариации
    cv_values = [stats['cv_coverage'] for stats in stability_analysis.values()]
    ax2.bar(x_pos, cv_values, alpha=0.7, color='lightcoral')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45)
    ax2.set_ylabel('Коэффициент вариации (%)')
    ax2.set_title('Коэффициент вариации покрытия (ниже = стабильнее)')
    ax2.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, cv in enumerate(cv_values):
        ax2.text(i, cv + 0.5, f'{cv:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Анализ стабильности методов сегментации - {test_name}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = f"stability_analysis_{test_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "stability_analysis.jpg"),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Анализ стабильности сохранен в {output_dir}/")

def add_segmentation_methods(tester: SegmentationTester):
    """Добавляет методы сегментации в тестер"""
    
    print("\nДобавление методов сегментации...")
    
    # CV2/Sklearn методы
    cv2_methods = {
        "Global_Threshold_CV2": OpenCVSegmenter("global_thresholding", threshold=0.5),
        "Adaptive_Threshold_CV2": OpenCVSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_Thresholding_CV2": OpenCVSegmenter("otsu_thresholding"),
        "Niblack_Thresholding_CV2": OpenCVSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "Sauvola_Thresholding_CV2": OpenCVSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
        "Sobel_CV2": OpenCVSegmenter("sobel_edge", threshold=0.1),
        "Canny_CV2": OpenCVSegmenter("canny_edge", low=0.1, high=0.3),
        "Region_Growing_CV2": OpenCVSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        "Split_And_Merge_CV2": OpenCVSegmenter("split_and_merge", min_size=50, threshold=0.1),
        "Floodfill_CV2": OpenCVSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        "KMeans_CV2": OpenCVSegmenter("kmeans_segmentation", k=3),
        "DBSCAN_CV2": OpenCVSegmenter("dbscan_segmentation", eps=0.1, min_samples=10),
        "Meanshift_CV2": OpenCVSegmenter("meanshift", spatial_radius=35, color_radius=60),
        "Active_Contour_CV2": OpenCVSegmenter("active_contour", iterations=10),
        "GVF_CV2": OpenCVSegmenter("gvf_contour", mu=0.1, iterations=50),
        "Morphological_Snakes_CV2": OpenCVSegmenter("morphological_snakes", iterations=100),
        "Chan_Vese_CV2": OpenCVSegmenter("chan_vese", mu=0.25, max_iter=100),
        "Watershed_CV2": OpenCVSegmenter("watershed"),
        "Random_Walker_CV2": OpenCVSegmenter("random_walker"),
        # "Quickshift_CV2": OpenCVSegmenter("quickshift", bandwidth=0.5),
        "Slic_CV2": OpenCVSegmenter("slic", region_size=20, ruler=10.0),
        "Felzenszwalb_CV2": OpenCVSegmenter("felzenszwalb"),
        "GrabCut_CV2": OpenCVSegmenter("grabcut", num_iterations=10),
    }

    # Sklearn методы С ПАРАМЕТРАМИ
    print("\n2. Загрузка методов SKlearn...")
    sklearn_methods = {
        "Global_Threshold_Sklearn": SklearnSegmenter("global_thresholding", threshold=0.5),
        "Adaptive_Threshold_Sklearn": SklearnSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_Thresholding_Sklearn": SklearnSegmenter("otsu_thresholding"),
        "Niblack_Thresholding_Sklearn": SklearnSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "Sauvola_Thresholding_Sklearn": SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
        "Sobel_Sklearn": SklearnSegmenter("sobel_edge", threshold=0.1),
        "Canny_Sklearn": SklearnSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0, use_quantiles=False),
        "Region_Growing_Sklearn": SklearnSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        "Split_And_Merge_Sklearn": SklearnSegmenter("split_and_merge", min_size=50, threshold=0.1),
        "Floodfill_Sklearn": SklearnSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        "KMeans_Sklearn": SklearnSegmenter("kmeans_segmentation", k=3),
        "DBSCAN_Sklearn": SklearnSegmenter("dbscan_segmentation", eps=0.1, min_samples=10),
        "MeanShift_Sklearn": SklearnSegmenter("meanshift", bandwidth=0.5),
        "Active_Contour_Sklearn": SklearnSegmenter("active_contour", alpha=0.015, beta=10, gamma=0.001, max_iterations=2000, w_edge=1, w_line=0),
        "GVF_Sklearn": SklearnSegmenter("gvf_contour", mu=0.1, iterations=50),
        "Morphological_Snakes_Sklearn": SklearnSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
        "Chan_Vese_Sklearn": SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
        "Watershed_Sklearn": SklearnSegmenter("watershed"),
        "Random_Walker_Sklearn": SklearnSegmenter("random_walker", beta=10),
        # "Quickshift_Sklearn": SklearnSegmenter("quickshift", kernel_size=5, max_dist=10, ratio=1.0),
        "Slic_Sklearn": SklearnSegmenter("slic", n_segments=100, compactness=10.0),
        "Felzenszwalb_Sklearn": SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
        "GrabCut_Sklearn": SklearnSegmenter("grabcut"),
    }
    
    for name, segmenter in {**cv2_methods, **sklearn_methods}.items():
        tester.add_method(name, segmenter)
        print(f"  ✅ {name}")
    
    # PyTorch методы (если доступны)
    try:
        torch_methods = {
            "Global_Threshold_Torch": TorchSegmenter("global_thresholding", threshold=0.5),
            "Adaptive_Threshold_Torch": TorchSegmenter("adaptive_thresholding", block_size=11, C=2),
            "Otsu_Thresholding_Torch": TorchSegmenter("otsu_thresholding"),
            "Niblack_Thresholding_Torch": TorchSegmenter("threshold_niblack", window_size=15, k=-0.2),
            "Sauvola_Thresholding_Torch": TorchSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
            "Sobel_Torch": TorchSegmenter("sobel_edge", threshold=0.1),
            "Canny_Torch": TorchSegmenter("canny_edge", low=0.1, high=0.3),
            "Region_Growing_Torch": TorchSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
            "Split_And_Merge_Torch": TorchSegmenter("split_and_merge", min_size=50, threshold=20),
            "Floodfill_Torch": TorchSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
            "KMeans_Torch": TorchSegmenter("kmeans_segmentation", k=3),
            "DBSCAN_Torch": TorchSegmenter("dbscan_segmentation", eps=0.5, min_samples=5),
            "MeanShift_Torch": TorchSegmenter("meanshift", bandwidth=0.5, spatial_radius=35, color_radius=60),
            "Active_Contour_Torch": TorchSegmenter("active_contour"),
            "GVF_Torch": TorchSegmenter("gvf_contour"),
            "Morphological_Snakes_Torch": TorchSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
            "Chan_Vese_Torch": TorchSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100, dt=0.5, eps=1.0),
            "Watershed_Torch": TorchSegmenter("watershed"),
            "Random_Walker_Torch": TorchSegmenter("random_walker", beta=130, tol=1e-3, max_iter=300, target_label=2),
            # "Quickshift_Torch": TorchSegmenter("quickshift", kernel_size=5, max_dist=10, ratio=1.0, sigma=0.0, convert2lab=True),
            "Slic_Torch": TorchSegmenter("slic", n_segments=100, compactness=10.0, max_iter=10, sigma=0.0, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3.0),
            "Felzenszwalb_Torch": TorchSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
            "GrabCut_Torch": TorchSegmenter("grabcut", rect=(100, 100, 200, 200), num_iterations=5),
        }
            
        for name, segmenter in torch_methods.items():
            tester.add_method(name, segmenter)
            print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ⚠️ PyTorch методы недоступны: {e}")
    
    # Нейросетевая сегментация (если доступна)
    try:
        neural_segmenter = NeuralSegmenter(
            local_path="/home/yamshchikov/models/segformer-b5-ready"
        )
        tester.add_method("Neural_SegFormer", neural_segmenter)
        print(f"  ✅ Neural_SegFormer")
    except Exception as e:
        print(f"  ⚠️ Нейросетевая сегментация недоступна: {e}")

def get_methods_for_comparison() -> List[str]:
    """Возвращает список методов для сравнения"""
    return [
        "Global_Threshold_CV2",
        "Otsu_CV2",
        "KMeans_CV2",
        "Watershed_CV2",
        "Canny_CV2",
        "GrabCut_CV2",
    ]

def create_summary_report(test_results: Dict[str, Any], output_dir: str = "summary_report"):
    """
    Создает сводный отчет по всем тестам
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем HTML отчет
    html_path = os.path.join(output_dir, "summary_report.html")
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Сводный отчет по сегментации</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .test-case { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metrics-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                .metrics-table th { background-color: #f2f2f2; }
                .good { background-color: #d4edda; }
                .medium { background-color: #fff3cd; }
                .poor { background-color: #f8d7da; }
                .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
                .image-grid img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>📊 Сводный отчет по тестированию методов сегментации</h1>
            <div class="summary">
                <p><strong>Дата тестирования:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p><strong>Всего тестов:</strong> """ + str(len(test_results)) + """</p>
            </div>
        """)
        
        # Добавляем результаты по каждому тесту
        for test_name, test_data in test_results.items():
            f.write(f"""
            <div class="test-case">
                <h2>{test_name}</h2>
            """)
            
            # Добавляем информацию о тесте
            if 'has_ground_truth' in test_data:
                if test_data['has_ground_truth']:
                    f.write("<p><strong>Тип:</strong> С Ground Truth</p>")
                else:
                    f.write("<p><strong>Тип:</strong> Без Ground Truth</p>")
            
            f.write("</div>")
        
        f.write("""
            <footer>
                <p>Сгенерировано автоматически. Для детальной информации смотрите папки с результатами.</p>
            </footer>
        </body>
        </html>
        """)
    
    print(f"📄 Сводный отчет создан: {html_path}")

def test_neural_segmentation_variants():
    """Тестирование различных вариантов нейросетевой сегментации"""
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ВАРИАНТОВ НЕЙРОСЕТЕВОЙ СЕГМЕНТАЦИИ")
    print("=" * 60)
    
    # Загрузка тестового изображения
    img_url = "https://i.pinimg.com/736x/17/e7/fc/1D7oZ9cqSef531ErnBAai8ZivwSPyqMCcs.jpg"
    
    try:
        response = requests.get(img_url)
        test_image = Image.open(BytesIO(response.content))
        
        # Создаем нейросетевой сегментатор
        segmenter = NeuralSegmenter(
            local_path="/home/yamshchikov/models/segformer-b5-ready"
        )
        
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
        
    except Exception as e:
        print(f"❌ Ошибка тестирования нейросетевых вариантов: {e}")
        print(traceback.format_exc())
        return None, None

def test_with_metrics():
    """Тестирование с метриками качества"""
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ С МЕТРИКАМИ КАЧЕСТВА")
    print("="*60)

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
    local_image_path = "test_image_0.jpg"
    image.save(local_image_path)
    print(f"✅ Изображение сохранено локально: {local_image_path}")

    local_mask_path = "test_mask_image_0.png"
    segmentation_map.save(local_mask_path)
    print(f"✅ Изображение маски сохранено локально: {local_mask_path}")
    
    # Инициализация тестера с ground truth
    tester = SegmentationTester(
        base_output_dir="segmentation_metrics_results",
        ground_truth_path=local_mask_path  # Путь к вашей ground truth маске
    )
    
    # Добавление методов
    # CV2/Sklearn методы
    cv2_methods = {
        "Global_Threshold_CV2": OpenCVSegmenter("global_thresholding", threshold=0.5),
        "Adaptive_Threshold_CV2": OpenCVSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_Thresholding_CV2": OpenCVSegmenter("otsu_thresholding"),
        "Niblack_Thresholding_CV2": OpenCVSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "Sauvola_Thresholding_CV2": OpenCVSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
        "Sobel_CV2": OpenCVSegmenter("sobel_edge", threshold=0.1),
        "Canny_CV2": OpenCVSegmenter("canny_edge", low=0.1, high=0.3),
        "Region_Growing_CV2": OpenCVSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        "Split_And_Merge_CV2": OpenCVSegmenter("split_and_merge", min_size=50, threshold=0.1),
        "Floodfill_CV2": OpenCVSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        "KMeans_CV2": OpenCVSegmenter("kmeans_segmentation", k=3),
        "DBSCAN_CV2": OpenCVSegmenter("dbscan_segmentation", eps=0.1, min_samples=10),
        "Meanshift_CV2": OpenCVSegmenter("meanshift", spatial_radius=35, color_radius=60),
        "Active_Contour_CV2": OpenCVSegmenter("active_contour", iterations=10),
        "GVF_CV2": OpenCVSegmenter("gvf_contour", mu=0.1, iterations=50),
        "Morphological_Snakes_CV2": OpenCVSegmenter("morphological_snakes", iterations=100),
        "Chan_Vese_CV2": OpenCVSegmenter("chan_vese", mu=0.25, max_iter=100),
        "Watershed_CV2": OpenCVSegmenter("watershed"),
        "Random_Walker_CV2": OpenCVSegmenter("random_walker"),
        # "Quickshift_CV2": OpenCVSegmenter("quickshift", bandwidth=0.5),
        "Slic_CV2": OpenCVSegmenter("slic", region_size=20, ruler=10.0),
        "Felzenszwalb_CV2": OpenCVSegmenter("felzenszwalb"),
        "GrabCut_CV2": OpenCVSegmenter("grabcut", num_iterations=10),
    }

    # Sklearn методы С ПАРАМЕТРАМИ
    print("\n2. Загрузка методов SKlearn...")
    sklearn_methods = {
        "Global_Threshold_Sklearn": SklearnSegmenter("global_thresholding", threshold=0.5),
        "Adaptive_Threshold_Sklearn": SklearnSegmenter("adaptive_thresholding", block_size=11, C=2),
        "Otsu_Thresholding_Sklearn": SklearnSegmenter("otsu_thresholding"),
        "Niblack_Thresholding_Sklearn": SklearnSegmenter("threshold_niblack", window_size=15, k=-0.2),
        "Sauvola_Thresholding_Sklearn": SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
        "Sobel_Sklearn": SklearnSegmenter("sobel_edge", threshold=0.1),
        "Canny_Sklearn": SklearnSegmenter("canny_edge", low=0.1, high=0.3, sigma=1.0, use_quantiles=False),
        "Region_Growing_Sklearn": SklearnSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        "Split_And_Merge_Sklearn": SklearnSegmenter("split_and_merge", min_size=50, threshold=0.1),
        "Floodfill_Sklearn": SklearnSegmenter("floodfill", seed=(100, 100), tolerance=0.15),
        "KMeans_Sklearn": SklearnSegmenter("kmeans_segmentation", k=3),
        "DBSCAN_Sklearn": SklearnSegmenter("dbscan_segmentation", eps=0.1, min_samples=10),
        "MeanShift_Sklearn": SklearnSegmenter("meanshift", bandwidth=0.5),
        "Active_Contour_Sklearn": SklearnSegmenter("active_contour", alpha=0.015, beta=10, gamma=0.001, max_iterations=2000, w_edge=1, w_line=0),
        "GVF_Sklearn": SklearnSegmenter("gvf_contour", mu=0.1, iterations=50),
        "Morphological_Snakes_Sklearn": SklearnSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
        "Chan_Vese_Sklearn": SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
        "Watershed_Sklearn": SklearnSegmenter("watershed"),
        "Random_Walker_Sklearn": SklearnSegmenter("random_walker", beta=10),
        # "Quickshift_Sklearn": SklearnSegmenter("quickshift", kernel_size=5, max_dist=10, ratio=1.0),
        "Slic_Sklearn": SklearnSegmenter("slic", n_segments=100, compactness=10.0),
        "Felzenszwalb_Sklearn": SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
        "GrabCut_Sklearn": SklearnSegmenter("grabcut"),
    }
    
    for name, segmenter in cv2_methods.items():
        tester.add_method(name, segmenter)
        print(f"   ✅ {name}")


    neural_segmenter = NeuralSegmenter(
        local_path="/home/yamshchikov/models/segformer-b5-ready"
    )
    tester.add_method("Neural_SegFormer", neural_segmenter)
    print(f"   ✅ Neural_SegFormer Added")

    method_names = ["Global_Threshold_CV2",
                    "Adaptive_Threshold_CV2",
                    "Otsu_CV2",
                    "Region_Growing_CV2",
                    "Split_and_Merge_CV2",
                    "Sobel_CV2", 
                    "Canny_CV2",
                    "KMeans_CV2",
                    "Active_Contour_CV2",
                    "GVF_Contour_CV2",
                    "GVF_Contour_CV2",
                    "Watershed_CV2",
                    "GrabCut_CV2",
                    "FloodFill_CV2",
                    "Morphological_Snakes_CV2",
                    "Slic_CV2",
                    "Felzenszwalb_CV2",
                    "Chan_Vese_CV2",
                    "Threshold_Niblack_CV2",
                    "Threshold_Sauvola_CV2",
                    "Random_Walker_CV2",
                    "Neural_SegFormer"
                    ]
    
    # Сравнение с метриками
    results = tester.compare_methods_with_metrics(
        image=local_image_path,
        method_names=method_names,
        test_name="metrics_comparison",
        show_plots=True
    )
    
    return tester, results

def save_metrics_report(metrics_all: Dict, path: str):
    """Сохранение отчёта с метриками"""
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False, default=str)

if __name__ == "__main__":
    # Основной тест
    print("ЗАПУСК ОСНОВНОГО ТЕСТА")
    print("=" * 60)
    tester, results, comparator = main()
    # tester, results = test_with_metrics()
    # Дополнительный тест нейросетевых вариантов
    # print("\n\nЗАПУСК ДОПОЛНИТЕЛЬНОГО ТЕСТА НЕЙРОСЕТЕВЫХ ВАРИАНТОВ")
    # print("=" * 60)
    # segmenter, detailed_result = test_neural_segmentation_variants()

