# main.py
from SegmentationTester import SegmentationTester
from TorchSegmenter import TorchSegmenter
from NeuralSegmenter import NeuralSegmenter
from cv2SklearnSegmenter import CV2SklearnSegmenter
from HelpModule import create_advanced_pipeline_example, analyze_pipeline_results, print_pipeline_analysis, compare_segmentation_methods_with_timing, original_compare_segmentation_methods
import pandas as pd
from typing import Union, Dict, Any
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import os
import shutil
import datetime
import traceback


def main() -> Union[SegmentationTester, Dict[str, Any], pd.DataFrame]:
    """Тестирование использования всех классов"""
    
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
        "Split_and_Merge_CV2": CV2SklearnSegmenter("split_and_merge", min_region_size=50, threshold=20),
        "Sobel_CV2": CV2SklearnSegmenter("sobel_edge", threshold=50),
        "Canny_CV2": CV2SklearnSegmenter("canny_edge", low=50, high=150),
        "KMeans_CV2": CV2SklearnSegmenter("kmeans_segmentation", k=3),
        # "DBSCAN_CV2": CV2SklearnSegmenter("dbscan_segmentation", eps=10, min_samples=100),
        "Active_Contour_CV2": CV2SklearnSegmenter("active_contour", alpha=0.01, beta=0.1, gamma=0.001, max_iterations=2000),
        "GVF_Contour_CV2": CV2SklearnSegmenter("gvf_contour", mu=0.2, iterations=100),
        "Watershed_CV2": CV2SklearnSegmenter("watershed"),
        # "Meanshift_CV2": CV2SklearnSegmenter("meanshift", bandwidth=0.5),
        "GrabCut_CV2": CV2SklearnSegmenter("grabcut", iterations=10),
        "FloodFill_CV2": CV2SklearnSegmenter("floodfill", seed=(100, 100), tolerance=20),
        "Morphological_Snakes_CV2": CV2SklearnSegmenter("morphological_snakes", iterations=100, smoothing=1, threshold=0.5),
        # "Quickshift_CV2": CV2SklearnSegmenter("quickshift", bandwidth=0.5),
        "Slic_CV2": CV2SklearnSegmenter("slic", n_segments=100, compactness=10.0),
        "Felzenszwalb_CV2": CV2SklearnSegmenter("felzenszwalb", scale=100, sigma=0.8, min_size=50),
        "Chan_Vese_CV2": CV2SklearnSegmenter("chan_vese", mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3, max_iter=100),
        "Threshold_Niblack_CV2": CV2SklearnSegmenter("threshold_niblack", window_size=15, k=0.2),
        "Threshold_Sauvola_CV2": CV2SklearnSegmenter("threshold_sauvola", window_size=15, k=0.2, r=128),
        "Random_Walker_CV2": CV2SklearnSegmenter("random_walker", scale=100, sigma=0.8, min_size=50),
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
        # "Region_Growing_Torch": TorchSegmenter("region_growing", seed=(100, 100), tolerance=0.1),
        "Split_and_Merge_Torch": TorchSegmenter("split_and_merge", min_size=50),
        "Sobel_Torch": TorchSegmenter("sobel_edge", threshold=0.1),
        "Canny_Torch": TorchSegmenter("canny_edge", low=0.1, high=0.3),
        "KMeans_Torch": TorchSegmenter("kmeans_segmentation", k=3),
        # "DBSCAN_Torch": TorchSegmenter("dbscan_segmentation", eps=10, min_samples=100),
        # "Active_Contour_Torch": TorchSegmenter("active_contour"),
        # "GVF_Contour_Torch": TorchSegmenter("gvf_contour"),
        "Watershed_Torch": TorchSegmenter("watershed"),
        # "Meanshift_Torch": TorchSegmenter("meanshift", bandwidth=0.5, spatial_radius=35, color_radius=60),
        "Grabcut_Torch": TorchSegmenter("grabcut", rect=(100, 100, 200, 200), num_iterations=5),
        "FloodFill_Torch": TorchSegmenter("floodfill", tolerance=0.15),
    }
    
    for name, segmenter in torch_methods.items():
        tester.add_method(name, segmenter)
        print(f"   ✅ {name}")

    # 3. Нейросетевая сегментация
    print("\n3. Загрузка нейросетевых методов...")
    try:
        # Используйте локальный путь к модели
        neural_segmenter = NeuralSegmenter(
            local_path="/home/yamshchikov/models/segformer-b5-ready"
        )
        tester.add_method("Neural_SegFormer", neural_segmenter)
        print(f"   ✅ Neural_SegFormer")
    except Exception as e:
        print(f"   ❌ Neural_SegFormer - ошибка: {e}")
        print(traceback.format_exc())
    
    print(f"\nВсего методов загружено: {len(tester.methods)}")
    
    # ============ ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ ============
    
    print("\n" + "=" * 60)
    print("ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ")
    print("=" * 60)
    
    # Загрузка тестового изображения
    try:
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

        print("Для дополнительных тестов Используем тестовое изображение по умолчанию...")
        img_url_1 = "https://i.pinimg.com/736x/17/e7/fc/17e7fc299466b2afd989e709fe7c9815.jpg"
        local_image_path_1 = "test_image_download_1.jpg"
        try:
            response_1 = requests.get(img_url_1)
            image_1 = Image.open(BytesIO(response_1.content))
            image_1.save(local_image_path_1)
            print(f"✅ Изображение загружено из URL: {img_url_1}")
            print(f"   Размер: {image_1.size}")
        except Exception as e1:
            print(f"❌ Ошибка загрузки из URL: {e1}")
            raise

        img_url_2 = "https://www.shutterstock.com/shutterstock/videos/1106252821/thumb/1.jpg?ip=x480"
        local_image_path_2 = "test_image_download_2.jpg"
        
        try:
            response_2 = requests.get(img_url_2)
            image_2 = Image.open(BytesIO(response_2.content))
            image_2.save(local_image_path_2)
            print(f"✅ Изображение загружено из URL: {img_url_2}")
            print(f"   Размер: {image_2.size}")
        except Exception as e2:
            print(f"❌ Ошибка загрузки из URL: {e2}")
            raise

        img_url_3 = "https://i.pinimg.com/736x/86/f6/07/86f60748d5d9ae4cb9092018d1321648.jpg"
        local_image_path_3 = "test_image_download_3.jpg"
        
        try:
            response_3 = requests.get(img_url_3)
            image_3 = Image.open(BytesIO(response_3.content))
            image_3.save(local_image_path_3)
            print(f"✅ Изображение загружено из URL: {img_url_3}")
            print(f"   Размер: {image_3.size}")
        except Exception as e3:
            print(f"❌ Ошибка загрузки из URL: {e3}")
            raise

        img_url_4 = "https://images.pond5.com/pov-car-and-truck-traffic-footage-190002081_iconl.jpeg"
        local_image_path_4 = "test_image_download_4.jpg"
        
        try:
            response_4 = requests.get(img_url_4)
            image_4 = Image.open(BytesIO(response_4.content))
            image_4.save(local_image_path_4)
            print(f"✅ Изображение загружено из URL: {img_url_4}")
            print(f"   Размер: {image_4.size}")
        except Exception as e4:
            print(f"❌ Ошибка загрузки из URL: {e4}")
            raise

        img_url_5 = "https://i.pinimg.com/736x/17/66/c4/1766c4f667af39f91172ef8eb21ab18a.jpg"
        local_image_path_5 = "test_image_download_5.jpg"
        
        try:
            response_5 = requests.get(img_url_5)
            image_5 = Image.open(BytesIO(response_5.content))
            image_5.save(local_image_path_5)
            print(f"✅ Изображение загружено из URL: {img_url_5}")
            print(f"   Размер: {image_5.size}")
        except Exception as e5:
            print(f"❌ Ошибка загрузки из URL: {e5}")
            raise

        img_url_6 = "https://i.pinimg.com/736x/f7/5a/f2/f75af26820b50c24600f50f3998eb02f.jpg"
        local_image_path_6 = "test_image_download_6.jpg"
        
        try:
            response_6 = requests.get(img_url_6)
            image_6 = Image.open(BytesIO(response_6.content))
            image_6.save(local_image_path_6)
            print(f"✅ Изображение загружено из URL: {img_url_6}")
            print(f"   Размер: {image_6.size}")
        except Exception as e6:
            print(f"❌ Ошибка загрузки из URL: {e6}")
            raise
        
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
    
    # ============ ТЕСТИРОВАНИЕ НЕЙРОСЕТЕВОЙ СЕГМЕНТАЦИИ ============
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ НЕЙРОСЕТЕВОЙ СЕГМЕНТАЦИИ")
    print("=" * 60)
    
    if "Neural_SegFormer" in tester.methods:
        try:
            neural_segmenter = tester.methods["Neural_SegFormer"]
            
            # Вариант 1: Простая сегментация (segment_image)
            print("\n1. Простая сегментация (segment_image)...")
            simple_result = neural_segmenter.segment_image(local_image_path, alpha=0.5)
            
            # Сохраняем результат
            simple_result.save("neural_segmentation_result.jpg")
            print(f"✅ Результат сохранен: neural_segmentation_result.jpg")

            segmented_image_path_1 = "test_segmented_image_1.jpg"
            simple_result_1 = neural_segmenter.segment_image(local_image_path_1, alpha=0.5)
            simple_result_1.save(segmented_image_path_1)
            print(f"✅ Результат сохранен: {segmented_image_path_1}")

            segmented_image_path_2 = "test_segmented_image_2.jpg"
            simple_result_2 = neural_segmenter.segment_image(local_image_path_2, alpha=0.5)
            simple_result_2.save(segmented_image_path_2)
            print(f"✅ Результат сохранен: {segmented_image_path_2}")

            segmented_image_path_3 = "test_segmented_image_3.jpg"
            simple_result_3 = neural_segmenter.segment_image(local_image_path_3, alpha=0.5)
            simple_result_3.save(segmented_image_path_3)
            print(f"✅ Результат сохранен: {segmented_image_path_3}")

            segmented_image_path_4 = "test_segmented_image_4.jpg"
            simple_result_4 = neural_segmenter.segment_image(local_image_path_4, alpha=0.5)
            simple_result_4.save(segmented_image_path_4)
            print(f"✅ Результат сохранен: {segmented_image_path_4}")

            segmented_image_path_5 = "test_segmented_image_5.jpg"
            simple_result_5 = neural_segmenter.segment_image(local_image_path_5, alpha=0.5)
            simple_result_5.save(segmented_image_path_5)
            print(f"✅ Результат сохранен: {segmented_image_path_5}")

            segmented_image_path_6 = "test_segmented_image_6.jpg"
            simple_result_6 = neural_segmenter.segment_image(local_image_path_6, alpha=0.5)
            simple_result_6.save(segmented_image_path_6)
            print(f"✅ Результат сохранен: {segmented_image_path_6}")
            
            # ============ СРАВНЕНИЕ С GROUND TRUTH ============
        
            print("\n2. Сравнение с Ground Truth...")
            
            # Получаем палитру из нейросетевого сегментатора
            palette = neural_segmenter.palette
            palette_array = np.array(palette, dtype=np.uint8)
            
            # 2.1 Ground Truth визуализация (как в оригинальном коде)
            print("\n  2.1 Визуализация Ground Truth...")
            ground_truth_seg = np.array(segmentation_map)  # 2D ground truth segmentation map
            
            ground_truth_color_seg = np.zeros(
                (ground_truth_seg.shape[0], ground_truth_seg.shape[1], 3), 
                dtype=np.uint8
            )
            
            # Для ADE20K: ground truth начинается с 1, а предсказание с 0
            # Поэтому используем ground_truth_seg - 1
            for label, color in enumerate(palette_array):
                ground_truth_color_seg[ground_truth_seg - 1 == label, :] = color
            
            # Конвертируем из BGR в RGB (если нужно)
            ground_truth_color_seg = ground_truth_color_seg[..., ::-1]
            
            # Создаем наложение ground truth на оригинал
            ground_truth_overlay = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
            ground_truth_overlay = ground_truth_overlay.astype(np.uint8)

            # Сохраняем ground truth overlay
            ground_truth_overlay_img = Image.fromarray(ground_truth_overlay)
            ground_truth_overlay_img.save("ground_truth_overlay.jpg")
            print(f"✅ Ground Truth Overlay сохранен: ground_truth_overlay.jpg")
            
            # Сохраняем цветную сегментацию ground truth
            ground_truth_color_img = Image.fromarray(ground_truth_color_seg)
            ground_truth_color_img.save("ground_truth_color.jpg")
            print(f"✅ Ground Truth Color Map сохранен: ground_truth_color.jpg")
            
            # 2.2 Neural Segmentation визуализация
            print("\n  2.2 Визуализация Neural Segmentation...")
            # Получаем сегментированное изображение с alpha=0.5 для сравнения
            neural_result = neural_segmenter.segment_image(local_image_path, alpha=0.5)
            neural_np = np.array(neural_result)
            
            # Сохраняем neural segmentation с alpha=0.5
            neural_result.save("neural_segmentation_alpha_0.5.jpg")
            print(f"✅ Neural Segmentation (alpha=0.5) сохранен: neural_segmentation_alpha_0.5.jpg")
            
            # 2.3 Сравнительная визуализация
            print("\n  2.3 Сравнительная визуализация...")
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Оригинальное изображение
            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            
            # Neural Segmentation (alpha=1)
            axes[0, 1].imshow(simple_result)
            axes[0, 1].set_title("Neural Segmentation (alpha=1)")
            axes[0, 1].axis('off')
            
            # Neural Segmentation (alpha=0.5) - для сравнения с GT
            axes[0, 2].imshow(neural_np)
            axes[0, 2].set_title("Neural Segmentation (alpha=0.5)")
            axes[0, 2].axis('off')
            
            # Ground Truth (цветная сегментация)
            axes[1, 0].imshow(ground_truth_color_seg)
            axes[1, 0].set_title("Ground Truth Color Map")
            axes[1, 0].axis('off')
            
            # Ground Truth Overlay
            axes[1, 1].imshow(ground_truth_overlay)
            axes[1, 1].set_title("Ground Truth Overlay (alpha=0.5)")
            axes[1, 1].axis('off')
            
            # Side-by-side сравнение
            comparison_img = np.hstack([neural_np, ground_truth_overlay])
            axes[1, 2].imshow(comparison_img)
            axes[1, 2].set_title("Comparison: Neural (left) vs GT (right)")
            axes[1, 2].axis('off')
            
            plt.suptitle("Neural Segmentation vs Ground Truth", fontsize=16)
            plt.tight_layout()
            
            # Сохраняем сводный график
            plt.savefig("neural_vs_gt_summary.jpg", dpi=150, bbox_inches='tight')
            print(f"✅ Сводный график сохранен: neural_vs_gt_summary.jpg")
            plt.show()
            
            # ============ РАСЧЕТ МЕТРИК ============
            
            print("\n3. Расчет метрик качества...")
            
            # Получаем предсказанную карту сегментации
            pred_seg_map = neural_segmenter.predict_segmentation_map(local_image_path)
            pred_seg_map_1 = neural_segmenter.predict_segmentation_map(local_image_path_1)
            pred_seg_map_2 = neural_segmenter.predict_segmentation_map(local_image_path_2)
            pred_seg_map_3 = neural_segmenter.predict_segmentation_map(local_image_path_3)
            pred_seg_map_4 = neural_segmenter.predict_segmentation_map(local_image_path_4)
            pred_seg_map_5 = neural_segmenter.predict_segmentation_map(local_image_path_5)
            pred_seg_map_6 = neural_segmenter.predict_segmentation_map(local_image_path_6)
            
            # Сохраняем карту сегментации
            predicted_segmentation_map_0 = "prediction_segmentation_map.png"
            pred_seg_map_normalized = (pred_seg_map / pred_seg_map.max() * 255).astype(np.uint8)
            pred_seg_map_img = Image.fromarray(pred_seg_map_normalized)
            pred_seg_map_img.save(predicted_segmentation_map_0)
            print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_0}")

            predicted_segmentation_map_1 = "prediction_segmentation_map_1.png"
            pred_seg_map_normalized_1 = (pred_seg_map_1 / pred_seg_map_1.max() * 255).astype(np.uint8)
            pred_seg_map_img_1 = Image.fromarray(pred_seg_map_normalized_1)
            pred_seg_map_img_1.save(predicted_segmentation_map_1)
            print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_1}")

            predicted_segmentation_map_2 = "prediction_segmentation_map_2.png"
            pred_seg_map_normalized_2 = (pred_seg_map_2 / pred_seg_map_2.max() * 255).astype(np.uint8)
            pred_seg_map_img_2 = Image.fromarray(pred_seg_map_normalized_2)
            pred_seg_map_img_2.save(predicted_segmentation_map_2)
            print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_2}")

            predicted_segmentation_map_3 = "prediction_segmentation_map_3.png"
            pred_seg_map_normalized_3 = (pred_seg_map_3 / pred_seg_map_3.max() * 255).astype(np.uint8)
            pred_seg_map_img_3 = Image.fromarray(pred_seg_map_normalized_3)
            pred_seg_map_img_3.save(predicted_segmentation_map_3)
            print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_3}")

            predicted_segmentation_map_4 = "prediction_segmentation_map_4.png"
            pred_seg_map_normalized_4 = (pred_seg_map_4 / pred_seg_map_4.max() * 255).astype(np.uint8)
            pred_seg_map_img_4 = Image.fromarray(pred_seg_map_normalized_4)
            pred_seg_map_img_4.save(predicted_segmentation_map_4)
            print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_4}")

            predicted_segmentation_map_5 = "prediction_segmentation_map_5.png"
            pred_seg_map_normalized_5 = (pred_seg_map_5 / pred_seg_map_5.max() * 255).astype(np.uint8)
            pred_seg_map_img_5 = Image.fromarray(pred_seg_map_normalized_5)
            pred_seg_map_img_5.save(predicted_segmentation_map_5)
            print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_5}")

            predicted_segmentation_map_6 = "prediction_segmentation_map_6.png"
            pred_seg_map_normalized_6 = (pred_seg_map_6 / pred_seg_map_6.max() * 255).astype(np.uint8)
            pred_seg_map_img_6 = Image.fromarray(pred_seg_map_normalized_6)
            pred_seg_map_img_6.save(predicted_segmentation_map_6)
            print(f"✅ Карта сегментации сохранена: {predicted_segmentation_map_6}")
            
            # Выравниваем размеры
            h, w = min(pred_seg_map.shape[0], ground_truth_seg.shape[0]), \
                min(pred_seg_map.shape[1], ground_truth_seg.shape[1])
            
            pred_flat = pred_seg_map[:h, :w].flatten()
            gt_flat = (ground_truth_seg[:h, :w] - 1).flatten()  # -1 для соответствия
            
            # Вычисляем метрики
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                # Общие классы
                common_classes = np.intersect1d(np.unique(pred_flat), np.unique(gt_flat))
                
                if len(common_classes) > 0:
                    accuracy = accuracy_score(gt_flat, pred_flat)
                    precision = precision_score(gt_flat, pred_flat, average='weighted', zero_division=0)
                    recall = recall_score(gt_flat, pred_flat, average='weighted', zero_division=0)
                    f1 = f1_score(gt_flat, pred_flat, average='weighted', zero_division=0)
                    
                    print(f"\n   Метрики качества:")
                    print(f"   - Accuracy:  {accuracy:.4f}")
                    print(f"   - Precision: {precision:.4f}")
                    print(f"   - Recall:    {recall:.4f}")
                    print(f"   - F1-Score:  {f1:.4f}")
                    print(f"   - Общих классов: {len(common_classes)}")
                    
                    # Сохраняем метрики в файл
                    metrics_file = "segmentation_metrics.txt"
                    with open(metrics_file, 'w') as f:
                        f.write("Segmentation Metrics\n")
                        f.write("="*50 + "\n")
                        f.write(f"Accuracy:  {accuracy:.4f}\n")
                        f.write(f"Precision: {precision:.4f}\n")
                        f.write(f"Recall:    {recall:.4f}\n")
                        f.write(f"F1-Score:  {f1:.4f}\n")
                        f.write(f"Common Classes: {len(common_classes)}\n")
                        f.write(f"Image: {local_image_path}\n")
                        f.write(f"Model: {neural_segmenter.model_name}\n")
                        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    print(f"✅ Метрики сохранены в: {metrics_file}")
                    
                    # Визуализация разницы
                    diff = np.abs(pred_seg_map[:h, :w] - (ground_truth_seg[:h, :w] - 1))
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Предсказание
                    axes[0].imshow(pred_seg_map[:h, :w], cmap='tab20')
                    axes[0].set_title(f"Prediction\nAccuracy: {accuracy:.3f}")
                    axes[0].axis('off')
                    
                    # Ground Truth
                    axes[1].imshow(ground_truth_seg[:h, :w] - 1, cmap='tab20')
                    axes[1].set_title("Ground Truth")
                    axes[1].axis('off')
                    
                    # Разность
                    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=10)
                    axes[2].set_title("Difference (Prediction - GT)")
                    axes[2].axis('off')
                    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                    
                    plt.suptitle(f"Segmentation Comparison - F1 Score: {f1:.3f}", fontsize=14)
                    plt.tight_layout()
                    
                    # Сохраняем график сравнения
                    plt.savefig("segmentation_comparison_detailed.jpg", dpi=150, bbox_inches='tight')
                    print(f"✅ Детальный график сравнения сохранен: segmentation_comparison_detailed.jpg")
                    plt.show()
                    
                    # Сохраняем разность как изображение
                    diff_normalized = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
                    diff_img = Image.fromarray(diff_normalized)
                    diff_img.save("segmentation_difference.png")
                    print(f"✅ Карта разности сохранена: segmentation_difference.png")
                    
                else:
                    print("   ⚠️ Нет общих классов для сравнения")
                    
            except ImportError:
                print("   ⚠️ scikit-learn не установлен. Пропускаем расчет метрик.")
                print("   Установите: pip install scikit-learn")

            # ============ ДОПОЛНИТЕЛЬНЫЕ СОХРАНЕНИЯ ============
            
            print("\n4. Дополнительные сохранения...")
            import os
            import shutil
            
            # Сохраняем оригинальное изображение
            image.save("original_image.jpg")
            print(f"✅ Оригинальное изображение сохранено: original_image.jpg")
            
            # Сохраняем ground truth как есть
            segmentation_map.save("ground_truth_original.png")
            print(f"✅ Ground Truth оригинал сохранен: ground_truth_original.png")
            
            # Сохраняем все в отдельную папку
            results_dir = "neural_segmentation_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Перемещаем все файлы в папку
            files_to_move = [
                "neural_segmentation_result.jpg",
                "neural_segmentation_alpha_0.5.jpg",
                "ground_truth_overlay.jpg",
                "ground_truth_color.jpg",
                "neural_vs_gt_summary.jpg",
                "segmentation_metrics.txt",
                "segmentation_comparison_detailed.jpg",
                "segmentation_difference.png",
                "prediction_segmentation_map.png",
                "original_image.jpg",
                "ground_truth_original.png"
            ]
            
            for file in files_to_move:
                if os.path.exists(file):
                    shutil.move(file, os.path.join(results_dir, file))
                    print(f"   Перемещен: {file} -> {results_dir}/{file}")
            
            print(f"\n✅ Все результаты сохранены в папку: {results_dir}")
            
            # ============ ФИНАЛЬНАЯ СВОДКА ============
            
            print("\n" + "="*60)
            print("ФИНАЛЬНАЯ СВОДКА РЕЗУЛЬТАТОВ")
            print("="*60)
            print(f"Папка с результатами: {results_dir}")
            print(f"Исходное изображение: {local_image_path}")
            print(f"Ground Truth: {segmentation_map_path}")
            print(f"Модель: {neural_segmenter.model_name}")
            
            if os.path.exists(os.path.join(results_dir, "segmentation_metrics.txt")):
                print("\nМетрики качества:")
                with open(os.path.join(results_dir, "segmentation_metrics.txt"), 'r') as f:
                    print(f.read())

            # ============ ДЕТАЛЬНАЯ СЕГМЕНТАЦИЯ ============

            # Вариант 2: Детальная сегментация
            print("\n2. Детальная сегментация...")
            detailed_result = neural_segmenter.detailed_segmentation(local_image_path)
            detailed_result_1 = neural_segmenter.detailed_segmentation(local_image_path_1)
            detailed_result_2 = neural_segmenter.detailed_segmentation(local_image_path_2)
            detailed_result_3 = neural_segmenter.detailed_segmentation(local_image_path_3)
            detailed_result_4 = neural_segmenter.detailed_segmentation(local_image_path_4)
            detailed_result_5 = neural_segmenter.detailed_segmentation(local_image_path_5)
            detailed_result_6 = neural_segmenter.detailed_segmentation(local_image_path_6)

            # Выводим информацию о классах
            print(f"Обнаружено классов по изображению 0: {detailed_result['total_classes']}")
            print(f"Обнаружено классов по изображению 1: {detailed_result_1['total_classes']}")
            print(f"Обнаружено классов по изображению 2: {detailed_result_2['total_classes']}")
            print(f"Обнаружено классов по изображению 3: {detailed_result_3['total_classes']}")
            print(f"Обнаружено классов по изображению 4: {detailed_result_4['total_classes']}")
            print(f"Обнаружено классов по изображению 5: {detailed_result_5['total_classes']}")
            print(f"Обнаружено классов по изображению 6: {detailed_result_6['total_classes']}")

            print("\nТоп-5 классов по площади (0):")

            sorted_classes = sorted(detailed_result['class_distribution'].items(), 
                                    key=lambda x: x[1]['pixel_count'], 
                                    reverse=True)[:5]

            for i, (class_name, info) in enumerate(sorted_classes, 1):
                print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

            # Сохраняем overlay изображение из detailed_result
            neural_segmentation_result_detailed_0 = "neural_segmentation_result_detailed.jpg"
            # Извлекаем overlay из словаря и сохраняем
            overlay_0 = Image.fromarray(detailed_result['overlay'])
            overlay_0.save(neural_segmentation_result_detailed_0)
            print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_0}")

            print("\nТоп-5 классов по площади (1):")
            sorted_classes_1 = sorted(detailed_result_1['class_distribution'].items(), 
                                    key=lambda x: x[1]['pixel_count'], 
                                    reverse=True)[:5]

            for i, (class_name, info) in enumerate(sorted_classes_1, 1):
                print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

            neural_segmentation_result_detailed_1 = "neural_segmentation_result_detailed_1.jpg"
            overlay_1 = Image.fromarray(detailed_result_1['overlay'])
            overlay_1.save(neural_segmentation_result_detailed_1)
            print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_1}")

            print("\nТоп-5 классов по площади (2):")
            sorted_classes_2 = sorted(detailed_result_2['class_distribution'].items(), 
                                    key=lambda x: x[1]['pixel_count'], 
                                    reverse=True)[:5]

            for i, (class_name, info) in enumerate(sorted_classes_2, 1):
                print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

            neural_segmentation_result_detailed_2 = "neural_segmentation_result_detailed_2.jpg"
            overlay_2 = Image.fromarray(detailed_result_2['overlay'])
            overlay_2.save(neural_segmentation_result_detailed_2)
            print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_2}")

            print("\nТоп-5 классов по площади (3):")
            sorted_classes_3 = sorted(detailed_result_3['class_distribution'].items(), 
                                    key=lambda x: x[1]['pixel_count'], 
                                    reverse=True)[:5]

            for i, (class_name, info) in enumerate(sorted_classes_3, 1):
                print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

            neural_segmentation_result_detailed_3 = "neural_segmentation_result_detailed_3.jpg"
            overlay_3 = Image.fromarray(detailed_result_3['overlay'])
            overlay_3.save(neural_segmentation_result_detailed_3)
            print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_3}")

            print("\nТоп-5 классов по площади (4):")
            sorted_classes_4 = sorted(detailed_result_4['class_distribution'].items(), 
                                    key=lambda x: x[1]['pixel_count'], 
                                    reverse=True)[:5]

            for i, (class_name, info) in enumerate(sorted_classes_4, 1):
                print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

            neural_segmentation_result_detailed_4 = "neural_segmentation_result_detailed_4.jpg"
            overlay_4 = Image.fromarray(detailed_result_4['overlay'])
            overlay_4.save(neural_segmentation_result_detailed_4)
            print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_4}")

            print("\nТоп-5 классов по площади (5):")
            sorted_classes_5 = sorted(detailed_result_5['class_distribution'].items(), 
                                    key=lambda x: x[1]['pixel_count'], 
                                    reverse=True)[:5]

            for i, (class_name, info) in enumerate(sorted_classes_5, 1):
                print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

            neural_segmentation_result_detailed_5 = "neural_segmentation_result_detailed_5.jpg"
            overlay_5 = Image.fromarray(detailed_result_5['overlay'])
            overlay_5.save(neural_segmentation_result_detailed_5)
            print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_5}")

            print("\nТоп-5 классов по площади (6):")
            sorted_classes_6 = sorted(detailed_result_6['class_distribution'].items(), 
                                    key=lambda x: x[1]['pixel_count'], 
                                    reverse=True)[:5]

            for i, (class_name, info) in enumerate(sorted_classes_6, 1):
                print(f"  {i}. {class_name}: {info['percentage']:.1f}% ({info['pixel_count']} пикселей)")

            neural_segmentation_result_detailed_6 = "neural_segmentation_result_detailed_6.jpg"
            overlay_6 = Image.fromarray(detailed_result_6['overlay'])
            overlay_6.save(neural_segmentation_result_detailed_6)
            print(f"✅ Результат сохранен: {neural_segmentation_result_detailed_6}")

            # Показываем overlay из detailed_segmentation_0
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(detailed_result['overlay'])
            axes[1].set_title("Detailed Segmentation (alpha=0.5)")
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Ошибка тестирования нейросетевой сегментации: {e}")
            # print(traceback.format_exc())
                
    else:
        print("⚠️ Нейросетевая сегментация не доступна")
    
    # ============ СРАВНЕНИЕ ВСЕХ МЕТОДОВ ============
    
    print("\n" + "=" * 60)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ СЕГМЕНТАЦИИ")
    print("=" * 60)
    
    # Выбираем подмножество методов для сравнения
    selected_methods = [
        "Global_Threshold_CV2",
        "Adaptive_Threshold_CV2",
        "Otsu_CV2",
        "Region_Growing_CV2",
        "Split_and_Merge_CV2",
        "Sobel_CV2",
        "Canny_CV2",
        "KMeans_CV2",
        # "DBSCAN_CV2",
        "Active_Contour_CV2",
        "GVF_Contour_CV2",
        "Watershed_CV2",
        # "Meanshift_CV2",
        "GrabCut_CV2",
        "FloodFill_CV2",
        "Morphological_Snakes_CV2",
        # "Quickshift_CV2",
        "Slic_CV2",
        "Felzenszwalb_CV2",
        "Chan_Vese_CV2",
        "Threshold_Niblack_CV2",
        "Threshold_Sauvola_CV2",
        "Random_Walker_CV2",
        "Global_Threshold_Torch",
        "Adaptive_Threshold_Torch",
        "Otsu_Torch",
        # "Region_Growing_Torch",
        "Split_and_Merge_Torch",
        "Sobel_Torch",
        "Canny_Torch",
        "KMeans_Torch",
        # "DBSCAN_Torch",
        # "Active_Contour_Torch",
        # "GVF_Contour_Torch",
        "Watershed_Torch",
        # "Meanshift_Torch",
        "Grabcut_Torch",
        "FloodFill_Torch",
    ]
    
    if "Neural_SegFormer" in tester.methods:
        selected_methods.append("Neural_SegFormer")
    
    print(f"Сравниваем {len(selected_methods)} методов...")
    
    try:
        results = tester.compare_methods(
            local_image_path, 
            method_names=selected_methods,
            figsize=(20, 15),
            test_name="full_comparison_0",  # Имя теста
            show_plots=True  # Показывать графики
        )
        
        # Визуализация сравнения
        tester.visualize_comparison(results, 
                                    show_masks=True,
                                    save_visualization=True,
                                    show_plots=True)

        results_1 = tester.compare_methods(
            local_image_path_1, 
            method_names=selected_methods,
            figsize=(20, 15),
            test_name="full_comparison_1",
            show_plots=True
        )
        tester.visualize_comparison(results_1, 
                                    show_masks=True,
                                    save_visualization=True,
                                    show_plots=True)

        results_2 = tester.compare_methods(
            local_image_path_2, 
            method_names=selected_methods,
            figsize=(20, 15),
            test_name="full_comparison_2",
            show_plots=True
        )
        tester.visualize_comparison(results_2, 
                                    show_masks=True,
                                    save_visualization=True,
                                    show_plots=True)

        results_3 = tester.compare_methods(
            local_image_path_3, 
            method_names=selected_methods,
            figsize=(20, 15),
            test_name="full_comparison_3",
            show_plots=True
        )
        tester.visualize_comparison(results_3, 
                                    show_masks=True,
                                    save_visualization=True,
                                    show_plots=True)

        results_4 = tester.compare_methods(
            local_image_path_4, 
            method_names=selected_methods,
            figsize=(20, 15),
            test_name="full_comparison_4",
            show_plots=True
        )
        tester.visualize_comparison(results_4, 
                                    show_masks=True,
                                    save_visualization=True,
                                    show_plots=True)

        results_5 = tester.compare_methods(
            local_image_path_5, 
            method_names=selected_methods,
            figsize=(20, 15),
            test_name="full_comparison_5",
            show_plots=True
        )
        tester.visualize_comparison(results_5, 
                                    show_masks=True,
                                    save_visualization=True,
                                    show_plots=True)

        results_6 = tester.compare_methods(
            local_image_path_6,
            method_names=selected_methods,
            figsize=(20, 15),
            test_name="full_comparison_6",
            show_plots=True
        )
        tester.visualize_comparison(results_6, 
                                    show_masks=True,
                                    save_visualization=True,
                                    show_plots=True)
        
    except Exception as e:
        print(f"❌ Ошибка при сравнении методов: {e}")
        results = {}
        results_1 = {}
        results_2 = {}
        results_3 = {}
        results_4 = {}
        results_5 = {}
        results_6 = {}

     # ============ Сравнение с таймингом ============
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ СРАВНЕНИЯ МЕТОДОВ С ТАЙМИНГОМ")
    print("=" * 60)
    
    try:
        # Тестируем на последнем изображении
        print(f"Тестируем на изображении (0): {local_image_path}")
        
        # Запускаем сравнение с таймингом
        methods_00, results_00, masks_00, times_00 = compare_segmentation_methods_with_timing(local_image_path)
        dict_compare_00 = original_compare_segmentation_methods(local_image_path)

        print(f"Тестируем на изображении (1): {local_image_path_1}")
        methods_01, results_01, masks_01, times_01 = compare_segmentation_methods_with_timing(local_image_path_1)
        dict_compare_01 = original_compare_segmentation_methods(local_image_path_1)

        print(f"Тестируем на изображении (2): {local_image_path_2}")
        methods_02, results_02, masks_02, times_02 = compare_segmentation_methods_with_timing(local_image_path_2)
        dict_compare_02 = original_compare_segmentation_methods(local_image_path_2)

        print(f"Тестируем на изображении (3): {local_image_path_3}")
        methods_03, results_03, masks_03, times_03 = compare_segmentation_methods_with_timing(local_image_path_3)
        dict_compare_03 = original_compare_segmentation_methods(local_image_path_3)

        print(f"Тестируем на изображении (4): {local_image_path_4}")
        methods_04, results_04, masks_04, times_04 = compare_segmentation_methods_with_timing(local_image_path_4)
        dict_compare_04 = original_compare_segmentation_methods(local_image_path_4)

        print(f"Тестируем на изображении (5): {local_image_path_5}")
        methods_05, results_05, masks_05, times_05 = compare_segmentation_methods_with_timing(local_image_path_5)
        dict_compare_05 = original_compare_segmentation_methods(local_image_path_5)

        print(f"Тестируем на изображении (6): {local_image_path_6}")
        methods_06, results_06, masks_06, times_06 = compare_segmentation_methods_with_timing(local_image_path_6)
        dict_compare_06 = original_compare_segmentation_methods(local_image_path_6)
        
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ С ТАЙМИНГОМ ЗАВЕРШЕНО")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании с таймингом: {e}")
        import traceback
        traceback.print_exc()
    
    # ============ БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ ============
    
    print("\n" + "=" * 60)
    print("БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    
    try:
        df = tester.benchmark_methods(local_image_path, 
                                      n_runs=2,
                                      test_name="performance_test_0",
                                      save_results=True)
    except Exception as ex0:
        print(f"❌ Ошибка бенчмарка (0): {ex0}")
        df = pd.DataFrame()

    try:
        df_1 = tester.benchmark_methods(local_image_path_1, 
                                        n_runs=2,
                                        test_name="performance_test_1",
                                        save_results=True)
    except Exception as ex1:
        print(f"❌ Ошибка бенчмарка (1): {ex1}")
        df_1 = pd.DataFrame()


    try:
        df_2 = tester.benchmark_methods(local_image_path_2, 
                                        n_runs=2,
                                        test_name="performance_test_2",
                                        save_results=True)
    except Exception as ex2:
        print(f"❌ Ошибка бенчмарка (2): {ex2}")
        df_2 = pd.DataFrame() 

    try:
        df_3 = tester.benchmark_methods(local_image_path_3, 
                                        n_runs=2,
                                        test_name="performance_test_3",
                                        save_results=True)
    except Exception as ex3:
        print(f"❌ Ошибка бенчмарка (3): {ex3}")
        df_3 = pd.DataFrame()

    try:
        df_4 = tester.benchmark_methods(local_image_path_4, 
                                        n_runs=2,
                                        test_name="performance_test_4",
                                        save_results=True)
    except Exception as ex4:
        print(f"❌ Ошибка бенчмарка (4): {ex4}")
        df_4 = pd.DataFrame()
        
    try:
        df_5 = tester.benchmark_methods(local_image_path_5, 
                                        n_runs=2,
                                        test_name="performance_test_5",
                                        save_results=True)
    except Exception as ex5:
        print(f"❌ Ошибка бенчмарка (5): {ex5}")
        df_5 = pd.DataFrame()
    
    try:
        df_6 = tester.benchmark_methods(local_image_path_6, 
                                        n_runs=2,
                                        test_name="performance_test_6",
                                        save_results=True)
    except Exception as ex6:
        print(f"❌ Ошибка бенчмарка (6): {ex6}")
        df_6 = pd.DataFrame()    
    
    
    # ============ СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ============
    
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    output_dir = "segmentation_comparison_results"
    try:
        tester.save_results(results, output_dir)
        
        # Дополнительно: сохраняем сводную таблицу
        if not df.empty:
            summary_path = os.path.join(output_dir, "summary.csv")
            df.to_csv(summary_path, index=False)
            print(f"✅ Сводная таблица сохранена (0): {summary_path}")
    except Exception as ex0:
        print(f"⚠️ Ошибка сохранения результатов: {ex0}")

    try:
        tester.save_results(results_1, output_dir)
        if not df_1.empty:
            summary_path = os.path.join(output_dir, "summary_1.csv")
            df_1.to_csv(summary_path, index=False)
            print(f"✅ Сводная таблица сохранена (1): {summary_path}")
    except Exception as ex1:
        print(f"⚠️ Ошибка сохранения результатов: {ex1}")

    try:
        tester.save_results(results_2, output_dir)
        if not df_2.empty:
            summary_path = os.path.join(output_dir, "summary_2.csv")
            df_2.to_csv(summary_path, index=False)
            print(f"✅ Сводная таблица сохранена (2): {summary_path}")
    except Exception as ex2:
        print(f"⚠️ Ошибка сохранения результатов: {ex2}")

    try:
        tester.save_results(results_3, output_dir)
        if not df_3.empty:
            summary_path = os.path.join(output_dir, "summary_3.csv")
            df_3.to_csv(summary_path, index=False)
            print(f"✅ Сводная таблица сохранена (3): {summary_path}")
    except Exception as ex3:
        print(f"⚠️ Ошибка сохранения результатов: {ex3}")

    try:
        tester.save_results(results_4, output_dir)
        if not df_4.empty:
            summary_path = os.path.join(output_dir, "summary_4.csv")
            df_4.to_csv(summary_path, index=False)
            print(f"✅ Сводная таблица сохранена (4): {summary_path}")
    except Exception as ex4:
        print(f"⚠️ Ошибка сохранения результатов: {ex4}")

    try:
        tester.save_results(results_5, output_dir)
        if not df_5.empty:
            summary_path = os.path.join(output_dir, "summary_5.csv")
            df_5.to_csv(summary_path, index=False)
            print(f"✅ Сводная таблица сохранена (5): {summary_path}")
    except Exception as ex5:
        print(f"⚠️ Ошибка сохранения результатов: {ex5}")

    try:
        tester.save_results(results_6, output_dir)
        if not df_6.empty:
            summary_path = os.path.join(output_dir, "summary_6.csv")
            df_6.to_csv(summary_path, index=False)
            print(f"✅ Сводная таблица сохранена (6): {summary_path}")
    except Exception as ex6:
        print(f"⚠️ Ошибка сохранения результатов: {ex6}")

    # ============ ВЫВОД ИНФОРМАЦИИ ============
    
    print("\n" + "=" * 60)
    print("РЕЗЮМЕ")
    print("=" * 60)
    print(f"✓ Загружено методов: {len(tester.methods)}")
    print(f"✓ Протестировано: {len(selected_methods)}")
    print(f"✓ Изображение (0): {local_image_path}")
    print(f"✓ Изображение (1): {local_image_path_1}")
    print(f"✓ Изображение (2): {local_image_path_2}")
    print(f"✓ Изображение (3): {local_image_path_3}")
    print(f"✓ Изображение (4): {local_image_path_4}")
    print(f"✓ Изображение (5): {local_image_path_5}")
    print(f"✓ Изображение (6): {local_image_path_6}")
    print(f"✓ Результаты сохранены в: {output_dir}")
    print("=" * 60)

    pipeline = create_advanced_pipeline_example()
    
    print(f"Создан пайплайн с шагами: {pipeline.get_step_names()}")
    
    # Запускаем пайплайн на тестовом изображении
    # (нужно иметь test_image.jpg в директории)
    try:
        results = pipeline.run("test_image_download_6.jpg", visualize=True)
        
        # Анализируем результаты
        analysis = analyze_pipeline_results(results)
        print_pipeline_analysis(analysis)
        
        # Сохраняем визуализацию
        pipeline.save_visualization(results, "test_image_download_6.jpg", "advanced_pipeline_results.jpg")
        
    except FileNotFoundError:
        print("⚠️ Файл test_image_download_6.jpg не найден.")
        print("Доступные файлы:")
        import os
        files = [f for f in os.listdir('.') if f.endswith('.jpg')]
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"❌ Ошибка выполнения пайплайна: {e}")
        import traceback
        traceback.print_exc()
    
    return tester, results, df


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


if __name__ == "__main__":
    # Основной тест
    print("ЗАПУСК ОСНОВНОГО ТЕСТА")
    print("=" * 60)
    tester, results, df = main()
    
    # Дополнительный тест нейросетевых вариантов
    # print("\n\nЗАПУСК ДОПОЛНИТЕЛЬНОГО ТЕСТА НЕЙРОСЕТЕВЫХ ВАРИАНТОВ")
    # print("=" * 60)
    # segmenter, detailed_result = test_neural_segmentation_variants()


def compare_with_ground_truth_simple(image_path: str, 
                                     gt_path: str, 
                                     neural_segmenter: NeuralSegmenter,
                                     save_results: bool = True):
    """
    Простое сравнение нейросетевой сегментации с ground truth
    """
    print("\n" + "="*60)
    print("ПРОСТОЕ СРАВНЕНИЕ С GROUND TRUTH")
    print("="*60)
    
    try:
        # Загрузка изображений
        image = Image.open(image_path)
        segmentation_map = Image.open(gt_path)
        
        print(f"Изображение: {image_path}")
        print(f"Ground Truth: {gt_path}")
        print(f"Размер изображения: {image.size}")
        print(f"Размер GT: {segmentation_map.size}")
        
        # 1. Получаем палитру
        palette = neural_segmenter.palette
        palette_array = np.array(palette, dtype=np.uint8)
        
        # 2. Ground Truth визуализация
        ground_truth_seg = np.array(segmentation_map)
        
        # Проверяем диапазон значений
        print(f"\nДиапазон значений Ground Truth: {ground_truth_seg.min()} - {ground_truth_seg.max()}")
        
        ground_truth_color_seg = np.zeros(
            (ground_truth_seg.shape[0], ground_truth_seg.shape[1], 3), 
            dtype=np.uint8
        )
        
        # Для ADE20K: ground truth начинается с 1
        for label, color in enumerate(palette_array):
            ground_truth_color_seg[ground_truth_seg - 1 == label, :] = color
        
        # Конвертируем в RGB
        ground_truth_color_seg = ground_truth_color_seg[..., ::-1]
        
        # 3. Neural Segmentation
        neural_result = neural_segmenter.segment_image(image_path, alpha=0.5)
        neural_np = np.array(neural_result)
        
        # 4. Создаем ground truth overlay
        ground_truth_overlay = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
        ground_truth_overlay = ground_truth_overlay.astype(np.uint8)
        
        # 5. Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Оригинал
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Neural Segmentation
        axes[0, 1].imshow(neural_np)
        axes[0, 1].set_title("Neural Segmentation (alpha=0.5)")
        axes[0, 1].axis('off')
        
        # Ground Truth
        axes[1, 0].imshow(ground_truth_overlay)
        axes[1, 0].set_title("Ground Truth Overlay (alpha=0.5)")
        axes[1, 0].axis('off')
        
        # Разность
        pred_seg_map = neural_segmenter.predict_segmentation_map(image_path)
        
        # Выравниваем размеры
        h, w = min(pred_seg_map.shape[0], ground_truth_seg.shape[0]), \
               min(pred_seg_map.shape[1], ground_truth_seg.shape[1])
        
        # Вычисляем разность
        pred_resized = pred_seg_map[:h, :w]
        gt_resized = ground_truth_seg[:h, :w] - 1
        
        diff = np.abs(pred_resized - gt_resized)
        
        # Нормализуем для визуализации
        if diff.max() > 0:
            diff_normalized = diff / diff.max()
        else:
            diff_normalized = diff
        
        im = axes[1, 1].imshow(diff_normalized, cmap='hot')
        axes[1, 1].set_title("Difference (Prediction vs GT)")
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.suptitle("Neural Segmentation vs Ground Truth", fontsize=14)
        plt.tight_layout()
        
        if save_results:
            plt.savefig("neural_vs_gt_comparison.jpg", dpi=150, bbox_inches='tight')
            print(f"✅ Сравнение сохранено: neural_vs_gt_comparison.jpg")
        
        plt.show()
        
        # 6. Простые метрики
        print("\nСтатистика:")
        print(f"  - Размер предсказания: {pred_seg_map.shape}")
        print(f"  - Размер Ground Truth: {ground_truth_seg.shape}")
        print(f"  - Классы в предсказании: {len(np.unique(pred_seg_map))}")
        print(f"  - Классы в GT: {len(np.unique(ground_truth_seg))}")
        
        # Простой подсчет совпадений
        if h > 0 and w > 0:
            matches = np.sum(pred_resized == gt_resized)
            total_pixels = h * w
            match_percentage = (matches / total_pixels) * 100
            
            print(f"  - Совпадение пикселей: {matches}/{total_pixels} ({match_percentage:.1f}%)")
        
        return {
            'image': image,
            'ground_truth': ground_truth_seg,
            'prediction': neural_result,
            'prediction_map': pred_seg_map,
            'gt_overlay': ground_truth_overlay
        }
        
    except Exception as e:
        print(f"❌ Ошибка сравнения с Ground Truth: {e}")
        print(traceback.format_exc())
        return None